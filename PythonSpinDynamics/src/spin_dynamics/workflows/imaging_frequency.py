"""Frequency-encoded imaging: spin-warp and RARE / fast spin echo.

The existing CPMG imaging workflows fill k-space one point per phase-encode
step. These workflows add a *readout* (frequency-encode) gradient applied during
acquisition, so each echo samples a whole k-space line. Built on the Lagrangian
motion engine (one static isochromat per voxel), they support arbitrary
two-axis gradient waveforms, which the scalar-gradient arbitrary-pulse kernels
cannot express.

* ``run_spin_warp_imaging`` -- a spin echo per phase-encode line: readout along
  x, phase encode along z. One image needs ``pz`` excitations.
* ``run_gradient_echo_imaging`` -- the matched Cartesian gradient-echo
  acquisition without a 180-degree pulse.
* ``run_rare_imaging`` -- Rapid Acquisition with Relaxation Enhancement (a.k.a.
  fast spin echo): a CPMG echo train where each echo reads a different k-space
  line, so ``echo_train_length`` lines are filled per excitation. The image is
  acquired in ``ceil(pz / echo_train_length)`` shots, at the cost of a T2 weight
  that varies across the phase-encode lines (the source of RARE blurring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.motion import (
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import MotionSequenceStep, run_motion_sequence
from spin_dynamics.workflows.imaging import reconstruct_image_from_kspace
from spin_dynamics.workflows.imaging_types import ImagingFieldMaps


PhaseEncodeOrder = Literal["linear", "centric"]


@dataclass(frozen=True)
class FrequencyEncodedImagingResult:
    """Result of a frequency-encoded (spin-warp or RARE) imaging simulation."""

    kspace: np.ndarray  # (px, pz, 1) centered k-space
    image: np.ndarray  # (px, pz, 1) complex reconstruction
    magnitude: np.ndarray  # (px, pz, 1)
    rho: np.ndarray
    echo_train_length: int
    num_shots: int
    line_echo_index: np.ndarray  # echo position (0-based) that filled each k_z line
    line_echo_time: np.ndarray  # echo time (s) at which each k_z line was sampled
    fov: tuple[float, float]
    echo_spacing: float
    method: str
    num_offsets: int  # sub-voxel B0 samples averaged per voxel (1 = none)
    offset_spread: float  # half-width of the sub-voxel B0 spread (rad/s)
    offset_model: str = "uniform"
    diffusion_coefficient: float = 0.0
    walkers_per_cell: int = 1

    def reconstruct(self) -> np.ndarray:
        """Return the complex image (alias for the stored reconstruction)."""

        return self.image[:, :, 0]


@dataclass(frozen=True)
class RadialUTEImagingResult:
    """Finite-pulse center-out radial short-TE acquisition and reconstruction."""

    samples: np.ndarray
    trajectory_rad_m: np.ndarray
    encoding_matrix: np.ndarray
    image: np.ndarray
    magnitude: np.ndarray
    rho: np.ndarray
    fov: tuple[float, float]
    num_spokes: int
    radial_samples: int
    echo_time_s: float
    readout_time_s: float
    excitation_duration_s: float
    flip_angle_rad: float
    regularization: float
    num_offsets: int
    offset_spread: float
    diffusion_coefficient: float
    walkers_per_cell: int

    def reconstruct(
        self,
        samples: np.ndarray | None = None,
        *,
        regularization: float | None = None,
    ) -> np.ndarray:
        """Reconstruct supplied or stored non-Cartesian samples."""

        data = self.samples if samples is None else np.asarray(samples, dtype=np.complex128)
        penalty = self.regularization if regularization is None else float(regularization)
        return _reconstruct_radial_samples(
            self.encoding_matrix,
            data,
            self.rho.shape,
            penalty,
        )


@dataclass(frozen=True)
class SliceSensitivityResult:
    """Real-space sensitive slice of an excitation in a non-uniform field.

    In a non-uniform B0 the spins that an RF pulse excites are those whose
    frequency falls in its band, so the excited region follows the curved
    iso-B0 contours rather than a flat plane, and its intensity varies with the
    transmit/receive B1. ``sensitivity`` is the excited (optionally refocused)
    transverse magnetization weighted by the receive sensitivity -- the relative
    signal each point contributes.
    """

    sensitivity: np.ndarray  # (px, pz) excited (+refocused) * receive weight
    excitation: np.ndarray  # (px, pz) |Mxy| excitation slice profile
    off_resonance: np.ndarray  # (px, pz) b0_map - center_frequency (rad/s)
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    center_frequency: float
    excitation_flip: float
    excitation_duration: float
    refocusing: bool
    fov: tuple[float, float]


def _resolve_maps(rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map):
    """Return ``(rho, t1, t2, b0, b1_tx, b1_rx)`` arrays from inputs.

    ``rho`` may be a 2D spin-density array (with optional map keywords) or an
    ``ImagingFieldMaps`` container shared with the phase-encoded workflows, in
    which case the map keywords must be omitted. ``b0_map`` is an absolute
    angular off-resonance map in rad/s.
    """

    if isinstance(rho, ImagingFieldMaps):
        if any(m is not None for m in (t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map)):
            raise ValueError(
                "do not provide map keywords when rho is an ImagingFieldMaps"
            )
        fm = rho
        return (
            np.asarray(fm.rho, dtype=np.float64),
            np.asarray(fm.t1_map, dtype=np.float64),
            np.asarray(fm.t2_map, dtype=np.float64),
            np.asarray(fm.b0_map, dtype=np.float64),
            np.asarray(fm.b1_tx_map, dtype=np.float64),
            np.asarray(fm.b1_rx_map, dtype=np.float64),
        )

    rho_arr = np.asarray(rho, dtype=np.float64)
    if rho_arr.ndim != 2 or min(rho_arr.shape) < 2:
        raise ValueError("rho must be a 2D array with at least 2x2 voxels")
    shape = rho_arr.shape

    def _opt(values, name, default, *, positive=False):
        if values is None:
            return np.full(shape, default, dtype=np.float64)
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != shape:
            raise ValueError(f"{name} must have the same shape as rho")
        if positive and np.any(arr <= 0.0):
            raise ValueError(f"{name} must be positive")
        return arr

    return (
        rho_arr,
        _opt(t1_map, "t1_map", np.inf, positive=True),
        _opt(t2_map, "t2_map", np.inf, positive=True),
        _opt(b0_map, "b0_map", 0.0),
        _opt(b1_tx_map, "b1_tx_map", 1.0),
        _opt(b1_rx_map, "b1_rx_map", 1.0),
    )


def _phase_encode_schedule(
    pz: int, echo_train_length: int, order: PhaseEncodeOrder
) -> list[list[int]]:
    """Group k_z line indices into shots of up to ``echo_train_length`` echoes."""

    etl = int(echo_train_length)
    if etl <= 0:
        raise ValueError("echo_train_length must be positive")
    num_shots = -(-pz // etl)  # ceil
    if order == "linear":
        ordered = list(range(pz))
    elif order == "centric":
        center = pz // 2
        ordered = sorted(range(pz), key=lambda line: (abs(line - center), line))
    else:
        raise ValueError("order must be 'linear' or 'centric'")
    # Interleave so each shot spans k-space and successive echoes step through it.
    shots: list[list[int]] = [[] for _ in range(num_shots)]
    for position, line in enumerate(ordered):
        shots[position % num_shots].append(line)
    return shots


def run_rare_imaging(
    rho,
    *,
    t1_map=None,
    t2_map=None,
    b0_map=None,
    b1_tx_map=None,
    b1_rx_map=None,
    fov: tuple[float, float] = (0.02, 0.02),
    echo_train_length: int = 8,
    phase_encode_order: PhaseEncodeOrder = "linear",
    readout_time: float = 2.0e-3,
    phase_time: float = 0.4e-3,
    excitation_duration: float = 50.0e-6,
    refocusing_duration: float = 100.0e-6,
    num_offsets: int = 1,
    offset_spread: float = 0.0,
    subvoxel_b0_offsets=None,
    subvoxel_density_weights=None,
    diffusion_coefficient: float = 0.0,
    walkers_per_cell: int = 1,
    seed: int | None = None,
    jitter: bool = False,
    refocusing: bool = True,
    gamma: float = 2.675e8,
    substeps_per_interval: int = 1,
) -> FrequencyEncodedImagingResult:
    """Simulate a RARE / fast-spin-echo frequency-encoded image.

    ``rho`` may be a 2D spin-density array or an ``ImagingFieldMaps`` container
    shared with the phase-encoded workflows. Readout is along x (frequency
    encode) and the phase encode is along z. Each CPMG echo reads one k_z line,
    so ``echo_train_length`` lines are acquired per excitation and the image
    needs ``ceil(pz / echo_train_length)`` shots. The T2 decay across the train
    weights the phase-encode lines, which broadens the point-spread function
    (RARE blurring).

    ``num_offsets`` (> 1) models an unresolved sub-voxel B0 spread by averaging
    that many isochromats per voxel, evenly spaced over ``+/- offset_spread``
    (rad/s) -- the counterpart of the ``ny`` / ``maxoffs`` off-resonance samples
    of the phase-encoded path. A spin echo refocuses the static spread at each
    echo, so the spread blurs the image along the readout axis (the T2*
    point-spread function) without decaying the echo train.

    ``subvoxel_b0_offsets`` supplies spatially varying unresolved offsets with
    shape ``(samples, px, pz)`` in rad/s.  It is useful for susceptibility
    fields whose intravoxel distribution varies across the image.  Each offset
    map is propagated through the complete pulse sequence and the resulting
    complex signals are averaged.  It is mutually exclusive with the scalar
    ``num_offsets``/``offset_spread`` model.

    ``subvoxel_density_weights`` optionally supplies a non-negative multiplier
    with the same shape.  It can exclude non-water inclusions independently at
    each subvoxel sample.  The weights modify spin density, not B0.

    ``diffusion_coefficient`` and ``walkers_per_cell`` activate an explicit
    random-walker Bloch--Torrey approximation.  This captures irreversible
    spin-echo attenuation when water diffuses through a spatially varying B0
    field.  ``seed`` makes the walker initialization and propagation
    reproducible; ``jitter`` distributes walkers within their source voxels.

    ``refocusing=False`` selects a single Cartesian gradient echo and requires
    ``echo_train_length=1``. The same finite RF, B0, subvoxel, diffusion, and
    readout models remain active.
    """

    rho_arr, t1_arr, t2_arr, b0_arr, b1_tx_arr, b1_rx_arr = _resolve_maps(
        rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map
    )
    px, pz = rho_arr.shape
    if min(px, pz) < 2:
        raise ValueError("rho must have at least 2x2 voxels")
    fov_x, fov_z = float(fov[0]), float(fov[1])
    if fov_x <= 0.0 or fov_z <= 0.0:
        raise ValueError("fov entries must be positive")
    if readout_time <= 0.0 or phase_time <= 0.0:
        raise ValueError("readout_time and phase_time must be positive")
    if num_offsets < 1:
        raise ValueError("num_offsets must be at least 1")
    if offset_spread < 0.0:
        raise ValueError("offset_spread must be non-negative")
    diffusion = float(diffusion_coefficient)
    if not np.isfinite(diffusion) or diffusion < 0.0:
        raise ValueError("diffusion_coefficient must be finite and non-negative")
    if int(walkers_per_cell) != walkers_per_cell or walkers_per_cell < 1:
        raise ValueError("walkers_per_cell must be a positive integer")
    if not refocusing and int(echo_train_length) != 1:
        raise ValueError("gradient echo requires echo_train_length=1")
    offset_maps = None
    if subvoxel_b0_offsets is not None:
        offset_maps = np.asarray(subvoxel_b0_offsets, dtype=np.float64)
        if (
            offset_maps.ndim != 3
            or offset_maps.shape[1:] != rho_arr.shape
            or offset_maps.shape[0] < 1
            or np.any(~np.isfinite(offset_maps))
        ):
            raise ValueError(
                "subvoxel_b0_offsets must be finite with shape (samples, px, pz)"
            )
        if int(num_offsets) != 1 or float(offset_spread) != 0.0:
            raise ValueError(
                "subvoxel_b0_offsets is mutually exclusive with num_offsets "
                "and offset_spread"
            )

    # Voxel positions use an integer center (px//2) so they sit on the DFT grid
    # that fftshift/ifftshift assume; a half-integer center would imprint a
    # (-1)^index half-pixel modulation on the reconstruction.
    x_axis = (np.arange(px) - px // 2) * (fov_x / px)
    z_axis = (np.arange(pz) - pz // 2) * (fov_z / pz)
    t2_particles = np.repeat(t2_arr.reshape(-1), int(walkers_per_cell))
    t1_particles = np.repeat(t1_arr.reshape(-1), int(walkers_per_cell))

    # k-space steps (rad/m) and the gradient "moments" gamma*G that realize them.
    dk_x = 2.0 * np.pi / fov_x
    dk_z = 2.0 * np.pi / fov_z
    moment_readout = px * dk_x / readout_time  # gamma*G_x during readout
    # Each echo is gradient-balanced: an x pre-dephase before the readout and an
    # x rewind after it return k_x to zero before the next 180, so the echo train
    # stays a clean Meiboom-Gill CPMG. The engine samples at the end of each
    # dwell, so sample m (0-based) sits at k_x = start + (m+1)*dk_x; pre-dephasing
    # to -(px//2 + 1)*dk_x lands the samples on the centered grid (m - px//2)*dk_x.
    moment_predephase = -(px // 2 + 1) * dk_x / phase_time
    moment_rewind = -(px - (px // 2 + 1)) * dk_x / phase_time  # k_x back to 0
    # Place the first 180 so the readout centre (k=0) sits at the spin-echo
    # refocus time; later echoes are then automatically centred. This makes the
    # sub-voxel B0 spread refocus at each echo, as a real spin echo does.
    pre_180_gap = max(
        0.0, phase_time + 0.5 * readout_time - 0.5 * excitation_duration
    )

    schedule = _phase_encode_schedule(pz, echo_train_length, phase_encode_order)
    num_shots = len(schedule)
    if refocusing:
        echo_spacing = refocusing_duration + 2.0 * phase_time + readout_time
    else:
        center_sample = (px // 2 + 1) / px
        echo_spacing = (
            0.5 * excitation_duration + phase_time + center_sample * readout_time
        )
    sub = int(substeps_per_interval)

    def _shot_steps(lines: list[int]) -> list[MotionSequenceStep]:
        steps: list[MotionSequenceStep] = [
            MotionSequenceStep(
                duration=excitation_duration,
                rf_amplitude=(0.5 * np.pi) / excitation_duration,
                rf_phase=np.pi / 2,
                substeps=max(1, sub),
                label="excitation_90",
            ),
        ]
        if refocusing and pre_180_gap > 0.0:
            steps.append(
                MotionSequenceStep(
                    duration=pre_180_gap, substeps=max(1, sub), label="te_centering"
                )
            )
        for line in lines:
            moment_pe = (line - pz // 2) * dk_z / phase_time
            if refocusing:
                steps.append(
                    MotionSequenceStep(
                        duration=refocusing_duration,
                        rf_amplitude=np.pi / refocusing_duration,
                        rf_phase=0.0,
                        substeps=max(1, sub),
                        label="refocus_180",
                    )
                )
            steps.extend(
                [
                    # x pre-dephase + z phase encode together (two-axis gradient).
                    MotionSequenceStep(
                        duration=phase_time,
                        gradient=(moment_predephase, moment_pe),
                        substeps=sub,
                        label="dephase_encode",
                    ),
                    MotionSequenceStep(
                        duration=readout_time,
                        gradient=(moment_readout, 0.0),
                        acquire=True,
                        num_samples=px,
                        substeps=sub,
                        label=f"readout_{line}",
                    ),
                    # x rewind + z rewind: return k-space to the origin pre-180.
                    MotionSequenceStep(
                        duration=phase_time,
                        gradient=(moment_rewind, -moment_pe),
                        substeps=sub,
                        label="rewind",
                    ),
                ]
            )
        return steps

    shot_steps = [(_shot_steps(lines), lines) for lines in schedule]

    # Sub-voxel B0 spread: average isochromats spaced over +/- offset_spread.
    if offset_maps is None:
        offsets = (
            np.zeros((1, px, pz), dtype=np.float64)
            if num_offsets == 1
            else np.broadcast_to(
                np.linspace(-offset_spread, offset_spread, num_offsets)[:, None, None],
                (num_offsets, px, pz),
            )
        )
        offset_model = "uniform"
        reported_spread = float(offset_spread)
    else:
        offsets = offset_maps
        offset_model = "spatial"
        reported_spread = float(np.max(np.abs(offset_maps)))
    if subvoxel_density_weights is None:
        density_weights = np.ones_like(offsets)
    else:
        density_weights = np.asarray(subvoxel_density_weights, dtype=np.float64)
        if (
            density_weights.shape != offsets.shape
            or np.any(~np.isfinite(density_weights))
            or np.any(density_weights < 0.0)
        ):
            raise ValueError(
                "subvoxel_density_weights must be finite, non-negative, and "
                "match the resolved subvoxel offsets"
            )
    kspace = np.zeros((px, pz), dtype=np.complex128)
    line_echo_index = np.zeros(pz, dtype=np.int64)
    line_echo_time = np.zeros(pz, dtype=np.float64)

    for offset_index, (offset, density_weight) in enumerate(
        zip(offsets, density_weights)
    ):
        sample_seed = None if seed is None else int(seed) + offset_index
        ensemble = initialize_ensemble_from_density(
            rho_arr * density_weight,
            x_axis,
            z_axis,
            walkers_per_cell=int(walkers_per_cell),
            diffusion_coefficient=diffusion,
            seed=sample_seed,
            jitter=bool(jitter),
        )
        motion_rng = np.random.default_rng(sample_seed)
        fields = make_motion_field_maps_2d(
            x_axis, z_axis,
            b0_map=b0_arr + offset,
            b1_tx_map=b1_tx_arr, b1_rx_map=b1_rx_arr,
        )
        for steps, lines in shot_steps:
            sequence = run_motion_sequence(
                ensemble, fields, steps,
                rng=motion_rng,
                t1=t1_particles, t2=t2_particles, default_substeps=max(1, sub),
            )
            signal = sequence.signal
            for echo_index, line in enumerate(lines):
                kspace[:, line] += signal[echo_index * px : (echo_index + 1) * px]
                line_echo_index[line] = echo_index
                line_echo_time[line] = (echo_index + 1) * echo_spacing
    kspace /= float(offsets.size)

    kspace3 = kspace[:, :, np.newaxis]
    image = reconstruct_image_from_kspace(kspace3, 0)[:, :, np.newaxis]
    return FrequencyEncodedImagingResult(
        kspace=kspace3,
        image=image,
        magnitude=np.abs(image),
        rho=rho_arr,
        echo_train_length=int(echo_train_length),
        num_shots=num_shots,
        line_echo_index=line_echo_index,
        line_echo_time=line_echo_time,
        fov=(fov_x, fov_z),
        echo_spacing=echo_spacing,
        method=(
            "rare"
            if echo_train_length > 1
            else ("spin_warp" if refocusing else "gradient_echo")
        ),
        num_offsets=int(offsets.shape[0]),
        offset_spread=reported_spread,
        offset_model=offset_model,
        diffusion_coefficient=diffusion,
        walkers_per_cell=int(walkers_per_cell),
    )


def run_spin_warp_imaging(rho, **kwargs) -> FrequencyEncodedImagingResult:
    """Simulate a spin-warp image (one spin echo per phase-encode line).

    This is the ``echo_train_length=1`` special case of RARE: every k_z line is
    acquired at the same echo time, so there is no RARE blurring (the reference
    image quality, at the cost of ``pz`` excitations).
    """

    kwargs.pop("echo_train_length", None)
    kwargs.pop("phase_encode_order", None)
    kwargs.pop("refocusing", None)
    return run_rare_imaging(rho, echo_train_length=1, refocusing=True, **kwargs)


def run_gradient_echo_imaging(rho, **kwargs) -> FrequencyEncodedImagingResult:
    """Simulate one Cartesian gradient echo per phase-encode line.

    Unlike ``run_spin_warp_imaging``, this sequence has no 180-degree pulse, so
    static susceptibility phase and intravoxel dephasing remain at readout.
    ``echo_spacing`` on the result stores the center-k-space echo time.
    """

    kwargs.pop("echo_train_length", None)
    kwargs.pop("phase_encode_order", None)
    kwargs.pop("refocusing", None)
    return run_rare_imaging(rho, echo_train_length=1, refocusing=False, **kwargs)


def _reconstruct_radial_samples(
    encoding_matrix: np.ndarray,
    samples: np.ndarray,
    shape: tuple[int, int],
    regularization: float,
) -> np.ndarray:
    matrix = np.asarray(encoding_matrix, dtype=np.complex128)
    data = np.asarray(samples, dtype=np.complex128)
    if data.shape != (matrix.shape[0],) or np.any(~np.isfinite(data)):
        raise ValueError("samples must be finite and match the radial trajectory")
    penalty_fraction = float(regularization)
    if not np.isfinite(penalty_fraction) or penalty_fraction < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    normal = matrix.conj().T @ matrix
    scale = float(np.linalg.norm(matrix, ord=2) ** 2)
    normal += penalty_fraction * scale * np.eye(matrix.shape[1])
    estimate = np.linalg.solve(normal, matrix.conj().T @ data)
    return estimate.reshape(shape)


def run_radial_ute_imaging(
    rho,
    *,
    t1_map=None,
    t2_map=None,
    b0_map=None,
    b1_tx_map=None,
    b1_rx_map=None,
    fov: tuple[float, float] = (0.02, 0.02),
    num_spokes: int | None = None,
    radial_samples: int | None = None,
    excitation_duration: float = 20e-6,
    readout_time: float = 0.4e-3,
    center_dwell: float = 1e-6,
    flip_angle: float = np.pi / 6.0,
    regularization: float = 1e-4,
    subvoxel_b0_offsets=None,
    subvoxel_density_weights=None,
    diffusion_coefficient: float = 0.0,
    walkers_per_cell: int = 1,
    seed: int | None = None,
    jitter: bool = False,
    gamma: float = 2.675e8,
    substeps_per_interval: int = 1,
) -> RadialUTEImagingResult:
    """Simulate a finite-pulse center-out radial short-TE acquisition.

    Each spoke starts from equilibrium, applies one finite RF excitation, takes
    a near-zero-gradient center sample, and immediately reads radially outward.
    This is a mechanistic UTE-like acquisition rather than a steady-state SPGR
    model: TR saturation and concentration-dependent T1 relaxivity are not
    inferred. Spatial subvoxel B0 offsets, water exclusion, and random-walker
    diffusion follow the Cartesian imaging conventions.
    """

    rho_arr, t1_arr, t2_arr, b0_arr, b1_tx_arr, b1_rx_arr = _resolve_maps(
        rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map
    )
    px, pz = rho_arr.shape
    fov_x, fov_z = float(fov[0]), float(fov[1])
    if fov_x <= 0.0 or fov_z <= 0.0:
        raise ValueError("fov entries must be positive")
    spokes = 2 * max(px, pz) if num_spokes is None else int(num_spokes)
    samples_per_spoke = (
        max(px, pz) // 2 if radial_samples is None else int(radial_samples)
    )
    if spokes < 4 or samples_per_spoke < 2:
        raise ValueError("num_spokes must be at least 4 and radial_samples at least 2")
    if excitation_duration <= 0.0 or readout_time <= 0.0 or center_dwell <= 0.0:
        raise ValueError("UTE timing values must be positive")
    if not np.isfinite(flip_angle) or not 0.0 < flip_angle <= np.pi:
        raise ValueError("flip_angle must be in (0, pi]")
    if int(walkers_per_cell) != walkers_per_cell or walkers_per_cell < 1:
        raise ValueError("walkers_per_cell must be a positive integer")
    diffusion = float(diffusion_coefficient)
    if not np.isfinite(diffusion) or diffusion < 0.0:
        raise ValueError("diffusion_coefficient must be finite and non-negative")

    if subvoxel_b0_offsets is None:
        offsets = np.zeros((1, px, pz), dtype=np.float64)
    else:
        offsets = np.asarray(subvoxel_b0_offsets, dtype=np.float64)
        if (
            offsets.ndim != 3
            or offsets.shape[1:] != rho_arr.shape
            or offsets.shape[0] < 1
            or np.any(~np.isfinite(offsets))
        ):
            raise ValueError(
                "subvoxel_b0_offsets must be finite with shape (samples, px, pz)"
            )
    if subvoxel_density_weights is None:
        density_weights = np.ones_like(offsets)
    else:
        density_weights = np.asarray(subvoxel_density_weights, dtype=np.float64)
        if (
            density_weights.shape != offsets.shape
            or np.any(~np.isfinite(density_weights))
            or np.any(density_weights < 0.0)
        ):
            raise ValueError(
                "subvoxel_density_weights must be finite, non-negative, and "
                "match subvoxel_b0_offsets"
            )

    x_axis = (np.arange(px) - px // 2) * (fov_x / px)
    z_axis = (np.arange(pz) - pz // 2) * (fov_z / pz)
    x_grid, z_grid = np.meshgrid(x_axis, z_axis, indexing="ij")
    positions = np.column_stack((x_grid.ravel(), z_grid.ravel()))
    dx = fov_x / px
    dz = fov_z / pz
    k_max = np.pi / min(dx, dz)
    radii = np.linspace(k_max / samples_per_spoke, k_max, samples_per_spoke)
    angles = np.linspace(0.0, 2.0 * np.pi, spokes, endpoint=False)
    trajectory_parts: list[np.ndarray] = []
    for angle in angles:
        direction = np.asarray((np.cos(angle), np.sin(angle)))
        trajectory_parts.append(np.zeros((1, 2)))
        trajectory_parts.append(radii[:, None] * direction[None, :])
    trajectory = np.vstack(trajectory_parts)
    encoding_matrix = np.exp(-1j * (trajectory @ positions.T))

    t1_particles = np.repeat(t1_arr.reshape(-1), int(walkers_per_cell))
    t2_particles = np.repeat(t2_arr.reshape(-1), int(walkers_per_cell))
    accumulated = np.zeros(trajectory.shape[0], dtype=np.complex128)
    substeps = max(1, int(substeps_per_interval))
    for offset_index, (offset, density_weight) in enumerate(
        zip(offsets, density_weights)
    ):
        sample_seed = None if seed is None else int(seed) + offset_index
        ensemble = initialize_ensemble_from_density(
            rho_arr * density_weight,
            x_axis,
            z_axis,
            walkers_per_cell=int(walkers_per_cell),
            diffusion_coefficient=diffusion,
            seed=sample_seed,
            jitter=bool(jitter),
        )
        fields = make_motion_field_maps_2d(
            x_axis,
            z_axis,
            b0_map=b0_arr + offset,
            b1_tx_map=b1_tx_arr,
            b1_rx_map=b1_rx_arr,
        )
        motion_rng = np.random.default_rng(sample_seed)
        cursor = 0
        for angle in angles:
            direction = np.asarray((np.cos(angle), np.sin(angle)))
            steps = (
                MotionSequenceStep(
                    duration=excitation_duration,
                    rf_amplitude=float(flip_angle) / excitation_duration,
                    rf_phase=np.pi / 2.0,
                    substeps=substeps,
                    label="ute_excitation",
                ),
                MotionSequenceStep(
                    duration=center_dwell,
                    acquire=True,
                    num_samples=1,
                    substeps=1,
                    label="ute_center",
                ),
                MotionSequenceStep(
                    duration=readout_time,
                    gradient=tuple((k_max / readout_time) * direction),
                    acquire=True,
                    num_samples=samples_per_spoke,
                    substeps=substeps,
                    label="ute_center_out_readout",
                ),
            )
            sequence = run_motion_sequence(
                ensemble,
                fields,
                steps,
                rng=motion_rng,
                t1=t1_particles,
                t2=t2_particles,
                default_substeps=substeps,
            )
            count = 1 + samples_per_spoke
            accumulated[cursor : cursor + count] += sequence.signal
            cursor += count
    accumulated /= float(offsets.shape[0])
    image = _reconstruct_radial_samples(
        encoding_matrix,
        accumulated,
        rho_arr.shape,
        regularization,
    )
    return RadialUTEImagingResult(
        samples=accumulated,
        trajectory_rad_m=trajectory,
        encoding_matrix=encoding_matrix,
        image=image,
        magnitude=np.abs(image),
        rho=rho_arr,
        fov=(fov_x, fov_z),
        num_spokes=spokes,
        radial_samples=samples_per_spoke,
        echo_time_s=0.5 * float(excitation_duration) + float(center_dwell),
        readout_time_s=float(readout_time),
        excitation_duration_s=float(excitation_duration),
        flip_angle_rad=float(flip_angle),
        regularization=float(regularization),
        num_offsets=int(offsets.shape[0]),
        offset_spread=float(np.max(np.abs(offsets))),
        diffusion_coefficient=diffusion,
        walkers_per_cell=int(walkers_per_cell),
    )


def imaging_slice_sensitivity(
    rho,
    *,
    center_frequency: float = 0.0,
    excitation_flip: float = np.pi / 2,
    excitation_duration: float = 100.0e-6,
    refocusing: bool = False,
    refocusing_flip: float = np.pi,
    refocusing_duration: float = 200.0e-6,
    t1_map=None,
    t2_map=None,
    b0_map=None,
    b1_tx_map=None,
    b1_rx_map=None,
    fov: tuple[float, float] = (0.02, 0.02),
) -> SliceSensitivityResult:
    """Map the real-space sensitive slice of an excitation in a non-uniform field.

    ``rho`` may be a spin-density array (with optional map keywords) or an
    ``ImagingFieldMaps`` container; only the B0/B1 maps are used. The excited
    transverse magnetization is computed from a rectangular RF pulse for every
    voxel at its own off-resonance ``b0_map - center_frequency`` (rad/s) and
    transmit-B1, so the slice profile (set by the pulse bandwidth ~ 1/duration)
    and the curvature (set by the B0 contours) emerge directly. The result is
    weighted by the receive B1. Set ``refocusing=True`` to also multiply by the
    refocusing efficiency of a 180-degree pulse (the spin-echo sensitive volume).

    The returned ``sensitivity`` is "neither flat nor uniform": it follows the
    curved iso-B0 contours and is shaded by the B1 maps.
    """

    from spin_dynamics.core.rotations import rf_matrix_elements

    rho_arr, _t1, _t2, b0_arr, b1_tx_arr, b1_rx_arr = _resolve_maps(
        rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map
    )
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("pulse durations must be positive")
    shape = b0_arr.shape
    off = b0_arr - float(center_frequency)

    w1_exc = (float(excitation_flip) / float(excitation_duration)) * b1_tx_arr
    excited = rf_matrix_elements(
        off.reshape(-1), w1_exc.reshape(-1), float(excitation_duration), np.pi / 2
    )
    excitation = np.abs(excited.R_m0).reshape(shape)  # |Mxy| from equilibrium
    sensitivity = excitation * b1_rx_arr
    if refocusing:
        w1_ref = (float(refocusing_flip) / float(refocusing_duration)) * b1_tx_arr
        refocused = rf_matrix_elements(
            off.reshape(-1), w1_ref.reshape(-1), float(refocusing_duration), 0.0
        )
        # |R_pm| is the M- -> M+ transfer (1 for an ideal on-resonance 180).
        sensitivity = sensitivity * np.abs(refocused.R_pm).reshape(shape)

    return SliceSensitivityResult(
        sensitivity=sensitivity,
        excitation=excitation,
        off_resonance=off,
        b0_map=b0_arr,
        b1_tx_map=b1_tx_arr,
        b1_rx_map=b1_rx_arr,
        center_frequency=float(center_frequency),
        excitation_flip=float(excitation_flip),
        excitation_duration=float(excitation_duration),
        refocusing=bool(refocusing),
        fov=(float(fov[0]), float(fov[1])),
    )
