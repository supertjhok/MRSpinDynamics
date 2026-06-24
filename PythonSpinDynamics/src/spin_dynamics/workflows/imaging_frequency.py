"""Frequency-encoded imaging: spin-warp and RARE / fast spin echo.

The existing CPMG imaging workflows fill k-space one point per phase-encode
step. These workflows add a *readout* (frequency-encode) gradient applied during
acquisition, so each echo samples a whole k-space line. Built on the Lagrangian
motion engine (one static isochromat per voxel), they support arbitrary
two-axis gradient waveforms, which the scalar-gradient arbitrary-pulse kernels
cannot express.

* ``run_spin_warp_imaging`` -- a spin echo per phase-encode line: readout along
  x, phase encode along z. One image needs ``pz`` excitations.
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

    def reconstruct(self) -> np.ndarray:
        """Return the complex image (alias for the stored reconstruction)."""

        return self.image[:, :, 0]


def _maps(rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map):
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
    gamma: float = 2.675e8,
    substeps_per_interval: int = 1,
) -> FrequencyEncodedImagingResult:
    """Simulate a RARE / fast-spin-echo frequency-encoded image.

    Readout is along x (frequency encode) and the phase encode is along z. Each
    CPMG echo reads one k_z line, so ``echo_train_length`` lines are acquired per
    excitation and the image needs ``ceil(pz / echo_train_length)`` shots. The
    T2 decay across the train weights the phase-encode lines, which broadens the
    point-spread function (RARE blurring).
    """

    rho_arr, t1_arr, t2_arr, b0_arr, b1_tx_arr, b1_rx_arr = _maps(
        rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map
    )
    px, pz = rho_arr.shape
    fov_x, fov_z = float(fov[0]), float(fov[1])
    if fov_x <= 0.0 or fov_z <= 0.0:
        raise ValueError("fov entries must be positive")
    if readout_time <= 0.0 or phase_time <= 0.0:
        raise ValueError("readout_time and phase_time must be positive")

    # Voxel positions use an integer center (px//2) so they sit on the DFT grid
    # that fftshift/ifftshift assume; a half-integer center would imprint a
    # (-1)^index half-pixel modulation on the reconstruction.
    x_axis = (np.arange(px) - px // 2) * (fov_x / px)
    z_axis = (np.arange(pz) - pz // 2) * (fov_z / pz)
    fields = make_motion_field_maps_2d(
        x_axis, z_axis, b0_map=b0_arr, b1_tx_map=b1_tx_arr, b1_rx_map=b1_rx_arr
    )
    ensemble = initialize_ensemble_from_density(
        rho_arr, x_axis, z_axis, walkers_per_cell=1, diffusion_coefficient=0.0
    )
    t2_particles = t2_arr.reshape(-1)
    t1_particles = t1_arr.reshape(-1)

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

    schedule = _phase_encode_schedule(pz, echo_train_length, phase_encode_order)
    num_shots = len(schedule)
    echo_spacing = refocusing_duration + 2.0 * phase_time + readout_time

    kspace = np.zeros((px, pz), dtype=np.complex128)
    line_echo_index = np.zeros(pz, dtype=np.int64)
    line_echo_time = np.zeros(pz, dtype=np.float64)
    sub = int(substeps_per_interval)

    for lines in schedule:
        steps: list[MotionSequenceStep] = [
            MotionSequenceStep(
                duration=excitation_duration,
                rf_amplitude=(0.5 * np.pi) / excitation_duration,
                rf_phase=np.pi / 2,
                substeps=max(1, sub),
                label="excitation_90",
            ),
        ]
        for echo_index, line in enumerate(lines):
            moment_pe = (line - pz // 2) * dk_z / phase_time
            steps.extend(
                [
                    MotionSequenceStep(
                        duration=refocusing_duration,
                        rf_amplitude=np.pi / refocusing_duration,
                        rf_phase=0.0,
                        substeps=max(1, sub),
                        label="refocus_180",
                    ),
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

        sequence = run_motion_sequence(
            ensemble,
            fields,
            steps,
            t1=t1_particles,
            t2=t2_particles,
            default_substeps=max(1, sub),
        )
        signal = sequence.signal
        for echo_index, line in enumerate(lines):
            kspace[:, line] = signal[echo_index * px : (echo_index + 1) * px]
            line_echo_index[line] = echo_index
            line_echo_time[line] = (echo_index + 1) * echo_spacing

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
        method="rare" if echo_train_length > 1 else "spin_warp",
    )


def run_spin_warp_imaging(rho, **kwargs) -> FrequencyEncodedImagingResult:
    """Simulate a spin-warp image (one spin echo per phase-encode line).

    This is the ``echo_train_length=1`` special case of RARE: every k_z line is
    acquired at the same echo time, so there is no RARE blurring (the reference
    image quality, at the cost of ``pz`` excitations).
    """

    kwargs.pop("echo_train_length", None)
    kwargs.pop("phase_encode_order", None)
    return run_rare_imaging(rho, echo_train_length=1, **kwargs)
