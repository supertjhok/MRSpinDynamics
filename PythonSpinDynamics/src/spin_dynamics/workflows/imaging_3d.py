"""Three-dimensional imaging by slice-selective multi-slice acquisition.

Two runners produce a 3-D volume as a stack of slice-selective 2-D images:

* :func:`run_multislice_imaging` -- the **primary, true-3-D** path. A single 3-D
  walker ensemble lives in a 3-D ``MotionFieldMaps`` carrying the actual
  ``(B0, B1)`` field, and every slice is excited and read out by running that
  ensemble through the moving-isochromat engine. Because the slice pulse selects
  spins by their *total* off-resonance (gradient plus local B0), a nonuniform B0
  curves and displaces the excited slice, and it distorts the readout, exactly as
  in a real magnet -- nothing is assumed flat or uniform.

* :func:`run_multislice_imaging_separable` -- a **fast approximation**. The slice
  pulse is reduced to a 1-D through-plane weight ``w(y)`` and each slice's
  in-plane image is formed by the validated 2-D spin-warp workflow on the
  ``w``-weighted density. It ignores in-plane B0/B1 variation (no slice
  curvature), trading that fidelity for speed on large volumes.

Genuinely Fourier-encoded 3-D (slab-select plus a second phase encode
reconstructed with ``ifftn``) is a separate workflow left for later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.motion import (
    initialize_ensemble_from_domain,
    make_motion_field_maps,
)
from spin_dynamics.fields import SpatialDomain
from spin_dynamics.sequences.motion import MotionSequenceStep, run_motion_sequence
from spin_dynamics.workflows.imaging import reconstruct_image_from_kspace
from spin_dynamics.workflows.imaging_frequency import run_spin_warp_imaging
from spin_dynamics.workflows.slice_selective import (
    Window,
    make_slice_selective_excitation,
    slice_excitation_weights,
)


@dataclass(frozen=True)
class MultiSliceImagingResult:
    """Result of a 2-D multi-slice 3-D imaging simulation."""

    image: np.ndarray  # (nx, n_slices, nz) complex volume
    magnitude: np.ndarray  # (nx, n_slices, nz)
    kspace: np.ndarray  # (nx, n_slices, nz) per-slice in-plane k-space
    rho: np.ndarray  # input (nx, ny, nz) volume
    slice_positions: np.ndarray  # (n_slices,) through-plane positions imaged
    slice_axis_positions: np.ndarray  # (ny,) physical positions of the volume slices
    slice_profiles: np.ndarray  # (n_slices, ny) nominal |through-plane weight|
    fov: tuple[float, float, float]  # (x, y, z)
    slice_gradient: float
    method: str  # "engine_3d" or "separable"


def _prepare_volume(rho, slice_axis, fov, b0_map, b1_tx_map, b1_rx_map):
    """Validate inputs and reorder everything to ``(x, slice, z)`` layout."""

    volume = np.asarray(rho, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError("rho must be a 3-D volume")
    if not 0 <= int(slice_axis) < 3:
        raise ValueError("slice_axis must be 0, 1, or 2")
    fov = tuple(float(f) for f in fov)
    if len(fov) != 3 or any(f <= 0.0 for f in fov):
        raise ValueError("fov must contain three positive values")

    def _move(arr, name, default):
        if arr is None:
            return np.full(volume.shape, default, dtype=np.float64)
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != volume.shape:
            raise ValueError(f"{name} must have the same shape as rho")
        return arr

    axis = int(slice_axis)
    moved = np.moveaxis(volume, axis, 1)
    b0 = np.moveaxis(_move(b0_map, "b0_map", 0.0), axis, 1)
    b1_tx = np.moveaxis(_move(b1_tx_map, "b1_tx_map", 1.0), axis, 1)
    b1_rx = np.moveaxis(_move(b1_rx_map, "b1_rx_map", 1.0), axis, 1)
    in_plane_fov = tuple(f for i, f in enumerate(fov) if i != axis)
    slice_fov = fov[axis]
    return volume, moved, b0, b1_tx, b1_rx, in_plane_fov, slice_fov, fov


def _slice_centers(slice_positions, slice_axis_positions):
    if slice_positions is None:
        return slice_axis_positions.copy()
    centers = np.asarray(slice_positions, dtype=np.float64).reshape(-1)
    if centers.size == 0:
        raise ValueError("slice_positions must not be empty")
    return centers


def _nominal_profiles(centers, slice_axis_positions, slice_gradient, pulse):
    """Uniform-field through-plane weights for diagnostics (B0 = 0)."""

    profiles = np.zeros((centers.size, slice_axis_positions.size), dtype=np.float64)
    for s, center in enumerate(centers):
        profiles[s, :] = np.abs(
            slice_excitation_weights(
                slice_axis_positions,
                slice_gradient=slice_gradient,
                center=float(center),
                **pulse,
            )
        )
    return profiles


def run_multislice_imaging(
    rho,
    *,
    slice_gradient: float,
    slice_axis: int = 1,
    fov: tuple[float, float, float] = (0.02, 0.02, 0.02),
    slice_positions=None,
    b0_map=None,
    b1_tx_map=None,
    b1_rx_map=None,
    t1_map=None,
    t2_map=None,
    slice_duration: float = 1.0e-3,
    flip_angle: float = np.pi / 2,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
    readout_time: float = 2.0e-3,
    phase_time: float = 0.4e-3,
    refocusing_duration: float = 100.0e-6,
    substeps_per_interval: int = 1,
) -> MultiSliceImagingResult:
    """Acquire a 3-D volume by true 3-D slice-selective multi-slice imaging.

    A single 3-D walker ensemble is built from ``rho`` and placed in a 3-D
    ``MotionFieldMaps`` carrying ``b0_map``/``b1_tx_map``/``b1_rx_map`` (each a
    3-D volume the shape of ``rho``; B0 in rad/s, B1 relative). For every slice
    the engine plays a slice-selective excitation (a windowed-sinc RF with the
    slice gradient on ``slice_axis``, its carrier offset to place the slice) then
    a spin-warp readout (readout along one in-plane axis, phase encode along the
    other), filling that slice's 2-D k-space line by line. Because the slice is
    selected by total off-resonance, a nonuniform B0 curves/shifts it and warps
    the readout -- the real-magnet behavior.

    ``slice_axis`` (default 1) is the through-plane axis of ``rho``; the two
    remaining axes are imaged in-plane (readout then phase encode). Returns a
    :class:`MultiSliceImagingResult` whose volume is ``(nx, n_slices, nz)``.
    """

    if not np.isfinite(slice_gradient) or slice_gradient == 0.0:
        raise ValueError("slice_gradient must be a nonzero finite value")

    (
        volume, moved, b0, b1_tx, b1_rx, in_plane_fov, slice_fov, fov3,
    ) = _prepare_volume(rho, slice_axis, fov, b0_map, b1_tx_map, b1_rx_map)
    nx, ny, nz = moved.shape
    if min(nx, nz) < 2:
        raise ValueError("the two in-plane axes must each have at least 2 voxels")
    t1_moved = np.moveaxis(np.asarray(t1_map, dtype=np.float64), int(slice_axis), 1) \
        if t1_map is not None else None
    t2_moved = np.moveaxis(np.asarray(t2_map, dtype=np.float64), int(slice_axis), 1) \
        if t2_map is not None else None

    fov_x, fov_z = in_plane_fov
    x_axis = (np.arange(nx) - nx // 2) * (fov_x / nx)
    y_axis = (np.arange(ny) - ny // 2) * (slice_fov / ny)
    z_axis = (np.arange(nz) - nz // 2) * (fov_z / nz)

    domain = SpatialDomain((x_axis, y_axis, z_axis))
    fields = make_motion_field_maps(domain, b0_map=b0, b1_tx_map=b1_tx, b1_rx_map=b1_rx)
    base = initialize_ensemble_from_domain(domain, moved)
    t1_particles = np.inf if t1_moved is None else t1_moved.reshape(-1)
    t2_particles = np.inf if t2_moved is None else t2_moved.reshape(-1)

    centers = _slice_centers(slice_positions, y_axis)
    sub = max(1, int(substeps_per_interval))

    # k-space steps and the gradient moments (gamma*G) that realize them.
    dk_x = 2.0 * np.pi / fov_x
    dk_z = 2.0 * np.pi / fov_z
    moment_readout = nx * dk_x / readout_time
    moment_predephase = -(nx // 2 + 1) * dk_x / phase_time
    moment_rewind = -(nx - (nx // 2 + 1)) * dk_x / phase_time
    exc_total = slice_duration + (rephase_fraction * slice_duration if rephase else 0.0)
    pre_180_gap = max(0.0, phase_time + 0.5 * readout_time - 0.5 * exc_total)

    def _slice_excitation(center: float):
        steps = list(
            make_slice_selective_excitation(
                duration=slice_duration,
                slice_gradient=slice_gradient,
                flip_angle=flip_angle,
                slice_axis=1,  # the through-plane axis in (x, y, z)
                ndim=3,
                time_bandwidth=time_bandwidth,
                num_substeps=num_substeps,
                phase=np.pi / 2,
                window=window,
                rephase=rephase,
                rephase_fraction=rephase_fraction,
            )
        )
        carrier = -float(slice_gradient) * float(center)

        def detuning(time: float) -> float:
            return carrier if time <= exc_total + 1e-12 else 0.0

        return steps, detuning

    def _readout_line(line: int):
        moment_pe = (line - nz // 2) * dk_z / phase_time
        return [
            MotionSequenceStep(
                duration=refocusing_duration,
                gradient=(0.0, 0.0, 0.0),
                rf_amplitude=np.pi / refocusing_duration,
                rf_phase=0.0,
                substeps=sub,
                label="refocus_180",
            ),
            MotionSequenceStep(
                duration=phase_time,
                gradient=(moment_predephase, 0.0, moment_pe),
                substeps=sub,
                label="dephase_encode",
            ),
            MotionSequenceStep(
                duration=readout_time,
                gradient=(moment_readout, 0.0, 0.0),
                acquire=True,
                num_samples=nx,
                substeps=sub,
                label=f"readout_{line}",
            ),
            MotionSequenceStep(
                duration=phase_time,
                gradient=(moment_rewind, 0.0, -moment_pe),
                substeps=sub,
                label="rewind",
            ),
        ]

    image = np.zeros((nx, centers.size, nz), dtype=np.complex128)
    kspace = np.zeros((nx, centers.size, nz), dtype=np.complex128)

    for s, center in enumerate(centers):
        exc_steps, detuning = _slice_excitation(center)
        slice_kspace = np.zeros((nx, nz), dtype=np.complex128)
        for line in range(nz):
            steps = list(exc_steps)
            if pre_180_gap > 0.0:
                steps.append(
                    MotionSequenceStep(
                        duration=pre_180_gap,
                        gradient=(0.0, 0.0, 0.0),
                        substeps=sub,
                        label="te_centering",
                    )
                )
            steps.extend(_readout_line(line))
            sequence = run_motion_sequence(
                base,
                fields,
                steps,
                t1=t1_particles,
                t2=t2_particles,
                default_substeps=sub,
                detuning_waveform=detuning,
            )
            slice_kspace[:, line] = sequence.signal[:nx]
        plane = slice_kspace[:, :, np.newaxis]
        image[:, s, :] = reconstruct_image_from_kspace(plane, 0)
        kspace[:, s, :] = slice_kspace

    pulse = dict(
        duration=slice_duration, flip_angle=flip_angle,
        time_bandwidth=time_bandwidth, num_substeps=num_substeps,
        window=window, rephase=rephase, rephase_fraction=rephase_fraction,
    )
    profiles = _nominal_profiles(centers, y_axis, slice_gradient, pulse)
    return MultiSliceImagingResult(
        image=image,
        magnitude=np.abs(image),
        kspace=kspace,
        rho=volume,
        slice_positions=centers,
        slice_axis_positions=y_axis,
        slice_profiles=profiles,
        fov=fov3,
        slice_gradient=float(slice_gradient),
        method="engine_3d",
    )


def run_multislice_imaging_separable(
    rho,
    *,
    slice_gradient: float,
    slice_axis: int = 1,
    fov: tuple[float, float, float] = (0.02, 0.02, 0.02),
    slice_positions=None,
    slice_duration: float = 1.0e-3,
    flip_angle: float = np.pi / 2,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
    **in_plane_kwargs,
) -> MultiSliceImagingResult:
    """Fast separable multi-slice approximation (uniform in-plane field).

    The slice pulse is reduced to a 1-D through-plane weight ``w(y)`` and each
    slice's in-plane image is formed by :func:`run_spin_warp_imaging` on the
    ``w``-weighted density. This ignores in-plane B0/B1 variation -- the slice
    stays flat -- so use :func:`run_multislice_imaging` when field inhomogeneity
    (slice curvature, readout distortion) matters. Extra keyword arguments are
    forwarded to :func:`run_spin_warp_imaging`.
    """

    if not np.isfinite(slice_gradient) or slice_gradient == 0.0:
        raise ValueError("slice_gradient must be a nonzero finite value")
    (
        volume, moved, _b0, _tx, _rx, in_plane_fov, slice_fov, fov3,
    ) = _prepare_volume(rho, slice_axis, fov, None, None, None)
    nx, ny, nz = moved.shape
    if min(nx, nz) < 2:
        raise ValueError("the two in-plane axes must each have at least 2 voxels")

    y_axis = (np.arange(ny) - ny // 2) * (slice_fov / ny)
    centers = _slice_centers(slice_positions, y_axis)
    in_plane_kwargs.setdefault("excitation_duration", 50.0e-6)
    pulse = dict(
        duration=slice_duration, flip_angle=flip_angle,
        time_bandwidth=time_bandwidth, num_substeps=num_substeps,
        window=window, rephase=rephase, rephase_fraction=rephase_fraction,
    )

    image = np.zeros((nx, centers.size, nz), dtype=np.complex128)
    kspace = np.zeros((nx, centers.size, nz), dtype=np.complex128)
    profiles = _nominal_profiles(centers, y_axis, slice_gradient, pulse)

    for s in range(centers.size):
        weights = profiles[s, :]
        effective_density = np.tensordot(weights, moved, axes=([0], [1]))
        slice_result = run_spin_warp_imaging(
            effective_density, fov=in_plane_fov, **in_plane_kwargs
        )
        image[:, s, :] = slice_result.image[:, :, 0]
        kspace[:, s, :] = slice_result.kspace[:, :, 0]

    return MultiSliceImagingResult(
        image=image,
        magnitude=np.abs(image),
        kspace=kspace,
        rho=volume,
        slice_positions=centers,
        slice_axis_positions=y_axis,
        slice_profiles=profiles,
        fov=fov3,
        slice_gradient=float(slice_gradient),
        method="separable",
    )
