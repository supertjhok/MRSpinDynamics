"""Susceptibility-aware spin-echo imaging of magnetic particle clouds.

This module is the physical-contrast counterpart to
``electropermanent_particle_imaging``.  The baseline workflow treats particle
density as directly detectable signal.  Here the particles carry no signal of
their own.  Their field-dependent magnetic moments perturb the tissue ``B0``
field, and a finite-duration 90--180 spin-warp acquisition converts that
perturbation into excitation/refocusing loss, distortion, and intravoxel
dephasing.

The model uses uniformly magnetized equivalent spheres and a paired
particle-free reference acquisition.  It remains a mesoscopic simulation: it
does not model microscopic water exchange, Néel/Brownian relaxation, particle
aggregation during the acquisition, or a calibrated receive chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import GAMMA_PROTON
from spin_dynamics.workflows.electropermanent_imaging import NonlinearEPMEncoding
from spin_dynamics.workflows.electropermanent_transport import (
    MU0,
    MagnetizationModel,
    SuperparamagneticParticle,
)
from spin_dynamics.workflows.imaging import reconstruct_image_from_kspace
from spin_dynamics.workflows.imaging_frequency import (
    FrequencyEncodedImagingResult,
    run_spin_warp_imaging,
)

__all__ = [
    "ParticleSpinEchoEstimate2D",
    "ParticleSusceptibilitySpinEchoResult",
    "estimate_particles_from_spin_echo_contrast",
    "particle_dipole_field_samples",
    "run_epm_particle_susceptibility_spin_echo",
]


def _axis(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 2
        or np.any(~np.isfinite(axis))
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, one-dimensional, and increasing")
    steps = np.diff(axis)
    if not np.allclose(steps, np.median(steps), rtol=1e-6, atol=0.0):
        raise ValueError(f"{name} must be uniformly spaced for spin-warp imaging")
    return axis


def _positions(values: np.ndarray) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[0] < 1
        or positions.shape[1] != 2
        or np.any(~np.isfinite(positions))
    ):
        raise ValueError("positions_m must be finite with shape (particles, 2)")
    return positions


def _map(values: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def _bilinear_sample(
    image: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    x = np.clip(positions[:, 0], x_axis[0], x_axis[-1])
    y = np.clip(positions[:, 1], y_axis[0], y_axis[-1])
    ix = np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, x_axis.size - 2)
    iy = np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, y_axis.size - 2)
    fx = (x - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
    fy = (y - y_axis[iy]) / (y_axis[iy + 1] - y_axis[iy])
    return (
        image[iy, ix] * (1.0 - fx) * (1.0 - fy)
        + image[iy, ix + 1] * fx * (1.0 - fy)
        + image[iy + 1, ix] * (1.0 - fx) * fy
        + image[iy + 1, ix + 1] * fx * fy
    )


def particle_dipole_field_samples(
    positions_m: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    particle: SuperparamagneticParticle,
    background_field_t: np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    subvoxel_grid_size: int = 3,
    magnetization_model: MagnetizationModel = "langevin",
) -> np.ndarray:
    """Return subvoxel samples of particle-induced ``delta B0`` in tesla.

    Each particle is represented by a uniformly magnetized equivalent sphere.
    Its moment is obtained from the local EPM field and the same magnetization
    law used by the transport model.  The returned array has shape
    ``(subvoxel_grid_size**2, y, x)``.
    """

    x_axis = _axis(x_m, "x_m")
    y_axis = _axis(y_m, "y_m")
    positions = _positions(positions_m)
    shape = (y_axis.size, x_axis.size)
    background = _map(background_field_t, shape, "background_field_t")
    if (
        np.any(positions[:, 0] < x_axis[0])
        or np.any(positions[:, 0] > x_axis[-1])
        or np.any(positions[:, 1] < y_axis[0])
        or np.any(positions[:, 1] > y_axis[-1])
    ):
        raise ValueError("positions_m must lie inside the imaging axes")
    if (
        int(subvoxel_grid_size) != subvoxel_grid_size
        or subvoxel_grid_size < 1
        or subvoxel_grid_size > 9
    ):
        raise ValueError("subvoxel_grid_size must be an integer from 1 through 9")
    direction = np.asarray(field_direction, dtype=np.float64)
    if direction.shape != (3,) or np.any(~np.isfinite(direction)):
        raise ValueError("field_direction must be a finite 3-vector")
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("field_direction must be nonzero")
    direction /= norm

    local_field = _bilinear_sample(background, x_axis, y_axis, positions)
    magnetization = particle.magnetization_a_m(
        np.abs(local_field), model=magnetization_model
    )
    signed_moment = (
        np.sign(local_field)
        * magnetization
        * particle.magnetic_volume_m3
    )
    signed_moment = np.where(local_field == 0.0, magnetization, signed_moment)

    nx = int(subvoxel_grid_size)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    offsets_x = ((np.arange(nx) + 0.5) / nx - 0.5) * dx
    offsets_y = ((np.arange(nx) + 0.5) / nx - 0.5) * dy
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    samples: list[np.ndarray] = []
    radius = particle.magnetic_core_radius_m
    prefactor = MU0 / (4.0 * np.pi)
    for offset_y in offsets_y:
        for offset_x in offsets_x:
            rx = x_grid[..., None] + offset_x - positions[:, 0]
            ry = y_grid[..., None] + offset_y - positions[:, 1]
            rz = np.zeros_like(rx)
            r2 = rx**2 + ry**2 + rz**2
            inside = r2 <= radius**2
            safe_r2 = np.where(inside, radius**2, r2)
            r = np.sqrt(safe_r2)
            projection = (
                rx * direction[0] + ry * direction[1] + rz * direction[2]
            ) / r
            outside_field = (
                prefactor
                * signed_moment
                * (3.0 * projection**2 - 1.0)
                / r**3
            )
            inside_field = (
                (2.0 / 3.0)
                * MU0
                * np.sign(signed_moment)
                * magnetization
            )
            samples.append(np.sum(np.where(inside, inside_field, outside_field), axis=2))
    return np.asarray(samples)


def _particle_water_mask_samples(
    positions: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    radius_m: float,
    subvoxel_grid_size: int,
) -> np.ndarray:
    """Return one water mask per subvoxel sample (particle cores are False)."""

    nx = int(subvoxel_grid_size)
    dx = float(np.median(np.diff(x_axis)))
    dy = float(np.median(np.diff(y_axis)))
    offsets_x = ((np.arange(nx) + 0.5) / nx - 0.5) * dx
    offsets_y = ((np.arange(nx) + 0.5) / nx - 0.5) * dy
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    masks: list[np.ndarray] = []
    for offset_y in offsets_y:
        for offset_x in offsets_x:
            distance_squared = (
                (x_grid[..., None] + offset_x - positions[:, 0]) ** 2
                + (y_grid[..., None] + offset_y - positions[:, 1]) ** 2
            )
            masks.append(np.all(distance_squared > radius_m**2, axis=2))
    return np.asarray(masks, dtype=np.float64)


@dataclass(frozen=True)
class ParticleSpinEchoEstimate2D:
    """Image-derived susceptibility foci and contrast-weighted control state."""

    contrast_image: np.ndarray
    positions_m: np.ndarray
    centroid_m: np.ndarray
    uncaptured_centroid_m: np.ndarray
    target_mask: np.ndarray
    capture_fraction: float
    resolved_focus_count: int
    support_threshold: float


@dataclass(frozen=True)
class ParticleSusceptibilitySpinEchoResult:
    """Paired spin-echo acquisitions and truth-only validation diagnostics."""

    reference_acquisition: FrequencyEncodedImagingResult
    particle_acquisition: FrequencyEncodedImagingResult
    reference_image: np.ndarray
    particle_image: np.ndarray
    signed_contrast_image: np.ndarray
    contrast_image: np.ndarray
    background_field_t: np.ndarray
    particle_delta_b0_samples_t: np.ndarray
    particle_water_mask_samples: np.ndarray
    estimate: ParticleSpinEchoEstimate2D
    ground_truth_positions_m: np.ndarray
    ground_truth_captured: np.ndarray | None
    target_center_m: np.ndarray
    target_radius_m: float
    snr_db: float | None

    @property
    def echo_time_s(self) -> float:
        """Time of the single acquired spin echo."""

        return self.particle_acquisition.echo_spacing

    @property
    def centroid_error_m(self) -> float:
        """Distance between contrast and true ensemble centroids."""

        truth = np.mean(self.ground_truth_positions_m, axis=0)
        return float(np.linalg.norm(self.estimate.centroid_m - truth))

    @property
    def ground_truth_capture_fraction(self) -> float:
        if self.ground_truth_captured is not None:
            return float(np.mean(self.ground_truth_captured))
        distance = np.linalg.norm(
            self.ground_truth_positions_m - self.target_center_m,
            axis=1,
        )
        return float(np.mean(distance <= self.target_radius_m))

    @property
    def capture_fraction_error(self) -> float:
        return self.estimate.capture_fraction - self.ground_truth_capture_fraction


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for row, col in np.argwhere(mask):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        cells: list[tuple[int, int]] = []
        while stack:
            current_row, current_col = stack.pop()
            cells.append((current_row, current_col))
            for next_row in range(
                max(0, current_row - 1), min(height, current_row + 2)
            ):
                for next_col in range(
                    max(0, current_col - 1), min(width, current_col + 2)
                ):
                    if mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
        components.append(np.asarray(cells, dtype=np.int64))
    return components


def estimate_particles_from_spin_echo_contrast(
    contrast_image: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    *,
    target_center_m: Sequence[float],
    target_radius_m: float,
    support_threshold_fraction: float = 0.10,
) -> ParticleSpinEchoEstimate2D:
    """Estimate susceptibility foci and their distribution from signal loss."""

    x_axis = _axis(x_m, "x_m")
    y_axis = _axis(y_m, "y_m")
    contrast = _map(
        contrast_image,
        (y_axis.size, x_axis.size),
        "contrast_image",
    )
    if np.any(contrast < 0.0):
        raise ValueError("contrast_image must be non-negative")
    fraction = float(support_threshold_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("support_threshold_fraction must be in [0, 1)")
    center = np.asarray(target_center_m, dtype=np.float64)
    radius = float(target_radius_m)
    if center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("target_center_m must be a finite 2-vector")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("target_radius_m must be finite and positive")
    peak = float(np.max(contrast))
    if peak <= 0.0:
        raise ValueError("contrast_image must contain positive susceptibility contrast")
    threshold = fraction * peak
    recovered = np.maximum(contrast - threshold, 0.0)
    total = float(np.sum(recovered))
    if total <= 0.0:
        raise ValueError("support threshold removed all susceptibility contrast")
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    centroid = np.asarray(
        (
            float(np.sum(recovered * x_grid) / total),
            float(np.sum(recovered * y_grid) / total),
        )
    )
    target_mask = (
        (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= radius**2
    )
    capture_fraction = float(np.sum(recovered[target_mask]) / total)
    outside = np.where(target_mask, 0.0, recovered)
    outside_total = float(np.sum(outside))
    if outside_total <= np.finfo(np.float64).eps * total:
        uncaptured_centroid = center.copy()
    else:
        uncaptured_centroid = np.asarray(
            (
                float(np.sum(outside * x_grid) / outside_total),
                float(np.sum(outside * y_grid) / outside_total),
            )
        )
    positions: list[np.ndarray] = []
    for component in _connected_components(recovered > 0.0):
        rows, cols = component[:, 0], component[:, 1]
        weights = recovered[rows, cols]
        positions.append(
            np.asarray(
                (
                    float(np.average(x_axis[cols], weights=weights)),
                    float(np.average(y_axis[rows], weights=weights)),
                )
            )
        )
    return ParticleSpinEchoEstimate2D(
        contrast_image=recovered,
        positions_m=np.asarray(positions),
        centroid_m=centroid,
        uncaptured_centroid_m=uncaptured_centroid,
        target_mask=target_mask,
        capture_fraction=capture_fraction,
        resolved_focus_count=len(positions),
        support_threshold=threshold,
    )


def _noisy_image(
    acquisition: FrequencyEncodedImagingResult,
    *,
    snr_db: float | None,
    seed: int,
) -> np.ndarray:
    kspace = acquisition.kspace.copy()
    if snr_db is not None:
        if not np.isfinite(snr_db) or snr_db <= 0.0:
            raise ValueError("snr_db must be finite and positive")
        rms = float(np.sqrt(np.mean(np.abs(kspace) ** 2)))
        noise_std = rms * 10.0 ** (-float(snr_db) / 20.0) / np.sqrt(2.0)
        rng = np.random.default_rng(seed)
        kspace += noise_std * (
            rng.normal(size=kspace.shape) + 1j * rng.normal(size=kspace.shape)
        )
    return reconstruct_image_from_kspace(kspace, 0).T


def run_epm_particle_susceptibility_spin_echo(
    encoding: NonlinearEPMEncoding,
    positions_m: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    particle: SuperparamagneticParticle,
    proton_density: np.ndarray,
    t1_s: np.ndarray,
    t2_s: np.ndarray,
    *,
    target_center_m: Sequence[float],
    target_radius_m: float,
    imaging_state_index: int = 0,
    subvoxel_grid_size: int = 3,
    magnetization_model: MagnetizationModel = "langevin",
    excitation_duration_s: float = 100e-6,
    refocusing_duration_s: float = 200e-6,
    readout_time_s: float = 2e-3,
    phase_encoding_time_s: float = 0.4e-3,
    water_diffusion_coefficient_m2_s: float = 2.3e-9,
    water_walkers_per_voxel: int = 4,
    sequence_substeps: int = 2,
    support_threshold_fraction: float = 0.10,
    snr_db: float | None = 40.0,
    seed: int = 0,
    ground_truth_captured: Sequence[bool] | np.ndarray | None = None,
) -> ParticleSusceptibilitySpinEchoResult:
    """Acquire paired reference/particle spin echoes and estimate particle foci."""

    x_axis = _axis(x_m, "x_m")
    y_axis = _axis(y_m, "y_m")
    positions = _positions(positions_m)
    shape = (y_axis.size, x_axis.size)
    if tuple(encoding.image_shape) != shape:
        raise ValueError("encoding image shape must match x_m and y_m")
    density = _map(proton_density, shape, "proton_density")
    t1_map = _map(t1_s, shape, "t1_s")
    t2_map = _map(t2_s, shape, "t2_s")
    if np.any(density < 0.0) or np.any(t1_map <= 0.0) or np.any(t2_map <= 0.0):
        raise ValueError("density must be non-negative and relaxation maps positive")
    state_index = int(imaging_state_index)
    if not 0 <= state_index < encoding.encoding_count:
        raise ValueError("imaging_state_index is out of range")
    background = encoding.projected_fields_t[state_index].reshape(shape)
    reference_index = encoding.reference_point_index
    reference_field = float(background.ravel()[reference_index])
    background_offresonance = encoding.gamma_rad_s_t * (background - reference_field)
    particle_fields = particle_dipole_field_samples(
        positions,
        x_axis,
        y_axis,
        particle,
        background,
        field_direction=encoding.field_direction,
        subvoxel_grid_size=subvoxel_grid_size,
        magnetization_model=magnetization_model,
    )
    particle_offsets = float(GAMMA_PROTON) * particle_fields
    particle_water_masks = _particle_water_mask_samples(
        positions,
        x_axis,
        y_axis,
        particle.magnetic_core_radius_m,
        subvoxel_grid_size,
    )
    fov = (
        float(np.median(np.diff(x_axis))) * x_axis.size,
        float(np.median(np.diff(y_axis))) * y_axis.size,
    )
    common = dict(
        t1_map=t1_map.T,
        t2_map=t2_map.T,
        b0_map=background_offresonance.T,
        fov=fov,
        readout_time=readout_time_s,
        phase_time=phase_encoding_time_s,
        excitation_duration=excitation_duration_s,
        refocusing_duration=refocusing_duration_s,
        gamma=float(GAMMA_PROTON),
        diffusion_coefficient=water_diffusion_coefficient_m2_s,
        walkers_per_cell=water_walkers_per_voxel,
        seed=seed + 2,
        jitter=water_walkers_per_voxel > 1,
        substeps_per_interval=sequence_substeps,
    )
    transposed_offsets = np.transpose(particle_offsets, (0, 2, 1))
    reference_acquisition = run_spin_warp_imaging(
        density.T,
        subvoxel_b0_offsets=np.zeros_like(transposed_offsets),
        **common,
    )
    particle_acquisition = run_spin_warp_imaging(
        density.T,
        subvoxel_b0_offsets=transposed_offsets,
        subvoxel_density_weights=np.transpose(particle_water_masks, (0, 2, 1)),
        **common,
    )
    reference_image = _noisy_image(
        reference_acquisition,
        snr_db=snr_db,
        seed=seed,
    )
    particle_image = _noisy_image(
        particle_acquisition,
        snr_db=snr_db,
        seed=seed + 1,
    )
    signed_contrast = np.abs(reference_image) - np.abs(particle_image)
    contrast = np.maximum(signed_contrast, 0.0)
    estimate = estimate_particles_from_spin_echo_contrast(
        contrast,
        x_axis,
        y_axis,
        target_center_m=target_center_m,
        target_radius_m=target_radius_m,
        support_threshold_fraction=support_threshold_fraction,
    )
    captured = None
    if ground_truth_captured is not None:
        captured = np.asarray(ground_truth_captured, dtype=bool)
        if captured.shape != (positions.shape[0],):
            raise ValueError("ground_truth_captured must match positions_m")
    return ParticleSusceptibilitySpinEchoResult(
        reference_acquisition=reference_acquisition,
        particle_acquisition=particle_acquisition,
        reference_image=reference_image,
        particle_image=particle_image,
        signed_contrast_image=signed_contrast,
        contrast_image=contrast,
        background_field_t=background.copy(),
        particle_delta_b0_samples_t=particle_fields,
        particle_water_mask_samples=particle_water_masks,
        estimate=estimate,
        ground_truth_positions_m=positions.copy(),
        ground_truth_captured=None if captured is None else captured.copy(),
        target_center_m=np.asarray(target_center_m, dtype=np.float64).copy(),
        target_radius_m=float(target_radius_m),
        snr_db=None if snr_db is None else float(snr_db),
    )
