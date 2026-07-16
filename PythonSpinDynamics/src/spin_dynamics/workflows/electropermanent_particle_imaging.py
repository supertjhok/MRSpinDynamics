"""Particle-sensitive nonlinear EPM imaging and state estimation.

The transport simulator represents a particle cloud with continuous positions.
This module converts that hidden state into a calibrated concentration image,
passes the image through the nonlinear EPM acquisition/reconstruction path, and
estimates the particle state from the reconstruction alone.  Ground truth is
retained on the result only for simulation diagnostics; control code should use
``estimate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.workflows.electropermanent_imaging import (
    EPMNonlinearImagingResult,
    NonlinearEPMEncoding,
    run_epm_nonlinear_imaging,
)

__all__ = [
    "EPMParticleImagingResult",
    "ParticleStateEstimate2D",
    "estimate_particle_state_from_image",
    "particle_distribution_image",
    "run_epm_particle_imaging",
]


def _strict_axis(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 2
        or np.any(~np.isfinite(axis))
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, one-dimensional, and increasing")
    return axis


def _positions(values: np.ndarray, name: str) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or np.any(~np.isfinite(positions)):
        raise ValueError(f"{name} must have shape (n_particles, 2) and be finite")
    if positions.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one particle")
    return positions


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def particle_distribution_image(
    positions_m: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    *,
    signal_per_particle: float = 1.0,
) -> np.ndarray:
    """Deposit continuous particle positions onto a Cartesian image grid.

    Cloud-in-cell (bilinear) deposition preserves total calibrated signal and
    the first spatial moment for particles inside the grid.  The resulting
    image is the hidden forward-model input, not an estimator output.
    """

    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    positions = _positions(positions_m, "positions_m")
    scale = _positive(signal_per_particle, "signal_per_particle")
    tolerance = 16.0 * np.finfo(np.float64).eps * max(
        1.0,
        float(np.max(np.abs(x_axis))),
        float(np.max(np.abs(y_axis))),
    )
    if (
        np.any(positions[:, 0] < x_axis[0] - tolerance)
        or np.any(positions[:, 0] > x_axis[-1] + tolerance)
        or np.any(positions[:, 1] < y_axis[0] - tolerance)
        or np.any(positions[:, 1] > y_axis[-1] + tolerance)
    ):
        raise ValueError("positions_m must lie inside the image axes")
    x_values = np.clip(positions[:, 0], x_axis[0], x_axis[-1])
    y_values = np.clip(positions[:, 1], y_axis[0], y_axis[-1])
    x_lower = np.clip(np.searchsorted(x_axis, x_values, side="right") - 1, 0, x_axis.size - 2)
    y_lower = np.clip(np.searchsorted(y_axis, y_values, side="right") - 1, 0, y_axis.size - 2)
    x_fraction = (x_values - x_axis[x_lower]) / (x_axis[x_lower + 1] - x_axis[x_lower])
    y_fraction = (y_values - y_axis[y_lower]) / (y_axis[y_lower + 1] - y_axis[y_lower])

    image = np.zeros((y_axis.size, x_axis.size), dtype=np.float64)
    for x_offset, y_offset, weights in (
        (0, 0, (1.0 - x_fraction) * (1.0 - y_fraction)),
        (1, 0, x_fraction * (1.0 - y_fraction)),
        (0, 1, (1.0 - x_fraction) * y_fraction),
        (1, 1, x_fraction * y_fraction),
    ):
        np.add.at(
            image,
            (y_lower + y_offset, x_lower + x_offset),
            scale * weights,
        )
    return image


@dataclass(frozen=True)
class ParticleStateEstimate2D:
    """Image-derived particle distribution and control summary.

    ``positions_m`` are representative image-resolved locations obtained by
    deterministic resampling of the recovered concentration.  They describe a
    distribution and do not preserve individual particle identities.
    """

    concentration_image: np.ndarray
    positions_m: np.ndarray
    centroid_m: np.ndarray
    uncaptured_centroid_m: np.ndarray
    target_mask: np.ndarray
    capture_fraction: float
    particle_count: int
    total_signal: float
    signal_per_particle: float
    support_threshold: float
    spatial_resolution_m: tuple[float, float]
    target_partial_volume_sensitivity: float


@dataclass(frozen=True)
class EPMParticleImagingResult:
    """Particle-sensitive acquisition, estimate, and truth-only diagnostics."""

    imaging: EPMNonlinearImagingResult
    estimate: ParticleStateEstimate2D
    ground_truth_positions_m: np.ndarray
    ground_truth_captured: np.ndarray | None
    target_center_m: np.ndarray
    target_radius_m: float

    @property
    def ground_truth_centroid_m(self) -> np.ndarray:
        """True ensemble centroid, exposed only for simulation diagnostics."""

        return np.mean(self.ground_truth_positions_m, axis=0)

    @property
    def ground_truth_capture_fraction(self) -> float:
        """True target capture used only to score the image-derived estimate."""

        if self.ground_truth_captured is not None:
            return float(np.mean(self.ground_truth_captured))
        distance = np.linalg.norm(
            self.ground_truth_positions_m - self.target_center_m,
            axis=1,
        )
        return float(np.mean(distance <= self.target_radius_m))

    @property
    def centroid_error_m(self) -> float:
        """Distance between estimated and ground-truth ensemble centroids."""

        return float(np.linalg.norm(self.estimate.centroid_m - self.ground_truth_centroid_m))

    @property
    def capture_fraction_error(self) -> float:
        """Estimated minus ground-truth target occupancy."""

        return self.estimate.capture_fraction - self.ground_truth_capture_fraction


def estimate_particle_state_from_image(
    reconstructed_image: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    *,
    target_center_m: Sequence[float],
    target_radius_m: float,
    signal_per_particle: float = 1.0,
    support_threshold_fraction: float = 0.01,
    boundary_capture_correction: bool = False,
) -> ParticleStateEstimate2D:
    """Estimate particle count, positions, centroid, and occupancy from an image.

    ``boundary_capture_correction`` applies a grid/target-only calibration for a
    model that immobilizes particles at first target entry.  It does not inspect
    ground-truth particle positions.
    """

    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    image = np.asarray(reconstructed_image, dtype=np.float64)
    if image.shape != (y_axis.size, x_axis.size) or np.any(~np.isfinite(image)):
        raise ValueError("reconstructed_image must be finite and match the axes")
    if np.any(image < 0.0):
        raise ValueError("reconstructed_image must be non-negative")
    center = np.asarray(target_center_m, dtype=np.float64)
    if center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("target_center_m must be a finite 2-vector")
    radius = _positive(target_radius_m, "target_radius_m")
    scale = _positive(signal_per_particle, "signal_per_particle")
    fraction = float(support_threshold_fraction)
    if not np.isfinite(fraction) or not 0.0 <= fraction < 1.0:
        raise ValueError("support_threshold_fraction must be in [0, 1)")
    peak = float(np.max(image))
    if peak <= 0.0:
        raise ValueError("reconstructed_image must contain positive particle signal")
    threshold = fraction * peak
    concentration = np.maximum(image - threshold, 0.0)
    total = float(np.sum(concentration))
    if total <= 0.0:
        raise ValueError("support threshold removed all reconstructed particle signal")

    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    resolution = (
        float(np.median(np.diff(x_axis))),
        float(np.median(np.diff(y_axis))),
    )
    centroid = np.asarray(
        (
            float(np.sum(concentration * x_grid) / total),
            float(np.sum(concentration * y_grid) / total),
        )
    )
    target_mask = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2 <= radius**2
    captured_mass = float(np.sum(concentration[target_mask]))
    target_sensitivity = 1.0
    if boundary_capture_correction:
        angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        boundary = center[None, :] + radius * (1.0 - 1e-9) * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
        inside = (
            (boundary[:, 0] >= x_axis[0])
            & (boundary[:, 0] <= x_axis[-1])
            & (boundary[:, 1] >= y_axis[0])
            & (boundary[:, 1] <= y_axis[-1])
        )
        if np.any(inside):
            boundary_image = particle_distribution_image(
                boundary[inside],
                x_axis,
                y_axis,
            )
            target_sensitivity = float(
                np.sum(boundary_image[target_mask]) / np.sum(boundary_image)
            )
            target_sensitivity = max(target_sensitivity, np.finfo(np.float64).eps)
    capture_fraction = float(
        np.clip(captured_mass / (total * target_sensitivity), 0.0, 1.0)
    )
    uncaptured = np.where(target_mask, 0.0, concentration)
    uncaptured_total = float(np.sum(uncaptured))
    if uncaptured_total <= np.finfo(np.float64).eps * total:
        uncaptured_centroid = center.copy()
    else:
        uncaptured_centroid = np.asarray(
            (
                float(np.sum(uncaptured * x_grid) / uncaptured_total),
                float(np.sum(uncaptured * y_grid) / uncaptured_total),
            )
        )

    count = max(1, int(np.rint(total / scale)))
    probabilities = concentration.ravel() / total
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    indices = np.searchsorted(cumulative, quantiles, side="left")
    positions = np.column_stack((x_grid.ravel()[indices], y_grid.ravel()[indices]))
    return ParticleStateEstimate2D(
        concentration_image=concentration,
        positions_m=positions,
        centroid_m=centroid,
        uncaptured_centroid_m=uncaptured_centroid,
        target_mask=target_mask,
        capture_fraction=capture_fraction,
        particle_count=count,
        total_signal=total,
        signal_per_particle=scale,
        support_threshold=threshold,
        spatial_resolution_m=resolution,
        target_partial_volume_sensitivity=target_sensitivity,
    )


def run_epm_particle_imaging(
    encoding: NonlinearEPMEncoding,
    positions_m: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    *,
    target_center_m: Sequence[float],
    target_radius_m: float,
    signal_per_particle: float = 1.0,
    support_threshold_fraction: float = 0.01,
    boundary_capture_correction: bool = False,
    regularization: float = 1e-4,
    snr_db: float | None = 35.0,
    seed: int = 0,
    ground_truth_captured: Sequence[bool] | np.ndarray | None = None,
) -> EPMParticleImagingResult:
    """Acquire a synthetic particle-sensitive image and estimate its state."""

    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    positions = _positions(positions_m, "positions_m")
    expected = particle_distribution_image(
        positions,
        x_axis,
        y_axis,
        signal_per_particle=signal_per_particle,
    )
    imaging = run_epm_nonlinear_imaging(
        encoding,
        expected,
        regularization=regularization,
        snr_db=snr_db,
        seed=seed,
    )
    estimate = estimate_particle_state_from_image(
        imaging.reconstructed_image,
        x_axis,
        y_axis,
        target_center_m=target_center_m,
        target_radius_m=target_radius_m,
        signal_per_particle=signal_per_particle,
        support_threshold_fraction=support_threshold_fraction,
        boundary_capture_correction=boundary_capture_correction,
    )
    center = np.asarray(target_center_m, dtype=np.float64)
    captured = None
    if ground_truth_captured is not None:
        captured = np.asarray(ground_truth_captured, dtype=bool)
        if captured.shape != (positions.shape[0],):
            raise ValueError("ground_truth_captured must match positions_m")
    return EPMParticleImagingResult(
        imaging=imaging,
        estimate=estimate,
        ground_truth_positions_m=positions.copy(),
        ground_truth_captured=None if captured is None else captured.copy(),
        target_center_m=center.copy(),
        target_radius_m=float(target_radius_m),
    )
