"""Correlated target-spin, sensor-environment, and technical field noise."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


CorrelationEnvelope = Literal["exponential", "power_law"]


@dataclass(frozen=True)
class FieldNoiseComponent:
    """One stationary field-noise component.

    ``rms_field_tesla`` is the one-axis RMS field. ``correlation_time_seconds``
    sets the exponential decay time or the power-law crossover. A finite
    ``spatial_correlation_length_nm`` multiplies the temporal kernel by
    ``exp(-distance/length)``; ``None`` represents a field common to all
    supplied positions, as appropriate for intrinsic noise of one scanning
    sensor.
    """

    label: str
    rms_field_tesla: float
    correlation_time_seconds: float
    larmor_frequency_hz: float = 0.0
    envelope: CorrelationEnvelope = "exponential"
    power_law_exponent: float = 1.5
    spatial_correlation_length_nm: float | None = None

    def __post_init__(self) -> None:
        rms = float(self.rms_field_tesla)
        correlation = float(self.correlation_time_seconds)
        frequency = float(self.larmor_frequency_hz)
        exponent = float(self.power_law_exponent)
        if rms < 0.0 or not np.isfinite(rms):
            raise ValueError("rms_field_tesla must be finite and non-negative")
        if correlation <= 0.0 or not np.isfinite(correlation):
            raise ValueError(
                "correlation_time_seconds must be positive and finite"
            )
        if not np.isfinite(frequency):
            raise ValueError("larmor_frequency_hz must be finite")
        if self.envelope not in {"exponential", "power_law"}:
            raise ValueError("envelope must be 'exponential' or 'power_law'")
        if exponent <= 0.0 or not np.isfinite(exponent):
            raise ValueError("power_law_exponent must be positive and finite")
        length = self.spatial_correlation_length_nm
        if length is not None:
            length = float(length)
            if length <= 0.0 or not np.isfinite(length):
                raise ValueError(
                    "spatial_correlation_length_nm must be positive and finite"
                )
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "rms_field_tesla", rms)
        object.__setattr__(self, "correlation_time_seconds", correlation)
        object.__setattr__(self, "larmor_frequency_hz", frequency)
        object.__setattr__(self, "power_law_exponent", exponent)
        object.__setattr__(self, "spatial_correlation_length_nm", length)

    def temporal_correlation(self, lag_seconds) -> np.ndarray:
        """Return the normalized stationary temporal correlation."""

        lag = np.abs(np.asarray(lag_seconds, dtype=np.float64))
        if not np.all(np.isfinite(lag)):
            raise ValueError("lag_seconds must be finite")
        scaled = lag / self.correlation_time_seconds
        if self.envelope == "exponential":
            decay = np.exp(-scaled)
        else:
            decay = (1.0 + scaled) ** (-self.power_law_exponent)
        return decay * np.cos(2.0 * np.pi * self.larmor_frequency_hz * lag)


@dataclass(frozen=True)
class CorrelatedFieldNoiseModel:
    """Sum of independent stationary magnetic-field noise components."""

    components: tuple[FieldNoiseComponent, ...]
    label: str = "correlated field noise"

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components or not all(
            isinstance(item, FieldNoiseComponent) for item in components
        ):
            raise ValueError("components must contain FieldNoiseComponent values")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "label", str(self.label))

    def covariance(self, times_seconds, *, positions_lab_nm=None) -> np.ndarray:
        """Return temporal/spatial covariance in tesla squared."""

        times = _times(times_seconds)
        positions = _positions(positions_lab_nm, times.size)
        lag = np.abs(times[:, np.newaxis] - times[np.newaxis, :])
        distance = None
        if positions is not None:
            displacement = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distance = np.linalg.norm(displacement, axis=2)
        covariance = np.zeros((times.size, times.size), dtype=np.float64)
        for component in self.components:
            kernel = component.temporal_correlation(lag)
            if (
                distance is not None
                and component.spatial_correlation_length_nm is not None
            ):
                kernel = kernel * np.exp(
                    -distance / component.spatial_correlation_length_nm
                )
            covariance += component.rms_field_tesla**2 * kernel
        return 0.5 * (covariance + covariance.T)


@dataclass(frozen=True, eq=False)
class FieldNoiseRealization:
    """Seeded correlated field samples and their prescribed covariance."""

    times_seconds: np.ndarray
    positions_lab_nm: np.ndarray | None
    field_tesla: np.ndarray
    covariance_t2: np.ndarray
    component_labels: tuple[str, ...]
    seed: int | None


def sample_correlated_field_noise(
    model: CorrelatedFieldNoiseModel,
    times_seconds,
    *,
    positions_lab_nm=None,
    mean_field_tesla=0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> FieldNoiseRealization:
    """Draw a reproducible multivariate-normal field record."""

    if not isinstance(model, CorrelatedFieldNoiseModel):
        raise TypeError("model must be a CorrelatedFieldNoiseModel")
    if seed is not None and rng is not None:
        raise ValueError("provide seed or rng, not both")
    times = _times(times_seconds)
    positions = _positions(positions_lab_nm, times.size)
    mean = np.asarray(mean_field_tesla, dtype=np.float64)
    if mean.ndim == 0:
        mean = np.full(times.size, float(mean))
    if mean.shape != (times.size,) or not np.all(np.isfinite(mean)):
        raise ValueError("mean_field_tesla must be finite and match times")
    covariance = model.covariance(times, positions_lab_nm=positions)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(np.finfo(float).tiny),
    )
    if float(np.min(eigenvalues, initial=0.0)) < -1.0e-12 * scale:
        raise ValueError("noise covariance is not positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    generator = np.random.default_rng(seed) if rng is None else rng
    field = mean + eigenvectors @ (
        np.sqrt(eigenvalues) * generator.standard_normal(times.size)
    )
    return FieldNoiseRealization(
        times_seconds=times.copy(),
        positions_lab_nm=None if positions is None else positions.copy(),
        field_tesla=field,
        covariance_t2=covariance,
        component_labels=tuple(item.label for item in model.components),
        seed=seed,
    )


def linear_field_covariance(forward_matrix, source_variances) -> np.ndarray:
    """Map independent source-amplitude variances through a linear operator."""

    matrix = np.asarray(forward_matrix, dtype=np.float64)
    variances = np.asarray(source_variances, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("forward_matrix must be a finite matrix")
    if variances.shape != (matrix.shape[1],) or not np.all(np.isfinite(variances)):
        raise ValueError("source_variances must match forward-matrix columns")
    if np.any(variances < 0.0):
        raise ValueError("source_variances must be non-negative")
    covariance = (matrix * variances[np.newaxis, :]) @ matrix.T
    return 0.5 * (covariance + covariance.T)


def effective_sample_size(covariance) -> float:
    """Return ``trace(C)^2 / trace(C^2)`` for a covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.size == 0
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("covariance must be a finite non-empty square matrix")
    matrix = 0.5 * (matrix + matrix.T)
    trace = float(np.trace(matrix))
    squared_trace = float(np.sum(matrix * matrix))
    if trace <= 0.0 or squared_trace <= 0.0:
        raise ValueError("covariance must have positive total variance")
    return trace**2 / squared_trace


def _times(values) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64)
    if (
        times.ndim != 1
        or times.size == 0
        or not np.all(np.isfinite(times))
    ):
        raise ValueError("times_seconds must be a finite non-empty vector")
    return times


def _positions(values, count: int) -> np.ndarray | None:
    if values is None:
        return None
    positions = np.asarray(values, dtype=np.float64)
    if positions.shape != (count, 3) or not np.all(np.isfinite(positions)):
        raise ValueError("positions_lab_nm must be finite with shape (N, 3)")
    return positions


__all__ = [
    "CorrelationEnvelope",
    "CorrelatedFieldNoiseModel",
    "FieldNoiseComponent",
    "FieldNoiseRealization",
    "effective_sample_size",
    "linear_field_covariance",
    "sample_correlated_field_noise",
]
