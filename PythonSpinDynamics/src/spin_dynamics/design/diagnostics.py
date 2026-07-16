"""Posterior summaries and stopping rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from spin_dynamics.design.posterior import ParticlePosterior
from spin_dynamics.design.types import ParameterParticles


QuantityOfInterest = Callable[[ParameterParticles], np.ndarray]


def quantity_values(
    posterior: ParticlePosterior, quantity: QuantityOfInterest
) -> np.ndarray:
    """Evaluate a scalar or vector QoI for every posterior particle."""

    values = np.asarray(quantity(posterior.parameters), dtype=np.float64)
    if values.ndim == 0 or values.shape[0] != posterior.particle_count:
        raise ValueError("quantity must return one value per posterior particle")
    if np.any(~np.isfinite(values)):
        raise ValueError("quantity values must be finite")
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values.reshape(values.shape[0], -1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.shape[0] != weights.size:
        raise ValueError("values and weights must share their leading dimension")
    return np.tensordot(weights, values, axes=(0, 0))


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mean = weighted_mean(values, weights)
    return weighted_mean((np.asarray(values) - mean) ** 2, weights)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probability: float
) -> float:
    """Return a left-continuous weighted scalar quantile."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie between zero and one")
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if values.size != weights.size:
        raise ValueError("values and weights must have equal size")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    cumulative[-1] = 1.0
    index = min(int(np.searchsorted(cumulative, probability, side="left")), values.size - 1)
    return float(values[order[index]])


@dataclass(frozen=True)
class PosteriorSummary:
    """Mean, standard deviation, and equal-tailed interval for a QoI."""

    mean: np.ndarray
    standard_deviation: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    probability: float
    effective_sample_size: float


def summarize_quantity(
    posterior: ParticlePosterior,
    quantity: QuantityOfInterest,
    *,
    probability: float = 0.95,
) -> PosteriorSummary:
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    values = quantity_values(posterior, quantity)
    weights = posterior.weights
    tail = 0.5 * (1.0 - probability)
    lower = np.array(
        [weighted_quantile(values[:, i], weights, tail) for i in range(values.shape[1])]
    )
    upper = np.array(
        [
            weighted_quantile(values[:, i], weights, 1.0 - tail)
            for i in range(values.shape[1])
        ]
    )
    return PosteriorSummary(
        mean=weighted_mean(values, weights),
        standard_deviation=np.sqrt(weighted_variance(values, weights)),
        lower=lower,
        upper=upper,
        probability=probability,
        effective_sample_size=posterior.effective_sample_size,
    )


@dataclass(frozen=True)
class CredibleIntervalStopping:
    """Stop when every QoI interval is no wider than ``maximum_width``."""

    quantity: QuantityOfInterest
    maximum_width: float | np.ndarray
    probability: float = 0.95

    def __post_init__(self) -> None:
        width = np.asarray(self.maximum_width, dtype=np.float64)
        if np.any(~np.isfinite(width)) or np.any(width < 0.0):
            raise ValueError("maximum_width must be finite and non-negative")
        if not 0.0 < self.probability < 1.0:
            raise ValueError("probability must lie strictly between zero and one")

    def reached(self, posterior: ParticlePosterior) -> bool:
        summary = summarize_quantity(
            posterior, self.quantity, probability=self.probability
        )
        return bool(np.all(summary.upper - summary.lower <= self.maximum_width))


@dataclass(frozen=True)
class PosteriorStandardDeviationStopping:
    """Stop when every QoI standard deviation is below a threshold."""

    quantity: QuantityOfInterest
    maximum_standard_deviation: float | np.ndarray

    def __post_init__(self) -> None:
        threshold = np.asarray(self.maximum_standard_deviation, dtype=np.float64)
        if np.any(~np.isfinite(threshold)) or np.any(threshold < 0.0):
            raise ValueError(
                "maximum_standard_deviation must be finite and non-negative"
            )

    def reached(self, posterior: ParticlePosterior) -> bool:
        values = quantity_values(posterior, self.quantity)
        standard_deviation = np.sqrt(weighted_variance(values, posterior.weights))
        return bool(np.all(standard_deviation <= self.maximum_standard_deviation))

