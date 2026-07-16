"""Observation likelihoods for Bayesian experiment design."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validated_sigma(value: float | np.ndarray) -> np.ndarray:
    sigma = np.asarray(value, dtype=np.float64)
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError("sigma must be finite and positive")
    return sigma


def _sum_event(values: np.ndarray, event_ndim: int) -> np.ndarray:
    if event_ndim == 0:
        return values
    if values.ndim < event_ndim:
        raise ValueError("prediction has fewer dimensions than event_ndim")
    axes = tuple(range(values.ndim - event_ndim, values.ndim))
    return np.sum(values, axis=axes)


@dataclass(frozen=True, eq=False)
class GaussianLikelihood:
    """Independent additive real Gaussian observation noise.

    ``event_ndim`` identifies trailing dimensions belonging to one observation.
    Use zero for scalar observations and one for a vector observation. Leading
    dimensions are preserved as batch or particle dimensions.
    """

    sigma: float | np.ndarray
    event_ndim: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _validated_sigma(self.sigma))
        if self.event_ndim < 0:
            raise ValueError("event_ndim must be non-negative")

    def sample(
        self, prediction: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.float64)
        return prediction + rng.normal(size=prediction.shape) * self.sigma

    def log_prob(
        self,
        observation: np.ndarray | float | complex,
        prediction: np.ndarray,
    ) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.float64)
        observation = np.asarray(observation, dtype=np.float64)
        residual = (observation - prediction) / self.sigma
        elementwise = (
            -0.5 * residual * residual
            - np.log(self.sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )
        return _sum_event(np.asarray(elementwise), self.event_ndim)


@dataclass(frozen=True, eq=False)
class ComplexGaussianLikelihood:
    """Circular complex Gaussian noise with ``sigma`` per real quadrature."""

    sigma: float | np.ndarray
    event_ndim: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma", _validated_sigma(self.sigma))
        if self.event_ndim < 0:
            raise ValueError("event_ndim must be non-negative")

    def sample(
        self, prediction: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.complex128)
        noise = rng.normal(size=prediction.shape) + 1j * rng.normal(
            size=prediction.shape
        )
        return prediction + self.sigma * noise

    def log_prob(
        self,
        observation: np.ndarray | float | complex,
        prediction: np.ndarray,
    ) -> np.ndarray:
        prediction = np.asarray(prediction, dtype=np.complex128)
        observation = np.asarray(observation, dtype=np.complex128)
        residual_power = np.abs(observation - prediction) ** 2
        elementwise = (
            -0.5 * residual_power / (self.sigma * self.sigma)
            - np.log(2.0 * np.pi * self.sigma * self.sigma)
        )
        return _sum_event(np.asarray(elementwise), self.event_ndim)

