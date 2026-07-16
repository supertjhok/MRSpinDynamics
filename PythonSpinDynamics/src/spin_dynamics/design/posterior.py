"""Weighted grid and particle posteriors for Bayesian design."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from spin_dynamics.design.types import ParameterParticles, Prior


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Stable NumPy log-sum-exp used by posterior and utility calculations."""

    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(finite, maximum + np.log(total), -np.inf)
    if axis is None:
        return np.asarray(result.reshape(()))
    return np.squeeze(result, axis=axis)


def normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return normalized one-dimensional log weights."""

    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    if np.any(np.isnan(values)):
        raise ValueError("log_weights must not contain NaN")
    normalizer = float(logsumexp(values))
    if not np.isfinite(normalizer):
        raise ValueError("log_weights must have positive finite total mass")
    if abs(normalizer) <= 1e-14:
        return values.copy()
    return values - normalizer


def _validated_particles(parameters: ParameterParticles) -> dict[str, np.ndarray]:
    values = {name: np.asarray(value) for name, value in parameters.items()}
    if not values:
        raise ValueError("parameters must not be empty")
    count: int | None = None
    for name, value in values.items():
        if not name:
            raise ValueError("parameter names must not be empty")
        if value.ndim == 0:
            raise ValueError(f"parameter {name!r} must have a particle dimension")
        if np.any(~np.isfinite(value)):
            raise ValueError(f"parameter {name!r} must contain finite values")
        if count is None:
            count = value.shape[0]
        elif value.shape[0] != count:
            raise ValueError("all parameter arrays must share their leading dimension")
    assert count is not None
    if count <= 0:
        raise ValueError("at least one particle is required")
    return {name: value.copy() for name, value in values.items()}


def take_particles(
    parameters: ParameterParticles, indices: np.ndarray | list[int]
) -> dict[str, np.ndarray]:
    """Select a common leading-dimension subset from named particles."""

    return {name: np.asarray(value)[indices] for name, value in parameters.items()}


class ParticlePosterior:
    """Weighted empirical posterior over named parameter particles."""

    kind = "particles"

    def __init__(
        self,
        parameters: ParameterParticles,
        log_weights: np.ndarray | None = None,
    ) -> None:
        self._parameters = _validated_particles(parameters)
        count = next(iter(self._parameters.values())).shape[0]
        if log_weights is None:
            self._log_weights = np.full(count, -np.log(float(count)))
        else:
            values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
            if values.size != count:
                raise ValueError("log_weights must match the particle count")
            self._log_weights = normalize_log_weights(values)

    @classmethod
    def from_prior(
        cls,
        prior: Prior,
        particles: int,
        rng: np.random.Generator,
    ) -> "ParticlePosterior":
        """Draw equally weighted particles from ``prior``."""

        return cls(prior.sample(particles, rng))

    @property
    def parameters(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._parameters.items()}

    @property
    def particle_count(self) -> int:
        return self._log_weights.size

    @property
    def log_weights(self) -> np.ndarray:
        return self._log_weights.copy()

    @property
    def weights(self) -> np.ndarray:
        return np.exp(self._log_weights)

    @property
    def effective_sample_size(self) -> float:
        weights = self.weights
        return float(1.0 / np.sum(weights * weights))

    def updated(self, log_likelihood: np.ndarray) -> "ParticlePosterior":
        """Return the importance-weight update for one observation."""

        likelihood = np.asarray(log_likelihood, dtype=np.float64).reshape(-1)
        if likelihood.size != self.particle_count:
            raise ValueError("log_likelihood must match the particle count")
        if np.any(np.isnan(likelihood)):
            raise ValueError("log_likelihood must not contain NaN")
        return ParticlePosterior(self._parameters, self._log_weights + likelihood)

    def resampled(self, rng: np.random.Generator) -> "ParticlePosterior":
        """Return an equally weighted systematic resample."""

        count = self.particle_count
        positions = (rng.random() + np.arange(count)) / count
        cumulative = np.cumsum(self.weights)
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, positions, side="right")
        return ParticlePosterior(take_particles(self._parameters, indices))

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": self.parameters,
            "log_weights": self.log_weights,
        }


class GridPosterior(ParticlePosterior):
    """Exact weighted posterior on a Cartesian parameter grid."""

    kind = "grid"

    def __init__(
        self,
        axes: Mapping[str, np.ndarray],
        weights: np.ndarray | None = None,
        *,
        log_weights: np.ndarray | None = None,
    ) -> None:
        normalized_axes: dict[str, np.ndarray] = {}
        for name, values in axes.items():
            axis = np.asarray(values, dtype=np.float64)
            if axis.ndim != 1 or axis.size == 0 or np.any(~np.isfinite(axis)):
                raise ValueError(f"axis {name!r} must be non-empty, finite, and 1-D")
            if np.unique(axis).size != axis.size:
                raise ValueError(f"axis {name!r} values must be unique")
            normalized_axes[name] = axis.copy()
        if not normalized_axes:
            raise ValueError("axes must not be empty")
        self.axes = normalized_axes
        self.shape = tuple(axis.size for axis in self.axes.values())
        meshes = np.meshgrid(*self.axes.values(), indexing="ij")
        parameters = {
            name: mesh.reshape(-1)
            for name, mesh in zip(self.axes, meshes)
        }
        if weights is not None and log_weights is not None:
            raise ValueError("provide weights or log_weights, not both")
        if weights is not None:
            probability = np.asarray(weights, dtype=np.float64)
            if probability.shape != self.shape:
                raise ValueError(f"weights must have shape {self.shape}")
            if np.any(~np.isfinite(probability)) or np.any(probability < 0.0):
                raise ValueError("weights must be finite and non-negative")
            flat = probability.reshape(-1)
            if np.sum(flat) <= 0.0:
                raise ValueError("weights must have positive total mass")
            initial = np.full(flat.size, -np.inf)
            positive = flat > 0.0
            initial[positive] = np.log(flat[positive])
            log_weights = initial
        super().__init__(parameters, log_weights)

    def updated(self, log_likelihood: np.ndarray) -> "GridPosterior":
        likelihood = np.asarray(log_likelihood, dtype=np.float64).reshape(-1)
        if likelihood.size != self.particle_count:
            raise ValueError("log_likelihood must match the grid size")
        return GridPosterior(
            self.axes,
            log_weights=self._log_weights + likelihood,
        )

    def marginal(self, parameter: str) -> np.ndarray:
        """Return probabilities along one named grid axis."""

        if parameter not in self.axes:
            raise KeyError(parameter)
        index = list(self.axes).index(parameter)
        probability = self.weights.reshape(self.shape)
        sum_axes = tuple(axis for axis in range(len(self.shape)) if axis != index)
        if not sum_axes:
            return probability.copy()
        return np.sum(probability, axis=sum_axes)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "axes": {name: values.copy() for name, values in self.axes.items()},
            "log_weights": self.log_weights,
        }


def posterior_from_state(data: Mapping[str, Any]) -> ParticlePosterior:
    """Reconstruct a grid or particle posterior from ``state_dict`` data."""

    kind = data.get("kind")
    if kind == "grid":
        return GridPosterior(data["axes"], log_weights=data["log_weights"])
    if kind == "particles":
        return ParticlePosterior(data["parameters"], data["log_weights"])
    raise ValueError(f"unknown posterior kind {kind!r}")

