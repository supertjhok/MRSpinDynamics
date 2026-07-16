"""NumPy-only prior distributions for experiment design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from spin_dynamics.design.types import ParameterParticles


class ScalarPrior(Protocol):
    """One-dimensional prior used by :class:`IndependentPrior`."""

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray: ...

    def log_prob(self, values: np.ndarray) -> np.ndarray: ...


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class NormalPrior:
    """Univariate normal prior."""

    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("mean must be finite")
        _positive_finite(self.standard_deviation, "standard_deviation")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean, self.standard_deviation, size=size)

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        residual = (values - self.mean) / self.standard_deviation
        return (
            -0.5 * residual * residual
            - np.log(self.standard_deviation)
            - 0.5 * np.log(2.0 * np.pi)
        )


@dataclass(frozen=True)
class UniformPrior:
    """Continuous uniform prior on a closed finite interval."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.low) or not np.isfinite(self.high):
            raise ValueError("uniform bounds must be finite")
        if self.high <= self.low:
            raise ValueError("high must exceed low")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        inside = (values >= self.low) & (values <= self.high)
        return np.where(inside, -np.log(self.high - self.low), -np.inf)


@dataclass(frozen=True)
class LogUniformPrior:
    """Continuous log-uniform prior on positive finite bounds."""

    low: float
    high: float

    def __post_init__(self) -> None:
        _positive_finite(self.low, "low")
        _positive_finite(self.high, "high")
        if self.high <= self.low:
            raise ValueError("high must exceed low")

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size=size))

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        inside = (values >= self.low) & (values <= self.high)
        normalization = np.log(np.log(self.high / self.low))
        safe = np.where(inside, values, 1.0)
        return np.where(inside, -np.log(safe) - normalization, -np.inf)


@dataclass(frozen=True, eq=False)
class DiscretePrior:
    """Finite scalar prior with user-supplied probabilities."""

    values: np.ndarray
    probabilities: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
            raise ValueError("values must be a non-empty finite one-dimensional array")
        if np.unique(values).size != values.size:
            raise ValueError("values must be unique")
        if self.probabilities is None:
            probabilities = np.full(values.size, 1.0 / values.size)
        else:
            probabilities = np.asarray(self.probabilities, dtype=np.float64)
            if probabilities.shape != values.shape:
                raise ValueError("probabilities must match values")
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0.0):
                raise ValueError("probabilities must be finite and non-negative")
            total = float(np.sum(probabilities))
            if total <= 0.0:
                raise ValueError("probabilities must have positive total mass")
            probabilities = probabilities / total
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probabilities)

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        indices = rng.choice(self.values.size, size=size, p=self.probabilities)
        return self.values[indices]

    def log_prob(self, values: np.ndarray) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        flat = query.reshape(-1)
        result = np.full(flat.size, -np.inf)
        for value, probability in zip(self.values, self.probabilities):
            if probability > 0.0:
                result[flat == value] = np.log(probability)
        return result.reshape(query.shape)


@dataclass(frozen=True)
class IndependentPrior:
    """Product prior over named scalar parameters."""

    components: Mapping[str, ScalarPrior]

    def __post_init__(self) -> None:
        components = dict(self.components)
        if not components:
            raise ValueError("components must not be empty")
        if any(not name for name in components):
            raise ValueError("parameter names must not be empty")
        object.__setattr__(self, "components", components)

    def sample(
        self, size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        if isinstance(size, bool) or int(size) != size or size <= 0:
            raise ValueError("size must be a positive integer")
        return {
            name: component.sample(int(size), rng)
            for name, component in self.components.items()
        }

    def log_prob(self, parameters: ParameterParticles) -> np.ndarray:
        missing = set(self.components) - set(parameters)
        if missing:
            raise KeyError(f"missing parameters: {sorted(missing)}")
        result: np.ndarray | None = None
        for name, component in self.components.items():
            term = np.asarray(component.log_prob(parameters[name]), dtype=np.float64)
            result = term if result is None else result + term
        assert result is not None
        return result

