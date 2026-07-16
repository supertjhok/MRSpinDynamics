"""Shared protocols and type aliases for Bayesian experiment design."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


ParameterParticles = Mapping[str, np.ndarray]
"""Named parameter arrays whose leading dimension indexes particles."""


@runtime_checkable
class Prior(Protocol):
    """Distribution capable of drawing named parameter particles."""

    def sample(
        self, size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]: ...

    def log_prob(self, parameters: ParameterParticles) -> np.ndarray: ...


@runtime_checkable
class ObservationLikelihood(Protocol):
    """Conditional observation distribution around model predictions."""

    def sample(
        self, prediction: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray: ...

    def log_prob(
        self, observation: np.ndarray | float | complex, prediction: np.ndarray
    ) -> np.ndarray: ...


@runtime_checkable
class PhysicalCost(Protocol):
    """Physical acquisition cost for one design action."""

    def seconds(self, design: Any) -> float: ...


@runtime_checkable
class StopRule(Protocol):
    """Posterior-dependent termination criterion."""

    def reached(self, posterior: Any) -> bool: ...

