"""Monte Carlo utilities for Bayesian experiment design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from spin_dynamics.design.diagnostics import (
    QuantityOfInterest,
    quantity_values,
    weighted_variance,
)
from spin_dynamics.design.models import PredictiveModel
from spin_dynamics.design.posterior import (
    ParticlePosterior,
    logsumexp,
    normalize_log_weights,
)


@dataclass(frozen=True, eq=False)
class UtilityEstimate:
    """Estimated expected utility and Monte Carlo standard error."""

    value: float
    standard_error: float
    samples: np.ndarray
    units: str


class DesignUtility(Protocol):
    """Utility estimator consumed by an adaptive design session."""

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate: ...


def _estimate(samples: np.ndarray, units: str) -> UtilityEstimate:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("utility samples must be non-empty and finite")
    standard_error = 0.0
    if values.size > 1:
        standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    return UtilityEstimate(
        value=float(np.mean(values)),
        standard_error=standard_error,
        samples=values.copy(),
        units=units,
    )


def _outer_indices(
    posterior: ParticlePosterior,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if isinstance(samples, bool) or int(samples) != samples or samples <= 0:
        raise ValueError("samples must be a positive integer")
    return rng.choice(
        posterior.particle_count,
        size=int(samples),
        p=posterior.weights,
    )


@dataclass(frozen=True)
class ExpectedInformationGain:
    """Expected information about the complete latent particle state.

    The nested Monte Carlo estimate is in nats. For a selected or derived
    quantity of interest use :class:`ExpectedVarianceReduction`; target-only
    EIG with nuisance marginalization is retained by the exact Phase 0
    reference and is planned for the generic core after density-estimator
    choices are validated.
    """

    samples: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or int(self.samples) != self.samples:
            raise ValueError("samples must be a positive integer")
        if self.samples <= 0:
            raise ValueError("samples must be a positive integer")

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate:
        predictions = model.predict(posterior.parameters, design)
        indices = _outer_indices(posterior, self.samples, rng)
        information = np.empty(indices.size, dtype=np.float64)
        log_weights = posterior.log_weights
        for sample_index, particle_index in enumerate(indices):
            observation = model.likelihood.sample(predictions[particle_index], rng)
            log_likelihood = np.asarray(
                model.likelihood.log_prob(observation, predictions),
                dtype=np.float64,
            ).reshape(-1)
            if log_likelihood.size != posterior.particle_count:
                raise ValueError("likelihood did not return one value per particle")
            log_predictive = float(logsumexp(log_weights + log_likelihood))
            information[sample_index] = (
                log_likelihood[particle_index] - log_predictive
            )
        return _estimate(information, "nats")


@dataclass(frozen=True, eq=False)
class ExpectedVarianceReduction:
    """Expected reduction in weighted squared-error Bayes risk for a QoI.

    ``scale`` nondimensionalizes vector components before their posterior
    variances are summed. A scalar scale applies to every component.
    """

    quantity: QuantityOfInterest
    samples: int = 128
    scale: float | np.ndarray = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or int(self.samples) != self.samples:
            raise ValueError("samples must be a positive integer")
        if self.samples <= 0:
            raise ValueError("samples must be a positive integer")
        scale = np.asarray(self.scale, dtype=np.float64)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "scale", scale)

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate:
        predictions = model.predict(posterior.parameters, design)
        quantity = quantity_values(posterior, self.quantity) / self.scale
        current_risk = float(np.sum(weighted_variance(quantity, posterior.weights)))
        indices = _outer_indices(posterior, self.samples, rng)
        reduction = np.empty(indices.size, dtype=np.float64)
        log_weights = posterior.log_weights
        for sample_index, particle_index in enumerate(indices):
            observation = model.likelihood.sample(predictions[particle_index], rng)
            log_likelihood = np.asarray(
                model.likelihood.log_prob(observation, predictions),
                dtype=np.float64,
            ).reshape(-1)
            updated_log_weights = normalize_log_weights(log_weights + log_likelihood)
            updated_weights = np.exp(updated_log_weights)
            posterior_risk = float(
                np.sum(weighted_variance(quantity, updated_weights))
            )
            reduction[sample_index] = current_risk - posterior_risk
        return _estimate(reduction, "scaled variance")

