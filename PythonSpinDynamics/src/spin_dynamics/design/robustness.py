"""Model discrepancy, mixtures, nuisance marginalization, and sensitivity.

These utilities make robustness assumptions explicit rather than hiding them
inside a simulator.  Discrete target information gain marginalizes nuisance
particles, model mixtures select a predictor through a posterior model label,
and sensitivity reports repeat the same decision under alternative priors or
models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from spin_dynamics.design.constraints import DesignConstraint
from spin_dynamics.design.costs import ConstantCost
from spin_dynamics.design.likelihoods import (
    ComplexGaussianLikelihood,
    GaussianLikelihood,
)
from spin_dynamics.design.models import PredictiveModel, Predictor
from spin_dynamics.design.posterior import ParticlePosterior, logsumexp
from spin_dynamics.design.session import AdaptiveDesignSession
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import PhysicalCost
from spin_dynamics.design.utilities import DesignUtility, UtilityEstimate


@dataclass(frozen=True, eq=False)
class GaussianDiscrepancyLikelihood:
    """Real Gaussian measurement noise plus independent model discrepancy."""

    measurement_sigma: float | np.ndarray
    discrepancy_sigma: float | np.ndarray
    event_ndim: int = 0

    def __post_init__(self) -> None:
        total = np.sqrt(
            np.asarray(self.measurement_sigma, dtype=np.float64) ** 2
            + np.asarray(self.discrepancy_sigma, dtype=np.float64) ** 2
        )
        object.__setattr__(self, "_likelihood", GaussianLikelihood(total, self.event_ndim))

    @property
    def sigma(self) -> np.ndarray:
        """Total standard deviation used by the likelihood."""

        return self._likelihood.sigma

    def sample(self, prediction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._likelihood.sample(prediction, rng)

    def log_prob(self, observation: Any, prediction: np.ndarray) -> np.ndarray:
        return self._likelihood.log_prob(observation, prediction)


@dataclass(frozen=True, eq=False)
class ComplexGaussianDiscrepancyLikelihood:
    """Circular-complex measurement noise plus independent discrepancy."""

    measurement_sigma: float | np.ndarray
    discrepancy_sigma: float | np.ndarray
    event_ndim: int = 0

    def __post_init__(self) -> None:
        total = np.sqrt(
            np.asarray(self.measurement_sigma, dtype=np.float64) ** 2
            + np.asarray(self.discrepancy_sigma, dtype=np.float64) ** 2
        )
        object.__setattr__(
            self, "_likelihood", ComplexGaussianLikelihood(total, self.event_ndim)
        )

    @property
    def sigma(self) -> np.ndarray:
        """Total per-quadrature standard deviation."""

        return self._likelihood.sigma

    def sample(self, prediction: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self._likelihood.sample(prediction, rng)

    def log_prob(self, observation: Any, prediction: np.ndarray) -> np.ndarray:
        return self._likelihood.log_prob(observation, prediction)


@dataclass(frozen=True)
class ModelMixturePredictor:
    """Dispatch particles to competing predictors using an integer model label."""

    predictors: tuple[Predictor, ...]
    model_parameter: str = "model_index"

    def __init__(
        self, predictors: Sequence[Predictor], model_parameter: str = "model_index"
    ) -> None:
        values = tuple(predictors)
        if not values:
            raise ValueError("predictors must not be empty")
        object.__setattr__(self, "predictors", values)
        object.__setattr__(self, "model_parameter", str(model_parameter))

    def __call__(
        self, parameters: Mapping[str, np.ndarray], design: Any
    ) -> np.ndarray:
        if self.model_parameter not in parameters:
            raise KeyError(f"missing model label parameter {self.model_parameter!r}")
        labels = np.asarray(parameters[self.model_parameter])
        if labels.ndim != 1 or np.any(labels != labels.astype(int)):
            raise ValueError("model labels must be an integer particle vector")
        labels = labels.astype(int)
        if np.any(labels < 0) or np.any(labels >= len(self.predictors)):
            raise ValueError("model label is outside the predictor range")
        physical = {
            name: np.asarray(values)
            for name, values in parameters.items()
            if name != self.model_parameter
        }
        result: np.ndarray | None = None
        for label, predictor in enumerate(self.predictors):
            mask = labels == label
            if not np.any(mask):
                continue
            subset = {name: values[mask] for name, values in physical.items()}
            prediction = np.asarray(predictor(subset, design))
            if prediction.shape[0] != int(np.sum(mask)):
                raise ValueError("mixture component returned the wrong particle count")
            if result is None:
                result = np.empty((labels.size, *prediction.shape[1:]), prediction.dtype)
            elif result.shape[1:] != prediction.shape[1:]:
                raise ValueError("mixture predictors must return the same event shape")
            elif np.iscomplexobj(prediction) and not np.iscomplexobj(result):
                result = result.astype(np.complex128)
            result[mask] = prediction
        assert result is not None
        return result


def _target_groups(values: np.ndarray) -> tuple[np.ndarray, int]:
    target = np.asarray(values)
    if target.ndim == 1:
        _, inverse = np.unique(target, return_inverse=True)
    else:
        flattened = target.reshape(target.shape[0], -1)
        _, inverse = np.unique(flattened, axis=0, return_inverse=True)
    return inverse.astype(int), int(np.max(inverse) + 1)


@dataclass(frozen=True)
class ExpectedTargetInformationGain:
    """Information about a discrete target after marginalizing nuisance state.

    ``target(parameters)`` must return one discrete label (or label row) per
    particle. Particles sharing a label represent nuisance uncertainty that is
    integrated out in ``p(y | target, design)``.
    """

    target: Callable[[Mapping[str, np.ndarray]], np.ndarray]
    samples: int = 128

    def __post_init__(self) -> None:
        if self.samples <= 0:
            raise ValueError("samples must be positive")

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate:
        predictions = model.predict(posterior.parameters, design)
        labels, groups = _target_groups(self.target(posterior.parameters))
        if labels.shape != (posterior.particle_count,):
            raise ValueError("target must return one label per particle")
        weights = posterior.weights
        group_probability = np.bincount(labels, weights=weights, minlength=groups)
        indices = rng.choice(
            posterior.particle_count, size=self.samples, p=weights
        )
        information = np.empty(self.samples, dtype=np.float64)
        for sample, particle_index in enumerate(indices):
            observation = model.likelihood.sample(predictions[particle_index], rng)
            log_likelihood = np.asarray(
                model.likelihood.log_prob(observation, predictions), dtype=np.float64
            ).reshape(-1)
            log_predictive = float(logsumexp(posterior.log_weights + log_likelihood))
            group = labels[particle_index]
            mask = labels == group
            log_target_predictive = float(
                logsumexp(posterior.log_weights[mask] + log_likelihood[mask])
                - np.log(group_probability[group])
            )
            information[sample] = log_target_predictive - log_predictive
        standard_error = (
            float(np.std(information, ddof=1) / np.sqrt(information.size))
            if information.size > 1
            else 0.0
        )
        return UtilityEstimate(
            float(np.mean(information)), standard_error, information, "target nats"
        )


@dataclass(frozen=True)
class SensitivityRecommendation:
    """One scenario's selected action and utility rate."""

    scenario: str
    design: Any
    design_index: int
    utility_rate: float


@dataclass(frozen=True)
class DesignSensitivityReport:
    """Recommendation stability across alternative priors or models."""

    recommendations: tuple[SensitivityRecommendation, ...]
    modal_design_index: int
    agreement_fraction: float


def analyze_design_sensitivity(
    *,
    scenarios: Mapping[str, tuple[PredictiveModel, ParticlePosterior]],
    design_space: CandidateDesignSpace,
    utility: DesignUtility,
    cost: PhysicalCost | None = None,
    constraints: Sequence[DesignConstraint] = (),
    seed: int = 0,
) -> DesignSensitivityReport:
    """Repeat one recommendation under named prior/model scenarios."""

    if not scenarios:
        raise ValueError("scenarios must not be empty")
    physical_cost = cost if cost is not None else ConstantCost(1.0)
    recommendations: list[SensitivityRecommendation] = []
    for name, (model, posterior) in scenarios.items():
        session = AdaptiveDesignSession(
            model=model,
            posterior=posterior,
            design_space=design_space,
            utility=utility,
            cost=physical_cost,
            constraints=constraints,
            seed=seed,
        )
        best = session.ask().best
        recommendations.append(
            SensitivityRecommendation(
                str(name),
                best.design,
                best.design_index,
                float(best.utility_rate),
            )
        )
    indices = np.asarray([item.design_index for item in recommendations])
    counts = np.bincount(indices, minlength=len(design_space.actions))
    modal = int(np.argmax(counts))
    return DesignSensitivityReport(
        tuple(recommendations), modal, float(counts[modal] / indices.size)
    )
