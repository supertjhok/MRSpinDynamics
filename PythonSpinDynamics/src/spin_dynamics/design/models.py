"""Likelihood-backed vectorized predictive models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from spin_dynamics.design.types import ObservationLikelihood, ParameterParticles


Predictor = Callable[[ParameterParticles, Any], np.ndarray]


def _particle_count(parameters: ParameterParticles) -> int:
    if not parameters:
        raise ValueError("parameters must not be empty")
    counts = {np.asarray(value).shape[0] for value in parameters.values()}
    if len(counts) != 1:
        raise ValueError("parameter arrays must share their leading dimension")
    return counts.pop()


@dataclass(frozen=True)
class PredictiveModel:
    """A vectorized deterministic predictor plus observation likelihood.

    ``predictor(parameters, design)`` must return an array whose leading
    dimension matches the parameter-particle count. Remaining dimensions are
    one observation event and must agree with ``likelihood.event_ndim`` when
    the likelihood exposes that attribute.
    """

    predictor: Predictor
    likelihood: ObservationLikelihood

    def predict(self, parameters: ParameterParticles, design: Any) -> np.ndarray:
        prediction = np.asarray(self.predictor(parameters, design))
        count = _particle_count(parameters)
        if prediction.ndim == 0 or prediction.shape[0] != count:
            raise ValueError(
                "predictor output must have the particle count as its leading dimension"
            )
        if np.any(~np.isfinite(prediction)):
            raise ValueError("predictor output must be finite")
        return prediction

    def log_likelihood(
        self,
        observation: np.ndarray | float | complex,
        parameters: ParameterParticles,
        design: Any,
    ) -> np.ndarray:
        prediction = self.predict(parameters, design)
        result = np.asarray(self.likelihood.log_prob(observation, prediction))
        if result.shape != (prediction.shape[0],):
            raise ValueError(
                "likelihood must return one log probability per parameter particle"
            )
        return result.astype(np.float64, copy=False)

