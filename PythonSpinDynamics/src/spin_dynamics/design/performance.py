"""Performance-oriented scoring, surrogates, and candidate screening.

The tools in this module accelerate the expensive *policy calculation* while
keeping the physical acquisition cost separate.  Surrogates are explicit and
must be validated against held-out exact predictions; two-stage sessions use a
cheap Laplace or low-sample utility to screen every action and a higher-fidelity
utility to rescore only the leaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from spin_dynamics.design.constraints import evaluate_constraints
from spin_dynamics.design.models import PredictiveModel, Predictor
from spin_dynamics.design.posterior import ParticlePosterior
from spin_dynamics.design.session import (
    AdaptiveDesignSession,
    CandidateScore,
    DesignRecommendation,
)
from spin_dynamics.design.utilities import (
    DesignUtility,
    ExpectedInformationGain,
    UtilityEstimate,
)


ParameterEncoder = Callable[[Mapping[str, np.ndarray]], np.ndarray]
DesignEncoder = Callable[[Any], np.ndarray]
JointEncoder = Callable[[Mapping[str, np.ndarray], Any], np.ndarray]


def scalar_parameter_encoder(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stack named scalar particle arrays in sorted-name order."""

    if not parameters:
        raise ValueError("parameters must not be empty")
    arrays = []
    count: int | None = None
    for name in sorted(parameters):
        values = np.asarray(parameters[name], dtype=np.float64)
        if values.ndim != 1:
            raise ValueError(f"parameter {name!r} must be a scalar-particle vector")
        if count is None:
            count = values.size
        elif values.size != count:
            raise ValueError("parameter arrays must have equal lengths")
        arrays.append(values)
    return np.column_stack(arrays)


def _encoded_design(encoder: DesignEncoder, design: Any) -> np.ndarray:
    values = np.asarray(encoder(design), dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("design encoder must return finite features")
    return values


def _polynomial_features(values: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("feature values must be a matrix")
    columns = [np.ones(values.shape[0], dtype=np.float64)]
    indices = range(values.shape[1])
    for order in range(1, degree + 1):
        for combination in combinations_with_replacement(indices, order):
            columns.append(np.prod(values[:, combination], axis=1))
    return np.column_stack(columns)


@dataclass(frozen=True)
class SurrogateValidation:
    """Held-out error metrics for a deterministic surrogate."""

    root_mean_square_error: float
    normalized_root_mean_square_error: float
    maximum_absolute_error: float
    correlation: float
    predictions: int


@dataclass(frozen=True, eq=False)
class PolynomialSurrogatePredictor:
    """Ridge-stabilized polynomial response surface for scalar particles.

    Use transformed features (often logarithms of positive relaxation times,
    diffusion coefficients, and delays) when the physics spans decades.  The
    predictor supports scalar, vector, real, or complex workflow observables.
    """

    parameter_encoder: ParameterEncoder
    design_encoder: DesignEncoder
    joint_encoder: JointEncoder | None
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    output_shape: tuple[int, ...]
    degree: int

    @classmethod
    def fit(
        cls,
        exact_predictor: Predictor,
        parameters: Mapping[str, np.ndarray],
        designs: Sequence[Any],
        design_encoder: DesignEncoder,
        *,
        parameter_encoder: ParameterEncoder = scalar_parameter_encoder,
        joint_encoder: JointEncoder | None = None,
        degree: int = 3,
        ridge: float = 1e-10,
    ) -> "PolynomialSurrogatePredictor":
        """Fit from the Cartesian product of training particles and actions."""

        if degree < 1:
            raise ValueError("degree must be positive")
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge must be finite and non-negative")
        design_values = tuple(designs)
        if not design_values:
            raise ValueError("designs must not be empty")
        parameter_features = np.asarray(parameter_encoder(parameters), dtype=np.float64)
        if parameter_features.ndim != 2 or parameter_features.shape[0] == 0:
            raise ValueError("parameter encoder must return a non-empty matrix")
        raw_rows: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        output_shape: tuple[int, ...] | None = None
        for design in design_values:
            if joint_encoder is None:
                design_features = _encoded_design(design_encoder, design)
                repeated = np.repeat(
                    design_features[None, :], parameter_features.shape[0], axis=0
                )
                encoded = np.column_stack((parameter_features, repeated))
            else:
                encoded = np.asarray(joint_encoder(parameters, design), dtype=np.float64)
                if encoded.ndim != 2 or encoded.shape[0] != parameter_features.shape[0]:
                    raise ValueError(
                        "joint encoder must return one feature row per particle"
                    )
            raw_rows.append(encoded)
            prediction = np.asarray(exact_predictor(parameters, design))
            if prediction.shape[0] != parameter_features.shape[0]:
                raise ValueError("exact predictor output has the wrong particle count")
            shape = prediction.shape[1:]
            if output_shape is None:
                output_shape = shape
            elif shape != output_shape:
                raise ValueError("exact predictor output shape changes across designs")
            targets.append(prediction.reshape(prediction.shape[0], -1))
        raw = np.vstack(raw_rows)
        target = np.vstack(targets)
        mean = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        matrix = _polynomial_features((raw - mean) / scale, degree)
        gram = matrix.T @ matrix
        penalty = np.eye(gram.shape[0]) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(gram + penalty, matrix.T @ target)
        assert output_shape is not None
        return cls(
            parameter_encoder=parameter_encoder,
            design_encoder=design_encoder,
            joint_encoder=joint_encoder,
            feature_mean=mean,
            feature_scale=scale,
            coefficients=coefficients,
            output_shape=output_shape,
            degree=int(degree),
        )

    def __call__(
        self, parameters: Mapping[str, np.ndarray], design: Any
    ) -> np.ndarray:
        parameter_features = np.asarray(
            self.parameter_encoder(parameters), dtype=np.float64
        )
        if self.joint_encoder is None:
            design_features = _encoded_design(self.design_encoder, design)
            repeated = np.repeat(
                design_features[None, :], parameter_features.shape[0], axis=0
            )
            raw = np.column_stack((parameter_features, repeated))
        else:
            raw = np.asarray(self.joint_encoder(parameters, design), dtype=np.float64)
            if raw.ndim != 2 or raw.shape[0] != parameter_features.shape[0]:
                raise ValueError("joint encoder must return one feature row per particle")
        if raw.shape[1] != self.feature_mean.size:
            raise ValueError("encoded feature count differs from surrogate training")
        matrix = _polynomial_features(
            (raw - self.feature_mean) / self.feature_scale, self.degree
        )
        values = matrix @ self.coefficients
        return values.reshape((parameter_features.shape[0], *self.output_shape))

    def validate(
        self,
        exact_predictor: Predictor,
        parameters: Mapping[str, np.ndarray],
        designs: Sequence[Any],
    ) -> SurrogateValidation:
        """Compare against exact predictions on held-out particle/action pairs."""

        exact_rows: list[np.ndarray] = []
        approximate_rows: list[np.ndarray] = []
        for design in designs:
            exact_rows.append(np.asarray(exact_predictor(parameters, design)).reshape(-1))
            approximate_rows.append(np.asarray(self(parameters, design)).reshape(-1))
        if not exact_rows:
            raise ValueError("validation designs must not be empty")
        exact = np.concatenate(exact_rows)
        approximate = np.concatenate(approximate_rows)
        residual = approximate - exact
        rmse = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
        scale = float(np.sqrt(np.mean(np.abs(exact - np.mean(exact)) ** 2)))
        normalized = rmse / scale if scale > 0.0 else (0.0 if rmse == 0.0 else np.inf)
        exact_real = np.column_stack((exact.real, exact.imag)).reshape(-1)
        approximate_real = np.column_stack(
            (approximate.real, approximate.imag)
        ).reshape(-1)
        if np.std(exact_real) == 0.0 or np.std(approximate_real) == 0.0:
            correlation = 1.0 if np.array_equal(exact_real, approximate_real) else 0.0
        else:
            correlation = float(np.corrcoef(exact_real, approximate_real)[0, 1])
        return SurrogateValidation(
            root_mean_square_error=rmse,
            normalized_root_mean_square_error=float(normalized),
            maximum_absolute_error=float(np.max(np.abs(residual))),
            correlation=correlation,
            predictions=int(exact.size),
        )


def _real_observation_features(prediction: np.ndarray) -> np.ndarray:
    flat = np.asarray(prediction).reshape(prediction.shape[0], -1)
    if np.iscomplexobj(flat):
        return np.column_stack((flat.real, flat.imag))
    return np.asarray(flat, dtype=np.float64)


@dataclass(frozen=True)
class LaplaceInformationGain:
    """Fast linear-Gaussian information approximation for candidate screening.

    The weighted predictive covariance supplies the local Fisher/Laplace
    approximation. ``backend='jax'`` optionally uses JAX for the eigensolve;
    prediction and posterior handling remain backend-neutral.
    """

    backend: str = "numpy"

    def __post_init__(self) -> None:
        if self.backend not in ("numpy", "jax"):
            raise ValueError("backend must be 'numpy' or 'jax'")

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate:
        del rng
        prediction = model.predict(posterior.parameters, design)
        features = _real_observation_features(prediction)
        sigma = np.asarray(getattr(model.likelihood, "sigma", None), dtype=np.float64)
        if sigma.size == 0 or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise TypeError("LaplaceInformationGain requires a Gaussian sigma")
        sigma_flat = np.broadcast_to(sigma, prediction.shape[1:] or ()).reshape(-1)
        if np.iscomplexobj(prediction):
            sigma_flat = np.concatenate((sigma_flat, sigma_flat))
        whitened = features / sigma_flat[None, :]
        mean = np.sum(posterior.weights[:, None] * whitened, axis=0)
        centered = whitened - mean
        covariance = (centered * posterior.weights[:, None]).T @ centered
        if self.backend == "jax":
            try:
                import jax.numpy as jnp
            except ImportError as exc:
                raise ImportError(
                    "backend='jax' requires the optional jax dependency"
                ) from exc
            eigenvalues = np.asarray(jnp.linalg.eigvalsh(jnp.asarray(covariance)))
        else:
            eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        value = float(0.5 * np.sum(np.log1p(eigenvalues)))
        return UtilityEstimate(value, 0.0, np.array([value]), "nats (Laplace)")


@dataclass(frozen=True)
class AdaptiveMonteCarloUtility:
    """Increase nested-Monte-Carlo effort until its standard error is resolved."""

    minimum_samples: int = 16
    maximum_samples: int = 256
    relative_tolerance: float = 0.1
    absolute_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0 or self.maximum_samples < self.minimum_samples:
            raise ValueError("sample limits must satisfy 0 < minimum <= maximum")
        if self.relative_tolerance < 0.0 or self.absolute_tolerance < 0.0:
            raise ValueError("Monte Carlo tolerances must be non-negative")

    def estimate(
        self,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design: Any,
        rng: np.random.Generator,
    ) -> UtilityEstimate:
        samples: list[np.ndarray] = []
        total = 0
        batch = self.minimum_samples
        units = "nats"
        while total < self.maximum_samples:
            count = min(batch, self.maximum_samples - total)
            estimate = ExpectedInformationGain(count).estimate(
                model, posterior, design, rng
            )
            samples.append(estimate.samples)
            total += count
            combined = np.concatenate(samples)
            mean = float(np.mean(combined))
            standard_error = (
                float(np.std(combined, ddof=1) / np.sqrt(combined.size))
                if combined.size > 1
                else 0.0
            )
            threshold = max(
                self.absolute_tolerance,
                self.relative_tolerance * abs(mean),
            )
            units = estimate.units
            if standard_error <= threshold:
                return UtilityEstimate(mean, standard_error, combined, units)
            batch *= 2
        combined = np.concatenate(samples)
        final_standard_error = (
            float(np.std(combined, ddof=1) / np.sqrt(combined.size))
            if combined.size > 1
            else 0.0
        )
        return UtilityEstimate(
            float(np.mean(combined)),
            final_standard_error,
            combined,
            units,
        )


class TwoStageDesignSession(AdaptiveDesignSession):
    """Screen all actions cheaply, then refine only the leading candidates."""

    def __init__(
        self,
        *,
        screening_utility: DesignUtility,
        refinement_utility: DesignUtility,
        finalists: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(utility=refinement_utility, **kwargs)
        if finalists <= 0:
            raise ValueError("finalists must be positive")
        self.screening_utility = screening_utility
        self.refinement_utility = refinement_utility
        self.finalists = int(finalists)

    def ask(self) -> DesignRecommendation:
        """Return refined rankings after a common-random-number screen."""

        evaluation_seed = int(
            self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        screened: list[CandidateScore] = []
        for index, design in enumerate(self.design_space.actions):
            feasible, messages = evaluate_constraints(design, self.constraints)
            if not feasible:
                screened.append(
                    CandidateScore(design, index, False, messages, None, None, None)
                )
                continue
            seconds = float(self.cost.seconds(design))
            if not np.isfinite(seconds) or seconds <= 0.0:
                raise ValueError("physical acquisition cost must be finite and positive")
            estimate = self.screening_utility.estimate(
                self.model,
                self.posterior,
                design,
                np.random.default_rng(evaluation_seed),
            )
            screened.append(
                CandidateScore(
                    design,
                    index,
                    True,
                    (),
                    estimate,
                    seconds,
                    estimate.value / seconds,
                )
            )
        feasible = [score for score in screened if score.feasible]
        if not feasible:
            reasons = sorted({message for score in screened for message in score.messages})
            raise ValueError("no feasible candidate designs: " + "; ".join(reasons))
        leaders = sorted(
            feasible,
            key=lambda score: (-float(score.utility_rate), score.design_index),
        )[: self.finalists]
        leader_indices = {score.design_index for score in leaders}
        refined: list[CandidateScore] = []
        for score in screened:
            if score.design_index not in leader_indices:
                refined.append(score)
                continue
            estimate = self.refinement_utility.estimate(
                self.model,
                self.posterior,
                score.design,
                np.random.default_rng(evaluation_seed),
            )
            refined.append(
                CandidateScore(
                    score.design,
                    score.design_index,
                    True,
                    (),
                    estimate,
                    score.cost_seconds,
                    estimate.value / float(score.cost_seconds),
                )
            )
        ranked_leaders = sorted(
            (score for score in refined if score.design_index in leader_indices),
            key=lambda score: (
                -float(score.utility_rate),
                -float(score.utility.value),
                score.design_index,
            ),
        )
        leader_rank = {
            score.design_index: rank for rank, score in enumerate(ranked_leaders)
        }
        scores = tuple(
            sorted(
                refined,
                key=lambda score: (
                    (
                        0
                        if score.design_index in leader_indices
                        else (1 if score.feasible else 2)
                    ),
                    leader_rank.get(score.design_index, score.design_index),
                ),
            )
        )
        self.ask_count += 1
        return DesignRecommendation(ranked_leaders[0], scores, evaluation_seed)
