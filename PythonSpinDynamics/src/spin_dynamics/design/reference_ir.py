"""Exact-grid Bayesian design reference for inversion recovery.

The Phase 0 model is

``y = baseline + amplitude * (1 - 2 exp(-delay / T1)) + error``

with Gaussian error. An observation is the average of ``repetitions``
independent acquisitions, so its standard deviation is
``sigma / sqrt(repetitions)``. The posterior is exact on a user-supplied
discrete joint grid over T1, amplitude, baseline, and single-acquisition sigma.

Candidate designs maximize goal-oriented information about T1, marginalizing
the nuisance dimensions. Expected information gain is evaluated
deterministically with Gauss-Hermite quadrature and divided by a physical-time
cost. This module is a small numerical reference, not an instrument driver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


_LOG_2PI = float(np.log(2.0 * np.pi))


def _axis(values: Iterable[float], name: str, *, positive: bool = False) -> np.ndarray:
    out = np.asarray(tuple(values), dtype=np.float64)
    if out.ndim != 1 or out.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if np.any(~np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(out <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    if np.any(np.diff(out) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return out


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.where(finite, values - maximum, -np.inf)
    total = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.where(finite, maximum + np.log(total), -np.inf)
    if axis is None:
        return np.asarray(result.reshape(()))
    return np.squeeze(result, axis=axis)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
    normalizer = float(_logsumexp(values))
    if not np.isfinite(normalizer):
        raise ValueError("posterior weights have zero or non-finite total mass")
    return values - normalizer


def _normal_logpdf(
    observations: np.ndarray,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> np.ndarray:
    residual = (observations - means) / standard_deviations
    return -0.5 * (residual * residual + _LOG_2PI) - np.log(standard_deviations)


def inversion_recovery_signal(
    delay_seconds: float | np.ndarray,
    t1_seconds: float | np.ndarray,
    amplitude: float | np.ndarray = 1.0,
    baseline: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Return the ideal inversion-recovery signal.

    The inversion efficiency is fixed at the ideal value of two. Imperfect
    inversion can be represented in Phase 0 by broad amplitude and baseline
    nuisance priors; an explicit efficiency parameter belongs in a later model.
    """

    delay = np.asarray(delay_seconds, dtype=np.float64)
    t1 = np.asarray(t1_seconds, dtype=np.float64)
    if np.any(delay < 0.0) or np.any(~np.isfinite(delay)):
        raise ValueError("delay_seconds must be finite and non-negative")
    if np.any(t1 <= 0.0) or np.any(~np.isfinite(t1)):
        raise ValueError("t1_seconds must be finite and positive")
    return np.asarray(baseline) + np.asarray(amplitude) * (
        1.0 - 2.0 * np.exp(-delay / t1)
    )


@dataclass(frozen=True)
class IRDesign:
    """One inversion-recovery acquisition action."""

    delay_seconds: float
    repetitions: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.delay_seconds) or self.delay_seconds < 0.0:
            raise ValueError("delay_seconds must be finite and non-negative")
        if isinstance(self.repetitions, bool) or int(self.repetitions) != self.repetitions:
            raise ValueError("repetitions must be a positive integer")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be a positive integer")


@dataclass(frozen=True)
class IRAcquisitionCost:
    """Physical time used to rank inversion-recovery actions.

    ``fixed_overhead_seconds`` is paid once per recommended action. Delay,
    readout, dead time, and recovery are paid for every repetition.
    """

    readout_seconds: float = 0.0
    dead_time_seconds: float = 0.0
    recovery_seconds: float = 0.0
    fixed_overhead_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "readout_seconds",
            "dead_time_seconds",
            "recovery_seconds",
            "fixed_overhead_seconds",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def seconds(self, design: IRDesign) -> float:
        """Return physical seconds required for ``design``."""

        per_repetition = (
            design.delay_seconds
            + self.readout_seconds
            + self.dead_time_seconds
            + self.recovery_seconds
        )
        total = self.fixed_overhead_seconds + design.repetitions * per_repetition
        if total <= 0.0:
            raise ValueError("the physical acquisition cost must be positive")
        return float(total)


@dataclass(frozen=True)
class IRMeasurement:
    """One averaged scalar observation and the action that produced it."""

    design: IRDesign
    value: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.value):
            raise ValueError("measurement value must be finite")


@dataclass(frozen=True, eq=False)
class IRGridPrior:
    """Discrete joint prior for the Phase 0 inversion-recovery model.

    ``weights`` has shape ``(n_t1, n_amplitude, n_baseline, n_sigma)``. If it
    is omitted, every Cartesian grid point receives equal probability.
    """

    t1_seconds: np.ndarray
    amplitude: np.ndarray
    baseline: np.ndarray
    sigma: np.ndarray
    weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        t1 = _axis(self.t1_seconds, "t1_seconds", positive=True)
        amplitude = _axis(self.amplitude, "amplitude")
        baseline = _axis(self.baseline, "baseline")
        sigma = _axis(self.sigma, "sigma", positive=True)
        object.__setattr__(self, "t1_seconds", t1)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "baseline", baseline)
        object.__setattr__(self, "sigma", sigma)

        shape = (t1.size, amplitude.size, baseline.size, sigma.size)
        if self.weights is None:
            weights = np.full(shape, 1.0 / float(np.prod(shape)))
        else:
            weights = np.asarray(self.weights, dtype=np.float64)
            if weights.shape != shape:
                raise ValueError(f"weights must have shape {shape}")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
                raise ValueError("weights must be finite and non-negative")
            total = float(np.sum(weights))
            if total <= 0.0:
                raise ValueError("weights must have positive total mass")
            weights = weights / total
        object.__setattr__(self, "weights", weights)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.t1_seconds.size,
            self.amplitude.size,
            self.baseline.size,
            self.sigma.size,
        )

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def to_dict(self) -> dict[str, Any]:
        return {
            "t1_seconds": self.t1_seconds.tolist(),
            "amplitude": self.amplitude.tolist(),
            "baseline": self.baseline.tolist(),
            "sigma": self.sigma.tolist(),
            "weights": np.asarray(self.weights).tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IRGridPrior":
        return cls(
            t1_seconds=np.asarray(data["t1_seconds"], dtype=np.float64),
            amplitude=np.asarray(data["amplitude"], dtype=np.float64),
            baseline=np.asarray(data["baseline"], dtype=np.float64),
            sigma=np.asarray(data["sigma"], dtype=np.float64),
            weights=np.asarray(data["weights"], dtype=np.float64),
        )


class IRGridPosterior:
    """Exact posterior on an :class:`IRGridPrior` Cartesian grid."""

    def __init__(
        self,
        prior: IRGridPrior,
        log_weights: np.ndarray | None = None,
    ) -> None:
        self.prior = prior
        meshes = np.meshgrid(
            prior.t1_seconds,
            prior.amplitude,
            prior.baseline,
            prior.sigma,
            indexing="ij",
        )
        self._t1 = meshes[0].reshape(-1)
        self._amplitude = meshes[1].reshape(-1)
        self._baseline = meshes[2].reshape(-1)
        self._sigma = meshes[3].reshape(-1)
        self._t1_index = np.indices(prior.shape, sparse=False)[0].reshape(-1)
        if log_weights is None:
            weights = np.asarray(prior.weights, dtype=np.float64).reshape(-1)
            initial = np.full(weights.shape, -np.inf)
            positive = weights > 0.0
            initial[positive] = np.log(weights[positive])
            self._log_weights = _normalize_log_weights(initial)
        else:
            values = np.asarray(log_weights, dtype=np.float64).reshape(-1)
            if values.size != prior.size:
                raise ValueError("log_weights size must match the prior grid")
            # Preserve an already normalized checkpoint bit for bit. Reapplying
            # log-sum-exp can otherwise introduce a last-bit offset during a
            # JSON round trip even though the probabilities are unchanged.
            normalizer = float(_logsumexp(values))
            if not np.isfinite(normalizer):
                raise ValueError("posterior weights have zero or non-finite total mass")
            if abs(normalizer) <= 1e-14:
                self._log_weights = values.copy()
            else:
                self._log_weights = values - normalizer

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

    def t1_marginal(self) -> np.ndarray:
        """Return posterior probabilities on ``prior.t1_seconds``."""

        return np.bincount(
            self._t1_index,
            weights=self.weights,
            minlength=self.prior.t1_seconds.size,
        )

    @property
    def t1_mean(self) -> float:
        return float(np.dot(self.t1_marginal(), self.prior.t1_seconds))

    @property
    def t1_standard_deviation(self) -> float:
        marginal = self.t1_marginal()
        centered = self.prior.t1_seconds - self.t1_mean
        return float(np.sqrt(np.dot(marginal, centered * centered)))

    def t1_credible_interval(self, probability: float = 0.95) -> tuple[float, float]:
        """Return an equal-tailed interval on the discrete T1 grid."""

        if not 0.0 < probability < 1.0:
            raise ValueError("probability must lie strictly between zero and one")
        marginal = self.t1_marginal()
        cumulative = np.cumsum(marginal)
        tail = 0.5 * (1.0 - probability)
        lo = min(int(np.searchsorted(cumulative, tail, side="left")), marginal.size - 1)
        hi = min(
            int(np.searchsorted(cumulative, 1.0 - tail, side="left")),
            marginal.size - 1,
        )
        return float(self.prior.t1_seconds[lo]), float(self.prior.t1_seconds[hi])

    def t1_credible_width(self, probability: float = 0.95) -> float:
        low, high = self.t1_credible_interval(probability)
        return high - low

    def predicted_means(self, design: IRDesign) -> np.ndarray:
        return inversion_recovery_signal(
            design.delay_seconds,
            self._t1,
            self._amplitude,
            self._baseline,
        ).reshape(-1)

    def predicted_standard_deviations(self, design: IRDesign) -> np.ndarray:
        return self._sigma / np.sqrt(float(design.repetitions))

    def updated(self, measurement: IRMeasurement) -> "IRGridPosterior":
        """Return the exact grid posterior after one independent observation."""

        means = self.predicted_means(measurement.design)
        standard_deviations = self.predicted_standard_deviations(measurement.design)
        likelihood = _normal_logpdf(measurement.value, means, standard_deviations)
        return IRGridPosterior(self.prior, self._log_weights + likelihood)

    def t1_expected_information_gain(
        self,
        design: IRDesign,
        *,
        quadrature_order: int = 12,
        chunk_size: int = 4096,
    ) -> float:
        """Return ``I(T1; Y | design, history)`` in nats.

        Amplitude, baseline, and sigma are marginalized under the current
        joint posterior. Gauss-Hermite quadrature integrates observations from
        every Gaussian mixture component. ``chunk_size`` limits temporary
        likelihood matrices without changing the calculation.
        """

        if quadrature_order < 3:
            raise ValueError("quadrature_order must be at least 3")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        means = self.predicted_means(design)
        standard_deviations = self.predicted_standard_deviations(design)
        weights = self.weights
        nodes, quadrature_weights = np.polynomial.hermite.hermgauss(quadrature_order)
        quadrature_weights = quadrature_weights / np.sqrt(np.pi)
        sqrt_two = np.sqrt(2.0)
        information = 0.0

        for t1_index in range(self.prior.t1_seconds.size):
            component_indices = np.flatnonzero(self._t1_index == t1_index)
            group_weights = weights[component_indices]
            group_probability = float(np.sum(group_weights))
            if group_probability <= 0.0:
                continue

            observations = (
                means[component_indices, np.newaxis]
                + sqrt_two
                * standard_deviations[component_indices, np.newaxis]
                * nodes[np.newaxis, :]
            ).reshape(-1)
            expectation_weights = (
                group_weights[:, np.newaxis]
                * quadrature_weights[np.newaxis, :]
            ).reshape(-1)
            conditional_log_weights = (
                self._log_weights[component_indices] - np.log(group_probability)
            )

            for start in range(0, observations.size, chunk_size):
                stop = min(start + chunk_size, observations.size)
                observed = observations[start:stop, np.newaxis]
                global_terms = _normal_logpdf(
                    observed,
                    means[np.newaxis, :],
                    standard_deviations[np.newaxis, :],
                ) + self._log_weights[np.newaxis, :]
                conditional_terms = _normal_logpdf(
                    observed,
                    means[component_indices][np.newaxis, :],
                    standard_deviations[component_indices][np.newaxis, :],
                ) + conditional_log_weights[np.newaxis, :]
                log_global = _logsumexp(global_terms, axis=1)
                log_conditional = _logsumexp(conditional_terms, axis=1)
                information += float(
                    np.dot(
                        expectation_weights[start:stop],
                        log_conditional - log_global,
                    )
                )

        # Quadrature and floating-point cancellation can yield a tiny negative
        # value even though mutual information is non-negative.
        return max(0.0, information)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior": self.prior.to_dict(),
            "log_weights": self._log_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IRGridPosterior":
        return cls(
            IRGridPrior.from_dict(data["prior"]),
            np.asarray(data["log_weights"], dtype=np.float64),
        )


@dataclass(frozen=True)
class IRDesignScore:
    """Goal-oriented information and physical cost for one candidate."""

    design: IRDesign
    expected_information_nats: float
    cost_seconds: float
    information_rate_nats_per_second: float


@dataclass(frozen=True)
class IRRecommendation:
    """Ranked Phase 0 candidate designs."""

    best: IRDesignScore
    scores: tuple[IRDesignScore, ...]


def recommend_ir_design(
    posterior: IRGridPosterior,
    designs: Sequence[IRDesign],
    cost: IRAcquisitionCost,
    *,
    quadrature_order: int = 12,
    chunk_size: int = 4096,
) -> IRRecommendation:
    """Rank candidates by expected T1 information per physical second."""

    candidates = tuple(designs)
    if not candidates:
        raise ValueError("designs must not be empty")
    scores: list[IRDesignScore] = []
    for design in candidates:
        information = posterior.t1_expected_information_gain(
            design,
            quadrature_order=quadrature_order,
            chunk_size=chunk_size,
        )
        seconds = cost.seconds(design)
        scores.append(
            IRDesignScore(
                design=design,
                expected_information_nats=information,
                cost_seconds=seconds,
                information_rate_nats_per_second=information / seconds,
            )
        )
    ranked = tuple(
        sorted(
            scores,
            key=lambda score: (
                -score.information_rate_nats_per_second,
                -score.expected_information_nats,
                score.cost_seconds,
                score.design.delay_seconds,
                score.design.repetitions,
            ),
        )
    )
    return IRRecommendation(best=ranked[0], scores=ranked)


class IRAdaptiveSession:
    """Replayable ask/tell loop for the Phase 0 reference problem."""

    def __init__(
        self,
        posterior: IRGridPosterior,
        designs: Sequence[IRDesign],
        cost: IRAcquisitionCost,
        *,
        quadrature_order: int = 12,
        chunk_size: int = 4096,
        history: Sequence[IRMeasurement] = (),
    ) -> None:
        self.posterior = posterior
        self.designs = tuple(designs)
        if not self.designs:
            raise ValueError("designs must not be empty")
        self.cost = cost
        if quadrature_order < 3:
            raise ValueError("quadrature_order must be at least 3")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.quadrature_order = int(quadrature_order)
        self.chunk_size = int(chunk_size)
        self.history = list(history)

    def ask(self) -> IRRecommendation:
        """Recommend the next action without changing the posterior."""

        return recommend_ir_design(
            self.posterior,
            self.designs,
            self.cost,
            quadrature_order=self.quadrature_order,
            chunk_size=self.chunk_size,
        )

    def tell(self, design: IRDesign, value: float) -> IRMeasurement:
        """Update the posterior with one completed acquisition."""

        if design not in self.designs:
            raise ValueError("design was not part of this session's candidate set")
        measurement = IRMeasurement(design=design, value=float(value))
        self.posterior = self.posterior.updated(measurement)
        self.history.append(measurement)
        return measurement

    @property
    def elapsed_seconds(self) -> float:
        return float(sum(self.cost.seconds(item.design) for item in self.history))

    def should_stop(
        self,
        *,
        credible_probability: float = 0.95,
        maximum_width_seconds: float | None = None,
        maximum_relative_width: float | None = None,
    ) -> bool:
        """Return whether the requested T1 precision has been reached."""

        if maximum_width_seconds is None and maximum_relative_width is None:
            raise ValueError("at least one stopping width must be supplied")
        if maximum_width_seconds is not None and maximum_width_seconds < 0.0:
            raise ValueError("maximum_width_seconds must be non-negative")
        if maximum_relative_width is not None and maximum_relative_width < 0.0:
            raise ValueError("maximum_relative_width must be non-negative")
        width = self.posterior.t1_credible_width(credible_probability)
        absolute_ok = (
            maximum_width_seconds is None or width <= maximum_width_seconds
        )
        relative_ok = (
            maximum_relative_width is None
            or width / self.posterior.t1_mean <= maximum_relative_width
        )
        return bool(absolute_ok and relative_ok)

    def to_dict(self) -> dict[str, Any]:
        return {
            "posterior": self.posterior.to_dict(),
            "designs": [
                {
                    "delay_seconds": item.delay_seconds,
                    "repetitions": item.repetitions,
                }
                for item in self.designs
            ],
            "cost": {
                "readout_seconds": self.cost.readout_seconds,
                "dead_time_seconds": self.cost.dead_time_seconds,
                "recovery_seconds": self.cost.recovery_seconds,
                "fixed_overhead_seconds": self.cost.fixed_overhead_seconds,
            },
            "quadrature_order": self.quadrature_order,
            "chunk_size": self.chunk_size,
            "history": [
                {
                    "design": {
                        "delay_seconds": item.design.delay_seconds,
                        "repetitions": item.design.repetitions,
                    },
                    "value": item.value,
                }
                for item in self.history
            ],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IRAdaptiveSession":
        def design_from(payload: Mapping[str, Any]) -> IRDesign:
            return IRDesign(
                delay_seconds=float(payload["delay_seconds"]),
                repetitions=int(payload["repetitions"]),
            )

        designs = tuple(design_from(item) for item in data["designs"])
        history = tuple(
            IRMeasurement(
                design=design_from(item["design"]),
                value=float(item["value"]),
            )
            for item in data["history"]
        )
        return cls(
            posterior=IRGridPosterior.from_dict(data["posterior"]),
            designs=designs,
            cost=IRAcquisitionCost(**data["cost"]),
            quadrature_order=int(data["quadrature_order"]),
            chunk_size=int(data["chunk_size"]),
            history=history,
        )

    @classmethod
    def from_json(cls, payload: str) -> "IRAdaptiveSession":
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class IRTruth:
    """Ground-truth values used only by synthetic Phase 0 benchmarks."""

    t1_seconds: float
    amplitude: float = 1.0
    baseline: float = 0.0
    sigma: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(self.t1_seconds) or self.t1_seconds <= 0.0:
            raise ValueError("t1_seconds must be finite and positive")
        if not np.isfinite(self.amplitude) or not np.isfinite(self.baseline):
            raise ValueError("amplitude and baseline must be finite")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")


def simulate_ir_observation(
    truth: IRTruth,
    design: IRDesign,
    rng: np.random.Generator,
) -> float:
    """Draw one averaged synthetic observation from the Phase 0 model."""

    mean = float(
        inversion_recovery_signal(
            design.delay_seconds,
            truth.t1_seconds,
            truth.amplitude,
            truth.baseline,
        )
    )
    standard_deviation = truth.sigma / np.sqrt(float(design.repetitions))
    return float(rng.normal(mean, standard_deviation))


@dataclass(frozen=True)
class IRBenchmarkResult:
    """Outcome of one adaptive or fixed synthetic reference trial."""

    policy: str
    reached_target: bool
    elapsed_seconds: float
    measurements: tuple[IRMeasurement, ...]
    posterior: IRGridPosterior


def run_ir_reference_trial(
    *,
    prior: IRGridPrior,
    truth: IRTruth,
    candidate_designs: Sequence[IRDesign],
    cost: IRAcquisitionCost,
    rng: np.random.Generator,
    policy: str = "adaptive",
    fixed_schedule: Sequence[IRDesign] | None = None,
    maximum_actions: int = 20,
    minimum_actions: int = 2,
    credible_probability: float = 0.95,
    maximum_width_seconds: float | None = None,
    maximum_relative_width: float | None = 0.25,
    quadrature_order: int = 12,
) -> IRBenchmarkResult:
    """Run one synthetic equal-model benchmark trial.

    ``policy='adaptive'`` maximizes target information rate. ``policy='fixed'``
    cycles through ``fixed_schedule`` without consulting the observations.
    Both policies use the same posterior update and stopping rule.
    """

    if policy not in ("adaptive", "fixed"):
        raise ValueError("policy must be 'adaptive' or 'fixed'")
    if maximum_actions <= 0:
        raise ValueError("maximum_actions must be positive")
    if minimum_actions < 0 or minimum_actions > maximum_actions:
        raise ValueError("minimum_actions must lie between zero and maximum_actions")
    candidates = tuple(candidate_designs)
    if not candidates:
        raise ValueError("candidate_designs must not be empty")
    schedule = tuple(fixed_schedule) if fixed_schedule is not None else candidates
    if policy == "fixed" and not schedule:
        raise ValueError("fixed_schedule must not be empty")
    if any(item not in candidates for item in schedule):
        raise ValueError("fixed_schedule must contain only candidate designs")

    session = IRAdaptiveSession(
        IRGridPosterior(prior),
        candidates,
        cost,
        quadrature_order=quadrature_order,
    )
    reached = False
    for action_index in range(maximum_actions):
        if policy == "adaptive":
            design = session.ask().best.design
        else:
            design = schedule[action_index % len(schedule)]
        value = simulate_ir_observation(truth, design, rng)
        session.tell(design, value)
        if len(session.history) >= minimum_actions and session.should_stop(
            credible_probability=credible_probability,
            maximum_width_seconds=maximum_width_seconds,
            maximum_relative_width=maximum_relative_width,
        ):
            reached = True
            break

    return IRBenchmarkResult(
        policy=policy,
        reached_target=reached,
        elapsed_seconds=session.elapsed_seconds,
        measurements=tuple(session.history),
        posterior=session.posterior,
    )
