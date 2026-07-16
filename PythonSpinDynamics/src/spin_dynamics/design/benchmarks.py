"""End-to-end adaptive-versus-fixed experiment-design benchmarks.

The benchmark clock is deliberately split into physical acquisition seconds,
Bayesian planning wall time, and one-time prediction-table construction.  A
table is built from exact Phase 2 adapter predictions and reused across paired
trials; this Phase 3 acceleration changes benchmark throughput, not the signals
or posterior updates being compared.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from spin_dynamics.design.constraints import DesignConstraint
from spin_dynamics.design.diagnostics import QuantityOfInterest, summarize_quantity
from spin_dynamics.design.models import PredictiveModel, Predictor
from spin_dynamics.design.posterior import ParticlePosterior, posterior_from_state
from spin_dynamics.design.session import AdaptiveDesignSession
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import ObservationLikelihood, PhysicalCost, StopRule
from spin_dynamics.design.utilities import DesignUtility


@dataclass(frozen=True, eq=False)
class CandidatePredictionTable:
    """Exact predictions for fixed particles and a finite candidate space."""

    parameters: Mapping[str, np.ndarray]
    design_space: CandidateDesignSpace
    predictions: tuple[np.ndarray, ...]
    build_seconds: float

    @classmethod
    def build(
        cls,
        predictor: Predictor,
        parameters: Mapping[str, np.ndarray],
        design_space: CandidateDesignSpace,
    ) -> "CandidatePredictionTable":
        """Evaluate each candidate once over the complete particle support."""

        reference = {name: np.asarray(value).copy() for name, value in parameters.items()}
        count = next(iter(reference.values())).shape[0]
        predictions: list[np.ndarray] = []
        start = perf_counter()
        event_shape: tuple[int, ...] | None = None
        for design in design_space.actions:
            value = np.asarray(predictor(reference, design))
            if value.ndim == 0 or value.shape[0] != count:
                raise ValueError("table predictor must lead with the particle count")
            if np.any(~np.isfinite(value)):
                raise ValueError("prediction table must contain finite values")
            if event_shape is None:
                event_shape = value.shape[1:]
            elif value.shape[1:] != event_shape:
                raise ValueError("prediction event shape changes across candidates")
            predictions.append(value.copy())
        elapsed = perf_counter() - start
        return cls(reference, design_space, tuple(predictions), float(elapsed))

    @property
    def particle_count(self) -> int:
        return next(iter(self.parameters.values())).shape[0]

    def __call__(
        self, parameters: Mapping[str, np.ndarray], design: Any
    ) -> np.ndarray:
        """Return cached predictions after checking particle identity and order."""

        if set(parameters) != set(self.parameters):
            raise ValueError("prediction-table parameter names differ from training")
        for name, reference in self.parameters.items():
            if not np.array_equal(np.asarray(parameters[name]), reference):
                raise ValueError(
                    "prediction table requires the original particle values and order"
                )
        return self.predictions[self.design_space.index(design)].copy()

    def truth_prediction(self, design: Any, particle_index: int) -> np.ndarray:
        """Return one exact table entry used as an equal-model synthetic truth."""

        if particle_index < 0 or particle_index >= self.particle_count:
            raise IndexError("truth particle index is out of range")
        return np.asarray(
            self.predictions[self.design_space.index(design)][particle_index]
        ).copy()


@dataclass(frozen=True, eq=False)
class AdapterBenchmarkProblem:
    """A fair finite-support comparison shared by adaptive and fixed policies."""

    name: str
    prediction_table: CandidatePredictionTable
    likelihood: ObservationLikelihood
    initial_posterior: ParticlePosterior
    utility: DesignUtility
    cost: PhysicalCost
    quantity: QuantityOfInterest
    stopping_rule: StopRule
    fixed_schedule: tuple[Any, ...]
    target_unit: str
    minimum_actions: int = 2
    maximum_actions: int = 10
    credible_probability: float = 0.9
    constraints: tuple[DesignConstraint, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("benchmark name must not be empty")
        if self.initial_posterior.particle_count != self.prediction_table.particle_count:
            raise ValueError("posterior and prediction table particle counts differ")
        if self.minimum_actions < 0 or self.maximum_actions < self.minimum_actions:
            raise ValueError("action limits must satisfy 0 <= minimum <= maximum")
        if not 0.0 < self.credible_probability < 1.0:
            raise ValueError("credible_probability must lie in (0, 1)")
        if not self.fixed_schedule:
            raise ValueError("fixed_schedule must not be empty")
        if any(
            design not in self.prediction_table.design_space.actions
            for design in self.fixed_schedule
        ):
            raise ValueError("fixed_schedule contains an unknown candidate")

    @property
    def model(self) -> PredictiveModel:
        return PredictiveModel(self.prediction_table, self.likelihood)


@dataclass(frozen=True)
class BenchmarkTracePoint:
    """Posterior state after one synthetic acquisition."""

    action_number: int
    design_index: int
    physical_seconds: float
    planning_seconds: float
    posterior_mean: float
    interval_lower: float
    interval_upper: float
    interval_width: float


@dataclass(frozen=True)
class BenchmarkTrial:
    """Outcome of one policy for one paired truth/noise realization."""

    policy: str
    trial_index: int
    truth_particle_index: int
    truth_value: float
    reached_target: bool
    physical_seconds: float
    planning_seconds: float
    posterior_mean: float
    interval_lower: float
    interval_upper: float
    interval_width: float
    covered: bool
    error: float
    trace: tuple[BenchmarkTracePoint, ...]


@dataclass(frozen=True)
class BenchmarkPolicySummary:
    """Aggregate performance of one policy over paired trials."""

    policy: str
    trials: int
    success_fraction: float
    median_physical_seconds: float
    median_success_seconds: float | None
    median_actions: float
    root_mean_square_error: float
    bias: float
    coverage_fraction: float
    median_interval_width: float
    median_planning_seconds: float


@dataclass(frozen=True)
class AdapterBenchmarkResult:
    """All paired trials, summaries, and one-time planning setup costs."""

    name: str
    target_unit: str
    prediction_table_seconds: float
    open_loop_schedule_seconds: float
    policies: tuple[str, ...]
    trials: tuple[BenchmarkTrial, ...]
    summaries: tuple[BenchmarkPolicySummary, ...]

    def summary(self, policy: str) -> BenchmarkPolicySummary:
        for item in self.summaries:
            if item.policy == policy:
                return item
        raise KeyError(policy)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible result without executable dependencies."""

        return {
            "name": self.name,
            "target_unit": self.target_unit,
            "prediction_table_seconds": self.prediction_table_seconds,
            "open_loop_schedule_seconds": self.open_loop_schedule_seconds,
            "policies": list(self.policies),
            "summaries": [vars(item) for item in self.summaries],
            "trials": [
                {
                    **{key: value for key, value in vars(trial).items() if key != "trace"},
                    "trace": [vars(point) for point in trial.trace],
                }
                for trial in self.trials
            ],
        }


def _fresh_posterior(posterior: ParticlePosterior) -> ParticlePosterior:
    return posterior_from_state(posterior.state_dict())


def _scalar_summary(
    posterior: ParticlePosterior,
    quantity: QuantityOfInterest,
    probability: float,
) -> tuple[float, float, float]:
    summary = summarize_quantity(posterior, quantity, probability=probability)
    if summary.mean.size != 1:
        raise ValueError("adapter benchmarks currently require a scalar target")
    return float(summary.mean[0]), float(summary.lower[0]), float(summary.upper[0])


def _prior_ranked_schedule(
    problem: AdapterBenchmarkProblem, seed: int
) -> tuple[tuple[Any, ...], float]:
    session = AdaptiveDesignSession(
        model=problem.model,
        posterior=_fresh_posterior(problem.initial_posterior),
        design_space=problem.prediction_table.design_space,
        utility=problem.utility,
        cost=problem.cost,
        constraints=problem.constraints,
        seed=seed,
    )
    start = perf_counter()
    recommendation = session.ask()
    elapsed = perf_counter() - start
    schedule = tuple(score.design for score in recommendation.scores if score.feasible)
    return schedule, float(elapsed)


def _run_trial(
    problem: AdapterBenchmarkProblem,
    *,
    policy: str,
    schedule: tuple[Any, ...],
    trial_index: int,
    truth_index: int,
    noise_seeds: np.ndarray,
    policy_seed: int,
) -> BenchmarkTrial:
    posterior = _fresh_posterior(problem.initial_posterior)
    session = AdaptiveDesignSession(
        model=problem.model,
        posterior=posterior,
        design_space=problem.prediction_table.design_space,
        utility=problem.utility,
        cost=problem.cost,
        constraints=problem.constraints,
        seed=policy_seed,
    )
    truth_values = np.asarray(problem.quantity(problem.initial_posterior.parameters))
    truth = float(truth_values.reshape(problem.initial_posterior.particle_count, -1)[truth_index, 0])
    planning_seconds = 0.0
    trace: list[BenchmarkTracePoint] = []
    reached = False
    for action_index in range(problem.maximum_actions):
        if policy == "adaptive":
            start = perf_counter()
            design = session.ask().best.design
            planning_seconds += perf_counter() - start
        else:
            design = schedule[action_index % len(schedule)]
        prediction = problem.prediction_table.truth_prediction(design, truth_index)
        observation = problem.likelihood.sample(
            prediction, np.random.default_rng(int(noise_seeds[action_index]))
        )
        session.tell(design, observation)
        mean, lower, upper = _scalar_summary(
            session.posterior, problem.quantity, problem.credible_probability
        )
        trace.append(
            BenchmarkTracePoint(
                action_number=action_index + 1,
                design_index=problem.prediction_table.design_space.index(design),
                physical_seconds=session.elapsed_seconds,
                planning_seconds=planning_seconds,
                posterior_mean=mean,
                interval_lower=lower,
                interval_upper=upper,
                interval_width=upper - lower,
            )
        )
        if len(trace) >= problem.minimum_actions and problem.stopping_rule.reached(
            session.posterior
        ):
            reached = True
            break
    final = trace[-1]
    return BenchmarkTrial(
        policy=policy,
        trial_index=trial_index,
        truth_particle_index=truth_index,
        truth_value=truth,
        reached_target=reached,
        physical_seconds=final.physical_seconds,
        planning_seconds=final.planning_seconds,
        posterior_mean=final.posterior_mean,
        interval_lower=final.interval_lower,
        interval_upper=final.interval_upper,
        interval_width=final.interval_width,
        covered=final.interval_lower <= truth <= final.interval_upper,
        error=final.posterior_mean - truth,
        trace=tuple(trace),
    )


def _summarize(policy: str, trials: Sequence[BenchmarkTrial]) -> BenchmarkPolicySummary:
    values = tuple(item for item in trials if item.policy == policy)
    if not values:
        raise ValueError(f"policy {policy!r} has no trials")
    success_times = [item.physical_seconds for item in values if item.reached_target]
    return BenchmarkPolicySummary(
        policy=policy,
        trials=len(values),
        success_fraction=float(np.mean([item.reached_target for item in values])),
        median_physical_seconds=float(np.median([item.physical_seconds for item in values])),
        median_success_seconds=(
            float(np.median(success_times)) if success_times else None
        ),
        median_actions=float(np.median([len(item.trace) for item in values])),
        root_mean_square_error=float(
            np.sqrt(np.mean([item.error * item.error for item in values]))
        ),
        bias=float(np.mean([item.error for item in values])),
        coverage_fraction=float(np.mean([item.covered for item in values])),
        median_interval_width=float(np.median([item.interval_width for item in values])),
        median_planning_seconds=float(
            np.median([item.planning_seconds for item in values])
        ),
    )


def run_adapter_benchmark(
    problem: AdapterBenchmarkProblem,
    *,
    trials: int = 64,
    seed: int = 0,
    policies: Sequence[str] = ("adaptive", "fixed", "prior_ranked"),
) -> AdapterBenchmarkResult:
    """Run paired prior-predictive trials under common truths and noise draws.

    ``fixed`` cycles through the declared laboratory schedule.
    ``prior_ranked`` freezes the initial adaptive ranking before any data are
    observed. ``adaptive`` recomputes the ranking after every observation.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    policy_values = tuple(str(policy) for policy in policies)
    allowed = {"adaptive", "fixed", "prior_ranked"}
    if not policy_values or any(policy not in allowed for policy in policy_values):
        raise ValueError(f"policies must be drawn from {sorted(allowed)}")
    ranked_schedule, ranked_seconds = _prior_ranked_schedule(problem, seed)
    truth_rng = np.random.default_rng(seed)
    truth_indices = truth_rng.choice(
        problem.initial_posterior.particle_count,
        size=trials,
        p=problem.initial_posterior.weights,
    )
    noise_rng = np.random.default_rng(seed + 1)
    noise_seeds = noise_rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=(trials, problem.maximum_actions),
        dtype=np.int64,
    )
    results: list[BenchmarkTrial] = []
    schedules = {
        "fixed": problem.fixed_schedule,
        "prior_ranked": ranked_schedule,
    }
    for trial_index, truth_index in enumerate(truth_indices):
        for policy_index, policy in enumerate(policy_values):
            schedule = schedules.get(policy, problem.fixed_schedule)
            results.append(
                _run_trial(
                    problem,
                    policy=policy,
                    schedule=schedule,
                    trial_index=trial_index,
                    truth_index=int(truth_index),
                    noise_seeds=noise_seeds[trial_index],
                    policy_seed=seed + 10_000 * trial_index + policy_index,
                )
            )
    summaries = tuple(_summarize(policy, results) for policy in policy_values)
    return AdapterBenchmarkResult(
        name=problem.name,
        target_unit=problem.target_unit,
        prediction_table_seconds=problem.prediction_table.build_seconds,
        open_loop_schedule_seconds=ranked_seconds,
        policies=policy_values,
        trials=tuple(results),
        summaries=summaries,
    )


def save_benchmark_results(
    path: str | Path, results: Sequence[AdapterBenchmarkResult]
) -> None:
    """Write benchmark results as strict, human-readable JSON."""

    payload = {"version": 1, "results": [result.to_dict() for result in results]}
    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
