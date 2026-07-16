"""End-to-end benchmarks for every Phase 2 Bayesian adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from spin_dynamics.design import (
    CallableCost,
    CandidateDesignSpace,
    ComplexGaussianLikelihood,
    CredibleIntervalStopping,
    ExpectedVarianceReduction,
    ParticlePosterior,
)
from spin_dynamics.design.adapter_benchmarks import (
    make_phase2_adapter_benchmarks,
)
from spin_dynamics.design.benchmarks import (
    AdapterBenchmarkProblem,
    CandidatePredictionTable,
    run_adapter_benchmark,
    save_benchmark_results,
)


@dataclass(frozen=True)
class _Action:
    x: float


def _quantity(parameters):
    return parameters["theta"]


def _small_problem() -> AdapterBenchmarkProblem:
    parameters = {
        "theta": np.array([-1.0, 0.0, 1.0]),
        "nuisance": np.array([0.0, 0.0, 0.0]),
    }
    posterior = ParticlePosterior(parameters)
    actions = CandidateDesignSpace([_Action(0.2), _Action(1.0)])

    def predictor(values, design):
        return values["theta"] * design.x + values["nuisance"]

    table = CandidatePredictionTable.build(predictor, parameters, actions)
    return AdapterBenchmarkProblem(
        name="linear",
        prediction_table=table,
        likelihood=ComplexGaussianLikelihood(0.1),
        initial_posterior=posterior,
        utility=ExpectedVarianceReduction(_quantity, samples=4),
        cost=CallableCost(lambda design: 1.0 + design.x),
        quantity=_quantity,
        stopping_rule=CredibleIntervalStopping(_quantity, 0.1, probability=0.9),
        fixed_schedule=actions.actions,
        target_unit="a.u.",
        minimum_actions=1,
        maximum_actions=3,
        credible_probability=0.9,
    )


def test_prediction_table_is_exact_and_particle_order_guarded() -> None:
    problem = _small_problem()
    table = problem.prediction_table

    assert np.array_equal(
        table(problem.initial_posterior.parameters, _Action(1.0)),
        np.array([-1.0, 0.0, 1.0]),
    )
    changed = problem.initial_posterior.parameters
    changed["theta"] = changed["theta"][::-1]
    with pytest.raises(ValueError, match="original particle"):
        table(changed, _Action(1.0))


def test_paired_benchmark_is_reproducible_except_wall_clock() -> None:
    problem = _small_problem()
    first = run_adapter_benchmark(problem, trials=6, seed=8)
    second = run_adapter_benchmark(problem, trials=6, seed=8)

    first_science = [
        (
            trial.policy,
            trial.trial_index,
            trial.truth_particle_index,
            trial.reached_target,
            trial.physical_seconds,
            trial.posterior_mean,
            trial.interval_width,
            trial.covered,
            tuple(point.design_index for point in trial.trace),
        )
        for trial in first.trials
    ]
    second_science = [
        (
            trial.policy,
            trial.trial_index,
            trial.truth_particle_index,
            trial.reached_target,
            trial.physical_seconds,
            trial.posterior_mean,
            trial.interval_width,
            trial.covered,
            tuple(point.design_index for point in trial.trace),
        )
        for trial in second.trials
    ]
    assert first_science == second_science
    for trial_index in range(6):
        truths = {
            trial.truth_particle_index
            for trial in first.trials
            if trial.trial_index == trial_index
        }
        assert len(truths) == 1


def test_benchmark_json_is_standard_and_complete(tmp_path) -> None:
    result = run_adapter_benchmark(_small_problem(), trials=3, seed=2)
    path = tmp_path / "benchmark.json"

    save_benchmark_results(path, [result])
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["version"] == 1
    assert payload["results"][0]["name"] == "linear"
    assert len(payload["results"][0]["summaries"]) == 3
    assert len(payload["results"][0]["trials"]) == 9


@pytest.mark.smoke
def test_every_phase2_adapter_has_a_runnable_benchmark() -> None:
    problems = make_phase2_adapter_benchmarks(profile="smoke", utility_samples=4)

    assert [problem.name for problem in problems] == [
        "CPMG-IR T1",
        "PGSE diffusion",
        "NQR site frequency",
        "ESR Hahn T2",
    ]
    for problem in problems:
        result = run_adapter_benchmark(problem, trials=2, seed=3)
        assert result.prediction_table_seconds >= 0.0
        assert len(result.summaries) == 3
        assert all(summary.trials == 2 for summary in result.summaries)
        assert all(0.0 <= summary.coverage_fraction <= 1.0 for summary in result.summaries)
        assert all(np.isfinite(summary.root_mean_square_error) for summary in result.summaries)
