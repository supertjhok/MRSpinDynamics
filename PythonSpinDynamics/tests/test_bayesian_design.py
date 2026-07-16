"""Phase 0 Bayesian experiment-design reference tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spin_dynamics.design import (
    IRAcquisitionCost,
    IRAdaptiveSession,
    IRDesign,
    IRGridPosterior,
    IRGridPrior,
    IRMeasurement,
    IRTruth,
    inversion_recovery_signal,
    recommend_ir_design,
    run_ir_reference_trial,
)


def _small_prior() -> IRGridPrior:
    return IRGridPrior(
        t1_seconds=np.array([0.08, 0.16, 0.32]),
        amplitude=np.array([0.9, 1.1]),
        baseline=np.array([-0.03, 0.03]),
        sigma=np.array([0.04, 0.08]),
    )


def test_inversion_recovery_signal_endpoints_and_validation() -> None:
    assert inversion_recovery_signal(0.0, 0.2) == pytest.approx(-1.0)
    assert inversion_recovery_signal(100.0, 0.2) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="delay_seconds"):
        inversion_recovery_signal(-1.0, 0.2)
    with pytest.raises(ValueError, match="t1_seconds"):
        inversion_recovery_signal(0.1, 0.0)


def test_grid_update_matches_direct_discrete_bayes_rule() -> None:
    prior = _small_prior()
    posterior = IRGridPosterior(prior)
    design = IRDesign(delay_seconds=0.12, repetitions=4)
    observed = 0.17
    updated = posterior.updated(IRMeasurement(design, observed))

    t1, amplitude, baseline, sigma = np.meshgrid(
        prior.t1_seconds,
        prior.amplitude,
        prior.baseline,
        prior.sigma,
        indexing="ij",
    )
    mean = inversion_recovery_signal(0.12, t1, amplitude, baseline)
    standard_deviation = sigma / 2.0
    likelihood = np.exp(
        -0.5 * ((observed - mean) / standard_deviation) ** 2
    ) / (np.sqrt(2.0 * np.pi) * standard_deviation)
    expected = np.asarray(prior.weights) * likelihood
    expected /= np.sum(expected)

    assert np.allclose(updated.weights.reshape(prior.shape), expected)
    assert np.sum(updated.t1_marginal()) == pytest.approx(1.0)


def test_goal_eig_is_zero_for_known_t1_and_positive_when_uncertain() -> None:
    design = IRDesign(delay_seconds=0.1, repetitions=2)
    known = IRGridPosterior(
        IRGridPrior(
            t1_seconds=np.array([0.2]),
            amplitude=np.array([0.8, 1.2]),
            baseline=np.array([-0.1, 0.1]),
            sigma=np.array([0.04, 0.08]),
        )
    )
    uncertain = IRGridPosterior(_small_prior())

    assert known.t1_expected_information_gain(design, quadrature_order=10) == pytest.approx(
        0.0, abs=1e-12
    )
    assert uncertain.t1_expected_information_gain(
        design, quadrature_order=10
    ) > 0.0


def test_goal_eig_matches_dense_predictive_integration() -> None:
    posterior = IRGridPosterior(
        IRGridPrior(
            t1_seconds=np.array([0.1, 0.3]),
            amplitude=np.array([1.0]),
            baseline=np.array([0.0]),
            sigma=np.array([0.1]),
            weights=np.array([[[[0.35]]], [[[0.65]]]]),
        )
    )
    design = IRDesign(delay_seconds=0.12)
    quadrature = posterior.t1_expected_information_gain(
        design, quadrature_order=64
    )

    means = inversion_recovery_signal(
        design.delay_seconds, posterior.prior.t1_seconds
    )
    axis = np.linspace(float(np.min(means) - 0.8), float(np.max(means) + 0.8), 50001)
    conditional = np.exp(-0.5 * ((axis[:, None] - means[None, :]) / 0.1) ** 2)
    conditional /= np.sqrt(2.0 * np.pi) * 0.1
    probabilities = posterior.t1_marginal()
    predictive = conditional @ probabilities
    integrand = conditional * np.log(conditional / predictive[:, None])
    density = integrand @ probabilities
    dense = float(np.sum(0.5 * (density[1:] + density[:-1]) * np.diff(axis)))

    assert quadrature == pytest.approx(dense, rel=2e-6, abs=1e-9)


def test_repetitions_increase_information_and_cost() -> None:
    posterior = IRGridPosterior(_small_prior())
    single = IRDesign(0.12, repetitions=1)
    averaged = IRDesign(0.12, repetitions=4)
    cost = IRAcquisitionCost(
        readout_seconds=0.01,
        recovery_seconds=0.2,
        fixed_overhead_seconds=0.05,
    )

    single_information = posterior.t1_expected_information_gain(
        single, quadrature_order=10
    )
    averaged_information = posterior.t1_expected_information_gain(
        averaged, quadrature_order=10
    )
    assert averaged_information >= single_information
    assert cost.seconds(averaged) > cost.seconds(single)


def test_recommendation_ranks_information_rate_deterministically() -> None:
    posterior = IRGridPosterior(_small_prior())
    designs = tuple(IRDesign(delay) for delay in (0.02, 0.08, 0.2, 0.5))
    cost = IRAcquisitionCost(readout_seconds=0.01, recovery_seconds=0.1)

    first = recommend_ir_design(
        posterior, designs, cost, quadrature_order=10, chunk_size=17
    )
    second = recommend_ir_design(
        posterior, designs, cost, quadrature_order=10, chunk_size=256
    )

    assert first.best.design == second.best.design
    assert first.best == first.scores[0]
    assert all(
        left.information_rate_nats_per_second
        >= right.information_rate_nats_per_second
        for left, right in zip(first.scores, first.scores[1:])
    )
    assert np.allclose(
        [item.expected_information_nats for item in first.scores],
        [item.expected_information_nats for item in second.scores],
    )


def test_session_json_round_trip_replays_posterior_and_recommendation() -> None:
    designs = tuple(IRDesign(delay) for delay in (0.02, 0.08, 0.2, 0.5))
    cost = IRAcquisitionCost(readout_seconds=0.01, recovery_seconds=0.1)
    session = IRAdaptiveSession(
        IRGridPosterior(_small_prior()),
        designs,
        cost,
        quadrature_order=9,
        chunk_size=23,
    )
    chosen = session.ask().best.design
    session.tell(chosen, 0.125)

    payload = session.to_json()
    restored = IRAdaptiveSession.from_json(payload)

    assert json.loads(payload)["quadrature_order"] == 9
    assert restored.history == session.history
    assert restored.elapsed_seconds == pytest.approx(session.elapsed_seconds)
    assert np.array_equal(restored.posterior.log_weights, session.posterior.log_weights)
    assert restored.ask().best.design == session.ask().best.design


def test_discrete_posterior_intervals_are_prior_predictive_calibrated() -> None:
    prior = IRGridPrior(
        t1_seconds=np.geomspace(0.05, 0.8, 17),
        amplitude=np.array([0.9, 1.0, 1.1]),
        baseline=np.array([-0.03, 0.0, 0.03]),
        sigma=np.array([0.04, 0.06]),
    )
    designs = tuple(IRDesign(delay) for delay in (0.03, 0.1, 0.3, 0.8))
    rng = np.random.default_rng(123)
    covered = 0
    trials = 200

    for _ in range(trials):
        truth = IRTruth(
            t1_seconds=float(rng.choice(prior.t1_seconds)),
            amplitude=float(rng.choice(prior.amplitude)),
            baseline=float(rng.choice(prior.baseline)),
            sigma=float(rng.choice(prior.sigma)),
        )
        posterior = IRGridPosterior(prior)
        for design in designs:
            value = float(
                rng.normal(
                    inversion_recovery_signal(
                        design.delay_seconds,
                        truth.t1_seconds,
                        truth.amplitude,
                        truth.baseline,
                    ),
                    truth.sigma,
                )
            )
            posterior = posterior.updated(IRMeasurement(design, value))
        low, high = posterior.t1_credible_interval(0.9)
        covered += int(low <= truth.t1_seconds <= high)

    # Equal-tailed intervals on a finite grid are conservative. The fixed-seed
    # prior-predictive check guards against under-coverage from a bad update.
    assert covered / trials >= 0.9


@pytest.mark.slow
def test_adaptive_reference_reaches_target_faster_than_fixed_long_delay_schedule() -> None:
    prior = IRGridPrior(
        t1_seconds=np.geomspace(0.06, 0.6, 13),
        amplitude=np.array([0.9, 1.0, 1.1]),
        baseline=np.array([-0.03, 0.0, 0.03]),
        sigma=np.array([0.05]),
    )
    candidates = tuple(IRDesign(delay) for delay in np.geomspace(0.02, 1.2, 9))
    fixed = tuple(reversed(candidates))
    cost = IRAcquisitionCost(readout_seconds=0.01, recovery_seconds=0.08)
    truths = [
        IRTruth(t1_seconds=float(t1), amplitude=1.0, sigma=0.05)
        for t1 in prior.t1_seconds[2:-2:2]
    ]

    adaptive_times: list[float] = []
    fixed_times: list[float] = []
    for seed, truth in enumerate(truths):
        adaptive = run_ir_reference_trial(
            prior=prior,
            truth=truth,
            candidate_designs=candidates,
            cost=cost,
            rng=np.random.default_rng(seed),
            policy="adaptive",
            maximum_actions=10,
            maximum_relative_width=0.35,
            quadrature_order=8,
        )
        baseline = run_ir_reference_trial(
            prior=prior,
            truth=truth,
            candidate_designs=candidates,
            fixed_schedule=fixed,
            cost=cost,
            rng=np.random.default_rng(seed),
            policy="fixed",
            maximum_actions=10,
            maximum_relative_width=0.35,
            quadrature_order=8,
        )
        adaptive_times.append(adaptive.elapsed_seconds)
        fixed_times.append(baseline.elapsed_seconds)

    assert np.median(adaptive_times) < np.median(fixed_times)
