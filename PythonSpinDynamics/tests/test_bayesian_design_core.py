"""Generic Phase 1 Bayesian experiment-design tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from spin_dynamics.design import (
    AdaptiveDesignSession,
    CallableConstraint,
    CallableCost,
    CandidateDesignSpace,
    ComplexGaussianLikelihood,
    CredibleIntervalStopping,
    DiscretePrior,
    ExpectedInformationGain,
    ExpectedVarianceReduction,
    GaussianLikelihood,
    GridPosterior,
    IndependentPrior,
    LogUniformPrior,
    NormalPrior,
    ParticlePosterior,
    PosteriorStandardDeviationStopping,
    PredictiveModel,
    UniformPrior,
    load_design_state,
    save_design_state,
    summarize_quantity,
)


@dataclass(frozen=True)
class LinearDesign:
    x: float
    duration_seconds: float = 1.0


def _linear_model(sigma: float = 0.2) -> PredictiveModel:
    def predict(parameters, design):
        return parameters["slope"] * design.x + parameters.get("offset", 0.0)

    return PredictiveModel(predict, GaussianLikelihood(sigma))


def test_scalar_and_independent_priors_sample_and_score() -> None:
    rng = np.random.default_rng(12)
    prior = IndependentPrior(
        {
            "normal": NormalPrior(1.0, 0.5),
            "uniform": UniformPrior(-2.0, 3.0),
            "log_uniform": LogUniformPrior(0.1, 10.0),
            "discrete": DiscretePrior(np.array([2.0, 4.0]), np.array([0.25, 0.75])),
        }
    )
    samples = prior.sample(200, rng)
    log_probability = prior.log_prob(samples)

    assert set(samples) == set(prior.components)
    assert all(values.shape == (200,) for values in samples.values())
    assert np.all(np.isfinite(log_probability))
    assert np.all((samples["uniform"] >= -2.0) & (samples["uniform"] <= 3.0))
    assert np.all((samples["log_uniform"] >= 0.1) & (samples["log_uniform"] <= 10.0))


def test_particle_update_and_systematic_resampling_are_normalized() -> None:
    posterior = ParticlePosterior({"theta": np.array([-1.0, 0.0, 1.0])})
    updated = posterior.updated(np.log(np.array([0.1, 0.2, 0.7])))
    resampled = updated.resampled(np.random.default_rng(4))

    assert np.sum(updated.weights) == pytest.approx(1.0)
    assert updated.weights == pytest.approx([0.1, 0.2, 0.7])
    assert resampled.parameters["theta"].shape == (3,)
    assert resampled.weights == pytest.approx(np.full(3, 1.0 / 3.0))


def test_grid_posterior_preserves_axes_and_exact_marginals() -> None:
    grid = GridPosterior(
        {"a": np.array([0.0, 1.0]), "b": np.array([10.0, 20.0, 30.0])},
        weights=np.array([[0.1, 0.1, 0.2], [0.05, 0.15, 0.4]]),
    )
    updated = grid.updated(np.log(np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])))

    assert isinstance(updated, GridPosterior)
    assert grid.marginal("a") == pytest.approx([0.4, 0.6])
    assert grid.marginal("b") == pytest.approx([0.15, 0.25, 0.6])
    assert updated.marginal("a")[1] > grid.marginal("a")[1]


def test_real_and_complex_gaussian_likelihood_event_shapes() -> None:
    real = GaussianLikelihood(np.array([0.1, 0.2]), event_ndim=1)
    prediction = np.array([[1.0, 2.0], [1.5, 2.5]])
    log_probability = real.log_prob(np.array([1.0, 2.0]), prediction)
    assert log_probability.shape == (2,)
    assert log_probability[0] > log_probability[1]

    complex_likelihood = ComplexGaussianLikelihood(0.1, event_ndim=1)
    complex_prediction = prediction.astype(np.complex128) * (1.0 + 1.0j)
    complex_probability = complex_likelihood.log_prob(
        complex_prediction[0], complex_prediction
    )
    draw = complex_likelihood.sample(
        complex_prediction[0], np.random.default_rng(8)
    )
    assert complex_probability.shape == (2,)
    assert complex_probability[0] > complex_probability[1]
    assert np.iscomplexobj(draw)


def test_predictive_model_requires_vectorized_particle_output() -> None:
    bad = PredictiveModel(lambda parameters, design: np.array(1.0), GaussianLikelihood(1.0))
    with pytest.raises(ValueError, match="particle count"):
        bad.predict({"theta": np.array([1.0, 2.0])}, None)


def test_eig_distinguishes_uninformative_and_informative_designs() -> None:
    posterior = GridPosterior({"slope": np.array([-1.0, 1.0])})
    utility = ExpectedInformationGain(samples=1000)
    model = _linear_model(sigma=0.15)

    zero = utility.estimate(
        model, posterior, LinearDesign(0.0), np.random.default_rng(5)
    )
    informative = utility.estimate(
        model, posterior, LinearDesign(1.0), np.random.default_rng(5)
    )

    assert zero.value == pytest.approx(0.0, abs=1e-12)
    assert informative.value > 0.65
    assert informative.standard_error >= 0.0
    assert informative.units == "nats"


def test_expected_variance_reduction_targets_selected_quantity() -> None:
    posterior = GridPosterior(
        {
            "slope": np.array([-1.0, 1.0]),
            "offset": np.array([-2.0, 2.0]),
        }
    )
    utility = ExpectedVarianceReduction(
        lambda parameters: parameters["slope"], samples=800
    )
    model = _linear_model(sigma=0.2)
    no_slope_information = utility.estimate(
        model, posterior, LinearDesign(0.0), np.random.default_rng(7)
    )
    slope_information = utility.estimate(
        model, posterior, LinearDesign(2.0), np.random.default_rng(7)
    )

    assert no_slope_information.value == pytest.approx(0.0, abs=1e-10)
    assert slope_information.value > 0.5
    assert slope_information.units == "scaled variance"


def test_session_enforces_constraints_and_ranks_utility_per_second() -> None:
    candidates = CandidateDesignSpace(
        (
            LinearDesign(1.0, 2.0),
            LinearDesign(1.0, 1.0),
            LinearDesign(5.0, 1.0),
        )
    )
    session = AdaptiveDesignSession(
        model=_linear_model(),
        posterior=GridPosterior({"slope": np.array([-1.0, 1.0])}),
        design_space=candidates,
        utility=ExpectedInformationGain(samples=256),
        cost=CallableCost(lambda design: design.duration_seconds),
        constraints=(
            CallableConstraint(lambda design: abs(design.x) <= 2.0, "x exceeds limit"),
        ),
        seed=19,
    )
    recommendation = session.ask()

    assert recommendation.best.design == LinearDesign(1.0, 1.0)
    rejected = next(score for score in recommendation.scores if not score.feasible)
    assert rejected.messages == ("x exceeds limit",)
    assert recommendation.scores[0].utility_rate >= recommendation.scores[1].utility_rate


def test_tell_updates_posterior_and_stopping_diagnostics() -> None:
    def quantity(parameters):
        return parameters["slope"]

    candidates = CandidateDesignSpace((LinearDesign(1.0),))
    session = AdaptiveDesignSession(
        model=_linear_model(sigma=0.05),
        posterior=GridPosterior({"slope": np.array([-1.0, 1.0])}),
        design_space=candidates,
        utility=ExpectedInformationGain(samples=32),
        cost=CallableCost(lambda design: design.duration_seconds),
        stopping_rule=PosteriorStandardDeviationStopping(quantity, 0.1),
        seed=3,
    )
    assert not session.should_stop()
    session.tell(candidates.actions[0], 1.02)
    summary = summarize_quantity(session.posterior, quantity, probability=0.9)

    assert session.should_stop()
    assert summary.mean[0] > 0.99
    assert summary.standard_deviation[0] < 0.1
    assert session.elapsed_seconds == pytest.approx(1.0)


def test_credible_interval_stopping_handles_discrete_qoi() -> None:
    posterior = ParticlePosterior(
        {"theta": np.array([0.0, 1.0, 2.0])},
        np.log(np.array([0.01, 0.98, 0.01])),
    )
    rule = CredibleIntervalStopping(
        lambda parameters: parameters["theta"], maximum_width=0.0, probability=0.95
    )
    assert rule.reached(posterior)


def test_session_checkpoint_restores_rng_and_next_recommendation(tmp_path) -> None:
    candidates = CandidateDesignSpace((LinearDesign(0.5), LinearDesign(1.5)))
    dependencies = {
        "model": _linear_model(),
        "design_space": candidates,
        "utility": ExpectedInformationGain(samples=64),
        "cost": CallableCost(lambda design: design.duration_seconds),
    }
    session = AdaptiveDesignSession(
        posterior=ParticlePosterior(
            {"slope": np.linspace(-2.0, 2.0, 101)}
        ),
        seed=42,
        **dependencies,
    )
    first = session.ask()
    session.tell(first.best.design, 0.75)
    path = tmp_path / "design-state.json"
    save_design_state(path, session.state_dict())
    state = load_design_state(path)
    restored = AdaptiveDesignSession.from_state(state, **dependencies)

    original_next = session.ask()
    restored_next = restored.ask()
    assert restored.history[0].design_index == session.history[0].design_index
    assert np.array_equal(restored.posterior.log_weights, session.posterior.log_weights)
    assert restored_next.evaluation_seed == original_next.evaluation_seed
    assert restored_next.best.design == original_next.best.design


def test_session_json_preserves_complex_observations() -> None:
    candidates = CandidateDesignSpace((LinearDesign(1.0),))
    model = PredictiveModel(
        lambda parameters, design: parameters["signal"],
        ComplexGaussianLikelihood(0.1),
    )
    dependencies = {
        "model": model,
        "design_space": candidates,
        "utility": ExpectedInformationGain(samples=8),
        "cost": CallableCost(lambda design: design.duration_seconds),
    }
    session = AdaptiveDesignSession(
        posterior=ParticlePosterior({"signal": np.array([1 + 1j, 2 + 2j])}),
        seed=2,
        **dependencies,
    )
    session.tell(candidates.actions[0], 1.2 + 0.8j)
    restored = AdaptiveDesignSession.from_json(session.to_json(), **dependencies)

    assert restored.history[0].observation == pytest.approx(1.2 + 0.8j)
