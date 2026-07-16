"""Phase 3 performance and robustness tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from spin_dynamics.design import (
    AdaptiveMonteCarloUtility,
    CallableCost,
    CandidateDesignSpace,
    ExpectedInformationGain,
    ExpectedTargetInformationGain,
    GaussianDiscrepancyLikelihood,
    GaussianLikelihood,
    LaplaceInformationGain,
    ModelMixturePredictor,
    PGSEAdapter,
    PGSEDesign,
    ParticlePosterior,
    PolynomialSurrogatePredictor,
    PredictiveModel,
    TwoStageDesignSession,
    analyze_design_sensitivity,
)
from spin_dynamics.design.adapters import ExperimentPredictor
from spin_dynamics.experiment import Experiment, PGSE, Sample


@dataclass(frozen=True)
class _Action:
    x: float


def _linear_predict(parameters, design):
    return parameters["slope"] * design.x + parameters["offset"]


@pytest.mark.smoke
def test_pgse_batch_matches_particle_facade_runs() -> None:
    adapter = PGSEAdapter(
        Experiment(
            PGSE(num_echoes=3),
            Sample(t2_seconds=0.1, diffusion_coefficient=1e-9),
        ),
        {"diffusion_coefficient": 1e-9, "t2_seconds": 0.1},
    )
    particles = {
        "diffusion_coefficient": np.array([0.6e-9, 1.0e-9, 1.6e-9]),
        "t2_seconds": np.array([0.07, 0.1, 0.14]),
        "signal_scale": np.array([0.9, 1.0, 1.1]),
        "baseline": np.array([-0.01, 0.0, 0.01]),
    }
    action = PGSEDesign(0.06, 2e-3, 18e-3)

    batched = ExperimentPredictor(adapter, cache=False, prefer_batch=True)(
        particles, action
    )
    facade = ExperimentPredictor(adapter, cache=False, prefer_batch=False)(
        particles, action
    )

    assert np.allclose(batched, facade, rtol=2e-15, atol=2e-15)

    fixed_t2_particles = {
        "diffusion_coefficient": particles["diffusion_coefficient"]
    }
    assert np.allclose(
        ExperimentPredictor(adapter, cache=False)(fixed_t2_particles, action),
        ExperimentPredictor(adapter, cache=False, prefer_batch=False)(
            fixed_t2_particles, action
        ),
    )


def test_polynomial_surrogate_fits_and_validates_quadratic_response() -> None:
    parameters = {"theta": np.linspace(-2.0, 2.0, 9)}
    designs = tuple(_Action(x) for x in (-1.0, 0.0, 1.0))

    def exact(values, design):
        theta = values["theta"]
        return theta**2 + 2.0 * theta * design.x + 0.5 * design.x**2

    surrogate = PolynomialSurrogatePredictor.fit(
        exact,
        parameters,
        designs,
        lambda design: np.array([design.x]),
        degree=2,
    )
    validation = surrogate.validate(
        exact, {"theta": np.array([-1.5, -0.25, 0.7, 1.8])}, [_Action(0.4)]
    )

    assert validation.normalized_root_mean_square_error < 1e-9
    assert validation.correlation == pytest.approx(1.0)


def test_laplace_screening_and_adaptive_monte_carlo() -> None:
    posterior = ParticlePosterior(
        {
            "slope": np.array([-1.0, 0.0, 1.0]),
            "offset": np.zeros(3),
        }
    )
    model = PredictiveModel(_linear_predict, GaussianLikelihood(0.2))
    rng = np.random.default_rng(3)

    zero = LaplaceInformationGain().estimate(model, posterior, _Action(0.0), rng)
    informative = LaplaceInformationGain().estimate(
        model, posterior, _Action(1.0), rng
    )
    adaptive = AdaptiveMonteCarloUtility(
        minimum_samples=4,
        maximum_samples=16,
        absolute_tolerance=10.0,
    ).estimate(model, posterior, _Action(1.0), rng)

    assert zero.value == pytest.approx(0.0)
    assert informative.value > 0.0
    assert adaptive.samples.size == 4


def test_two_stage_session_refines_only_leaders() -> None:
    posterior = ParticlePosterior(
        {
            "slope": np.array([-1.0, 0.0, 1.0]),
            "offset": np.zeros(3),
        }
    )
    session = TwoStageDesignSession(
        model=PredictiveModel(_linear_predict, GaussianLikelihood(0.2)),
        posterior=posterior,
        design_space=CandidateDesignSpace(_Action(x) for x in (0.0, 0.5, 1.0)),
        screening_utility=LaplaceInformationGain(),
        refinement_utility=ExpectedInformationGain(8),
        finalists=2,
        cost=CallableCost(lambda design: 1.0 + abs(design.x)),
        seed=5,
    )

    recommendation = session.ask()

    assert recommendation.best.design in (_Action(0.5), _Action(1.0))
    assert sum(score.utility.samples.size == 8 for score in recommendation.scores) == 2


def test_discrepancy_adds_variances() -> None:
    likelihood = GaussianDiscrepancyLikelihood(3.0, 4.0)

    assert likelihood.sigma == pytest.approx(5.0)
    log_probability = likelihood.log_prob(0.0, np.array([0.0, 1.0]))
    assert log_probability.shape == (2,)


def test_model_mixture_dispatches_particle_subsets() -> None:
    predictor = ModelMixturePredictor(
        (
            lambda p, d: p["theta"] + d.x,
            lambda p, d: 2.0 * p["theta"] - d.x,
        )
    )
    values = predictor(
        {
            "model_index": np.array([0, 1, 0, 1]),
            "theta": np.array([1.0, 1.0, 2.0, 2.0]),
        },
        _Action(0.5),
    )

    assert np.array_equal(values, np.array([1.5, 1.5, 2.5, 3.5]))


def test_target_information_marginalizes_nuisance_particles() -> None:
    posterior = ParticlePosterior(
        {
            "target": np.array([0, 0, 1, 1]),
            "nuisance": np.array([-2.0, 2.0, -2.0, 2.0]),
        }
    )

    def nuisance_only(parameters, design):
        del design
        return parameters["nuisance"]

    model = PredictiveModel(nuisance_only, GaussianLikelihood(0.1))
    estimate = ExpectedTargetInformationGain(
        lambda parameters: parameters["target"], samples=32
    ).estimate(model, posterior, _Action(1.0), np.random.default_rng(8))

    assert estimate.value == pytest.approx(0.0, abs=1e-12)


def test_prior_model_sensitivity_reports_agreement() -> None:
    model = PredictiveModel(_linear_predict, GaussianLikelihood(0.3))
    narrow = ParticlePosterior(
        {"slope": np.array([-0.5, 0.0, 0.5]), "offset": np.zeros(3)}
    )
    wide = ParticlePosterior(
        {"slope": np.array([-2.0, 0.0, 2.0]), "offset": np.zeros(3)}
    )
    report = analyze_design_sensitivity(
        scenarios={"narrow": (model, narrow), "wide": (model, wide)},
        design_space=CandidateDesignSpace([_Action(0.0), _Action(1.0)]),
        utility=LaplaceInformationGain(),
        seed=2,
    )

    assert report.modal_design_index == 1
    assert report.agreement_fraction == pytest.approx(1.0)
