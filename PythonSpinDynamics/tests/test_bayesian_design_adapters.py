"""Phase 2 experiment-facade adapter tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spin_dynamics.design import (
    CPMGIRAdapter,
    CPMGIRDesign,
    ESRDelayDesign,
    ESRHahnAdapter,
    ExpectedInformationGain,
    ExperimentAdapterCost,
    ExperimentPlanConstraint,
    ExperimentPredictor,
    ComplexGaussianLikelihood,
    NQRFIDAdapter,
    NQRFrequencyDesign,
    PGSEAdapter,
    PGSEDesign,
    ParticlePosterior,
    CandidateDesignSpace,
    make_adapter_session,
)
from spin_dynamics.esr import ESRSpinSystem
from spin_dynamics.experiment import (
    Acquisition,
    CPMGIRTrain,
    ESRHahnEcho,
    Experiment,
    Hardware,
    NQRFID,
    PGSE,
    Sample,
    UniformB0,
)
from spin_dynamics.nqr import QuadrupolarSite


def _cpmg_adapter(**kwargs) -> CPMGIRAdapter:
    template = Experiment(
        sequence=CPMGIRTrain(num_echoes=2, echo_spacing_seconds=1e-3),
        sample=Sample(t1_seconds=0.1, t2_seconds=0.05),
        acquisition=Acquisition(numpts=21, maxoffs=5.0, rephase_action="ignore"),
    )
    return CPMGIRAdapter(
        template,
        {"t1_seconds": 0.1, "t2_seconds": 0.05},
        **kwargs,
    )


@pytest.mark.smoke
def test_cpmg_ir_adapter_binds_parameters_extracts_and_costs() -> None:
    adapter = _cpmg_adapter(echo_index=0, recovery_seconds=0.02)
    design = CPMGIRDesign(0.01)
    experiment = adapter.build_experiment(
        {"t1_seconds": 0.2, "t2_seconds": 0.08}, design
    )

    assert experiment.sequence.tauvect == (0.01,)
    assert experiment.sample.t1_seconds == pytest.approx(0.2)
    assert experiment.sample.t2_seconds == pytest.approx(0.08)
    assert adapter.plan(design).ok
    assert adapter.physical_seconds(design) == pytest.approx(0.032)
    assert np.asarray(adapter.simulate(adapter.nominal_parameters, design)).shape == ()


@pytest.mark.smoke
def test_pgse_adapter_vectorizes_real_workflow_and_caches() -> None:
    adapter = PGSEAdapter(
        Experiment(
            PGSE(num_echoes=2),
            sample=Sample(t2_seconds=0.1, diffusion_coefficient=1e-9),
        ),
        {"diffusion_coefficient": 1e-9, "t2_seconds": 0.1},
    )
    action = PGSEDesign(0.05, 2e-3, 20e-3)
    predictor = ExperimentPredictor(adapter)
    particles = {
        "diffusion_coefficient": np.array([0.8e-9, 1.2e-9]),
        "t2_seconds": np.array([0.08, 0.12]),
    }

    first = predictor(particles, action)
    cache_size = len(predictor._cache)
    second = predictor(particles, action)

    assert first.shape == (2, 2)
    assert np.array_equal(first, second)
    assert len(predictor._cache) == cache_size == 2
    assert adapter.physical_seconds(action) == pytest.approx(0.08)


@pytest.mark.smoke
def test_nqr_frequency_adapter_binds_site_and_pulse() -> None:
    site = QuadrupolarSite(
        spin=1.0, quadrupole_frequency_hz=900e3, eta=0.3
    )
    adapter = NQRFIDAdapter(
        Experiment(
            NQRFID(10e3, 25e-6, 50e-6, num_points=8),
            sample=Sample(site=site),
        ),
        {"quadrupole_frequency_hz": 900e3, "eta": 0.3},
    )
    action = NQRFrequencyDesign(910e3, pulse_duration_seconds=20e-6)
    experiment = adapter.build_experiment(
        {"quadrupole_frequency_hz": 905e3, "eta": 0.2}, action
    )

    assert experiment.sequence.rf_frequency_hz == pytest.approx(910e3)
    assert experiment.sequence.pulse_duration_seconds == pytest.approx(20e-6)
    assert experiment.sample.site.quadrupole_frequency_hz == pytest.approx(905e3)
    assert experiment.sample.site.eta == pytest.approx(0.2)
    assert adapter.plan(action).ok
    assert adapter.simulate(adapter.nominal_parameters, action).shape == (8,)


@pytest.mark.smoke
def test_esr_hahn_adapter_binds_t2_g_factor_and_delay() -> None:
    adapter = ESRHahnAdapter(
        Experiment(
            ESRHahnEcho(25e6, 10e-9, 200e-9, num_points=8),
            sample=Sample(esr_system=ESRSpinSystem()),
            hardware=Hardware(b0=UniformB0(field_tesla=0.35)),
        ),
        {"t2_seconds": 1e-6, "g_factor": 2.0023},
        sample_index=3,
    )
    action = ESRDelayDesign(300e-9)
    experiment = adapter.build_experiment(
        {"t2_seconds": 2e-6, "g_factor": 2.01}, action
    )

    assert experiment.sequence.tau_seconds == pytest.approx(300e-9)
    assert experiment.sequence.t2_seconds == pytest.approx(2e-6)
    assert experiment.sample.esr_system.g_tensor[0, 0] == pytest.approx(2.01)
    assert adapter.simulate(adapter.nominal_parameters, action).shape == ()
    assert adapter.physical_seconds(action) == pytest.approx(930e-9)


def test_planner_constraint_rejects_invalid_nominal_experiment() -> None:
    adapter = _cpmg_adapter()
    invalid = replace(
        adapter,
        template=replace(adapter.template, hardware=Hardware(probe="not-a-probe")),
    )
    result = ExperimentPlanConstraint(invalid).evaluate(CPMGIRDesign(0.01))

    assert not result.feasible
    assert result.message


def test_adapter_session_mandates_planner_constraint_and_ranks() -> None:
    adapter = _cpmg_adapter(echo_index=0)
    posterior = ParticlePosterior(
        {
            "t1_seconds": np.array([0.06, 0.1, 0.18]),
            "t2_seconds": np.full(3, 0.05),
        }
    )
    session = make_adapter_session(
        adapter=adapter,
        likelihood=ComplexGaussianLikelihood(0.2),
        posterior=posterior,
        design_space=CandidateDesignSpace(
            [CPMGIRDesign(0.01), CPMGIRDesign(0.05)]
        ),
        utility=ExpectedInformationGain(samples=4),
        seed=4,
    )

    recommendation = session.ask()

    assert isinstance(session.constraints[0], ExperimentPlanConstraint)
    assert isinstance(session.cost, ExperimentAdapterCost)
    assert recommendation.best.feasible
    assert recommendation.best.cost_seconds > 0.0


def test_action_and_template_validation() -> None:
    with pytest.raises(ValueError, match="at least"):
        PGSEDesign(0.05, 5e-3, 2e-3)
    with pytest.raises(ValueError, match="acquisition.noise"):
        CPMGIRAdapter(
            replace(
                _cpmg_adapter().template,
                acquisition=replace(_cpmg_adapter().template.acquisition, noise=0.1),
            ),
            {"t1_seconds": 0.1, "t2_seconds": 0.05},
        )
