"""Reference benchmark problems for the four Phase 2 experiment adapters."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from spin_dynamics.design.adapters import (
    CPMGIRAdapter,
    CPMGIRDesign,
    ESRDelayDesign,
    ESRHahnAdapter,
    ExperimentAdapterCost,
    ExperimentPlanConstraint,
    ExperimentPredictor,
    NQRFIDAdapter,
    NQRFrequencyDesign,
    PGSEAdapter,
    PGSEDesign,
)
from spin_dynamics.design.benchmarks import (
    AdapterBenchmarkProblem,
    CandidatePredictionTable,
)
from spin_dynamics.design.diagnostics import CredibleIntervalStopping
from spin_dynamics.design.likelihoods import ComplexGaussianLikelihood
from spin_dynamics.design.posterior import GridPosterior
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.utilities import ExpectedVarianceReduction
from spin_dynamics.esr import ESRSpinSystem, resonance_frequency_hz
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


_COVERAGE_ORDER = (0, 8, 4, 2, 6, 1, 3, 5, 7)


def _target(name: str):
    def quantity(parameters: Mapping[str, np.ndarray]) -> np.ndarray:
        return np.asarray(parameters[name], dtype=np.float64)

    return quantity


def _table_problem(
    *,
    name: str,
    adapter: Any,
    posterior: GridPosterior,
    actions: CandidateDesignSpace,
    likelihood: ComplexGaussianLikelihood,
    target_name: str,
    maximum_width: float,
    target_unit: str,
    utility_scale: float,
    utility_samples: int,
    minimum_actions: int,
    maximum_actions: int,
) -> AdapterBenchmarkProblem:
    predictor = ExperimentPredictor(adapter, cache=False, prefer_batch=True)
    table = CandidatePredictionTable.build(predictor, posterior.parameters, actions)
    quantity = _target(target_name)
    fixed_schedule = tuple(actions.actions[index] for index in _COVERAGE_ORDER)
    return AdapterBenchmarkProblem(
        name=name,
        prediction_table=table,
        likelihood=likelihood,
        initial_posterior=posterior,
        utility=ExpectedVarianceReduction(
            quantity, samples=utility_samples, scale=utility_scale
        ),
        cost=ExperimentAdapterCost(adapter),
        quantity=quantity,
        stopping_rule=CredibleIntervalStopping(
            quantity, maximum_width=maximum_width, probability=0.9
        ),
        fixed_schedule=fixed_schedule,
        target_unit=target_unit,
        minimum_actions=minimum_actions,
        maximum_actions=maximum_actions,
        credible_probability=0.9,
        constraints=(ExperimentPlanConstraint(adapter),),
    )


def _counts(profile: str) -> tuple[int, int]:
    if profile == "reference":
        return 15, 3
    if profile == "smoke":
        return 5, 2
    raise ValueError("profile must be 'reference' or 'smoke'")


def make_cpmg_ir_benchmark(
    *, profile: str = "reference", utility_samples: int = 24
) -> AdapterBenchmarkProblem:
    """Benchmark adaptive inversion-delay selection for T1 estimation."""

    target_count, nuisance_count = _counts(profile)
    template = Experiment(
        CPMGIRTrain(num_echoes=2, echo_spacing_seconds=1e-3),
        Sample(t1_seconds=0.15, t2_seconds=0.06),
        acquisition=Acquisition(numpts=21, maxoffs=5.0, rephase_action="ignore"),
    )
    adapter = CPMGIRAdapter(
        template,
        {"t1_seconds": 0.15, "t2_seconds": 0.06, "signal_scale": 1.0},
        echo_index=0,
        recovery_seconds=30e-3,
        fixed_overhead_seconds=2e-3,
    )
    posterior = GridPosterior(
        {
            "t1_seconds": np.geomspace(0.05, 0.5, target_count),
            "t2_seconds": np.geomspace(0.045, 0.08, nuisance_count),
            "signal_scale": np.array([0.9, 1.1]),
        }
    )
    actions = CandidateDesignSpace(
        CPMGIRDesign(float(delay)) for delay in np.geomspace(5e-3, 0.8, 9)
    )
    return _table_problem(
        name="CPMG-IR T1",
        adapter=adapter,
        posterior=posterior,
        actions=actions,
        likelihood=ComplexGaussianLikelihood(1.5),
        target_name="t1_seconds",
        maximum_width=0.07,
        target_unit="s",
        utility_scale=0.1,
        utility_samples=utility_samples,
        minimum_actions=2,
        maximum_actions=8,
    )


def make_pgse_benchmark(
    *, profile: str = "reference", utility_samples: int = 24
) -> AdapterBenchmarkProblem:
    """Benchmark adaptive gradient selection for diffusion estimation."""

    target_count, nuisance_count = _counts(profile)
    template = Experiment(
        PGSE(num_echoes=2, diffusion_time=20e-3),
        Sample(t2_seconds=0.1, diffusion_coefficient=1.0e-9),
    )
    adapter = PGSEAdapter(
        template,
        {"diffusion_coefficient": 1.0e-9, "t2_seconds": 0.1},
        echo_index=0,
        recovery_seconds=20e-3,
        fixed_overhead_seconds=2e-3,
    )
    posterior = GridPosterior(
        {
            "diffusion_coefficient": np.geomspace(0.4e-9, 2.2e-9, target_count),
            "t2_seconds": np.geomspace(0.06, 0.16, nuisance_count),
        }
    )
    actions = CandidateDesignSpace(
        PGSEDesign(float(amplitude), 2e-3, 20e-3)
        for amplitude in np.linspace(0.0, 0.12, 9)
    )
    return _table_problem(
        name="PGSE diffusion",
        adapter=adapter,
        posterior=posterior,
        actions=actions,
        likelihood=ComplexGaussianLikelihood(0.02),
        target_name="diffusion_coefficient",
        maximum_width=0.5e-9,
        target_unit="m^2/s",
        utility_scale=1e-9,
        utility_samples=utility_samples,
        minimum_actions=2,
        maximum_actions=8,
    )


def make_nqr_fid_benchmark(
    *, profile: str = "reference", utility_samples: int = 24
) -> AdapterBenchmarkProblem:
    """Benchmark adaptive NQR carrier selection for site-frequency estimation."""

    target_count, nuisance_count = _counts(profile)
    site = QuadrupolarSite(
        spin=1.0, quadrupole_frequency_hz=900e3, eta=0.3
    )
    template = Experiment(
        NQRFID(10e3, 25e-6, 80e-6, num_points=12),
        Sample(site=site),
    )
    adapter = NQRFIDAdapter(
        template,
        {"quadrupole_frequency_hz": 900e3, "eta": 0.3},
        sample_index=0,
        recovery_seconds=2e-3,
        fixed_overhead_seconds=0.2e-3,
    )
    posterior = GridPosterior(
        {
            "quadrupole_frequency_hz": np.linspace(820e3, 980e3, target_count),
            "eta": np.linspace(0.1, 0.5, nuisance_count),
        }
    )
    actions = CandidateDesignSpace(
        NQRFrequencyDesign(float(frequency))
        for frequency in np.linspace(760e3, 1.04e6, 9)
    )
    return _table_problem(
        name="NQR site frequency",
        adapter=adapter,
        posterior=posterior,
        actions=actions,
        likelihood=ComplexGaussianLikelihood(0.04),
        target_name="quadrupole_frequency_hz",
        maximum_width=25e3,
        target_unit="Hz",
        utility_scale=100e3,
        utility_samples=utility_samples,
        minimum_actions=2,
        maximum_actions=8,
    )


def make_esr_hahn_benchmark(
    *, profile: str = "reference", utility_samples: int = 24
) -> AdapterBenchmarkProblem:
    """Benchmark adaptive Hahn-delay selection for electron-spin T2."""

    target_count, nuisance_count = _counts(profile)
    system = ESRSpinSystem()
    b0 = 0.35
    carrier = resonance_frequency_hz(system, (0.0, 0.0, b0))
    template = Experiment(
        ESRHahnEcho(25e6, 10e-9, 200e-9, num_points=16),
        Sample(esr_system=system),
        Hardware(b0=UniformB0(field_tesla=b0)),
    )
    adapter = ESRHahnAdapter(
        template,
        {"t2_seconds": 2e-6, "g_factor": 2.00231930436256},
        sample_index=8,
        recovery_seconds=10e-6,
        fixed_overhead_seconds=1e-6,
    )
    posterior = GridPosterior(
        {
            "t2_seconds": np.geomspace(0.5e-6, 6e-6, target_count),
            "g_factor": np.linspace(1.995, 2.01, nuisance_count),
        }
    )
    actions = CandidateDesignSpace(
        ESRDelayDesign(float(tau), rf_frequency_hz=carrier)
        for tau in np.geomspace(50e-9, 5e-6, 9)
    )
    return _table_problem(
        name="ESR Hahn T2",
        adapter=adapter,
        posterior=posterior,
        actions=actions,
        likelihood=ComplexGaussianLikelihood(0.03),
        target_name="t2_seconds",
        maximum_width=0.7e-6,
        target_unit="s",
        utility_scale=1e-6,
        utility_samples=utility_samples,
        minimum_actions=2,
        maximum_actions=8,
    )


def make_phase2_adapter_benchmarks(
    *, profile: str = "reference", utility_samples: int = 24
) -> tuple[AdapterBenchmarkProblem, ...]:
    """Build benchmark problems for every Phase 2 adapter."""

    return (
        make_cpmg_ir_benchmark(profile=profile, utility_samples=utility_samples),
        make_pgse_benchmark(profile=profile, utility_samples=utility_samples),
        make_nqr_fid_benchmark(profile=profile, utility_samples=utility_samples),
        make_esr_hahn_benchmark(profile=profile, utility_samples=utility_samples),
    )
