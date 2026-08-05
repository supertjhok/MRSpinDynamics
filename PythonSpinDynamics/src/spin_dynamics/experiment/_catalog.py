"""Registry entries for the CPMG workflow family (PR-1 scope).

Each entry wraps one validated ``spin_dynamics.workflows`` function. Kwarg
builders only forward spec fields the workflow honors, and only when the
user set them, so a default ``Experiment`` reproduces the direct workflow
call bit for bit.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from spin_dynamics.experiment.estimate import CostModel
from spin_dynamics.experiment.io import register_result_type
from spin_dynamics.experiment.hardware import RxArray
from spin_dynamics.experiment.registry import WorkflowEntry, register_workflow
from spin_dynamics.experiment.serialization import register_serializable
from spin_dynamics.esr import ESRSpinSystem, ESRTransition
from spin_dynamics.esr.pulsed import ESRFIDResult, ESRHahnEchoResult
from spin_dynamics.esr.systems import ESREigensystem
from spin_dynamics.experiment import (
    esr_adapter,
    esr_multidim_adapter,
    nano_mr_adapter,
    nqr_adapter,
    sequence_adapter,
)
from spin_dynamics.experiment.specs import (
    CPMG,
    CPMGImaging,
    CPMGIRTrain,
    CPMGTrain,
    ESRFID,
    ESRHahnEcho,
    ESRCWSweep,
    ESRDEER,
    ESRDaviesENDOR,
    ESRHYSCORE,
    ESRMimsENDOR,
    ESRThreePulseESEEM,
    ESRTwoPulseESEEM,
    Experiment,
    NQRSLSE,
    NQRSORC,
    NQRFID,
    NQRPopulationTransfer,
    NanoMRQdyne,
    NanoMRStatisticalSpectrum,
    PGSE,
    PGSEWalkers,
    SequenceIRExecution,
)
from spin_dynamics.experiment.wiring import (
    solve_for_experiment,
    solve_receive_sensitivities,
)
from spin_dynamics.nqr import QuadrupolarSite
from spin_dynamics.nqr.full_dynamics import FullNQRFIDResult, FullNQRSLSEResult
from spin_dynamics.nqr.simulation import PopulationTransferResult, SLSEResult, SORCResult
from spin_dynamics.esr.spectra import ESRFieldSweepResult
from spin_dynamics.esr.eseem import HyperfineCoupling
from spin_dynamics.esr.endor import EndorSpectrum
from spin_dynamics.nqr.systems import NQREigensystem, NQRTransition
from spin_dynamics.noise import NoiseMetadata, NoiseSpec
from spin_dynamics.sequences import (
    ADCEvent,
    GradientWaveform,
    HardwareEffectsPolicy,
    RFPulse,
    SequenceBlock,
    SequenceIR,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.phase_cycling import PhaseCycle, PhaseStep
from spin_dynamics.workflows import (
    run_ideal_cpmg,
    run_ideal_cpmg_imaging,
    run_ideal_receiver_array_cpmg_imaging,
    run_ideal_cpmg_ir_train,
    run_ideal_cpmg_train,
    run_matched_cpmg,
    run_matched_cpmg_imaging,
    run_matched_cpmg_ir_train,
    run_matched_cpmg_train,
    run_pgse_moment,
    run_pgse_walkers,
    run_tuned_cpmg,
    run_tuned_cpmg_imaging,
    run_tuned_cpmg_ir_train,
    run_tuned_cpmg_train,
    run_untuned_cpmg,
    run_untuned_cpmg_ir_train,
    run_untuned_cpmg_train,
)
from spin_dynamics.workflows.cpmg import (
    CPMGResult,
    CPMGTrainResult,
    ideal_cpmg_train_max_time,
    probe_cpmg_train_max_time,
)
from spin_dynamics.workflows.cpmg_ir import (
    CPMGIRTrainResult,
    MatchedCPMGIRTrainResult,
    cpmg_ir_train_max_time,
    default_ir_tauvect,
)
from spin_dynamics.workflows.imaging_types import (
    IdealCPMGImagingResult,
    ProbeCPMGImagingResult,
    ReceiverArrayCPMGImagingResult,
)
from spin_dynamics.workflows.pgse import PGSEMomentResult, PGSEWalkerResult
from spin_dynamics.nano_mr import (
    OpticalReadoutResult,
    QdyneResult,
    StatisticalSpectrumResult,
)

register_result_type(CPMGResult)
register_result_type(CPMGTrainResult)
register_result_type(CPMGIRTrainResult)
register_result_type(MatchedCPMGIRTrainResult)
register_result_type(IdealCPMGImagingResult)
register_result_type(ProbeCPMGImagingResult)
register_result_type(ReceiverArrayCPMGImagingResult)
register_result_type(SLSEResult)
register_result_type(SORCResult)
register_result_type(FullNQRSLSEResult)
register_result_type(FullNQRFIDResult)
register_result_type(PopulationTransferResult)
register_result_type(ESRFIDResult)
register_result_type(ESRHahnEchoResult)
register_result_type(ESRFieldSweepResult)
register_result_type(esr_adapter.ESRDEERResult)
register_result_type(esr_multidim_adapter.ESRESEEMResult)
register_result_type(esr_multidim_adapter.ESRHYSCOREResult)
register_result_type(EndorSpectrum)
register_result_type(PGSEMomentResult)
register_result_type(PGSEWalkerResult)
register_result_type(sequence_adapter.SequenceIRResult)
register_result_type(StatisticalSpectrumResult)
register_result_type(QdyneResult)

register_serializable(NoiseSpec)
register_serializable(NoiseMetadata)
register_serializable(PhaseStep)
register_serializable(PhaseCycle)
register_serializable(QuadrupolarSite)
register_serializable(NQRTransition)
register_serializable(NQREigensystem)
register_serializable(ESRSpinSystem)
register_serializable(ESRTransition)
register_serializable(ESREigensystem)
register_serializable(HyperfineCoupling)
register_serializable(HardwareEffectsPolicy)
register_serializable(RFPulse)
register_serializable(GradientWaveform)
register_serializable(ADCEvent)
register_serializable(SequenceBlock)
register_serializable(SequenceIR)
register_serializable(OpticalReadoutResult)

_ACQ_GRID = frozenset({"acquisition.numpts", "acquisition.maxoffs"})
_ACQ_REPHASE = frozenset(
    {
        "acquisition.auto_refine_grid",
        "acquisition.rephase_safety_factor",
        "acquisition.rephase_action",
    }
)
_SAMPLE_RELAX = frozenset({"sample.t1_seconds", "sample.t2_seconds"})
_PROBE_CIRCUIT = frozenset({"hardware.q_value", "hardware.mistuning_offset"})


def _cpmg_kwargs(experiment: Experiment) -> dict[str, Any]:
    acq = experiment.acquisition
    kwargs: dict[str, Any] = {"numpts": acq.numpts, "maxoffs": acq.maxoffs}
    if acq.noise is not None:
        kwargs["noise"] = acq.noise
    return kwargs


def _train_kwargs(experiment: Experiment) -> dict[str, Any]:
    acq = experiment.acquisition
    sample = experiment.sample
    kwargs = _cpmg_kwargs(experiment)
    kwargs.update(
        num_echoes=experiment.sequence.num_echoes,
        auto_refine_grid=acq.auto_refine_grid,
        rephase_safety_factor=acq.rephase_safety_factor,
        rephase_action=acq.rephase_action,
    )
    if sample.t1_seconds is not None:
        kwargs["t1_seconds"] = sample.t1_seconds
    if sample.t2_seconds is not None:
        kwargs["t2_seconds"] = sample.t2_seconds
    if experiment.hardware.absolute_phase is not None:
        kwargs["absolute_phase"] = experiment.hardware.absolute_phase
    return kwargs


def _probe_train_kwargs(
    experiment: Experiment, *, radiation_damping: bool
) -> dict[str, Any]:
    hardware = experiment.hardware
    kwargs = _train_kwargs(experiment)
    if hardware.q_value is not None:
        kwargs["q_value"] = hardware.q_value
    if hardware.mistuning_offset is not None:
        kwargs["mistuning_offset"] = hardware.mistuning_offset
    if radiation_damping and hardware.radiation_damping is not None:
        kwargs["radiation_damping"] = hardware.radiation_damping
    return kwargs


def _ir_train_kwargs(experiment: Experiment) -> dict[str, Any]:
    acq = experiment.acquisition
    sample = experiment.sample
    sequence = experiment.sequence
    kwargs: dict[str, Any] = {
        "num_echoes": sequence.num_echoes,
        "echo_spacing_seconds": sequence.echo_spacing_seconds,
        "numpts": acq.numpts,
        "maxoffs": acq.maxoffs,
        "auto_refine_grid": acq.auto_refine_grid,
        "rephase_safety_factor": acq.rephase_safety_factor,
        "rephase_action": acq.rephase_action,
    }
    if sequence.tauvect is not None:
        kwargs["tauvect"] = sequence.tauvect
    if sample.t1_seconds is not None:
        kwargs["t1_seconds"] = sample.t1_seconds
    if sample.t2_seconds is not None:
        kwargs["t2_seconds"] = sample.t2_seconds
    return kwargs


def _ideal_pp(numpts: int) -> Any:
    return set_params_ideal(numpts=numpts)[1]


def _tuned_pp(numpts: int) -> Any:
    return set_params_tuned_orig(numpts=numpts)[2]


def _untuned_pp(numpts: int) -> Any:
    return set_params_untuned_orig(numpts=numpts)[2]


def _matched_pp(numpts: int) -> Any:
    return set_params_matched_orig(numpts=numpts)[1]


_PP_GETTERS = {
    "ideal": _ideal_pp,
    "tuned": _tuned_pp,
    "untuned": _untuned_pp,
    "matched": _matched_pp,
}


def _make_train_max_time(probe: str):
    getter = _PP_GETTERS[probe]
    formula = ideal_cpmg_train_max_time if probe == "ideal" else probe_cpmg_train_max_time

    def _max_time(experiment: Experiment) -> float:
        pp0 = getter(experiment.acquisition.numpts)
        return formula(pp0, experiment.sequence.num_echoes)

    return _max_time


def _make_ir_max_time(probe: str):
    getter = _PP_GETTERS[probe]

    def _max_time(experiment: Experiment) -> float:
        sequence = experiment.sequence
        pp0 = getter(experiment.acquisition.numpts)
        tau = default_ir_tauvect(sequence.tauvect)
        return cpmg_ir_train_max_time(
            pp0, sequence.num_echoes, sequence.echo_spacing_seconds, tau
        )

    return _max_time


def _nacq(pp0: Any) -> int:
    return int(round(float(np.ravel(pp0.tacq)[0]) / float(pp0.tdw))) + 1


_PROBE_COST_NOTE = (
    "probe-circuit setup time is not modeled and can dominate small grids",
)
_PHASE_BRANCHES = 2  # default two-step CPMG/PAP excitation cycle


def _make_train_cost(probe: str):
    getter = _PP_GETTERS[probe]
    notes = () if probe == "ideal" else _PROBE_COST_NOTE

    def _cost(experiment: Experiment) -> CostModel:
        numpts = experiment.acquisition.numpts
        num_echoes = experiment.sequence.num_echoes
        segments = 2 + 3 * num_echoes
        units = float(_PHASE_BRANCHES * numpts * segments)
        nacq = _nacq(getter(numpts))
        # complex128 isochromat phase matrix + spectra/echoes + kernel state
        memory = 16 * numpts * (nacq + num_echoes + 40)
        return CostModel(work_units=units, memory_bytes=memory, notes=notes)

    return _cost


def _make_ir_cost(probe: str):
    getter = _PP_GETTERS[probe]
    notes = () if probe == "ideal" else _PROBE_COST_NOTE

    def _cost(experiment: Experiment) -> CostModel:
        numpts = experiment.acquisition.numpts
        sequence = experiment.sequence
        num_taus = default_ir_tauvect(sequence.tauvect).size
        segments = 4 + 3 * sequence.num_echoes
        units = float(_PHASE_BRANCHES * num_taus * numpts * segments)
        nacq = _nacq(getter(numpts))
        memory = 16 * numpts * (nacq + num_taus * sequence.num_echoes + 40)
        return CostModel(work_units=units, memory_bytes=memory, notes=notes)

    return _cost


_CPMG_HONORS = _ACQ_GRID | {"acquisition.noise"}
_TRAIN_HONORS = (
    _CPMG_HONORS | _ACQ_REPHASE | _SAMPLE_RELAX | {"hardware.absolute_phase"}
)
_IR_HONORS = _ACQ_GRID | _ACQ_REPHASE | _SAMPLE_RELAX

_CPMG_FUNCS = {
    "ideal": run_ideal_cpmg,
    "tuned": run_tuned_cpmg,
    "untuned": run_untuned_cpmg,
    "matched": run_matched_cpmg,
}
for _probe, _func in _CPMG_FUNCS.items():
    register_workflow(
        WorkflowEntry(
            name=_func.__name__,
            sequence_type=CPMG,
            probe=_probe,
            func=_func,
            build_kwargs=_cpmg_kwargs,
            honors=_CPMG_HONORS,
        )
    )

_TRAIN_FUNCS = {
    "ideal": (run_ideal_cpmg_train, frozenset(), False),
    "tuned": (run_tuned_cpmg_train, _PROBE_CIRCUIT | {"hardware.radiation_damping"}, True),
    "untuned": (run_untuned_cpmg_train, _PROBE_CIRCUIT, False),
    "matched": (run_matched_cpmg_train, _PROBE_CIRCUIT | {"hardware.radiation_damping"}, True),
}
for _probe, (_func, _extra, _rad) in _TRAIN_FUNCS.items():
    if _probe == "ideal":
        _builder = _train_kwargs
    else:
        def _builder(experiment: Experiment, *, _rad_flag: bool = _rad) -> dict[str, Any]:
            return _probe_train_kwargs(experiment, radiation_damping=_rad_flag)

    register_workflow(
        WorkflowEntry(
            name=_func.__name__,
            sequence_type=CPMGTrain,
            probe=_probe,
            func=_func,
            build_kwargs=_builder,
            honors=_TRAIN_HONORS | _extra,
            execution_kwargs=frozenset({"num_workers"}),
            max_time=_make_train_max_time(_probe),
            cost=_make_train_cost(_probe),
        )
    )

def _imaging_kwargs(experiment: Experiment) -> dict[str, Any]:
    sequence = experiment.sequence
    kwargs: dict[str, Any] = {
        # The container carries rho/t1/t2 plus any solved coil maps; the
        # workflows accept it in place of the raw rho array.
        "rho": solve_for_experiment(experiment),
        "num_echoes": sequence.num_echoes,
        "echo_spacing_seconds": sequence.echo_spacing_seconds,
        "gradient_duration_seconds": sequence.gradient_duration_seconds,
        "fov": sequence.fov,
        "ny": sequence.ny,
        "maxoffs": sequence.maxoffs,
    }
    if isinstance(experiment.hardware.rx_coil, RxArray):
        sensitivities = solve_receive_sensitivities(
            experiment.sample.phantom, experiment.hardware
        )
        kwargs.update(
            receiver_sensitivities=sensitivities.normalized_complex,
            channel_labels=sensitivities.channel_labels,
            receiver_network=experiment.hardware.rx_coil.network,
            noise_std=experiment.acquisition.receiver_noise_std,
            noise_covariance=experiment.acquisition.receiver_noise_covariance,
            noise_seed=experiment.acquisition.receiver_noise_seed,
            sense_acceleration=experiment.acquisition.sense_acceleration,
            sense_axis=experiment.acquisition.sense_axis,
            sense_offset=experiment.acquisition.sense_offset,
            sense_regularization=experiment.acquisition.sense_regularization,
        )
    elif experiment.acquisition.noise is not None:
        kwargs["noise"] = experiment.acquisition.noise
    return kwargs


_IMAGING_HONORS = frozenset(
    {
        "sample.phantom",
        "sample.t1_seconds",
        "sample.t2_seconds",
        "hardware.b0",
        "hardware.tx_coil",
        "hardware.rx_coil",
        "hardware.plane",
        "acquisition.noise",
        "acquisition.receiver_noise_covariance",
        "acquisition.receiver_noise_seed",
        "acquisition.receiver_noise_std",
        "acquisition.sense_acceleration",
        "acquisition.sense_axis",
        "acquisition.sense_offset",
        "acquisition.sense_regularization",
    }
)

def _resolve_imaging_func(experiment: Experiment) -> Any:
    if (
        isinstance(experiment.hardware.rx_coil, RxArray)
        and experiment.hardware.probe == "ideal"
    ):
        return run_ideal_receiver_array_cpmg_imaging
    return _IMAGING_FUNCS[experiment.hardware.probe]

_IMAGING_FUNCS = {
    "ideal": run_ideal_cpmg_imaging,
    "tuned": run_tuned_cpmg_imaging,
    "matched": run_matched_cpmg_imaging,
}
for _probe, _func in _IMAGING_FUNCS.items():
    register_workflow(
        WorkflowEntry(
            name=_func.__name__,
            sequence_type=CPMGImaging,
            probe=_probe,
            func=_func,
            build_kwargs=_imaging_kwargs,
            honors=_IMAGING_HONORS,
            execution_kwargs=frozenset({"num_workers", "phase_workers"}),
            resolve_func=_resolve_imaging_func,
        )
    )


def _pgse_kwargs(experiment: Experiment) -> dict[str, Any]:
    sequence = experiment.sequence
    sample = experiment.sample
    kwargs: dict[str, Any] = {
        "num_echoes": sequence.num_echoes,
        "gradient_amplitude": sequence.gradient_amplitude,
        "gradient_duration": sequence.gradient_duration,
        "diffusion_time": sequence.diffusion_time,
        "first_echo_time_seconds": sequence.first_echo_time_seconds,
        "echo_spacing_seconds": sequence.echo_spacing_seconds,
        "gamma": sequence.gamma,
    }
    if sample.diffusion_coefficient is not None:
        kwargs["diffusion_coefficient"] = sample.diffusion_coefficient
    if sample.t2_seconds is not None:
        kwargs["t2_seconds"] = sample.t2_seconds
    return kwargs


def _pgse_cost(experiment: Experiment) -> CostModel:
    num_echoes = experiment.sequence.num_echoes
    return CostModel(
        work_units=float(max(1, num_echoes)),
        memory_bytes=256 + 32 * max(1, num_echoes),
        notes=("closed-form moment backend; dispatch dominates small runs",),
    )


register_workflow(
    WorkflowEntry(
        name="run_pgse_moment",
        sequence_type=PGSE,
        probe="ideal",
        func=run_pgse_moment,
        build_kwargs=_pgse_kwargs,
        honors=frozenset({"sample.diffusion_coefficient", "sample.t2_seconds"}),
        cost=_pgse_cost,
    )
)


def _pgse_walkers_kwargs(experiment: Experiment) -> dict[str, Any]:
    sequence = experiment.sequence
    sample = experiment.sample
    domain = sample.transport_domain
    kwargs: dict[str, Any] = {
        "rho": domain.rho,
        "x_axis": domain.x_axis,
        "z_axis": domain.z_axis,
        "num_echoes": sequence.num_echoes,
        "gradient_amplitude": sequence.gradient_amplitude,
        "gradient_duration": sequence.gradient_duration,
        "diffusion_time": sequence.diffusion_time,
        "gamma": sequence.gamma,
        "gradient_axis": sequence.gradient_axis,
        "walkers_per_cell": sequence.walkers_per_cell,
        "seed": sequence.seed,
        "jitter": sequence.jitter,
        "excitation_duration": sequence.excitation_duration,
        "refocusing_duration": sequence.refocusing_duration,
        "echo_spacing_seconds": sequence.echo_spacing_seconds,
        "boundary": sequence.boundary,
        "substeps_per_interval": sequence.substeps_per_interval,
    }
    if sample.diffusion_coefficient is not None:
        kwargs["diffusion_coefficient"] = sample.diffusion_coefficient
    if sample.t1_seconds is not None:
        kwargs["t1_seconds"] = sample.t1_seconds
    if sample.t2_seconds is not None:
        kwargs["t2_seconds"] = sample.t2_seconds
    if sample.flow is not None:
        kwargs["velocity"] = sample.flow.as_array()
    return kwargs


def _pgse_walkers_cost(experiment: Experiment) -> CostModel:
    sequence = experiment.sequence
    domain = experiment.sample.transport_domain
    occupied_cells = int(np.count_nonzero(domain.rho > 0.0))
    particles = occupied_cells * sequence.walkers_per_cell
    intervals = 5 + 3 * max(0, sequence.num_echoes - 1)
    units = float(particles * intervals * sequence.substeps_per_interval)
    memory = particles * 96 + sequence.num_echoes * 32
    return CostModel(
        work_units=units,
        memory_bytes=memory,
        notes=("seeded random-walker transport with explicit 2-D domain",),
    )


register_workflow(
    WorkflowEntry(
        name="run_pgse_walkers",
        sequence_type=PGSEWalkers,
        probe="ideal",
        func=run_pgse_walkers,
        build_kwargs=_pgse_walkers_kwargs,
        honors=frozenset(
            {
                "sample.diffusion_coefficient",
                "sample.t1_seconds",
                "sample.t2_seconds",
                "sample.transport_domain",
                "sample.flow",
            }
        ),
        cost=_pgse_walkers_cost,
    )
)

# NQR entries: no probe circuit is modeled, so they register under the
# default "ideal" probe only. SLSE dispatches reduced-vs-full per spec/site
# via the adapter; SORC has a reduced implementation only.
_NQR_HONORS = frozenset({"sample.site"})
register_workflow(
    WorkflowEntry(
        name="simulate_slse",
        sequence_type=NQRSLSE,
        probe="ideal",
        func=nqr_adapter.simulate_slse,
        build_kwargs=nqr_adapter.slse_kwargs,
        honors=_NQR_HONORS,
        resolve_func=nqr_adapter.resolve_slse_func,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_sorc",
        sequence_type=NQRSORC,
        probe="ideal",
        func=nqr_adapter.simulate_sorc,
        build_kwargs=nqr_adapter.sorc_kwargs,
        honors=_NQR_HONORS,
    )
)


def _sequence_ir_cost(experiment: Experiment) -> CostModel:
    compiled = sequence_adapter.compile_for_experiment(experiment)
    sequence = experiment.sequence
    domain = experiment.sample.sequence_domain
    particles = int(np.count_nonzero(domain.density > 0.0)) * sequence.walkers_per_cell
    intervals = max(1, compiled.durations_seconds.size)
    units = float(particles * intervals * sequence.default_substeps)
    memory = particles * 112 + intervals * 96 + compiled.adc.times_seconds.size * 64
    return CostModel(
        work_units=units,
        memory_bytes=memory,
        notes=("compiled SequenceIR on an explicit moving-isochromat domain",),
    )


register_workflow(
    WorkflowEntry(
        name="run_sequence_ir",
        sequence_type=SequenceIRExecution,
        probe="ideal",
        func=sequence_adapter.run_sequence_ir,
        build_kwargs=sequence_adapter.sequence_kwargs,
        honors=frozenset(
            {
                "sample.t1_seconds",
                "sample.t2_seconds",
                "sample.diffusion_coefficient",
                "sample.sequence_domain",
                "acquisition.noise",
            }
        ),
        cost=_sequence_ir_cost,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_full_fid",
        sequence_type=NQRFID,
        probe="ideal",
        func=nqr_adapter.simulate_full_fid,
        build_kwargs=nqr_adapter.fid_kwargs,
        honors=_NQR_HONORS,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_population_transfer",
        sequence_type=NQRPopulationTransfer,
        probe="ideal",
        func=nqr_adapter.simulate_population_transfer,
        build_kwargs=nqr_adapter.population_transfer_kwargs,
        honors=_NQR_HONORS,
    )
)

# ESR entries: pulsed FID / Hahn echo on a single electron spin; no probe
# circuit is modeled, so they register under the default "ideal" probe. The
# static field (hardware.b0 with field_tesla) sets the Larmor frequency.
_ESR_HONORS = frozenset({"sample.esr_system", "hardware.b0"})
register_workflow(
    WorkflowEntry(
        name="simulate_fid",
        sequence_type=ESRFID,
        probe="ideal",
        func=esr_adapter.simulate_fid,
        build_kwargs=esr_adapter.fid_kwargs,
        honors=_ESR_HONORS,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_hahn_echo",
        sequence_type=ESRHahnEcho,
        probe="ideal",
        func=esr_adapter.simulate_hahn_echo,
        build_kwargs=esr_adapter.hahn_kwargs,
        honors=_ESR_HONORS,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_field_sweep",
        sequence_type=ESRCWSweep,
        probe="ideal",
        func=esr_adapter.simulate_field_sweep,
        build_kwargs=esr_adapter.cw_sweep_kwargs,
        honors=frozenset({"sample.esr_system"}),
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_deer",
        sequence_type=ESRDEER,
        probe="ideal",
        func=esr_adapter.run_deer,
        build_kwargs=esr_adapter.deer_kwargs,
        honors=frozenset({"sample.deer_distribution"}),
    )
)

_HYPERFINE_HONORS = frozenset({"sample.hyperfine_coupling"})


def _eseem_cost(experiment: Experiment) -> CostModel:
    points = experiment.sequence.num_points
    return CostModel(
        work_units=float(points),
        memory_bytes=points * 64,
        notes=("analytic or small density-matrix electron-nuclear model",),
    )


def _hyscore_cost(experiment: Experiment) -> CostModel:
    sequence = experiment.sequence
    points = sequence.num_points1 * sequence.num_points2
    transformed = (
        sequence.zero_fill * sequence.num_points1
        * sequence.zero_fill
        * sequence.num_points2
    )
    return CostModel(
        work_units=float(points),
        memory_bytes=8 * (points + transformed),
        notes=("nested 2-D density-matrix evolution plus zero-filled FFT",),
    )
register_workflow(
    WorkflowEntry(
        name="run_two_pulse_eseem",
        sequence_type=ESRTwoPulseESEEM,
        probe="ideal",
        func=esr_multidim_adapter.run_two_pulse_eseem,
        build_kwargs=esr_multidim_adapter.two_pulse_kwargs,
        honors=_HYPERFINE_HONORS,
        cost=_eseem_cost,
    )
)
register_workflow(
    WorkflowEntry(
        name="run_three_pulse_eseem",
        sequence_type=ESRThreePulseESEEM,
        probe="ideal",
        func=esr_multidim_adapter.run_three_pulse_eseem,
        build_kwargs=esr_multidim_adapter.three_pulse_kwargs,
        honors=_HYPERFINE_HONORS,
        cost=_eseem_cost,
    )
)
register_workflow(
    WorkflowEntry(
        name="run_hyscore",
        sequence_type=ESRHYSCORE,
        probe="ideal",
        func=esr_multidim_adapter.run_hyscore,
        build_kwargs=esr_multidim_adapter.hyscore_kwargs,
        honors=_HYPERFINE_HONORS,
        cost=_hyscore_cost,
    )
)
register_workflow(
    WorkflowEntry(
        name="davies_endor_spectrum",
        sequence_type=ESRDaviesENDOR,
        probe="ideal",
        func=esr_multidim_adapter.davies_endor_spectrum,
        build_kwargs=esr_multidim_adapter.endor_kwargs,
        honors=_HYPERFINE_HONORS,
    )
)
register_workflow(
    WorkflowEntry(
        name="mims_endor_spectrum",
        sequence_type=ESRMimsENDOR,
        probe="ideal",
        func=esr_multidim_adapter.mims_endor_spectrum,
        build_kwargs=esr_multidim_adapter.mims_endor_kwargs,
        honors=_HYPERFINE_HONORS,
    )
)

_IR_FUNCS = {
    "ideal": run_ideal_cpmg_ir_train,
    "tuned": run_tuned_cpmg_ir_train,
    "untuned": run_untuned_cpmg_ir_train,
    "matched": run_matched_cpmg_ir_train,
}
for _probe, _func in _IR_FUNCS.items():
    register_workflow(
        WorkflowEntry(
            name=_func.__name__,
            sequence_type=CPMGIRTrain,
            probe=_probe,
            func=_func,
            build_kwargs=_ir_train_kwargs,
            honors=_IR_HONORS,
            execution_kwargs=frozenset({"num_workers", "tau_workers"}),
            max_time=_make_ir_max_time(_probe),
            cost=_make_ir_cost(_probe),
        )
    )


def _nano_statistical_cost(experiment: Experiment) -> CostModel:
    sequence = experiment.sequence
    components = len(experiment.sample.nano_mr_layer.components)
    points = sequence.num_points
    work = float(components * points)
    arrays = 2 * components * points + 4 * components + points
    return CostModel(
        work_units=work,
        memory_bytes=8 * arrays,
        notes=("analytic planar spectrum; no dense spin propagation",),
    )


def _nano_qdyne_cost(experiment: Experiment) -> CostModel:
    sequence = experiment.sequence
    pulses = sequence.xy_order * sequence.xy_repetitions
    points = sequence.shot_count
    work = float(points * (pulses + np.log2(max(points, 2))))
    arrays = 7 * points + 2 * (points // 2 + 1)
    if experiment.hardware.nano_mr_optical_readout is not None:
        arrays += 3 * points
    return CostModel(
        work_units=work,
        memory_bytes=8 * arrays,
        notes=("clocked analytic Qdyne record plus real FFT",),
    )


_NANO_SENSOR = frozenset({"hardware.nano_mr_sensor"})
_NANO_STATISTICAL_HONORS = _NANO_SENSOR | frozenset(
    {
        "sample.nano_mr_layer",
        "sequence.b0_vector_tesla_lab",
        "sequence.angular_frequency_min_rad_s",
        "sequence.angular_frequency_max_rad_s",
        "sequence.num_points",
    }
)
_NANO_QDYNE_HONORS = _NANO_SENSOR | frozenset(
    {
        "hardware.nano_mr_optical_readout",
        *(f"sequence.{field.name}" for field in dataclasses.fields(NanoMRQdyne)),
    }
)

register_workflow(
    WorkflowEntry(
        name="simulate_statistical_spectrum",
        sequence_type=NanoMRStatisticalSpectrum,
        probe="ideal",
        func=nano_mr_adapter.simulate_statistical_spectrum,
        build_kwargs=nano_mr_adapter.statistical_kwargs,
        honors=_NANO_STATISTICAL_HONORS,
        cost=_nano_statistical_cost,
    )
)
register_workflow(
    WorkflowEntry(
        name="simulate_qdyne",
        sequence_type=NanoMRQdyne,
        probe="ideal",
        func=nano_mr_adapter.simulate_qdyne,
        build_kwargs=nano_mr_adapter.qdyne_kwargs,
        honors=_NANO_QDYNE_HONORS,
        cost=_nano_qdyne_cost,
    )
)
