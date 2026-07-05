"""Registry entries for the CPMG workflow family (PR-1 scope).

Each entry wraps one validated ``spin_dynamics.workflows`` function. Kwarg
builders only forward spec fields the workflow honors, and only when the
user set them, so a default ``Experiment`` reproduces the direct workflow
call bit for bit.
"""

from __future__ import annotations

from typing import Any

from spin_dynamics.experiment.io import register_result_type
from spin_dynamics.experiment.registry import WorkflowEntry, register_workflow
from spin_dynamics.experiment.serialization import register_serializable
from spin_dynamics.experiment.specs import CPMG, CPMGIRTrain, CPMGTrain, Experiment
from spin_dynamics.noise import NoiseMetadata, NoiseSpec
from spin_dynamics.phase_cycling import PhaseCycle, PhaseStep
from spin_dynamics.workflows import (
    run_ideal_cpmg,
    run_ideal_cpmg_ir_train,
    run_ideal_cpmg_train,
    run_matched_cpmg,
    run_matched_cpmg_ir_train,
    run_matched_cpmg_train,
    run_tuned_cpmg,
    run_tuned_cpmg_ir_train,
    run_tuned_cpmg_train,
    run_untuned_cpmg,
    run_untuned_cpmg_ir_train,
    run_untuned_cpmg_train,
)
from spin_dynamics.workflows.cpmg import CPMGResult, CPMGTrainResult
from spin_dynamics.workflows.cpmg_ir import CPMGIRTrainResult, MatchedCPMGIRTrainResult

register_result_type(CPMGResult)
register_result_type(CPMGTrainResult)
register_result_type(CPMGIRTrainResult)
register_result_type(MatchedCPMGIRTrainResult)

register_serializable(NoiseSpec)
register_serializable(NoiseMetadata)
register_serializable(PhaseStep)
register_serializable(PhaseCycle)

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
        )
    )
