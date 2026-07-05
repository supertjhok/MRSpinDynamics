"""Unified declarative experiment facade over the validated workflows.

Describe an experiment with frozen dataclass specs, front-load validation
with ``plan()``, execute via the existing ``run_*`` workflows with
``run()``, and persist results with provenance::

    from spin_dynamics.experiment import Experiment, CPMGTrain, Hardware

    study = Experiment(sequence=CPMGTrain(num_echoes=8),
                       hardware=Hardware(probe="tuned"))
    print(study.plan().report())
    record = study.run()
    record.save("run1.npz")

Design and milestones: ``docs/unified_workflow_plan.md``.
"""

from spin_dynamics.experiment.estimate import (
    CostModel,
    RuntimeEstimate,
    calibrate,
    set_calibration,
)
from spin_dynamics.experiment.hardware import (
    ImagingPlane,
    PlanarSpiralCoil,
    RxCoil,
    SolenoidCoil,
    TxCoil,
    UniformB0,
)
from spin_dynamics.experiment.io import LoadedRun, load_run, register_result_type
from spin_dynamics.experiment.plan import ExperimentPlan, plan_experiment
from spin_dynamics.experiment.registry import (
    WorkflowEntry,
    available_workflows,
    register_workflow,
)
from spin_dynamics.experiment.rules import (
    DEFAULT_RULES,
    RuleFinding,
    run_rules,
)
from spin_dynamics.experiment.runner import (
    ExperimentPlanError,
    RunRecord,
    run_experiment,
)
from spin_dynamics.experiment.serialization import (
    SerializationError,
    register_serializable,
)
from spin_dynamics.experiment.specs import (
    CPMG,
    Acquisition,
    CPMGImaging,
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    Hardware,
    Phantom,
    Sample,
)
from spin_dynamics.experiment.wiring import solve_imaging_field_maps

from spin_dynamics.experiment import _catalog  # noqa: F401  (registers workflows)

__all__ = [
    "Acquisition",
    "CPMG",
    "CPMGIRTrain",
    "CPMGImaging",
    "CPMGTrain",
    "CostModel",
    "DEFAULT_RULES",
    "Experiment",
    "ExperimentPlan",
    "ExperimentPlanError",
    "Hardware",
    "ImagingPlane",
    "LoadedRun",
    "Phantom",
    "PlanarSpiralCoil",
    "RuleFinding",
    "RunRecord",
    "RuntimeEstimate",
    "RxCoil",
    "Sample",
    "SerializationError",
    "SolenoidCoil",
    "TxCoil",
    "UniformB0",
    "WorkflowEntry",
    "available_workflows",
    "calibrate",
    "load_run",
    "plan_experiment",
    "set_calibration",
    "register_result_type",
    "register_serializable",
    "register_workflow",
    "run_experiment",
    "run_rules",
    "solve_imaging_field_maps",
]
