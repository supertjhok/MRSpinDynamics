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

from spin_dynamics.experiment.io import LoadedRun, load_run, register_result_type
from spin_dynamics.experiment.plan import ExperimentPlan, plan_experiment
from spin_dynamics.experiment.registry import (
    WorkflowEntry,
    available_workflows,
    register_workflow,
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
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    Hardware,
    Sample,
)

from spin_dynamics.experiment import _catalog  # noqa: F401  (registers workflows)

__all__ = [
    "Acquisition",
    "CPMG",
    "CPMGIRTrain",
    "CPMGTrain",
    "Experiment",
    "ExperimentPlan",
    "ExperimentPlanError",
    "Hardware",
    "LoadedRun",
    "RunRecord",
    "Sample",
    "SerializationError",
    "WorkflowEntry",
    "available_workflows",
    "load_run",
    "plan_experiment",
    "register_result_type",
    "register_serializable",
    "register_workflow",
    "run_experiment",
]
