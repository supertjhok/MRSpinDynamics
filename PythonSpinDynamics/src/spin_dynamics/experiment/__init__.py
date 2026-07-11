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

from spin_dynamics.experiment.config import (
    ConfigError,
    experiment_from_config,
    experiment_to_config,
    load_config,
    save_config,
)
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
from spin_dynamics.experiment.io import (
    LoadedRun,
    ReproductionReport,
    load_run,
    register_result_type,
)
from spin_dynamics.experiment.plan import ExperimentPlan, plan_experiment
from spin_dynamics.experiment.registry import (
    WorkflowEntry,
    available_workflows,
    register_workflow,
)
from spin_dynamics.experiment.provenance import (
    experiment_fingerprint,
    result_fingerprint,
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
    DEERDistribution,
    ESRCWSweep,
    ESRDEER,
    ESRDaviesENDOR,
    ESRFID,
    ESRHYSCORE,
    ESRHahnEcho,
    ESRMimsENDOR,
    ESRThreePulseESEEM,
    ESRTwoPulseESEEM,
    Experiment,
    Hardware,
    NQRFID,
    NQRPopulationTransfer,
    NQRSLSE,
    NQRSORC,
    PGSE,
    PGSEWalkers,
    Phantom,
    Sample,
    SampledB0,
    SequenceDomain,
    SequenceIRExecution,
    TransportDomain2D,
    UniformFlow2D,
)
from spin_dynamics.experiment.wiring import (
    sampled_b0_from_solution,
    solve_imaging_field_maps,
)

from spin_dynamics.experiment import _catalog  # noqa: F401  (registers workflows)

__all__ = [
    "Acquisition",
    "CPMG",
    "CPMGIRTrain",
    "CPMGImaging",
    "CPMGTrain",
    "ConfigError",
    "CostModel",
    "DEFAULT_RULES",
    "DEERDistribution",
    "ESRCWSweep",
    "ESRDEER",
    "ESRDaviesENDOR",
    "ESRFID",
    "ESRHYSCORE",
    "ESRHahnEcho",
    "ESRMimsENDOR",
    "ESRThreePulseESEEM",
    "ESRTwoPulseESEEM",
    "Experiment",
    "ExperimentPlan",
    "ExperimentPlanError",
    "Hardware",
    "ImagingPlane",
    "LoadedRun",
    "ReproductionReport",
    "NQRSLSE",
    "NQRSORC",
    "NQRFID",
    "NQRPopulationTransfer",
    "PGSE",
    "PGSEWalkers",
    "Phantom",
    "PlanarSpiralCoil",
    "RuleFinding",
    "RunRecord",
    "RuntimeEstimate",
    "RxCoil",
    "Sample",
    "SampledB0",
    "SequenceDomain",
    "SequenceIRExecution",
    "TransportDomain2D",
    "UniformFlow2D",
    "sampled_b0_from_solution",
    "SerializationError",
    "SolenoidCoil",
    "TxCoil",
    "UniformB0",
    "WorkflowEntry",
    "available_workflows",
    "calibrate",
    "experiment_from_config",
    "experiment_fingerprint",
    "experiment_to_config",
    "load_config",
    "load_run",
    "plan_experiment",
    "save_config",
    "set_calibration",
    "register_result_type",
    "register_serializable",
    "register_workflow",
    "result_fingerprint",
    "run_experiment",
    "run_rules",
    "solve_imaging_field_maps",
]
