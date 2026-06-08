"""High-level workflow entry points mirroring canonical MATLAB examples."""

from spin_dynamics.workflows.acquisition import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
)
from spin_dynamics.workflows.cpmg import (
    CPMGResult,
    CPMGTrainResult,
    run_ideal_cpmg,
    run_ideal_cpmg_train,
    run_matched_cpmg,
    run_matched_cpmg_train,
    run_tuned_cpmg,
    run_tuned_cpmg_train,
    run_untuned_cpmg,
    run_untuned_cpmg_train,
)
from spin_dynamics.workflows.sweeps import (
    CPMGParameterSweepResult,
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)

__all__ = [
    "CPMGResult",
    "CPMGParameterSweepResult",
    "CPMGTrainResult",
    "calc_macq_ideal_probe_relax4",
    "calc_macq_matched_probe_relax4",
    "calc_macq_tuned_probe_relax4",
    "calc_macq_untuned_probe_relax4",
    "run_ideal_cpmg",
    "run_ideal_cpmg_train",
    "run_matched_cpmg",
    "run_matched_cpmg_train",
    "run_matched_mistuning_sweep",
    "run_matched_q_sweep",
    "run_tuned_cpmg",
    "run_tuned_cpmg_train",
    "run_tuned_mistuning_sweep",
    "run_tuned_q_sweep",
    "run_untuned_cpmg",
    "run_untuned_cpmg_train",
]
