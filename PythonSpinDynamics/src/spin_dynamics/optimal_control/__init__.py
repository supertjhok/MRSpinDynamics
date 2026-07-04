"""GRAPE (gradient ascent pulse engineering) quantum optimal control.

Optimizes the full piecewise-constant RF control waveform (amplitude and
phase per segment) of an arbitrary pulse against a state-transfer or
propagator/gate fidelity target, via reverse-mode autodiff through a
differentiable Hilbert-space propagator chain. Phase-only control (a fixed
nutation rate, only phase optimized) is the primary mode, matching hardware
that uses switching power amplifiers with no continuously variable amplitude;
joint amplitude+phase control is supported as a generalization.

``hamiltonians.py`` is plain NumPy and has no extra dependency. Everything
else here requires the optional ``jax`` extra (reverse-mode autodiff is the
entire point -- finite-differencing a multi-segment waveform is far more
expensive). Install with ``python -m pip install -e .[jax]`` (or ``.[perf]``).
"""

from spin_dynamics.optimal_control.drivers import GrapeMultiStartResult, run_grape_multistart
from spin_dynamics.optimal_control.hamiltonians import (
    ControlHamiltonianModel,
    coupled_spin_control_model,
)
from spin_dynamics.optimal_control.objectives import (
    average_gate_fidelity,
    bloch_vector_to_ket,
    make_grape_objective,
    robust_ensemble_fidelity,
    state_transfer_fidelity,
    su2_effective_axis,
)
from spin_dynamics.optimal_control.parameterization import (
    ControlBounds,
    amplitude_phase_bounds,
    export_to_pulse_shape,
    phase_only_bounds,
    random_phase_starts,
    rectangular_seed_phase,
)
from spin_dynamics.optimal_control.solvers import (
    GrapeOptimizationResult,
    grape_optimize,
    grape_optimize_phase_only,
)

__all__ = [
    "ControlBounds",
    "ControlHamiltonianModel",
    "GrapeMultiStartResult",
    "GrapeOptimizationResult",
    "amplitude_phase_bounds",
    "average_gate_fidelity",
    "bloch_vector_to_ket",
    "coupled_spin_control_model",
    "export_to_pulse_shape",
    "grape_optimize",
    "grape_optimize_phase_only",
    "make_grape_objective",
    "phase_only_bounds",
    "random_phase_starts",
    "rectangular_seed_phase",
    "robust_ensemble_fidelity",
    "run_grape_multistart",
    "state_transfer_fidelity",
    "su2_effective_axis",
]
