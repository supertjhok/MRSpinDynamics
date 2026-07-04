"""GRAPE solver entry points.

Wraps :func:`optimal_control.objectives.make_grape_objective` and
``optimization._bounded.scipy_maximize_with_grad`` -- this module does not
reimplement any optimizer logic, only parameterization/objective wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.optimal_control.hamiltonians import ControlHamiltonianModel
from spin_dynamics.optimal_control.objectives import make_grape_objective
from spin_dynamics.optimal_control.parameterization import (
    ControlBounds,
    assemble_control_bounds,
    phase_only_bounds,
)
from spin_dynamics.optimization._bounded import scipy_maximize_with_grad


@dataclass(frozen=True)
class GrapeOptimizationResult:
    """Result of a GRAPE control-waveform optimization."""

    mode: str
    n_segments: int
    dt: np.ndarray
    optimize_amplitude: bool
    bounds: ControlBounds
    initial_controls: np.ndarray
    best_controls: np.ndarray
    best_fidelity: float
    initial_fidelity: float
    history_scores: np.ndarray
    history_controls: tuple[np.ndarray, ...]
    iterations: int
    improved: bool
    optimizer_method: str
    optimizer_success: bool
    optimizer_message: str
    optimize_gradient: bool = False

    @property
    def best_amplitude(self) -> np.ndarray | None:
        """The amplitude channel of ``best_controls``, or ``None`` if it was fixed."""

        if not self.optimize_amplitude:
            return None
        return self.best_controls[: self.n_segments]

    @property
    def best_phase(self) -> np.ndarray:
        """The phase channel of ``best_controls``.

        Layout is ``concat([amplitude?, phase, gradient?])``, so the phase block
        starts after the amplitude block (present only when
        ``optimize_amplitude``) -- not simply the tail, which would be the
        gradient block when one is optimized.
        """

        offset = self.n_segments if self.optimize_amplitude else 0
        return self.best_controls[offset : offset + self.n_segments]

    @property
    def best_gradient(self) -> np.ndarray | None:
        """The gradient channel of ``best_controls``, or ``None`` if it was fixed."""

        if not self.optimize_gradient:
            return None
        offset = (self.n_segments if self.optimize_amplitude else 0) + self.n_segments
        return self.best_controls[offset : offset + self.n_segments]


def grape_optimize(
    model: ControlHamiltonianModel,
    initial_phase: np.ndarray,
    *,
    dt: float | np.ndarray,
    target: np.ndarray,
    psi0: np.ndarray | None = None,
    mode: Literal["state_transfer", "gate"] = "state_transfer",
    optimize_amplitude: bool = False,
    fixed_amplitude: float | np.ndarray | None = None,
    initial_amplitude: np.ndarray | None = None,
    amplitude_max_hz: float | None = None,
    optimize_gradient: bool = False,
    fixed_gradient: float | np.ndarray | None = None,
    initial_gradient: np.ndarray | None = None,
    gradient_max: float | None = None,
    gradient_operator_batch: Sequence[np.ndarray] | None = None,
    phase_bound_rad: float = 4 * np.pi,
    hamiltonian_batch: Sequence[np.ndarray] | None = None,
    ensemble_reduction: Literal["mean", "worst_case"] = "mean",
    phase_smoothness_weight: float = 0.0,
    gradient_smoothness_weight: float = 0.0,
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> GrapeOptimizationResult:
    """Optimize a piecewise-constant RF (and optionally gradient) waveform.

    Phase-only by default (``fixed_amplitude`` required: a hertz nutation rate
    held constant -- the primary mode for switching-power-amplifier hardware
    that cannot vary RF amplitude). Pass ``optimize_amplitude=True`` with
    ``initial_amplitude`` and ``amplitude_max_hz`` (the peak-B1 hardware
    limit, box-constrained) for joint amplitude+phase GRAPE.
    ``hamiltonian_batch`` turns this into a robust/ensemble optimization (e.g.
    a B0-offset and B1-scale-factor grid of ``H_drift`` variants sharing the
    same controls), reduced via ``ensemble_reduction``.

    **Gradient control channel.** Pass ``gradient_operator_batch`` (per-case,
    position-scaled gradient operators, e.g. from
    ``hamiltonians.position_gradient_batch``) to activate the gradient channel.
    ``optimize_gradient=True`` optimizes the shared gradient waveform ``g(t)``
    (needs ``initial_gradient`` and ``gradient_max``); otherwise it is held at
    ``fixed_gradient``. With a gradient channel, ``target`` may be per-case
    (shape ``(batch, dim)``), e.g. inverted in-slice and untouched out-of-slice
    for a slice-selective pulse. The control vector is
    ``concat([amplitude?, phase, gradient?])``.
    """

    initial_phase = np.asarray(initial_phase, dtype=np.float64).reshape(-1)
    n_segments = initial_phase.size
    if n_segments == 0:
        raise ValueError("initial_phase must not be empty")

    if optimize_gradient and gradient_operator_batch is None:
        raise ValueError(
            "optimize_gradient=True requires gradient_operator_batch "
            "(the per-case gradient operators to drive)"
        )

    blocks: list[np.ndarray] = []
    if optimize_amplitude:
        if amplitude_max_hz is None:
            raise ValueError(
                "amplitude_max_hz (the peak-B1 hardware limit) is required "
                "when optimize_amplitude=True"
            )
        if initial_amplitude is None:
            raise ValueError("initial_amplitude is required when optimize_amplitude=True")
        initial_amplitude = np.asarray(initial_amplitude, dtype=np.float64).reshape(-1)
        if initial_amplitude.size != n_segments:
            raise ValueError("initial_amplitude must match initial_phase in length")
        blocks.append(initial_amplitude)
    elif fixed_amplitude is None:
        raise ValueError(
            "fixed_amplitude is required when optimize_amplitude=False "
            "(phase-only control needs a fixed nutation rate)"
        )
    blocks.append(initial_phase)
    if optimize_gradient:
        if gradient_max is None:
            raise ValueError("gradient_max is required when optimize_gradient=True")
        if initial_gradient is None:
            raise ValueError("initial_gradient is required when optimize_gradient=True")
        initial_gradient = np.asarray(initial_gradient, dtype=np.float64).reshape(-1)
        if initial_gradient.size != n_segments:
            raise ValueError("initial_gradient must match initial_phase in length")
        blocks.append(initial_gradient)

    if optimize_gradient or optimize_amplitude:
        bounds = assemble_control_bounds(
            n_segments,
            optimize_amplitude=optimize_amplitude,
            amplitude_max_hz=amplitude_max_hz,
            optimize_gradient=optimize_gradient,
            gradient_max=gradient_max,
            phase_bound_rad=phase_bound_rad,
        )
    else:
        bounds = phase_only_bounds(n_segments, phase_bound_rad=phase_bound_rad)
    x0 = blocks[0] if len(blocks) == 1 else np.concatenate(blocks)

    value_and_grad_fn = make_grape_objective(
        model,
        n_segments=n_segments,
        dt=dt,
        target=target,
        psi0=psi0,
        mode=mode,
        optimize_amplitude=optimize_amplitude,
        fixed_amplitude=fixed_amplitude,
        optimize_gradient=optimize_gradient,
        fixed_gradient=fixed_gradient,
        gradient_operator_batch=gradient_operator_batch,
        hamiltonian_batch=hamiltonian_batch,
        ensemble_reduction=ensemble_reduction,
        phase_smoothness_weight=phase_smoothness_weight,
        gradient_smoothness_weight=gradient_smoothness_weight,
    )

    initial_score, _initial_grad = value_and_grad_fn(x0)
    initial_score = float(initial_score)

    run = scipy_maximize_with_grad(
        value_and_grad_fn,
        x0,
        bounds=bounds.as_pairs(),
        scipy_method=scipy_method,
        options=scipy_options,
    )

    dt_arr = np.broadcast_to(np.asarray(dt, dtype=np.float64), (n_segments,)).copy()

    return GrapeOptimizationResult(
        mode=mode,
        n_segments=n_segments,
        dt=dt_arr,
        optimize_amplitude=optimize_amplitude,
        optimize_gradient=optimize_gradient,
        bounds=bounds,
        initial_controls=x0.copy(),
        best_controls=run.best_x,
        best_fidelity=run.best_score,
        initial_fidelity=initial_score,
        history_scores=run.history_scores,
        history_controls=run.history_x,
        iterations=run.iterations,
        improved=run.improved,
        optimizer_method=run.method,
        optimizer_success=run.success,
        optimizer_message=run.message,
    )


def grape_optimize_phase_only(
    model: ControlHamiltonianModel,
    initial_phase: np.ndarray,
    *,
    fixed_amplitude: float | np.ndarray,
    dt: float | np.ndarray,
    target: np.ndarray,
    psi0: np.ndarray | None = None,
    mode: Literal["state_transfer", "gate"] = "state_transfer",
    phase_bound_rad: float = 4 * np.pi,
    hamiltonian_batch: Sequence[np.ndarray] | None = None,
    ensemble_reduction: Literal["mean", "worst_case"] = "mean",
    phase_smoothness_weight: float = 0.0,
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> GrapeOptimizationResult:
    """Explicit phase-only GRAPE entry point.

    The expected common case: switching-power-amplifier hardware that cannot
    vary RF amplitude, so only phase is agile per segment.
    """

    return grape_optimize(
        model,
        initial_phase,
        dt=dt,
        target=target,
        psi0=psi0,
        mode=mode,
        optimize_amplitude=False,
        fixed_amplitude=fixed_amplitude,
        phase_bound_rad=phase_bound_rad,
        hamiltonian_batch=hamiltonian_batch,
        ensemble_reduction=ensemble_reduction,
        phase_smoothness_weight=phase_smoothness_weight,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
    )
