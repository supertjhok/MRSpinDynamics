"""Continuous refocusing-pulse phase optimization helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_tuned.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_untuned.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_matched.m
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.optimization._bounded import maximize_bounded, validate_bounds
from spin_dynamics.optimization.spa import (
    MatchedRefocusingEvaluation,
    TunedRefocusingEvaluation,
    UntunedRefocusingEvaluation,
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
)


RefocusingEvaluation = (
    TunedRefocusingEvaluation
    | UntunedRefocusingEvaluation
    | MatchedRefocusingEvaluation
)
RefocusingEvaluator = Callable[..., RefocusingEvaluation]


@dataclass(frozen=True)
class RefocusingOptimizationResult:
    """Result of bounded fixed-amplitude refocusing phase optimization."""

    probe: str
    initial_phases: np.ndarray
    best_phases: np.ndarray
    best_score: float
    initial_score: float
    best_evaluation: RefocusingEvaluation
    history_scores: np.ndarray
    history_phases: tuple[np.ndarray, ...]
    iterations: int
    improved: bool
    final_step: float
    bounds: tuple[float, float]
    optimizer_method: str
    optimizer_success: bool
    optimizer_message: str


def _evaluate_score(
    evaluator: RefocusingEvaluator,
    phases: np.ndarray,
    *,
    segment_fraction: float,
    numpts: int,
    excitation_amplitude: float,
) -> tuple[float, RefocusingEvaluation]:
    evaluation = evaluator(
        phases,
        segment_fraction=segment_fraction,
        numpts=numpts,
        excitation_amplitude=excitation_amplitude,
    )
    score = float(evaluation.snr)
    if not np.isfinite(score):
        score = -np.inf
    return score, evaluation


def _optimize_refocusing_phase_program(
    probe: str,
    initial_phases: np.ndarray | list[float],
    evaluator: RefocusingEvaluator,
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    excitation_amplitude: float = 6.0,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> RefocusingOptimizationResult:
    """Optimize fixed-amplitude refocusing phases with a bounded optimizer."""

    lower, upper = validate_bounds(bounds)
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    def score_fn(phases: np.ndarray) -> float:
        score, _evaluation = _evaluate_score(
            evaluator,
            phases,
            segment_fraction=segment_fraction,
            numpts=numpts,
            excitation_amplitude=excitation_amplitude,
        )
        return score

    run = maximize_bounded(
        score_fn,
        initial,
        bounds=(lower, upper),
        optimizer=optimizer,
        initial_step=initial_step,
        step_decay=step_decay,
        min_step=min_step,
        max_passes=max_passes,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
    )
    _best_score, best_evaluation = _evaluate_score(
        evaluator,
        run.best_x,
        segment_fraction=segment_fraction,
        numpts=numpts,
        excitation_amplitude=excitation_amplitude,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )

    return RefocusingOptimizationResult(
        probe=probe,
        initial_phases=initial,
        best_phases=run.best_x,
        best_score=run.best_score,
        initial_score=initial_score,
        best_evaluation=best_evaluation,
        history_scores=run.history_scores,
        history_phases=run.history_x,
        iterations=run.iterations,
        improved=run.improved,
        final_step=run.final_step,
        bounds=(lower, upper),
        optimizer_method=run.method,
        optimizer_success=run.success,
        optimizer_message=run.message,
    )


def optimize_tuned_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize tuned-probe fixed-amplitude refocusing phases."""

    return _optimize_refocusing_phase_program(
        "tuned",
        initial_phases,
        evaluate_tuned_refocusing_pulse,
        **kwargs,
    )


def optimize_untuned_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize untuned-probe fixed-amplitude refocusing phases."""

    return _optimize_refocusing_phase_program(
        "untuned",
        initial_phases,
        evaluate_untuned_refocusing_pulse,
        **kwargs,
    )


def optimize_matched_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize matched-probe fixed-amplitude refocusing phases.

    The matched transient solver is slower than the tuned and untuned paths, so
    the default grid and pass count are intentionally conservative.
    """

    options = {"numpts": 21, "max_passes": 3}
    options.update(kwargs)
    return _optimize_refocusing_phase_program(
        "matched",
        initial_phases,
        evaluate_matched_refocusing_pulse,
        **options,
    )
