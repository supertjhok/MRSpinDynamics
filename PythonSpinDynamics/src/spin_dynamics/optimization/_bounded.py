"""Bounded continuous optimization backends for phase programs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


ScoreFunction = Callable[[np.ndarray], float]
BoundsSpec = tuple[float, float] | Sequence[tuple[float, float]]


@dataclass(frozen=True)
class BoundedOptimizationRun:
    """Backend-agnostic result for maximizing a bounded phase objective."""

    best_x: np.ndarray
    best_score: float
    history_scores: np.ndarray
    history_x: tuple[np.ndarray, ...]
    iterations: int
    improved: bool
    final_step: float
    method: str
    success: bool
    message: str


def validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Validate scalar lower/upper phase bounds."""

    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bounds must be finite")
    if lower >= upper:
        raise ValueError("bounds must be ordered as (lower, upper)")
    return lower, upper


def _is_scalar_bound_pair(bounds: BoundsSpec) -> bool:
    """Return True for a single ``(lower, upper)`` scalar pair.

    False for a sequence of per-parameter ``(lower, upper)`` pairs (each
    element itself a length-2 sequence rather than a bare number).
    """

    if len(bounds) != 2:
        return False
    return np.isscalar(bounds[0]) and np.isscalar(bounds[1])


def _resolve_bounds(
    bounds: BoundsSpec, size: int
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Resolve a scalar-pair or per-parameter bounds spec against ``size``.

    A scalar ``(lower, upper)`` pair is broadcast to every one of the ``size``
    flattened parameters (the original, sole behavior of this module). A
    sequence of ``size`` per-parameter ``(lower, upper)`` pairs is passed
    straight through -- needed by callers with heterogeneous per-parameter
    bounds (e.g. GRAPE's amplitude vs. phase channels), which a single
    broadcast pair cannot express. Returns ``(lower_array, upper_array,
    scipy_bounds_pairs)``.
    """

    if _is_scalar_bound_pair(bounds):
        lower, upper = validate_bounds(bounds)  # type: ignore[arg-type]
        return np.full(size, lower), np.full(size, upper), [(lower, upper)] * size

    pairs = [(float(lo), float(hi)) for lo, hi in bounds]
    if len(pairs) != size:
        raise ValueError("per-parameter bounds must have exactly `size` entries")
    lower_arr = np.array([lo for lo, _ in pairs], dtype=np.float64)
    upper_arr = np.array([hi for _, hi in pairs], dtype=np.float64)
    if not np.all(np.isfinite(lower_arr)) or not np.all(np.isfinite(upper_arr)):
        raise ValueError("bounds must be finite")
    if np.any(lower_arr >= upper_arr):
        raise ValueError("each (lower, upper) pair must be ordered as lower < upper")
    return lower_arr, upper_arr, pairs


def pattern_search_maximize(
    score_fn: ScoreFunction,
    initial: np.ndarray,
    *,
    bounds: tuple[float, float],
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
) -> BoundedOptimizationRun:
    """Maximize a bounded objective with deterministic coordinate search."""

    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    if initial_step <= 0:
        raise ValueError("initial_step must be positive")
    if min_step <= 0:
        raise ValueError("min_step must be positive")
    if not 0 < step_decay < 1:
        raise ValueError("step_decay must be between 0 and 1")

    lower, upper = validate_bounds(bounds)
    current = np.clip(np.asarray(initial, dtype=np.float64).reshape(-1), lower, upper)
    if current.size == 0:
        raise ValueError("initial_phases must not be empty")

    best_score = float(score_fn(current.copy()))
    if not np.isfinite(best_score):
        best_score = -np.inf
    history_scores = [best_score]
    history_x = [current.copy()]
    improved = False
    step = float(initial_step)

    for _pass in range(int(max_passes)):
        pass_improved = False
        for idx in range(current.size):
            local_best_score = best_score
            local_best = current.copy()
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[idx] = np.clip(candidate[idx] + direction * step, lower, upper)
                score = float(score_fn(candidate.copy()))
                if not np.isfinite(score):
                    score = -np.inf
                history_scores.append(score)
                history_x.append(candidate.copy())
                if score > local_best_score:
                    local_best_score = score
                    local_best = candidate
            if local_best_score > best_score:
                current = local_best
                best_score = local_best_score
                pass_improved = True
                improved = True
        if pass_improved:
            continue
        step *= float(step_decay)
        if step < min_step:
            break

    return BoundedOptimizationRun(
        best_x=current,
        best_score=best_score,
        history_scores=np.asarray(history_scores, dtype=np.float64),
        history_x=tuple(history_x),
        iterations=max(0, len(history_scores) - 1),
        improved=improved,
        final_step=step,
        method="pattern",
        success=np.isfinite(best_score),
        message="coordinate pattern search completed",
    )


def scipy_maximize(
    score_fn: ScoreFunction,
    initial: np.ndarray,
    *,
    bounds: tuple[float, float],
    scipy_method: str = "L-BFGS-B",
    options: dict[str, object] | None = None,
) -> BoundedOptimizationRun:
    """Maximize a bounded objective with SciPy when the optional extra is present."""

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for optimizer='scipy'. Install the optional "
            "optimization dependencies with `python -m pip install -e .[opt]`."
        ) from exc

    lower, upper = validate_bounds(bounds)
    x0 = np.clip(np.asarray(initial, dtype=np.float64).reshape(-1), lower, upper)
    if x0.size == 0:
        raise ValueError("initial_phases must not be empty")

    history_scores: list[float] = []
    history_x: list[np.ndarray] = []

    def objective(x: np.ndarray) -> float:
        phases = np.asarray(x, dtype=np.float64).reshape(-1)
        score = float(score_fn(phases.copy()))
        if not np.isfinite(score):
            score = -np.inf
        history_scores.append(score)
        history_x.append(phases.copy())
        return -score

    result = minimize(
        objective,
        x0,
        method=scipy_method,
        bounds=[(lower, upper)] * x0.size,
        options=options,
    )
    best_x = np.asarray(result.x, dtype=np.float64).reshape(-1)
    best_score = float(score_fn(best_x.copy()))
    if not np.isfinite(best_score):
        best_score = -np.inf
    history_scores.append(best_score)
    history_x.append(best_x.copy())
    initial_score = history_scores[0] if history_scores else best_score

    return BoundedOptimizationRun(
        best_x=best_x,
        best_score=best_score,
        history_scores=np.asarray(history_scores, dtype=np.float64),
        history_x=tuple(history_x),
        iterations=int(getattr(result, "nfev", max(0, len(history_scores) - 1))),
        improved=best_score > initial_score,
        final_step=0.0,
        method=f"scipy:{scipy_method}",
        success=bool(result.success),
        message=str(result.message),
    )


def scipy_maximize_with_grad(
    value_and_grad_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    initial: np.ndarray,
    *,
    bounds: BoundsSpec,
    scipy_method: str = "L-BFGS-B",
    options: dict[str, object] | None = None,
) -> BoundedOptimizationRun:
    """Maximize using a caller-supplied analytic gradient (e.g. ``jax.grad``).

    ``value_and_grad_fn(x)`` returns ``(score, grad)``; the gradient is handed to
    SciPy via ``jac=True`` so each step costs one forward+backward evaluation
    instead of the ``len(x)+1`` forward evaluations finite differencing needs.

    ``bounds`` accepts either a single scalar ``(lower, upper)`` pair
    (broadcast to every parameter, the original behavior) or a sequence of
    ``len(initial)`` per-parameter ``(lower, upper)`` pairs -- needed by
    callers with heterogeneous per-parameter bounds, such as GRAPE's
    amplitude vs. phase channels.
    """

    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "SciPy is required for the analytic-gradient optimizer. Install the "
            "optional optimization dependencies with `python -m pip install -e .[opt]`."
        ) from exc

    x0_raw = np.asarray(initial, dtype=np.float64).reshape(-1)
    if x0_raw.size == 0:
        raise ValueError("initial_phases must not be empty")
    lower_arr, upper_arr, bounds_pairs = _resolve_bounds(bounds, x0_raw.size)
    x0 = np.clip(x0_raw, lower_arr, upper_arr)

    history_scores: list[float] = []
    history_x: list[np.ndarray] = []

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        phases = np.asarray(x, dtype=np.float64).reshape(-1)
        score, grad = value_and_grad_fn(phases)
        score = float(score)
        grad = np.asarray(grad, dtype=np.float64).reshape(-1)
        if not np.isfinite(score):
            score = -np.inf
        history_scores.append(score)
        history_x.append(phases.copy())
        return -score, -grad

    result = minimize(
        objective,
        x0,
        method=scipy_method,
        jac=True,
        bounds=bounds_pairs,
        options=options,
    )
    best_x = np.asarray(result.x, dtype=np.float64).reshape(-1)
    best_score, _grad = value_and_grad_fn(best_x)
    best_score = float(best_score)
    if not np.isfinite(best_score):
        best_score = -np.inf
    history_scores.append(best_score)
    history_x.append(best_x.copy())
    initial_score = history_scores[0] if history_scores else best_score

    return BoundedOptimizationRun(
        best_x=best_x,
        best_score=best_score,
        history_scores=np.asarray(history_scores, dtype=np.float64),
        history_x=tuple(history_x),
        iterations=int(getattr(result, "nfev", max(0, len(history_scores) - 1))),
        improved=best_score > initial_score,
        final_step=0.0,
        method=f"jax+scipy:{scipy_method}",
        success=bool(result.success),
        message=str(result.message),
    )


def maximize_bounded(
    score_fn: ScoreFunction,
    initial: np.ndarray,
    *,
    bounds: tuple[float, float],
    optimizer: str = "auto",
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> BoundedOptimizationRun:
    """Maximize a bounded objective with the selected backend."""

    if optimizer not in {"auto", "pattern", "scipy"}:
        raise ValueError("optimizer must be 'auto', 'pattern', or 'scipy'")
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    if initial_step <= 0:
        raise ValueError("initial_step must be positive")
    if min_step <= 0:
        raise ValueError("min_step must be positive")
    if not 0 < step_decay < 1:
        raise ValueError("step_decay must be between 0 and 1")
    if optimizer == "scipy":
        return scipy_maximize(
            score_fn,
            initial,
            bounds=bounds,
            scipy_method=scipy_method,
            options=scipy_options,
        )
    if optimizer == "auto":
        try:
            return scipy_maximize(
                score_fn,
                initial,
                bounds=bounds,
                scipy_method=scipy_method,
                options=scipy_options,
            )
        except ImportError:
            pass
    return pattern_search_maximize(
        score_fn,
        initial,
        bounds=bounds,
        initial_step=initial_step,
        step_decay=step_decay,
        min_step=min_step,
        max_passes=max_passes,
    )
