"""Isochromat grid helpers."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class RephasingAnalysis:
    """Rephasing estimate for a uniformly spaced offset grid."""

    spacing: float
    rephase_time: float
    max_time: float
    safety_factor: float
    required_spacing: float
    recommended_numpts: int | None
    ok: bool


def offset_spacing(del_w: np.ndarray) -> float:
    """Return the uniform offset spacing for an isochromat grid."""

    grid = np.asarray(del_w, dtype=np.float64).reshape(-1)
    if grid.size < 2:
        return np.inf

    diffs = np.diff(np.sort(grid))
    spacing = float(np.median(diffs))
    if spacing <= 0:
        raise ValueError("offset grid must contain distinct points")
    if not np.allclose(diffs, spacing, rtol=1e-6, atol=1e-12):
        raise ValueError("offset grid must be uniformly spaced for rephasing analysis")
    return spacing


def estimate_rephase_time(del_w: np.ndarray) -> float:
    """Estimate the normalized rephasing time for a uniform offset grid.

    The Python kernels use angular normalized offsets, so adjacent isochromats
    reacquire the same phase after approximately ``2*pi / spacing``.
    """

    spacing = offset_spacing(del_w)
    if not np.isfinite(spacing):
        return np.inf
    return float(2 * np.pi / spacing)


def recommended_numpts_for_rephasing(
    maxoffs: float,
    max_time: float,
    safety_factor: float = 1.25,
) -> int:
    """Return the minimum grid size that keeps rephasing beyond max time."""

    if maxoffs <= 0:
        raise ValueError("maxoffs must be positive")
    if max_time <= 0:
        return 2
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")

    required_spacing = 2 * np.pi / (float(safety_factor) * float(max_time))
    return int(np.ceil(2 * float(maxoffs) / required_spacing)) + 2


def analyze_rephasing(
    del_w: np.ndarray,
    max_time: float,
    safety_factor: float = 1.25,
) -> RephasingAnalysis:
    """Analyze whether a grid is fine enough for the requested simulation time."""

    if max_time < 0:
        raise ValueError("max_time must be non-negative")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")

    grid = np.asarray(del_w, dtype=np.float64).reshape(-1)
    spacing = offset_spacing(grid)
    rephase_time = estimate_rephase_time(grid)
    required_spacing = (
        np.inf if max_time == 0 else 2 * np.pi / (float(safety_factor) * float(max_time))
    )
    ok = bool(rephase_time > float(safety_factor) * float(max_time))

    recommended = None
    if not ok and grid.size >= 2:
        maxoffs = float(np.max(np.abs(grid)))
        recommended = recommended_numpts_for_rephasing(
            maxoffs=maxoffs,
            max_time=float(max_time),
            safety_factor=float(safety_factor),
        )

    return RephasingAnalysis(
        spacing=float(spacing),
        rephase_time=float(rephase_time),
        max_time=float(max_time),
        safety_factor=float(safety_factor),
        required_spacing=float(required_spacing),
        recommended_numpts=recommended,
        ok=ok,
    )


def check_rephasing(
    del_w: np.ndarray,
    max_time: float,
    safety_factor: float = 1.25,
    action: str = "warn",
) -> RephasingAnalysis:
    """Warn or raise when the isochromat grid may produce rephasing artifacts."""

    analysis = analyze_rephasing(del_w, max_time, safety_factor)
    if analysis.ok or action == "ignore":
        return analysis

    message = (
        "Isochromat grid may rephase before the requested simulation finishes: "
        f"spacing={analysis.spacing:.6g}, rephase_time={analysis.rephase_time:.6g}, "
        f"max_time={analysis.max_time:.6g}, safety_factor={analysis.safety_factor:.6g}. "
        f"Use at least numpts={analysis.recommended_numpts} over this offset span, "
        "or pass auto_refine_grid=True in supported workflows."
    )
    if action == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif action == "raise":
        raise RuntimeError(message)
    else:
        raise ValueError("action must be 'ignore', 'warn', or 'raise'")
    return analysis
