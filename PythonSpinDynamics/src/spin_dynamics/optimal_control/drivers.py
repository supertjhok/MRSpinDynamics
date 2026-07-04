"""GRAPE multistart driver.

Mirrors ``optimization.drivers``'s multistart pattern: GRAPE landscapes are
multimodal, so a single-start result is not a trustworthy validation baseline
on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.optimal_control.parameterization import random_phase_starts
from spin_dynamics.optimal_control.solvers import GrapeOptimizationResult, grape_optimize


@dataclass(frozen=True)
class GrapeMultiStartResult:
    """Result of repeated random-phase-start GRAPE optimization."""

    initial_phases: np.ndarray
    results: tuple[GrapeOptimizationResult, ...]
    best_index: int
    best_result: GrapeOptimizationResult
    best_fidelity: float


def run_grape_multistart(
    model: Any,
    n_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | None = None,
    phase_bounds: tuple[float, float] = (-np.pi, np.pi),
    **grape_optimize_kwargs: Any,
) -> GrapeMultiStartResult:
    """Run repeated random-phase-start GRAPE optimization and rank by fidelity.

    ``**grape_optimize_kwargs`` are forwarded to ``grape_optimize`` verbatim
    (``dt``, ``target``, ``psi0``, ``fixed_amplitude``, ``mode``, ...).
    """

    n_segments = int(n_segments)
    if initial_phases is None:
        starts = random_phase_starts(
            num_starts, n_segments, bounds=phase_bounds, seed=seed, rng=rng
        )
    else:
        starts = np.asarray(initial_phases, dtype=np.float64)
        if starts.ndim != 2 or starts.shape[1] != n_segments:
            raise ValueError("initial_phases must have shape (num_starts, n_segments)")
        if starts.shape[0] == 0:
            raise ValueError("initial_phases must include at least one start")

    results = tuple(
        grape_optimize(model, start, **grape_optimize_kwargs) for start in starts
    )
    scores = np.array([float(r.best_fidelity) for r in results], dtype=np.float64)
    best_index = int(np.nanargmax(scores))
    return GrapeMultiStartResult(
        initial_phases=starts,
        results=results,
        best_index=best_index,
        best_result=results[best_index],
        best_fidelity=float(scores[best_index]),
    )
