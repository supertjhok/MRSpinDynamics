"""Multi-start optimization driver scaffolds for OCT-style workflows.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_*_repeat.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned_repeat.m
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import spin_dynamics.optimization.excitation as excitation_module
import spin_dynamics.optimization.refocusing as refocusing_module


@dataclass(frozen=True)
class MultiStartOptimizationResult:
    """Array-returning result for repeated random-start phase optimization."""

    pulse_kind: str
    probe: str
    initial_phases: np.ndarray
    results: tuple[Any, ...]
    best_index: int
    best_result: Any
    best_score: float
    bounds: tuple[float, float]


def _validate_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("bounds must be finite")
    if lower >= upper:
        raise ValueError("bounds must be ordered as (lower, upper)")
    return lower, upper


def random_phase_starts(
    num_starts: int,
    num_segments: int,
    *,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate reproducible random phase starts within bounded phase limits."""

    if num_starts <= 0:
        raise ValueError("num_starts must be positive")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    lower, upper = _validate_bounds(bounds)
    generator = np.random.default_rng(seed) if rng is None else rng
    return generator.uniform(lower, upper, size=(int(num_starts), int(num_segments)))


def _prepare_starts(
    num_segments: int,
    *,
    num_starts: int,
    bounds: tuple[float, float],
    seed: int | None,
    rng: np.random.Generator | None,
    initial_phases: np.ndarray | list[list[float]] | None,
) -> np.ndarray:
    if initial_phases is None:
        return random_phase_starts(
            num_starts,
            num_segments,
            bounds=bounds,
            seed=seed,
            rng=rng,
        )
    starts = np.asarray(initial_phases, dtype=np.float64)
    if starts.ndim != 2:
        raise ValueError("initial_phases must have shape (num_starts, num_segments)")
    if starts.shape[0] == 0:
        raise ValueError("initial_phases must include at least one start")
    if starts.shape[1] != int(num_segments):
        raise ValueError("initial_phases second dimension must match num_segments")
    if not np.all(np.isfinite(starts)):
        raise ValueError("initial_phases must be finite")
    return starts.copy()


def _rank_results(results: tuple[Any, ...]) -> tuple[int, Any, float]:
    scores = np.array([float(result.best_score) for result in results], dtype=np.float64)
    if scores.size == 0:
        raise ValueError("results must not be empty")
    best_index = int(np.nanargmax(scores))
    return best_index, results[best_index], float(scores[best_index])


def _run_refocusing_multistart(
    probe: str,
    optimizer: Any,
    num_segments: int,
    *,
    num_starts: int,
    seed: int | None,
    rng: np.random.Generator | None,
    initial_phases: np.ndarray | list[list[float]] | None,
    bounds: tuple[float, float],
    optimizer_kwargs: dict[str, Any],
) -> MultiStartOptimizationResult:
    starts = _prepare_starts(
        num_segments,
        num_starts=num_starts,
        bounds=bounds,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
    )
    lower, upper = _validate_bounds(bounds)
    results = tuple(
        optimizer(start, bounds=(lower, upper), **optimizer_kwargs)
        for start in starts
    )
    best_index, best_result, best_score = _rank_results(results)
    return MultiStartOptimizationResult(
        pulse_kind="refocusing",
        probe=probe,
        initial_phases=starts,
        results=results,
        best_index=best_index,
        best_result=best_result,
        best_score=best_score,
        bounds=(lower, upper),
    )


def run_tuned_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start tuned refocusing phase searches."""

    return _run_refocusing_multistart(
        "tuned",
        refocusing_module.optimize_tuned_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_ideal_v0crit_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start ideal v0crit refocusing phase searches."""

    return _run_refocusing_multistart(
        "ideal_v0crit",
        refocusing_module.optimize_ideal_v0crit_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_ideal_v0crit_excited_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated ideal v0crit searches with a fixed excitation vector."""

    return _run_refocusing_multistart(
        "ideal_v0crit_excited",
        refocusing_module.optimize_ideal_v0crit_excited_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_ideal_time_varying_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start ideal time-varying refocusing searches."""

    return _run_refocusing_multistart(
        "ideal_time_varying",
        refocusing_module.optimize_ideal_time_varying_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_untuned_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start untuned refocusing phase searches."""

    return _run_refocusing_multistart(
        "untuned",
        refocusing_module.optimize_untuned_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_matched_refocusing_multistart(
    num_segments: int,
    *,
    num_starts: int = 4,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start matched refocusing phase searches."""

    return _run_refocusing_multistart(
        "matched",
        refocusing_module.optimize_matched_refocusing_phases,
        num_segments,
        num_starts=num_starts,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
        bounds=bounds,
        optimizer_kwargs=optimizer_kwargs,
    )


def run_tuned_excitation_multistart(
    num_segments: int,
    neff: np.ndarray | list[list[float]],
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run repeated random-start tuned excitation phase searches."""

    starts = _prepare_starts(
        num_segments,
        num_starts=num_starts,
        bounds=bounds,
        seed=seed,
        rng=rng,
        initial_phases=initial_phases,
    )
    lower, upper = _validate_bounds(bounds)
    results = tuple(
        excitation_module.optimize_tuned_excitation_phases(
            start,
            neff,
            bounds=(lower, upper),
            **optimizer_kwargs,
        )
        for start in starts
    )
    best_index, best_result, best_score = _rank_results(results)
    return MultiStartOptimizationResult(
        pulse_kind="excitation",
        probe="tuned",
        initial_phases=starts,
        results=results,
        best_index=best_index,
        best_result=best_result,
        best_score=best_score,
        bounds=(lower, upper),
    )


def _inverse_start(
    center: np.ndarray,
    generator: np.random.Generator,
    random_fraction: float,
    bounds: tuple[float, float],
) -> np.ndarray:
    lower, upper = bounds
    random_phase = generator.uniform(0.0, 2 * np.pi, size=center.size)
    start = (1.0 - random_fraction) * center + random_fraction * random_phase
    return np.clip(start, lower, upper)


def run_tuned_inverse_excitation_multistart(
    num_segments: int,
    neff: np.ndarray | list[list[float]],
    target_mrx: np.ndarray | list[complex],
    target_snr: float,
    reference_phases: np.ndarray | list[float],
    *,
    num_starts: int = 24,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    initial_phases: np.ndarray | list[list[float]] | None = None,
    random_fraction: float = 0.3,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    **optimizer_kwargs: Any,
) -> MultiStartOptimizationResult:
    """Run MATLAB-style repeated starts for tuned inverse excitation search.

    The first generated start is the deterministic ``pi + reference_phases``
    phase-flipped candidate. Each later generated start is centered near the
    best inverse pulse found so far, following the refinement strategy used by
    the MATLAB repeat driver.
    """

    if num_starts <= 0:
        raise ValueError("num_starts must be positive")
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    if random_fraction < 0 or random_fraction > 1:
        raise ValueError("random_fraction must be between 0 and 1")
    reference = np.asarray(reference_phases, dtype=np.float64).reshape(-1)
    if reference.size != int(num_segments):
        raise ValueError("reference_phases must match num_segments")
    if not np.all(np.isfinite(reference)):
        raise ValueError("reference_phases must be finite")

    lower, upper = _validate_bounds(bounds)
    if initial_phases is not None:
        starts = _prepare_starts(
            num_segments,
            num_starts=num_starts,
            bounds=(lower, upper),
            seed=seed,
            rng=rng,
            initial_phases=initial_phases,
        )
        results = tuple(
            excitation_module.optimize_tuned_inverse_excitation_phases(
                start,
                neff,
                target_mrx,
                target_snr,
                bounds=(lower, upper),
                **optimizer_kwargs,
            )
            for start in starts
        )
    else:
        generator = np.random.default_rng(seed) if rng is None else rng
        starts_list: list[np.ndarray] = []
        results_list: list[Any] = []
        best_result: Any | None = None
        best_score = -np.inf
        center = np.mod(np.pi + reference, 2 * np.pi)
        for start_index in range(int(num_starts)):
            if start_index == 0:
                start = np.clip(center, lower, upper)
            else:
                start = _inverse_start(
                    center,
                    generator,
                    float(random_fraction),
                    (lower, upper),
                )
            result = excitation_module.optimize_tuned_inverse_excitation_phases(
                start,
                neff,
                target_mrx,
                target_snr,
                bounds=(lower, upper),
                **optimizer_kwargs,
            )
            starts_list.append(start)
            results_list.append(result)
            if float(result.best_score) > best_score:
                best_result = result
                best_score = float(result.best_score)
            center = np.asarray(best_result.best_phases, dtype=np.float64)
        starts = np.vstack(starts_list)
        results = tuple(results_list)

    best_index, best_result, best_score = _rank_results(results)
    return MultiStartOptimizationResult(
        pulse_kind="inverse_excitation",
        probe="tuned",
        initial_phases=starts,
        results=results,
        best_index=best_index,
        best_result=best_result,
        best_score=best_score,
        bounds=(lower, upper),
    )
