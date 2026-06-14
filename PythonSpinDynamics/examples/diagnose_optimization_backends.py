"""Compare pulse-optimization backends on compact tuned-probe cases.

This diagnostic is useful when SciPy is available locally. It compares the
dependency-light pattern search against SciPy's bounded optimizer and a random
phase baseline, then prints residual metrics for the inverse-excitation
objective. A good inverse pulse should drive residual/target area below 1.
"""

from __future__ import annotations

import argparse
import importlib.util

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rot_axis_arba3
from spin_dynamics.optimization import (
    evaluate_tuned_excitation_pulse,
    evaluate_tuned_inverse_excitation_pulse,
    optimize_tuned_excitation_phases,
    optimize_tuned_inverse_excitation_phases,
)


def _ideal_refocusing_axis(numpts: int) -> tuple[np.ndarray, np.ndarray]:
    del_w = np.linspace(-10.0, 10.0, int(numpts))
    neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
    return del_w, neff


def _residual_ratio(target, inverse_eval) -> float:
    residual = target.mrx + inverse_eval.excitation.mrx
    target_norm = trapezoid(np.abs(target.mrx), target.del_w)
    if target_norm == 0:
        return np.inf
    return float(trapezoid(np.abs(residual), target.del_w) / target_norm)


def _random_inverse_baseline(
    target,
    neff: np.ndarray,
    *,
    num_samples: int,
    seed: int,
    numpts: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    best_mismatch = np.inf
    best_ratio = np.inf
    for _idx in range(int(num_samples)):
        phases = rng.uniform(0.0, 2 * np.pi, size=target.phases.size)
        evaluation = evaluate_tuned_inverse_excitation_pulse(
            phases,
            neff,
            target.mrx,
            target.snr,
            numpts=numpts,
        )
        if evaluation.mismatch < best_mismatch:
            best_mismatch = float(evaluation.mismatch)
            best_ratio = _residual_ratio(target, evaluation)
    return best_mismatch, best_ratio


def _print_backend_report(
    backend: str,
    *,
    num_segments: int,
    numpts: int,
    max_passes: int,
    random_samples: int,
    seed: int,
) -> None:
    _del_w, neff = _ideal_refocusing_axis(numpts)
    initial = np.zeros(int(num_segments), dtype=np.float64)

    target_result = optimize_tuned_excitation_phases(
        initial,
        neff,
        numpts=numpts,
        optimizer=backend,
        max_passes=max_passes,
    )
    target = target_result.best_evaluation

    phase_shift = np.mod(target.phases + np.pi, 2 * np.pi)
    shifted = evaluate_tuned_excitation_pulse(phase_shift, neff, numpts=numpts)
    phase_shift_ratio = trapezoid(
        np.abs(target.mrx + shifted.mrx),
        target.del_w,
    ) / trapezoid(np.abs(target.mrx), target.del_w)

    initial_inverse = evaluate_tuned_inverse_excitation_pulse(
        phase_shift,
        neff,
        target.mrx,
        target.snr,
        numpts=numpts,
    )
    inverse_result = optimize_tuned_inverse_excitation_phases(
        phase_shift,
        neff,
        target.mrx,
        target.snr,
        numpts=numpts,
        optimizer=backend,
        max_passes=max_passes,
    )
    random_mismatch, random_ratio = _random_inverse_baseline(
        target,
        neff,
        num_samples=random_samples,
        seed=seed,
        numpts=numpts,
    )

    print(f"[{backend}]")
    print(
        "  excitation: "
        f"initial={target_result.initial_score:.6g} "
        f"best={target_result.best_score:.6g} "
        f"evals={target_result.history_scores.size}"
    )
    print(f"  phase_shift_residual_ratio={phase_shift_ratio:.6g}")
    print(
        "  inverse initial: "
        f"mismatch={initial_inverse.mismatch:.6g} "
        f"residual_ratio={_residual_ratio(target, initial_inverse):.6g}"
    )
    print(
        "  inverse optimized: "
        f"mismatch={inverse_result.best_evaluation.mismatch:.6g} "
        f"residual_ratio={_residual_ratio(target, inverse_result.best_evaluation):.6g} "
        f"evals={inverse_result.history_scores.size}"
    )
    print(
        "  random baseline: "
        f"mismatch={random_mismatch:.6g} residual_ratio={random_ratio:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=21, help="Offset grid size.")
    parser.add_argument(
        "--segments",
        type=int,
        default=3,
        help="Excitation phase segments.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=8,
        help="Pattern-search passes; ignored by SciPy.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=200,
        help="Random inverse candidates for a baseline.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random baseline seed.")
    parser.add_argument(
        "--backend",
        choices=["all", "pattern", "scipy"],
        default="all",
        help="Backend to run.",
    )
    args = parser.parse_args()

    backends = ["pattern", "scipy"] if args.backend == "all" else [args.backend]
    for backend in backends:
        if backend == "scipy" and importlib.util.find_spec("scipy") is None:
            print("[scipy]")
            print("  skipped: SciPy is not installed")
            continue
        _print_backend_report(
            backend,
            num_segments=args.segments,
            numpts=args.numpts,
            max_passes=args.max_passes,
            random_samples=args.random_samples,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
