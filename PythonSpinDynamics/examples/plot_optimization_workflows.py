"""Plot diagnostic tuned-probe OCT optimization helpers.

The inverse-excitation panel is intentionally diagnostic. It exposes objective
and residual behavior while MATLAB parity for the inverse pulse-design workflow
is still being validated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.core.rotations import calc_rot_axis_arba3
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.optimization import (
    run_tuned_excitation_multistart,
    run_tuned_inverse_excitation_multistart,
    run_tuned_refocusing_multistart,
)




def _ideal_refocusing_axis(numpts: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a simple reference axis used by the excitation examples."""

    del_w = np.linspace(-10.0, 10.0, int(numpts))
    # A hard pi pulse is a cheap stand-in for an optimized refocusing axis.
    # For production studies, pass a higher-resolution refocusing-axis array
    # produced by the tuned-probe axis calculation or a MATLAB fixture.
    neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
    return del_w, neff


def _plot_history(ax, result, label: str) -> None:
    scores = np.asarray(result.history_scores, dtype=np.float64)
    ax.plot(np.arange(scores.size), scores, marker="o", label=label)


def _inverse_residual_ratio(target, inverse_result) -> float:
    inverse_eval = inverse_result.best_evaluation.excitation
    target_norm = trapezoid(np.abs(target.mrx), target.del_w)
    if target_norm == 0:
        return np.inf
    residual_norm = trapezoid(np.abs(target.mrx + inverse_eval.mrx), target.del_w)
    return float(residual_norm / target_norm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=11, help="Offset grid size.")
    parser.add_argument("--segments", type=int, default=2, help="Refocusing phase segments.")
    parser.add_argument("--starts", type=int, default=2, help="Random starts for refocusing.")
    parser.add_argument(
        "--excitation-segments",
        type=int,
        default=3,
        help="Fixed-amplitude phase segments in the excitation target.",
    )
    parser.add_argument(
        "--excitation-starts",
        type=int,
        default=2,
        help="Random starts for target excitation optimization.",
    )
    parser.add_argument(
        "--inverse-starts",
        type=int,
        default=4,
        help="MATLAB-style phase-flipped starts for inverse excitation.",
    )
    parser.add_argument(
        "--inverse-random-fraction",
        type=float,
        default=0.3,
        help="Fraction of random phase mixed into each generated inverse start.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=2,
        help="Pattern-search passes for each single-start optimizer.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for multi-start phases.")
    parser.add_argument(
        "--optimizer",
        choices=["auto", "pattern", "scipy"],
        default="pattern",
        help="Backend used by the single-start phase optimizers.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    # This is a diagnostic plot, not a finished pulse-design recipe. The small
    # defaults make failures visible quickly; useful pulse design generally
    # needs more offset points, more segments, more starts, and the SciPy
    # optimizer backend.
    refocusing = run_tuned_refocusing_multistart(
        args.segments,
        num_starts=args.starts,
        seed=args.seed,
        numpts=args.numpts,
        max_passes=args.max_passes,
        initial_step=0.4,
        optimizer=args.optimizer,
    )

    del_w, neff = _ideal_refocusing_axis(args.numpts)

    # The excitation optimizer requires a refocusing axis. We use the cheap
    # reference axis above so the example is deterministic and fast, then run a
    # small multi-start search to build a nontrivial target pulse.
    excitation = run_tuned_excitation_multistart(
        args.excitation_segments,
        neff,
        num_starts=args.excitation_starts,
        seed=args.seed + 1,
        numpts=args.numpts,
        max_passes=args.max_passes,
        initial_step=0.4,
        optimizer=args.optimizer,
    )

    # The inverse workflow is diagnostic, not a validated design recipe. It
    # always tries pi + the target phases first, then perturbs the best inverse
    # found so far for each later run. The spectra panel reports whether that
    # actually cancels the target for this simplified setup.
    target = excitation.best_result.best_evaluation
    inverse = run_tuned_inverse_excitation_multistart(
        args.excitation_segments,
        neff,
        target.mrx,
        target.snr,
        target.phases,
        num_starts=args.inverse_starts,
        seed=args.seed + 2,
        random_fraction=args.inverse_random_fraction,
        numpts=args.numpts,
        max_passes=args.max_passes,
        initial_step=0.4,
        optimizer=args.optimizer,
    )
    inverse_residual_ratios = np.array(
        [_inverse_residual_ratio(target, result) for result in inverse.results],
        dtype=np.float64,
    )
    residual_best_index = int(np.nanargmin(inverse_residual_ratios))
    residual_best = inverse.results[residual_best_index]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # Panel 1: compare all random-start refocusing runs and mark the best score.
    for idx, result in enumerate(refocusing.results, start=1):
        _plot_history(axes[0, 0], result, f"start {idx}")
    axes[0, 0].axhline(refocusing.best_score, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Tuned Refocusing Multi-Start")
    axes[0, 0].set_xlabel("Objective evaluation")
    axes[0, 0].set_ylabel("SNR score")
    axes[0, 0].legend()

    # Panel 2: excitation and inverse objectives use different signs. The
    # inverse optimizer maximizes negative mismatch, so higher is still better.
    _plot_history(axes[0, 1], excitation.best_result, "target best")
    for idx, result in enumerate(inverse.results, start=1):
        is_objective_best = idx - 1 == inverse.best_index
        is_residual_best = idx - 1 == residual_best_index
        alpha = 1.0 if is_objective_best or is_residual_best else 0.35
        linewidth = 2.0 if is_objective_best or is_residual_best else 1.0
        suffix = ""
        if is_objective_best:
            suffix += ", obj"
        if is_residual_best:
            suffix += ", resid"
        axes[0, 1].plot(
            np.arange(result.history_scores.size),
            result.history_scores,
            marker="o",
            alpha=alpha,
            linewidth=linewidth,
            label=f"inverse {idx} ({inverse_residual_ratios[idx - 1]:.2f}{suffix})",
        )
    axes[0, 1].set_title("Excitation Objective Histories")
    axes[0, 1].set_xlabel("Objective evaluation")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].legend()

    # Panel 3: received spectra show how the inverse pulse is trying to cancel
    # the target spectrum across the offset band.
    inverse_eval = residual_best.best_evaluation.excitation
    residual = target.mrx + inverse_eval.mrx
    target_norm = trapezoid(np.abs(target.mrx), del_w)
    residual_norm = trapezoid(np.abs(residual), del_w)
    axes[1, 0].plot(del_w, np.abs(target.mrx), label="target")
    axes[1, 0].plot(del_w, np.abs(inverse_eval.mrx), label="inverse")
    axes[1, 0].plot(del_w, np.abs(residual), label="residual")
    axes[1, 0].set_title("Tuned Excitation Spectra (Residual-Best Inverse)")
    axes[1, 0].set_xlabel("Normalized offset")
    axes[1, 0].set_ylabel("Received magnitude")
    axes[1, 0].legend()
    axes[1, 0].text(
        0.02,
        0.95,
        f"residual/target area = {residual_norm / target_norm:.2f}",
        transform=axes[1, 0].transAxes,
        va="top",
    )
    axes[1, 0].text(
        0.02,
        0.86,
        f"objective-best start = {inverse.best_index + 1}, residual-best start = {residual_best_index + 1}",
        transform=axes[1, 0].transAxes,
        va="top",
    )

    # Panel 4: time-domain echo magnitudes are often easier to interpret than
    # complex offset-domain spectra when checking whether a pulse looks sane.
    axes[1, 1].plot(target.tvect, np.abs(target.echo), label="target echo")
    axes[1, 1].plot(inverse_eval.tvect, np.abs(inverse_eval.echo), label="inverse echo")
    axes[1, 1].set_title("Excitation Echoes")
    axes[1, 1].set_xlabel("Normalized time")
    axes[1, 1].set_ylabel("Magnitude")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
        print(f"objective-best inverse start: {inverse.best_index + 1}")
        print(f"residual-best inverse start: {residual_best_index + 1}")
        print(f"best inverse residual/target area: {residual_norm / target_norm:.3f}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
