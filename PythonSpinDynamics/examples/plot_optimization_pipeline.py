"""Plot the selected-refocusing to excitation/inverse optimization pipeline.

This example is intentionally compact. It demonstrates the workflow handoff:
run a small ideal-v0crit refocusing search, convert it to MATLAB-style result
cells, select the best refocusing pulse, then run tuned excitation and inverse
excitation searches from the reconstructed refocusing axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.optimization import (
    multistart_to_matlab_results,
    run_ideal_v0crit_refocusing_multistart,
    run_tuned_excitation_inverse_pipeline,
    select_matlab_result_program,
    summarize_matlab_results,
)




def _plot_history(ax, result, *, label: str, color: str | None = None) -> None:
    scores = np.asarray(result.history_scores, dtype=np.float64)
    ax.plot(
        np.arange(scores.size),
        scores,
        marker="o",
        linewidth=1.4,
        markersize=4,
        label=label,
        color=color,
    )


def _step_edges(values: np.ndarray, heights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    heights = np.asarray(heights, dtype=np.float64).reshape(-1)
    edges = np.concatenate([[0.0], np.cumsum(values)])
    return edges, np.concatenate([heights, [0.0]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=11, help="Offset grid size.")
    parser.add_argument(
        "--refocusing-segments",
        type=int,
        default=2,
        help="Ideal-v0crit refocusing phase segments.",
    )
    parser.add_argument(
        "--refocusing-starts",
        type=int,
        default=2,
        help="Random starts for the refocusing search.",
    )
    parser.add_argument(
        "--excitation-segments",
        type=int,
        default=2,
        help="Tuned excitation phase segments.",
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
        default=3,
        help="Phase-flipped and perturbed starts for inverse excitation.",
    )
    parser.add_argument(
        "--max-passes",
        type=int,
        default=2,
        help="Pattern-search passes for each single-start optimizer.",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=0.4,
        help="Initial phase-search step in radians.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["auto", "pattern", "scipy"],
        default="pattern",
        help="Backend used by the single-start phase optimizers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=11,
        help="Random seed for reproducible multi-start phases.",
    )
    parser.add_argument(
        "--maxoffs",
        type=float,
        default=10.0,
        help="Normalized offset half-width for reconstructing cell-based axes.",
    )
    parser.add_argument(
        "--free-precession",
        type=float,
        default=1.5,
        help="Free-precession padding stored in MATLAB-style refocusing cells.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    refocusing = run_ideal_v0crit_refocusing_multistart(
        args.refocusing_segments,
        num_starts=args.refocusing_starts,
        seed=args.seed,
        numpts=args.numpts,
        maxoffs=args.maxoffs,
        max_passes=args.max_passes,
        initial_step=args.initial_step,
        optimizer=args.optimizer,
    )
    refocusing_cells = multistart_to_matlab_results(
        refocusing,
        free_precession_t180=args.free_precession,
    )
    refocusing_summary = summarize_matlab_results(refocusing_cells)
    selected = select_matlab_result_program(refocusing_cells)

    pipeline = run_tuned_excitation_inverse_pipeline(
        refocusing_cells,
        excitation_segments=args.excitation_segments,
        excitation_starts=args.excitation_starts,
        inverse_starts=args.inverse_starts,
        seed=args.seed + 100,
        numpts=args.numpts,
        maxoffs=args.maxoffs,
        excitation_kwargs={
            "max_passes": args.max_passes,
            "initial_step": args.initial_step,
            "optimizer": args.optimizer,
        },
        inverse_kwargs={
            "max_passes": args.max_passes,
            "initial_step": args.initial_step,
            "optimizer": args.optimizer,
        },
    )

    target = pipeline.excitation.best_result.best_evaluation
    objective_best = pipeline.inverse.best_result.best_evaluation.excitation
    residual_best = pipeline.residual_best_result.best_evaluation.excitation
    residual = target.mrx + residual_best.mrx

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    program = selected.refocusing
    if program is None:
        raise RuntimeError("selected refocusing result did not include a pulse program")
    edges, x_amp = _step_edges(program.times, program.amplitudes * np.cos(program.phases))
    _edges, y_amp = _step_edges(program.times, program.amplitudes * np.sin(program.phases))
    ax.step(edges, x_amp, where="post", label="I amplitude")
    ax.step(edges, y_amp, where="post", label="Q amplitude")
    ax.set_title("Selected Refocusing Pulse")
    ax.set_xlabel("Time / T180")
    ax.set_ylabel("Normalized amplitude")
    ax.text(
        0.02,
        0.95,
        (
            f"pulse {selected.pulse_number}, score={selected.score:.3g}\n"
            f"best score={refocusing_summary.best_score:.3g}"
        ),
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    _plot_history(ax, pipeline.excitation.best_result, label="target excitation")
    for index, result in enumerate(pipeline.inverse.results, start=1):
        label = f"inverse {index}, residual={pipeline.inverse_residual_ratios[index - 1]:.2f}"
        linewidth = 2.0 if index - 1 == pipeline.residual_best_index else 1.0
        alpha = 1.0 if index - 1 == pipeline.residual_best_index else 0.45
        scores = np.asarray(result.history_scores, dtype=np.float64)
        ax.plot(
            np.arange(scores.size),
            scores,
            marker="o",
            linewidth=linewidth,
            markersize=4,
            alpha=alpha,
            label=label,
        )
    ax.set_title("Excitation and Inverse Histories")
    ax.set_xlabel("Objective evaluation")
    ax.set_ylabel("Score")
    ax.legend(fontsize="small")

    ax = axes[1, 0]
    ax.plot(pipeline.del_w, np.abs(target.mrx), label="target")
    ax.plot(pipeline.del_w, np.abs(objective_best.mrx), label="objective-best inverse")
    ax.plot(pipeline.del_w, np.abs(residual_best.mrx), label="residual-best inverse")
    ax.plot(pipeline.del_w, np.abs(residual), color="black", linestyle="--", label="residual")
    ax.set_title("Received Spectra")
    ax.set_xlabel("Normalized offset")
    ax.set_ylabel("Magnitude")
    ax.text(
        0.02,
        0.95,
        f"best residual/target area = {pipeline.residual_best_ratio:.3g}",
        transform=ax.transAxes,
        va="top",
    )
    ax.legend(fontsize="small")

    ax = axes[1, 1]
    ax.plot(target.tvect, np.abs(target.echo), label="target echo")
    ax.plot(objective_best.tvect, np.abs(objective_best.echo), label="objective-best")
    ax.plot(residual_best.tvect, np.abs(residual_best.echo), label="residual-best")
    ax.set_title("Excitation Echo Magnitudes")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Magnitude")
    ax.legend(fontsize="small")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
        print(f"selected refocusing pulse: {selected.pulse_number}")
        print(f"refocusing best score: {refocusing_summary.best_score:.6g}")
        print(f"residual-best inverse start: {pipeline.residual_best_index + 1}")
        print(f"best inverse residual/target area: {pipeline.residual_best_ratio:.6g}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
