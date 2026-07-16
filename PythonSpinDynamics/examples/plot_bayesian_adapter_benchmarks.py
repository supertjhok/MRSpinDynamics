"""Benchmark adaptive, fixed, and prior-ranked protocols for all Phase 2 adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.design import summarize_quantity  # noqa: E402
from spin_dynamics.design.adapter_benchmarks import (  # noqa: E402
    make_phase2_adapter_benchmarks,
)
from spin_dynamics.design.benchmarks import (  # noqa: E402
    run_adapter_benchmark,
    save_benchmark_results,
)


POLICIES = ("adaptive", "fixed", "prior_ranked")
LABELS = {
    "adaptive": "adaptive",
    "fixed": "fixed coverage",
    "prior_ranked": "prior-ranked fixed",
}
COLORS = {"adaptive": "C0", "fixed": "C1", "prior_ranked": "C2"}
STYLES = {"adaptive": "-", "fixed": "--", "prior_ranked": ":"}


def _step_median(problem, result, policy: str, points: int = 100):
    trials = [trial for trial in result.trials if trial.policy == policy]
    maximum = max(trial.physical_seconds for trial in trials)
    grid = np.linspace(0.0, maximum, points)
    prior = summarize_quantity(
        problem.initial_posterior,
        problem.quantity,
        probability=problem.credible_probability,
    )
    initial_width = float(prior.upper[0] - prior.lower[0])
    curves = np.empty((len(trials), points))
    for row, trial in enumerate(trials):
        times = np.array([0.0, *[point.physical_seconds for point in trial.trace]])
        widths = np.array([initial_width, *[point.interval_width for point in trial.trace]])
        indices = np.searchsorted(times, grid, side="right") - 1
        curves[row] = widths[np.clip(indices, 0, widths.size - 1)]
    threshold = float(np.asarray(problem.stopping_rule.maximum_width).reshape(-1)[0])
    normalized = np.median(curves, axis=0) / threshold
    return grid, np.maximum(normalized, 1e-3)


def _success_ecdf(result, policy: str):
    trials = [trial for trial in result.trials if trial.policy == policy]
    reached = np.sort(
        [trial.physical_seconds for trial in trials if trial.reached_target]
    )
    if reached.size == 0:
        return np.array([]), np.array([])
    fraction = np.arange(1, reached.size + 1, dtype=np.float64) / len(trials)
    maximum = max(trial.physical_seconds for trial in trials)
    if reached[-1] < maximum:
        reached = np.append(reached, maximum)
        fraction = np.append(fraction, fraction[-1])
    return reached, fraction


def _print_summary(results) -> None:
    print(
        "adapter                 policy              success  median lab s  "
        "actions      RMSE   coverage  planner s"
    )
    for result in results:
        for summary in result.summaries:
            print(
                f"{result.name:23s} {LABELS[summary.policy]:19s} "
                f"{summary.success_fraction:7.1%}  "
                f"{summary.median_physical_seconds:12.4g}  "
                f"{summary.median_actions:7.1f}  "
                f"{summary.root_mean_square_error:8.3g}  "
                f"{summary.coverage_fraction:8.1%}  "
                f"{summary.median_planning_seconds:9.3g}"
            )
        print(
            f"  exact table build: {result.prediction_table_seconds:.3g} s; "
            f"prior-ranked schedule build: {result.open_loop_schedule_seconds:.3g} s"
        )


def _display_time(name: str) -> tuple[float, str]:
    if name.startswith("NQR"):
        return 1e3, "ms"
    if name.startswith("ESR"):
        return 1e6, "us"
    return 1.0, "s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--utility-samples", type=int, default=24)
    parser.add_argument(
        "--profile", choices=("reference", "smoke"), default="reference"
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    if args.trials <= 0 or args.utility_samples <= 0:
        parser.error("trials and utility-samples must be positive")

    plt = load_matplotlib(headless=args.output is not None)
    problems = make_phase2_adapter_benchmarks(
        profile=args.profile, utility_samples=args.utility_samples
    )
    results = tuple(
        run_adapter_benchmark(problem, trials=args.trials, seed=args.seed + index)
        for index, problem in enumerate(problems)
    )
    _print_summary(results)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4))
    for column, (problem, result) in enumerate(zip(problems, results)):
        top = axes[0, column]
        bottom = axes[1, column]
        time_scale, time_unit = _display_time(problem.name)
        for policy in POLICIES:
            time, width = _step_median(problem, result, policy)
            top.plot(
                time_scale * time,
                width,
                color=COLORS[policy],
                linestyle=STYLES[policy],
                linewidth=2,
                label=LABELS[policy],
            )
            reached_time, fraction = _success_ecdf(result, policy)
            if reached_time.size:
                bottom.step(
                    time_scale * reached_time,
                    fraction,
                    where="post",
                    color=COLORS[policy],
                    linestyle=STYLES[policy],
                    linewidth=2,
                )
        top.axhline(1.0, color="0.25", linewidth=1, alpha=0.6)
        top.set_yscale("log")
        top.set_title(problem.name)
        top.grid(True, alpha=0.2)
        bottom.set_ylim(0.0, 1.02)
        bottom.grid(True, alpha=0.2)
        if column == 0:
            top.set_ylabel("Median interval width / target")
            bottom.set_ylabel("Fraction reaching target")
        top.set_xlabel(f"Accumulated experiment time ({time_unit})")
        bottom.set_xlabel(f"Time to precision target ({time_unit})")
        adaptive = result.summary("adaptive")
        ranked = result.summary("prior_ranked")
        bottom.text(
            0.97,
            0.05,
            f"success: {adaptive.success_fraction:.0%} vs {ranked.success_fraction:.0%}\n"
            f"coverage: {adaptive.coverage_fraction:.0%}",
            transform=bottom.transAxes,
            ha="right",
            va="bottom",
        )
    axes[0, 0].legend(loc="upper right")
    fig.suptitle(
        "Phase 2 adapters: posterior precision versus physical experiment time"
    )
    fig.tight_layout()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        save_benchmark_results(args.json_output, results)
        print(f"saved: {args.json_output}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
