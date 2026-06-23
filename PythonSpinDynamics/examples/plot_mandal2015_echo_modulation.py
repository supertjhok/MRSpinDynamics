"""Plot echo modulation from absolute-phase-resolved probe pulse shapes.

Each refocusing pulse shape is solved in the rotating frame at its actual
absolute RF phase, discretized into small pulse segments, and passed through
the standard finite CPMG spin-dynamics machinery.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from _mandal2015_absolute_phase import (  # noqa: E402
    matched_filter_ratio,
    run_phase_resolved_probe_case,
)


def _case_label(step_cycles: float) -> str:
    return f"{step_cycles:g} cycles/echo"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "untuned", "matched"], default="tuned")
    parser.add_argument("--numpts", type=int, default=17)
    parser.add_argument("--num-echoes", type=int, default=64)
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=None,
        help="Optional number of absolute-phase bins used to reuse pulse solves.",
    )
    parser.add_argument(
        "--phase-steps",
        type=float,
        nargs="+",
        default=[0.0, 0.125, 0.25],
        help="RF absolute-phase advances per refocusing pulse, in cycles.",
    )
    parser.add_argument(
        "--no-auto-refine-grid",
        dest="auto_refine_grid",
        action="store_false",
        help="Keep the requested numpts even when the offset grid may rephase.",
    )
    parser.add_argument(
        "--rephase-safety-factor",
        type=float,
        default=1.25,
        help="Safety factor for the discrete-offset rephasing check.",
    )
    parser.add_argument(
        "--rephase-action",
        choices=["warn", "ignore", "raise"],
        default="raise",
        help="Action if auto-refinement is disabled and the grid may rephase.",
    )
    parser.add_argument("--initial-phase", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.set_defaults(auto_refine_grid=True)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    baseline = run_phase_resolved_probe_case(
        probe=args.probe,
        numpts=args.numpts,
        num_echoes=args.num_echoes,
        phase_step_cycles=0.0,
        initial_refocus_phase_rad=args.initial_phase,
        phase_bins=args.phase_bins,
        auto_refine_grid=args.auto_refine_grid,
        rephase_safety_factor=args.rephase_safety_factor,
        rephase_action=args.rephase_action,
    )
    results = []
    for step_cycles in args.phase_steps:
        results.append(
            run_phase_resolved_probe_case(
                probe=args.probe,
                numpts=args.numpts,
                num_echoes=args.num_echoes,
                phase_step_cycles=step_cycles,
                initial_refocus_phase_rad=args.initial_phase,
                phase_bins=args.phase_bins,
                auto_refine_grid=args.auto_refine_grid,
                rephase_safety_factor=args.rephase_safety_factor,
                rephase_action=args.rephase_action,
            )
        )
    print(f"effective num offsets: {baseline.result.del_w.size}")

    echo_numbers = np.arange(1, args.num_echoes + 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(
        echo_numbers,
        np.ones_like(echo_numbers, dtype=np.float64),
        color="0.35",
        linewidth=2,
        label="baseline",
    )
    for step_cycles, result in zip(args.phase_steps, results, strict=True):
        label = _case_label(step_cycles)
        ratio = matched_filter_ratio(result, baseline)
        phase_driver = result.refocus_phase_cycles

        axes[0, 0].plot(
            echo_numbers,
            np.abs(ratio),
            marker="o",
            label=label,
        )
        axes[0, 1].plot(
            echo_numbers,
            phase_driver,
            marker="o",
            label=label,
        )
        axes[1, 0].plot(echo_numbers, np.real(ratio), label=f"{label} real")
        axes[1, 0].plot(echo_numbers, np.imag(ratio), linestyle="--")
        if step_cycles in {args.phase_steps[0], args.phase_steps[-1]}:
            spectrum = np.abs(result.result.mrx[-1])
            norm = np.max(np.abs(baseline.result.mrx[-1]))
            axes[1, 1].plot(result.result.del_w, spectrum / norm, label=label)

    axes[1, 1].plot(
        baseline.result.del_w,
        np.abs(baseline.result.mrx[-1])
        / np.max(np.abs(baseline.result.mrx[-1])),
        color="0.35",
        linewidth=2,
        label="baseline",
    )

    axes[0, 0].set_title("Echo Envelope")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("Matched-filter amplitude / baseline")
    axes[0, 0].legend()

    axes[0, 1].set_title("Refocusing Absolute Phase")
    axes[0, 1].set_xlabel("Echo number")
    axes[0, 1].set_ylabel("Cycles modulo 1")
    axes[0, 1].legend()

    axes[1, 0].set_title("Matched-Filter Echo Ratio")
    axes[1, 0].set_xlabel("Echo number")
    axes[1, 0].set_ylabel("perturbed / baseline")
    axes[1, 0].legend(fontsize="small", ncol=2)

    axes[1, 1].set_title("Last-Echo Offset Spectrum")
    axes[1, 1].set_xlabel("Normalized offset")
    axes[1, 1].set_ylabel("|mrx| / baseline max")
    axes[1, 1].legend()

    fig.suptitle(
        "Absolute-phase-driven CPMG modulation "
        f"({args.probe} probe pulse shapes)"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
