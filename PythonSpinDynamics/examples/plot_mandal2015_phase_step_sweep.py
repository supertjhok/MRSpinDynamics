"""Plot CPMG sensitivity to refocusing-pulse absolute phase advance.

This example follows the direct simulation strategy from Mandal 2015: solve the
probe waveform in the rotating frame for each refocusing pulse's absolute RF
phase, discretize the waveform into half-RF-cycle-scale pulse segments, and
feed those matrices through the usual finite CPMG machinery.
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


def _tail_rms(values: np.ndarray, *, tail_fraction: float = 0.5) -> float:
    start = int((1.0 - tail_fraction) * values.size)
    tail = np.asarray(values[start:], dtype=np.float64)
    return float(np.sqrt(np.mean(tail**2)))


def _run_train(
    *,
    probe: str,
    numpts: int,
    num_echoes: int,
    phase_step_cycles: float,
    initial_phase_rad: float,
    phase_bins: int | None,
    auto_refine_grid: bool,
    rephase_safety_factor: float,
    rephase_action: str,
) -> object:
    return run_phase_resolved_probe_case(
        probe=probe,
        numpts=numpts,
        num_echoes=num_echoes,
        phase_step_cycles=phase_step_cycles,
        initial_refocus_phase_rad=initial_phase_rad,
        phase_bins=phase_bins,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )


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
    parser.add_argument("--initial-phase", type=float, default=0.0)
    parser.add_argument(
        "--phase-steps",
        type=float,
        nargs="+",
        default=[0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5],
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
    echo_numbers = np.arange(1, args.num_echoes + 1)

    results = [
        _run_train(
            probe=args.probe,
            numpts=args.numpts,
            num_echoes=args.num_echoes,
            phase_step_cycles=step,
            initial_phase_rad=args.initial_phase,
            phase_bins=args.phase_bins,
            auto_refine_grid=args.auto_refine_grid,
            rephase_safety_factor=args.rephase_safety_factor,
            rephase_action=args.rephase_action,
        )
        for step in args.phase_steps
    ]
    print(f"effective num offsets: {baseline.result.del_w.size}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(
        echo_numbers,
        np.ones_like(echo_numbers, dtype=np.float64),
        color="0.35",
        linewidth=2,
        label="synchronized reference",
    )
    selected_steps = {0.0, 0.0625, 0.25, 0.5}
    for step, result in zip(args.phase_steps, results, strict=True):
        if step not in selected_steps:
            continue
        ratio = np.abs(matched_filter_ratio(result, baseline))
        axes[0, 0].plot(echo_numbers, ratio, marker="o", label=f"{step:g}")

    axes[0, 0].set_title("Echo Attenuation")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("Matched-filter amplitude / synchronized")
    axes[0, 0].legend(title="Phase step")

    tail_ratio = [
        _tail_rms(np.abs(matched_filter_ratio(result, baseline))) for result in results
    ]
    axes[0, 1].plot(args.phase_steps, tail_ratio, marker="o")

    axes[0, 1].set_title("Tail RMS Echo Amplitude")
    axes[0, 1].set_xlabel("Absolute phase advance (cycles/echo)")
    axes[0, 1].set_ylabel("Tail RMS matched-filter amplitude")

    for step, result in zip(args.phase_steps, results, strict=True):
        if step not in selected_steps:
            continue
        axes[1, 0].plot(
            echo_numbers,
            result.refocus_phase_cycles,
            marker="o",
            label=f"{step:g}",
        )
        ratio = matched_filter_ratio(result, baseline)
        axes[1, 1].plot(
            echo_numbers,
            np.abs(ratio),
            marker="o",
            label=f"{step:g}",
        )

    axes[1, 0].set_title("Refocusing Absolute Phase")
    axes[1, 0].set_xlabel("Echo number")
    axes[1, 0].set_ylabel("Cycles modulo 1")
    axes[1, 0].legend(title="Phase step")

    axes[1, 1].set_title("Matched-Filter Magnitude")
    axes[1, 1].set_xlabel("Echo number")
    axes[1, 1].set_ylabel("|filtered echo| / baseline")
    axes[1, 1].legend(title="Phase step")

    fig.suptitle(
        "Mandal 2015 absolute-phase-resolved "
        f"{args.probe} probe pulse shapes"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
