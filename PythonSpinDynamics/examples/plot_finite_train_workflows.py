"""Plot finite CPMG echo-train workflows across probe models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import (
    run_ideal_cpmg_train,
    run_matched_cpmg_train,
    run_tuned_cpmg_train,
    run_untuned_cpmg_train,
)




def _run_train(probe: str, numpts: int, num_echoes: int):
    # Use the public workflow wrappers rather than stitching lower-level
    # acquisition calls together. This keeps the example close to user-facing
    # code and exercises probe pulse shaping plus receiver filtering where
    # applicable.
    common = {"numpts": numpts, "num_echoes": num_echoes, "rephase_action": "ignore"}
    if probe == "ideal":
        return run_ideal_cpmg_train(**common)
    if probe == "tuned":
        return run_tuned_cpmg_train(**common)
    if probe == "untuned":
        return run_untuned_cpmg_train(**common)
    if probe == "matched":
        return run_matched_cpmg_train(**common)
    raise ValueError("unknown probe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=17, help="Offset grid size.")
    parser.add_argument("--num-echoes", type=int, default=4, help="Number of CPMG echoes.")
    parser.add_argument(
        "--probes",
        nargs="+",
        choices=["ideal", "tuned", "untuned", "matched"],
        default=["ideal", "tuned", "untuned", "matched"],
        help="Probe models to compare.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    results = [_run_train(probe, args.numpts, args.num_echoes) for probe in args.probes]
    echo_numbers = np.arange(1, args.num_echoes + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for result in results:
        # Echo integrals are compact scalar summaries of each acquired echo.
        # They are the most convenient values to compare in sweeps or fits.
        axes[0, 0].plot(
            echo_numbers,
            np.abs(result.echo_integrals),
            marker="o",
            label=result.probe,
        )
        axes[0, 1].plot(
            echo_numbers,
            np.real(result.echo_integrals),
            marker="o",
            label=result.probe,
        )
        axes[1, 0].plot(
            result.sequence_time,
            np.abs(result.echo_integrals),
            marker="o",
            label=result.probe,
        )

        # Show one representative offset-domain spectrum: the last acquired echo
        # is often the most sensitive to accumulated relaxation and probe effects.
        axes[1, 1].plot(result.del_w, np.abs(result.mrx[-1]), label=result.probe)

    axes[0, 0].set_title("Echo Integral Magnitude")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("|integral|")
    axes[0, 0].legend()

    axes[0, 1].set_title("Echo Integral Real Part")
    axes[0, 1].set_xlabel("Echo number")
    axes[0, 1].set_ylabel("real(integral)")
    axes[0, 1].legend()

    axes[1, 0].set_title("Echo Decay vs Sequence Time")
    axes[1, 0].set_xlabel("Sequence time (s)")
    axes[1, 0].set_ylabel("|integral|")
    axes[1, 0].legend()

    axes[1, 1].set_title("Last-Echo Received Spectrum")
    axes[1, 1].set_xlabel("Normalized offset")
    axes[1, 1].set_ylabel("|mrx|")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
