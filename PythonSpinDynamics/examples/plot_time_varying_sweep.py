"""Plot ideal CPMG sensitivity to time-varying B0 offsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import (
    run_ideal_time_varying_amplitude_sweep,
    sinusoidal_field_waveform,
)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=51, help="Offset grid size.")
    parser.add_argument("--num-echoes", type=int, default=12, help="Number of CPMG echoes.")
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0],
        help="Normalized B0 fluctuation amplitudes.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    waveform = sinusoidal_field_waveform(args.num_echoes)
    result = run_ideal_time_varying_amplitude_sweep(
        amplitudes=args.amplitudes,
        waveform=waveform,
        numpts=args.numpts,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # The waveform is normalized; amplitudes scale it before the CPMG runner
    # applies it echo-by-echo as a time-varying B0 offset.
    axes[0, 0].plot(np.arange(1, args.num_echoes + 1), waveform, marker="o")

    axes[0, 1].plot(result.amplitudes, np.abs(result.matched_signal), marker="o")
    axes[1, 0].plot(result.amplitudes, np.abs(result.echo_integrals), marker="o")

    # Plot final echo magnitudes for each amplitude. The final echo is what the
    # time-varying workflow is designed to compare.
    for idx, amplitude in enumerate(result.amplitudes):
        axes[1, 1].plot(
            result.tvect,
            np.abs(result.echo[idx]),
            label=f"amp={amplitude:g}",
        )

    axes[0, 0].set_title("Normalized B0 Waveform")
    axes[0, 0].set_xlabel("Echo index")
    axes[0, 0].set_ylabel("Relative offset")

    axes[0, 1].set_title("Matched Signal vs Fluctuation")
    axes[0, 1].set_xlabel("Amplitude")
    axes[0, 1].set_ylabel("|matched signal|")

    axes[1, 0].set_title("Final Echo Integral vs Fluctuation")
    axes[1, 0].set_xlabel("Amplitude")
    axes[1, 0].set_ylabel("|echo integral|")

    axes[1, 1].set_title("Final Echo Magnitudes")
    axes[1, 1].set_xlabel("Normalized time")
    axes[1, 1].set_ylabel("|echo|")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
