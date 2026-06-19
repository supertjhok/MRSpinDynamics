"""Plot a compact WURST inversion and matched-probe CPMG workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import (
    run_ideal_wurst_inversion,
    run_matched_wurst_cpmg,
    run_matched_wurst_inversion,
)




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=41, help="Offset grid size.")
    parser.add_argument("--num-steps", type=int, default=64, help="WURST pulse segments.")
    parser.add_argument(
        "--sweep-width",
        type=float,
        default=20.0,
        help="Total WURST sweep width normalized to nominal w1.",
    )
    parser.add_argument(
        "--duration-us",
        type=float,
        default=500.0,
        help="WURST pulse duration in microseconds.",
    )
    parser.add_argument("--num-echoes", type=int, default=2, help="Matched CPMG echoes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()
    duration_seconds = args.duration_us * 1e-6

    ideal = run_ideal_wurst_inversion(
        numpts=args.numpts,
        num_steps=args.num_steps,
        duration_seconds=duration_seconds,
        sweep_width_normalized=args.sweep_width,
    )
    matched = run_matched_wurst_inversion(
        numpts=args.numpts,
        num_steps=args.num_steps,
        duration_seconds=duration_seconds,
        sweep_width_normalized=args.sweep_width,
    )
    cpmg = run_matched_wurst_cpmg(
        num_echoes=args.num_echoes,
        numpts=args.numpts,
        num_steps=max(8, args.num_steps // 2),
        duration_seconds=duration_seconds,
        sweep_width_normalized=args.sweep_width,
        rephase_action="ignore",
    )

    pulse_time_us = 1e6 * np.cumsum(ideal.pulse.duration)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    axes[0, 0].plot(pulse_time_us, ideal.pulse.amplitude, label="amplitude")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("Normalized amplitude")
    ax_freq = axes[0, 0].twinx()
    ax_freq.plot(
        pulse_time_us,
        ideal.pulse.frequency_offset / (2 * np.pi * 1e3),
        color="tab:red",
        alpha=0.75,
        label="frequency",
    )
    ax_freq.set_ylabel("Offset (kHz)")
    axes[0, 0].set_title("WURST Pulse")

    axes[0, 1].plot(ideal.del_w, ideal.mz, label="ideal")
    axes[0, 1].plot(matched.del_w, matched.mz, label="matched")
    axes[0, 1].set_xlabel("Normalized offset")
    axes[0, 1].set_ylabel("Mz after pulse")
    axes[0, 1].set_title("Inversion Profile")
    axes[0, 1].legend()

    axes[1, 0].plot(1e6 * matched.rotating_time, np.real(matched.rotating_current), label="real")
    axes[1, 0].plot(1e6 * matched.rotating_time, np.imag(matched.rotating_current), label="imag")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel("Rotating-frame current")
    axes[1, 0].set_title("Matched-Probe WURST Response")
    axes[1, 0].legend()

    for echo_idx in range(args.num_echoes):
        axes[1, 1].plot(
            cpmg.tvect,
            np.real(cpmg.echo[echo_idx]),
            label=f"echo {echo_idx + 1}",
        )
    axes[1, 1].set_xlabel("Normalized acquisition time")
    axes[1, 1].set_ylabel("Real echo")
    axes[1, 1].set_title("Matched WURST-CPMG Echoes")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
