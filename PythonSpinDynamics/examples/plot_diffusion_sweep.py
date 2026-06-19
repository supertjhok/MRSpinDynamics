"""Plot a compact matched-probe diffusion CPMG Q sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import run_matched_diffusion_q_sweep




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=17, help="Offset grid size.")
    parser.add_argument("--num-echoes", type=int, default=3, help="Number of CPMG echoes.")
    parser.add_argument(
        "--q-values",
        type=float,
        nargs="+",
        default=[20.0, 50.0, 100.0],
        help="Matched-probe Q values. Keep these within the validated compact range.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib()

    result = run_matched_diffusion_q_sweep(
        q_values=args.q_values,
        numpts=args.numpts,
        num_echoes=args.num_echoes,
    )
    echo_numbers = np.arange(1, args.num_echoes + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # Echo integrals summarize the combined diffusion attenuation, probe pulse
    # shape, receiver response, and finite echo-train assembly.
    for idx, q_value in enumerate(result.values):
        axes[0, 0].plot(
            echo_numbers,
            np.abs(result.echo_integrals[idx]),
            marker="o",
            label=f"Q={q_value:g}",
        )
        axes[0, 1].plot(
            result.tvect[0],
            np.abs(result.echo[idx, 0]),
            label=f"Q={q_value:g}, echo 1",
        )

    axes[1, 0].plot(result.values, np.abs(result.echo_integrals[:, -1]), marker="o")
    heat = axes[1, 1].imshow(
        np.abs(result.echo_integrals),
        aspect="auto",
        origin="lower",
        extent=[1, args.num_echoes, result.values[0], result.values[-1]],
    )
    fig.colorbar(heat, ax=axes[1, 1], label="|integral|")

    axes[0, 0].set_title("Diffusion Echo Integral Decay")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("|integral|")
    axes[0, 0].legend()

    axes[0, 1].set_title("First-Echo Time Traces")
    axes[0, 1].set_xlabel("Normalized time")
    axes[0, 1].set_ylabel("|echo|")
    axes[0, 1].legend()

    axes[1, 0].set_title("Last Echo vs Q")
    axes[1, 0].set_xlabel("Q")
    axes[1, 0].set_ylabel("|last echo integral|")

    axes[1, 1].set_title("Echo Integral Heatmap")
    axes[1, 1].set_xlabel("Echo number")
    axes[1, 1].set_ylabel("Q")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
