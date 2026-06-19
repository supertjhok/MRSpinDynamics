"""Plot ideal TANGO-B J-filter selectivity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.coupling import tango_b_filter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=160.0, help="Target J coupling in Hz.")
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Odd TANGO-B filter orders to display.",
    )
    parser.add_argument("--min-j", type=float, default=100.0, help="Minimum J value in Hz.")
    parser.add_argument("--max-j", type=float, default=180.0, help="Maximum J value in Hz.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    couplings = np.linspace(args.min_j, args.max_j, 801)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for order in args.orders:
        response = tango_b_filter(couplings, target_coupling_hz=args.target, order=order)
        axes[0].plot(couplings, response, label=f"n={order}")

    markers = np.array([125.0, 150.0, 160.0])
    marker_response = {
        order: tango_b_filter(markers, target_coupling_hz=args.target, order=order)
        for order in args.orders
    }

    axes[0].axvline(125.0, color="tab:blue", linestyle="--", alpha=0.45)
    axes[0].axvline(160.0, color="tab:orange", linestyle="--", alpha=0.45)
    axes[0].set_title("Ideal TANGO-B Filter Response")
    axes[0].set_xlabel("J coupling (Hz)")
    axes[0].set_ylabel("Relative transverse amplitude")
    axes[0].legend(title="order")

    width = 0.8 / max(len(args.orders), 1)
    x = np.arange(markers.size)
    for idx, order in enumerate(args.orders):
        axes[1].bar(
            x + (idx - (len(args.orders) - 1) / 2) * width,
            marker_response[order],
            width=width,
            label=f"n={order}",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{value:g} Hz" for value in markers])
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Selected Coupling Classes")
    axes[1].set_ylabel("Relative amplitude")
    axes[1].legend(title="order")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
