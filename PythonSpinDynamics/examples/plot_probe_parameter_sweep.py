"""Plot a tuned or matched CPMG probe-parameter sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (  # noqa: E402
    CPMGParameterSweepResult,
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
)


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this example. Install the optional "
            "plot dependency, for example: pip install matplotlib"
        ) from exc
    return plt


def _parse_values(text: str) -> np.ndarray:
    values = np.asarray([float(part) for part in text.split(",") if part.strip()])
    if values.size == 0:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _run_sweep(args: argparse.Namespace) -> CPMGParameterSweepResult:
    if args.sweep == "q":
        q_values = args.values
        if q_values is None:
            start, stop, count = args.range
            q_values = np.linspace(start, stop, int(count))
        if args.probe == "tuned":
            return run_tuned_q_sweep(
                q_values=q_values,
                numpts=args.numpts,
                maxoffs=args.maxoffs,
                num_workers=args.workers,
            )
        return run_matched_q_sweep(
            q_values=q_values,
            numpts=args.numpts,
            maxoffs=args.maxoffs,
            num_workers=args.workers,
        )

    offsets = args.values
    if offsets is None:
        start, stop, count = args.range
        offsets = np.linspace(start, stop, int(count))
    if args.probe == "tuned":
        return run_tuned_mistuning_sweep(
            offsets=offsets,
            numpts=args.numpts,
            maxoffs=args.maxoffs,
            num_workers=args.workers,
        )
    return run_matched_mistuning_sweep(
        offsets=offsets,
        numpts=args.numpts,
        maxoffs=args.maxoffs,
        num_workers=args.workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="tuned")
    parser.add_argument("--sweep", choices=["q", "mistuning"], default="q")
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Offset half-width.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel sweep-point workers.")
    parser.add_argument(
        "--values",
        type=_parse_values,
        default=None,
        help="Comma-separated sweep values. Overrides --range.",
    )
    parser.add_argument(
        "--range",
        type=float,
        nargs=3,
        metavar=("START", "STOP", "COUNT"),
        default=None,
        help="Sweep range as START STOP COUNT.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path.")
    args = parser.parse_args()

    if args.range is None:
        args.range = (1.0, 100.0, 25.0) if args.sweep == "q" else (-5.0, 5.0, 25.0)
    if args.range[2] <= 0:
        raise SystemExit("range COUNT must be positive")

    plt = _load_matplotlib()
    result = _run_sweep(args)
    order = np.argsort(result.values)
    y = result.values[order]
    mrx = result.mrx[order]
    echo_data = result.echo[order]
    tvect = result.tvect / np.pi

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    spec = axes[0].imshow(
        np.abs(mrx),
        aspect="auto",
        origin="lower",
        extent=[result.del_w[0], result.del_w[-1], y[0], y[-1]],
    )
    axes[0].set_title("Received Spectrum")
    axes[0].set_xlabel("Normalized offset")
    axes[0].set_ylabel(result.value_label)
    fig.colorbar(spec, ax=axes[0], label="Magnitude")

    echo = axes[1].imshow(
        np.abs(echo_data),
        aspect="auto",
        origin="lower",
        extent=[tvect[0], tvect[-1], y[0], y[-1]],
    )
    axes[1].set_title("Echo")
    axes[1].set_xlabel("Time / pi")
    axes[1].set_ylabel(result.value_label)
    fig.colorbar(echo, ax=axes[1], label="Magnitude")

    axes[2].plot(result.values, result.snr, marker="o")
    axes[2].set_title("SNR")
    axes[2].set_xlabel(result.value_label)
    axes[2].set_ylabel("SNR")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{result.probe.capitalize()} {result.sweep} sweep")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
