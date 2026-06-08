"""Plot ideal, tuned, untuned, and matched CPMG workflow results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import (
    CPMGResult,
    run_ideal_cpmg,
    run_matched_cpmg,
    run_tuned_cpmg,
    run_untuned_cpmg,
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


def _label(result: CPMGResult) -> str:
    return f"{result.probe} (SNR {result.snr:.3g})" if result.snr is not None else result.probe


def _masy_component(result: CPMGResult, component: str) -> np.ndarray:
    if component == "real":
        return np.real(result.masy)
    if component == "imag":
        return np.imag(result.masy)
    if component == "phase":
        return np.angle(result.masy)
    return np.abs(result.masy)


def _masy_ylabel(component: str) -> str:
    labels = {
        "real": "Real component",
        "imag": "Imaginary component",
        "phase": "Phase (rad)",
        "magnitude": "Magnitude",
    }
    return labels[component]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Offset half-width.")
    parser.add_argument(
        "--masy-component",
        choices=["magnitude", "real", "imag", "phase"],
        default="magnitude",
        help="Asymptotic magnetization component shown in the top-left panel.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path.")
    args = parser.parse_args()

    plt = _load_matplotlib()
    results = [
        run_ideal_cpmg(args.numpts, args.maxoffs),
        run_tuned_cpmg(args.numpts, args.maxoffs),
        run_untuned_cpmg(args.numpts, args.maxoffs),
        run_matched_cpmg(args.numpts, args.maxoffs),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for result in results:
        axes[0, 0].plot(
            result.del_w,
            _masy_component(result, args.masy_component),
            label=result.probe,
        )
        axes[0, 1].plot(result.del_w, np.abs(result.mrx), label=_label(result))
        axes[1, 0].plot(result.tvect, np.abs(result.echo), label=result.probe)
        peak = np.max(np.abs(result.echo))
        norm_echo = np.abs(result.echo) / peak if peak > 0 else np.abs(result.echo)
        axes[1, 1].plot(result.tvect, norm_echo, label=result.probe)

    axes[0, 0].set_title("Asymptotic Magnetization")
    axes[0, 0].set_xlabel("Normalized offset")
    axes[0, 0].set_ylabel(_masy_ylabel(args.masy_component))
    axes[0, 0].legend()

    axes[0, 1].set_title("Received Spectrum")
    axes[0, 1].set_xlabel("Normalized offset")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].legend()

    axes[1, 0].set_title("Echo Magnitude")
    axes[1, 0].set_xlabel("Normalized time")
    axes[1, 0].set_ylabel("Magnitude")
    axes[1, 0].legend()

    axes[1, 1].set_title("Normalized Echo Magnitude")
    axes[1, 1].set_xlabel("Normalized time")
    axes[1, 1].set_ylabel("Normalized magnitude")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
