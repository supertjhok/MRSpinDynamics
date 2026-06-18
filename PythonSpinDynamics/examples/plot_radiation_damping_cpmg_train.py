"""Plot clean and radiation-damped finite CPMG train results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_cpmg_train, run_tuned_cpmg_train


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this example. Install the optional "
            "plot dependency, for example: pip install matplotlib"
        ) from exc
    return plt


def _runner(probe: str):
    if probe == "matched":
        return run_matched_cpmg_train
    return run_tuned_cpmg_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="tuned")
    parser.add_argument("--numpts", type=int, default=31)
    parser.add_argument("--num-echoes", type=int, default=6)
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--equilibrium-magnetization", type=float, default=0.8)
    parser.add_argument("--model", choices=["instant", "circuit"], default="instant")
    parser.add_argument(
        "--apply-during-pulses",
        action="store_true",
        help="Interleave radiation damping with pulse matrices.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = _load_matplotlib()
    runner = _runner(args.probe)
    common = {
        "numpts": args.numpts,
        "num_echoes": args.num_echoes,
        "rephase_action": "ignore",
    }
    clean = runner(**common)
    damped = runner(
        **common,
        radiation_damping={
            "fill_factor": args.fill_factor,
            "equilibrium_magnetization": args.equilibrium_magnetization,
            "model": args.model,
            "apply_during_pulses": args.apply_during_pulses,
        },
    )
    if damped.radiation_damping is None:
        raise RuntimeError("radiation damping was not enabled")

    echo_index = min(1, clean.echo.shape[0] - 1)
    clean_peak = np.max(np.abs(clean.echo), axis=1)
    damped_peak = np.max(np.abs(damped.echo), axis=1)
    ratio = np.divide(
        damped.echo_integrals,
        clean.echo_integrals,
        out=np.zeros_like(damped.echo_integrals),
        where=np.abs(clean.echo_integrals) > 0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(clean.del_w, np.abs(clean.mrx[echo_index]), label="clean")
    axes[0, 0].plot(clean.del_w, np.abs(damped.mrx[echo_index]), label="RD")
    axes[0, 0].set_title(f"Echo {echo_index + 1} Spectrum")
    axes[0, 0].set_xlabel("Normalized offset")
    axes[0, 0].set_ylabel("|mrx|")
    axes[0, 0].legend()

    axes[0, 1].plot(clean.tvect, np.abs(clean.echo[echo_index]), label="clean")
    axes[0, 1].plot(clean.tvect, np.abs(damped.echo[echo_index]), label="RD")
    axes[0, 1].set_title(f"Echo {echo_index + 1} Time Trace")
    axes[0, 1].set_xlabel("Normalized time")
    axes[0, 1].set_ylabel("|echo|")
    axes[0, 1].legend()

    axes[1, 0].plot(clean.sequence_time, clean_peak, marker="o", label="clean")
    axes[1, 0].plot(clean.sequence_time, damped_peak, marker="o", label="RD")
    axes[1, 0].set_title("Echo Peak Train")
    axes[1, 0].set_xlabel("Sequence time (s)")
    axes[1, 0].set_ylabel("Peak |echo|")
    axes[1, 0].legend()

    axes[1, 1].plot(
        clean.sequence_time,
        np.abs(ratio),
        marker="o",
        color="tab:red",
    )
    axes[1, 1].set_title("RD / Clean Echo Integral")
    axes[1, 1].set_xlabel("Sequence time (s)")
    axes[1, 1].set_ylabel("Magnitude ratio")

    pulse_text = "with pulse RD" if args.apply_during_pulses else "free-window RD"
    fig.suptitle(
        f"{args.probe.capitalize()} CPMG, {args.model} {pulse_text}, "
        f"Trd={damped.radiation_damping.probe.trd:.3g} s"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
