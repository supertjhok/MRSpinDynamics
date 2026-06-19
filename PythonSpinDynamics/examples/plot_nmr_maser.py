"""Plot threshold behavior for an idealized pumped NMR maser."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.radiation_damping import (
    radiation_damping_probe_from_matched,
    radiation_damping_probe_from_tuned,
    simulate_nmr_maser,
)




def _build_probe(
    probe: str,
    *,
    fill_factor: float,
    equilibrium_magnetization: float,
    detuning: float,
):
    if probe == "matched":
        sp, _pp = set_params_matched_orig(numpts=21)
        return radiation_damping_probe_from_matched(
            sp,
            fill_factor=fill_factor,
            equilibrium_magnetization=equilibrium_magnetization,
            detuning=detuning,
        )
    _params, sp, _pp = set_params_tuned_orig(numpts=21)
    return radiation_damping_probe_from_tuned(
        sp,
        fill_factor=fill_factor,
        equilibrium_magnetization=equilibrium_magnetization,
        detuning=detuning,
    )


def _pump_multipliers(value: str) -> np.ndarray:
    multipliers = np.array([float(part) for part in value.split(",")], dtype=np.float64)
    if multipliers.size == 0:
        raise argparse.ArgumentTypeError("at least one pump multiplier is required")
    if np.any(multipliers <= 0.0):
        raise argparse.ArgumentTypeError("pump multipliers must be positive")
    return multipliers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="matched")
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--equilibrium-magnetization", type=float, default=0.8)
    parser.add_argument("--seed", type=float, default=1e-6)
    parser.add_argument("--t2-trd", type=float, default=20.0)
    parser.add_argument("--t1-trd", type=float, default=6.0)
    parser.add_argument("--duration-trd", type=float, default=45.0)
    parser.add_argument("--points", type=int, default=1200)
    parser.add_argument(
        "--pump-multipliers",
        type=_pump_multipliers,
        default=_pump_multipliers("0.5,2.0,16.0"),
        help=(
            "Comma-separated positive multiples of the threshold inversion. "
            "Values are plotted as negative pump mz levels."
        ),
    )
    parser.add_argument("--detuning", type=float, default=0.0)
    parser.add_argument("--model", choices=["instant", "circuit"], default="instant")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.t1_trd <= 0 or args.t2_trd <= 0:
        raise SystemExit("--t1-trd and --t2-trd must be positive")
    if args.points < 2:
        raise SystemExit("--points must be at least 2")

    plt = load_matplotlib()
    probe = _build_probe(
        args.probe,
        fill_factor=args.fill_factor,
        equilibrium_magnetization=args.equilibrium_magnetization,
        detuning=args.detuning,
    )
    t1 = args.t1_trd * probe.trd
    t2 = args.t2_trd * probe.trd
    threshold = probe.trd / t2
    pump_levels = -args.pump_multipliers * threshold
    time = np.linspace(0.0, args.duration_trd * probe.trd, args.points)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.0), constrained_layout=True)
    for pump_mz in pump_levels:
        result = simulate_nmr_maser(
            time,
            probe,
            seed_mxy=-1j * args.seed,
            initial_mz=float(pump_mz),
            pump_mz=float(pump_mz),
            t1=t1,
            t2=t2,
            model=args.model,
        )
        label = f"pump mz={pump_mz:.3g}"
        time_trd = result.time / probe.trd
        axes[0].semilogy(time_trd, result.envelope, label=label)
        axes[1].plot(time_trd, result.mz, label=label)
        axes[2].plot(time_trd, np.abs(result.feedback) * probe.trd, label=label)

    axes[0].axhline(args.seed, color="0.3", linewidth=0.8, linestyle=":")
    axes[0].set_title("Seed Growth and Saturation")
    axes[0].set_xlabel("Time / Trd")
    axes[0].set_ylabel("|mxy|")
    axes[0].legend()
    axes[1].axhline(-threshold, color="0.3", linewidth=0.8, linestyle=":")
    axes[1].set_title("Longitudinal Inversion")
    axes[1].set_xlabel("Time / Trd")
    axes[1].set_ylabel("mz")
    axes[1].legend()
    axes[2].set_title("Probe Feedback Amplitude")
    axes[2].set_xlabel("Time / Trd")
    axes[2].set_ylabel("|feedback| Trd")
    axes[2].legend()
    fig.suptitle(
        f"NMR maser threshold, {args.probe} probe, "
        f"|mz| threshold={threshold:.3g}, model={args.model}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
