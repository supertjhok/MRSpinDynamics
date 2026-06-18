"""Run a compact pumped NMR maser example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="matched")
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--equilibrium-magnetization", type=float, default=0.8)
    parser.add_argument("--pump-mz", type=float, default=-0.8)
    parser.add_argument("--seed", type=float, default=1e-6)
    parser.add_argument("--t2-trd", type=float, default=20.0)
    parser.add_argument("--t1-trd", type=float, default=6.0)
    parser.add_argument("--duration-trd", type=float, default=40.0)
    parser.add_argument("--points", type=int, default=1001)
    parser.add_argument("--detuning", type=float, default=0.0)
    parser.add_argument("--model", choices=["instant", "circuit"], default="instant")
    parser.add_argument("--save-npz", type=Path, default=None)
    args = parser.parse_args()

    probe = _build_probe(
        args.probe,
        fill_factor=args.fill_factor,
        equilibrium_magnetization=args.equilibrium_magnetization,
        detuning=args.detuning,
    )
    if args.t1_trd <= 0 or args.t2_trd <= 0:
        raise SystemExit("--t1-trd and --t2-trd must be positive")
    if args.points < 2:
        raise SystemExit("--points must be at least 2")

    time = np.linspace(0.0, args.duration_trd * probe.trd, args.points)
    t1 = args.t1_trd * probe.trd
    t2 = args.t2_trd * probe.trd
    result = simulate_nmr_maser(
        time,
        probe,
        seed_mxy=-1j * args.seed,
        initial_mz=args.pump_mz,
        pump_mz=args.pump_mz,
        t1=t1,
        t2=t2,
        model=args.model,
    )
    threshold = probe.trd / t2
    growth = result.envelope[-1] / result.envelope[0]

    print("NMR maser example")
    print(f"probe: {probe.name}")
    print(f"model: {result.model}")
    print(f"Q: {probe.q:.6g}")
    print(f"Trd seconds: {probe.trd:.12g}")
    print(f"T2 / Trd: {args.t2_trd:.12g}")
    print(f"T1 / Trd: {args.t1_trd:.12g}")
    print(f"threshold inversion |mz|: {threshold:.12g}")
    print(f"pump mz: {args.pump_mz:.12g}")
    print(f"initial |mxy|: {result.envelope[0]:.12g}")
    print(f"final |mxy|: {result.envelope[-1]:.12g}")
    print(f"peak |mxy|: {np.max(result.envelope):.12g}")
    print(f"amplitude growth factor: {growth:.12g}")
    print(f"final mz: {result.mz[-1]:.12g}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            time=result.time,
            mxy=result.mxy,
            mz=result.mz,
            feedback=result.feedback,
            threshold_inversion=threshold,
            trd=probe.trd,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
