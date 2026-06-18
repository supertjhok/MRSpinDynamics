"""Compare finite CPMG trains with and without radiation damping."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_cpmg_train, run_tuned_cpmg_train


def _runner(probe: str):
    if probe == "matched":
        return run_matched_cpmg_train
    return run_tuned_cpmg_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="tuned")
    parser.add_argument("--numpts", type=int, default=21)
    parser.add_argument("--num-echoes", type=int, default=4)
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--equilibrium-magnetization", type=float, default=0.8)
    parser.add_argument("--model", choices=["instant", "circuit"], default="instant")
    parser.add_argument("--save-npz", type=Path, default=None)
    args = parser.parse_args()

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
        },
    )
    if damped.radiation_damping is None:
        raise RuntimeError("radiation damping was not enabled")

    delta = damped.mrx - clean.mrx
    clean_peak = np.max(np.abs(clean.echo), axis=1)
    damped_peak = np.max(np.abs(damped.echo), axis=1)

    print("Radiation damping CPMG train")
    print(f"probe: {args.probe}")
    print(f"model: {args.model}")
    print(f"num offsets: {clean.del_w.size}")
    print(f"num echoes: {clean.mrx.shape[0]}")
    print(f"Trd normalized: {damped.radiation_damping.trd:.12g}")
    print(f"Trd seconds: {damped.radiation_damping.probe.trd:.12g}")
    print(f"max |delta mrx|: {np.max(np.abs(delta)):.12g}")
    print(f"clean echo peaks: {np.array2string(clean_peak, precision=6, separator=', ')}")
    print(f"damped echo peaks: {np.array2string(damped_peak, precision=6, separator=', ')}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=clean.del_w,
            clean_mrx=clean.mrx,
            damped_mrx=damped.mrx,
            clean_echo=clean.echo,
            damped_echo=damped.echo,
            tvect=clean.tvect,
            sequence_time=clean.sequence_time,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
