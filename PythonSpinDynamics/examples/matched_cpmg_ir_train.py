"""Run a compact matched-probe CPMG-IR finite echo train."""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_cpmg_ir_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=21)
    parser.add_argument("--num-echoes", type=int, default=4)
    parser.add_argument("--num-tau", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tau-workers", type=int, default=1)
    args = parser.parse_args()

    tauvect = np.linspace(0.5e-3, 4e-3, args.num_tau)
    result = run_matched_cpmg_ir_train(
        num_echoes=args.num_echoes,
        tauvect=tauvect,
        numpts=args.numpts,
        num_workers=args.workers,
        tau_workers=args.tau_workers,
        rephase_action="ignore",
    )
    peak = np.unravel_index(abs(result.echo_integrals).argmax(), result.echo_integrals.shape)
    print("Matched CPMG-IR finite train")
    print(f"num offsets: {result.del_w.size}")
    print(f"num tau: {result.tauvect.size}")
    print(f"num echoes: {result.sequence_time.size}")
    print(f"mrx shape: {result.mrx.shape}")
    print(f"echo shape: {result.echo.shape}")
    print(f"peak tau: {result.tauvect[peak[0]]:.6g}")
    print(f"peak echo time: {result.sequence_time[peak[1]]:.6g}")
    print(f"peak integral: {result.echo_integrals[peak]:.6g}")


if __name__ == "__main__":
    main()
