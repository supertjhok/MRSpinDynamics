"""Run a finite matched-probe CPMG echo-train example using the Python port."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_matched_cpmg_train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--num-echoes", type=int, default=8, help="Number of echoes.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Maximum normalized offset.")
    parser.add_argument("--t1", type=float, default=2.0, help="T1 in seconds.")
    parser.add_argument("--t2", type=float, default=2.0, help="T2 in seconds.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    # The matched-probe runner includes the matching-network response and the
    # same finite echo-train acquisition surface as the ideal/tuned examples.
    result = run_matched_cpmg_train(
        numpts=args.numpts,
        maxoffs=args.maxoffs,
        num_echoes=args.num_echoes,
        t1_seconds=args.t1,
        t2_seconds=args.t2,
    )

    # Shapes follow (echo, offset) for `mrx` and (echo, time) for `echo`.
    # The matched path uses practical tolerances in validation because its
    # nonlinear solve is independent of MATLAB's toolbox implementation.
    peak = np.max(np.abs(result.echo), axis=1)
    print("Finite matched-probe CPMG train")
    print(f"num offsets: {result.del_w.size}")
    print(f"num echoes: {result.mrx.shape[0]}")
    print(f"mrx shape: {result.mrx.shape}")
    print(f"echo shape: {result.echo.shape}")
    print(f"tvect range: {result.tvect[0]:.6g} to {result.tvect[-1]:.6g}")
    print(f"peak echo magnitudes: {np.array2string(peak, precision=6, separator=', ')}")
    print(
        "echo integrals real: "
        f"{np.array2string(np.real(result.echo_integrals), precision=6, separator=', ')}"
    )

    if args.save_npz is not None:
        # Keep the saved archive complete enough for plotting without rerunning
        # the matched-probe transient calculation.
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=result.del_w,
            mrx=result.mrx,
            echo=result.echo,
            tvect=result.tvect,
            echo_integrals=result.echo_integrals,
            sequence_time=result.sequence_time,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
