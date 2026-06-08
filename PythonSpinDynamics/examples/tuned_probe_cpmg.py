"""Compare ideal and tuned-probe CPMG spectra using the Python port."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_ideal_cpmg, run_tuned_cpmg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Offset half-width.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    ideal = run_ideal_cpmg(args.numpts, args.maxoffs)
    tuned = run_tuned_cpmg(args.numpts, args.maxoffs)
    ideal_peak = int(np.argmax(np.abs(ideal.echo)))
    tuned_peak = int(np.argmax(np.abs(tuned.echo)))

    print("Ideal vs tuned-probe CPMG")
    print(f"num offsets: {args.numpts}")
    print(f"offset range: {-args.maxoffs:.6g} to {args.maxoffs:.6g}")
    print(f"ideal sum |masy|: {np.sum(np.abs(ideal.masy)):.12g}")
    print(f"ideal peak echo: {ideal.echo[ideal_peak]}")
    print(f"tuned sum |masy|: {np.sum(np.abs(tuned.masy)):.12g}")
    print(f"tuned sum |mrx|: {np.sum(np.abs(tuned.mrx)):.12g}")
    print(f"tuned peak echo: {tuned.echo[tuned_peak]}")
    print(f"tuned SNR: {tuned.snr:.12g}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=ideal.del_w,
            ideal_masy=ideal.masy,
            ideal_echo=ideal.echo,
            ideal_tvect=ideal.tvect,
            tuned_masy=tuned.masy,
            tuned_mrx=tuned.mrx,
            tuned_echo=tuned.echo,
            tuned_tvect=tuned.tvect,
            tuned_snr=tuned.snr,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
