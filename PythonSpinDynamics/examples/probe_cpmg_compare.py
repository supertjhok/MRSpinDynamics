"""Compare ideal, tuned-probe, untuned-probe, and matched-probe CPMG spectra."""

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


def _peak(echo: np.ndarray) -> complex:
    return echo[int(np.argmax(np.abs(echo)))]


def _print_result(result: CPMGResult) -> None:
    print(f"{result.probe} sum |mrx|: {np.sum(np.abs(result.mrx)):.12g}")
    print(f"{result.probe} peak echo: {_peak(result.echo)}")
    if result.snr is not None:
        print(f"{result.probe} SNR: {result.snr:.12g}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--maxoffs", type=float, default=10.0, help="Offset half-width.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    results = [
        run_ideal_cpmg(args.numpts, args.maxoffs),
        run_tuned_cpmg(args.numpts, args.maxoffs),
        run_untuned_cpmg(args.numpts, args.maxoffs),
        run_matched_cpmg(args.numpts, args.maxoffs),
    ]

    print("CPMG probe comparison")
    print(f"num offsets: {args.numpts}")
    print(f"offset range: {-args.maxoffs:.6g} to {args.maxoffs:.6g}")
    for result in results:
        _print_result(result)

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            **{
                f"{result.probe}_{name}": value
                for result in results
                for name, value in {
                    "del_w": result.del_w,
                    "masy": result.masy,
                    "mrx": result.mrx,
                    "echo": result.echo,
                    "tvect": result.tvect,
                    "snr": np.nan if result.snr is None else result.snr,
                }.items()
            },
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
