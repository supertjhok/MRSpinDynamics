"""Run a small ideal CPMG example using the Python port."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.cpmg import calc_masy_ideal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    # Build the same ideal spin/probe parameter structures used by the MATLAB
    # `set_params_ideal` reference. `del_w` is the normalized frequency-offset
    # grid, and increasing `numpts` refines that grid.
    sp, pp = set_params_ideal(numpts=args.numpts)

    # `masy` is the complex asymptotic transverse magnetization over offsets.
    # The echo helper Fourier-sums that spectrum onto a normalized time axis.
    masy = calc_masy_ideal(sp, pp)
    echo, tvect = calc_time_domain_echo(masy, sp.del_w)

    # Print compact diagnostics that are stable enough for quick comparisons
    # with validation arrays or MATLAB reference runs.
    peak_idx = int(np.argmax(np.abs(echo)))
    print("Ideal CPMG example")
    print(f"num offsets: {sp.del_w.size}")
    print(f"del_w range: {sp.del_w[0]:.6g} to {sp.del_w[-1]:.6g}")
    print(f"masy shape: {masy.shape}")
    print(f"masy first 5 real: {np.array2string(np.real(masy[:5]), precision=6, separator=', ')}")
    print(f"masy first 5 imag: {np.array2string(np.imag(masy[:5]), precision=6, separator=', ')}")
    print(f"echo shape: {echo.shape}")
    print(f"peak echo index: {peak_idx}")
    print(f"peak echo time: {tvect[peak_idx]:.12g}")
    print(f"peak echo value: {echo[peak_idx]}")
    print(f"sum |masy|: {np.sum(np.abs(masy)):.12g}")
    print(f"sum |echo|: {np.sum(np.abs(echo)):.12g}")

    if args.save_npz is not None:
        # Save arrays in NumPy's native archive format for notebooks, plotting,
        # or direct numerical comparison outside this script.
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=sp.del_w,
            masy=masy,
            echo=echo,
            tvect=tvect,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
