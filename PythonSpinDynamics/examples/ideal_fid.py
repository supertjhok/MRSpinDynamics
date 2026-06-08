"""Run a small ideal FID example using the Python port."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.parameters import set_params_ideal_fid
from spin_dynamics.workflows.fid import sim_fid_ideal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    sp, pp = set_params_ideal_fid(numpts=args.numpts)
    macq, fid, tvect = sim_fid_ideal(sp, pp)

    peak_idx = int(np.argmax(np.abs(fid)))
    print("Ideal FID example")
    print(f"num offsets: {sp.del_w.size}")
    print(f"del_w range: {sp.del_w[0]:.6g} to {sp.del_w[-1]:.6g}")
    print(f"macq shape: {macq.shape}")
    macq_real = np.array2string(np.real(macq[0, :5]), precision=6, separator=", ")
    macq_imag = np.array2string(np.imag(macq[0, :5]), precision=6, separator=", ")
    print(f"macq first 5 real: {macq_real}")
    print(f"macq first 5 imag: {macq_imag}")
    print(f"fid shape: {fid.shape}")
    print(f"peak fid index: {peak_idx}")
    print(f"peak fid time: {tvect[peak_idx]:.12g}")
    print(f"peak fid value: {fid[peak_idx]}")
    print(f"sum |macq|: {np.sum(np.abs(macq)):.12g}")
    print(f"sum |fid|: {np.sum(np.abs(fid)):.12g}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            del_w=sp.del_w,
            macq=macq,
            fid=fid,
            tvect=tvect,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
