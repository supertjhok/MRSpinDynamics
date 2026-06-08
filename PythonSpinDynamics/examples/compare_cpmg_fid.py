"""Compare the currently validated ideal CPMG and FID workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import set_params_ideal, set_params_ideal_fid
from spin_dynamics.workflows.cpmg import calc_masy_ideal
from spin_dynamics.workflows.fid import sim_fid_ideal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    parser.add_argument("--save-npz", type=Path, default=None, help="Optional output .npz path.")
    args = parser.parse_args()

    sp_cpmg, pp_cpmg = set_params_ideal(numpts=args.numpts)
    masy = calc_masy_ideal(sp_cpmg, pp_cpmg)
    echo_cpmg, tvect_cpmg = calc_time_domain_echo(masy, sp_cpmg.del_w)

    sp_fid, pp_fid = set_params_ideal_fid(numpts=args.numpts)
    macq_fid, fid, tvect_fid = sim_fid_ideal(sp_fid, pp_fid)

    cpmg_peak = int(np.argmax(np.abs(echo_cpmg)))
    fid_peak = int(np.argmax(np.abs(fid)))

    print("Ideal CPMG vs FID")
    print(f"num offsets: {args.numpts}")
    print(f"CPMG masy shape: {masy.shape}")
    print(f"CPMG echo shape: {echo_cpmg.shape}")
    print(f"CPMG peak time: {tvect_cpmg[cpmg_peak]:.12g}")
    print(f"CPMG peak value: {echo_cpmg[cpmg_peak]}")
    print(f"CPMG sum |masy|: {np.sum(np.abs(masy)):.12g}")
    print(f"CPMG sum |echo|: {np.sum(np.abs(echo_cpmg)):.12g}")
    print(f"FID macq shape: {macq_fid.shape}")
    print(f"FID trace shape: {fid.shape}")
    print(f"FID peak time: {tvect_fid[fid_peak]:.12g}")
    print(f"FID peak value: {fid[fid_peak]}")
    print(f"FID sum |macq|: {np.sum(np.abs(macq_fid)):.12g}")
    print(f"FID sum |trace|: {np.sum(np.abs(fid)):.12g}")

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            cpmg_del_w=sp_cpmg.del_w,
            cpmg_masy=masy,
            cpmg_echo=echo_cpmg,
            cpmg_tvect=tvect_cpmg,
            fid_del_w=sp_fid.del_w,
            fid_macq=macq_fid,
            fid_trace=fid,
            fid_tvect=tvect_fid,
        )
        print(f"saved: {args.save_npz}")


if __name__ == "__main__":
    main()
