"""Export compact arrays from the currently validated Python workflows."""

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
    parser.add_argument("output", type=Path, help="Output .npz file.")
    parser.add_argument("--numpts", type=int, default=101, help="Number of offset points.")
    args = parser.parse_args()

    sp_cpmg, pp_cpmg = set_params_ideal(numpts=args.numpts)
    masy = calc_masy_ideal(sp_cpmg, pp_cpmg)
    echo_cpmg, tvect_cpmg = calc_time_domain_echo(masy, sp_cpmg.del_w)

    sp_fid, pp_fid = set_params_ideal_fid(numpts=args.numpts)
    macq_fid, fid, tvect_fid = sim_fid_ideal(sp_fid, pp_fid)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        cpmg_del_w=sp_cpmg.del_w,
        cpmg_masy=masy,
        cpmg_echo=echo_cpmg,
        cpmg_tvect=tvect_cpmg,
        fid_del_w=sp_fid.del_w,
        fid_macq=macq_fid,
        fid_trace=fid,
        fid_tvect=tvect_fid,
    )
    print(f"saved: {args.output}")
    print(f"num offsets: {args.numpts}")
    print(f"CPMG echo samples: {echo_cpmg.size}")
    print(f"FID trace samples: {fid.size}")


if __name__ == "__main__":
    main()
