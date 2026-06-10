"""Plot the currently validated ideal CPMG and FID workflows."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import set_params_ideal, set_params_ideal_fid
from spin_dynamics.workflows.cpmg import calc_masy_ideal
from spin_dynamics.workflows.fid import sim_fid_ideal


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this example. Install the optional "
            "plot dependency, for example: pip install matplotlib"
        ) from exc
    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=201, help="Number of offset points.")
    parser.add_argument(
        "--fid-maxoffs",
        type=float,
        default=2.0,
        help="FID offset half-width for plotting. Use 10 for MATLAB defaults.",
    )
    parser.add_argument(
        "--raw-fid-scale",
        action="store_true",
        help="Plot the raw FID trace amplitude instead of normalizing it.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output image path.")
    args = parser.parse_args()

    plt = _load_matplotlib()

    # CPMG plot data: offset-domain asymptotic magnetization and its
    # time-domain echo.
    sp_cpmg, pp_cpmg = set_params_ideal(numpts=args.numpts)
    masy = calc_masy_ideal(sp_cpmg, pp_cpmg)
    echo_cpmg, tvect_cpmg = calc_time_domain_echo(masy, sp_cpmg.del_w)

    # FID defaults span a wider offset range than is visually helpful here, so
    # the plot uses a narrower range unless `--fid-maxoffs 10` is requested.
    sp_fid, pp_fid = set_params_ideal_fid(numpts=args.numpts)
    fid_del_w = np.linspace(-args.fid_maxoffs, args.fid_maxoffs, args.numpts)
    sp_fid = replace(sp_fid, maxoffs=args.fid_maxoffs, del_w=fid_del_w)
    macq_fid, fid, tvect_fid = sim_fid_ideal(sp_fid, pp_fid)
    fid_plot = fid if args.raw_fid_scale else fid / np.max(np.abs(fid))

    # The four panels line up the two domains for both workflows:
    # offset spectra on the left/top pair, time-domain traces on the right.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axes[0, 0].plot(sp_cpmg.del_w, np.real(masy), label="real")
    axes[0, 0].plot(sp_cpmg.del_w, np.imag(masy), label="imag")
    axes[0, 0].set_title("CPMG Asymptotic Magnetization")
    axes[0, 0].set_xlabel("Normalized offset")
    axes[0, 0].legend()

    axes[0, 1].plot(tvect_cpmg, np.real(echo_cpmg), label="real")
    axes[0, 1].plot(tvect_cpmg, np.abs(echo_cpmg), label="magnitude")
    axes[0, 1].set_title("CPMG Echo")
    axes[0, 1].set_xlabel("Normalized time")
    axes[0, 1].legend()

    axes[1, 0].plot(sp_fid.del_w, np.real(macq_fid[0]), label="real")
    axes[1, 0].plot(sp_fid.del_w, np.imag(macq_fid[0]), label="imag")
    axes[1, 0].set_title("FID Acquired Spectrum")
    axes[1, 0].set_xlabel("Normalized offset")
    axes[1, 0].legend()

    axes[1, 1].plot(tvect_fid, np.real(fid_plot), label="real")
    axes[1, 1].plot(tvect_fid, np.imag(fid_plot), label="imag")
    axes[1, 1].plot(tvect_fid, np.abs(fid_plot), label="magnitude")
    axes[1, 1].set_title("FID Trace")
    axes[1, 1].set_xlabel("Normalized time")
    axes[1, 1].set_ylabel("Normalized amplitude" if not args.raw_fid_scale else "Amplitude")
    axes[1, 1].legend()

    if args.output is not None:
        # Saving is useful for docs or CI artifacts; otherwise show interactively.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
