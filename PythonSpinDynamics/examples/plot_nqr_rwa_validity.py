"""Map where the single-band RWA agrees with exact lab-frame RF dynamics.

The two panels use the same NaNO2 and NaClO3 sites as the static crossover
example.  Color reports the density-matrix response error after a short pulse;
contours mark 1% and 10% error.  The dashed curve is the normalized spacing to
the nearest distinct RF-active line, a useful multiband warning scale.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import QuadrupolarSite, scan_rwa_validity  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"


def _nano2_site() -> QuadrupolarSite:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                return QuadrupolarSite(
                    spin=1.0,
                    isotope="14N",
                    label="NaNO2 N",
                    quadrupole_frequency_hz=0.75
                    * float(row["mean_abs_cq_mhz"])
                    * 1.0e6,
                    eta=float(row["mean_eta"]),
                    gamma_hz_per_t=3.0766e6,
                )
    raise RuntimeError(f"could not find NaNO2 14N parameters in {NANO2_SUMMARY}")


def _naclo3_site() -> QuadrupolarSite:
    return QuadrupolarSite(
        spin=1.5,
        isotope="35Cl",
        label="NaClO3 Cl",
        quadrupole_frequency_hz=30.656e6,
        eta=0.0,
        gamma_hz_per_t=4.1717e6,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-points", type=int, default=25)
    parser.add_argument("--rf-points", type=int, default=18)
    parser.add_argument("--samples-per-cycle", type=int, default=60)
    parser.add_argument("--pulse-cycles", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.interaction_points < 3 or args.rf_points < 3:
        raise ValueError("interaction-points and rf-points must be at least three")
    interaction = np.logspace(-2.0, 1.5, args.interaction_points)
    rf_strength = np.logspace(-3.0, -0.7, args.rf_points)
    sites = (_nano2_site(), _naclo3_site())
    labels = (r"NaNO$_2$ $^{14}$N ($I=1$)", r"NaClO$_3$ $^{35}$Cl ($I=3/2$)")

    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0), constrained_layout=True)
    mesh = None
    for axis, site, label in zip(axes, sites, labels, strict=True):
        validity = scan_rwa_validity(
            site,
            interaction,
            rf_strength,
            b0_direction_pas=(1.0, 1.0, 1.0),
            b1_direction_pas=(1.0, -1.0, 0.0),
            duration_in_carrier_cycles=args.pulse_cycles,
            samples_per_carrier_cycle=args.samples_per_cycle,
        )
        error_percent = 100.0 * validity.relative_response_error.T
        mesh = axis.pcolormesh(
            interaction,
            rf_strength,
            np.log10(np.clip(error_percent, 1.0e-2, 1.0e2)),
            shading="auto",
            cmap="magma",
            vmin=-2.0,
            vmax=2.0,
            rasterized=True,
        )
        contours = axis.contour(
            interaction,
            rf_strength,
            error_percent,
            levels=(1.0, 10.0),
            colors=("white", "cyan"),
            linewidths=1.1,
        )
        axis.clabel(contours, fmt={1.0: "1%", 10.0: "10%"}, fontsize=8)
        isolation_ratio = (
            validity.minimum_target_isolation_hz / site.quadrupole_frequency_hz
        )
        axis.plot(
            interaction,
            isolation_ratio,
            color="deepskyblue",
            linestyle="--",
            linewidth=1.25,
            label="nearest-line spacing / $\\nu_Q$",
        )
        axis.axvspan(0.1, 10.0, color="white", alpha=0.04)
        axis.axvline(1.0, color="white", linewidth=0.8, alpha=0.7)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(rf_strength[0], rf_strength[-1])
        axis.set_title(label)
        axis.set_xlabel(r"Static interaction ratio $\nu_L/\nu_Q$")
        axis.grid(alpha=0.12, which="both")
        axis.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel(r"Peak RF ratio $\gamma B_{1,\mathrm{peak}}/\nu_Q$")
    colorbar = fig.colorbar(mesh, ax=axes, pad=0.02)
    colorbar.set_label("Lab-frame / RWA response error (%)")
    colorbar.set_ticks([-2, -1, 0, 1, 2], labels=["0.01", "0.1", "1", "10", "100"])
    fig.suptitle(
        "Single-band RWA validity through the NQR–NMR crossover\n"
        f"{args.pulse_cycles:g}-carrier-cycle pulse, tilted single crystal"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
