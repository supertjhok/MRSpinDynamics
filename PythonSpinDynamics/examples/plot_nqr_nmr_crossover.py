"""Plot the exact NQR-to-NMR crossover for NaNO2 and NaClO3.

The spin-1 panel uses the tracked ABINIT EFG summary for 14N in NaNO2. The
spin-3/2 panel uses the 35Cl NaClO3 parameters already used by the Chen et al.
weak-field relaxation validation. Both examples use a tilted single-crystal
geometry so transitions that borrow intensity in the intermediate regime are
visible. Run with ``--output nqr_nmr_crossover.png`` to save the figure.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    CrossoverOrientation,
    QuadrupolarSite,
    simulate_crossover_powder_sweep,
    track_crossover_field_sweep,
)


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"


@dataclass(frozen=True)
class MaterialSite:
    """One plotted quadrupolar material and its parameter provenance."""

    name: str
    site: QuadrupolarSite
    provenance: str


def _load_nano2() -> MaterialSite:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                cq_hz = float(row["mean_abs_cq_mhz"]) * 1.0e6
                eta = float(row["mean_eta"])
                return MaterialSite(
                    name=r"NaNO$_2$ $^{14}$N ($I=1$)",
                    site=QuadrupolarSite(
                        spin=1.0,
                        isotope="14N",
                        label="NaNO2 N",
                        quadrupole_frequency_hz=0.75 * cq_hz,
                        eta=eta,
                        gamma_hz_per_t=3.0766e6,
                    ),
                    provenance=(
                        "ABINIT ICSD 82857: "
                        f"C_Q={cq_hz / 1e6:.4f} MHz, eta={eta:.4f}"
                    ),
                )
    raise RuntimeError(f"could not find NaNO2 14N parameters in {NANO2_SUMMARY}")


def _naclo3() -> MaterialSite:
    return MaterialSite(
        name=r"NaClO$_3$ $^{35}$Cl ($I=3/2$)",
        site=QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            label="NaClO3 Cl",
            quadrupole_frequency_hz=30.656e6,
            eta=0.0,
            gamma_hz_per_t=4.1717e6,
        ),
        provenance="Chen 2020 validation: nu_Q=30.656 MHz, eta=0",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-ratio", type=float, default=1.0e-3)
    parser.add_argument("--max-ratio", type=float, default=30.0)
    parser.add_argument("--points", type=int, default=180)
    parser.add_argument("--temperature-k", type=float, default=293.15)
    parser.add_argument(
        "--powder",
        action="store_true",
        help="Replace transition sticks with an optional powder-averaged spectrum map.",
    )
    parser.add_argument("--powder-n-theta", type=int, default=4)
    parser.add_argument("--powder-n-phi", type=int, default=8)
    parser.add_argument("--powder-n-chi", type=int, default=4)
    parser.add_argument("--powder-frequency-points", type=int, default=384)
    parser.add_argument("--powder-broadening-khz", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _shade_regimes(axis, minimum_ratio: float, maximum_ratio: float) -> None:
    axis.axvspan(1.0e-6, 0.1, color="tab:blue", alpha=0.055)
    axis.axvspan(0.1, 10.0, color="tab:orange", alpha=0.075)
    axis.axvspan(10.0, 1.0e4, color="tab:green", alpha=0.055)
    axis.axvline(1.0, color="0.35", linewidth=0.9, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlim(minimum_ratio, maximum_ratio)
    axis.grid(alpha=0.18, which="both")


def _plot_material(
    axes,
    material: MaterialSite,
    ratios: np.ndarray,
    temperature: float,
    args: argparse.Namespace,
):
    orientation = CrossoverOrientation(
        b0_direction_pas=(1.0, 1.0, 1.0),
        transmit_direction_pas=(1.0, -1.0, 0.0),
    )
    fields = ratios * material.site.quadrupole_frequency_hz / abs(
        material.site.gamma_hz_per_t
    )
    sweep = track_crossover_field_sweep(
        material.site,
        fields,
        orientation=orientation,
        temperature_kelvin=temperature,
    )

    for state in range(material.site.dimension):
        axes[0].plot(
            ratios,
            sweep.levels_hz[:, state] / material.site.quadrupole_frequency_hz,
            linewidth=1.45,
        )
    axes[0].set_title(material.name)
    axes[0].set_ylabel(r"Energy / $h\nu_Q$")
    axes[0].text(
        np.sqrt(ratios[0] * 0.1),
        0.97,
        "NQR",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].text(
        1.0,
        0.97,
        "crossover",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].text(
        np.sqrt(10.0 * ratios[-1]),
        0.97,
        "NMR",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
    )
    axes[0].text(
        0.02,
        0.03,
        material.provenance,
        transform=axes[0].transAxes,
        fontsize=8,
        va="bottom",
    )

    if args.powder:
        powder = simulate_crossover_powder_sweep(
            material.site,
            fields,
            n_theta=args.powder_n_theta,
            n_phi=args.powder_n_phi,
            n_chi=args.powder_n_chi,
            temperature_kelvin=temperature,
            broadening_hz=args.powder_broadening_khz * 1.0e3,
            frequency_points=args.powder_frequency_points,
        )
        scatter = axes[1].pcolormesh(
            ratios,
            powder.frequencies_hz / material.site.quadrupole_frequency_hz,
            np.abs(powder.spectra).T,
            shading="auto",
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
            rasterized=True,
        )
    else:
        intensities = sweep.transition_intensities
        scale = float(np.max(intensities))
        relative = intensities / scale if scale > 0.0 else intensities
        x_values = np.repeat(ratios, len(sweep.state_pairs))
        y_values = (
            sweep.transition_frequencies_hz / material.site.quadrupole_frequency_hz
        ).reshape(-1)
        colors = np.log10(np.clip(relative.reshape(-1), 1.0e-7, 1.0))
        sizes = 5.0 + 23.0 * np.sqrt(relative.reshape(-1))
        visible = relative.reshape(-1) > 1.0e-7
        scatter = axes[1].scatter(
            x_values[visible],
            y_values[visible],
            c=colors[visible],
            s=sizes[visible],
            vmin=-7.0,
            vmax=0.0,
            cmap="viridis",
            linewidths=0.0,
            rasterized=True,
        )
    axes[1].set_xlabel(r"Interaction ratio $\nu_L/\nu_Q$")
    axes[1].set_ylabel(r"Transition frequency / $\nu_Q$")
    axes[1].set_ylim(bottom=0.0)
    for axis in axes:
        _shade_regimes(axis, ratios[0], ratios[-1])
    return sweep, scatter


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parse_args()
    if args.min_ratio <= 0.0 or args.max_ratio <= args.min_ratio:
        raise ValueError("require 0 < min-ratio < max-ratio")
    if args.points < 3:
        raise ValueError("points must be at least three")
    ratios = np.logspace(
        np.log10(args.min_ratio),
        np.log10(args.max_ratio),
        args.points,
    )
    materials = (_load_nano2(), _naclo3())
    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)

    sweeps = []
    scatter = None
    for column, material in enumerate(materials):
        sweep, scatter = _plot_material(
            axes[:, column],
            material,
            ratios,
            args.temperature_k,
            args,
        )
        sweeps.append(sweep)

    fig.suptitle(
        "Exact NQR → Zeeman-perturbed NQR → quadrupolar NMR crossover\n"
        + (
            "Tilted single-crystal levels; powder-averaged spectra"
            if args.powder
            else "Tilted single-crystal geometry; marker color and size encode intensity"
        ),
        fontsize=13,
    )
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=axes[1, :], shrink=0.88, pad=0.02)
        colorbar.set_label(
            "Normalized powder intensity"
            if args.powder
            else r"$\log_{10}$ relative equilibrium intensity"
        )

    for material, sweep in zip(materials, sweeps):
        bq = material.site.quadrupole_frequency_hz / abs(
            material.site.gamma_hz_per_t
        )
        print(
            f"{material.site.label}: B_Q={bq:.4g} T; "
            f"minimum adjacent-state overlap={np.min(sweep.minimum_state_overlap[1:]):.6f}"
        )

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
