"""Plot a NaNO2 SLSE response in four NQR--NMR field regimes.

The calculation uses exact finite-sideband Floquet pulses and the
field-dependent unified-GKLS relaxation model during free evolution. Dashed
curves show the fully secular Davies limit for comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    FieldDependentDaviesRelaxationModel,
    FieldDependentNonsecularRelaxationModel,
    QuadrupolarSite,
    nqr_hamiltonian,
    simulate_crossover_slse,
)


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"


def _nano2_site() -> tuple[QuadrupolarSite, str]:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                cq_hz = float(row["mean_abs_cq_mhz"]) * 1.0e6
                eta = float(row["mean_eta"])
                return (
                    QuadrupolarSite(
                        spin=1.0,
                        isotope="14N",
                        label="NaNO2 N",
                        quadrupole_frequency_hz=0.75 * cq_hz,
                        eta=eta,
                        gamma_hz_per_t=3.0766e6,
                    ),
                    f"ABINIT ICSD 82857: C_Q={cq_hz / 1e6:.4f} MHz, eta={eta:.4f}",
                )
    raise RuntimeError(f"could not find NaNO2 parameters in {NANO2_SUMMARY}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echoes", type=int, default=12)
    parser.add_argument("--echo-spacing-us", type=float, default=150.0)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--cluster-width-khz", type=float, default=20.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.echoes <= 0 or args.echo_spacing_us <= 0.0:
        raise ValueError("echoes and echo spacing must be positive")
    site, provenance = _nano2_site()
    direction_b0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    direction_b1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    regimes = (
        (0.0, "zero-field NQR"),
        (0.1, "Zeeman-perturbed NQR"),
        (1.0, "intermediate crossover"),
        (10.0, "quadrupolar NMR"),
    )
    common_relaxation = dict(
        spin=site.spin,
        temperature_kelvin=args.temperature_k,
        magnetic_rate_per_second=2.0e3,
        efg_rate_per_second=5.0e2,
        correlation_time_seconds=0.2e-6,
        secular_tolerance_hz=1.0e-3,
    )
    unified = FieldDependentNonsecularRelaxationModel(
        **common_relaxation,
        frequency_cluster_width_hz=args.cluster_width_khz * 1.0e3,
    )
    secular = FieldDependentDaviesRelaxationModel(**common_relaxation)
    nutation = 0.02 * site.quadrupole_frequency_hz
    sequence = dict(
        nutation_hz=nutation,
        excitation_duration_seconds=0.15 / nutation,
        refocus_duration_seconds=0.30 / nutation,
        echo_spacing_seconds=args.echo_spacing_us * 1.0e-6,
        num_echoes=args.echoes,
        b1_direction_pas=direction_b1,
        floquet_sidebands=5,
    )

    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8), constrained_layout=True)
    for axis, (ratio, regime) in zip(axes.flat, regimes, strict=True):
        field = ratio * site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
        b0 = field * direction_b0
        nonsecular_result = simulate_crossover_slse(
            site,
            b0,
            relaxation=unified,
            **sequence,
        )
        secular_result = simulate_crossover_slse(
            site,
            b0,
            relaxation=secular,
            **sequence,
        )
        time_ms = nonsecular_result.echo_times_seconds * 1.0e3
        nonsecular_signal = np.abs(nonsecular_result.echo_amplitudes) * 1.0e6
        secular_signal = np.abs(secular_result.echo_amplitudes) * 1.0e6
        difference = np.linalg.norm(
            nonsecular_result.echo_amplitudes - secular_result.echo_amplitudes
        ) / max(np.linalg.norm(secular_result.echo_amplitudes), 1.0e-30)
        hamiltonian = nqr_hamiltonian(site, b0)
        gibbs_residual = unified.gibbs_stationarity_error(hamiltonian)
        axis.plot(
            time_ms,
            nonsecular_signal,
            "o-",
            linewidth=1.6,
            markersize=4.0,
            label="unified GKLS",
        )
        axis.plot(
            time_ms,
            secular_signal,
            "--",
            linewidth=1.2,
            label="secular Davies",
        )
        axis.set_title(
            f"{regime}: $\\nu_L/\\nu_Q={ratio:g}$, $B_0={field:.3g}$ T"
        )
        axis.text(
            0.97,
            0.95,
            f"carrier {nonsecular_result.rf_frequency_hz / 1e6:.3f} MHz\n"
            f"relative model difference {100 * difference:.3g}%\n"
            f"Gibbs residual {gibbs_residual:.2g} s$^{{-1}}$",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
        axis.set_xlabel("Echo time (ms)")
        axis.set_ylabel(r"$|\langle I_+\rangle-\langle I_+\rangle_{eq}|$ ($10^{-6}$)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle(
        r"NaNO$_2$ $^{14}$N SLSE across the NQR--NMR crossover" + "\n"
        + provenance
        + f"; cluster width={args.cluster_width_khz:g} kHz"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
