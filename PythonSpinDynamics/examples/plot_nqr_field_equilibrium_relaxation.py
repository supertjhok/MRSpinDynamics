"""Plot field-dependent Gibbs polarization and thermal relaxation rates.

The NaNO2 and NaClO3 sites match the static crossover examples. A low spin
temperature makes the equilibrium crossover visible; the rate panel uses the
finite-temperature Davies model with magnetic and EFG fluctuation channels.
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
    QuadrupolarSite,
    field_dependent_equilibrium,
    nqr_hamiltonian,
)


ROOT = Path(__file__).resolve().parents[2]
NANO2_SUMMARY = ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"


def _nano2_site() -> QuadrupolarSite:
    with NANO2_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] == "nano2_icsd82857_efg" and row["isotope"] == "14N":
                return QuadrupolarSite(
                    spin=1.0,
                    isotope="14N",
                    quadrupole_frequency_hz=(
                        0.75 * float(row["mean_abs_cq_mhz"]) * 1.0e6
                    ),
                    eta=float(row["mean_eta"]),
                    gamma_hz_per_t=3.0766e6,
                )
    raise RuntimeError(f"could not find NaNO2 parameters in {NANO2_SUMMARY}")


def _naclo3_site() -> QuadrupolarSite:
    return QuadrupolarSite(
        spin=1.5,
        isotope="35Cl",
        quadrupole_frequency_hz=30.656e6,
        eta=0.0,
        gamma_hz_per_t=4.1717e6,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=100)
    parser.add_argument("--temperature-mk", type=float, default=1.0)
    parser.add_argument("--correlation-time-us", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _slowest_decay_rate(generator: np.ndarray) -> float:
    rates = -np.real(np.linalg.eigvals(generator))
    decaying = rates[rates > 1.0e-8]
    return float(np.min(decaying)) if decaying.size else np.nan


def main() -> None:
    args = _parse_args()
    if args.points < 3:
        raise ValueError("points must be at least three")
    temperature = args.temperature_mk * 1.0e-3
    correlation_time = args.correlation_time_us * 1.0e-6
    if temperature <= 0.0 or correlation_time < 0.0:
        raise ValueError("temperature must be positive and correlation time non-negative")

    ratios = np.logspace(-3.0, 1.5, args.points)
    direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    sites = (_nano2_site(), _naclo3_site())
    labels = (r"NaNO$_2$ $^{14}$N ($I=1$)", r"NaClO$_3$ $^{35}$Cl ($I=3/2$)")
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.3), sharex=True, constrained_layout=True)

    for site, label in zip(sites, labels, strict=True):
        model = FieldDependentDaviesRelaxationModel(
            spin=site.spin,
            temperature_kelvin=temperature,
            magnetic_rate_per_second=100.0,
            efg_rate_per_second=30.0,
            correlation_time_seconds=correlation_time,
            secular_tolerance_hz=1.0e-3,
        )
        polarization = np.empty_like(ratios)
        decay_rate = np.empty_like(ratios)
        for index, ratio in enumerate(ratios):
            field = ratio * site.quadrupole_frequency_hz / abs(site.gamma_hz_per_t)
            b0 = field * direction
            equilibrium = field_dependent_equilibrium(
                site,
                b0,
                temperature_kelvin=temperature,
            )
            polarization[index] = (
                np.dot(equilibrium.spin_expectation_pas, direction) / site.spin
            )
            hamiltonian = nqr_hamiltonian(site, b0)
            decay_rate[index] = _slowest_decay_rate(model.superoperator(hamiltonian))
        axes[0].plot(ratios, polarization, linewidth=1.8, label=label)
        axes[1].plot(ratios, decay_rate, linewidth=1.8, label=label)

    for axis in axes:
        axis.axvspan(0.1, 10.0, color="tab:orange", alpha=0.07)
        axis.axvline(1.0, color="0.35", linestyle="--", linewidth=0.9)
        axis.set_xscale("log")
        axis.grid(alpha=0.2, which="both")
        axis.legend()
    axes[0].set_ylabel(r"Equilibrium $\langle I_{B_0}\rangle/I$")
    axes[1].set_ylabel(r"Slowest Davies decay rate (s$^{-1}$)")
    axes[1].set_xlabel(r"Static interaction ratio $\nu_L/\nu_Q$")
    fig.suptitle(
        "Field-dependent equilibrium and relaxation through the NQR–NMR crossover\n"
        f"T = {args.temperature_mk:g} mK, correlation time = "
        f"{args.correlation_time_us:g} µs"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved {args.output}")


if __name__ == "__main__":
    main()
