"""Run a transition-resolved quadrupolar-relaxation temperature sweep.

This example starts from an isotope-aware NQR site, supplies one Arrhenius law
for the EFG correlation time, and asks the workflow for the initial population
(``R1``) and coherence (``R2``) decay rate of every allowed transition. The
absolute scale depends on the fluctuating-EFG amplitude; transition and
temperature trends are meaningful only within that stated model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import quadrupolar_site  # noqa: E402
from spin_dynamics.workflows import run_arrhenius_quadrupolar_relaxation  # noqa: E402


def build_data(args: argparse.Namespace):
    """Build the site and run the microscopic temperature sweep."""

    temperature = np.linspace(args.temp_min_k, args.temp_max_k, args.points)
    site = quadrupolar_site(
        args.isotope,
        cq_hz=args.cq_mhz * 1.0e6,
        eta=args.eta,
    )
    return run_arrhenius_quadrupolar_relaxation(
        site,
        temperature,
        tau_ref_seconds=args.tau_ref_ns * 1.0e-9,
        reference_temperature_kelvin=args.reference_temp_k,
        activation_energy_j_per_mol=args.activation_energy_kj_mol * 1.0e3,
        fluctuation_amplitude_hz=args.efg_fluctuation_khz * 1.0e3,
        vibration_frequency_hz=args.vibration_mhz * 1.0e6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isotope", default="14N")
    parser.add_argument("--cq-mhz", type=float, default=3.0)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--temp-min-k", type=float, default=260.0)
    parser.add_argument("--temp-max-k", type=float, default=340.0)
    parser.add_argument("--reference-temp-k", type=float, default=300.0)
    parser.add_argument("--tau-ref-ns", type=float, default=1.0)
    parser.add_argument("--activation-energy-kj-mol", type=float, default=18.0)
    parser.add_argument("--efg-fluctuation-khz", type=float, default=20.0)
    parser.add_argument("--vibration-mhz", type=float, default=0.0)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    result = build_data(args)

    # Each column belongs to one transition; keeping labels and frequencies in
    # the result prevents rate curves from being detached from spectroscopy.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for index, (label, frequency) in enumerate(
        zip(result.transition_labels, result.transition_frequencies_hz)
    ):
        legend = f"{label}: {frequency / 1e6:.3f} MHz"
        axes[0].semilogy(
            result.temperature_kelvin,
            result.r1_per_second[:, index],
            label=legend,
        )
        axes[1].semilogy(
            result.temperature_kelvin,
            result.r2_per_second[:, index],
            label=legend,
        )
    axes[0].set_title("Population-difference initial decay (R1)")
    axes[1].set_title("Coherence initial decay (R2)")
    for axis in axes:
        axis.set_xlabel("Temperature (K)")
        axis.set_ylabel("Rate (1/s)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)

    print("Quadrupolar relaxation workflow")
    print(
        f"{result.site.isotope}, spin={result.site.spin:g}, "
        f"C_Q={args.cq_mhz:g} MHz, eta={result.site.eta:g}"
    )
    for label, frequency in zip(
        result.transition_labels,
        result.transition_frequencies_hz,
    ):
        print(f"  {label}: {frequency / 1e6:.6g} MHz")
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
