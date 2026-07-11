"""Plot a temperature-resolved field-cycling NMRD experiment.

The workflow converts magnetic field to Larmor frequency with the supplied
gyromagnetic ratio, evaluates a shared Arrhenius correlation time at each
temperature, and returns a temperature-by-field grid. This is a BPP baseline:
real samples may also need multiple motions, CSA, exchange, surface relaxation,
or instrument transfer effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import run_field_cycling_nmrd  # noqa: E402


def build_data(args: argparse.Namespace):
    """Construct a field grid including zero and run the NMRD workflow."""

    nonzero = np.geomspace(args.field_min_t, args.field_max_t, args.points - 1)
    fields = np.concatenate(([0.0], nonzero))
    return run_field_cycling_nmrd(
        fields,
        np.asarray(args.temperatures_k, dtype=np.float64),
        gamma_hz_per_t=args.gamma_mhz_per_t * 1.0e6,
        tau_ref_seconds=args.tau_ref_ns * 1.0e-9,
        reference_temperature_kelvin=args.reference_temp_k,
        activation_energy_j_per_mol=args.activation_energy_kj_mol * 1.0e3,
        coupling_scale_per_second2=args.coupling_scale,
        baseline_r1_per_second=args.baseline_r1,
        baseline_r2_per_second=args.baseline_r2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-min-t", type=float, default=1.0e-6)
    parser.add_argument("--field-max-t", type=float, default=3.0)
    parser.add_argument("--gamma-mhz-per-t", type=float, default=42.57747892)
    parser.add_argument("--temperatures-k", type=float, nargs="+", default=[280, 300, 320])
    parser.add_argument("--reference-temp-k", type=float, default=300.0)
    parser.add_argument("--tau-ref-ns", type=float, default=8.0)
    parser.add_argument("--activation-energy-kj-mol", type=float, default=16.0)
    parser.add_argument("--coupling-scale", type=float, default=5.0e8)
    parser.add_argument("--baseline-r1", type=float, default=0.0)
    parser.add_argument("--baseline-r2", type=float, default=0.0)
    parser.add_argument("--points", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.points < 3:
        parser.error("--points must be at least 3")
    plt = load_matplotlib(headless=args.output is not None)
    result = build_data(args)

    # Frequency, rather than field, is the natural NMRD x-axis. The result also
    # retains the field grid so experimental schedules can be reconstructed.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    frequency_mhz = result.larmor_frequency_hz / 1.0e6
    for row, temperature in enumerate(result.temperature_kelvin):
        axes[0].semilogx(
            frequency_mhz[1:],
            result.r1_per_second[row, 1:],
            label=f"{temperature:g} K",
        )
        axes[1].semilogx(
            frequency_mhz[1:],
            result.t1_seconds[row, 1:],
            label=f"{temperature:g} K",
        )
    axes[0].set_ylabel("R1 (1/s)")
    axes[1].set_ylabel("T1 (s)")
    for axis in axes:
        axis.set_xlabel("Larmor frequency (MHz)")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    axes[0].set_title("NMRD rate dispersion")
    axes[1].set_title("Equivalent field-cycling T1")

    print("Field-cycling/NMRD workflow")
    print(f"grid: {result.relaxivity_shape[0]} temperatures x {result.relaxivity_shape[1]} fields")
    for temperature, tau in zip(
        result.temperature_kelvin,
        result.correlation_time_seconds,
    ):
        print(f"  {temperature:g} K: tau_c={tau * 1e9:.6g} ns")
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
