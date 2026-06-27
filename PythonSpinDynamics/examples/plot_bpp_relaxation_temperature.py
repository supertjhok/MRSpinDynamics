"""Plot BPP T1 and T2 relaxation versus temperature."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.relaxation import BPPRelaxationModel  # noqa: E402


def build_temperature_sweep(args: argparse.Namespace):
    """Return temperatures and BPP rates for the requested CLI parameters."""

    temperatures = np.linspace(args.temp_min_k, args.temp_max_k, int(args.points))
    model = BPPRelaxationModel(
        angular_frequency_rad_per_s=2.0 * np.pi * args.larmor_mhz * 1e6,
        tau_ref_seconds=args.tau_ref_ns * 1e-9,
        reference_temperature_kelvin=args.reference_temp_k,
        activation_energy_j_per_mol=args.activation_energy_kj_mol * 1e3,
        coupling_scale_per_second2=args.coupling_scale,
        baseline_r1_per_second=args.baseline_r1,
        baseline_r2_per_second=args.baseline_r2,
    )
    return temperatures, model.rates(temperatures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-min-k", type=float, default=250.0)
    parser.add_argument("--temp-max-k", type=float, default=360.0)
    parser.add_argument("--reference-temp-k", type=float, default=300.0)
    parser.add_argument("--points", type=int, default=220)
    parser.add_argument("--larmor-mhz", type=float, default=20.0)
    parser.add_argument("--tau-ref-ns", type=float, default=8.0)
    parser.add_argument("--activation-energy-kj-mol", type=float, default=16.0)
    parser.add_argument("--coupling-scale", type=float, default=5.0e8)
    parser.add_argument("--baseline-r1", type=float, default=0.0)
    parser.add_argument("--baseline-r2", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    temperatures, rates = build_temperature_sweep(args)
    omega_tau = 2.0 * np.pi * args.larmor_mhz * 1e6 * rates.correlation_time_seconds

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    axes[0, 0].semilogy(temperatures, rates.t1_seconds, label="T1")
    axes[0, 0].semilogy(temperatures, rates.t2_seconds, label="T2")
    axes[0, 0].set_xlabel("Temperature (K)")
    axes[0, 0].set_ylabel("Relaxation time (s)")
    axes[0, 0].set_title("BPP Relaxation Times")
    axes[0, 0].legend()

    axes[0, 1].plot(temperatures, rates.r1_per_second, label="R1")
    axes[0, 1].plot(temperatures, rates.r2_per_second, label="R2")
    axes[0, 1].set_xlabel("Temperature (K)")
    axes[0, 1].set_ylabel("Relaxation rate (1/s)")
    axes[0, 1].set_title("Rates")
    axes[0, 1].legend()

    axes[1, 0].semilogy(temperatures, rates.correlation_time_seconds * 1e9)
    axes[1, 0].set_xlabel("Temperature (K)")
    axes[1, 0].set_ylabel("tau_c (ns)")
    axes[1, 0].set_title("Arrhenius Correlation Time")

    axes[1, 1].semilogy(temperatures, rates.j0_seconds, label="J(0)")
    axes[1, 1].semilogy(temperatures, rates.jw_seconds, label="J(w0)")
    axes[1, 1].semilogy(temperatures, rates.j2w_seconds, label="J(2w0)")
    axes[1, 1].semilogy(temperatures, omega_tau / np.max(omega_tau) * np.max(rates.j0_seconds),
                         "--", color="0.35", label="scaled w0 tau_c")
    axes[1, 1].set_xlabel("Temperature (K)")
    axes[1, 1].set_ylabel("Spectral density (s)")
    axes[1, 1].set_title("Spectral Density Terms")
    axes[1, 1].legend()

    t1_min_idx = int(np.argmin(rates.t1_seconds))
    t2_min_idx = int(np.argmin(rates.t2_seconds))
    print("BPP relaxation temperature sweep")
    print(f"Larmor frequency MHz: {args.larmor_mhz:g}")
    print(f"tau_ref ns at {args.reference_temp_k:g} K: {args.tau_ref_ns:g}")
    print(f"activation energy kJ/mol: {args.activation_energy_kj_mol:g}")
    print(
        "T1 minimum: "
        f"{rates.t1_seconds[t1_min_idx]:.6g} s at {temperatures[t1_min_idx]:.3g} K"
    )
    print(
        "T2 minimum: "
        f"{rates.t2_seconds[t2_min_idx]:.6g} s at {temperatures[t2_min_idx]:.3g} K"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
