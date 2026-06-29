"""Four-pulse DEER: forward dipolar trace, Pake spectrum, and distance recovery.

Builds a Gaussian distance distribution between two nitroxide-like electron
spins, computes the powder-averaged DEER form factor, shows its dipolar (Pake)
spectrum, and recovers the distance distribution by regularized inversion. A
single-orientation panel overlays the independent two-electron density-matrix
simulation on the analytic kernel to show they agree.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    deer_dipolar_spectrum,
    deer_pair_trace,
    deer_pair_trace_quantum,
    dipolar_frequency_hz,
    extract_distance_distribution,
    gaussian_distance_distribution,
    simulate_deer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-nm", type=float, default=2.5, help="Mean spin-spin distance (nm).")
    parser.add_argument("--sigma-nm", type=float, default=0.2, help="Distance spread (nm).")
    parser.add_argument("--lambda-depth", type=float, default=0.35, help="Modulation depth.")
    parser.add_argument("--max-time-us", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--min-distance-nm", type=float, default=1.5)
    parser.add_argument("--max-distance-nm", type=float, default=4.5)
    parser.add_argument("--distance-points", type=int, default=80)
    parser.add_argument("--snr", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    times = np.linspace(0.0, args.max_time_us * 1e-6, args.points)
    distances = np.linspace(args.min_distance_nm, args.max_distance_nm, args.distance_points)
    truth = gaussian_distance_distribution(distances, args.mean_nm, args.sigma_nm)

    form_factor = simulate_deer(times, distances, truth, lambda_depth=args.lambda_depth)
    frequencies, spectrum = deer_dipolar_spectrum(times, form_factor)
    recovery = extract_distance_distribution(
        times, form_factor, distances, lambda_depth=args.lambda_depth, snr=args.snr
    )

    recovered = recovery.distribution / max(float(np.sum(recovery.distribution)), 1e-30)
    recovered_mean = float(np.sum(distances * recovered))
    nu_perp = dipolar_frequency_hz(args.mean_nm)
    print(
        f"DEER: true mean {args.mean_nm:.2f} nm, recovered mean {recovered_mean:.2f} nm, "
        f"nu_perp(mean) {nu_perp / 1e6:.2f} MHz, residual {recovery.residual_norm:.2e}"
    )

    # Single-orientation cross-check: analytic kernel vs density-matrix simulation.
    pair_times = np.linspace(0.0, 2.0e-6, 60)
    analytic = deer_pair_trace(pair_times, args.mean_nm, np.pi / 2.0, lambda_depth=1.0)
    quantum = deer_pair_trace_quantum(pair_times, args.mean_nm, np.pi / 2.0, pump_flip_rad=np.pi)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(1e6 * times, form_factor, color="tab:blue", label="simulated")
    axes[0, 0].plot(
        1e6 * times, recovery.fitted_form_factor, "--", color="tab:orange", label="fit"
    )
    axes[0, 0].set_xlabel("Pump position t (us)")
    axes[0, 0].set_ylabel("DEER form factor F(t)")
    axes[0, 0].set_title("Form Factor")
    axes[0, 0].legend()

    axes[0, 1].plot(frequencies / 1e6, spectrum, color="tab:green")
    axes[0, 1].axvline(nu_perp / 1e6, color="tab:red", linestyle="--", label="nu_perp(mean)")
    axes[0, 1].set_xlim(0.0, 5.0 * nu_perp / 1e6)
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_ylabel("|FT|")
    axes[0, 1].set_title("Dipolar (Pake) Spectrum")
    axes[0, 1].legend()

    axes[1, 0].plot(distances, truth, color="tab:blue", label="true P(r)")
    axes[1, 0].plot(distances, recovered, color="tab:orange", label="recovered P(r)")
    axes[1, 0].set_xlabel("Distance (nm)")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].set_title("Distance Distribution")
    axes[1, 0].legend()

    axes[1, 1].plot(1e6 * pair_times, analytic, color="tab:blue", label="analytic kernel")
    axes[1, 1].plot(
        1e6 * pair_times, quantum, "x", color="tab:red", markersize=4, label="density matrix"
    )
    axes[1, 1].set_xlabel("Pump position t (us)")
    axes[1, 1].set_ylabel("F(t), single pair (theta=90)")
    axes[1, 1].set_title("Kernel vs Spin Simulation")
    axes[1, 1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
