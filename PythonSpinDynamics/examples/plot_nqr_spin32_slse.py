"""Spin-3/2 chlorine SLSE from the full density-matrix NQR model.

The single zero-field NQR line of a spin-3/2 nucleus (for example 35Cl) connects
two doubly degenerate Kramers doublets, so a selective pulse acts on a four-state
manifold and the embedded two-level ``simulate_slse`` does not apply. This
example uses ``simulate_full_slse``, which propagates the full ``(2I+1)`` density
matrix, to compute the powder spin-lock spin-echo (SLSE) train, and shows how a
weak static Zeeman field detunes the crystallites and reshapes the decay.

Run with ``--output nqr_spin32_slse.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


@dataclass(frozen=True)
class SpinThreeHalfSLSE:
    """Powder SLSE echo trains for several static-field strengths."""

    echo_times: np.ndarray
    b0_values_tesla: np.ndarray
    echo_trains: np.ndarray
    quadrupole_frequency_hz: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the spin-3/2 (35Cl) powder SLSE echo train with the full "
            "density-matrix NQR model and show the effect of a weak Zeeman field."
        )
    )
    parser.add_argument(
        "--quadrupole-mhz",
        type=float,
        default=1.0,
        help="Zero-field quadrupole line frequency in MHz.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="EFG asymmetry parameter.",
    )
    parser.add_argument(
        "--nutation-khz",
        type=float,
        default=10.0,
        help="Bare RF nutation gamma*B1/(2*pi) in kHz.",
    )
    parser.add_argument(
        "--excitation-us",
        type=float,
        default=25.0,
        help="Excitation pulse duration in microseconds.",
    )
    parser.add_argument(
        "--refocus-us",
        type=float,
        default=50.0,
        help="Refocusing pulse duration in microseconds.",
    )
    parser.add_argument(
        "--echo-spacing-us",
        type=float,
        default=400.0,
        help="SLSE echo spacing in microseconds.",
    )
    parser.add_argument(
        "--num-echoes",
        type=int,
        default=12,
        help="Number of SLSE echoes.",
    )
    parser.add_argument(
        "--t2e-ms",
        type=float,
        default=3.0,
        help="Phenomenological SLSE decay envelope in milliseconds.",
    )
    parser.add_argument(
        "--b0-mt",
        type=float,
        nargs="+",
        default=[0.0, 5.0, 15.0],
        help="Static Zeeman field strengths in millitesla.",
    )
    parser.add_argument(
        "--gamma-mhz-per-t",
        type=float,
        default=4.17,
        help="Nuclear gyromagnetic ratio in MHz/T (35Cl ~ 4.17).",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=8,
        help="Powder polar samples.",
    )
    parser.add_argument(
        "--n-phi",
        type=int,
        default=16,
        help="Powder azimuthal samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _simulate(args: argparse.Namespace) -> SpinThreeHalfSLSE:
    from spin_dynamics.nqr import (
        QuadrupolarSite,
        b0_powder_average_grid,
        simulate_full_slse,
    )

    nu_q = float(args.quadrupole_mhz) * 1e6
    gamma = float(args.gamma_mhz_per_t) * 1e6
    grid = b0_powder_average_grid(int(args.n_theta), int(args.n_phi))
    b0_values = np.asarray(args.b0_mt, dtype=np.float64) * 1e-3

    slse_kwargs = dict(
        nutation_hz=float(args.nutation_khz) * 1e3,
        excitation_duration_seconds=float(args.excitation_us) * 1e-6,
        refocus_duration_seconds=float(args.refocus_us) * 1e-6,
        echo_spacing_seconds=float(args.echo_spacing_us) * 1e-6,
        num_echoes=int(args.num_echoes),
        t2e_seconds=float(args.t2e_ms) * 1e-3,
    )

    trains = []
    echo_times = None
    for b0 in b0_values:
        site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=nu_q,
            eta=float(args.eta),
            gamma_hz_per_t=gamma,
        )
        result = simulate_full_slse(
            site, orientations=grid, b0_tesla=float(b0), **slse_kwargs
        )
        trains.append(np.abs(result.echo_amplitudes))
        echo_times = result.echo_times
    trains = np.asarray(trains, dtype=np.float64)
    normalizer = trains[0, 0] if trains[0, 0] > 0 else 1.0
    return SpinThreeHalfSLSE(
        echo_times=np.asarray(echo_times, dtype=np.float64),
        b0_values_tesla=b0_values,
        echo_trains=trains / normalizer,
        quadrupole_frequency_hz=nu_q,
    )


def _plot_results(plt, sim: SpinThreeHalfSLSE):
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    for b0, train in zip(sim.b0_values_tesla, sim.echo_trains):
        ax.semilogy(
            sim.echo_times * 1e3,
            train,
            marker="o",
            ms=3,
            label=f"B0 = {b0 * 1e3:.0f} mT",
        )
    ax.set_xlabel("echo time (ms)")
    ax.set_ylabel("normalized powder echo amplitude")
    ax.set_title(
        f"$^{{35}}$Cl spin-3/2 powder SLSE "
        f"($\\nu_Q$ = {sim.quadrupole_frequency_hz / 1e6:.2f} MHz)"
    )
    ax.legend()
    return fig


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 0:
        raise SystemExit("--num-echoes must be positive")
    if any(b < 0.0 for b in args.b0_mt):
        raise SystemExit("--b0-mt values must be non-negative")

    plt = load_matplotlib(headless=bool(args.output))
    sim = _simulate(args)

    print(f"35Cl quadrupole line: {sim.quadrupole_frequency_hz / 1e6:.3f} MHz")
    print("normalized final echo amplitude by static field:")
    for b0, train in zip(sim.b0_values_tesla, sim.echo_trains):
        print(f"  B0 = {b0 * 1e3:5.1f} mT -> {train[-1]:.3f}")

    fig = _plot_results(plt, sim)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
