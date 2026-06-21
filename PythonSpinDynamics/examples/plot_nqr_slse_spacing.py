"""Plot spin-1 NQR SLSE response versus pulse period."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    powder_average_grid,
    simulate_slse_spacing_sweep,
    single_crystal_orientation,
)


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (2.0 * np.pi * nutation_hz)


def _orientations(args):
    if args.orientation == "single":
        return single_crystal_orientation(alpha=args.alpha, beta=args.beta)
    return powder_average_grid(args.n_theta, args.n_phi)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Spin support: this pulsed example currently supports spin=1 only.",
    )
    parser.add_argument("--transition", choices=["x", "y", "z"], default="x")
    parser.add_argument("--eta", type=float, default=0.3)
    parser.add_argument("--quadrupole-khz", type=float, default=900.0)
    parser.add_argument("--nutation-khz", type=float, default=10.0)
    parser.add_argument("--pulse-angle", type=float, default=90.0)
    parser.add_argument("--min-spacing-us", type=float, default=150.0)
    parser.add_argument("--max-spacing-us", type=float, default=1500.0)
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--num-echoes", type=int, default=16)
    parser.add_argument("--echo-index", type=int, default=-1)
    parser.add_argument("--rf-offset-khz", type=float, default=0.0)
    parser.add_argument("--t2-ms", type=float, default=20.0)
    parser.add_argument("--t1-ms", type=float, default=np.inf)
    parser.add_argument(
        "--orientation",
        choices=["powder", "single"],
        default="powder",
        help="Use a powder average or one fixed EFG orientation.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Single-crystal azimuthal orientation angle in radians.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=np.pi / 2.0,
        help="Single-crystal polar orientation angle in radians.",
    )
    parser.add_argument("--n-theta", type=int, default=6)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
    )
    orientations = _orientations(args)
    nutation_hz = args.nutation_khz * 1e3
    pulse_duration = _pulse_duration(args.pulse_angle, nutation_hz)

    # SLSE timing is often discussed through the cycle period tc. Sweeping tc
    # changes both where offset maxima occur and how much time relaxation has
    # to act during each cycle.
    spacing_seconds = np.linspace(
        args.min_spacing_us * 1e-6,
        args.max_spacing_us * 1e-6,
        args.points,
    )
    result = simulate_slse_spacing_sweep(
        site,
        args.transition,
        spacing_seconds,
        pulse_duration_seconds=pulse_duration,
        nutation_hz=nutation_hz,
        num_echoes=args.num_echoes,
        rf_offset_hz=args.rf_offset_khz * 1e3,
        orientations=orientations,
        relaxation=NQRRelaxationModel(
            t1_seconds=args.t1_ms * 1e-3,
            t2_seconds=args.t2_ms * 1e-3,
        ),
        echo_index=args.echo_index,
    )

    spacing_us = result.sweep_values * 1e6
    amplitude = result.selected_echo_amplitudes
    magnitude = np.abs(amplitude)
    magnitude_scale = float(np.max(magnitude)) if magnitude.size else 0.0
    normalized_magnitude = (
        magnitude / magnitude_scale if magnitude_scale > 0 else magnitude
    )
    cycle_rate_khz = 1.0 / result.sweep_values / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)

    axes[0].plot(spacing_us, normalized_magnitude)
    axes[0].set_xlabel("SLSE cycle period tc (us)")
    axes[0].set_ylabel("Normalized selected-echo magnitude")
    axes[0].set_title("Pulse-Period Response")

    # This re-plots the same data against 1 / tc, matching the modulation
    # language used in the NQR SLSE chapter.
    axes[1].plot(cycle_rate_khz, normalized_magnitude)
    axes[1].set_xlabel("Cycle frequency 1 / tc (kHz)")
    axes[1].set_ylabel("Normalized selected-echo magnitude")
    axes[1].set_title("Same Sweep vs 1 / tc")

    axes[2].plot(spacing_us, result.effective_t2eff_seconds * 1e3)
    axes[2].set_xlabel("SLSE cycle period tc (us)")
    axes[2].set_ylabel("Effective cycle decay time (ms)")
    axes[2].set_title("Cycle-Derived T2eff")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
