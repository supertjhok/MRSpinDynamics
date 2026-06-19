"""Plot a diagnostic powder NQR nutation curve using one SLSE pulse."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    powder_average_grid,
    simulate_slse,
    slse_sequence,
)


def _spin_three_halves_bessel_curve(theta: np.ndarray) -> np.ndarray:
    """Return the normalized spin-1 powder nutation envelope from J_3/2."""

    theta = np.asarray(theta, dtype=np.float64)
    out = np.zeros_like(theta)
    nonzero = theta != 0
    x = theta[nonzero]
    # sqrt(pi / (2x)) J_{3/2}(x) = sin(x) / x^2 - cos(x) / x
    out[nonzero] = np.sin(x) / (x * x) - np.cos(x) / x
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transition",
        choices=["x", "y", "z"],
        default="x",
        help="NQR transition to detect.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.3,
        help="EFG asymmetry parameter for the spin-1 site.",
    )
    parser.add_argument(
        "--quadrupole-khz",
        type=float,
        default=900.0,
        help="Quadrupole frequency parameter in kHz.",
    )
    parser.add_argument(
        "--nutation-khz",
        type=float,
        default=10.0,
        help="Nominal RF nutation frequency in kHz.",
    )
    parser.add_argument(
        "--max-angle",
        type=float,
        default=720.0,
        help="Maximum nominal flip angle in degrees.",
    )
    parser.add_argument("--points", type=int, default=121, help="Flip-angle samples.")
    parser.add_argument("--n-theta", type=int, default=12, help="Powder polar samples.")
    parser.add_argument("--n-phi", type=int, default=24, help="Powder azimuthal samples.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(
        spin=1,
        isotope="14N",
        quadrupole_frequency_hz=args.quadrupole_khz * 1e3,
        eta=args.eta,
    )
    orientations = powder_average_grid(args.n_theta, args.n_phi)
    angles_deg = np.linspace(0.0, args.max_angle, args.points)
    angles_rad = np.deg2rad(angles_deg)
    nutation_hz = args.nutation_khz * 1e3

    signal = np.empty(args.points, dtype=np.complex128)
    for idx, angle_rad in enumerate(angles_rad):
        duration = angle_rad / (2.0 * np.pi * nutation_hz)
        sequence = slse_sequence(
            args.transition,
            pulse_duration_seconds=duration,
            nutation_hz=nutation_hz,
            echo_spacing_seconds=1e-3,
            num_echoes=1,
        )
        result = simulate_slse(site, sequence, orientations=orientations)
        signal[idx] = result.echo_amplitudes[0]

    reference = _spin_three_halves_bessel_curve(angles_rad)
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        phase = np.angle(signal[int(np.argmax(np.abs(signal)))])
        aligned = signal * np.exp(-1j * phase)
        signal_plot = np.real(aligned) / peak
    else:
        aligned = signal
        signal_plot = np.real(aligned)
    if np.max(np.abs(reference)) > 0:
        reference = reference / np.max(np.abs(reference))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].plot(angles_deg, signal_plot, label="phase-aligned simulation")
    axes[0].plot(angles_deg, np.abs(signal) / np.max(np.abs(signal)), label="simulated magnitude")
    axes[0].plot(angles_deg, reference, "--", label="J3/2 envelope")
    axes[0].axvline(120.0, color="0.4", linewidth=1, alpha=0.6)
    axes[0].set_xlabel("Nominal flip angle (degrees)")
    axes[0].set_ylabel("Normalized first-echo amplitude")
    axes[0].set_title(f"Powder Nutation, {args.transition} Transition")
    axes[0].legend()

    axes[1].plot(angles_deg, np.real(signal), label="real")
    axes[1].plot(angles_deg, np.imag(signal), label="imag")
    axes[1].set_xlabel("Nominal flip angle (degrees)")
    axes[1].set_ylabel("Raw averaged echo amplitude")
    axes[1].set_title("Complex Signal Components")
    axes[1].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
