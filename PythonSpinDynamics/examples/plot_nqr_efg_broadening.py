"""Plot spin-1 NQR time-domain and spectral response from static EFG disorder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    SelectivePulse,
    gaussian_efg_distribution,
    powder_average_grid,
    simulate_fid_efg_distribution,
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
    parser.add_argument("--nuq-std-khz", type=float, default=2.0)
    parser.add_argument("--eta-std", type=float, default=0.0)
    parser.add_argument("--efg-samples", type=int, default=41)
    parser.add_argument("--sigma-span", type=float, default=3.0)
    parser.add_argument("--nutation-khz", type=float, default=100.0)
    parser.add_argument("--pulse-angle", type=float, default=90.0)
    parser.add_argument("--duration-ms", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=512)
    parser.add_argument("--t2-ms", type=float, default=np.inf)
    parser.add_argument(
        "--orientation",
        choices=["powder", "single"],
        default="single",
        help="Use a powder average or one fixed EFG orientation.",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=np.pi / 2.0)
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
    distribution = gaussian_efg_distribution(
        site,
        quadrupole_std_hz=args.nuq_std_khz * 1e3,
        eta_std=args.eta_std,
        samples=args.efg_samples,
        sigma_span=args.sigma_span,
    )
    times = np.linspace(0.0, args.duration_ms * 1e-3, args.points)
    nutation_hz = args.nutation_khz * 1e3

    # This is the direct NQR analogue of an NMR isochromat sum: each static EFG
    # variant has its own resonance frequency, accumulates its own phase, and
    # contributes to the complex ensemble FID with a fixed weight.
    result = simulate_fid_efg_distribution(
        distribution,
        args.transition,
        times,
        excitation=SelectivePulse(
            args.transition,
            duration_seconds=_pulse_duration(args.pulse_angle, nutation_hz),
            nutation_hz=nutation_hz,
        ),
        orientations=_orientations(args),
        t2_seconds=args.t2_ms * 1e-3,
    )

    time_ms = result.times * 1e3
    spectrum_offset_khz = result.spectrum_frequencies_hz / 1e3
    spectrum = np.abs(result.spectrum)
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    axes[0].plot(time_ms, np.real(result.signal), label="real")
    axes[0].plot(time_ms, np.imag(result.signal), label="imag")
    axes[0].plot(time_ms, np.abs(result.signal), label="magnitude")
    axes[0].set_xlabel("Time after pulse (ms)")
    axes[0].set_ylabel("Complex FID amplitude")
    axes[0].set_title("Time-Domain EFG Isochromat Sum")
    axes[0].legend()

    axes[1].plot(spectrum_offset_khz, spectrum)
    axes[1].set_xlabel("Frequency offset from carrier (kHz)")
    axes[1].set_ylabel("Normalized spectrum")
    axes[1].set_title("FFT Spectrum")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
