"""Plot spin-1 SLSE echo trains from static EFG inhomogeneous broadening."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    diagonalize_site,
    gaussian_efg_distribution,
    powder_average_grid,
    simulate_slse_acquisition_spectrum,
    single_crystal_orientation,
    slse_sequence,
)


def _pulse_duration(angle_degrees: float, nutation_hz: float) -> float:
    return np.deg2rad(angle_degrees) / (2.0 * np.pi * nutation_hz)


def _orientations(args):
    if args.orientation == "single":
        return single_crystal_orientation(alpha=args.alpha, beta=args.beta)
    return powder_average_grid(args.n_theta, args.n_phi)


def _normalized_magnitude(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    scale = float(np.max(magnitude)) if magnitude.size else 0.0
    return magnitude / scale if scale > 0 else magnitude


def _normalized_real(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return np.real(values) / scale if scale > 0 else np.real(values)


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
    parser.add_argument("--nutation-khz", type=float, default=10.0)
    parser.add_argument("--pulse-angle", type=float, default=90.0)
    parser.add_argument("--echo-spacing-us", type=float, default=500.0)
    parser.add_argument("--num-echoes", type=int, default=96)
    parser.add_argument(
        "--echo-index",
        type=int,
        default=-1,
        help="Echo index to acquire; negative values count from the end.",
    )
    parser.add_argument(
        "--acq-us",
        type=float,
        default=200.0,
        help="Acquisition window centered on the selected echo.",
    )
    parser.add_argument("--acq-points", type=int, default=256)
    parser.add_argument("--zero-fill", type=int, default=2)
    parser.add_argument(
        "--noise-snr",
        type=float,
        default=None,
        help="Optional target SNR for complex white time-domain receiver noise.",
    )
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument(
        "--deconvolve",
        action="store_true",
        help="Overlay regularized deconvolution of the finite acquisition window.",
    )
    parser.add_argument("--deconv-strength", type=float, default=1e-2)
    parser.add_argument("--t2-ms", type=float, default=20.0)
    parser.add_argument("--t1-ms", type=float, default=np.inf)
    parser.add_argument(
        "--rephase-action",
        choices=["warn", "raise", "ignore"],
        default="warn",
        help="Action when the EFG isochromat grid may discretely rephase.",
    )
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
    carrier_hz = diagonalize_site(site).transition(args.transition).frequency_hz
    broad = gaussian_efg_distribution(
        site,
        quadrupole_std_hz=args.nuq_std_khz * 1e3,
        eta_std=args.eta_std,
        samples=args.efg_samples,
        sigma_span=args.sigma_span,
    )
    narrow = gaussian_efg_distribution(site, samples=1)

    nutation_hz = args.nutation_khz * 1e3
    sequence = slse_sequence(
        args.transition,
        pulse_duration_seconds=_pulse_duration(args.pulse_angle, nutation_hz),
        nutation_hz=nutation_hz,
        echo_spacing_seconds=args.echo_spacing_us * 1e-6,
        num_echoes=args.num_echoes,
        rf_frequency_hz=carrier_hz,
    )
    relaxation = NQRRelaxationModel(
        t1_seconds=args.t1_ms * 1e-3,
        t2_seconds=args.t2_ms * 1e-3,
    )
    orientations = _orientations(args)

    # This is the pulsed analogue of an NMR isochromat calculation. The RF
    # carrier stays fixed at the central line while each EFG variant sees its
    # own detuning during the SLSE cycle.
    noise = None
    if args.noise_snr is not None:
        noise = {
            "target_snr": args.noise_snr,
            "seed": args.noise_seed,
            "domain": "time",
        }
    deconv_strength = args.deconv_strength if args.deconvolve else None
    broad_acq = simulate_slse_acquisition_spectrum(
        broad,
        sequence,
        acquisition_duration_seconds=args.acq_us * 1e-6,
        acquisition_points=args.acq_points,
        echo_index=args.echo_index,
        carrier_frequency_hz=carrier_hz,
        orientations=orientations,
        relaxation=relaxation,
        zero_fill_factor=args.zero_fill,
        noise=noise,
        deconvolution_strength=deconv_strength,
        rephase_action=args.rephase_action,
    )
    narrow_acq = simulate_slse_acquisition_spectrum(
        narrow,
        sequence,
        acquisition_duration_seconds=args.acq_us * 1e-6,
        acquisition_points=args.acq_points,
        echo_index=args.echo_index,
        carrier_frequency_hz=carrier_hz,
        orientations=orientations,
        relaxation=relaxation,
        zero_fill_factor=args.zero_fill,
        rephase_action=args.rephase_action,
    )
    broad_result = broad_acq.echo_train
    narrow_result = narrow_acq.echo_train

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    echo_index = np.arange(1, args.num_echoes + 1)

    axes[0].plot(
        echo_index,
        _normalized_magnitude(narrow_result.echo_amplitudes),
        label="zero-width EFG",
    )
    axes[0].plot(
        echo_index,
        _normalized_magnitude(broad_result.echo_amplitudes),
        label="broadened EFG",
    )
    axes[0].set_xlabel("Echo number")
    axes[0].set_ylabel("Normalized echo magnitude")
    axes[0].set_title("SLSE Echo Train")
    axes[0].legend()

    axes[1].plot(
        narrow_acq.acquisition_times_seconds * 1e6,
        _normalized_real(narrow_acq.clean_echo_signal),
        label="zero-width EFG",
    )
    axes[1].plot(
        broad_acq.acquisition_times_seconds * 1e6,
        _normalized_real(broad_acq.clean_echo_signal),
        label="broadened EFG",
    )
    if noise is not None:
        axes[1].plot(
            broad_acq.acquisition_times_seconds * 1e6,
            _normalized_real(broad_acq.echo_signal),
            color="0.4",
            alpha=0.65,
            label=f"broadened + noise ({args.noise_snr:g} SNR)",
        )
    axes[1].set_xlabel("Time from echo center (us)")
    axes[1].set_ylabel("Normalized real signal")
    axes[1].set_title("Acquired Echo Window")
    axes[1].legend()

    axes[2].plot(
        narrow_acq.spectrum_frequencies_hz / 1e3,
        _normalized_magnitude(narrow_acq.clean_spectrum),
        label="zero-width EFG",
    )
    axes[2].plot(
        broad_acq.spectrum_frequencies_hz / 1e3,
        _normalized_magnitude(broad_acq.clean_spectrum),
        label="broadened EFG",
    )
    if noise is not None:
        axes[2].plot(
            broad_acq.spectrum_frequencies_hz / 1e3,
            _normalized_magnitude(broad_acq.spectrum),
            color="0.35",
            alpha=0.65,
            label="broadened + noise",
        )
    if broad_acq.deconvolution is not None:
        axes[2].plot(
            broad_acq.spectrum_frequencies_hz / 1e3,
            _normalized_magnitude(broad_acq.deconvolution.deconvolved_spectrum),
            "--",
            label="regularized deconv.",
        )
    axes[2].set_xlabel("RF offset from central line (kHz)")
    axes[2].set_ylabel("Normalized magnitude")
    axes[2].set_title("Acquired Echo Spectrum")
    axes[2].legend()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
