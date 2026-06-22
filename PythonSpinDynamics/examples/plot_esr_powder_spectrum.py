"""Plot powder ESR spectra for an anisotropic spin-1/2 g tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    powder_average_grid,
    simulate_field_sweep_distribution,
    simulate_frequency_spectrum_distribution,
    static_disorder_grid,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return values / scale if scale > 0 else values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g",
        type=float,
        nargs=3,
        default=[2.00, 2.08, 2.24],
        metavar=("GX", "GY", "GZ"),
        help="Principal g values for the electron spin.",
    )
    parser.add_argument("--microwave-ghz", type=float, default=9.5)
    parser.add_argument("--b0-mt", type=float, default=340.0)
    parser.add_argument("--field-broadening-mt", type=float, default=0.35)
    parser.add_argument("--frequency-broadening-mhz", type=float, default=8.0)
    parser.add_argument("--points", type=int, default=1401)
    parser.add_argument("--n-theta", type=int, default=16)
    parser.add_argument("--n-phi", type=int, default=32)
    parser.add_argument("--n-chi", type=int, default=4)
    parser.add_argument(
        "--lineshape",
        choices=["gaussian", "lorentzian"],
        default="gaussian",
        help="Absorption lineshape used to broaden orientation-resolved lines.",
    )
    parser.add_argument(
        "--detection-mode",
        choices=["absorption", "derivative"],
        default="absorption",
        help="Plot absorption or first-derivative CW ESR spectra.",
    )
    parser.add_argument(
        "--g-strain",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DGX", "DGY", "DGZ"),
        help="Gaussian standard deviations for diagonal g strain.",
    )
    parser.add_argument(
        "--field-strain-mt",
        type=float,
        default=0.0,
        help="Gaussian applied-field offset standard deviation in mT.",
    )
    parser.add_argument("--strain-points", type=int, default=3)
    parser.add_argument(
        "--b1-b0-angle",
        type=float,
        default=90.0,
        help="Lab microwave B1 angle relative to B0 in degrees.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = ESRSpinSystem(g_tensor=args.g)
    disorder = static_disorder_grid(
        system,
        g_std=args.g_strain,
        field_std_tesla=args.field_strain_mt * 1e-3,
        g_points=args.strain_points,
        field_points=args.strain_points,
    )
    orientations = powder_average_grid(
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        n_chi=args.n_chi,
        b1_b0_angle=np.deg2rad(args.b1_b0_angle),
    )

    # Conventional continuous-wave ESR is usually field swept at fixed
    # microwave frequency.  The anisotropic g tensor turns one spin-1/2
    # transition into a powder pattern because each crystallite orientation has
    # its own effective g value and therefore its own resonant field.  Optional
    # g strain and field offsets add static-disorder samples on top of the
    # orientation grid.
    field_result = simulate_field_sweep_distribution(
        disorder,
        args.microwave_ghz * 1e9,
        orientations=orientations,
        broadening_tesla=args.field_broadening_mt * 1e-3,
        points=args.points,
        lineshape=args.lineshape,
        detection_mode=args.detection_mode,
    )

    # The same orientation distribution can be viewed as a frequency-swept
    # spectrum at fixed B0.  This is a useful cross-check because the high-g
    # edge moves to lower field in a field sweep but to higher frequency in a
    # frequency sweep.
    frequency_result = simulate_frequency_spectrum_distribution(
        disorder,
        args.b0_mt * 1e-3,
        orientations=orientations,
        broadening_hz=args.frequency_broadening_mhz * 1e6,
        points=args.points,
        lineshape=args.lineshape,
        detection_mode=args.detection_mode,
    )

    line_fields = np.array([line.field_tesla for line in field_result.lines])
    line_frequencies = np.array([line.frequency_hz for line in frequency_result.lines])
    line_intensities = np.array([line.intensity for line in field_result.lines])

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)

    axes[0, 0].plot(
        1e3 * field_result.fields_tesla,
        _normalized(field_result.spectrum),
        color="tab:blue",
    )
    axes[0, 0].set_xlabel("Static field B0 (mT)")
    axes[0, 0].set_ylabel("Normalized signal")
    axes[0, 0].set_title(f"Field Sweep at {args.microwave_ghz:g} GHz")

    axes[0, 1].plot(
        frequency_result.frequencies_hz / 1e9,
        _normalized(frequency_result.spectrum),
        color="tab:orange",
    )
    axes[0, 1].set_xlabel("Frequency (GHz)")
    axes[0, 1].set_ylabel("Normalized signal")
    axes[0, 1].set_title(f"Frequency Sweep at {args.b0_mt:g} mT")

    axes[1, 0].hist(
        1e3 * line_fields,
        bins=64,
        weights=line_intensities,
        color="tab:blue",
        alpha=0.8,
    )
    axes[1, 0].set_xlabel("Orientation-resolved resonance field (mT)")
    axes[1, 0].set_ylabel("Weighted counts")
    axes[1, 0].set_title("Powder Orientation Distribution")

    axes[1, 1].hist(
        line_frequencies / 1e9,
        bins=64,
        weights=line_intensities,
        color="tab:orange",
        alpha=0.8,
    )
    axes[1, 1].set_xlabel("Orientation-resolved frequency (GHz)")
    axes[1, 1].set_ylabel("Weighted counts")
    axes[1, 1].set_title("Same Orientations at Fixed Field")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
