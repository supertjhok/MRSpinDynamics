"""Plot static weak-B0 NQR spectra for spin-1 or spin-3/2 sites."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    b0_b1_powder_average_grid,
    simulate_weak_b0_spectrum,
    single_crystal_orientation,
)


def _orientations(args):
    if args.orientation == "single":
        return single_crystal_orientation(
            alpha=0.0,
            beta=np.pi / 2.0,
            b0_alpha=args.b0_alpha,
            b0_beta=args.b0_beta,
        )
    return b0_b1_powder_average_grid(
        args.n_theta,
        args.n_phi,
        args.n_chi,
        b1_b0_angle=np.deg2rad(args.b1_b0_angle),
    )


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    return values / scale if scale > 0 else values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Spin support: this static-transition example supports spin=1 and "
            "spin=3/2. It does not simulate spin-3/2 pulsed dynamics."
        ),
    )
    parser.add_argument("--spin", choices=["1", "1.5"], default="1.5")
    parser.add_argument("--isotope", default="35Cl")
    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--quadrupole-mhz", type=float, default=30.0)
    parser.add_argument(
        "--gamma-mhz-per-t",
        type=float,
        default=4.171,
        help="Gyromagnetic ratio gamma / 2pi in MHz/T.",
    )
    parser.add_argument(
        "--transition",
        default=None,
        help="Optional zero-field transition label, for example x for spin-1.",
    )
    parser.add_argument(
        "--b0-mt",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0],
        help="Static field magnitudes in mT.",
    )
    parser.add_argument("--broadening-hz", type=float, default=200.0)
    parser.add_argument("--points", type=int, default=1024)
    parser.add_argument(
        "--weak-ratio-threshold",
        type=float,
        default=0.05,
        help="Warn when |gamma B0| / nu_ref exceeds this value.",
    )
    parser.add_argument(
        "--orientation",
        choices=["powder", "single"],
        default="powder",
        help="Average static-field directions over a powder or use one PAS direction.",
    )
    parser.add_argument("--b0-alpha", type=float, default=0.0)
    parser.add_argument("--b0-beta", type=float, default=0.0)
    parser.add_argument("--n-theta", type=int, default=8)
    parser.add_argument("--n-phi", type=int, default=16)
    parser.add_argument("--n-chi", type=int, default=8)
    parser.add_argument(
        "--b1-b0-angle",
        type=float,
        default=90.0,
        help="Lab RF-field angle relative to B0 in degrees for powder averaging.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    site = QuadrupolarSite(
        spin=float(args.spin),
        isotope=args.isotope,
        quadrupole_frequency_hz=args.quadrupole_mhz * 1e6,
        eta=args.eta,
        gamma_hz_per_t=args.gamma_mhz_per_t * 1e6,
    )
    orientations = _orientations(args)

    results = [
        simulate_weak_b0_spectrum(
            site,
            b0_mt * 1e-3,
            orientations=orientations,
            transition_label=args.transition,
            broadening_hz=args.broadening_hz,
            points=args.points,
            weak_ratio_threshold=args.weak_ratio_threshold,
        )
        for b0_mt in args.b0_mt
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    for b0_mt, result in zip(args.b0_mt, results):
        axes[0].plot(
            result.offsets_hz / 1e3,
            _normalized(result.spectrum),
            label=(
                f"{b0_mt:g} mT, "
                f"|gamma B0|/nu={result.max_perturbation_ratio:.2e}"
            ),
        )
    axes[0].set_xlabel("Offset from zero-field line (kHz)")
    axes[0].set_ylabel("Normalized intensity")
    axes[0].set_title(f"Weak-B0 NQR Spectrum, I={args.spin}")
    axes[0].legend()

    strongest = results[-1]
    shifts = np.array(
        [
            item.frequency_hz - strongest.reference_frequency_hz
            for item in strongest.transitions
        ]
    )
    intensities = np.array([item.intensity for item in strongest.transitions])
    axes[1].vlines(shifts / 1e3, 0.0, _normalized(intensities), color="tab:blue")
    axes[1].set_xlabel("Offset from zero-field line (kHz)")
    axes[1].set_ylabel("Relative transition intensity")
    axes[1].set_title(f"Resolved Transitions at {args.b0_mt[-1]:g} mT")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
