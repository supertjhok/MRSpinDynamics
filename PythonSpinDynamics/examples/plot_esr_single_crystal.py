"""Plot single-crystal ESR resonance shifts from an anisotropic g tensor."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    effective_g_value,
    resonance_field_tesla,
    resonance_frequency_hz,
    simulate_field_sweep,
    single_crystal_orientation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--g",
        type=float,
        nargs=3,
        default=[2.00, 2.06, 2.22],
        metavar=("GX", "GY", "GZ"),
        help="Principal g values for the electron spin.",
    )
    parser.add_argument(
        "--microwave-ghz",
        type=float,
        default=9.5,
        help="Fixed microwave frequency for the field sweep.",
    )
    parser.add_argument(
        "--broadening-mt",
        type=float,
        default=0.25,
        help="Gaussian field broadening in mT.",
    )
    parser.add_argument("--points", type=int, default=1201)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    system = ESRSpinSystem(g_tensor=args.g)
    microwave_hz = args.microwave_ghz * 1e9

    # Rotate B0 in the x-z plane of the g-tensor principal-axis frame.  For a
    # diagonal g tensor, this sweeps smoothly between gx and gz.
    angles = np.linspace(0.0, 90.0, 181)
    beta = np.deg2rad(angles)
    directions = np.column_stack(
        [np.sin(beta), np.zeros_like(beta), np.cos(beta)]
    )
    g_eff = np.array([effective_g_value(system, direction) for direction in directions])
    resonance_fields = np.array(
        [
            resonance_field_tesla(system, microwave_hz, direction)
            for direction in directions
        ]
    )

    # Pick three representative orientations to show as actual spectra.  The
    # transition is the same spin-1/2 line, but the field axis moves because
    # the effective g value changes with orientation.
    selected_angles = [0.0, 45.0, 90.0]
    spectra = []
    for angle in selected_angles:
        orientation = single_crystal_orientation(alpha=0.0, beta=np.deg2rad(angle))
        spectra.append(
            simulate_field_sweep(
                system,
                microwave_hz,
                orientations=orientation,
                broadening_tesla=args.broadening_mt * 1e-3,
                points=args.points,
            )
        )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)

    axes[0].plot(angles, g_eff, color="tab:blue", label="g_eff")
    axes[0].set_xlabel("B0 polar angle in g frame (deg)")
    axes[0].set_ylabel("Effective g value")
    axes[0].set_title("Single-Crystal Angular Dependence")
    ax_field = axes[0].twinx()
    ax_field.plot(angles, 1e3 * resonance_fields, color="tab:orange", label="B_res")
    ax_field.set_ylabel("Resonance field (mT)")

    handles, labels = axes[0].get_legend_handles_labels()
    extra_handles, extra_labels = ax_field.get_legend_handles_labels()
    axes[0].legend(handles + extra_handles, labels + extra_labels, loc="best")

    for angle, result in zip(selected_angles, spectra):
        scale = float(np.max(result.spectrum))
        spectrum = result.spectrum / scale if scale > 0 else result.spectrum
        axes[1].plot(
            1e3 * result.fields_tesla,
            spectrum,
            label=f"{angle:g} deg",
        )
    axes[1].set_xlabel("Static field B0 (mT)")
    axes[1].set_ylabel("Normalized intensity")
    axes[1].set_title(f"{args.microwave_ghz:g} GHz Field Sweeps")
    axes[1].legend(title="B0 angle")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
