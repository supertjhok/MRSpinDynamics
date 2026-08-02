"""Plot absolute complex reciprocity maps for a two-channel receive array.

Two orthogonal solenoids are sampled on a transverse imaging plane. The example
shows the absolute ``|B1-|`` sensitivity in microtesla per ampere, its complex
phase, and the legacy rho-mean-normalized magnitude maps. At the center the two channels differ by 90 degrees, so
the example also provides a concrete convention check for quadrature receive.

This is the Phase-1 uncoupled geometry model. It preserves receiver identity and
complex phase but does not yet propagate a channel axis through CPMG k-space or
include mutual coupling and correlated receiver noise.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot complex B1- maps for two orthogonal receive coils."
    )
    parser.add_argument("--pixels", type=int, default=81, help="Grid size per side.")
    parser.add_argument(
        "--extent-mm",
        type=float,
        default=20.0,
        help="Square transverse field of view in millimeters.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path. If omitted, show the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.experiment import (
        Hardware,
        ImagingPlane,
        Phantom,
        RxArray,
        RxCoil,
        SolenoidCoil,
        solve_receive_sensitivities,
    )

    pixels = int(args.pixels)
    if pixels < 3:
        raise ValueError("pixels must be at least 3")
    extent_m = float(args.extent_mm) * 1e-3
    if extent_m <= 0:
        raise ValueError("extent-mm must be positive")

    axis_m = np.linspace(-extent_m / 2.0, extent_m / 2.0, pixels)
    xx, yy = np.meshgrid(axis_m, axis_m, indexing="ij")
    sample_radius = 0.42 * extent_m
    rho = ((xx**2 + yy**2) <= sample_radius**2).astype(np.float64)
    phantom = Phantom(rho=rho)

    coil_common = dict(radius_m=0.015, length_m=0.03, turns=10)
    receivers = RxArray(
        (
            RxCoil(SolenoidCoil(axis="x", **coil_common)),
            RxCoil(SolenoidCoil(axis="y", **coil_common)),
        )
    )
    hardware = Hardware(
        b0=None,  # default B0 direction is +z
        rx_coil=receivers,
        plane=ImagingPlane(
            extent_m=(extent_m, extent_m),
            plane="xy",
        ),
    )
    sensitivity = solve_receive_sensitivities(phantom, hardware)

    center = (pixels // 2, pixels // 2)
    center_values = sensitivity.b1_minus_t_per_a[(slice(None), *center)]
    phase_difference = np.angle(center_values[1] / center_values[0])
    print("Two-channel reciprocal receive sensitivity")
    for index, (value, normalization) in enumerate(
        zip(center_values, sensitivity.normalization_t_per_a)
    ):
        print(
            f"  rx{index}: center B1-={value * 1e6:.6g} uT/A; "
            f"rho-mean |B1-|={normalization * 1e6:.6g} uT/A"
        )
    print(f"  center phase(rx1/rx0)={np.degrees(phase_difference):.3f} deg")

    extent_mm = 1e3 * np.array([axis_m[0], axis_m[-1], axis_m[0], axis_m[-1]])
    absolute_ut_per_a = 1e6 * np.abs(sensitivity.b1_minus_t_per_a)
    phase_deg = np.degrees(np.angle(sensitivity.b1_minus_t_per_a))
    normalized = sensitivity.normalized_magnitude
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    for channel in range(2):
        magnitude_image = axes[channel, 0].imshow(
            absolute_ut_per_a[channel].T,
            origin="lower",
            extent=extent_mm,
            cmap="magma",
            aspect="equal",
        )
        axes[channel, 0].set_title(f"rx{channel} |B1-| (uT/A)")
        fig.colorbar(magnitude_image, ax=axes[channel, 0], fraction=0.046, pad=0.04)

        phase_image = axes[channel, 1].imshow(
            phase_deg[channel].T,
            origin="lower",
            extent=extent_mm,
            cmap="twilight",
            vmin=-180.0,
            vmax=180.0,
            aspect="equal",
        )
        axes[channel, 1].set_title(f"rx{channel} phase(B1-) (deg)")
        fig.colorbar(phase_image, ax=axes[channel, 1], fraction=0.046, pad=0.04)

        normalized_image = axes[channel, 2].imshow(
            normalized[channel].T,
            origin="lower",
            extent=extent_mm,
            cmap="viridis",
            aspect="equal",
        )
        axes[channel, 2].set_title(f"rx{channel} normalized magnitude")
        fig.colorbar(normalized_image, ax=axes[channel, 2], fraction=0.046, pad=0.04)

    for axis in axes.flat:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    fig.suptitle(
        "Uncoupled reciprocal receive array: absolute complex B1- and legacy maps"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
