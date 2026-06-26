"""Finite-length four-rod Halbach dipole field map.

This example uses the lowest-order Halbach approximation: four transverse,
diametrically magnetized rods around a cylindrical bore. The rods may be
cylinders or square rods, and their finite length is included by summing a
volume cubature of point dipoles. It shows the 3-D field sampled in the bore:

1. The mid-plane Larmor-frequency map.
2. The mid-plane field components.
3. The on-axis finite-length falloff.
4. A transverse center-line profile and a uniformity map.

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
        description=(
            "Simulate the 3-D field of a finite four-rod Halbach dipole "
            "using sampled diametrically magnetized cylinders or square rods."
        )
    )
    parser.add_argument("--pixels", type=int, default=31, help="x/y grid points.")
    parser.add_argument("--z-pixels", type=int, default=41, help="z grid points.")
    parser.add_argument(
        "--center-radius",
        type=float,
        default=30.0,
        help="Rod-center radius from the bore axis (mm).",
    )
    parser.add_argument(
        "--rod-radius",
        type=float,
        default=8.0,
        help="Cylinder radius, or half the default square-rod width (mm).",
    )
    parser.add_argument(
        "--rod-width",
        type=float,
        help="Square-rod width (mm); defaults to 2 * --rod-radius.",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=80.0,
        help="Magnet length along z (mm).",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=28.0,
        help="Square transverse field of view in the bore (mm).",
    )
    parser.add_argument(
        "--z-span",
        type=float,
        default=100.0,
        help="Axial field-of-view length (mm).",
    )
    parser.add_argument(
        "--remanence",
        type=float,
        default=1.30,
        help="Magnet remanence Br (T); NdFeB N42 is about 1.30.",
    )
    parser.add_argument(
        "--rod-shape",
        choices=("cylinder", "square"),
        default="cylinder",
        help="Use cylindrical rods or axis-aligned square rods.",
    )
    parser.add_argument(
        "--field-angle",
        type=float,
        default=0.0,
        help="Desired central bore-field angle in degrees from +x.",
    )
    parser.add_argument(
        "--n-cross",
        type=int,
        default=5,
        help="Cross-section cubature samples per diameter/side.",
    )
    parser.add_argument(
        "--n-length",
        type=int,
        default=21,
        help="Cubature samples along each rod length.",
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

    from spin_dynamics.fields.magnetostatics import sample_halbach_dipole_field

    x_axis = np.linspace(-0.5 * args.fov, 0.5 * args.fov, args.pixels) * 1e-3
    y_axis = np.linspace(-0.5 * args.fov, 0.5 * args.fov, args.pixels) * 1e-3
    z_axis = np.linspace(-0.5 * args.z_span, 0.5 * args.z_span, args.z_pixels) * 1e-3

    fm = sample_halbach_dipole_field(
        x_axis,
        y_axis,
        z_axis,
        center_radius=args.center_radius * 1e-3,
        rod_radius=args.rod_radius * 1e-3,
        rod_width=None if args.rod_width is None else args.rod_width * 1e-3,
        length=args.length * 1e-3,
        remanence=args.remanence,
        rod_shape=args.rod_shape,
        field_angle=np.deg2rad(args.field_angle),
        n_cross=args.n_cross,
        n_length=args.n_length,
    )

    ix0 = int(np.argmin(np.abs(x_axis)))
    iy0 = int(np.argmin(np.abs(y_axis)))
    iz0 = int(np.argmin(np.abs(z_axis)))
    b_center = fm.b0_magnitude[ix0, iy0, iz0]
    freq_mhz = fm.larmor_hz / 1e6
    ppm = 1e6 * (fm.b0_magnitude[:, :, iz0] - b_center) / b_center
    extent_xy = [
        x_axis[0] * 1e3,
        x_axis[-1] * 1e3,
        y_axis[0] * 1e3,
        y_axis[-1] * 1e3,
    ]

    print(
        f"finite Halbach dipole: {args.rod_shape} rods, "
        f"R={args.center_radius:.1f} mm, length={args.length:.1f} mm"
    )
    print(
        f"  center |B0|: {b_center * 1e3:.1f} mT "
        f"({fm.larmor_hz[ix0, iy0, iz0] / 1e6:.2f} MHz)"
    )
    print(
        f"  mid-plane uniformity over FOV: "
        f"{float(np.nanmin(ppm)):.0f}..{float(np.nanmax(ppm)):.0f} ppm"
    )

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))

    im = axes[0, 0].imshow(
        freq_mhz[:, :, iz0].T,
        origin="lower",
        extent=extent_xy,
        aspect="equal",
        cmap="viridis",
    )
    axes[0, 0].set_title("mid-plane Larmor frequency (MHz)")
    axes[0, 0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    bx_mt = 1e3 * fm.b0_vector[:, :, iz0, 0]
    by_mt = 1e3 * fm.b0_vector[:, :, iz0, 1]
    im = axes[0, 1].imshow(
        bx_mt.T,
        origin="lower",
        extent=extent_xy,
        aspect="equal",
        cmap="coolwarm",
    )
    skip = max(1, args.pixels // 13)
    axes[0, 1].quiver(
        x_axis[::skip] * 1e3,
        y_axis[::skip] * 1e3,
        bx_mt[::skip, ::skip].T,
        by_mt[::skip, ::skip].T,
        color="k",
        width=0.003,
        scale=2.0e3,
    )
    axes[0, 1].set_title("Bx (mT) with transverse B direction")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(
        ppm.T,
        origin="lower",
        extent=extent_xy,
        aspect="equal",
        cmap="RdBu_r",
    )
    axes[0, 2].set_title("mid-plane |B0| deviation (ppm)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(z_axis * 1e3, fm.b0_magnitude[ix0, iy0, :] * 1e3, color="tab:blue")
    ax.axvline(-0.5 * args.length, color="0.5", linestyle="--", linewidth=1.0)
    ax.axvline(0.5 * args.length, color="0.5", linestyle="--", linewidth=1.0)
    ax.set_title("on-axis finite-length falloff")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("|B0| (mT)")

    ax = axes[1, 1]
    ax.plot(x_axis * 1e3, fm.b0_vector[:, iy0, iz0, 0] * 1e3, label="Bx")
    ax.plot(x_axis * 1e3, fm.b0_vector[:, iy0, iz0, 1] * 1e3, label="By")
    ax.plot(x_axis * 1e3, fm.b0_magnitude[:, iy0, iz0] * 1e3, label="|B0|")
    ax.set_title("transverse center-line field")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("field (mT)")
    ax.legend(frameon=False)

    im = axes[1, 2].imshow(
        fm.b0_gradient[:, :, iz0].T,
        origin="lower",
        extent=extent_xy,
        aspect="equal",
        cmap="magma",
    )
    axes[1, 2].set_title("mid-plane |grad |B0|| (T/m)")
    axes[1, 2].set_xlabel("x (mm)")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.flat:
        if ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]):
            ax.set_xlabel("x (mm)")
        if ax in (axes[0, 1], axes[0, 2], axes[1, 2]):
            ax.set_ylabel("y (mm)")

    fig.suptitle(
        "Finite-length four-rod Halbach dipole field from sampled magnetostatics",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
