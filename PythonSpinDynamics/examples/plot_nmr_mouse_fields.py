"""NMR-MOUSE B0/B1 field maps and the depth-resolved sensitive slice.

The NMR-MOUSE is a single-sided (open) magnet: two antiparallel permanent-magnet
bars on an iron yoke produce a stray field *above* the surface with a strong,
smoothly decaying static gradient. There is no homogeneous volume -- instead the
field defines a curved sensitive slice at each excitation frequency, and sweeping
the frequency (or moving the probe) scans that slice through depth. That static
gradient is also a built-in, always-on diffusion encoder.

This example builds the field from first principles with
``spin_dynamics.fields.magnetostatics`` (analytic charged-sheet B0 of the bars +
an iron-yoke image for the flux return path, Biot-Savart B1 of a surface coil),
then maps the sensitive slice at several depths with ``imaging_slice_sensitivity``.
It shows:

1. ``|B0|`` above the probe with iso-frequency contours -- each is a depth slice.
2. The static gradient ``|grad |B0|||`` (T/m), which sets depth resolution and
   diffusion weighting.
3. The transverse B1 of the surface coil.
4. The sensitive slices at several frequencies (curved bands at different depths).
5. ``|B0|``, gradient, and on-axis sensitivity versus depth.

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
            "NMR-MOUSE B0/B1 field maps (analytic permanent magnets + iron yoke, "
            "Biot-Savart coil) and the depth-resolved sensitive slice."
        )
    )
    parser.add_argument("--pixels", type=int, default=121, help="Grid size per side.")
    parser.add_argument("--gap", type=float, default=12.0,
                        help="Gap between the two magnet bars (mm).")
    parser.add_argument("--magnet-mm", type=float, default=20.0,
                        help="Bar width and height (mm).")
    parser.add_argument("--remanence", type=float, default=1.30,
                        help="Magnet remanence Br (T); NdFeB N42 ~ 1.30.")
    parser.add_argument("--coil-radius", type=float, default=8.0,
                        help="Surface-coil radius (mm).")
    parser.add_argument("--excitation-duration", type=float, default=10.0e-6,
                        help="RF pulse duration (s); bandwidth ~ 1/duration.")
    parser.add_argument("--num-slices", type=int, default=5,
                        help="Sensitive slices (excitation frequencies) to show.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields.magnetostatics import (
        circular_loop, nmr_mouse_magnets, sample_magnet_field, GAMMA_PROTON,
    )
    from spin_dynamics.workflows import imaging_slice_sensitivity

    mw = args.magnet_mm * 1e-3
    gap = args.gap * 1e-3
    bars, yoke = nmr_mouse_magnets(
        magnet_width=mw, magnet_height=mw, gap=gap, remanence=args.remanence,
    )
    # Sample the stray field above the magnet faces (the accessible region).
    span = gap + mw
    x_axis = np.linspace(-span, span, int(args.pixels))
    y0 = mw + 1.0e-3
    y_axis = np.linspace(y0, y0 + 1.2 * span, int(args.pixels))
    coil = circular_loop((0.0, y0 + 0.5 * span, 0.0), args.coil_radius * 1e-3,
                         axis="y", n_segments=120)

    fm = sample_magnet_field(x_axis, y_axis, bars, yoke_y=yoke,
                             coil_segments=coil, coil_current=1.0)
    b0_mhz = fm.larmor_hz / 1e6
    rho = np.ones_like(fm.b0_magnitude)
    b0_rad = GAMMA_PROTON * fm.b0_magnitude
    b1_rel = fm.b1_transverse / (fm.b1_transverse.max() or 1.0)
    fov = (float(x_axis[-1] - x_axis[0]), float(y_axis[-1] - y_axis[0]))

    # Pick excitation frequencies spanning the accessible depth (on the x=0 axis).
    ix0 = int(np.argmin(np.abs(x_axis)))
    f_axis_mhz = b0_mhz[ix0, :]
    f_lo, f_hi = np.percentile(f_axis_mhz, 20), np.percentile(f_axis_mhz, 80)
    f_centers = np.linspace(f_lo, f_hi, int(args.num_slices))

    def _slice(fc_mhz):
        return imaging_slice_sensitivity(
            rho, center_frequency=2 * np.pi * fc_mhz * 1e6,
            excitation_duration=args.excitation_duration,
            b0_map=b0_rad, b1_tx_map=b1_rel, b1_rx_map=b1_rel, fov=fov,
        ).sensitivity

    composite = np.zeros_like(rho)
    for fc in f_centers:
        composite = np.maximum(composite, _slice(fc))
    f_mid = float(np.median(f_centers))
    single = _slice(f_mid)

    print(f"NMR-MOUSE: gap {args.gap:.0f} mm, bars {args.magnet_mm:.0f} mm, "
          f"Br {args.remanence:.2f} T")
    print(f"  |B0| over sample : {fm.b0_magnitude.min()*1e3:.0f}..{fm.b0_magnitude.max()*1e3:.0f} mT "
          f"({b0_mhz.min():.1f}..{b0_mhz.max():.1f} MHz)")
    print(f"  static gradient  : {fm.b0_gradient.min():.1f}..{fm.b0_gradient.max():.1f} T/m")
    print(f"  showing {args.num_slices} slices at {f_lo:.1f}..{f_hi:.1f} MHz")

    extent = [x_axis[0] * 1e3, x_axis[-1] * 1e3, y_axis[0] * 1e3, y_axis[-1] * 1e3]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 9.0))

    im = axes[0, 0].imshow(b0_mhz.T, origin="lower", extent=extent, aspect="auto",
                           cmap="viridis")
    cs = axes[0, 0].contour(x_axis * 1e3, y_axis * 1e3, b0_mhz.T,
                            levels=f_centers, colors="w", linewidths=0.8)
    axes[0, 0].clabel(cs, fontsize="x-small", fmt="%.1f")
    axes[0, 0].set_title("|B0| Larmor freq (MHz) + iso-freq depth slices")
    axes[0, 0].set_ylabel("depth y (mm)")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(fm.b0_gradient.T, origin="lower", extent=extent,
                           aspect="auto", cmap="magma")
    axes[0, 1].set_title("static gradient |grad|B0|| (T/m)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow((b1_rel).T, origin="lower", extent=extent, aspect="auto",
                           cmap="cividis", vmin=0, vmax=1)
    axes[0, 2].set_title("transverse B1 (relative)")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(single.T, origin="lower", extent=extent, aspect="auto",
                           cmap="inferno", vmin=0)
    axes[1, 0].set_title(f"sensitive slice at {f_mid:.1f} MHz")
    axes[1, 0].set_xlabel("lateral x (mm)")
    axes[1, 0].set_ylabel("depth y (mm)")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(composite.T, origin="lower", extent=extent, aspect="auto",
                           cmap="inferno", vmin=0)
    axes[1, 1].set_title(f"{args.num_slices} sensitive slices vs frequency (depth)")
    axes[1, 1].set_xlabel("lateral x (mm)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    ax = axes[1, 2]
    yz = y_axis * 1e3
    ax.plot(yz, fm.b0_magnitude[ix0, :] * 1e3, "-", color="tab:blue", label="|B0| (mT)")
    ax.set_xlabel("depth y (mm)")
    ax.set_ylabel("|B0| (mT)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(yz, fm.b0_gradient[ix0, :], "--", color="tab:red", label="gradient (T/m)")
    ax2.set_ylabel("gradient (T/m)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    axes[1, 2].set_title("on-axis depth profile (x=0)")

    fig.suptitle("NMR-MOUSE: single-sided B0/B1 and the depth-resolved sensitive slice",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
