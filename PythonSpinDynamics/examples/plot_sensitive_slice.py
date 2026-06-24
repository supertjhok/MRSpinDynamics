"""The sensitive slice in a non-uniform field is neither flat nor uniform.

In an inhomogeneous B0 (single-sided / ex-situ NMR, internal gradients), an RF
pulse excites the spins whose frequency falls in its band, so the excited region
follows the curved iso-B0 contours rather than a flat plane, and its intensity
varies with the transmit/receive B1. ``imaging_slice_sensitivity`` maps this
sensitive slice in real space from the same B0/B1 maps the imaging workflows use.

This example builds a single-sided-like field (B0 rising with depth and curving
across the probe) and a surface-coil B1, then shows:

1. The B0 field with iso-frequency contours -- each is a candidate slice, and
   they are curved.
2. The sensitive slice at one excitation frequency -- a curved band shaded by B1.
3. Several slices at different excitation frequencies -- curved bands at
   different depths (frequency selects depth, but not a flat plane).
4. The slice center and peak intensity versus position -- quantifying that the
   slice is neither flat (curved center) nor uniform (varying intensity).

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
            "Visualize the curved, non-uniform sensitive slice of an excitation "
            "in a non-uniform B0/B1 field."
        )
    )
    parser.add_argument("--pixels", type=int, default=61, help="Grid size per side.")
    parser.add_argument("--b0-depth-hz", type=float, default=24000.0,
                        help="B0 change with depth across the FOV (Hz).")
    parser.add_argument("--b0-curvature-hz", type=float, default=14000.0,
                        help="B0 curvature across the probe (Hz), sets slice bend.")
    parser.add_argument("--excitation-duration", type=float, default=100.0e-6,
                        help="Hard 90 pulse duration (s); bandwidth ~ 1/duration.")
    parser.add_argument("--num-slices", type=int, default=4,
                        help="Excitation frequencies for the multi-slice panel.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _fields(args):
    n = int(args.pixels)
    axis = np.linspace(-1.0, 1.0, n)
    xx = axis[:, np.newaxis] * np.ones((1, n))  # across the probe
    zz = np.ones((n, 1)) * axis[np.newaxis, :]  # depth
    # Single-sided-like B0 (Hz): rises with depth, curves across the probe.
    b0_hz = args.b0_depth_hz * 0.5 * zz + args.b0_curvature_hz * xx**2
    b0 = 2.0 * np.pi * b0_hz
    # Surface-coil B1: strongest near the probe (low depth) and on axis.
    b1 = np.exp(-(zz + 1.0) / 1.1) * np.exp(-((xx / 1.3) ** 2))
    return n, axis, b0_hz, b0, b1


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.workflows import imaging_slice_sensitivity

    n, axis, b0_hz, b0, b1 = _fields(args)
    rho = np.ones((n, n))
    maps = dict(b0_map=b0, b1_tx_map=b1, b1_rx_map=b1,
                excitation_duration=args.excitation_duration)

    # On-axis frequency range used for the multi-slice sweep.
    f_lo, f_hi = float(np.percentile(b0_hz, 25)), float(np.percentile(b0_hz, 75))
    f_centers = np.linspace(f_lo, f_hi, int(args.num_slices))
    f0 = float(np.median(f_centers))

    slice0 = imaging_slice_sensitivity(rho, center_frequency=2 * np.pi * f0, **maps)
    composite = np.zeros((n, n))
    for fc in f_centers:
        composite = np.maximum(
            composite,
            imaging_slice_sensitivity(rho, center_frequency=2 * np.pi * fc, **maps).sensitivity,
        )

    print(f"grid {n}x{n}; B0 {b0_hz.min()/1e3:.1f}..{b0_hz.max()/1e3:.1f} kHz; "
          f"pulse {args.excitation_duration*1e6:.0f} us (~{1/args.excitation_duration/1e3:.0f} kHz BW)")
    print(f"slice at f0={f0/1e3:.1f} kHz: peak sensitivity {slice0.sensitivity.max():.3f}")

    # Slice center is the on-resonance depth (where off-resonance crosses zero),
    # a purely geometric quantity; peak intensity is the B1-driven amplitude.
    sens = slice0.sensitivity
    off = np.abs(slice0.off_resonance)
    centers = axis[np.argmin(off, axis=1)]  # curved slice depth vs position
    peaks = sens.max(axis=1)  # non-uniform intensity vs position

    extent = [axis[0], axis[-1], axis[0], axis[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 9.4))

    im0 = axes[0, 0].imshow(b0_hz.T / 1e3, origin="lower", extent=extent,
                            aspect="equal", cmap="coolwarm")
    cs = axes[0, 0].contour(axis, axis, b0_hz.T / 1e3,
                            levels=f_centers / 1e3, colors="k", linewidths=1.0)
    axes[0, 0].clabel(cs, fontsize="x-small", fmt="%.0f")
    axes[0, 0].set_title("B0 field (kHz) + iso-frequency contours")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(sens.T, origin="lower", extent=extent, aspect="equal",
                            cmap="magma", vmin=0)
    axes[0, 1].contour(axis, axis, b0_hz.T / 1e3, levels=[f0 / 1e3],
                       colors="cyan", linewidths=1.2)
    axes[0, 1].set_title(f"sensitive slice at {f0/1e3:.1f} kHz (+ on-res contour)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(composite.T, origin="lower", extent=extent,
                            aspect="equal", cmap="magma", vmin=0)
    axes[1, 0].set_title(f"{int(args.num_slices)} slices vs excitation frequency")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    ax.plot(axis, centers, "o-", color="#1f77b4", markersize=3, label="slice depth")
    ax.set_xlabel("position across probe")
    ax.set_ylabel("slice depth (center)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax.set_ylim(axis[0], axis[-1])
    ax2 = ax.twinx()
    ax2.plot(axis, peaks, "s-", color="#d62728", markersize=3, label="peak intensity")
    ax2.set_ylabel("peak sensitivity", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(0, np.nanmax(peaks) * 1.1 if np.isfinite(np.nanmax(peaks)) else 1)
    axes[1, 1].set_title("slice is curved (depth) and non-uniform (intensity)")

    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
