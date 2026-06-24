"""Inhomogeneity effects in frequency-encoded imaging (spin-warp).

The frequency-encoded workflows accept the same B0/B1 maps as the phase-encoded
ones, so off-resonance and B1 inhomogeneity can be evaluated directly. This
example images one phantom four ways with ``run_spin_warp_imaging``:

1. Uniform fields -- the reference reconstruction.
2. A linear B0 gradient along the readout axis -- geometric distortion (the
   object stretches/shifts along readout, scaling as 1 / readout gradient).
3. A sub-voxel B0 spread (``num_offsets`` / ``offset_spread``) -- the T2* point-
   spread function, which blurs along the readout axis. A spin echo refocuses
   the static spread at the echo, so it does not decay the train.
4. A transmit-B1 shading -- a flip-angle gradient that shades the image.

Readout is along x and phase encode along z, so B0 effects appear along x.
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
            "Show how B0 distortion, sub-voxel B0 spread (T2*), and B1 shading "
            "affect a frequency-encoded (spin-warp) image."
        )
    )
    parser.add_argument("--pixels", type=int, default=32, help="Image size.")
    parser.add_argument("--b0-gradient-hz", type=float, default=1500.0,
                        help="Edge-to-edge B0 gradient along phase encode (Hz).")
    parser.add_argument("--b0-spread-hz", type=float, default=1500.0,
                        help="Sub-voxel B0 spread half-width (Hz).")
    parser.add_argument("--num-offsets", type=int, default=9,
                        help="Sub-voxel B0 samples averaged per voxel.")
    parser.add_argument("--b1-min", type=float, default=0.4,
                        help="Minimum transmit-B1 fraction across the FOV.")
    parser.add_argument("--readout-time", type=float, default=2.0e-3,
                        help="Readout duration (s); shorter resists B0 distortion.")
    parser.add_argument("--fov", type=float, default=0.02, help="Field of view (m).")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _phantom(n: int) -> np.ndarray:
    """A grid of bars plus a disk, so distortion and blurring are visible."""

    rho = np.zeros((n, n), dtype=np.float64)
    yy, xx = np.mgrid[0:n, 0:n]
    disk = (xx - n * 0.34) ** 2 + (yy - n * 0.5) ** 2 <= (n * 0.16) ** 2
    rho[disk] = 1.0
    for col in range(int(n * 0.58), int(n * 0.86), 3):
        rho[(xx >= col) & (xx < col + 1) & (yy >= n * 0.25) & (yy < n * 0.75)] = 0.9
    return rho


def _panel(ax, image, rho, title):
    img = np.abs(image)
    if img.max() > 0:
        img = img / img.max() * rho.max()
    ax.imshow(img.T, origin="lower", cmap="gray", vmin=0, vmax=rho.max())
    ax.set_title(title, fontsize="medium")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.workflows import run_spin_warp_imaging

    n = int(args.pixels)
    rho = _phantom(n)
    fov = (args.fov, args.fov)
    common = dict(fov=fov, readout_time=args.readout_time)

    # B0 linear gradient along the phase-encode axis (z = second index), in
    # rad/s: each z row gets a constant readout offset, so its readout position
    # shifts along x -> the object shears (cleaner to read than a pure stretch).
    ramp = np.linspace(-0.5, 0.5, n) * (2.0 * np.pi * args.b0_gradient_hz)
    b0_gradient = np.ones((n, 1)) * ramp[np.newaxis, :]

    # Transmit-B1 shading along x.
    b1 = np.linspace(args.b1_min, 1.0, n)[:, np.newaxis] * np.ones((1, n))

    reference = run_spin_warp_imaging(rho, **common)
    distorted = run_spin_warp_imaging(rho, b0_map=b0_gradient, **common)
    blurred = run_spin_warp_imaging(
        rho, num_offsets=int(args.num_offsets),
        offset_spread=2.0 * np.pi * args.b0_spread_hz, **common,
    )
    shaded = run_spin_warp_imaging(rho, b1_tx_map=b1, **common)

    print(f"image {n}x{n}, FOV {args.fov*1e3:.0f} mm, readout {args.readout_time*1e3:.1f} ms")
    print(f"readout bandwidth ~ {n/args.readout_time/1e3:.1f} kHz; "
          f"B0 gradient {args.b0_gradient_hz:.0f} Hz edge-to-edge "
          f"(~{args.b0_gradient_hz*args.readout_time:.1f} px peak shift)")
    print(f"sub-voxel B0 spread +/-{args.b0_spread_hz:.0f} Hz over {args.num_offsets} "
          f"samples -> readout T2* blur")

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.8))
    _panel(axes[0, 0], reference.image[:, :, 0], rho, "uniform fields (reference)")
    _panel(axes[0, 1], distorted.image[:, :, 0], rho,
            f"B0 gradient {args.b0_gradient_hz:.0f} Hz -> readout shear")
    _panel(axes[1, 0], blurred.image[:, :, 0], rho,
            f"B0 spread +/-{args.b0_spread_hz:.0f} Hz -> T2* readout blur")
    _panel(axes[1, 1], shaded.image[:, :, 0], rho,
            f"B1 shading {args.b1_min:.1f}->1.0 -> intensity shading")
    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
