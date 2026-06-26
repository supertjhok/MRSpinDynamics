"""Multi-slice 3-D imaging of a structured phantom in a Halbach-like field.

A Halbach magnet has a compact, mostly-uniform B0 with a *mild* residual
inhomogeneity (a smooth saddle across the bore) and a transmit/receive B1 that
falls off gently away from the coil axis. This example builds such a field over a
small 3-D phantom and acquires it with the true-3-D slice-selective multi-slice
workflow (``run_multislice_imaging``): one 3-D spin ensemble lives in the actual
``(B0, B1)`` field, and every slice is excited and read out through the
moving-isochromat engine. Because the slice is selected by *total* off-resonance
(slice gradient plus local B0), the mild B0 gently shifts/curves the slices and
warps the readout -- the real-magnet behaviour, not a flat-slice cartoon.

It shows:

1. A few acquired slices (magnitude) next to the ground-truth slices.
2. The Halbach B0 saddle and the B1 shading that produce the distortions.
3. A 3-D voxel rendering of the reconstructed volume.

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
            "True-3-D multi-slice imaging of a 3-D phantom in a mild Halbach "
            "(B0, B1) field, showing acquired slices and the 3-D reconstruction."
        )
    )
    parser.add_argument("--pixels", type=int, default=16,
                        help="In-plane grid size per side (readout x phase encode).")
    parser.add_argument("--slices", type=int, default=5,
                        help="Number of through-plane slices.")
    parser.add_argument("--fov", type=float, default=0.02,
                        help="Isotropic field of view (m).")
    parser.add_argument("--b0-inhomogeneity-hz", type=float, default=300.0,
                        help="Peak Halbach B0 saddle amplitude over the FOV (Hz).")
    parser.add_argument("--b1-inhomogeneity", type=float, default=0.25,
                        help="Fractional B1 falloff toward the bore edge (0..1).")
    parser.add_argument("--slice-thickness-voxels", type=float, default=1.2,
                        help="Target slice thickness in through-plane voxels.")
    parser.add_argument("--num-substeps", type=int, default=40,
                        help="RF samples in the shaped slice pulse.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _phantom(nx, ny, nz):
    """A structured 3-D phantom: spheres and a bar at different slices."""

    rho = np.zeros((nx, ny, nz))
    xx, yy, zz = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )

    def sphere(cx, cy, cz, r, value=1.0):
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= r**2
        rho[mask] = value

    cy = ny // 2
    sphere(nx * 0.30, max(cy - 1, 0), nz * 0.32, max(nx, nz) * 0.16, 1.0)
    sphere(nx * 0.66, cy, nz * 0.62, max(nx, nz) * 0.13, 0.8)
    sphere(nx * 0.42, min(cy + 1, ny - 1), nz * 0.50, max(nx, nz) * 0.10, 0.6)
    # A bar spanning the readout axis in the centre slice, for in-plane structure.
    bar_z = int(round(nz * 0.30))
    rho[int(nx * 0.2):int(nx * 0.8), cy, bar_z] = 0.9
    return rho


def _halbach_fields(nx, ny, nz, fov, b0_amp_hz, b1_falloff):
    """Mild Halbach-like B0 saddle (rad/s) and B1 shading (relative)."""

    ax_x = np.linspace(-1.0, 1.0, nx)
    ax_y = np.linspace(-1.0, 1.0, ny)
    ax_z = np.linspace(-1.0, 1.0, nz)
    X, Y, Z = np.meshgrid(ax_x, ax_y, ax_z, indexing="ij")
    # Saddle: a smooth, sign-changing residual typical of a shimmed Halbach bore.
    saddle = (X**2 + Z**2 - 2.0 * Y**2) / 3.0
    b0_hz = b0_amp_hz * saddle
    b0 = 2.0 * np.pi * b0_hz
    # B1 strongest on the coil axis (r=0), falling off toward the bore edge.
    r2 = X**2 + Y**2 + Z**2
    b1 = 1.0 - float(b1_falloff) * (r2 / r2.max())
    return b0, b1, b0_hz


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.workflows import run_multislice_imaging

    nx = nz = int(args.pixels)
    ny = int(args.slices)
    fov = float(args.fov)
    rho = _phantom(nx, ny, nz)
    b0, b1, b0_hz = _halbach_fields(
        nx, ny, nz, fov, args.b0_inhomogeneity_hz, args.b1_inhomogeneity
    )

    # Slice gradient (rad/s per metre) for the requested thickness. The shaped
    # pulse used here has an empirical FWHM ~ 51000 / gradient in position units.
    voxel_y = fov / ny
    slice_gradient = 51000.0 / (float(args.slice_thickness_voxels) * voxel_y)

    print(f"phantom {nx}x{ny}x{nz}; FOV {fov*1e3:.0f} mm; "
          f"B0 saddle +/-{args.b0_inhomogeneity_hz:.0f} Hz; "
          f"B1 falloff {args.b1_inhomogeneity*100:.0f}%")
    print(f"slice gradient {slice_gradient:.2e} rad/s/m "
          f"(~{args.slice_thickness_voxels:.1f} voxel slices)")
    print("acquiring slices through the true-3-D engine (B0 curves the slices)...")

    result = run_multislice_imaging(
        rho,
        slice_gradient=slice_gradient,
        slice_axis=1,
        fov=(fov, fov, fov),
        b0_map=b0,
        b1_tx_map=b1,
        b1_rx_map=b1,
        num_substeps=int(args.num_substeps),
        readout_time=2.0e-3,
        phase_time=0.4e-3,
    )
    recon = result.magnitude  # (nx, n_slices, nz)
    recon_norm = recon / (recon.max() or 1.0)
    truth = rho / (rho.max() or 1.0)
    print(f"reconstructed volume {recon.shape}; peak signal {recon.max():.3f}")

    # Slices to display (evenly spaced through the stack).
    n_show = min(4, ny)
    show_idx = np.linspace(0, ny - 1, n_show).round().astype(int)

    fig = plt.figure(figsize=(3.0 * n_show, 9.2))
    gs = fig.add_gridspec(3, n_show, height_ratios=[1.0, 1.0, 1.35])
    extent = [-fov * 1e3 / 2, fov * 1e3 / 2, -fov * 1e3 / 2, fov * 1e3 / 2]

    for col, s in enumerate(show_idx):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(recon_norm[:, s, :].T, origin="lower", extent=extent,
                  cmap="inferno", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"acquired slice {s}")
        if col == 0:
            ax.set_ylabel("phase encode z (mm)")
        axt = fig.add_subplot(gs[1, col])
        axt.imshow(truth[:, s, :].T, origin="lower", extent=extent,
                   cmap="inferno", vmin=0, vmax=1, aspect="equal")
        axt.set_title(f"ground truth slice {s}")
        axt.set_xlabel("readout x (mm)")
        if col == 0:
            axt.set_ylabel("phase encode z (mm)")

    # B0 saddle (central slice) with iso-frequency contours.
    ax_b0 = fig.add_subplot(gs[2, 0])
    mid = ny // 2
    im = ax_b0.imshow(b0_hz[:, mid, :].T, origin="lower", extent=extent,
                      cmap="coolwarm", aspect="equal")
    ax_b0.contour(np.linspace(extent[0], extent[1], nx),
                  np.linspace(extent[2], extent[3], nz),
                  b0_hz[:, mid, :].T, colors="k", linewidths=0.6)
    ax_b0.set_title("Halbach B0 (Hz)")
    fig.colorbar(im, ax=ax_b0, fraction=0.046, pad=0.04)

    # B1 shading (central slice).
    ax_b1 = fig.add_subplot(gs[2, 1])
    im2 = ax_b1.imshow(b1[:, mid, :].T, origin="lower", extent=extent,
                       cmap="viridis", aspect="equal", vmin=b1.min(), vmax=1.0)
    ax_b1.set_title("B1 shading (rel.)")
    fig.colorbar(im2, ax=ax_b1, fraction=0.046, pad=0.04)

    # 3-D voxel rendering of the reconstruction.
    ax3d = fig.add_subplot(gs[2, 2:], projection="3d")
    threshold = 0.18
    mask = recon_norm > threshold
    if mask.any():
        cmap = plt.get_cmap("inferno")
        colors = cmap(recon_norm)
        colors[..., 3] = np.clip(recon_norm, 0.0, 1.0)  # alpha by intensity
        ax3d.voxels(mask, facecolors=colors, edgecolor=(1, 1, 1, 0.05))
    ax3d.set_title("3-D reconstruction")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("slice")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=22, azim=-60)

    fig.suptitle(
        "Multi-slice 3-D imaging in a mild Halbach (B0, B1) field", fontsize=13
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
