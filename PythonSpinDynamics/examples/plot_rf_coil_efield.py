"""RF coil B1, induced E-field, and eddy-current power deposition.

The Biot-Savart solver gives an RF coil's B1 but not the E-field that accompanies
it -- and it is the E-field that drives eddy currents and resistive power loss in
a conductive sample. This example maps both a solenoid and a planar-spiral
(pancake) coil: the transverse B1, the induced |E| = |dA/dt|, and the eddy Joule
density sigma|E|^2 in a cylindrical conductive sample, with the total deposited
power.

First-order / Born model (see docs/field_solvers_quasistatic.md): the induced E
is the primary-field response only; it ignores the eddy currents' reaction field,
so it is valid for low conductivity / thin samples / skin depth >> sample and
OVER-estimates deposition at high conductivity (no skin-effect shielding).
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import coils
from spin_dynamics.fields.magnetostatics import biot_savart
from spin_dynamics.fields.quasistatic import eddy_currents, induced_efield


def _slice_maps(segments, xs, zs, dIdt):
    """|B|, |E| on the y=0 (x-z) plane."""

    X, Z = np.meshgrid(xs, zs, indexing="ij")
    pts = np.stack([X, np.zeros_like(X), Z], axis=-1)
    b = biot_savart(pts, segments, 1.0)
    e = induced_efield(pts, segments, dIdt)
    return np.linalg.norm(b, axis=-1), np.linalg.norm(e, axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extent-mm", type=float, default=40.0)
    parser.add_argument("--grid", type=int, default=61, help="2-D slice resolution.")
    parser.add_argument("--sample-grid", type=int, default=25, help="3-D sample resolution.")
    parser.add_argument("--frequency-mhz", type=float, default=10.0)
    parser.add_argument("--current-a", type=float, default=1.0)
    parser.add_argument("--conductivity", type=float, default=0.6, help="Sample conductivity (S/m, ~saline).")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    ext = args.extent_mm * 1e-3
    dIdt = 2.0 * np.pi * args.frequency_mhz * 1e6 * args.current_a  # peak sinusoidal slew

    solenoid = coils.solenoid(radius=0.02, length=0.05, turns=8, axis="z", n_segments=48)
    spiral = coils.planar_spiral(r_inner=0.004, r_outer=0.02, turns=6, center=(0, 0, -0.02), axis="z")

    xs = np.linspace(-ext, ext, args.grid)
    zs = np.linspace(-ext, ext, args.grid)

    # 3-D conductive sample: a cylinder on the coil axis.
    ng = args.sample_grid
    axg = np.linspace(-ext, ext, ng)
    dg = axg[1] - axg[0]
    grid = np.stack(np.meshgrid(axg, axg, axg, indexing="ij"), axis=-1)
    rho_xy = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2)
    sample_mask = (rho_xy <= 0.015) & (np.abs(grid[..., 2]) <= 0.015)

    print("RF coil E-field and eddy-current power deposition")
    print(f"  drive: {args.frequency_mhz:.1f} MHz, {args.current_a:.1f} A (dI/dt = {dIdt:.2e} A/s)")
    print(f"  sample: sigma = {args.conductivity:.2f} S/m cylinder (r=15 mm, h=30 mm)")
    results = {}
    for name, seg in (("solenoid", solenoid), ("planar spiral", spiral)):
        bmag, emag = _slice_maps(seg, xs, zs, dIdt)
        res = eddy_currents(
            grid, seg, dIdt, conductivity=args.conductivity, mask=sample_mask,
            spacing=(dg, dg, dg), charge_correction=True,
        )
        results[name] = (bmag, emag, res)
        print(f"  {name:14s}: deposited power = {res.power * 1e3:.3f} mW")

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        extent = [-args.extent_mm, args.extent_mm, -args.extent_mm, args.extent_mm]
        for row, name in enumerate(("solenoid", "planar spiral")):
            bmag, emag, res = results[name]
            joule = args.conductivity * np.sum(np.abs(res.e_field) ** 2, axis=-1)
            mid = res.e_field.shape[1] // 2
            panels = [
                (bmag.T * 1e6, f"{name}: |B1| (uT/A)", "viridis"),
                (emag.T, f"{name}: |E| (V/m)", "magma"),
                (joule[:, mid, :].T, f"{name}: eddy Joule sigma|E|^2 (W/m^3)", "inferno"),
            ]
            for col, (data, title, cmap) in enumerate(panels):
                # Clip the colour scale to the 99th percentile: Biot-Savart is
                # singular ON the filaments, and those on-wire spikes would
                # otherwise dominate the scale and hide the field structure.
                vmax = float(np.percentile(data[data > 0], 99)) if np.any(data > 0) else None
                im = axes[row, col].imshow(
                    data, origin="lower", extent=extent, aspect="equal", cmap=cmap, vmax=vmax
                )
                axes[row, col].set_title(title, fontsize=9)
                axes[row, col].set_xlabel("x (mm)")
                axes[row, col].set_ylabel("z (mm)")
                fig.colorbar(im, ax=axes[row, col], fraction=0.046)
        fig.suptitle("First-order (Born) induced E-field -- no skin-effect shielding", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    main()
