"""NMR-MOUSE stray-field magnet simulated with the 3D scalar-potential solver.

The NMR-MOUSE (MObile Universal Surface Explorer, Bluemich et al., *Magn. Reson.
Imaging* **16**, 479 (1998)) is a single-sided "inside-out" NMR sensor: two
permanent magnets of **anti-parallel** magnetization sit on a soft-iron yoke with
a gap between them, and a surface RF coil in the gap excites a sensitive volume in
the stray field a few millimetres *above* the device. The static field ``B0`` arcs
over the gap so that its dominant component (along the gap, ``z``) is roughly
orthogonal to the coil's ``B1`` (vertical, ``y``) -- the geometry that makes the
sensor work.

This is a showcase for :class:`spin_dynamics.fields.ReducedScalarPotential3D`
because it combines the two ingredients only a magnetostatic solver can treat
together: **permanent magnets** (remanence ``B_r``) and a **soft-magnetic yoke**
(the low-reluctance return path). Analytic dipole superposition can place the
magnets but cannot represent the iron yoke -- which here boosts the useful field
by ~10 %.

Geometry follows Fig. 1 of the paper: two 26 mm-wide x 32 mm-tall magnets, a 13 mm
gap, and a 15 mm iron yoke; the coordinate frame has ``z`` across the gap, ``y``
vertical (out of the surface), ``x`` along the magnet bars.

Panels:
1. ``B0_z`` in the ``x = 0`` cross-section: the field arching out of the gap.
2. Depth profile ``|B0|`` vs height ``y`` on the central axis, with and without the
   yoke -- the penetration-depth gradient that maps RF frequency to depth, and the
   yoke's field boost.
3. Lateral profile ``B0_z`` vs ``z`` at a fixed height: the near-quadratic field
   across the gap (paper Fig. 2), with and without the yoke.

Accuracy note (see docs/reduced_scalar_potential_3d.md): the sensitive spot sits
just above the sharp magnet corners, whose staircasing on a structured grid gives
the absolute field a ~10 % resolution sensitivity; the field *shape*, the yoke
effect, and the ~0.4 T / 17.5 MHz operating point are robust. The high-mu_r yoke
is in the cancellation-error regime, but the reported fields are in the air above
it, which is unaffected; the yoke runs below saturation, so a linear mu_r matches
a saturable-iron model here (available via ``soft_iron()``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.nonlinear_magnetostatics import (  # noqa: E402
    MU0,
    linear_material,
    ndfeb,
)
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D  # noqa: E402

GAMMA_1H_MHZ_PER_T = 42.577  # proton gyromagnetic ratio / 2pi


def build_problem(args, *, with_yoke: bool):
    """Assemble the NMR-MOUSE: two anti-parallel magnets, optional iron yoke."""

    mm = 1e-3
    h = args.h_mm * mm

    def axis(lo, hi):
        n = int(round((hi - lo) / h)) + 1
        return np.linspace(lo, hi, n)

    x = axis(-args.box_x_mm * mm, args.box_x_mm * mm)
    y = axis(-args.box_below_mm * mm, args.box_above_mm * mm)
    z = axis(-args.box_z_mm * mm, args.box_z_mm * mm)
    prob = ReducedScalarPotential3D(x, y, z)

    bar_x = (-0.5 * args.bar_len_mm * mm, 0.5 * args.bar_len_mm * mm)
    half_gap = 0.5 * args.gap_mm * mm
    magnet_outer = half_gap + args.magnet_w_mm * mm
    top, bot = 0.0, -args.magnet_h_mm * mm
    # Left magnet magnetized +y (N up), right magnet -y (S up): anti-parallel.
    prob.add_material(
        prob.box(bar_x, (bot, top), (-magnet_outer, -half_gap)),
        ndfeb(args.remanence_t), remanence_direction=(0.0, 1.0, 0.0),
    )
    prob.add_material(
        prob.box(bar_x, (bot, top), (half_gap, magnet_outer)),
        ndfeb(args.remanence_t), remanence_direction=(0.0, -1.0, 0.0),
    )
    if with_yoke:
        yoke_bot = bot - args.yoke_h_mm * mm
        prob.add_material(
            prob.box(bar_x, (yoke_bot, bot), (-magnet_outer, magnet_outer)),
            linear_material(args.yoke_mu_r),
        )
    return prob, (x, y, z)


def build_data(args) -> dict:
    prob, (x, y, z) = build_problem(args, with_yoke=True)
    sol = prob.solve()
    prob_bare, _ = build_problem(args, with_yoke=False)
    sol_bare = prob_bare.solve()

    mm = 1e-3
    ix0 = int(np.argmin(np.abs(x)))
    # Cross-section map (x = 0): B0_z(y, z). Mask the magnet/yoke interiors so the
    # (~10x weaker) stray field in the air is the visual focus.
    bz_plane = sol.b_z[ix0, :, :].astype(float).copy()
    bz_plane[prob.mu[ix0, :, :] > MU0 * 1.001] = np.nan

    # Depth profile on the central axis (x = z = 0), a few mm above the surface.
    y_probe = np.linspace(0.3 * mm, args.depth_max_mm * mm, 120)
    zeros = np.zeros_like(y_probe)
    b_depth = _bmag(sol, zeros, y_probe, zeros)
    b_depth_bare = _bmag(sol_bare, zeros, y_probe, zeros)

    # Lateral profile across the gap at the sensor height.
    z_probe = np.linspace(-0.5 * args.gap_mm * mm, 0.5 * args.gap_mm * mm, 81)
    yq = np.full_like(z_probe, args.sensor_mm * mm)
    xq = np.zeros_like(z_probe)
    bz_lat = sol.sample(xq, yq, z_probe)[2]
    bz_lat_bare = sol_bare.sample(xq, yq, z_probe)[2]

    # Operating point and gradients at the sensor.
    b_sensor = float(_bmag(sol, [0.0], [args.sensor_mm * mm], [0.0])[0])
    b_sensor_bare = float(_bmag(sol_bare, [0.0], [args.sensor_mm * mm], [0.0])[0])
    depth_grad = _local_slope(y_probe, b_depth, args.sensor_mm * mm)  # T/m
    lat_fit = np.polyfit(z_probe, bz_lat, 2)

    return {
        "y_m": y, "z_m": z, "bz_plane": bz_plane,
        "y_depth_mm": y_probe * 1e3, "b_depth": b_depth, "b_depth_bare": b_depth_bare,
        "z_lat_mm": z_probe * 1e3, "bz_lat": bz_lat, "bz_lat_bare": bz_lat_bare,
        "b_sensor": b_sensor, "b_sensor_bare": b_sensor_bare,
        "depth_grad": depth_grad, "lat_curv": lat_fit[0],
        "iters": sol.iterations, "n": x.size * y.size * z.size,
        "geom": _geom(args),
    }


def _bmag(sol, xq, yq, zq):
    bx, by, bz = sol.sample(np.asarray(xq), np.asarray(yq), np.asarray(zq))
    return np.sqrt(bx**2 + by**2 + bz**2)


def _local_slope(xs, ys, x0):
    i = int(np.argmin(np.abs(xs - x0)))
    lo, hi = max(0, i - 5), min(len(xs) - 1, i + 5)
    return float(np.polyfit(xs[lo:hi + 1], ys[lo:hi + 1], 1)[0])


def _geom(args):
    half_gap = 0.5 * args.gap_mm
    return {
        "half_gap": half_gap,
        "magnet_outer": half_gap + args.magnet_w_mm,
        "magnet_h": args.magnet_h_mm,
        "yoke_h": args.yoke_h_mm,
    }


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Geometry (paper Fig. 1 defaults, mm).
    parser.add_argument("--gap-mm", type=float, default=13.0)
    parser.add_argument("--magnet-w-mm", type=float, default=26.0)
    parser.add_argument("--magnet-h-mm", type=float, default=32.0)
    parser.add_argument("--yoke-h-mm", type=float, default=15.0)
    parser.add_argument("--bar-len-mm", type=float, default=40.0)
    parser.add_argument("--remanence-t", type=float, default=1.2)
    parser.add_argument("--yoke-mu-r", type=float, default=1000.0)
    # Sampling and domain.
    parser.add_argument("--sensor-mm", type=float, default=2.5, help="sensor height above surface")
    parser.add_argument("--depth-max-mm", type=float, default=15.0)
    parser.add_argument("--h-mm", type=float, default=1.5, help="grid spacing")
    parser.add_argument("--box-z-mm", type=float, default=90.0)
    parser.add_argument("--box-x-mm", type=float, default=55.0)
    parser.add_argument("--box-above-mm", type=float, default=55.0)
    parser.add_argument("--box-below-mm", type=float, default=75.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)
    g = data["geom"]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)

    # Panel 1: cross-section B0_z map.
    extent = [data["z_m"][0] * 1e3, data["z_m"][-1] * 1e3,
              data["y_m"][0] * 1e3, data["y_m"][-1] * 1e3]
    # Scale to the stray field above the surface, not the (~10x larger) internal
    # magnet field, so the arching sensitive-region field is visible; magnet
    # interiors saturate to solid colour (they are outlined and labelled N/S).
    above = data["y_m"] > 0
    vmax = max(100.0, float(np.nanpercentile(np.abs(data["bz_plane"][above, :]), 98)) * 1e3)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.85")  # magnet/yoke interiors (masked)
    # bz_plane is [iy, iz]; rows map to the vertical (y) axis of extent, so no
    # transpose (transposing swaps y<->z and mis-orients the field).
    im = axes[0].imshow(data["bz_plane"] * 1e3, origin="lower", extent=extent,
                        cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal")
    _draw_geometry(axes[0], g, plt)
    axes[0].set_xlim(-g["magnet_outer"] - 15, g["magnet_outer"] + 15)
    axes[0].set_ylim(-(g["magnet_h"] + g["yoke_h"]) - 8, 30)
    axes[0].set_xlabel("z (mm)  [across gap]")
    axes[0].set_ylabel("y (mm)  [depth]")
    axes[0].set_title("B0_z cross-section (x=0)")
    fig.colorbar(im, ax=axes[0], label="B0_z (mT)", fraction=0.046)

    # Panel 2: depth profile and penetration.
    f_sensor = GAMMA_1H_MHZ_PER_T * data["b_sensor"]
    axes[1].plot(data["b_depth"] * 1e3, data["y_depth_mm"], color="tab:blue", label="with yoke")
    axes[1].plot(data["b_depth_bare"] * 1e3, data["y_depth_mm"], "--", color="tab:gray",
                 label="magnets only")
    axes[1].axhline(args.sensor_mm, color="k", ls=":", lw=0.8)
    axes[1].plot([data["b_sensor"] * 1e3], [args.sensor_mm], "ko", ms=5)
    axes[1].set_xlabel("|B0| (mT)")
    axes[1].set_ylabel("height above surface y (mm)")
    axes[1].set_title(f"Depth profile  (grad ~{abs(data['depth_grad']):.0f} T/m)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    # Panel 3: lateral profile across the gap.
    axes[2].plot(data["z_lat_mm"], data["bz_lat"] * 1e3, color="tab:blue", label="with yoke")
    axes[2].plot(data["z_lat_mm"], data["bz_lat_bare"] * 1e3, "--", color="tab:gray",
                 label="magnets only")
    axes[2].axvspan(-g["half_gap"], g["half_gap"], color="orange", alpha=0.08, label="gap")
    axes[2].set_xlabel("z (mm)  [across gap]")
    axes[2].set_ylabel("B0_z (mT)")
    axes[2].set_title(f"Lateral field at y={args.sensor_mm:g} mm (~quadratic)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.2)

    print("NMR-MOUSE stray-field magnet (3D reduced scalar potential)")
    print(f"  grid                        : {data['n']:,} nodes @ {args.h_mm:g} mm")
    print(f"  B0 at sensor (y={args.sensor_mm:g} mm)   : {data['b_sensor'] * 1e3:.1f} mT"
          f"  ->  {f_sensor:.2f} MHz (1H)   [paper: 0.41 T, 17.5 MHz]")
    boost = 100.0 * (data["b_sensor"] / data["b_sensor_bare"] - 1.0)
    print(f"  iron-yoke field boost       : {data['b_sensor_bare'] * 1e3:.1f} -> "
          f"{data['b_sensor'] * 1e3:.1f} mT  (+{boost:.0f} %)")
    print(f"  depth (penetration) gradient: {abs(data['depth_grad']):.0f} T/m"
          f"   [paper: 10-14 T/m]")
    print(f"  lateral profile curvature   : {data['lat_curv']:.0f} T/m^2 (quadratic across gap)")
    print(f"  nonlinear/linear iterations : {data['iters']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _draw_geometry(ax, g, plt):
    """Outline the two magnets, the gap, the yoke, and the surface coil."""

    ho, hg, mh, yh = g["magnet_outer"], g["half_gap"], g["magnet_h"], g["yoke_h"]
    style = dict(fill=False, edgecolor="k", lw=0.8)
    ax.add_patch(plt.Rectangle((-ho, -mh), ho - hg, mh, **style))   # left magnet
    ax.add_patch(plt.Rectangle((hg, -mh), ho - hg, mh, **style))    # right magnet
    ax.add_patch(plt.Rectangle((-ho, -mh - yh), 2 * ho, yh, **style))  # yoke
    ax.add_patch(plt.Rectangle((-hg, -3), 2 * hg, 3, facecolor="orange",
                               alpha=0.35, edgecolor="none"))       # surface coil
    for zc, s in ((-(ho + hg) / 2, "N"), ((ho + hg) / 2, "S")):
        ax.text(zc, -mh * 0.25, s, ha="center", va="center", fontsize=9, weight="bold")
    for zc, s in ((-(ho + hg) / 2, "S"), ((ho + hg) / 2, "N")):
        ax.text(zc, -mh * 0.8, s, ha="center", va="center", fontsize=9, weight="bold")


if __name__ == "__main__":
    main()
