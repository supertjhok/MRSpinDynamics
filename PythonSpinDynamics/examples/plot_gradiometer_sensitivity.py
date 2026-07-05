"""Gradiometer pickup: the excited/sensitive region shrinks with order.

By reciprocity a pickup's receive sensitivity to a source at ``r`` equals the
field it would transmit at ``r`` per unit current -- so the region a gradiometer
"excites" is the region it detects. This example uses the field-solver loop
fields (``spin_dynamics.fields.magnetostatics`` via
``spin_dynamics.detection.Gradiometer``) to map that region for three coaxial
pickups and show how winding order localizes it:

* a **magnetometer** (single loop) -- broad reach, sensitivity falls as 1/r^3;
* a **first-order** axial gradiometer ``(+1, -1)`` -- nulls a uniform field,
  1/r^4;
* a **second-order** axial gradiometer ``(+1, -2, +1)`` -- the Clarke 2007 SQUID
  pickup; nulls a uniform field and its first gradient, 1/r^5.

All three couple to a sample at the bottom loop, but the gradiometers reject
distant/ambient sources far more strongly -- the basis of the ~1000x common-mode
rejection in SQUID microtesla MRI.

A summary (orders, uniform-field response, far-field falloff exponents) prints
always; ``--save`` writes a four-panel figure: the three reciprocal-sensitivity
maps in the axial plane plus the on-axis falloff curves.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.detection import Gradiometer  # noqa: E402


def build_pickups(*, radius_m, baseline_m, n_segments):
    return [
        Gradiometer.magnetometer(radius_m=radius_m, n_segments=n_segments),
        Gradiometer.first_order_axial(
            radius_m=radius_m, baseline_m=baseline_m, n_segments=n_segments
        ),
        Gradiometer.second_order_axial(
            radius_m=radius_m, baseline_m=baseline_m, n_segments=n_segments
        ),
    ]


def axial_plane_map(g, x, z):
    """|axial sensitivity| (T/A) on the y=0 plane over grids ``x`` (X) and ``z`` (axis)."""

    xx, zz = np.meshgrid(x, z, indexing="ij")
    pts = np.stack([xx, np.zeros_like(xx), zz], axis=-1)  # y = 0
    return np.abs(g.axial_sensitivity(pts))


def on_axis_falloff(g, z):
    pts = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1)
    return np.abs(g.axial_sensitivity(pts))


def far_slope(g, r1=20.0):
    s1 = on_axis_falloff(g, np.array([r1]))[0]
    s2 = on_axis_falloff(g, np.array([2.0 * r1]))[0]
    return float(np.log(s1 / s2) / np.log(2.0))


def print_summary(pickups) -> None:
    print("Gradiometer pickups (reciprocal spatial sensitivity)")
    for g in pickups:
        print(
            f"  {g.name:26s}: order {g.order}, "
            f"uniform response {g.uniform_field_response():+.3e} m^2, "
            f"far-field 1/r^{far_slope(g):.2f}"
        )


def make_figure(plt, pickups, *, radius_m, baseline_m):
    from matplotlib.colors import LogNorm

    x = np.linspace(-5.0 * radius_m, 5.0 * radius_m, 161)
    z = np.linspace(-1.5 * radius_m, 2.0 * baseline_m + 2.5 * radius_m, 201)
    maps = [axial_plane_map(g, x, z) for g in pickups]
    vmax = max(m.max() for m in maps)
    vmin = vmax * 1e-4

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, g, m in zip(axes[:3], pickups, maps):
        pcm = ax.pcolormesh(
            x * 1e3,
            z * 1e3,
            np.clip(m, vmin, vmax).T,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            shading="auto",
            cmap="magma",
        )
        # draw loop rims (+/- radius) at each loop position
        for coord, w in zip(g.positions_m, g.weights):
            style = "-" if w > 0 else "--"
            ax.plot([-radius_m * 1e3, radius_m * 1e3], [coord * 1e3, coord * 1e3],
                    style, color="cyan", lw=1.5)
        ax.set_title(f"{g.name}\n(order {g.order}, 1/r^{far_slope(g):.1f})", fontsize=9)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z along axis (mm)")
        fig.colorbar(pcm, ax=ax, label="|axial sensitivity| (T/A)")

    ax = axes[3]
    zc = np.logspace(np.log10(2.0 * radius_m), np.log10(50.0 * baseline_m), 200)
    for g in pickups:
        ax.loglog(zc * 1e3, on_axis_falloff(g, zc), label=f"{g.name} (1/r^{far_slope(g):.1f})")
    ax.set_xlabel("on-axis distance z (mm)")
    ax.set_ylabel("|axial sensitivity| (T/A)")
    ax.set_title("On-axis falloff")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Gradiometer reciprocal sensitivity: winding order localizes the excited region")
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-mm", type=float, default=25.0, help="loop radius (mm)")
    parser.add_argument("--baseline-mm", type=float, default=50.0, help="gradiometer baseline (mm)")
    parser.add_argument("--n-segments", type=int, default=72, help="chords per loop")
    parser.add_argument("--save", type=str, default=None,
                        help="path to save the figure; if omitted, only prints a summary")
    args = parser.parse_args()

    radius_m = args.radius_mm * 1e-3
    baseline_m = args.baseline_mm * 1e-3
    pickups = build_pickups(radius_m=radius_m, baseline_m=baseline_m, n_segments=args.n_segments)
    print_summary(pickups)

    if args.save:
        plt = load_matplotlib(required=True, headless=True)
        fig = make_figure(plt, pickups, radius_m=radius_m, baseline_m=baseline_m)
        fig.savefig(args.save, dpi=150)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
