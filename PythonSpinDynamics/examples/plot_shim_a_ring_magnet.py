"""Shim-a-ring permanent magnet: a diametric NdFeB ring inside a soft-iron ring.

A uniformly (diametrically) magnetized NdFeB *tube* produces essentially **zero**
field in its bore -- the outer and inner bound surface currents (both a
``sin(theta)`` sheet) generate equal and opposite uniform interior fields that
cancel (a solid magnetized cylinder minus the missing inner cylinder). The bore
field is created by a **concentric soft-iron ring**: the high-permeability iron
provides a low-reluctance return path that breaks the cancellation and sets up a
transverse field across the bore. This makes the iron an *essential* nonlinear
element, and its **saturation** limits the achievable field -- both are captured
by the structured-grid nonlinear magnetostatic solver
(:class:`spin_dynamics.fields.PlanarMagnetostatics`), which analytic dipole
superposition cannot represent.

Panels:
1. Transverse ``B_x`` map of the assembled ring (bore field + iron return path).
2. Bore field along ``x``: magnet-only (~0) versus magnet + iron ring.
3. Iron-thickness sweep: bore field and peak ``|B|`` in the iron -- the field
   rises with iron until the ring saturates, the nonlinear performance knee.

The bore homogeneity of a plain circular iron ring is modest (reported in ppm);
shaped iron pole pieces would improve it, a natural extension.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.nonlinear_magnetostatics import (  # noqa: E402
    PlanarMagnetostatics,
    ndfeb,
    soft_iron,
)


def build_problem(args, iron_outer_m: float | None, n: int | None = None):
    """Assemble a diametric NdFeB ring, optionally inside a soft-iron ring."""

    half = args.box_mm * 1e-3
    n = int(args.n if n is None else n)
    prob = PlanarMagnetostatics(np.linspace(-half, half, n), np.linspace(-half, half, n))
    prob.add_material(
        prob.annulus((0.0, 0.0), args.bore_mm * 1e-3, args.magnet_outer_mm * 1e-3),
        ndfeb(args.remanence_t, mu_rec=1.05),
        remanence_angle_deg=0.0,
    )
    if iron_outer_m is not None:
        prob.add_material(
            prob.annulus((0.0, 0.0), args.iron_inner_mm * 1e-3, iron_outer_m),
            soft_iron(args.iron_curve),
        )
    return prob


def build_data(args) -> dict:
    """Build the field map, bore profiles, and iron-thickness sweep."""

    iron_outer = args.iron_outer_mm * 1e-3

    prob_iron = build_problem(args, iron_outer)
    sol_iron = prob_iron.solve(max_iter=args.max_iter, relax=args.relax, n_load_steps=args.load_steps)

    prob_bare = build_problem(args, None)
    sol_bare = prob_bare.solve(max_iter=args.max_iter, relax=args.relax)

    xs = np.linspace(-args.magnet_outer_mm * 1e-3, args.magnet_outer_mm * 1e-3, 121)
    zeros = np.zeros_like(xs)
    bx_iron, _ = sol_iron.sample(xs, zeros)
    bx_bare, _ = sol_bare.sample(xs, zeros)

    iron_mask = prob_iron.annulus(
        (0.0, 0.0), args.iron_inner_mm * 1e-3, iron_outer
    )
    peak_iron = float(np.max(sol_iron.b_magnitude[iron_mask]))
    bore_bx = float(sol_iron.sample([0.0], [0.0])[0][0])
    ppm = sol_iron.homogeneity_ppm(radius_m=args.dsv_mm * 1e-3)

    # Iron-thickness sweep: field vs saturation knee.
    thicknesses = np.linspace(args.sweep_min_mm, args.sweep_max_mm, int(args.sweep_points))
    sweep_bore = np.empty_like(thicknesses)
    sweep_mean = np.empty_like(thicknesses)
    for idx, t_mm in enumerate(thicknesses):
        outer = (args.iron_inner_mm + t_mm) * 1e-3
        prob = build_problem(args, outer, n=args.sweep_n)
        sol = prob.solve(max_iter=args.max_iter, relax=args.relax, n_load_steps=args.load_steps)
        sweep_bore[idx] = float(sol.sample([0.0], [0.0])[0][0])
        mask = prob.annulus((0.0, 0.0), args.iron_inner_mm * 1e-3, outer)
        # Mean |B| over the iron is a robust saturation proxy; the pointwise peak
        # is contaminated by staircase-corner spikes on the structured grid.
        sweep_mean[idx] = float(np.mean(sol.b_magnitude[mask]))

    win = np.abs(prob_iron.x) <= 1.1 * iron_outer
    sub = np.ix_(win, win)
    return {
        "x_m": prob_iron.x[win],
        "bx_map": sol_iron.b_x[sub],
        "xs_mm": xs * 1e3,
        "bx_iron_mt": bx_iron * 1e3,
        "bx_bare_mt": bx_bare * 1e3,
        "bore_bx_mt": bore_bx * 1e3,
        "peak_iron_t": peak_iron,
        "ppm": ppm,
        "iters": sol_iron.iterations,
        "sweep_thickness_mm": thicknesses,
        "sweep_bore_mt": sweep_bore * 1e3,
        "sweep_mean_t": sweep_mean,
        "radii_mm": (
            args.bore_mm,
            args.magnet_outer_mm,
            args.iron_inner_mm,
            args.iron_outer_mm,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bore-mm", type=float, default=20.0)
    parser.add_argument("--magnet-outer-mm", type=float, default=45.0)
    parser.add_argument("--iron-inner-mm", type=float, default=50.0)
    parser.add_argument("--iron-outer-mm", type=float, default=90.0)
    parser.add_argument("--remanence-t", type=float, default=1.3)
    parser.add_argument("--iron-curve", type=str, default="soft_iron")
    parser.add_argument("--box-mm", type=float, default=400.0)
    parser.add_argument("--n", type=int, default=181)
    parser.add_argument("--sweep-n", type=int, default=121)
    parser.add_argument("--dsv-mm", type=float, default=6.0)
    parser.add_argument("--relax", type=float, default=0.7)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--load-steps", type=int, default=1)
    parser.add_argument("--sweep-min-mm", type=float, default=5.0)
    parser.add_argument("--sweep-max-mm", type=float, default=45.0)
    parser.add_argument("--sweep-points", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    extent = [data["x_m"][0] * 1e3, data["x_m"][-1] * 1e3] * 2
    im = axes[0].imshow(
        data["bx_map"].T * 1e3, origin="lower", extent=extent, cmap="RdBu_r"
    )
    for radius in data["radii_mm"]:
        axes[0].add_patch(plt.Circle((0, 0), radius, fill=False, color="k", lw=0.6))
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Transverse B_x (mT)")
    fig.colorbar(im, ax=axes[0])

    axes[1].plot(data["xs_mm"], data["bx_bare_mt"], label="magnet ring only")
    axes[1].plot(data["xs_mm"], data["bx_iron_mt"], label="magnet + iron ring")
    axes[1].axvspan(-args.bore_mm, args.bore_mm, color="gray", alpha=0.1, label="bore")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("B_x (mT)")
    axes[1].set_title("Bore field along x")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    ax2 = axes[2]
    ax2b = ax2.twinx()
    ax2.plot(data["sweep_thickness_mm"], data["sweep_bore_mt"], "o-", color="tab:blue",
             label="bore B_x")
    ax2b.plot(data["sweep_thickness_mm"], data["sweep_mean_t"], "s--", color="tab:red",
              label="mean |B| in iron")
    ax2b.axhline(1.5, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax2.set_xlabel("Iron ring thickness (mm)")
    ax2.set_ylabel("Bore B_x (mT)", color="tab:blue")
    ax2b.set_ylabel("Mean |B| in iron (T)", color="tab:red")
    ax2.set_title("Field builds as the iron de-saturates")
    ax2.grid(True, alpha=0.2)

    print("Shim-a-ring: diametric NdFeB ring + soft-iron return ring")
    print(f"  bore B_x (magnet + iron)   : {data['bore_bx_mt']:.1f} mT")
    print(f"  bore B_x (magnet only)     : {float(data['bx_bare_mt'][len(data['bx_bare_mt'])//2]):.2f} mT (tube cancellation)")
    print(f"  homogeneity over DSV r={args.dsv_mm:g}mm : {data['ppm']:.0f} ppm")
    print(f"  peak |B| in iron ring      : {data['peak_iron_t']:.2f} T")
    print(f"  nonlinear iterations       : {data['iters']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
