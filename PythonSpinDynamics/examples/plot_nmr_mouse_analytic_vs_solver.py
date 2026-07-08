"""NMR-MOUSE field: analytic model vs 3D magnetostatic solver -- when to use which.

Two ways to get the NMR-MOUSE static field, with different assumptions:

* **Analytic** (:func:`spin_dynamics.fields.magnetostatics.sample_magnet_field`
  with :func:`nmr_mouse_magnets`): closed-form magnetic-charge-sheet field of the
  bars plus a soft-iron yoke by the method of images (a perfect ``mu -> infinity``
  plane). It is essentially instant, but assumes the bars are **infinitely long**
  (2-D, translationally invariant) and the yoke ideal.

* **3-D solver** (:class:`spin_dynamics.fields.ReducedScalarPotential3D`): the
  reduced-scalar-potential solve of the actual **finite** magnets and a **finite**
  iron yoke. It costs seconds but models the real geometry -- finite bar length,
  finite/saturable yoke permeability, arbitrary shapes.

This script quantifies the trade-off on-axis. It finds that for this device the
two differ mostly because of **finite bar length**: the analytic (infinite-bar)
model overestimates the field of a finite MOUSE, and the solver *converges to the
analytic value as the bars are lengthened* (validating both). The yoke
permeability barely matters here (a 15 mm ``mu_r = 1000`` slab is already
near-ideal), so the length is the deciding physics.

Panels:
1. On-axis ``|B0|`` vs depth: analytic (infinite bars) versus the solver at the
   real bar length -- the finite-length shortfall the analytic cannot capture.
2. Solver ``|B0|`` at a fixed depth versus bar length, approaching the analytic
   (dashed) as bars lengthen; the legend reports the solve times.

Guidance printed at the end: use the analytic model for speed and for long bars /
near-ideal yokes; use the solver when finite geometry, a finite/saturable yoke, or
an unusual shape matters.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.magnetostatics import (  # noqa: E402
    GAMMA_PROTON,
    bar_array_b0,
    nmr_mouse_magnets,
)
from spin_dynamics.fields.nonlinear_magnetostatics import linear_material, ndfeb  # noqa: E402
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D  # noqa: E402

MM = 1e-3


def analytic_profile(args, depths):
    """On-axis |B0|(depth) from the infinite-bar analytic model, and its time."""

    bars, yoke = nmr_mouse_magnets(
        magnet_width=args.magnet_w_mm * MM, magnet_height=args.magnet_h_mm * MM,
        gap=args.gap_mm * MM, remanence=args.remanence_t,
    )
    t0 = time.perf_counter()
    b0 = np.hypot(*bar_array_b0(np.zeros_like(depths), depths, bars, yoke_y=yoke))
    return b0, time.perf_counter() - t0


def solve_profile(args, depths, bar_len_mm, h_mm):
    """On-axis |B0|(depth) from the 3D solver at a given bar length, and its time.

    Geometry matches the analytic model: bars in y in [0, H] (bottom on the yoke),
    sample above the bars; a finite mu_r yoke slab replaces the image plane.
    """

    h = h_mm * MM
    half = 0.5 * bar_len_mm * MM
    hh, hg = args.magnet_h_mm * MM, 0.5 * args.gap_mm * MM
    outer = hg + args.magnet_w_mm * MM

    def axis(lo, hi):
        return np.linspace(lo, hi, int(round((hi - lo) / h)) + 1)

    x = axis(-(half + 20 * MM), half + 20 * MM)
    y = axis(-args.yoke_h_mm * MM - 25 * MM, hh + args.sample_max_mm * MM + 20 * MM)
    z = axis(-args.box_z_mm * MM, args.box_z_mm * MM)
    prob = ReducedScalarPotential3D(x, y, z)
    bar_x = (-half, half)
    prob.add_material(prob.box(bar_x, (0.0, hh), (-outer, -hg)),
                      ndfeb(args.remanence_t), remanence_direction=(0, 1, 0))
    prob.add_material(prob.box(bar_x, (0.0, hh), (hg, outer)),
                      ndfeb(args.remanence_t), remanence_direction=(0, -1, 0))
    prob.add_material(prob.box(bar_x, (-args.yoke_h_mm * MM, 0.0), (-outer, outer)),
                      linear_material(args.yoke_mu_r))
    t0 = time.perf_counter()
    sol = prob.solve()
    dt = time.perf_counter() - t0
    bx, by, bz = sol.sample(np.zeros_like(depths), depths, np.zeros_like(depths))
    return np.sqrt(bx**2 + by**2 + bz**2), dt, x.size * y.size * z.size


def build_data(args) -> dict:
    hh = args.magnet_h_mm * MM
    depths = np.linspace(hh + 1 * MM, hh + args.sample_max_mm * MM, 40)
    sensor = hh + args.sensor_mm * MM

    b0_an, t_an = analytic_profile(args, depths)
    b0_solve, t_solve, n = solve_profile(args, depths, args.bar_len_mm, args.h_mm)

    # Bar-length convergence at the sensor depth.
    lengths = np.array(args.bar_lengths_mm, dtype=float)
    sensor_b0, sensor_t = [], []
    for length in lengths:
        b0_l, dt_l, _ = solve_profile(args, np.array([sensor]), length, args.h_mm)
        sensor_b0.append(float(b0_l[0]))
        sensor_t.append(dt_l)
    b0_an_sensor = float(np.interp(sensor, depths, b0_an))

    i_sensor = int(np.argmin(np.abs(depths - sensor)))
    return {
        "depth_mm": (depths - hh) * 1e3, "b0_an": b0_an, "b0_solve": b0_solve,
        "t_an": t_an, "t_solve": t_solve, "n": n,
        "lengths_mm": lengths, "sensor_b0": np.array(sensor_b0), "sensor_t": np.array(sensor_t),
        "b0_an_sensor": b0_an_sensor,
        "sensor_ratio": float(b0_solve[i_sensor] / b0_an[i_sensor]),
        "sensor_mm": args.sensor_mm, "bar_len_mm": args.bar_len_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gap-mm", type=float, default=13.0)
    parser.add_argument("--magnet-w-mm", type=float, default=26.0)
    parser.add_argument("--magnet-h-mm", type=float, default=32.0)
    parser.add_argument("--yoke-h-mm", type=float, default=15.0)
    parser.add_argument("--yoke-mu-r", type=float, default=1000.0)
    parser.add_argument("--remanence-t", type=float, default=1.2)
    parser.add_argument("--bar-len-mm", type=float, default=40.0)
    parser.add_argument("--bar-lengths-mm", type=float, nargs="+",
                        default=[40.0, 80.0, 160.0, 320.0])
    parser.add_argument("--h-mm", type=float, default=2.5)
    parser.add_argument("--box-z-mm", type=float, default=90.0)
    parser.add_argument("--sensor-mm", type=float, default=3.0, help="depth above bars")
    parser.add_argument("--sample-max-mm", type=float, default=28.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    axes[0].plot(data["depth_mm"], data["b0_an"] * 1e3, color="tab:blue",
                 label=f"analytic (inf. bars)  {data['t_an'] * 1e3:.2f} ms")
    axes[0].plot(data["depth_mm"], data["b0_solve"] * 1e3, color="tab:red",
                 label=f"solver ({data['bar_len_mm']:g} mm bars)  {data['t_solve']:.1f} s")
    axes[0].set_xlabel("depth above bars (mm)")
    axes[0].set_ylabel("|B0| (mT)")
    axes[0].set_title("On-axis field: analytic vs 3D solver")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].axhline(data["b0_an_sensor"] * 1e3, color="tab:blue", ls="--",
                    label="analytic (infinite bars)")
    for length, b0, dt in zip(data["lengths_mm"], data["sensor_b0"], data["sensor_t"]):
        axes[1].plot(length, b0 * 1e3, "o", color="tab:red")
        axes[1].annotate(f"{dt:.1f}s", (length, b0 * 1e3), textcoords="offset points",
                         xytext=(4, -10), fontsize=7, color="0.4")
    axes[1].plot(data["lengths_mm"], data["sensor_b0"] * 1e3, "-", color="tab:red",
                 label="solver (finite bars)")
    axes[1].set_xlabel("bar length (mm)")
    axes[1].set_ylabel(f"|B0| at {data['sensor_mm']:g} mm (mT)")
    axes[1].set_title("Solver converges to analytic as bars lengthen")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    f_an = GAMMA_PROTON * data["b0_an"][int(np.argmin(np.abs(data["depth_mm"] - data["sensor_mm"])))] / (2 * np.pi)
    print("NMR-MOUSE field: analytic model vs 3D solver")
    print(f"  analytic: {data['t_an'] * 1e3:.2f} ms   solver: {data['t_solve']:.1f} s"
          f"  ({data['n']:,} nodes @ {args.h_mm:g} mm)")
    print(f"  at the {args.sensor_mm:g} mm sensor, {args.bar_len_mm:g} mm bars:")
    print(f"    analytic |B0| = {data['b0_an_sensor'] * 1e3:.0f} mT ({f_an / 1e6:.1f} MHz, infinite bars)")
    print(f"    solver   |B0| = {data['sensor_b0'][0] * 1e3:.0f} mT"
          f"  ({100 * data['sensor_ratio']:.0f}% of analytic -- finite-length shortfall)")
    print(f"    solver at {data['lengths_mm'][-1]:g} mm bars = "
          f"{100 * data['sensor_b0'][-1] / data['b0_an_sensor']:.0f}% (converging to analytic)")
    print("  Guidance: analytic model = instant, exact for long bars + ideal yoke -> use for")
    print("  design sweeps; 3D solver = seconds, models finite length / finite-saturable yoke /")
    print("  arbitrary shapes -> use for realistic finite devices.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
