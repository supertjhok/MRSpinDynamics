"""High-permeability accuracy and AMG scaling for the 3D scalar-potential solver.

Two linked findings about :class:`spin_dynamics.fields.ReducedScalarPotential3D`,
demonstrated on a permeable sphere in a uniform applied field (analytic interior
``B = mu0 * 3 mu_r/(mu_r+2) * H0``):

1. **The reduced-potential cancellation error at high ``mu_r`` is cured by grid
   refinement, not by the Simkin-Trowbridge total/reduced split.** On this
   centered nodal finite-volume grid the total and reduced formulations are an
   exact linear variable shift apart, so they give the *identical* discrete
   solution (verified to machine precision) -- the split removes none of the
   error. What remains is discretization error, which falls as the grid refines.

2. **AMG (pyamg) is what makes that refinement affordable.** Sparse-LU fill-in
   caps 3D grids at ~50-64^3; AMG-preconditioned CG has a grid-independent
   iteration count and scales to >10^6 unknowns, so the fine grids that reduce
   the high-``mu_r`` error are reachable.

Panels:
1. Interior-``B`` error versus resolution ``R/h`` (log-log) -- the error decays
   with refinement.
2. Solve time versus number of unknowns for each linear solver -- AMG's near-linear
   scaling versus the sparse-LU wall.

Requires pyamg for the finer grids (``pip install pyamg``); without it the script
runs the sparse-LU points only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.nonlinear_magnetostatics import (  # noqa: E402
    MU0,
    _HAVE_PYAMG,
    linear_material,
)
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D  # noqa: E402


def analytic_b_in(mu_r: float, h0: float) -> float:
    return MU0 * mu_r * 3.0 * h0 / (mu_r + 2.0)


def solve_at(n: int, mu_r: float, h0: float, half_m: float, radius_m: float, solver: str):
    g = np.linspace(-half_m, half_m, n)
    prob = ReducedScalarPotential3D(g, g, g)
    prob.add_material(prob.sphere((0.0, 0.0, 0.0), radius_m), linear_material(mu_r))
    prob.add_uniform_source_field((0.0, 0.0, h0))
    t0 = time.perf_counter()
    sol = prob.solve(linear_solver=solver, cg_tol=1e-10)
    dt = time.perf_counter() - t0
    b_in = float(sol.mean_b_in(prob.sphere((0.0, 0.0, 0.0), 0.4 * radius_m))[2])
    r_over_h = radius_m / (2.0 * half_m / (n - 1))
    return {"n": n, "N": n**3, "r_over_h": r_over_h, "b_in": b_in, "time": dt}


def build_data(args) -> dict:
    exact = analytic_b_in(args.mu_r, args.h0)
    # Refinement study: sparse LU where it is affordable, AMG for the finer grids.
    rows = []
    for n in args.grids:
        solver = "amg" if (_HAVE_PYAMG and n**3 > 45**3) else "splu"
        rec = solve_at(n, args.mu_r, args.h0, args.box_mm * 1e-3, args.radius_mm * 1e-3, solver)
        rec["solver"] = solver
        rec["err"] = abs(rec["b_in"] / exact - 1.0)
        rows.append(rec)
        print(
            f"  n={rec['n']:3d}  N={rec['N']:>8d}  R/h={rec['r_over_h']:4.1f}  "
            f"{solver:4s}  err={100 * rec['err']:5.1f}%  {rec['time']:5.1f}s"
        )
    # Solver-scaling: time vs N for each available solver on a few grids.
    scaling = {}
    for solver in (["splu", "amg", "cg"] if _HAVE_PYAMG else ["splu", "cg"]):
        pts = []
        for n in args.scaling_grids:
            if solver == "splu" and n**3 > args.splu_cap**3:
                continue  # keep sparse LU off grids that would exhaust memory/time
            rec = solve_at(n, args.mu_r, args.h0, args.box_mm * 1e-3, args.radius_mm * 1e-3, solver)
            pts.append((rec["N"], rec["time"]))
            print(f"  scaling {solver:4s} n={n:3d} N={rec['N']:>8d}  {rec['time']:6.2f}s")
        scaling[solver] = np.array(pts)
    return {"exact": exact, "rows": rows, "scaling": scaling}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mu-r", type=float, default=1000.0)
    parser.add_argument("--h0", type=float, default=1000.0)
    parser.add_argument("--radius-mm", type=float, default=50.0)
    parser.add_argument("--box-mm", type=float, default=150.0)
    parser.add_argument("--grids", type=int, nargs="+", default=[41, 61, 81, 101])
    parser.add_argument("--scaling-grids", type=int, nargs="+", default=[31, 41, 51, 61])
    parser.add_argument("--splu-cap", type=int, default=45, help="max n per axis for sparse LU")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    exact_mt = analytic_b_in(args.mu_r, args.h0) * 1e3
    print(f"High-mu_r={args.mu_r:g} sphere; analytic interior B = {exact_mt:.2f} mT")
    if not _HAVE_PYAMG:
        print("  (pyamg not installed -- sparse-LU points only; finer grids skipped)")
    data = build_data(args)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    r_over_h = np.array([r["r_over_h"] for r in data["rows"]])
    errs = np.array([100 * r["err"] for r in data["rows"]])
    axes[0].loglog(r_over_h, errs, "o-", color="tab:red")
    # O(h) reference through the first point.
    ref = errs[0] * (r_over_h[0] / r_over_h)
    axes[0].loglog(r_over_h, ref, "k--", lw=0.8, label="O(h)")
    axes[0].set_xlabel("resolution  R / h")
    axes[0].set_ylabel("interior B error (%)")
    axes[0].set_title(f"High-mu_r error falls with refinement (mu_r={args.mu_r:g})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.2)

    colors = {"splu": "tab:blue", "amg": "tab:red", "cg": "tab:green"}
    for solver, pts in data["scaling"].items():
        if pts.size:
            axes[1].loglog(pts[:, 0], pts[:, 1], "o-", color=colors[solver], label=solver)
    axes[1].set_xlabel("unknowns N")
    axes[1].set_ylabel("solve time (s)")
    axes[1].set_title("Linear-solver scaling")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.2)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
