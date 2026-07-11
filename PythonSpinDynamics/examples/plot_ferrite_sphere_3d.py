"""3D flux concentration by a soft-magnetic sphere (reduced scalar potential).

A soft-magnetic sphere placed in a uniform applied field ``H0`` concentrates the
flux inside it and radiates a dipole-like distortion outside -- a genuinely 3D
magnetostatics problem with no translational or axial symmetry, so neither 2D
solver in :mod:`spin_dynamics.fields.nonlinear_magnetostatics` can represent it.
:class:`spin_dynamics.fields.ReducedScalarPotential3D` solves it on a structured
grid via the reduced scalar potential ``H = H_s - grad(psi)``.

The uniform interior field has the closed form ``B_in = mu0 * 3 mu_r/(mu_r+2) H0``
(Clausius-Mossotti / magnetic sphere), which lets the example *quantify* the
solver's accuracy -- and its documented limit. RSP carries a cancellation error
inside high-permeability material that grows as ``~ mu_r (h/L)^2``
(``docs/reduced_scalar_potential_3d.md``, issue 1), so the sphere is accurate at
low ``mu_r`` and degrades predictably as ``mu_r`` rises. The example shows exactly
that, rather than hiding it.

Panels:
1. ``B_z`` in the ``z = 0`` plane: interior concentration + external dipole lobes.
2. Interior ``B`` vs ``mu_r``: solver versus the analytic sphere -- agreement at
   low ``mu_r``, the RSP cancellation error opening up as ``mu_r`` grows.
3. On-axis ``B_z(z)`` through the sphere at a representative ``mu_r``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.nonlinear_magnetostatics import MU0, linear_material  # noqa: E402
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D  # noqa: E402


def analytic_b_in(mu_r: float, h0: float) -> float:
    """Uniform interior ``B`` of a permeable sphere in a uniform applied field."""

    return MU0 * mu_r * 3.0 * h0 / (mu_r + 2.0)


def solve_sphere(mu_r: float, h0: float, half_m: float, radius_m: float, n: int):
    """Solve a ``mu_r`` sphere of ``radius_m`` in a uniform ``H0 = h0 z`` field."""

    g = np.linspace(-half_m, half_m, n)
    prob = ReducedScalarPotential3D(g, g, g)
    prob.add_material(prob.sphere((0.0, 0.0, 0.0), radius_m), linear_material(mu_r))
    prob.add_uniform_source_field((0.0, 0.0, h0))
    sol = prob.solve()
    return prob, sol


def build_data(args) -> dict:
    half = args.box_mm * 1e-3
    radius = args.radius_mm * 1e-3

    # Field map at a representative mu_r.
    prob, sol = solve_sphere(args.mu_r, args.h0, half, radius, args.n)
    kz = int(np.argmin(np.abs(prob.z)))
    bz_plane = sol.b_z[:, :, kz]
    inner = prob.sphere((0.0, 0.0, 0.0), 0.5 * radius)
    b_in_map = float(sol.mean_b_in(inner)[2])

    # On-axis profile B_z(z) at x = y = 0.
    ic = int(np.argmin(np.abs(prob.x)))
    z_mm = prob.z * 1e3
    bz_axis = sol.b_z[ic, ic, :]

    # Accuracy sweep vs mu_r (coarser grid to keep the sweep quick).
    mu_list = np.array(args.mu_sweep, dtype=float)
    solved, exact = [], []
    for mu_r in mu_list:
        p, s = solve_sphere(mu_r, args.h0, half, radius, args.sweep_n)
        solved.append(float(s.mean_b_in(p.sphere((0.0, 0.0, 0.0), 0.4 * radius))[2]))
        exact.append(analytic_b_in(mu_r, args.h0))

    return {
        "x_mm": prob.x * 1e3,
        "bz_plane": bz_plane,
        "b_in_map": b_in_map,
        "b_in_exact": analytic_b_in(args.mu_r, args.h0),
        "z_mm": z_mm,
        "bz_axis": bz_axis,
        "radius_mm": args.radius_mm,
        "mu_list": mu_list,
        "solved_mt": np.array(solved) * 1e3,
        "exact_mt": np.array(exact) * 1e3,
        "iters": sol.iterations,
    }


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mu-r", type=float, default=5.0, help="mu_r for the map/profile panels")
    parser.add_argument("--h0", type=float, default=1000.0, help="applied field H0 (A/m)")
    parser.add_argument("--radius-mm", type=float, default=45.0)
    parser.add_argument("--box-mm", type=float, default=150.0)
    parser.add_argument("--n", type=int, default=41)
    parser.add_argument("--sweep-n", type=int, default=31)
    parser.add_argument("--mu-sweep", type=float, nargs="+",
                        default=[2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)

    extent = [data["x_mm"][0], data["x_mm"][-1], data["x_mm"][0], data["x_mm"][-1]]
    im = axes[0].imshow(data["bz_plane"].T * 1e3, origin="lower", extent=extent, cmap="RdBu_r")
    axes[0].add_patch(plt.Circle((0, 0), data["radius_mm"], fill=False, color="k", lw=0.8))
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title(f"B_z in z=0 plane (mu_r={args.mu_r:g})")
    fig.colorbar(im, ax=axes[0], label="B_z (mT)")

    ax1 = axes[1]
    ax1.plot(data["mu_list"], data["exact_mt"], "k-o", label="analytic sphere")
    ax1.plot(data["mu_list"], data["solved_mt"], "s--", color="tab:red", label="RSP solver")
    ax1.set_xlabel("mu_r")
    ax1.set_ylabel("interior B_z (mT)")
    ax1.set_title("Accuracy vs mu_r (cancellation error grows)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[2]
    ax2.plot(data["z_mm"], data["bz_axis"] * 1e3, color="tab:blue")
    ax2.axvspan(-data["radius_mm"], data["radius_mm"], color="gray", alpha=0.12, label="sphere")
    ax2.axhline(data["b_in_exact"] * 1e3, color="k", ls=":", lw=0.8, label="analytic interior")
    ax2.set_xlabel("z (mm)")
    ax2.set_ylabel("B_z (mT)")
    ax2.set_title(f"On-axis B_z (mu_r={args.mu_r:g})")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    err = 100.0 * (data["b_in_map"] / data["b_in_exact"] - 1.0)
    print("Soft-magnetic sphere in a uniform field (reduced scalar potential)")
    print(f"  mu_r                       : {args.mu_r:g}")
    print(f"  interior B_z (solver)      : {data['b_in_map'] * 1e3:.2f} mT")
    print(f"  interior B_z (analytic)    : {data['b_in_exact'] * 1e3:.2f} mT")
    print(f"  relative error             : {err:+.1f} %  (grows ~ mu_r (h/L)^2)")
    print(f"  nonlinear/linear iterations: {data['iters']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
