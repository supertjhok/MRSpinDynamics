"""Double diffusion encoding (DDE / double-PGSE) reveals microscopic anisotropy.

Single-PGSE measures the diffusion tensor, so in an orientationally disordered
("powder") sample of anisotropic compartments it sees only the isotropic
orientation average -- the microscopic anisotropy is hidden. Double diffusion
encoding applies two gradient pairs separated by a mixing time and varies the
angle ``psi`` between them. The echo then carries a ``cos 2*psi`` modulation
whose amplitude reports the *microscopic* anisotropy of the local geometry, and
because it depends only on the relative angle it survives powder averaging.

This example uses the random-walker DDE backend (``run_dde_walkers``) with an
elliptical reflecting pore (``make_elliptical_reflector``), contrasted with an
equal-area circular pore:

1. Geometry -- walkers fill the elliptical and circular pores; the two encoding
   directions ``q1`` and ``q2`` are drawn for one ``psi``.
2. Single orientation -- ``E(psi)`` for the ellipse develops a ``cos 2*psi``
   modulation that the isotropic circle lacks.
3. Powder average -- rotating the gradient pair (equivalent to averaging over
   pore orientations) removes the orientation-dependent part, yet the ellipse
   retains a ``cos 2*psi`` term while the circle stays flat: microscopic
   anisotropy that single-PGSE cannot see.

The ``cos 2*psi`` amplitude is a higher-order (``q^4``) effect, so strong
diffusion weighting (large ``--gradient-amplitude``) is needed to see it in the
powder average. Run with ``--output figure.png`` to save, or omit to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8  # rad/(s*T), proton gyromagnetic ratio


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Random-walker double diffusion encoding (DDE) in an elliptical "
            "pore: the cos(2 psi) angular modulation reveals microscopic "
            "anisotropy that survives powder averaging."
        )
    )
    parser.add_argument("--semi-major", type=float, default=8.0e-6,
                        help="Ellipse semi-axis along x (m).")
    parser.add_argument("--semi-minor", type=float, default=3.0e-6,
                        help="Ellipse semi-axis along z (m).")
    parser.add_argument("--gradient-amplitude", type=float, default=1.0,
                        help="Gradient amplitude |G| per block (T/m).")
    parser.add_argument("--gradient-duration", type=float, default=1.0e-3,
                        help="Gradient-pulse duration delta (s).")
    parser.add_argument("--diffusion-time", type=float, default=12.0e-3,
                        help="Per-block diffusion time Delta (s).")
    parser.add_argument("--mixing-time", type=float, default=1.0e-3,
                        help="Mixing time between the two encoding blocks (s).")
    parser.add_argument("--diffusion-coefficient", type=float, default=2.0e-9,
                        help="Bulk diffusion coefficient (m^2/s).")
    parser.add_argument("--num-angles", type=int, default=14,
                        help="Number of psi samples over 0..360 degrees.")
    parser.add_argument("--num-orientations", type=int, default=5,
                        help="Pore orientations averaged for the powder panel.")
    parser.add_argument("--grid", type=int, default=15,
                        help="Spatial cells per axis seeding walkers in the pore.")
    parser.add_argument("--walkers-per-cell", type=int, default=64,
                        help="Random walkers per spatial cell.")
    parser.add_argument("--substeps", type=int, default=10,
                        help="Diffusion substeps per interval (refine vs pore size).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed, reused across angles for smooth curves.")
    parser.add_argument("--output", type=Path,
                        help="Optional output PNG path. If omitted, show the plot.")
    return parser.parse_args()


def _build_pore(semi_axes: tuple[float, float], grid: int):
    from spin_dynamics.motion import make_motion_field_maps_2d

    ax, az = semi_axes
    x = np.linspace(-ax, ax, int(grid))
    z = np.linspace(-az, az, int(grid))
    xx, zz = np.meshgrid(x, z, indexing="ij")
    rho = ((xx / ax) ** 2 + (zz / az) ** 2 <= 1.0).astype(np.float64)
    return rho, x, z, make_motion_field_maps_2d(x, z)


def _echo(args, reflector, pore, angle1: float, angle2: float) -> float:
    from spin_dynamics.workflows import run_dde_walkers

    rho, x, z, fields = pore
    result = run_dde_walkers(
        rho=rho, x_axis=x, z_axis=z, fields=fields,
        gradient_amplitude=args.gradient_amplitude,
        gradient_duration=args.gradient_duration,
        diffusion_time=args.diffusion_time,
        mixing_time=args.mixing_time,
        angle1=angle1, angle2=angle2,
        diffusion_coefficient=args.diffusion_coefficient,
        walkers_per_cell=args.walkers_per_cell, seed=args.seed, jitter=True,
        excitation_duration=40.0e-6, refocusing_duration=80.0e-6,
        boundary=reflector, substeps_per_interval=args.substeps,
    )
    return abs(result.signal[0]) / float(rho.sum())


def _fit_harmonics(psi: np.ndarray, e_values: np.ndarray):
    """Least-squares fit E0 + E1 cos(psi) + E2 cos(2 psi)."""

    design = np.column_stack([np.ones_like(psi), np.cos(psi), np.cos(2.0 * psi)])
    coeffs, *_ = np.linalg.lstsq(design, e_values, rcond=None)
    return coeffs


def _sweep_single(args, reflector, pore, psi: np.ndarray) -> np.ndarray:
    return np.array([_echo(args, reflector, pore, 0.0, p) for p in psi])


def _sweep_powder(args, reflector, pore, psi: np.ndarray) -> np.ndarray:
    # Rotating the (q1, q2) pair by phi is equivalent to rotating the pore, so
    # averaging over phi performs the powder (orientation) average.
    phis = np.linspace(0.0, np.pi, int(args.num_orientations), endpoint=False)
    out = np.zeros_like(psi)
    for index, p in enumerate(psi):
        out[index] = float(np.mean([_echo(args, reflector, pore, phi, phi + p)
                                     for phi in phis]))
    return out


def _plot(plt, args, *, psi, ell_pore, ell_single, circ_single,
          ell_powder, circ_powder):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    psi_deg = np.degrees(psi)
    dense = np.linspace(0.0, 2.0 * np.pi, 361)
    dense_deg = np.degrees(dense)

    def model(coeffs, angle):
        return coeffs[0] + coeffs[1] * np.cos(angle) + coeffs[2] * np.cos(2.0 * angle)

    # Panel 1: geometry -- elliptical pore walkers + equal-area circle + q-vectors.
    rho, x, z, _ = ell_pore
    xx, zz = np.meshgrid(x, z, indexing="ij")
    inside = rho > 0
    axes[0].scatter(xx[inside] * 1e6, zz[inside] * 1e6, s=6, color="#d62728", alpha=0.35)
    a_um, b_um = args.semi_major * 1e6, args.semi_minor * 1e6
    r_circ_um = float(np.sqrt(a_um * b_um))
    theta = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(a_um * np.cos(theta), b_um * np.sin(theta), "k-", linewidth=1.2)
    axes[0].plot(r_circ_um * np.cos(theta), r_circ_um * np.sin(theta), "b--",
                 linewidth=1.0, label="equal-area circle")
    scale = 0.9 * a_um
    axes[0].annotate("", xy=(scale, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2))
    axes[0].annotate("", xy=(scale * np.cos(np.pi / 3), scale * np.sin(np.pi / 3)),
                     xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2))
    axes[0].text(scale, -0.9, "q1", color="#1f77b4", fontsize="small")
    axes[0].text(scale * np.cos(np.pi / 3) + 0.4, scale * np.sin(np.pi / 3), "q2",
                 color="#2ca02c", fontsize="small")
    axes[0].set_aspect("equal")
    axes[0].set_xlim(-a_um * 1.1, a_um * 1.1)
    axes[0].set_ylim(-a_um * 1.1, a_um * 1.1)
    axes[0].set_xlabel("x (um)")
    axes[0].set_ylabel("z (um)")
    axes[0].set_title(f"Elliptical pore {a_um:.0f} x {b_um:.0f} um")
    axes[0].legend(fontsize="x-small", loc="upper right")

    # Panel 2: single-orientation E(psi); ellipse develops a cos(2 psi) term.
    ce = _fit_harmonics(psi, ell_single)
    cc = _fit_harmonics(psi, circ_single)
    axes[1].plot(psi_deg, ell_single, "o", color="#d62728", markersize=4, label="ellipse")
    axes[1].plot(psi_deg, circ_single, "s", color="#1f77b4", markersize=4, label="circle")
    axes[1].plot(dense_deg, model(ce, dense), "-", color="#d62728", linewidth=1.0, alpha=0.7)
    axes[1].plot(dense_deg, model(cc, dense), "-", color="#1f77b4", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("angle between gradients psi (deg)")
    axes[1].set_ylabel("E = |S| / M0")
    axes[1].set_title("Single orientation")
    axes[1].text(0.03, 0.04,
                 f"cos2psi amp: ellipse {ce[2]:+.3f}\n              circle {cc[2]:+.3f}",
                 transform=axes[1].transAxes, fontsize="x-small", va="bottom",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="small", loc="upper right")

    # Panel 3: powder average -- cos(2 psi) survives only for the ellipse.
    cep = _fit_harmonics(psi, ell_powder)
    ccp = _fit_harmonics(psi, circ_powder)
    axes[2].plot(psi_deg, ell_powder, "o", color="#d62728", markersize=4, label="ellipse")
    axes[2].plot(psi_deg, circ_powder, "s", color="#1f77b4", markersize=4, label="circle")
    axes[2].plot(dense_deg, model(cep, dense), "-", color="#d62728", linewidth=1.0, alpha=0.7)
    axes[2].plot(dense_deg, model(ccp, dense), "-", color="#1f77b4", linewidth=1.0, alpha=0.7)
    axes[2].set_xlabel("angle between gradients psi (deg)")
    axes[2].set_ylabel("E = |S| / M0")
    axes[2].set_title("Powder average (orientation-averaged)")
    axes[2].text(0.03, 0.04,
                 f"cos2psi amp: ellipse {cep[2]:+.3f}\n              circle {ccp[2]:+.3f}",
                 transform=axes[2].transAxes, fontsize="x-small", va="bottom",
                 bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.8))
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize="small", loc="upper right")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.motion import make_circular_reflector, make_elliptical_reflector

    semi_axes = (args.semi_major, args.semi_minor)
    r_circ = float(np.sqrt(args.semi_major * args.semi_minor))  # equal area
    ell_pore = _build_pore(semi_axes, args.grid)
    circ_pore = _build_pore((r_circ, r_circ), args.grid)
    ell_reflector = make_elliptical_reflector((0.0, 0.0), semi_axes)
    circ_reflector = make_circular_reflector((0.0, 0.0), r_circ)

    psi = np.linspace(0.0, 2.0 * np.pi, int(args.num_angles), endpoint=False)
    ell_single = _sweep_single(args, ell_reflector, ell_pore, psi)
    circ_single = _sweep_single(args, circ_reflector, circ_pore, psi)
    ell_powder = _sweep_powder(args, ell_reflector, ell_pore, psi)
    # The circle is isotropic, so rotating the gradient pair leaves the echo
    # unchanged: its powder average equals its single-orientation curve.
    circ_powder = circ_single

    b_block = GAMMA**2 * (args.gradient_amplitude * args.gradient_duration) ** 2 * (
        args.diffusion_time - args.gradient_duration / 3.0
    )
    print(f"per-block b: {b_block:.3e} s/m^2")
    print(f"ellipse cos2psi amplitude: single {_fit_harmonics(psi, ell_single)[2]:+.4f}, "
          f"powder {_fit_harmonics(psi, ell_powder)[2]:+.4f}")
    print(f"circle  cos2psi amplitude: single {_fit_harmonics(psi, circ_single)[2]:+.4f}, "
          f"powder {_fit_harmonics(psi, circ_powder)[2]:+.4f}")
    print("Nonzero powder cos2psi for the ellipse is the microscopic-anisotropy "
          "signature single-PGSE cannot resolve.")

    fig = _plot(plt, args, psi=psi, ell_pore=ell_pore, ell_single=ell_single,
                circ_single=circ_single, ell_powder=ell_powder, circ_powder=circ_powder)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
