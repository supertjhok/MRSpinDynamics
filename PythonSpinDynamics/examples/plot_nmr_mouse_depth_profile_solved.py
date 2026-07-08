"""Single-sided depth profiling driven by a 3D-solved NMR-MOUSE field.

This ties the 3D magnetostatic solver into the single-sided (NMR-MOUSE) workflow:
the depth-resolved CPMG measurement is driven by a full
:class:`spin_dynamics.fields.ReducedScalarPotential3D` solve of the magnets *and*
the finite iron yoke (wrapped as a
:class:`spin_dynamics.workflows.SolvedMouseField`), instead of the analytic
image-yoke model.

**The measured signal is two factors, and they pull in opposite directions.**

1. *Intrinsic slice response* -- what the moving-walker sim returns: spin density,
   T2/diffusion contrast, and the slice volume. This *rises* with depth: the
   frequency-selected slice thickness is (pulse bandwidth)/gradient, and the
   static ``B0`` gradient weakens with depth, so the excited slice -- and thus the
   sample volume -- grows.
2. *Geometric detection sensitivity* -- the depth falloff the walker sim does not
   include (it uses an idealized uniform, per-slice-optimized RF and a
   density-only M0). By reciprocity and the Curie law the detected echo scales as

       S(d) ~ B0(d)^2 * B1(d)^2 ,

   with one ``B0`` from thermal polarization, one from the ``omega_0`` reception
   (Faraday), one ``B1`` from transmit efficiency and one from coil reception.
   For a surface coil ``B1`` falls steeply with depth, so ``S`` collapses.

The *measured* profile is their product. ``S(d)`` (B1-dominated) wins, so the real
signal *decreases* with depth even though the intrinsic response rises -- which is
why a raw walker profile alone looks suspiciously inverted. Panels:

1. The sensitivity factors vs depth (``B0``, ``B1``, and ``S = B0^2 B1^2``),
   showing the steep, B1-dominated falloff.
2. The depth profile of a layered phantom: the intrinsic (walker) response versus
   the measured signal (intrinsic x ``S``). Both resolve a buried density gap; only
   the measured one has the physical depth decay.

See ``plot_nmr_mouse_3d.py`` for the field and ``docs/reduced_scalar_potential_3d.md``
for the solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, circular_loop  # noqa: E402
from spin_dynamics.fields.nonlinear_magnetostatics import linear_material, ndfeb  # noqa: E402
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    LayeredSample,
    SampleLayer,
    SolvedMouseField,
    mouse_depth_profile,
)


def build_field(args) -> SolvedMouseField:
    """Solve the 3D NMR-MOUSE (two anti-parallel bars on an iron yoke)."""

    mm = 1e-3
    h = args.h_mm * mm

    def axis(lo, hi):
        return np.linspace(lo, hi, int(round((hi - lo) / h)) + 1)

    x = axis(-args.box_x_mm * mm, args.box_x_mm * mm)
    y = axis(-args.box_below_mm * mm, args.box_above_mm * mm)
    z = axis(-args.box_z_mm * mm, args.box_z_mm * mm)
    prob = ReducedScalarPotential3D(x, y, z)
    bar_x = (-0.5 * args.bar_len_mm * mm, 0.5 * args.bar_len_mm * mm)
    hg = 0.5 * args.gap_mm * mm
    outer = hg + args.magnet_w_mm * mm
    top, bot = 0.0, -args.magnet_h_mm * mm
    prob.add_material(prob.box(bar_x, (bot, top), (-outer, -hg)),
                      ndfeb(args.remanence_t), remanence_direction=(0, 1, 0))
    prob.add_material(prob.box(bar_x, (bot, top), (hg, outer)),
                      ndfeb(args.remanence_t), remanence_direction=(0, -1, 0))
    prob.add_material(prob.box(bar_x, (bot - args.yoke_h_mm * mm, bot), (-outer, outer)),
                      linear_material(args.yoke_mu_r))
    return SolvedMouseField(prob.solve(), depth_range=(0.5e-3, args.depth_max_mm * 1e-3))


def build_data(args) -> dict:
    field = build_field(args)
    # Surface RF coil in the gap: a loop in the x-z plane makes B1 along y (depth),
    # transverse to the dominant B0_z. B1 falls steeply with depth.
    coil = circular_loop((0.0, 0.0, 0.0), args.coil_radius_mm * 1e-3, axis="y", n_segments=64)

    # On-axis sensitivity factors vs depth (B0 from the field, B1 from the coil).
    depths = np.linspace(0.5e-3, args.depth_max_mm * 1e-3, 200)
    fm = field.plane_field_maps(np.array([0.0]), depths, coil_segments=coil,
                                coil_current=1.0, gamma=GAMMA_PROTON)
    b0 = fm.b0_magnitude[0]
    b1 = fm.b1_transverse[0]
    sensitivity = b0**2 * b1**2  # reciprocity + Curie (see module docstring)
    larmor = GAMMA_PROTON * b0 / (2.0 * np.pi)

    # Frequencies resonating at target depths spanning the phantom.
    targets = np.linspace(args.profile_min_mm, args.profile_max_mm, args.n_freq) * 1e-3
    freqs = np.interp(targets, depths, larmor)

    gap = (args.gap_lo_mm * 1e-3, args.gap_hi_mm * 1e-3)
    sample = LayeredSample([
        SampleLayer(0.0, gap[0], rho=1.0, t2=args.t2),
        SampleLayer(gap[0], gap[1], rho=0.0),        # buried density gap
        SampleLayer(gap[1], 20e-3, rho=1.0, t2=args.t2),
    ])
    # Intrinsic response: uniform-B1 walker sim (RF sensitivity applied separately).
    prof = mouse_depth_profile(
        field, sample, freqs, echo_time=args.echo_ms * 1e-3, num_echoes=args.num_echoes,
        depth_halfwidth=0.5e-3, n_depth=41, walkers_per_cell=16,
        substeps_per_interval=2, depth_range=(0.5e-3, args.depth_max_mm * 1e-3), seed=0,
    )
    s_at_depth = np.interp(prof.depths, depths, sensitivity)
    measured = prof.signal * s_at_depth
    return {
        "depth_mm": depths * 1e3,
        "b0_n": b0 / b0[0], "b1_n": b1 / b1[0], "sens_n": sensitivity / sensitivity[0],
        "prof_depth_mm": prof.depths * 1e3,
        "intrinsic": prof.signal / (prof.signal.max() or 1.0),
        "measured": measured / (measured.max() or 1.0),
        "gap_mm": (args.gap_lo_mm, args.gap_hi_mm),
        "b0_sensor": float(np.interp(args.sensor_mm * 1e-3, depths, b0)),
        "sens_falloff": float(sensitivity[0] / sensitivity[np.argmin(np.abs(depths - 10e-3))]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gap-mm", type=float, default=13.0)
    parser.add_argument("--magnet-w-mm", type=float, default=26.0)
    parser.add_argument("--magnet-h-mm", type=float, default=32.0)
    parser.add_argument("--yoke-h-mm", type=float, default=15.0)
    parser.add_argument("--bar-len-mm", type=float, default=40.0)
    parser.add_argument("--remanence-t", type=float, default=1.2)
    parser.add_argument("--yoke-mu-r", type=float, default=1000.0)
    parser.add_argument("--coil-radius-mm", type=float, default=6.0)
    parser.add_argument("--h-mm", type=float, default=2.0)
    parser.add_argument("--box-z-mm", type=float, default=90.0)
    parser.add_argument("--box-x-mm", type=float, default=50.0)
    parser.add_argument("--box-above-mm", type=float, default=50.0)
    parser.add_argument("--box-below-mm", type=float, default=70.0)
    parser.add_argument("--sensor-mm", type=float, default=2.5)
    parser.add_argument("--depth-max-mm", type=float, default=10.0)
    parser.add_argument("--profile-min-mm", type=float, default=1.5)
    parser.add_argument("--profile-max-mm", type=float, default=8.0)
    parser.add_argument("--n-freq", type=int, default=12)
    parser.add_argument("--gap-lo-mm", type=float, default=3.5)
    parser.add_argument("--gap-hi-mm", type=float, default=5.0)
    parser.add_argument("--t2", type=float, default=0.05)
    parser.add_argument("--echo-ms", type=float, default=0.2)
    parser.add_argument("--num-echoes", type=int, default=6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    data = build_data(args)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    axes[0].semilogy(data["depth_mm"], data["b0_n"], color="tab:blue", label="B0 (norm)")
    axes[0].semilogy(data["depth_mm"], data["b1_n"], color="tab:green", label="B1 (norm)")
    axes[0].semilogy(data["depth_mm"], data["sens_n"], color="tab:red", lw=2,
                     label="S = B0$^2$B1$^2$")
    axes[0].axvspan(*data["gap_mm"], color="gray", alpha=0.15, label="density gap")
    axes[0].set_xlabel("depth below surface (mm)")
    axes[0].set_ylabel("normalized sensitivity")
    axes[0].set_title("Detection sensitivity falls with depth (B1-dominated)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].plot(data["prof_depth_mm"], data["intrinsic"], "o--", color="tab:gray",
                 label="intrinsic (walker)")
    axes[1].plot(data["prof_depth_mm"], data["measured"], "o-", color="tab:red",
                 label="measured (x sensitivity)")
    axes[1].axvspan(*data["gap_mm"], color="gray", alpha=0.15, label="density gap")
    axes[1].set_xlabel("resonant depth (mm)")
    axes[1].set_ylabel("signal (norm.)")
    axes[1].set_title("Measured = intrinsic contrast x sensitivity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    print("Single-sided depth profile from a 3D-solved NMR-MOUSE field")
    print(f"  B0 at {args.sensor_mm:g} mm sensor : {data['b0_sensor'] * 1e3:.1f} mT"
          f"  ({GAMMA_PROTON * data['b0_sensor'] / (2 * np.pi) / 1e6:.1f} MHz)")
    print(f"  sensitivity falloff 0.5->10 mm : {data['sens_falloff']:.0f}x"
          f"  (B1-dominated; the walker sim omits this)")
    print("  intrinsic response RISES with depth (thicker slice as gradient weakens);")
    print("  measured signal FALLS with depth once the B0^2 B1^2 sensitivity is applied.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
