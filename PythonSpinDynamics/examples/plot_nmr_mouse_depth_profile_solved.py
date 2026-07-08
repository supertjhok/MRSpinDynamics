"""Single-sided depth profiling driven by a 3D-solved NMR-MOUSE field.

This ties the 3D magnetostatic solver into the single-sided (NMR-MOUSE) workflow:
instead of the analytic bar-magnet model (which adds the iron yoke by the method
of images), the depth-resolved CPMG measurement is driven by a full
:class:`spin_dynamics.fields.ReducedScalarPotential3D` solve of the magnets *and*
the finite iron yoke, wrapped as a
:class:`spin_dynamics.workflows.SolvedMouseField`. The workflow is otherwise
unchanged -- the same moving-walker engine, the same frequency sweep -- so this
is a drop-in, higher-fidelity field source.

Panels:
1. On-axis ``|B0|`` vs depth from the solved field, with the excitation
   frequencies marked at the depths they select (frequency ↔ depth is the
   penetration-depth encoding).
2. The reconstructed depth profile of a layered phantom: the excited signal traces
   spin density, so a buried density gap reads as a hole.

The 3D solve (magnets + yoke) runs once via AMG; the frequency sweep then reuses
it. See ``plot_nmr_mouse_3d.py`` for the field itself and
``docs/reduced_scalar_potential_3d.md`` for the solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.magnetostatics import GAMMA_PROTON  # noqa: E402
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

    # On-axis |B0| vs depth and its frequency map.
    depths, larmor = field.larmor_profile(0.5e-3, args.depth_max_mm * 1e-3, GAMMA_PROTON)
    b0_axis = 2.0 * np.pi * larmor / GAMMA_PROTON  # |B0| (T)

    # Frequencies chosen to resonate at a set of target depths spanning the phantom.
    targets = np.linspace(args.profile_min_mm, args.profile_max_mm, args.n_freq) * 1e-3
    freqs = np.interp(targets, depths, larmor)

    gap = (args.gap_lo_mm * 1e-3, args.gap_hi_mm * 1e-3)
    sample = LayeredSample([
        SampleLayer(0.0, gap[0], rho=1.0, t2=args.t2),
        SampleLayer(gap[0], gap[1], rho=0.0),
        SampleLayer(gap[1], 20e-3, rho=1.0, t2=args.t2),
    ])
    prof = mouse_depth_profile(
        field, sample, freqs, echo_time=args.echo_ms * 1e-3, num_echoes=args.num_echoes,
        depth_halfwidth=0.4e-3, n_depth=31, walkers_per_cell=6,
        substeps_per_interval=2, depth_range=(0.5e-3, args.depth_max_mm * 1e-3), seed=0,
    )
    return {
        "depth_mm": depths * 1e3, "b0_mt": b0_axis * 1e3,
        "freqs_mhz": freqs / 1e6, "target_mm": targets * 1e3,
        "prof_depth_mm": prof.depths * 1e3, "signal": prof.signal,
        "gap_mm": (args.gap_lo_mm, args.gap_hi_mm),
        "b0_sensor": float(np.interp(args.sensor_mm * 1e-3, depths, b0_axis)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Magnet geometry (paper Fig. 1).
    parser.add_argument("--gap-mm", type=float, default=13.0)
    parser.add_argument("--magnet-w-mm", type=float, default=26.0)
    parser.add_argument("--magnet-h-mm", type=float, default=32.0)
    parser.add_argument("--yoke-h-mm", type=float, default=15.0)
    parser.add_argument("--bar-len-mm", type=float, default=40.0)
    parser.add_argument("--remanence-t", type=float, default=1.2)
    parser.add_argument("--yoke-mu-r", type=float, default=1000.0)
    parser.add_argument("--h-mm", type=float, default=2.0)
    parser.add_argument("--box-z-mm", type=float, default=90.0)
    parser.add_argument("--box-x-mm", type=float, default=50.0)
    parser.add_argument("--box-above-mm", type=float, default=50.0)
    parser.add_argument("--box-below-mm", type=float, default=70.0)
    # Sampling / phantom.
    parser.add_argument("--sensor-mm", type=float, default=2.5)
    parser.add_argument("--depth-max-mm", type=float, default=12.0)
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

    axes[0].plot(data["b0_mt"], data["depth_mm"], color="tab:blue")
    axes[0].scatter(np.interp(data["target_mm"], data["depth_mm"], data["b0_mt"]),
                    data["target_mm"], c=data["freqs_mhz"], cmap="viridis", zorder=3, s=25)
    axes[0].axhspan(*data["gap_mm"], color="gray", alpha=0.15, label="density gap")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("|B0| (mT)")
    axes[0].set_ylabel("depth below surface (mm)")
    axes[0].set_title("Solved B0 vs depth (points = excitation freqs)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(data["prof_depth_mm"], data["signal"] / (data["signal"].max() or 1.0),
                 "o-", color="tab:red")
    axes[1].axvspan(*data["gap_mm"], color="gray", alpha=0.15, label="density gap")
    axes[1].set_xlabel("resonant depth (mm)")
    axes[1].set_ylabel("excited signal (norm.)")
    axes[1].set_title("Depth profile of a layered phantom")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.2)

    print("Single-sided depth profile from a 3D-solved NMR-MOUSE field")
    print(f"  B0 at {args.sensor_mm:g} mm sensor : {data['b0_sensor'] * 1e3:.1f} mT"
          f"  ({GAMMA_PROTON * data['b0_sensor'] / (2 * np.pi) / 1e6:.1f} MHz)")
    print(f"  frequency sweep            : {data['freqs_mhz'].min():.1f}"
          f"-{data['freqs_mhz'].max():.1f} MHz over {args.n_freq} points")
    gap_mask = (data["prof_depth_mm"] >= data["gap_mm"][0]) & (data["prof_depth_mm"] <= data["gap_mm"][1])
    if gap_mask.any():
        print(f"  mean signal in the gap     : "
              f"{data['signal'][gap_mask].mean() / (data['signal'].max() or 1):.2f} of peak")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
