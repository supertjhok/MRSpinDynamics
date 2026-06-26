"""Depth-resolved relaxation and diffusion in the NMR-MOUSE field.

This drives the moving-isochromat engine with the magnet's own ``B0`` field
(from ``spin_dynamics.fields.magnetostatics``): walkers are seeded around the
frequency-selected sensitive slice and diffuse through the *real* static gradient,
so the slice selection, the echo train, and its diffusion attenuation all emerge
from spins moving through the spatially structured field -- physics that a fixed
off-resonance distribution cannot reproduce.

On a layered phantom (a water layer, a gap, and a gel layer) it shows:

1. The depth profile of CPMG signal -- the gap appears as a hole, the layers as
   plateaus (depth resolution from the frequency-selected slice).
2. The apparent T2 versus depth.
3. The diffusion coefficient versus depth, measured by the diffusion-on/off echo
   ratio, recovering the fast (water) and slow (gel) layers -- with the accuracy
   honestly falling off with depth as the static gradient weakens.

Run with ``--output figure.png`` to save, or omit it to show interactively.
The walker Monte-Carlo makes this example take ~1-2 minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Depth-resolved T2 and diffusion in the NMR-MOUSE field, simulated by "
            "walkers moving through the magnet's real static gradient."
        )
    )
    parser.add_argument("--gap", type=float, default=12.0, help="Magnet gap (mm).")
    parser.add_argument("--magnet-mm", type=float, default=20.0, help="Bar size (mm).")
    parser.add_argument("--remanence", type=float, default=1.30, help="Br (T).")
    parser.add_argument("--num-depths", type=int, default=13,
                        help="Frequencies (depths) in the profile sweep.")
    parser.add_argument("--num-d-depths", type=int, default=5,
                        help="Depths at which to measure diffusion (slower).")
    parser.add_argument("--seeds", type=int, default=4, help="Walker seeds to average.")
    parser.add_argument("--walkers", type=int, default=14, help="Walkers per cell.")
    parser.add_argument("--output", type=Path, help="Optional output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields.magnetostatics import (
        nmr_mouse_magnets, bar_array_b0, GAMMA_PROTON,
    )
    from spin_dynamics.workflows.single_sided import (
        SampleLayer, LayeredSample, mouse_depth_profile, measure_diffusion_at_depth,
    )

    mw = args.magnet_mm * 1e-3
    bars, yoke = nmr_mouse_magnets(magnet_width=mw, magnet_height=mw,
                                   gap=args.gap * 1e-3, remanence=args.remanence)
    # Layered phantom: water (fast D, long T2) / gap (no signal) / gel (slow D, short T2).
    sample = LayeredSample([
        SampleLayer(0.022, 0.030, rho=1.0, t2=0.060, diffusion=2.3e-9),
        SampleLayer(0.030, 0.034, rho=0.0, t2=0.060, diffusion=0.0),
        SampleLayer(0.034, 0.044, rho=1.0, t2=0.015, diffusion=0.5e-9),
    ])

    def freqs_for_depths(ys):
        b = np.hypot(*bar_array_b0(np.zeros_like(ys), ys, bars, yoke_y=yoke))
        return GAMMA_PROTON * b / (2 * np.pi)

    prof_depths = np.linspace(0.0225, 0.043, int(args.num_depths))
    prof_freqs = freqs_for_depths(prof_depths)
    print(f"depth profile: {len(prof_freqs)} frequencies "
          f"({prof_freqs.min()/1e6:.1f}..{prof_freqs.max()/1e6:.1f} MHz)")
    profile = mouse_depth_profile(
        bars, sample, prof_freqs, yoke_y=yoke, echo_time=2.0e-4, num_echoes=48,
        depth_halfwidth=0.6e-3, n_depth=101, walkers_per_cell=args.walkers,
        substeps_per_interval=3, seed=1,
    )

    # Diffusion at a few depths (skip the gap, where there is no signal).
    d_depths = np.linspace(0.024, 0.042, int(args.num_d_depths))
    d_depths = d_depths[(d_depths < 0.030) | (d_depths > 0.034)]
    d_freqs = freqs_for_depths(d_depths)
    print(f"measuring diffusion at {len(d_freqs)} depths (ratio method)...")

    def local_gradient(y):  # on-axis |d|B0|/dy| (T/m)
        dy = 1e-5
        bu = np.hypot(*bar_array_b0([0.0], [y + dy], bars, yoke_y=yoke))[0]
        bd = np.hypot(*bar_array_b0([0.0], [y - dy], bars, yoke_y=yoke))[0]
        return abs(bu - bd) / (2.0 * dy)

    d_meas = []
    for f, y in zip(d_freqs, d_depths):
        # Tune the echo time to the local gradient (keep gamma*G*tE ~ const) so the
        # diffusion weighting is comparable at every depth -- the experimental knob.
        tE = float(np.clip(1.2e-4 * (20.0 / local_gradient(y)), 6e-5, 3.0e-4))
        r = measure_diffusion_at_depth(
            bars, sample, float(f), yoke_y=yoke, echo_time=tE, num_echoes=40,
            n_seeds=int(args.seeds), depth_halfwidth=0.8e-3, n_depth=121,
            walkers_per_cell=args.walkers + 2, substeps_per_interval=4,
        )
        d_meas.append(r)
        print(f"  depth {r.depth*1e3:4.1f} mm  G={r.local_gradient:5.1f} T/m  "
              f"D={r.diffusion*1e9:.2f}e-9")

    # ---- plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0))
    dz = profile.depths * 1e3

    ax = axes[0, 0]
    ax.plot(dz, profile.signal, "o-", color="tab:blue")
    ax.axvspan(30, 34, color="gray", alpha=0.2, label="gap (no signal)")
    ax.set(xlabel="depth (mm)", ylabel="CPMG signal (first echo)",
           title="Depth profile: signal vs depth")
    ax.legend()

    ax = axes[0, 1]
    truth = sample.properties(profile.depths)
    ax.plot(dz, profile.t2_eff * 1e3, "s-", color="tab:red", label="apparent T2 (sim)")
    ax.plot(dz, truth["t2"] * 1e3, "k--", alpha=0.5, label="intrinsic T2 (true)")
    ax.set(xlabel="depth (mm)", ylabel="T2 (ms)", ylim=(0, 80),
           title="Apparent T2 vs depth (diffusion-shortened where G is large)")
    ax.legend()

    ax = axes[1, 0]
    dd = np.array([r.depth for r in d_meas]) * 1e3
    df = np.array([r.diffusion for r in d_meas]) * 1e9
    dt = sample.properties(np.array([r.depth for r in d_meas]))["diffusion"] * 1e9
    ax.plot(dd, df, "o", ms=8, color="tab:green", label="D fitted (sim)")
    ax.plot(dd, dt, "k_", ms=18, label="D true")
    ax.set(xlabel="depth (mm)", ylabel=r"D ($10^{-9}$ m$^2$/s)",
           title="Diffusion vs depth (ratio method)")
    ax.legend()

    ax = axes[1, 1]
    for i in (1, len(profile.depths) - 2):
        ax.semilogy(profile.echo_times * 1e3, profile.echo_amplitudes[i]
                    / profile.echo_amplitudes[i][0],
                    label=f"depth {profile.depths[i]*1e3:.0f} mm")
    ax.set(xlabel="echo time (ms)", ylabel="echo amplitude (norm.)",
           title="CPMG echo trains (single-sided shape)")
    ax.legend()

    fig.suptitle("NMR-MOUSE: depth-resolved relaxation and diffusion "
                 "(walkers in the real magnet field)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
