"""Optimize a constant-amplitude CPMG refocusing pulse for a B0 gradient.

Physical setup: a static B0 gradient spreads the sample across a band of
resonance offsets. A plain rectangular 180 deg refocusing pulse only inverts
spins near resonance, so off-resonant spins (the edges of the gradient) refocus
poorly and the CPMG echo train decays faster across the band. Here we
**phase-modulate** a composite refocusing pulse — splitting the pulse into
equal-length segments and optimizing each segment's phase — to maximize the
refocused transverse magnetization integrated across the offset band, i.e. the
CPMG echo amplitude in the gradient.

Crucially the segment amplitudes are all held at 1 (the ideal-probe v0crit
objective uses unit-amplitude RF segments), so **peak RF power is constant** —
only the phases change. This is the physically meaningful constraint for a
power-limited probe.

The optimization uses the ideal v0crit refocusing objective. With JAX installed
the phase gradient is computed by reverse-mode autodiff (the Phase 3 backend);
otherwise it falls back to the SciPy/pattern optimizer. See
``docs/performance.md``.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.optimization._jax_objectives import JAX_AVAILABLE
from spin_dynamics.optimization.drivers import (
    run_ideal_v0crit_refocusing_multistart,
)
from spin_dynamics.optimization.refocusing import (
    evaluate_ideal_v0crit_refocusing_pulse,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=10, help="Number of constant-amplitude phase segments.")
    parser.add_argument("--starts", type=int, default=8, help="Random restarts for the optimizer.")
    parser.add_argument("--numpts", type=int, default=201, help="Offset isochromats across the gradient band.")
    parser.add_argument("--maxoffs", type=float, default=5.0, help="Half-width of the offset band (gradient spread).")
    parser.add_argument("--segment-fraction", type=float, default=0.1, help="Segment length in units of a 180 pulse.")
    parser.add_argument(
        "--v0crit-weight",
        type=float,
        default=0.1,
        help="Weight of the v0crit smoothness term; small => maximize refocused signal.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the restarts.")
    parser.add_argument("--save", type=str, default=None, help="Optional path to save the refocusing-profile figure.")
    args = parser.parse_args()

    optimizer = "jax" if JAX_AVAILABLE else "auto"
    common = dict(
        segment_fraction=args.segment_fraction,
        numpts=args.numpts,
        maxoffs=args.maxoffs,
        v0crit_weight=args.v0crit_weight,
    )

    # Rectangular reference: equal-length, constant phase (= a plain block pulse
    # of the same total length and the same peak amplitude). With the default
    # 10 segments x 0.1 this is a true 1.0 x T180 rectangular refocusing pulse.
    rect = evaluate_ideal_v0crit_refocusing_pulse(
        np.zeros(args.segments, dtype=np.float64), **common
    )

    # Seed the search with the rectangular pulse plus random restarts, so the
    # reported optimum is guaranteed no worse than the rectangular baseline.
    rng = np.random.default_rng(args.seed)
    random_starts = rng.uniform(0.0, 2 * np.pi, size=(max(args.starts - 1, 1), args.segments))
    initial_phases = np.vstack([np.zeros((1, args.segments)), random_starts])

    result = run_ideal_v0crit_refocusing_multistart(
        args.segments,
        initial_phases=initial_phases,
        optimizer=optimizer,
        **common,
    )
    best = result.best_result.best_evaluation

    pulse_length_t180 = args.segments * args.segment_fraction
    improvement = best.axis_rms / rect.axis_rms if rect.axis_rms else float("nan")

    grad_backend = "autodiff (JAX)" if optimizer == "jax" else "finite-difference / pattern"
    print("Constant-amplitude refocusing-pulse optimization in a B0 gradient")
    print(f"gradient band (normalized offset): +/- {args.maxoffs}")
    print(f"segments: {args.segments}  (pulse length = {pulse_length_t180:.2f} x T180, peak RF held constant)")
    print(f"restarts: {args.starts}   gradient backend: {grad_backend}")
    print()
    print(f"refocused signal across band (integral |Masy|^2, higher = stronger echoes):")
    print(f"  rectangular 180 : {rect.axis_rms:.6f}")
    print(f"  optimized phases: {best.axis_rms:.6f}")
    print(f"  improvement     : {improvement:.2f}x")
    print()
    print(
        "optimized phases (rad): "
        f"{np.array2string(np.mod(best.phases, 2 * np.pi), precision=3, separator=', ')}"
    )

    # Edge-of-band refocusing is where the gain shows up most.
    edge = np.abs(best.del_w) > 0.6 * args.maxoffs
    rect_edge = float(np.mean(np.abs(rect.masy)[edge]))
    opt_edge = float(np.mean(np.abs(best.masy)[edge]))
    print(
        f"\nmean |Masy| at band edges (|offset| > {0.6 * args.maxoffs:.1f}): "
        f"rectangular {rect_edge:.4f} -> optimized {opt_edge:.4f}"
    )

    if args.save is not None:
        plt = load_matplotlib(required=True, headless=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rect.del_w, np.abs(rect.masy), label="rectangular 180", lw=1.5)
        ax.plot(best.del_w, np.abs(best.masy), label="optimized phases", lw=1.5)
        ax.set_xlabel("normalized resonance offset (B0 gradient)")
        ax.set_ylabel("|refocused transverse magnetization|")
        ax.set_title("CPMG refocusing across a B0 gradient (constant peak RF)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.save, dpi=150)
        print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
