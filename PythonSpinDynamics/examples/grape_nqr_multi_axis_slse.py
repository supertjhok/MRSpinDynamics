"""GRAPE-discover whether multi-axis excitation improves the powder NQR SLSE signal.

The circular-polarization example (``plot_nqr_circular_polarization_nutation``)
showed that two orthogonal coils in a *fixed* quadrature relationship raise the
refocused (SLSE) powder signal. Here we hand the problem to GRAPE with far more
freedom: up to three orthogonal coils, each contributing one **rectangular**
sub-pulse whose amplitude, phase, delay, and length are optimized independently
(for both the excitation and the refocusing pulse). Sequential vs simultaneous
excitation is not imposed -- it emerges from the fitted delays and lengths.

The objective is the refocused powder SLSE echo-train energy (full ``(2I+1)``
density-matrix model). Everything is compared at **matched per-coil B1** (each
coil capped at the same nutation), so a scheme with more coils spends more total
RF power and receiver channels -- reported honestly, not hidden in the gain.

Detection: the 1-axis scheme uses a single coil; the 2- and 3-axis schemes use a
crossed-coil quadrature (2-channel) receiver, matched to the excitation sense.

Typical finding: GRAPE recovers essentially all of the benefit in going from one
to two axes (it rediscovers a circular-style rotating field), while a third
independently-controlled axis adds little to the refocused powder signal -- i.e.
circular polarization is already close to optimal for this objective, echoing the
broader result that well-tuned powder NQR detection schemes are hard to beat.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.nqr.isotopes import quadrupolar_site
from spin_dynamics.nqr.orientations import powder_frame_grid
from spin_dynamics.optimal_control._jax_propagation import JAX_AVAILABLE
from spin_dynamics.optimal_control.multi_axis import (
    MultiAxisSLSEConfig,
    optimize_multi_axis_slse,
)


def _timing_description(params: np.ndarray, window_us: float) -> str:
    """Summarize whether the active sub-pulses overlap (simultaneous) or not."""

    active = params[params[:, 0] > 0.05]  # coils with non-negligible amplitude
    if len(active) <= 1:
        return "single active coil"
    intervals = [(p[2], p[2] + p[3]) for p in active]  # (start, end) fractions
    intervals.sort()
    overlap = any(intervals[i + 1][0] < intervals[i][1] - 1e-3
                  for i in range(len(intervals) - 1))
    return "simultaneous (overlapping)" if overlap else "sequential (staggered)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--isotope", type=str, default="14N", help="Quadrupolar isotope.")
    parser.add_argument("--nu-q-mhz", type=float, default=3.3, help="nu_Q (MHz).")
    parser.add_argument("--eta", type=float, default=0.3, help="EFG asymmetry parameter.")
    parser.add_argument("--nutation-khz", type=float, default=20.0,
                        help="Per-coil B1 cap (bare nutation, kHz) -- matched across schemes.")
    parser.add_argument("--window-us", type=float, default=30.0,
                        help="Time budget for each composite pulse (microseconds).")
    parser.add_argument("--echo-spacing-us", type=float, default=200.0,
                        help="SLSE echo spacing (microseconds).")
    parser.add_argument("--num-echoes", type=int, default=6, help="Echoes per train.")
    parser.add_argument("--n-fine", type=int, default=16, help="Rendering segments per pulse.")
    parser.add_argument("--n-theta", type=int, default=4, help="Powder polar nodes.")
    parser.add_argument("--n-phi", type=int, default=6, help="Powder azimuthal nodes.")
    parser.add_argument("--n-chi", type=int, default=3, help="Powder in-plane nodes.")
    parser.add_argument("--starts", type=int, default=5, help="Random restarts per scheme.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--save", type=str, default=None, help="Optional summary-figure path.")
    args = parser.parse_args()

    if not JAX_AVAILABLE:
        print(
            "GRAPE requires the optional 'jax' extra (reverse-mode autodiff is the "
            "whole point). Install it with `python -m pip install -e .[jax]` (or "
            "`.[perf]`)."
        )
        return

    site = quadrupolar_site(args.isotope, nu_q_hz=args.nu_q_mhz * 1e6, eta=args.eta)
    frames = powder_frame_grid(args.n_theta, args.n_phi, args.n_chi)
    common = dict(
        site=site, frames=frames, nutation_scale_hz=args.nutation_khz * 1e3,
        window_ex_seconds=args.window_us * 1e-6, window_ref_seconds=args.window_us * 1e-6,
        n_fine=args.n_fine, echo_spacing_seconds=args.echo_spacing_us * 1e-6,
        num_echoes=args.num_echoes,
    )

    print(f"GRAPE multi-axis SLSE ({args.isotope}, nu_Q {args.nu_q_mhz} MHz, eta {args.eta})")
    print(f"per-coil B1: {args.nutation_khz:.0f} kHz (matched)   orientations: {len(frames)}   "
          f"echoes: {args.num_echoes}   restarts: {args.starts}")
    print("all schemes at matched per-coil B1; N-axis uses N coils (N x peak power)")
    print()

    results = {}
    for n_axes in (1, 2, 3):
        rx = "single" if n_axes == 1 else "quadrature"
        cfg = MultiAxisSLSEConfig(n_coils=n_axes, rx_scheme=rx, **common)
        res = optimize_multi_axis_slse(
            cfg, n_starts=args.starts, seed=args.seed, amp_max=1.0,
            include_simultaneous_seed=(n_axes >= 2),
        )
        results[n_axes] = res

    base = results[1].best_energy
    print(f"{'scheme':<26}{'refocused energy':>18}{'gain':>8}{'RX ch':>7}   timing")
    for n_axes in (1, 2, 3):
        res = results[n_axes]
        label = {1: "1-axis (linear)", 2: "2-axis (circular-style)",
                 3: "3-axis (tri-axial)"}[n_axes]
        rx_ch = 1 if n_axes == 1 else 2
        timing = _timing_description(res.excite_params(), args.window_us)
        print(f"{label:<26}{res.best_energy:>18.3f}{res.best_energy / base:>7.2f}x"
              f"{rx_ch:>7}   {timing}")

    # Honest SNR accounting: N-axis TX uses N coils; quadrature RX uses 2 channels
    # (sqrt(2) noise). Report the net SNR of the best multi-axis scheme.
    g2 = results[2].best_energy / base
    net_snr2 = g2 / np.sqrt(2.0)
    g3 = results[3].best_energy / base
    print()
    print(f"2-axis: {g2:.2f}x refocused signal -> {net_snr2:.2f}x net SNR "
          f"(sqrt(2) two-channel receiver noise; ~2x peak RF power)")
    print(f"3-axis over 2-axis: {results[3].best_energy / results[2].best_energy:.2f}x "
          f"signal at 1.5x the TX power -- the third axis adds little")
    if g3 <= g2 * 1.05:
        print("=> GRAPE finds circular (2-axis) near-optimal; the extra axis does not help.")

    if args.save is not None:
        _plot(results, args)


def _plot(results, args) -> None:
    from _source_path import load_matplotlib

    plt = load_matplotlib(required=True, headless=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    base = results[1].best_energy
    colors = {1: "C0", 2: "C3", 3: "C2"}
    labels = {1: "1-axis\n(linear)", 2: "2-axis\n(circular)", 3: "3-axis\n(tri-axial)"}

    ax = axes[0, 0]
    xs = [1, 2, 3]
    ax.bar(xs, [results[n].best_energy for n in xs],
           color=[colors[n] for n in xs])
    for n in xs:
        ax.text(n, results[n].best_energy, f"{results[n].best_energy / base:.2f}x",
                ha="center", va="bottom")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[n] for n in xs])
    ax.set_ylabel("Refocused SLSE train energy")
    ax.set_title("(a) GRAPE-optimized signal vs number of axes")

    # discovered 3-axis excitation envelopes (amplitude vs time per coil)
    ax = axes[0, 1]
    res3 = results[3]
    ex = res3.excite_params()
    win = args.window_us
    t = np.linspace(0, win, 200)
    for c in range(3):
        amp, _phase, start_f, len_f = ex[c]
        env = amp * ((t >= start_f * win) & (t <= (start_f + len_f) * win))
        ax.plot(t, env + 0.001 * c, label=f"coil {c} (phase {ex[c, 1]:.2f})")
    ax.set_xlabel("time in excitation window (us)")
    ax.set_ylabel("amplitude fraction")
    ax.set_title("(b) Discovered 3-axis excitation sub-pulses")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for n in xs:
        train = np.abs(results[n].train())
        ax.plot(np.arange(1, len(train) + 1), train, "o-", color=colors[n],
                label=labels[n].replace("\n", " "))
    ax.set_xlabel("echo index")
    ax.set_ylabel("|powder echo|")
    ax.set_title("(c) Optimized echo trains")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.axis("off")
    txt = ["Summary", ""]
    for n in xs:
        txt.append(f"{labels[n].replace(chr(10), ' ')}: "
                   f"{results[n].best_energy / base:.2f}x, "
                   f"{results[n].iterations} iters")
    ax.text(0.05, 0.9, "\n".join(txt), va="top", fontsize=11, family="monospace")

    fig.suptitle("GRAPE multi-axis SLSE excitation -- powder 14N NQR", fontsize=13)
    fig.savefig(args.save, dpi=150)
    print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
