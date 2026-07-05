"""Reproduce the key results of Brill et al., Nonresonant Multiple Spin Echoes (2002).

Nonresonant NMR manipulates spins *without RF*, by suddenly switching and adiabatically
rotating applied magnetic fields (CSAR). Its Achilles heel is a non-reversible
background field -- Earth's field, unshieldable outside a magnet -- that survives the
field reversals and dephases the echoes within a few milliseconds. This example
reproduces the paper's central figures with the classical Bloch field-reversal engine
(``spin_dynamics.nonresonant``):

* **Fig. 1C / Fig. 2** -- the basic reversal sequence decays within tens of
  milliseconds from the Earth's-field residual phase, while the 90-degree CSAR sequence
  refocuses every *second* echo and sustains the train out to the intrinsic T2. The
  even echoes carry a ~50% component decaying as exp(-t/T2) plus a ~50% component with a
  slow, Bessel-like odd-even modulation.
* **Fig. 3** -- comparison of CSAR variants: 90-degree rotations (strong even-odd
  modulation), 180-degree rotations (both echoes refocused), and a supercycle of
  alternating rotation senses (better early compensation).

The fitted T2 of the CSAR even-echo envelope is printed against the input T2 -- the
long echo train is exactly what enables an accurate relaxation-time measurement in the
paper.

Note on the Fig. 3 waveforms (panel d): the (A) single-reversal sequence follows the
paper's Fig. 1D drive directly. The (B) 2-pi and (C) supercycle sequences here are
built from the same primitives (adiabatic rotations + sudden reversals) and reproduce
the same *effects* -- both echo parities refocusing (2-pi), and a supercycle that
suppresses the imperfection decay down to the T2 limit -- but their exact coil-current
waveforms differ from the specific implementations in Brill 2002 Fig. 3B/3C, which use
180-degree rotations and a particular four-unit sense supercycle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nonresonant import (  # noqa: E402
    NonresonantFieldModel,
    basic_reversal_sequence,
    csar_double_reversal_sequence,
    csar_sequence,
    csar_supercycle_sequence,
    effective_rotation,
    sample_isochromats,
    sequence_waveform,
    simulate_field_reversal_echoes,
)


def _fit_t2(times: np.ndarray, magnitudes: np.ndarray) -> float:
    """Fit an exponential envelope to (times, magnitudes); return T2 in seconds."""

    good = magnitudes > 1e-6
    if good.sum() < 3:
        return float("nan")
    slope = np.polyfit(times[good], np.log(magnitudes[good]), 1)[0]
    return float(-1.0 / slope) if slope < 0 else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-isochromats", type=int, default=600, help="Isochromat count.")
    parser.add_argument("--num-echoes", type=int, default=256, help="CSAR echoes.")
    parser.add_argument("--echo-spacing-us", type=float, default=2500.0,
                        help="Echo spacing in microseconds.")
    parser.add_argument("--t2-ms", type=float, default=510.0, help="Intrinsic T2 (ms).")
    parser.add_argument("--coil-mt", type=float, default=2.14,
                        help="Coil field magnitude at full current (mT).")
    parser.add_argument("--background-ut", type=float, default=50.0,
                        help="Non-reversible background (Earth's) field (uT).")
    parser.add_argument("--tau-rev-us", type=float, default=8.0,
                        help="Finite field-reversal time (us); the dominant imperfection.")
    parser.add_argument("--tilt-deg", type=float, default=18.0,
                        help="Coil-direction inhomogeneity (deg) setting the alpha spread.")
    parser.add_argument("--adiabatic-steps", type=int, default=160,
                        help="Sub-steps per adiabatic rotation (accuracy).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG.")
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)

    spacing = args.echo_spacing_us * 1e-6
    t2 = args.t2_ms * 1e-3
    tau_rev = args.tau_rev_us * 1e-6
    model = NonresonantFieldModel(
        coil_a_tesla=args.coil_mt * 1e-3, coil_b_tesla=args.coil_mt * 1e-3,
        background_tesla=args.background_ut * 1e-6)
    ens = sample_isochromats(model, args.n_isochromats, b_inhomogeneity=0.25,
                             direction_tilt_deg=args.tilt_deg, seed=args.seed)
    steps = dict(adiabatic_steps=args.adiabatic_steps)

    # Basic sequence -- rapid decay from the background field (fewer echoes needed).
    n_basic = min(args.num_echoes, max(40, int(0.12 / spacing)))
    basic = basic_reversal_sequence(model, echo_spacing_seconds=spacing)
    r_basic = simulate_field_reversal_echoes(model, ens, basic, num_echoes=n_basic,
                                             t2_seconds=t2)

    # CSAR (Fig. 1D / Fig. 2) -- sustained train with even/odd modulation.
    csar = csar_sequence(model, echo_spacing_seconds=spacing, tau_rev_seconds=tau_rev,
                         **steps)
    r_csar = simulate_field_reversal_echoes(model, ens, csar, num_echoes=args.num_echoes,
                                            t2_seconds=t2)
    n = np.arange(1, args.num_echoes + 1)
    even = n % 2 == 0
    odd = ~even
    times_ms = r_csar.echo_times * 1e3
    mag = r_csar.magnitude
    norm = mag[even].max() if mag[even].size else mag.max()
    mag = mag / norm

    # The odd-even modulation *exchanges* energy between neighbouring echoes rather
    # than losing it, so the combined amplitude of consecutive pairs cancels the
    # modulation and follows the intrinsic exp(-t/T2) -- the paper's "average decay".
    npair = 2 * (args.num_echoes // 2)
    raw = r_csar.magnitude[:npair]
    combined = np.sqrt(raw[0::2] ** 2 + raw[1::2] ** 2)
    pair_times = r_csar.echo_times[1:npair:2]
    # Fit the late portion, where the modulated component's ensemble dephasing has
    # damped and only the T2-decaying refocused component (~50%) remains.
    late = pair_times > 0.35 * pair_times.max()
    fit_t2 = _fit_t2(pair_times[late], combined[late])

    # Effective-rotation-axis statistics (the paper's analysis).
    axangles = [effective_rotation(ens, csar, k)
                for k in range(0, ens.size, max(1, ens.size // 60))]
    refocused_fraction = float(np.mean([ax[0] ** 2 for ax, _ in axangles]))
    mean_angle = float(np.rad2deg(np.mean([a for _, a in axangles])))

    print("Nonresonant field-reversal echoes (Brill et al., Science 297, 369, 2002)")
    print(f"proton, coil {args.coil_mt:.2f} mT, background {args.background_ut:.0f} uT, "
          f"echo spacing {args.echo_spacing_us / 1e3:.1f} ms")
    print(f"isochromats: {ens.size}   CSAR echoes: {args.num_echoes}   "
          f"finite reversal: {args.tau_rev_us:.0f} us")
    print()
    print(f"basic sequence: decays to 10% of initial by "
          f"{_time_to_fraction(r_basic.magnitude, r_basic.echo_times, 0.1):.0f} ms "
          f"(Earth's-field dephasing)")
    print(f"CSAR even-echo net rotation: {mean_angle:.1f} deg (~180 deg = pi refocusing)")
    print(f"refocused fraction <(n.x)^2>: {refocused_fraction:.2f} (~0.5, paper)")
    print(f"CSAR even-echo fitted T2: {fit_t2 * 1e3:.0f} ms  (input {args.t2_ms:.0f} ms)")

    if plt is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    ax = axes[0, 0]
    tb = r_basic.echo_times * 1e3
    ax.plot(tb, r_basic.magnitude / r_basic.magnitude.max(), "-", color="0.5",
            label="basic reversal (Fig. 1B)")
    ax.plot(times_ms[even], mag[even], "o", ms=3, color="C0", label="CSAR even echoes")
    ax.plot(times_ms[odd], mag[odd], "o", ms=3, mfc="none", color="C3",
            label="CSAR odd echoes")
    ax.plot(pair_times * 1e3, combined / norm / np.sqrt(2.0), "-", color="C2", lw=1.5,
            label="even+odd envelope")
    ax.plot(times_ms, np.exp(-r_csar.echo_times / t2), "--", color="k", lw=1,
            label=f"exp(-t/T2), T2={args.t2_ms:.0f} ms")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("echo magnitude (norm.)")
    ax.set_title("(a) Basic vs CSAR echo trains (Fig. 1C / Fig. 2)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    early = times_ms < min(120.0, times_ms.max())
    ax.plot(times_ms[even & early], mag[even & early], "o-", ms=3, color="C0",
            label="even")
    ax.plot(times_ms[odd & early], mag[odd & early], "o-", ms=3, mfc="none", color="C3",
            label="odd")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("echo magnitude (norm.)")
    ax.set_title("(b) Odd-even (Bessel-like) modulation, early echoes")
    ax.legend(fontsize=8)

    variants = [
        ("(A) single reversal (pi): even echoes", csar_sequence(
            model, echo_spacing_seconds=spacing, tau_rev_seconds=tau_rev, **steps), "C0"),
        ("(B) double reversal (2pi): both echoes", csar_double_reversal_sequence(
            model, echo_spacing_seconds=spacing, tau_rev_seconds=tau_rev), "C3"),
        ("(C) supercycle: sustained", csar_supercycle_sequence(
            model, echo_spacing_seconds=spacing, tau_rev_seconds=tau_rev, **steps), "C2"),
    ]

    ax = axes[1, 0]
    n_cmp = min(args.num_echoes, 100)
    for label, seq, color in variants:
        rv = simulate_field_reversal_echoes(model, ens, seq, num_echoes=n_cmp,
                                            t2_seconds=t2)
        mv = rv.magnitude
        # refocused-signal envelope: pair each echo with its neighbour so the fast
        # even-odd modulation folds into one clean sustain curve per sequence.
        npr = 2 * (n_cmp // 2)
        pair = np.sqrt(mv[0:npr:2] ** 2 + mv[1:npr:2] ** 2)
        pair = pair / pair[0]
        ax.plot(rv.echo_times[1:npr:2] * 1e3, pair, "-", lw=1.6, color=color, label=label)
    tt = np.linspace(0, n_cmp * spacing, 100)
    ax.plot(tt * 1e3, np.exp(-tt / t2), "--", color="k", lw=1,
            label=f"exp(-t/T2), T2={args.t2_ms:.0f} ms")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("refocused-signal envelope (norm.)")
    ax.set_title("(c) CSAR sequence comparison (Fig. 3); supercycle -> T2 limit")
    ax.legend(fontsize=7.5, loc="upper right")

    # (d) The drive current waveforms: solid = coil B, dashed = coil A (as in Fig. 3
    # insets). Stacked with vertical offsets over two echo periods.
    ax = axes[1, 1]
    coil = args.coil_mt
    for i, (label, seq, color) in enumerate(variants):
        offset = -3.2 * coil * i
        t_wf, i_a, i_b = sequence_waveform(seq, repeats=2)
        ax.plot(t_wf * 1e3, i_b * 1e3 + offset, "-", color=color, lw=1.3)
        ax.plot(t_wf * 1e3, i_a * 1e3 + offset, "--", color=color, lw=1.0, alpha=0.7)
        ax.text(0.0, offset + 1.4 * coil, label.split(":")[0], fontsize=7.5, color=color)
    ax.plot([], [], "-", color="0.3", label="I_B (coil B, solid)")
    ax.plot([], [], "--", color="0.3", label="I_A (coil A, dashed)")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("coil field (mT, offset)")
    ax.set_yticks([])
    ax.set_title("(d) Drive current waveforms")
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle("Nonresonant field-reversal spin echoes (Brill 2002)", fontsize=13)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"\nsaved: {args.output}")
    else:
        plt.show()


def _time_to_fraction(magnitude, times, fraction) -> float:
    """Return the time (ms) at which the normalized magnitude first drops below fraction."""

    norm = magnitude / magnitude.max()
    below = np.nonzero(norm < fraction)[0]
    return float(times[below[0]] * 1e3) if below.size else float(times[-1] * 1e3)


if __name__ == "__main__":
    main()
