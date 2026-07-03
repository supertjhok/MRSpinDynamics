r"""Zero-field NQR of higher-spin quadrupolar nuclei (spin 5/2, 7/2, 9/2).

Most quadrupolar nuclei people want to simulate have spin > 3/2 -- ``27Al`` (5/2),
``51V`` (7/2), ``93Nb`` (9/2), and many more. Their zero-field level structure is a
ladder of Kramers doublets, so a half-integer spin ``I`` shows ``I - 1/2`` NQR
lines: for an axial EFG they fall at ``nu_Q, 2 nu_Q, 3 nu_Q, ...`` (the ``1/2<->3/2``,
``3/2<->5/2``, ... transitions), and a non-zero asymmetry ``eta`` shifts them and
switches on a weak combination line. This example builds real nuclei from the
isotope presets and shows:

1. The energy-level ladder for I = 1, 3/2, 5/2, 7/2 (each level a Kramers doublet
   for half-integer spin).
2. The transition ladder for 27Al (5/2) and 51V (7/2): stems at nu_Q, 2 nu_Q,
   3 nu_Q with height set by the transverse matrix element (the fundamental line
   is the strongest -- why it is the natural detection line).
3. A pulsed full-density-matrix FID on the 27Al fundamental line at two carrier
   detunings, demodulated to baseband: the beat frequency equals the detuning.
4. The eta-dependence of the fundamental line for 5/2 and 7/2, plus a spin-echo on
   the 27Al fundamental to show the pulsed dynamics run for higher spin.

The single-carrier full model is faithful on the fundamental (lowest) line, which
is what this example drives; higher/satellite lines are shown analytically (level
and transition structure) but not pulsed. Run with ``--output figure.png`` to save.
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
            "Zero-field NQR of higher-spin quadrupolar nuclei: level ladder, "
            "transition ratios, a pulsed FID on the fundamental line, and the "
            "eta-dependence of the fundamental line."
        )
    )
    parser.add_argument("--nu-q-mhz", type=float, default=1.0,
                        help="Fundamental NQR line nu_Q of the pulsed nucleus (MHz).")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Asymmetry parameter of the pulsed 27Al site.")
    parser.add_argument("--nutation-khz", type=float, default=10.0)
    parser.add_argument("--output", type=Path, help="Optional output PNG path.")
    return parser.parse_args()


def _levels_hz(spin: float, nu_q_hz: float, eta: float = 0.0) -> np.ndarray:
    from spin_dynamics.nqr import QuadrupolarSite, diagonalize_site

    site = QuadrupolarSite(spin=spin, quadrupole_frequency_hz=nu_q_hz, eta=eta)
    return diagonalize_site(site).levels_hz


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.nqr import (
        diagonalize_site,
        quadrupolar_site,
        simulate_full_echo,
        simulate_full_fid,
    )

    nu_q = args.nu_q_mhz * 1e6
    nutation = args.nutation_khz * 1e3

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)

    # (a) Energy-level ladder for several spins (axial, eta = 0).
    spins = [1.0, 1.5, 2.5, 3.5]
    for i, spin in enumerate(spins):
        levels = np.unique(np.round(_levels_hz(spin, nu_q) / 1e6, 6))
        for e in levels:
            ax[0, 0].hlines(e, i - 0.32, i + 0.32, color=f"C{i}", lw=2.2)
        degen = "doublets" if round(2 * spin) % 2 == 1 else "levels"
        ax[0, 0].text(i, levels.max() + 0.35 * nu_q / 1e6,
                      f"I={spin:g}\n({len(levels)} {degen})", ha="center", fontsize=8)
    ax[0, 0].set_xticks(range(len(spins)))
    ax[0, 0].set_xticklabels([f"{s:g}" for s in spins])
    ax[0, 0].set_xlabel("nuclear spin I")
    ax[0, 0].set_ylabel("energy / h (MHz)")
    ax[0, 0].set_title("Zero-field level ladder (axial)")

    # (b) Transition ladder for 27Al (5/2) and 51V (7/2): strength-weighted stems.
    for col, iso, spin in (("C2", "27Al", 2.5), ("C3", "51V", 3.5)):
        site = quadrupolar_site(iso, nu_q_hz=nu_q, eta=0.0)
        trans = diagonalize_site(site).transitions
        # collapse the degenerate partners: keep the strongest per distinct line
        by_line: dict[int, float] = {}
        for t in trans:
            key = int(round(t.frequency_hz / (0.01 * nu_q)))
            by_line[key] = max(by_line.get(key, 0.0), t.strength)
        for key, strength in by_line.items():
            freq = key * 0.01 * nu_q
            ax[0, 1].stem([freq / nu_q], [strength], linefmt=col, markerfmt=col + "o",
                          basefmt=" ")
        ax[0, 1].plot([], [], col, label=f"{iso} (I={spin:g})")
    ax[0, 1].set_xlabel(r"transition frequency / $\nu_Q$")
    ax[0, 1].set_ylabel("transverse matrix element")
    ax[0, 1].set_title(r"Transition ladder ($\nu_Q$, $2\nu_Q$, $3\nu_Q$)")
    ax[0, 1].set_xticks([1, 2, 3])
    ax[0, 1].legend(fontsize=8)

    # (c) Pulsed FID on the 27Al fundamental line at two carrier detunings.
    al = quadrupolar_site("27Al", nu_q_hz=nu_q, eta=args.eta)
    nu1 = min(t.frequency_hz for t in diagonalize_site(al).transitions)
    dur = np.deg2rad(90.0) / (2.0 * np.pi * nutation)
    times = np.linspace(0.0, 400e-6, 2000)
    for detuning, col in ((20e3, "C0"), (50e3, "C4")):
        fid = simulate_full_fid(al, nutation_hz=nutation, pulse_duration_seconds=dur,
                                times_seconds=times, rf_frequency_hz=nu1 + detuning)
        ax[1, 0].plot(times * 1e6, np.real(fid.signal), col, lw=0.9,
                      label=f"detuning {detuning/1e3:.0f} kHz")
    ax[1, 0].set_xlabel("time (us)")
    ax[1, 0].set_ylabel("baseband signal (Re)")
    ax[1, 0].set_title(f"27Al (5/2) FID on the fundamental (eta={args.eta:g})")
    ax[1, 0].legend(fontsize=8)

    # (d) eta-dependence of the fundamental line + a fundamental-line spin echo.
    etas = np.linspace(0.0, 1.0, 21)
    for col, spin, name in (("C2", 2.5, "27Al (5/2)"), ("C3", 3.5, "51V (7/2)")):
        fundamentals = [
            min(t.frequency_hz for t in diagonalize_site(
                quadrupolar_site("27Al" if spin == 2.5 else "51V", nu_q_hz=nu_q, eta=float(e))
            ).transitions) / nu_q
            for e in etas
        ]
        ax[1, 1].plot(etas, fundamentals, col, label=name)
    ax[1, 1].set_xlabel(r"asymmetry $\eta$")
    ax[1, 1].set_ylabel(r"fundamental line / $\nu_Q$")
    ax[1, 1].set_title("Fundamental line vs. asymmetry")
    ax[1, 1].legend(fontsize=8, loc="upper left")

    echo_times = np.linspace(0.0, 300e-6, 400)
    echo = simulate_full_echo(al, nutation_hz=nutation, excitation_duration_seconds=dur,
                              refocus_duration_seconds=2 * dur, echo_spacing_seconds=300e-6,
                              times_seconds=echo_times, rf_frequency_hz=nu1)
    axins = ax[1, 1].inset_axes([0.55, 0.12, 0.42, 0.4])
    axins.plot(echo_times * 1e6, np.abs(echo.signal), "k", lw=0.8)
    axins.set_title("27Al echo |s|", fontsize=7)
    axins.tick_params(labelsize=6)

    fig.suptitle("Higher-spin quadrupolar NQR: level ladder, transitions, and pulsed dynamics")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
