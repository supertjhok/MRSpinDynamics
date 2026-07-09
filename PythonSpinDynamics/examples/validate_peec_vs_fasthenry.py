"""Validate the PEEC coil solver against FastHenry (the reference partial-element solver).

Runs the SAME geometries through the package's PEEC solver
(``fields.coil_peec.extract_impedance``) and, if a Windows FastHenry2 install with COM
automation is present, through FastHenry itself (``fields.fasthenry_interop``), and prints
the per-frequency L and R side by side. Two geometries:

1. a straight square bar (exact cross-section match to FastHenry), and
2. a helical solenoid (round wire, exported as an equal-area square bar).

Without FastHenry/pywin32 the script still prints the PEEC results and the reference
numbers recorded from a prior FastHenry run, so the comparison is visible everywhere.

Expected agreement: R within ~1% at low frequency (both near the DC value), L within a few
percent (a GMD-discretization offset vs FastHenry's exact Hoer-Love formulas), and growing
divergence in the deep-skin regime where both solvers under-resolve unless the filament
count is raised.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.fields.coil_peec import Conductor, extract_impedance, helical_solenoid
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER

# Reference FastHenry results recorded on Windows (FastHenry2 via COM) for the straight bar
# below (1x1 mm square, 100 mm long, copper, nwinc=nhinc=8).
_BAR_FH_REF = {
    "frequency": [1e4, 1e5, 1e6, 1e7],
    "inductance_nH": [102.091, 100.597, 97.922, 97.076],
    "resistance_mOhm": [1.7441, 2.8974, 8.3070, 21.4900],
}


def _try_fasthenry(conductor, freqs, **kw):
    try:
        from spin_dynamics.fields.fasthenry_interop import run_fasthenry

        return run_fasthenry(conductor, freqs, **kw)
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"  (FastHenry not run: {exc})")
        return None


def _print_table(freqs, peec, fh_L_nH, fh_R_mO):
    print("  %8s %10s %10s %7s   %10s %10s %7s"
          % ("f (Hz)", "L_FH nH", "L_PEEC", "dL%", "R_FH mO", "R_PEEC", "dR%"))
    for i, f in enumerate(freqs):
        lp, rp = peec.inductance[i] * 1e9, peec.resistance[i] * 1e3
        lf, rf = fh_L_nH[i], fh_R_mO[i]
        dl = 100 * (lp - lf) / lf if lf else float("nan")
        dr = 100 * (rp - rf) / rf if rf else float("nan")
        print("  %8.0e %10.3f %10.3f %7.1f   %10.4f %10.4f %7.1f" % (f, lf, lp, dl, rf, rp, dr))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=6)
    args = parser.parse_args()

    # --- 1. Straight square bar (exact geometry match) ---
    print("Straight square bar: 1x1 mm, 100 mm, copper, 8x8 filaments")
    freqs = np.array(_BAR_FH_REF["frequency"])
    bar = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                    cross_section="rect", width=1e-3, height=1e-3, n_width=8, n_height=8)
    peec_bar = extract_impedance(bar, freqs)
    fh = _try_fasthenry(bar, freqs)
    if fh is not None:
        _print_table(freqs, peec_bar, fh.inductance * 1e9, fh.resistance * 1e3)
    else:
        print("  Using recorded FastHenry reference numbers:")
        _print_table(freqs, peec_bar, _BAR_FH_REF["inductance_nH"], _BAR_FH_REF["resistance_mOhm"])

    # --- 2. Helical solenoid (round wire -> equal-area square bar in FastHenry) ---
    print(f"\nHelical solenoid: D=20 mm, l=30 mm, {args.turns} turns, 1 mm wire")
    sol_freqs = np.array([1e5, 1e6])
    sol = helical_solenoid(diameter=20e-3, length=30e-3, turns=args.turns, wire_radius=0.5e-3,
                           material=ANNEALED_COPPER, n_per_turn=10, n_radial=3, n_angular=6)
    peec_sol = extract_impedance(sol, sol_freqs)
    fh_sol = _try_fasthenry(sol, sol_freqs, nwinc=4, nhinc=4)
    if fh_sol is not None:
        _print_table(sol_freqs, peec_sol, fh_sol.inductance * 1e9, fh_sol.resistance * 1e3)
    else:
        print("  (install FastHenry2 + pywin32 on Windows to run the solenoid comparison)")
        for i, f in enumerate(sol_freqs):
            print("  %8.0e  L_PEEC=%.3f uH  R_PEEC=%.4f ohm"
                  % (f, peec_sol.inductance[i] * 1e6, peec_sol.resistance[i]))


if __name__ == "__main__":
    main()
