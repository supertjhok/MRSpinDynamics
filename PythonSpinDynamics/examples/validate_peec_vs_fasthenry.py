"""Validate the PEEC coil solver against FastHenry (the reference partial-element solver).

Runs the SAME geometries through the package's PEEC solver
(``fields.coil_peec.extract_impedance``) and, if a Windows FastHenry2 install with COM
automation is present, through FastHenry itself (``fields.fasthenry_interop``), and prints
the per-frequency L and R side by side. Three studies:

1. a straight square bar,
2. a helical solenoid (both square cross-section -- exact geometry match to FastHenry),
   comparing BOTH formulations: ``chain`` (reduced, skin only) and ``full`` (per-segment,
   FastHenry's own system, resolves turn-to-turn proximity), and
3. the AC-resistance mesh-convergence study.

Without FastHenry/pywin32 the script still prints the PEEC results and the reference
numbers recorded from a prior FastHenry run, so the comparison is visible everywhere.

Mesh-matching note: FastHenry silently grades its ``nwinc x nhinc`` filaments toward the
surface with adjacent-size ratio 2 unless told otherwise (its ``readGeom.c`` defaults
``rw = rh = 2.0``); all matched comparisons here pass ``rw = rh = 1.0`` so both solvers use
the identical uniform tiling.

Expected agreement (uniform-vs-uniform): inductance within ~1% for both geometries (the
segment-pair mutual uses the exact closed form ported from FastHenry's ``mutualfil``);
straight-bar resistance within ~2% at every frequency; solenoid ``full`` resistance within
a few % (the residual is the near-field kernel: FastHenry integrates exact rectangular
cross-sections for close parallel filament pairs, this solver uses curved thin filaments
with GMD self-terms). For *tightly wound* coils FastHenry itself under-predicts the
proximity loss against a continuum-FEM reference -- see the proximity note printed by the
script and docs/coil_peec.md.
"""

from __future__ import annotations

import argparse

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from scipy.special import bei, beip, ber, berp

from spin_dynamics.fields.coil_peec import Conductor, extract_impedance
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER
from spin_dynamics.fields.magnetostatics import MU0


def _kelvin_rac_over_rdc(a, delta):
    """Exact AC/DC resistance ratio of an isolated round wire (Kelvin functions)."""
    q = np.sqrt(2.0) * a / delta
    return (q / 2.0) * (ber(q) * beip(q) - bei(q) * berp(q)) / (berp(q) ** 2 + beip(q) ** 2)

# Reference FastHenry results recorded on Windows (FastHenry2 via COM) for the straight bar
# below (1x1 mm square, 100 mm long, copper, nwinc=nhinc=8, rw=rh=1 i.e. uniform filaments
# matching the PEEC tiling).
_BAR_FH_REF = {
    "frequency": [1e4, 1e5, 1e6, 1e7],
    "inductance_nH": [102.090, 100.585, 98.219, 97.886],
    "resistance_mOhm": [1.7441, 2.8228, 6.5847, 8.2802],
}

# Recorded FastHenry results for the 6-turn solenoid below (uniform 3x3 filaments).
_SOL_FH_REF = {
    "frequency": [1e5, 1e6],
    "inductance_nH": [401.71, 396.09],
    "resistance_mOhm": [9.102, 14.379],
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

    # --- 1. Straight square bar (exact geometry match; uniform filaments both sides) ---
    print("Straight square bar: 1x1 mm, 100 mm, copper, 8x8 uniform filaments")
    freqs = np.array(_BAR_FH_REF["frequency"])
    bar = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                    cross_section="rect", width=1e-3, height=1e-3, n_width=8, n_height=8)
    peec_bar = extract_impedance(bar, freqs)
    fh = _try_fasthenry(bar, freqs, rw=1.0, rh=1.0)
    if fh is not None:
        _print_table(freqs, peec_bar, fh.inductance * 1e9, fh.resistance * 1e3)
    else:
        print("  Using recorded FastHenry reference numbers:")
        _print_table(freqs, peec_bar, _BAR_FH_REF["inductance_nH"], _BAR_FH_REF["resistance_mOhm"])

    # --- 1b. Capacitance vs FasterCap (isolated straight wire) ---
    print("\nStraight wire self-capacitance vs FasterCap")
    from spin_dynamics.fields.coil_peec import capacitance_to_ground
    wire = Conductor(np.column_stack([np.zeros(200), np.zeros(200), np.linspace(0, 1.0, 200)]),
                     wire_radius=1e-3, material=ANNEALED_COPPER)
    c_peec = capacitance_to_ground(wire)
    try:
        from spin_dynamics.fields.fastercap_interop import run_fastercap

        c_fc = run_fastercap(wire, n_theta=24)
        print("  1 m, 1 mm wire:  C_PEEC=%.3f pF  C_FasterCap=%.3f pF  (%.1f%%)"
              % (c_peec * 1e12, c_fc * 1e12, 100 * (c_peec - c_fc) / c_fc))
    except Exception as exc:  # pragma: no cover
        print(f"  C_PEEC={c_peec * 1e12:.3f} pF  (FasterCap not run: {exc})")

    # --- 2. Helical solenoid, square wire (exact geometry match to FastHenry) ---
    # Both formulations: `chain` reduces the sub-filaments to path-constant currents (skin
    # only), `full` keeps one branch per (segment, sub-filament) -- FastHenry's own system --
    # so the turn-to-turn proximity redistribution is resolved.
    print(f"\nHelical solenoid: D=20 mm, l=30 mm, {args.turns} turns, 1 mm square wire, "
          "uniform 3x3 filaments")
    sol_freqs = np.array([1e5, 1e6])
    diam, length, turns, side = 20e-3, 30e-3, args.turns, 1e-3
    n_per = 12
    th = np.linspace(0, 2 * np.pi * turns, turns * n_per + 1)
    z = np.linspace(-length / 2, length / 2, th.size)
    path = np.column_stack([(diam / 2) * np.cos(th), (diam / 2) * np.sin(th), z])
    sol = Conductor(path, material=ANNEALED_COPPER, cross_section="rect",
                    width=side, height=side, n_width=3, n_height=3)
    peec_chain = extract_impedance(sol, sol_freqs)
    peec_full = extract_impedance(sol, sol_freqs, formulation="full")
    fh_sol = _try_fasthenry(sol, sol_freqs, nwinc=3, nhinc=3, rw=1.0, rh=1.0)
    if fh_sol is not None:
        fh_L, fh_R = fh_sol.inductance * 1e9, fh_sol.resistance * 1e3
    else:
        print("  Using recorded FastHenry reference numbers:")
        fh_L, fh_R = _SOL_FH_REF["inductance_nH"], _SOL_FH_REF["resistance_mOhm"]
    print("  chain formulation (reduced; misses part of the proximity loss):")
    _print_table(sol_freqs, peec_chain, fh_L, fh_R)
    print("  full formulation (per-segment, FastHenry's system):")
    _print_table(sol_freqs, peec_full, fh_L, fh_R)

    # --- 2b. Tight-coil proximity: where the references themselves disagree ---
    # For a CLOSELY-wound coil (pitch 1.5 mm, wire 1 mm) the proximity factor
    # Phi = R_coil / R_straight_wire from different references at 0.5 / 2 MHz:
    #   FEMM 4.2 (axisymmetric continuum FEM, 10-ring stack): 1.720 / 1.803
    #   this solver, volume-full (round wire):                1.718 /  --  (0.1% vs FEMM)
    #   this solver, SIBC-full:                               1.54  / 1.69 (-6% at deep skin)
    #   FastHenry (graded 8x8, its converged mesh):           1.43  / 1.47 (~-20%)
    #   Medhurst's measured table (HF asymptote):                2.27      (~+25%)
    # The full-PEEC volume solve matches the continuum FEM; FastHenry under-predicts
    # tight-coil proximity and Medhurst's empirical table reads high vs field solvers
    # (his measured Q included real-coil losses beyond the eddy-current proximity).
    print("\nTight-coil proximity factor (see docs/coil_peec.md 'Loss modelling'):")
    print("  FEMM continuum reference 1.720 @0.5 MHz; PEEC volume-full 1.718;")
    print("  FastHenry ~1.43 (under); Medhurst table 2.27 (over).")

    # --- 3. AC-resistance convergence vs filament count (deep skin) ---
    # Why does R "diverge" between PEEC and FastHenry at high frequency? Both are
    # partial-element solvers: the cross-section must be tiled finer than the skin depth,
    # and BOTH under-resolve at a coarse mesh -- they converge to the same (exact Kelvin)
    # value as the filament count rises. Inductance, by contrast, is already converged.
    r_freq = 1e6
    a_eq = np.sqrt(1e-3 * 1e-3 / np.pi)  # round wire of equal area, for the Kelvin anchor
    delta = np.sqrt(2 * ANNEALED_COPPER.resistivity / (2 * np.pi * r_freq * MU0))
    r_dc = ANNEALED_COPPER.resistivity * 0.1 / (np.pi * a_eq**2)
    kelvin = _kelvin_rac_over_rdc(a_eq, delta) * r_dc
    print(f"\nAC-resistance convergence, 1x1 mm bar at {r_freq:.0e} Hz "
          f"(skin depth {delta * 1e6:.0f} um, side/delta {1e-3 / delta:.0f} -- deep skin)")
    print(f"  Kelvin exact (round-equivalent) ~ {kelvin * 1e3:.2f} mOhm")
    print("  %8s %12s %12s %10s" % ("n x n", "R_PEEC mO", "R_FH mO", "L_PEEC nH"))
    for n in (4, 8, 16):
        bar_n = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                          cross_section="rect", width=1e-3, height=1e-3, n_width=n, n_height=n)
        imp_n = extract_impedance(bar_n, [r_freq])
        fh_n = _try_fasthenry(bar_n, [r_freq], nwinc=n, nhinc=n, rw=1.0, rh=1.0)
        r_fh = f"{fh_n.resistance[0] * 1e3:12.3f}" if fh_n is not None else f"{'--':>12}"
        print("  %8s %12.3f %s %10.2f"
              % (f"{n}x{n}", imp_n.resistance[0] * 1e3, r_fh, imp_n.inductance[0] * 1e9))
    print("  -> both under-resolve at coarse mesh and climb toward Kelvin; L stays put.")


if __name__ == "__main__":
    main()
