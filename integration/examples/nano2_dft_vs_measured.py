"""NaNO2 14N: ab initio prediction vs measured database lines.

Demonstrates the full cross-project loop on sodium nitrite, whose 14N
parameters exist on all three sides:

- DFT: a QuadrupolarDFT ABINIT EFG run (ICSD 82857 structure);
- simulation: spin_dynamics diagonalizes the resulting quadrupolar site;
- measurement: the NQRDatabase export holds the literature 14N lines.

Run:
    python integration/examples/nano2_dft_vs_measured.py
"""

from __future__ import annotations

from mr_integration import compare_dft_to_measured, predicted_lines

# ---- ab initio parameters (QuadrupolarDFT, nano2_icsd82857_efg run) ----------
# Mean over the two equivalent 14N atoms in results/nano2_efg_results.md.
DFT_CQ_HZ = -5.034045e6
DFT_ETA = 0.111906

# ---- literature parameters (also recorded in the database) -------------------
LIT_CQ_HZ = 5.497e6  # QCC from the database site record (kHz -> Hz)
LIT_ETA = 0.378


def main() -> None:
    print("=" * 64)
    print("Cross-implementation self-consistency (simulator vs DFT module)")
    print("=" * 64)
    pl = predicted_lines(cq_hz=DFT_CQ_HZ, eta=DFT_ETA, spin=1.0, isotope="14N")
    print(f"  nu_Q = {pl.nu_q_hz / 1e6:.6f} MHz (from C_Q = {DFT_CQ_HZ / 1e6:+.4f} MHz)")
    print(f"  simulator lines (MHz): {[round(float(x) / 1e6, 6) for x in pl.simulator_hz]}")
    print(f"  DFT module lines (MHz): {[round(float(x) / 1e6, 6) for x in pl.dft_hz]}")
    print(f"  max discrepancy: {pl.max_abs_discrepancy_hz:.3e} Hz")
    print(f"  self-consistent (< 1 Hz): {pl.self_consistent()}")
    print()

    print("=" * 64)
    print("Ab initio prediction vs measured database lines")
    print("=" * 64)
    print("[A] using the DFT (ICSD 82857) parameters")
    report_dft = compare_dft_to_measured(
        compound="Sodium Nitrite",
        cq_hz=DFT_CQ_HZ,
        eta=DFT_ETA,
        spin=1.0,
        isotope="14N",
    )
    print(report_dft.format_table())
    print()

    print("[B] using the literature C_Q/eta (sanity check: should match closely)")
    report_lit = compare_dft_to_measured(
        compound="Sodium Nitrite",
        cq_hz=LIT_CQ_HZ,
        eta=LIT_ETA,
        spin=1.0,
        isotope="14N",
    )
    print(report_lit.format_table())


if __name__ == "__main__":
    main()
