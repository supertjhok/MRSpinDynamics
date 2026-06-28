"""Validate the whole NQR database against the spin-dynamics simulator.

For every site that stores both quadrupolar parameters (qcc, eta) and measured
lines, simulate the lines implied by the parameters and compare. Sites whose
stored parameters and lines disagree are flagged as likely transcription/OCR
errors (or mismatched sources). For spin-1 sites the parameters implied by the
lines are shown alongside the stored ones, localizing the inconsistent field.

Run:
    python integration/examples/database_consistency.py
"""

from __future__ import annotations

from mr_integration import summarize, validate_database


def main() -> None:
    reports = validate_database()
    print(summarize(reports, worst=12))
    print()

    # Detail on the single worst site, with per-line residuals.
    if reports and reports[0].flagged():
        worst = reports[0]
        print(f"Worst site: {worst.site.compound}")
        print(
            f"  isotope={worst.site.isotope} spin={worst.spin:g}  "
            f"stored qcc={worst.site.qcc_hz / 1e6:.4f} MHz eta={worst.site.eta:.3f}"
        )
        if worst.implied_qcc_hz is not None:
            print(
                f"  line-implied qcc={worst.implied_qcc_hz / 1e6:.4f} MHz "
                f"eta={worst.implied_eta:.3f}  "
                f"(eta off by {worst.eta_error:+.3f})"
            )
        print("  measured(MHz)  predicted(MHz)   diff(kHz)")
        for measured, predicted, diff in worst.matches:
            print(
                f"  {measured / 1e6:11.4f}  {predicted / 1e6:9.4f}   {diff / 1e3:+8.1f}"
            )


if __name__ == "__main__":
    main()
