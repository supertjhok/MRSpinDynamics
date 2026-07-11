"""Survey every structure-backed cross-project NQR target.

The report joins the QuadrupolarDFT structure inventory and result summary with
the measured NQR database. It identifies comparisons that can run now, compounds
ready for a DFT calculation, and database records that first need isotope
metadata. Every comparison-ready DFT row is then passed through the simulator
and compared with measurement.

Run:
    python integration/examples/integration_target_survey.py
"""

from __future__ import annotations

from mr_integration import (
    compare_available_dft,
    format_target_survey,
    survey_integration_targets,
)


def main() -> None:
    targets = survey_integration_targets()
    print("Cross-project integration target survey")
    print(format_target_survey(targets))

    comparisons = compare_available_dft(targets)
    print(f"\nExecutable DFT comparisons: {len(comparisons)}")
    for comparison in comparisons:
        record = comparison.dft_record
        report = comparison.report
        print(
            f"  {record.case_id} {record.isotope}: "
            f"RMS={report.rms_difference_hz / 1e3:.1f} kHz, "
            f"max={report.max_abs_difference_hz / 1e3:.1f} kHz"
        )

    queued = [target for target in targets if target.status == "dft-needed"]
    print(f"\nStructure + isotope-tagged measurement, awaiting DFT: {len(queued)}")
    for target in queued:
        isotopes = ", ".join(
            f"{isotope} ({count} lines)"
            for isotope, count in target.isotope_line_counts
        )
        print(f"  {target.spec.structure_name}: {isotopes}")


if __name__ == "__main__":
    main()
