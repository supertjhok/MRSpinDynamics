"""Propagate an illustrative DFT parameter uncertainty to NaNO2 NQR lines.

The standard deviations below are deliberately labelled as a sensitivity model,
not a calibrated ABINIT posterior. Replace them with convergence-, structure-,
or ensemble-derived uncertainties for a quantitative study. The example shows
how such uncertainties become line-frequency intervals and how measured lines
are scored for interval coverage.

Run:
    python integration/examples/nano2_uncertainty.py
"""

from __future__ import annotations

from mr_integration import (
    QuadrupolarParameterDistribution,
    compare_uncertain_dft_to_measured,
)


def main() -> None:
    sensitivity_model = QuadrupolarParameterDistribution(
        cq_mean_hz=5.034045e6,
        cq_std_hz=0.15e6,
        eta_mean=0.111906,
        eta_std=0.04,
        correlation=0.0,
    )
    report = compare_uncertain_dft_to_measured(
        compound="Sodium Nitrite",
        distribution=sensitivity_model,
        spin=1.0,
        isotope="14N",
        sample_count=1000,
        confidence=0.95,
        seed=2026,
    )
    print("Illustrative sensitivity model; parameter widths are not calibrated")
    print(report.format_table())
    print(f"Measured-line interval coverage: {report.coverage_fraction:.0%}")
    print(
        "Cross-implementation discrepancy: "
        f"{report.prediction.max_cross_implementation_discrepancy_hz:.3e} Hz"
    )


if __name__ == "__main__":
    main()
