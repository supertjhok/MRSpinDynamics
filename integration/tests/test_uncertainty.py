"""Tests for DFT-parameter uncertainty propagation."""

from __future__ import annotations

import unittest

import numpy as np

from mr_integration import (
    QuadrupolarParameterDistribution,
    compare_uncertain_dft_to_measured,
    predicted_lines,
    propagate_parameter_uncertainty,
)
from mr_integration.database import default_database_path

_HAS_DB = default_database_path().exists()


class ParameterUncertaintyTests(unittest.TestCase):
    def test_zero_width_distribution_reproduces_nominal_lines(self) -> None:
        distribution = QuadrupolarParameterDistribution(
            cq_mean_hz=5.497e6,
            cq_std_hz=0.0,
            eta_mean=0.378,
            eta_std=0.0,
        )
        result = propagate_parameter_uncertainty(
            distribution,
            spin=1.0,
            isotope="14N",
            sample_count=8,
        )
        nominal = predicted_lines(
            cq_hz=5.497e6,
            eta=0.378,
            spin=1.0,
            isotope="14N",
        )
        np.testing.assert_allclose(result.median_hz, nominal.simulator_hz, atol=1e-9)
        for interval in result.intervals:
            self.assertAlmostEqual(interval.lower_hz, interval.median_hz)
            self.assertAlmostEqual(interval.upper_hz, interval.median_hz)

    def test_nonzero_distribution_produces_ordered_intervals(self) -> None:
        distribution = QuadrupolarParameterDistribution(
            cq_mean_hz=5.0e6,
            cq_std_hz=0.1e6,
            eta_mean=0.25,
            eta_std=0.03,
            correlation=-0.4,
        )
        result = propagate_parameter_uncertainty(
            distribution,
            spin=1.0,
            isotope="14N",
            sample_count=200,
            confidence=0.9,
            seed=12,
        )
        self.assertEqual(len(result.intervals), 3)
        for interval in result.intervals:
            self.assertLess(interval.lower_hz, interval.median_hz)
            self.assertLess(interval.median_hz, interval.upper_hz)
        self.assertLess(result.max_cross_implementation_discrepancy_hz, 1.0)

    def test_invalid_distribution_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "eta_mean"):
            QuadrupolarParameterDistribution(5.0e6, 1.0e5, 1.1, 0.1)
        with self.assertRaisesRegex(ValueError, "standard deviations"):
            QuadrupolarParameterDistribution(5.0e6, -1.0, 0.2, 0.1)
        with self.assertRaisesRegex(ValueError, "correlation"):
            QuadrupolarParameterDistribution(5.0e6, 1.0, 0.2, 0.1, 1.1)

    @unittest.skipUnless(_HAS_DB, "NQR SQLite export not available")
    def test_measured_lines_can_be_scored_against_intervals(self) -> None:
        report = compare_uncertain_dft_to_measured(
            compound="Sodium Nitrite",
            distribution=QuadrupolarParameterDistribution(
                cq_mean_hz=5.497e6,
                cq_std_hz=20e3,
                eta_mean=0.378,
                eta_std=0.01,
            ),
            spin=1.0,
            isotope="14N",
            sample_count=300,
            seed=4,
        )
        self.assertEqual(len(report.matches), 3)
        self.assertEqual(report.coverage_fraction, 1.0)
        self.assertIn("prediction intervals", report.format_table())


if __name__ == "__main__":
    unittest.main()
