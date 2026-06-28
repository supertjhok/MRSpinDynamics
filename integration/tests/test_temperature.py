"""Tests for predicted-vs-measured temperature-coefficient comparison."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from mr_integration import (
    compare_temperature_coefficients,
    measured_temperature_coefficients,
    slopes_from_temperature_points,
)
from mr_integration.database import default_database_path


@dataclass
class _Point:
    temperature_k: float
    frequencies_hz: np.ndarray


class SlopesFromPointsTests(unittest.TestCase):
    def test_slopes_from_two_points(self):
        low = _Point(0.0, np.array([1.0e6, 3.0e6]))
        high = _Point(100.0, np.array([0.9e6, 2.8e6]))
        slopes = slopes_from_temperature_points([high, low])  # unordered input
        self.assertEqual(len(slopes), 2)
        # line at 1.0 MHz drops 0.1 MHz over 100 K -> -1000 Hz/K.
        self.assertAlmostEqual(slopes[0][0], 1.0e6)
        self.assertAlmostEqual(slopes[0][1], -1000.0)
        self.assertAlmostEqual(slopes[1][1], -2000.0)

    def test_requires_two_points(self):
        with self.assertRaises(ValueError):
            slopes_from_temperature_points([_Point(0.0, np.array([1.0e6]))])


class CompareTests(unittest.TestCase):
    def test_match_dataclass_difference_and_signs(self):
        from mr_integration.temperature import (
            MeasuredTemperatureCoefficient,
            TemperatureCoefficientComparison,
            TemperatureCoefficientMatch,
        )

        measured = MeasuredTemperatureCoefficient("X", "14N", 4.6e6, -2200.0, 293.0)
        match = TemperatureCoefficientMatch(measured, 4.62e6, -1600.0)
        comparison = TemperatureCoefficientComparison("X", "14N", (match,))
        self.assertAlmostEqual(match.difference_hz_per_k, 600.0)
        self.assertAlmostEqual(match.frequency_offset_hz, 2.0e4)
        self.assertTrue(comparison.signs_agree)
        self.assertIn("temperature coefficients", comparison.format_table())


@unittest.skipUnless(default_database_path().exists(), "NQR SQLite export absent")
class RealDatabaseTests(unittest.TestCase):
    def test_nano2_measured_coefficients(self):
        coeffs = measured_temperature_coefficients(
            "Nitrous acid sodium salt", isotope="14N", temperature_k=293.0
        )
        self.assertTrue(coeffs)
        # All NaNO2 14N coefficients are negative (Bayer behaviour).
        self.assertTrue(all(c.dnu_dt_hz_per_k < 0 for c in coeffs))

    def test_compare_signs_agree_with_negative_prediction(self):
        comparison = compare_temperature_coefficients(
            compound="Nitrous acid sodium salt",
            isotope="14N",
            temperature_k=293.0,
            predicted=[(3.6e6, -900.0), (4.65e6, -1600.0)],
        )
        self.assertTrue(comparison.matches)
        self.assertTrue(comparison.signs_agree)


if __name__ == "__main__":
    unittest.main()
