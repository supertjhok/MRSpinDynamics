"""Database-backed pipeline tests (skip if the SQLite export is absent)."""

from __future__ import annotations

import unittest

from mr_integration import compare_dft_to_measured, measured_lines
from mr_integration.database import default_database_path


_HAS_DB = default_database_path().exists()


@unittest.skipUnless(_HAS_DB, "NQR SQLite export not available")
class DatabasePipelineTests(unittest.TestCase):
    def test_nano2_measured_lines_present(self) -> None:
        lines = measured_lines("Sodium Nitrite", isotope="14N")
        self.assertTrue(lines)
        freqs_mhz = sorted(round(line.frequency_hz / 1e6, 3) for line in lines)
        # Literature NaNO2 14N lines: ~1.038, 3.604, 4.642 MHz.
        self.assertIn(1.038, freqs_mhz)
        self.assertIn(3.604, freqs_mhz)
        self.assertIn(4.642, freqs_mhz)

    def test_match_by_formula(self) -> None:
        lines = measured_lines("NaNO2", isotope="14N")
        self.assertTrue(lines)

    def test_literature_params_reproduce_measured_lines(self) -> None:
        # Feeding the literature C_Q/eta through the simulator should land
        # within a few kHz of the measured lines (rounding of the stored QCC).
        report = compare_dft_to_measured(
            compound="Sodium Nitrite",
            cq_hz=5.497e6,
            eta=0.378,
            spin=1.0,
            isotope="14N",
        )
        self.assertEqual(len(report.measured), 3)
        self.assertLess(report.max_abs_difference_hz, 15e3)

    def test_dft_params_are_worse_than_literature(self) -> None:
        # The starter DFT geometry underestimates eta, so its RMS error should
        # exceed the literature-parameter RMS error. This guards the whole loop.
        dft = compare_dft_to_measured(
            compound="Sodium Nitrite",
            cq_hz=5.034045e6,
            eta=0.111906,
            spin=1.0,
            isotope="14N",
        )
        lit = compare_dft_to_measured(
            compound="Sodium Nitrite",
            cq_hz=5.497e6,
            eta=0.378,
            spin=1.0,
            isotope="14N",
        )
        self.assertGreater(dft.rms_difference_hz, lit.rms_difference_hz)


if __name__ == "__main__":
    unittest.main()
