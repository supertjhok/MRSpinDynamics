"""Tests for the database self-consistency validator."""

from __future__ import annotations

import unittest

import numpy as np

from mr_integration import (
    SiteRecord,
    check_site,
    spin1_parameters_from_lines,
    summarize,
    validate_database,
)
from mr_integration.conversions import quadrupolar_site_from_cq
from mr_integration.cross_validation import _unique_within
from mr_integration.database import default_database_path
from spin_dynamics.nqr import diagonalize_site


def _consistent_spin1_record(qcc_hz: float, eta: float) -> SiteRecord:
    """Build a record whose stored lines exactly match its parameters."""

    site = quadrupolar_site_from_cq(cq_hz=qcc_hz, eta=eta, spin=1.0, isotope="14N")
    lines = _unique_within(
        np.asarray([t.frequency_hz for t in diagonalize_site(site).transitions]), 1.0
    )
    return SiteRecord(
        site_id="s1",
        compound="Synthetic",
        isotope="14N",
        site_label=None,
        qcc_hz=qcc_hz,
        eta=eta,
        temperature_k=None,
        measured_hz=tuple(float(x) for x in lines),
    )


class InversionTests(unittest.TestCase):
    def test_spin1_inversion_roundtrips(self) -> None:
        record = _consistent_spin1_record(3.5e6, 0.42)
        qcc, eta = spin1_parameters_from_lines(record.measured_hz)
        self.assertAlmostEqual(qcc, 3.5e6, delta=1.0)
        self.assertAlmostEqual(eta, 0.42, places=6)

    def test_inversion_requires_three_lines(self) -> None:
        with self.assertRaises(ValueError):
            spin1_parameters_from_lines([1.0e6, 2.0e6])


class CheckSiteTests(unittest.TestCase):
    def test_consistent_site_not_flagged(self) -> None:
        report = check_site(_consistent_spin1_record(3.5e6, 0.42))
        self.assertIsNotNone(report)
        self.assertLess(report.max_abs_diff_hz, 1.0)
        self.assertFalse(report.flagged())

    def test_wrong_eta_is_flagged_and_localized(self) -> None:
        # Lines consistent with eta=0.366 but stored eta=0.55 (Ampicillin-like).
        good = _consistent_spin1_record(3.533e6, 0.366)
        bad = SiteRecord(
            site_id="s2",
            compound="Bad eta",
            isotope="14N",
            site_label=None,
            qcc_hz=3.533e6,
            eta=0.55,
            temperature_k=None,
            measured_hz=good.measured_hz,
        )
        report = check_site(bad)
        self.assertTrue(report.flagged())
        # The implied parameters recover the true eta, localizing the error.
        self.assertAlmostEqual(report.implied_eta, 0.366, places=3)
        self.assertGreater(abs(report.eta_error), 0.1)
        self.assertLess(abs(report.qcc_error_hz), 1.0e3)

    def test_unsupported_isotope_returns_none(self) -> None:
        record = SiteRecord(
            site_id="s3",
            compound="Unsupported",
            isotope="127I",
            site_label=None,
            qcc_hz=1.0e6,
            eta=0.1,
            temperature_k=None,
            measured_hz=(1.0e6,),
        )
        self.assertIsNone(check_site(record))


@unittest.skipUnless(default_database_path().exists(), "NQR SQLite export absent")
class RealDatabaseTests(unittest.TestCase):
    def test_majority_of_sites_consistent(self) -> None:
        reports = validate_database(isotope="14N")
        self.assertGreater(len(reports), 20)
        consistent = sum(1 for r in reports if not r.flagged())
        # The curated 14N data should be overwhelmingly self-consistent.
        self.assertGreaterEqual(consistent / len(reports), 0.8)

    def test_summary_is_nonempty_string(self) -> None:
        reports = validate_database(isotope="14N")
        text = summarize(reports)
        self.assertIn("Checked", text)


if __name__ == "__main__":
    unittest.main()
