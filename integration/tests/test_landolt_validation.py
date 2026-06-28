"""Tests for Landolt entry-level consistency validation and review flagging."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path


from mr_integration import (
    LandoltSetRecord,
    check_landolt_set,
    parse_nucleus,
    write_landolt_review_flags,
)
from mr_integration.conversions import quadrupolar_site_from_cq
from mr_integration.database import default_database_path
from mr_integration.landolt_review_export import FLAG_TABLE, ISSUE_FLAG
from spin_dynamics.nqr import diagonalize_site


def _strong_lines(qcc_hz: float, eta: float) -> tuple[float, float]:
    site = quadrupolar_site_from_cq(cq_hz=qcc_hz, eta=eta, spin=1.0, isotope="14N")
    ascending = sorted(t.frequency_hz for t in diagonalize_site(site).transitions)
    return ascending[-2], ascending[-1]


class ParseNucleusTests(unittest.TestCase):
    def test_n14(self) -> None:
        self.assertEqual(parse_nucleus("N-14"), ("14N", 1.0))

    def test_cl35(self) -> None:
        self.assertEqual(parse_nucleus("Cl-35"), ("35Cl", 1.5))

    def test_unsupported_or_missing(self) -> None:
        self.assertIsNone(parse_nucleus(None))
        self.assertIsNone(parse_nucleus("O-17"))  # spin 5/2, unsupported
        self.assertIsNone(parse_nucleus("garbage"))


class CheckLandoltSetTests(unittest.TestCase):
    def _record(self, freqs_mhz, pairs) -> LandoltSetRecord:
        return LandoltSetRecord(
            measurement_set_id="ms1",
            entry_id="e1",
            nucleus="N-14",
            isotope="14N",
            spin=1.0,
            substance_name="Test",
            frequencies_hz=tuple(f * 1e6 for f in freqs_mhz),
            qcc_eta_pairs=tuple(pairs),
        )

    def test_consistent_set_not_flagged(self) -> None:
        nu_minus, nu_plus = _strong_lines(3.0e6, 0.3)
        record = self._record(
            [nu_plus / 1e6, nu_minus / 1e6], [(3.0, 0.3)]
        )
        report = check_landolt_set(record)
        self.assertLess(report.max_gap_hz, 1.0e3)
        self.assertFalse(report.flagged())

    def test_extra_measured_lines_are_harmless(self) -> None:
        # A stray temperature (77.0) in the list must not trigger a flag.
        nu_minus, nu_plus = _strong_lines(3.0e6, 0.3)
        record = self._record(
            [nu_plus / 1e6, nu_minus / 1e6, 77.0], [(3.0, 0.3)]
        )
        self.assertFalse(check_landolt_set(record).flagged())

    def test_bad_qcc_is_flagged(self) -> None:
        # QCC OCR error (313 instead of ~3.13): predicted lines are far off.
        record = self._record([2.68, 0.39], [(313.0, 0.235)])
        report = check_landolt_set(record)
        self.assertTrue(report.flagged())
        self.assertGreater(report.max_gap_hz, 1.0e6)

    def test_missing_eta_pair_ignored(self) -> None:
        # Pairs with no eta are dropped upstream; here a valid set still checks.
        nu_minus, nu_plus = _strong_lines(2.19e6, 0.64)
        record = self._record(
            [nu_plus / 1e6, nu_minus / 1e6], [(2.19, 0.64)]
        )
        self.assertFalse(check_landolt_set(record).flagged())


@unittest.skipUnless(default_database_path().exists(), "NQR SQLite export absent")
class LandoltReviewExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self.db = self._tmp / "nqr.sqlite"
        shutil.copy(default_database_path(), self.db)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_flags_routed_into_queue(self) -> None:
        summary = write_landolt_review_flags(
            database_path=self.db, jsonl_path=self._tmp / "f.jsonl"
        )
        self.assertGreater(summary.sets_checked, 50)
        self.assertGreater(summary.entries_flagged, 0)
        self.assertEqual(summary.queue_rows_updated, summary.entries_flagged)
        conn = sqlite3.connect(self.db)
        try:
            in_queue = conn.execute(
                "SELECT COUNT(*) FROM landolt_review_queue "
                "WHERE issue_flags_json LIKE ?",
                [f"%{ISSUE_FLAG}%"],
            ).fetchone()[0]
            in_table = conn.execute(
                f"SELECT COUNT(*) FROM {FLAG_TABLE}"
            ).fetchone()[0]
            bumped = conn.execute(
                "SELECT COUNT(*) FROM landolt_review_queue "
                "WHERE issue_flags_json LIKE ? AND priority = 1",
                [f"%{ISSUE_FLAG}%"],
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(in_queue, summary.entries_flagged)
        self.assertEqual(in_table, summary.entries_flagged)
        self.assertEqual(bumped, summary.entries_flagged)

    def test_rerun_idempotent_no_duplicate_flags(self) -> None:
        write_landolt_review_flags(database_path=self.db, write_jsonl=False)
        write_landolt_review_flags(database_path=self.db, write_jsonl=False)
        conn = sqlite3.connect(self.db)
        try:
            rows = [
                json.loads(r[0])
                for r in conn.execute(
                    "SELECT issue_flags_json FROM landolt_review_queue "
                    "WHERE issue_flags_json LIKE ?",
                    [f"%{ISSUE_FLAG}%"],
                )
            ]
        finally:
            conn.close()
        self.assertTrue(rows)
        self.assertEqual(max(flags.count(ISSUE_FLAG) for flags in rows), 1)

    def test_existing_issue_flags_preserved(self) -> None:
        # An unrelated pre-existing flag must survive our edit.
        conn = sqlite3.connect(self.db)
        try:
            entry = conn.execute(
                "SELECT q.id, q.entry_id FROM landolt_review_queue q "
                "JOIN landolt_compound_entries e ON e.id = q.entry_id "
                "WHERE e.nucleus = 'N-14' LIMIT 1"
            ).fetchone()
            conn.execute(
                "UPDATE landolt_review_queue SET issue_flags_json = ? WHERE id = ?",
                [json.dumps(["missing_cas"]), entry[0]],
            )
            conn.commit()
        finally:
            conn.close()
        write_landolt_review_flags(database_path=self.db, write_jsonl=False)
        conn = sqlite3.connect(self.db)
        try:
            flags = json.loads(
                conn.execute(
                    "SELECT issue_flags_json FROM landolt_review_queue WHERE id = ?",
                    [entry[0]],
                ).fetchone()[0]
            )
        finally:
            conn.close()
        self.assertIn("missing_cas", flags)


if __name__ == "__main__":
    unittest.main()
