"""Tests for writing the consistency-flag overlay into the database."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mr_integration import write_consistency_flags
from mr_integration.database import default_database_path
from mr_integration.database_validation import describe
from mr_integration.flag_export import FLAG_TABLE


@unittest.skipUnless(default_database_path().exists(), "NQR SQLite export absent")
class FlagExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self.db = self._tmp / "nqr.sqlite"
        shutil.copy(default_database_path(), self.db)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_writes_table_and_jsonl(self) -> None:
        jsonl = self._tmp / "flags.jsonl"
        summary = write_consistency_flags(database_path=self.db, jsonl_path=jsonl)
        self.assertGreater(summary.sites_written, 0)
        self.assertGreater(summary.flagged, 0)
        self.assertTrue(jsonl.exists())
        self.assertEqual(
            summary.sites_written, sum(1 for _ in jsonl.open(encoding="utf-8"))
        )

    def test_table_rows_match_summary(self) -> None:
        summary = write_consistency_flags(database_path=self.db, write_jsonl=False)
        conn = sqlite3.connect(self.db)
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {FLAG_TABLE}").fetchone()[0]
            flagged = conn.execute(
                f"SELECT COUNT(*) FROM {FLAG_TABLE} WHERE flagged = 1"
            ).fetchone()[0]
            detail = conn.execute(
                f"SELECT detail FROM {FLAG_TABLE} WHERE flagged = 1 LIMIT 1"
            ).fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(total, summary.sites_written)
        self.assertEqual(flagged, summary.flagged)
        self.assertTrue(detail)

    def test_rerun_is_idempotent(self) -> None:
        first = write_consistency_flags(database_path=self.db, write_jsonl=False)
        second = write_consistency_flags(database_path=self.db, write_jsonl=False)
        self.assertEqual(first.sites_written, second.sites_written)
        conn = sqlite3.connect(self.db)
        try:
            total = conn.execute(f"SELECT COUNT(*) FROM {FLAG_TABLE}").fetchone()[0]
        finally:
            conn.close()
        # A rerun replaces rather than appends.
        self.assertEqual(total, second.sites_written)


class DescribeTests(unittest.TestCase):
    def test_describe_verified_phrasing(self) -> None:
        from mr_integration.database import SiteRecord
        from mr_integration import check_site
        from mr_integration.conversions import quadrupolar_site_from_cq
        from mr_integration.cross_validation import _unique_within
        from spin_dynamics.nqr import diagonalize_site
        import numpy as np

        site = quadrupolar_site_from_cq(cq_hz=3.5e6, eta=0.4, spin=1.0, isotope="14N")
        lines = _unique_within(
            np.asarray([t.frequency_hz for t in diagonalize_site(site).transitions]),
            1.0,
        )
        record = SiteRecord(
            site_id="s1",
            compound="X",
            isotope="14N",
            site_label=None,
            qcc_hz=3.5e6,
            eta=0.4,
            temperature_k=None,
            measured_hz=tuple(float(x) for x in lines),
        )
        report = check_site(record)
        self.assertIn("verified", describe(report))


if __name__ == "__main__":
    unittest.main()
