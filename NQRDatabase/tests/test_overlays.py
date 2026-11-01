"""The shipped sqlite must carry the overlay tables whose JSONL is shipped."""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from validate_database import OVERLAY_TABLES, validate_overlays  # noqa: E402

DATABASE = PROJECT / "data" / "exports" / "nqr.sqlite"
NORMALIZED = PROJECT / "data" / "normalized"


class OverlayExportTests(unittest.TestCase):
    def test_shipped_sqlite_matches_overlay_jsonl(self) -> None:
        # Regression: site_consistency_flags.jsonl (61 rows) and
        # landolt_consistency_flags.jsonl (26 rows) were shipped while the
        # sqlite export had neither table, so the explorer showed no badges.
        counts = validate_overlays(DATABASE, NORMALIZED)
        self.assertEqual(
            counts, {"site_consistency_flags": 61, "landolt_consistency_flags": 26}
        )

    def test_missing_overlay_table_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "nqr.sqlite"
            shutil.copyfile(DATABASE, copy)
            with sqlite3.connect(copy) as conn:
                for table in OVERLAY_TABLES:
                    conn.execute(f"DROP TABLE {table}")
            with self.assertRaisesRegex(AssertionError, "lacks table"):
                validate_overlays(copy, NORMALIZED)


if __name__ == "__main__":
    unittest.main()
