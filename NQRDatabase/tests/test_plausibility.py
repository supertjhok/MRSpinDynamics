"""Unit-plausibility checks in scripts/validate_database.py."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from validate_database import validate_plausibility  # noqa: E402

NORMALIZED = PROJECT / "data" / "normalized"


def _rewrite_line(directory: Path, line_id_fragment: str, **changes) -> None:
    path = directory / "lines.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    matched = 0
    for row in rows:
        if line_id_fragment in row["id"]:
            row.update(changes)
            matched += 1
    assert matched >= 1, line_id_fragment
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class PlausibilityTests(unittest.TestCase):
    def test_shipped_exports_pass(self) -> None:
        counts = validate_plausibility(NORMALIZED)
        self.assertGreater(counts["14n_lines"], 900)
        self.assertGreater(counts["t2_values"], 50)
        self.assertGreaterEqual(counts["form_isotopes"], 1)

    def _assert_rejected(self, fragment: str, message: str, **changes) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for name in ("sites.jsonl", "lines.jsonl"):
                shutil.copyfile(NORMALIZED / name, directory / name)
            _rewrite_line(directory, fragment, **changes)
            with self.assertRaisesRegex(AssertionError, message):
                validate_plausibility(directory)

    def test_rejects_previous_defects(self) -> None:
        # The three defects found in review, re-injected into a copy:
        # glycine T2 stored as 12.5 s (ms quoted as s) ...
        self._assert_rejected("glycine-nqr-database", "likely ms", t2_s=12.5)
        # ... a 36 MHz 35Cl line on a 14N site ...
        self._assert_rejected(
            "furosemide-lasix-prescription-default-1-1-cwru-html-furosemide-lasix-prescription-nqr-database-3563-6",
            "14N line at",
            frequency_khz=36226.0,
        )
        # ... and a Form that names a nucleus other than the site's.
        self._assert_rejected(
            "furosemide-lasix-prescription-default-1-1-cwru-html-furosemide-lasix-prescription-nqr-database-3563-6",
            "form names 35Cl",
            form="Chlorine line",
        )
        # T2 can never exceed 2 T1.
        self._assert_rejected(
            "sodium-nitrite-nqr-database", "exceeds 2 T1", t1_s=0.001, t2_s=0.0051
        )


if __name__ == "__main__":
    unittest.main()
