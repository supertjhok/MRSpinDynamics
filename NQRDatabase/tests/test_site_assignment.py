"""Regression tests for isotope assignment and export consistency."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from build_cwru_database import isotope_from_form  # noqa: E402

NORMALIZED = PROJECT / "data" / "normalized"

# The largest 14N quadrupole couplings are ~6-7 MHz, so a 14N line cannot sit
# above ~1.05 * C_Q (nu+ at eta = 1); 6.5 MHz is a generous physical ceiling.
MAX_14N_LINE_KHZ = 6500.0


def _load(name: str) -> list[dict]:
    with (NORMALIZED / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class FormIsotopeTests(unittest.TestCase):
    def test_isotope_from_form(self) -> None:
        self.assertEqual(isotope_from_form("Chlorine line"), "35Cl")
        self.assertEqual(isotope_from_form("37Cl line"), "37Cl")
        self.assertIsNone(isotope_from_form("phase 1"))
        self.assertIsNone(isotope_from_form("A"))
        self.assertIsNone(isotope_from_form(None))

    def test_no_14n_line_above_physical_ceiling(self) -> None:
        # Regression: furosemide's 36.226 MHz "Chlorine line" was stored on a
        # 14N site because extract_isotope defaults labelled sites to 14N.
        sites = {row["id"]: row for row in _load("sites.jsonl")}
        offenders = [
            (line["id"], line["frequency_khz"])
            for line in _load("lines.jsonl")
            if sites[line["site_id"]]["isotope"] == "14N"
            and line["frequency_khz"] is not None
            and line["frequency_khz"] > MAX_14N_LINE_KHZ
        ]
        self.assertEqual(offenders, [])

    def test_form_named_nucleus_matches_site_isotope(self) -> None:
        sites = {row["id"]: row for row in _load("sites.jsonl")}
        checked = 0
        for line in _load("lines.jsonl"):
            isotope = isotope_from_form(line["form"])
            if isotope is None:
                continue
            checked += 1
            self.assertEqual(sites[line["site_id"]]["isotope"], isotope, line["id"])
        self.assertGreaterEqual(checked, 1)

    def test_line_records_mirror_line_site(self) -> None:
        sites = {row["id"]: row for row in _load("sites.jsonl")}
        lines = {row["id"]: row for row in _load("lines.jsonl")}
        records = _load("line_records.jsonl")
        self.assertEqual({row["line_id"] for row in records}, set(lines))
        for record in records:
            site = sites[lines[record["line_id"]]["site_id"]]
            self.assertEqual(record["site"]["id"], site["id"])
            self.assertEqual(record["site"]["isotope"], site["isotope"])


if __name__ == "__main__":
    unittest.main()
