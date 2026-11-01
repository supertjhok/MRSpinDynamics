"""Regression tests for relaxation-time unit handling in the NQR database."""

from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from build_cwru_database import (  # noqa: E402
    CWRU_UNITLESS_TIME_SCALE_S,
    first_number,
    parse_time_s,
)

NORMALIZED = PROJECT / "data" / "normalized"
DATABASE = PROJECT / "data" / "exports" / "nqr.sqlite"


def _load(name: str) -> list[dict]:
    with (NORMALIZED / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ParseTimeTests(unittest.TestCase):
    def test_explicit_units(self) -> None:
        cases = {
            "4.220 ± 1.148 s": 4.22,
            "ca 2 s": 2.0,
            "15 ms": 0.015,
            "<2 ms": 0.002,
            "250 us": 250e-6,
            "250 µs": 250e-6,  # micro sign
            "250 μs": 250e-6,  # Greek mu
            "2 hours": 7200.0,  # used to match "us" and be divided by 1e6
            "3 min": 180.0,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_time_s(text), expected, delta=1e-12)

    def test_unitless_value_uses_table_convention(self) -> None:
        self.assertEqual(parse_time_s("328"), 328.0)
        self.assertAlmostEqual(parse_time_s("328", CWRU_UNITLESS_TIME_SCALE_S), 0.328)
        # An explicit unit always wins over the table default.
        self.assertEqual(parse_time_s("4 s", CWRU_UNITLESS_TIME_SCALE_S), 4.0)
        self.assertIsNone(parse_time_s(""))
        self.assertIsNone(parse_time_s("several"))

    def test_scientific_notation_and_leading_dot(self) -> None:
        # Regression: the number pattern read ".5 ms" as 5 ms and "1.2e-3 s" as
        # 1.2 s (no leading-dot decimals, no exponents).
        cases = {
            ".5 ms": 0.5e-3,
            "1.2e-3 s": 1.2e-3,
            "2.5E+2 us": 2.5e-4,
            "3e1 ms": 0.03,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertAlmostEqual(parse_time_s(text), expected, delta=1e-15)
        self.assertAlmostEqual(parse_time_s("5e-3", CWRU_UNITLESS_TIME_SCALE_S), 5e-6)
        self.assertEqual(first_number("1,234.5 kHz"), 1234.5)
        self.assertEqual(first_number("1.5-2.0"), 1.5)
        self.assertEqual(first_number("-.25"), -0.25)
        self.assertEqual(first_number("12. kHz"), 12.0)


class CrossSourceRelaxationTests(unittest.TestCase):
    """CWRU relaxation times must agree with the independent NRL line summary."""

    def _by_compound_frequency(self, lines: list[dict]) -> dict:
        sites = {row["id"]: row for row in _load("sites.jsonl")}
        samples = {row["id"]: row for row in _load("samples.jsonl")}
        grouped: dict[tuple[str, float], dict[str, list[dict]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for line in lines:
            compound = samples[sites[line["site_id"]]["sample_id"]]["compound_id"]
            source = line["source_id"]
            family = "nrl" if source == "nrl_line_summary_pdf" else (
                "cwru" if source.startswith("cwru_") else None
            )
            if family and line["frequency_khz"] is not None:
                grouped[(compound, line["frequency_khz"])][family].append(line)
        return grouped

    def test_cwru_and_nrl_t1_t2_agree_for_shared_lines(self) -> None:
        grouped = self._by_compound_frequency(_load("lines.jsonl"))
        compared = 0
        for (compound, frequency), families in grouped.items():
            for cwru in families.get("cwru", []):
                for nrl in families.get("nrl", []):
                    for key in ("t1_s", "t2_s"):
                        if cwru[key] is None or nrl[key] is None:
                            continue
                        compared += 1
                        with self.subTest(compound=compound, frequency=frequency, key=key):
                            self.assertAlmostEqual(
                                cwru[key] / nrl[key], 1.0, delta=0.01
                            )
        # Sodium nitrite (3 lines) and glycine (2 lines), T1 and T2 each.
        self.assertGreaterEqual(compared, 10)

    def test_glycine_relaxation_times_are_milliseconds(self) -> None:
        # Regression: glycine T1 = 43.4/50 and T2 = 12.5/17.1 are ms (curator
        # confirmed) but were stored as seconds in both the CWRU and NRL rows.
        # 14N T2 in a protonated solid is ms-scale, so require T2 < 0.1 s and the
        # quoted millisecond values in seconds.
        sites = {row["id"]: row for row in _load("sites.jsonl")}
        samples = {row["id"]: row for row in _load("samples.jsonl")}
        expected = {1052.0: (0.0434, 0.0125), 737.0: (0.050, 0.0171)}
        seen = defaultdict(set)
        for line in _load("lines.jsonl"):
            compound = samples[sites[line["site_id"]]["sample_id"]]["compound_id"]
            if compound != "compound:glycine" or line["frequency_khz"] not in expected:
                continue
            t1, t2 = expected[line["frequency_khz"]]
            with self.subTest(line=line["id"]):
                self.assertLess(line["t2_s"], 0.1)
                self.assertAlmostEqual(line["t1_s"], t1, delta=1e-12)
                self.assertAlmostEqual(line["t2_s"], t2, delta=1e-12)
            seen[line["frequency_khz"]].add(line["source_id"])
        for frequency in expected:
            self.assertEqual(
                seen[frequency],
                {"cwru_html_glycine-nqr-database", "nrl_line_summary_pdf"},
            )

    def test_nrl_errata_rows_are_scaled(self) -> None:
        from build_cwru_database import (
            NRL_RELAXATION_UNIT_ERRATA,
            nrl_relaxation_time_scale,
        )

        self.assertEqual(set(NRL_RELAXATION_UNIT_ERRATA), {("Glycine", 1052.0), ("Glycine", 737.0)})
        self.assertEqual(nrl_relaxation_time_scale("Glycine", "1052"), 1.0e-3)
        self.assertEqual(nrl_relaxation_time_scale("Glycine", "815"), 1.0)
        self.assertEqual(nrl_relaxation_time_scale("Diglycine HCl", "737"), 1.0)

    def test_sqlite_lines_match_normalized_jsonl(self) -> None:
        expected = {
            line["id"]: (line["t1_s"], line["t2_s"], line["t2_star_s"])
            for line in _load("lines.jsonl")
        }
        with sqlite3.connect(DATABASE) as conn:
            rows = conn.execute("SELECT id, t1_s, t2_s, t2_star_s FROM lines").fetchall()
        self.assertEqual(len(rows), len(expected))
        for line_id, *values in rows:
            self.assertEqual(tuple(values), expected[line_id], line_id)
        records = {
            row["line_id"]: (row["line"]["t1_s"], row["line"]["t2_s"], row["line"]["t2_star_s"])
            for row in _load("line_records.jsonl")
        }
        self.assertEqual(records, expected)


if __name__ == "__main__":
    unittest.main()
