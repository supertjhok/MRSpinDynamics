"""Validate generated NQR database artifacts.

This script is intentionally lightweight: it checks that the generated SQLite
database opens, core tables have plausible row counts, normalized JSONL exports
are parseable, overlay tables match their JSONL, and normalized values pass
physical unit-plausibility checks. It is meant for CI and for quick local
rebuild checks.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_cwru_database import isotope_from_form  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]

# Unit-plausibility envelopes. The largest 14N quadrupole couplings are ~6-7 MHz
# and nu+ <= C_Q, so a 14N line above 6.5 MHz is a mis-assigned nucleus or a
# unit error. Quadrupolar T2 in solids is ms-scale; a T2 of 1 s or more is almost
# certainly milliseconds stored as seconds. T2 <= 2 T1 always holds.
MAX_14N_LINE_KHZ = 6500.0
MAX_T2_SECONDS = 1.0
DEFAULT_DATABASE = PROJECT / "data" / "exports" / "nqr.sqlite"
DEFAULT_NORMALIZED_DIR = PROJECT / "data" / "normalized"

MINIMUM_ROW_COUNTS = {
    "sources": 40,
    "compounds": 180,
    "samples": 240,
    "sites": 500,
    "lines": 900,
    "literature_references": 100,
    "reference_links": 1200,
    "nqr_transition_equations": 5,
    "landolt_compound_entries": 160,
    "landolt_measurement_sets": 200,
    "landolt_frequency_records": 600,
    "landolt_review_queue": 150,
}

REQUIRED_JSONL_EXPORTS = (
    "sources.jsonl",
    "compounds.jsonl",
    "samples.jsonl",
    "sites.jsonl",
    "lines.jsonl",
    "line_records.jsonl",
    "literature_references.jsonl",
    "reference_links.jsonl",
    "landolt_compound_entries.jsonl",
    "landolt_measurement_sets.jsonl",
    "landolt_frequency_records.jsonl",
    "landolt_review_queue.jsonl",
)


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
    value = cursor.fetchone()[0]
    return int(value)


def validate_sqlite(database: Path) -> dict[str, int]:
    if not database.exists():
        raise FileNotFoundError(f"SQLite export not found: {database}")

    counts: dict[str, int] = {}
    with sqlite3.connect(database) as conn:
        conn.execute("PRAGMA integrity_check")
        for table, minimum in MINIMUM_ROW_COUNTS.items():
            count = _table_count(conn, table)
            if count < minimum:
                raise AssertionError(
                    f"{table} has {count} rows, expected at least {minimum}"
                )
            counts[table] = count

        orphan_lines = conn.execute(
            """
            SELECT COUNT(*)
            FROM lines
            LEFT JOIN sites ON lines.site_id = sites.id
            WHERE sites.id IS NULL
            """
        ).fetchone()[0]
        if orphan_lines:
            raise AssertionError(f"lines contains {orphan_lines} orphan site links")

    return counts


# Post-build overlays written by integration/scripts/write_consistency_flags.py
# and write_landolt_review_flags.py. Each writes a sqlite table plus a JSONL
# export; when the JSONL is shipped the sqlite table must be shipped too, with
# the same rows (the explorer and review GUIs read only the sqlite table).
OVERLAY_TABLES = {
    "site_consistency_flags": "site_id",
    "landolt_consistency_flags": "entry_id",
}


def validate_overlays(database: Path, normalized_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, key in OVERLAY_TABLES.items():
            path = normalized_dir / f"{table}.jsonl"
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                exported = [json.loads(line) for line in handle if line.strip()]
            if table not in tables:
                raise AssertionError(
                    f"{path.name} is exported but the sqlite export lacks table {table}; "
                    "re-run the integration overlay scripts against the shipped database"
                )
            stored = [dict(row) for row in conn.execute(f"SELECT * FROM {table}")]

            def keyed(rows: list[dict]) -> dict:
                return {
                    (row[key], row.get("measurement_set_id")): {
                        name: value for name, value in row.items() if name != "generated_at"
                    }
                    for row in rows
                }

            if keyed(stored) != keyed(exported):
                raise AssertionError(f"sqlite table {table} differs from {path.name}")
            counts[table] = len(stored)
    return counts


def validate_plausibility(normalized_dir: Path) -> dict[str, int]:
    """Check normalized lines for unit and assignment plausibility.

    Raises ``AssertionError`` listing every offending line id; returns how many
    values each check examined.
    """

    def load(name: str) -> list[dict]:
        with (normalized_dir / name).open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    sites = {row["id"]: row for row in load("sites.jsonl")}
    problems: list[str] = []
    counts = {"14n_lines": 0, "t2_values": 0, "t2_t1_pairs": 0, "form_isotopes": 0}
    for line in load("lines.jsonl"):
        site = sites.get(line["site_id"], {})
        isotope = site.get("isotope")
        frequency = line.get("frequency_khz")
        if isotope == "14N" and frequency is not None:
            counts["14n_lines"] += 1
            if frequency > MAX_14N_LINE_KHZ:
                problems.append(f"{line['id']}: 14N line at {frequency} kHz")
        for key in ("t1_s", "t2_s", "t2_star_s"):
            value = line.get(key)
            if value is not None and value <= 0.0:
                problems.append(f"{line['id']}: non-positive {key} = {value}")
        for key in ("t2_s", "t2_star_s"):
            value = line.get(key)
            if value is not None:
                counts["t2_values"] += 1
                if value >= MAX_T2_SECONDS:
                    problems.append(f"{line['id']}: {key} = {value} s (likely ms)")
        t1, t2 = line.get("t1_s"), line.get("t2_s")
        if t1 is not None and t2 is not None:
            counts["t2_t1_pairs"] += 1
            if t2 > 2.0 * t1:
                problems.append(f"{line['id']}: T2 {t2} s exceeds 2 T1 ({t1} s)")
        form_isotope = isotope_from_form(line.get("form"))
        if form_isotope is not None:
            counts["form_isotopes"] += 1
            if form_isotope != isotope:
                problems.append(
                    f"{line['id']}: form names {form_isotope} but site isotope is {isotope}"
                )
    if problems:
        raise AssertionError(
            "unit-plausibility check failed:\n  " + "\n  ".join(problems)
        )
    return counts


def validate_jsonl_exports(normalized_dir: Path) -> dict[str, int]:
    if not normalized_dir.exists():
        raise FileNotFoundError(f"Normalized export directory not found: {normalized_dir}")

    counts: dict[str, int] = {}
    for filename in REQUIRED_JSONL_EXPORTS:
        path = normalized_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"JSONL export not found: {path}")
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
                rows += 1
        if rows == 0:
            raise AssertionError(f"{path} is empty")
        counts[filename] = rows
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--normalized-dir", type=Path, default=DEFAULT_NORMALIZED_DIR)
    args = parser.parse_args()

    sqlite_counts = validate_sqlite(args.database)
    jsonl_counts = validate_jsonl_exports(args.normalized_dir)
    overlay_counts = validate_overlays(args.database, args.normalized_dir)
    plausibility_counts = validate_plausibility(args.normalized_dir)

    print("sqlite tables:")
    for table, count in sqlite_counts.items():
        print(f"  {table}={count}")
    print("jsonl exports:")
    for filename, count in jsonl_counts.items():
        print(f"  {filename}={count}")
    print("overlays:")
    for table, count in overlay_counts.items():
        print(f"  {table}={count}")
    print("plausibility checks:")
    for check, count in plausibility_counts.items():
        print(f"  {check}={count}")


if __name__ == "__main__":
    main()
