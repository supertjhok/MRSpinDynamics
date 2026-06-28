"""Validate generated NQR database artifacts.

This script is intentionally lightweight: it checks that the generated SQLite
database opens, core tables have plausible row counts, and normalized JSONL
exports are parseable. It is meant for CI and for quick local rebuild checks.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
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

    print("sqlite tables:")
    for table, count in sqlite_counts.items():
        print(f"  {table}={count}")
    print("jsonl exports:")
    for filename, count in jsonl_counts.items():
        print(f"  {filename}={count}")


if __name__ == "__main__":
    main()
