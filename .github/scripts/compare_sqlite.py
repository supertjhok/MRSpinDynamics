#!/usr/bin/env python3
"""Compare two SQLite databases logically (schema + row multisets).

Used by the NQRDatabase workflow to check that the tracked
``NQRDatabase/data/exports/nqr.sqlite`` matches a fresh rebuild. A byte diff is
not meaningful for SQLite: page layout and free lists can vary with the library
version while the content is identical.

Usage:
    python .github/scripts/compare_sqlite.py EXPECTED.sqlite ACTUAL.sqlite \
        [--ignore-column generated_at]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _snapshot(path: Path, ignore_columns: set[str]) -> dict[str, tuple]:
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        objects = connection.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        snapshot: dict[str, tuple] = {}
        for kind, name, sql in objects:
            snapshot[f"{kind}:{name}:schema"] = (sql,)
            if kind != "table":
                continue
            columns = [
                row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')
            ]
            keep = [column for column in columns if column not in ignore_columns]
            selected = ", ".join(f'"{column}"' for column in keep) or "1"
            rows = connection.execute(f'SELECT {selected} FROM "{name}"').fetchall()
            snapshot[f"table:{name}:rows"] = tuple(sorted(rows, key=repr))
        return snapshot
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--ignore-column", action="append", default=[])
    args = parser.parse_args(argv)

    ignore = set(args.ignore_column)
    expected = _snapshot(args.expected, ignore)
    actual = _snapshot(args.actual, ignore)
    differences = sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    if differences:
        print(
            f"{args.actual} differs from {args.expected} in {len(differences)} "
            "object(s):",
            file=sys.stderr,
        )
        for key in differences[:50]:
            rows_e = len(expected.get(key, ()))
            rows_a = len(actual.get(key, ()))
            print(f"  {key} (expected {rows_e} item(s), actual {rows_a})", file=sys.stderr)
        return 1
    tables = sum(1 for key in expected if key.endswith(":rows"))
    print(f"OK: {tables} tables identical ({args.expected} == {args.actual})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
