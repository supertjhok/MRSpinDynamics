"""Generate simulator-derived consistency flags and write them to the database.

Run this *after* building the NQR database. It adds a ``site_consistency_flags``
table (and a matching JSONL export) that the explorer surfaces on each site.

    python integration/scripts/write_consistency_flags.py
    python integration/scripts/write_consistency_flags.py --database path/to/nqr.sqlite
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mr_integration import write_consistency_flags
from mr_integration.database_validation import DEFAULT_FLAG_THRESHOLD_HZ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=None,
        help="Path to nqr.sqlite (defaults to the repo's NQRDatabase export).",
    )
    parser.add_argument(
        "--threshold-khz",
        type=float,
        default=DEFAULT_FLAG_THRESHOLD_HZ / 1e3,
        help="Flag sites whose worst line mismatch exceeds this (kHz).",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip writing the JSONL export alongside the SQLite table.",
    )
    args = parser.parse_args()

    summary = write_consistency_flags(
        database_path=args.database,
        threshold_hz=args.threshold_khz * 1e3,
        write_jsonl=not args.no_jsonl,
    )
    print(f"Wrote consistency flags to {summary.database_path}")
    print(
        f"  {summary.sites_written} sites checked, "
        f"{summary.flagged} flagged (> {args.threshold_khz:g} kHz)"
    )
    if summary.jsonl_path is not None:
        print(f"  JSONL export: {summary.jsonl_path}")
    print(f"  generated at {summary.generated_at}")


if __name__ == "__main__":
    main()
