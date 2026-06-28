"""Flag Landolt entries whose lines disagree with their QCC/eta for review.

Run after building the NQR database. Adds a ``quad_consistency_mismatch`` issue
flag (and raises priority) on the affected ``landolt_review_queue`` rows so they
surface in the Landolt review GUI, and records the diagnostics in a
``landolt_consistency_flags`` table plus JSONL.

    python integration/scripts/write_landolt_review_flags.py
    python integration/scripts/write_landolt_review_flags.py --threshold-khz 100
    python integration/scripts/write_landolt_review_flags.py --no-priority
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mr_integration import write_landolt_review_flags
from mr_integration.landolt_validation import DEFAULT_LANDOLT_THRESHOLD_HZ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument(
        "--threshold-khz",
        type=float,
        default=DEFAULT_LANDOLT_THRESHOLD_HZ / 1e3,
        help="Flag a set when a predicted strong line misses every measured "
        "line by more than this (kHz).",
    )
    parser.add_argument(
        "--no-priority",
        action="store_true",
        help="Do not raise review priority on flagged entries.",
    )
    parser.add_argument("--no-jsonl", action="store_true")
    args = parser.parse_args()

    summary = write_landolt_review_flags(
        database_path=args.database,
        threshold_hz=args.threshold_khz * 1e3,
        raise_priority=not args.no_priority,
        write_jsonl=not args.no_jsonl,
    )
    print(f"Checked {summary.sets_checked} Landolt sets with lines + QCC/eta")
    print(
        f"  flagged {summary.entries_flagged} entries "
        f"(> {args.threshold_khz:g} kHz), updated "
        f"{summary.queue_rows_updated} review-queue rows"
    )
    if summary.jsonl_path is not None:
        print(f"  JSONL export: {summary.jsonl_path}")
    print(f"  generated at {summary.generated_at}")


if __name__ == "__main__":
    main()
