"""Feed Landolt consistency flags into the review queue.

For every Landolt measurement set whose tabulated lines disagree with its
reported ``(QCC, eta)`` (see :mod:`mr_integration.landolt_validation`), this:

1. adds a ``quad_consistency_mismatch`` code to the entry's
   ``landolt_review_queue.issue_flags_json`` so it shows up in the review GUI;
2. raises that entry's review priority so it surfaces for re-examination;
3. records the full diagnostic in a derived ``landolt_consistency_flags`` table
   and matching JSONL export.

Steps 1-2 mutate the build-generated ``landolt_review_queue`` as a post-build
overlay (like the site flags): re-run after a database build.  The issue-flag
edit is idempotent -- the code is cleared from every row first, then re-added to
the currently-flagged rows -- so reruns converge.  Human review *decisions*
(status, reviewer notes) live in a separate decisions journal and are never
touched here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .database import default_database_path
from .landolt_validation import (
    DEFAULT_LANDOLT_THRESHOLD_HZ,
    LandoltConsistencyReport,
    describe_landolt,
    validate_landolt_sets,
)

ISSUE_FLAG = "quad_consistency_mismatch"
FLAG_TABLE = "landolt_consistency_flags"
FLAGGED_PRIORITY = 1

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {FLAG_TABLE} (
    entry_id TEXT PRIMARY KEY,
    measurement_set_id TEXT,
    nucleus TEXT,
    isotope TEXT,
    spin REAL,
    substance_name TEXT,
    flagged INTEGER NOT NULL,
    max_gap_hz REAL,
    threshold_hz REAL,
    measured_mhz TEXT,
    predicted_strong_mhz TEXT,
    qcc_eta TEXT,
    detail TEXT,
    generated_at TEXT
)
"""


@dataclass(frozen=True)
class LandoltReviewSummary:
    """Outcome of a Landolt review-flag export run."""

    database_path: Path
    jsonl_path: Path | None
    sets_checked: int
    entries_flagged: int
    queue_rows_updated: int
    generated_at: str


def _worst_per_entry(
    reports: list[LandoltConsistencyReport],
    threshold_hz: float,
) -> dict[str, LandoltConsistencyReport]:
    """Keep, per entry, the flagged report with the largest gap."""

    worst: dict[str, LandoltConsistencyReport] = {}
    for report in reports:
        if not report.flagged(threshold_hz):
            continue
        entry_id = report.record.entry_id
        current = worst.get(entry_id)
        if current is None or report.max_gap_hz > current.max_gap_hz:
            worst[entry_id] = report
    return worst


def _flag_row(report: LandoltConsistencyReport, *, threshold_hz: float, generated_at: str) -> dict:
    record = report.record
    return {
        "entry_id": record.entry_id,
        "measurement_set_id": record.measurement_set_id,
        "nucleus": record.nucleus,
        "isotope": record.isotope,
        "spin": record.spin,
        "substance_name": record.substance_name,
        "flagged": 1,
        "max_gap_hz": float(report.max_gap_hz),
        "threshold_hz": threshold_hz,
        "measured_mhz": json.dumps([round(f / 1e6, 6) for f in record.frequencies_hz]),
        "predicted_strong_mhz": json.dumps(
            [round(float(p) / 1e6, 6) for p in report.predicted_strong_hz]
        ),
        "qcc_eta": json.dumps([[q, e] for q, e in record.qcc_eta_pairs]),
        "detail": describe_landolt(report, threshold_hz=threshold_hz),
        "generated_at": generated_at,
    }


def _clear_issue_flag(connection: sqlite3.Connection) -> None:
    """Remove our issue-flag code from every queue row (for idempotency)."""

    rows = connection.execute(
        "SELECT id, issue_flags_json FROM landolt_review_queue"
    ).fetchall()
    for review_id, flags_json in rows:
        flags = json.loads(flags_json or "[]")
        if ISSUE_FLAG in flags:
            flags = [flag for flag in flags if flag != ISSUE_FLAG]
            connection.execute(
                "UPDATE landolt_review_queue SET issue_flags_json = ? WHERE id = ?",
                [json.dumps(flags), review_id],
            )


def _apply_issue_flag(
    connection: sqlite3.Connection,
    entry_id: str,
    *,
    raise_priority: bool,
) -> int:
    """Add the issue flag (and optionally raise priority) for one entry."""

    rows = connection.execute(
        "SELECT id, issue_flags_json, priority FROM landolt_review_queue "
        "WHERE entry_id = ?",
        [entry_id],
    ).fetchall()
    updated = 0
    for review_id, flags_json, priority in rows:
        flags = json.loads(flags_json or "[]")
        if ISSUE_FLAG not in flags:
            flags.append(ISSUE_FLAG)
        new_priority = (
            min(int(priority), FLAGGED_PRIORITY) if raise_priority else priority
        )
        connection.execute(
            "UPDATE landolt_review_queue SET issue_flags_json = ?, priority = ? "
            "WHERE id = ?",
            [json.dumps(flags), new_priority, review_id],
        )
        updated += 1
    return updated


def write_landolt_review_flags(
    *,
    database_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    threshold_hz: float = DEFAULT_LANDOLT_THRESHOLD_HZ,
    raise_priority: bool = True,
    write_jsonl: bool = True,
) -> LandoltReviewSummary:
    """Validate Landolt entries and route inconsistencies into the review queue."""

    db_path = Path(database_path) if database_path else default_database_path()
    if not db_path.exists():
        raise FileNotFoundError(f"NQR database not found at {db_path}")

    reports = validate_landolt_sets(database_path=db_path, threshold_hz=threshold_hz)
    worst = _worst_per_entry(reports, threshold_hz)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        _flag_row(report, threshold_hz=threshold_hz, generated_at=generated_at)
        for report in worst.values()
    ]

    connection = sqlite3.connect(db_path)
    queue_updated = 0
    try:
        connection.execute(_CREATE_TABLE)
        connection.execute(f"DELETE FROM {FLAG_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {FLAG_TABLE} (
                entry_id, measurement_set_id, nucleus, isotope, spin,
                substance_name, flagged, max_gap_hz, threshold_hz,
                measured_mhz, predicted_strong_mhz, qcc_eta, detail, generated_at
            ) VALUES (
                :entry_id, :measurement_set_id, :nucleus, :isotope, :spin,
                :substance_name, :flagged, :max_gap_hz, :threshold_hz,
                :measured_mhz, :predicted_strong_mhz, :qcc_eta, :detail, :generated_at
            )
            """,
            rows,
        )
        _clear_issue_flag(connection)
        for entry_id in worst:
            queue_updated += _apply_issue_flag(
                connection, entry_id, raise_priority=raise_priority
            )
        connection.commit()
    finally:
        connection.close()

    resolved_jsonl: Path | None = None
    if write_jsonl:
        resolved_jsonl = (
            Path(jsonl_path)
            if jsonl_path is not None
            else db_path.parent.parent / "normalized" / "landolt_consistency_flags.jsonl"
        )
        if resolved_jsonl.parent.exists():
            with resolved_jsonl.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            resolved_jsonl = None

    return LandoltReviewSummary(
        database_path=db_path,
        jsonl_path=resolved_jsonl,
        sets_checked=len(reports),
        entries_flagged=len(worst),
        queue_rows_updated=queue_updated,
        generated_at=generated_at,
    )
