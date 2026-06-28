"""Write simulator-derived consistency flags back into the NQR database.

The flags are a *derived overlay*: they are computed from the database's own
stored parameters and lines via the spin-dynamics simulator, then written back
as a ``site_consistency_flags`` table (and a matching JSONL export, following
the database's dual SQLite/JSONL convention).

Because the SQLite file is a generated artifact, this overlay is meant to be
(re)generated *after* a database build.  The explorer reads it if present and
degrades gracefully if it is absent, so the database build itself never has to
depend on the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from .database import default_database_path
from .database_validation import (
    DEFAULT_FLAG_THRESHOLD_HZ,
    SiteConsistencyReport,
    describe,
    validate_database,
)

FLAG_TABLE = "site_consistency_flags"

_CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS {FLAG_TABLE} (
    site_id TEXT PRIMARY KEY,
    compound TEXT,
    isotope TEXT,
    spin REAL,
    flagged INTEGER NOT NULL,
    max_abs_diff_hz REAL,
    rms_diff_hz REAL,
    threshold_hz REAL,
    stored_qcc_hz REAL,
    stored_eta REAL,
    implied_qcc_hz REAL,
    implied_eta REAL,
    n_lines INTEGER,
    detail TEXT,
    generated_at TEXT
)
"""


@dataclass(frozen=True)
class FlagExportSummary:
    """Outcome of a flag-export run."""

    database_path: Path
    jsonl_path: Path | None
    sites_written: int
    flagged: int
    generated_at: str


def _report_row(
    report: SiteConsistencyReport,
    *,
    threshold_hz: float,
    generated_at: str,
) -> dict:
    return {
        "site_id": report.site.site_id,
        "compound": report.site.compound,
        "isotope": report.site.isotope,
        "spin": report.spin,
        "flagged": int(report.flagged(threshold_hz)),
        "max_abs_diff_hz": _finite_or_none(report.max_abs_diff_hz),
        "rms_diff_hz": _finite_or_none(report.rms_diff_hz),
        "threshold_hz": threshold_hz,
        "stored_qcc_hz": report.site.qcc_hz,
        "stored_eta": report.site.eta,
        "implied_qcc_hz": report.implied_qcc_hz,
        "implied_eta": report.implied_eta,
        "n_lines": len(report.site.measured_hz),
        "detail": describe(report, threshold_hz=threshold_hz),
        "generated_at": generated_at,
    }


def _finite_or_none(value: float) -> float | None:
    return float(value) if value == value and value not in (float("inf"),) else None


def write_consistency_flags(
    *,
    database_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    threshold_hz: float = DEFAULT_FLAG_THRESHOLD_HZ,
    write_jsonl: bool = True,
) -> FlagExportSummary:
    """Compute consistency flags and write them into the database.

    Returns a summary. ``jsonl_path`` defaults to
    ``<db_dir>/../normalized/site_consistency_flags.jsonl`` to sit beside the
    other normalized exports.
    """

    db_path = Path(database_path) if database_path else default_database_path()
    if not db_path.exists():
        raise FileNotFoundError(f"NQR database not found at {db_path}")

    reports = validate_database(database_path=db_path)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = [
        _report_row(report, threshold_hz=threshold_hz, generated_at=generated_at)
        for report in reports
    ]

    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_CREATE_TABLE)
        connection.execute(f"DELETE FROM {FLAG_TABLE}")
        connection.executemany(
            f"""
            INSERT INTO {FLAG_TABLE} (
                site_id, compound, isotope, spin, flagged,
                max_abs_diff_hz, rms_diff_hz, threshold_hz,
                stored_qcc_hz, stored_eta, implied_qcc_hz, implied_eta,
                n_lines, detail, generated_at
            ) VALUES (
                :site_id, :compound, :isotope, :spin, :flagged,
                :max_abs_diff_hz, :rms_diff_hz, :threshold_hz,
                :stored_qcc_hz, :stored_eta, :implied_qcc_hz, :implied_eta,
                :n_lines, :detail, :generated_at
            )
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()

    resolved_jsonl: Path | None = None
    if write_jsonl:
        if jsonl_path is not None:
            resolved_jsonl = Path(jsonl_path)
        else:
            resolved_jsonl = (
                db_path.parent.parent / "normalized" / "site_consistency_flags.jsonl"
            )
        if resolved_jsonl.parent.exists():
            with resolved_jsonl.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            resolved_jsonl = None

    return FlagExportSummary(
        database_path=db_path,
        jsonl_path=resolved_jsonl,
        sites_written=len(rows),
        flagged=sum(row["flagged"] for row in rows),
        generated_at=generated_at,
    )
