"""Read measured NQR lines from the ``NQRDatabase`` SQLite export.

The integration layer treats the database as a read-only library of validation
targets.  It does not depend on the ``NQRDatabase`` Python package — only on the
exported ``nqr.sqlite`` file and its stable schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

# Default location of the exported database relative to the repository root
# (``integration/`` and ``NQRDatabase/`` are siblings).
_DEFAULT_DB = (
    Path(__file__).resolve().parents[3]
    / "NQRDatabase"
    / "data"
    / "exports"
    / "nqr.sqlite"
)


@dataclass(frozen=True)
class SiteRecord:
    """A measured site with its stored quadrupolar parameters and lines."""

    site_id: str
    compound: str
    isotope: str
    site_label: str | None
    qcc_hz: float
    eta: float
    temperature_k: float | None
    measured_hz: tuple[float, ...]


@dataclass(frozen=True)
class MeasuredLine:
    """One measured NQR line for a compound, with its site context."""

    compound: str
    isotope: str
    site_label: str | None
    frequency_hz: float
    qcc_hz: float | None
    eta: float | None
    temperature_k: float | None
    transition_label: str | None


def default_database_path() -> Path:
    """Return the conventional path to the exported NQR SQLite database."""

    return _DEFAULT_DB


def _connect(database_path: str | Path | None) -> sqlite3.Connection:
    path = Path(database_path) if database_path is not None else _DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(
            f"NQR database not found at {path}; pass database_path= explicitly"
        )
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def measured_lines(
    compound: str,
    *,
    isotope: str | None = None,
    database_path: str | Path | None = None,
) -> list[MeasuredLine]:
    """Return measured lines for ``compound`` (matched on name or formula).

    ``compound`` is matched case-insensitively against the canonical name,
    formula, or compound id.  Pass ``isotope`` (e.g. ``"14N"``) to filter.
    Duplicate ``(isotope, frequency)`` rows from multiple sources are collapsed.
    """

    query = """
        SELECT s.isotope        AS isotope,
               s.site_label      AS site_label,
               s.qcc_khz         AS qcc_khz,
               s.eta             AS eta,
               sa.temperature_k  AS temperature_k,
               l.frequency_khz   AS frequency_khz,
               l.transition_label AS transition_label
        FROM lines l
        JOIN sites s   ON l.site_id = s.id
        JOIN samples sa ON s.sample_id = sa.id
        JOIN compounds c ON sa.compound_id = c.id
        WHERE lower(c.canonical_name) = lower(:needle)
           OR lower(c.formula) = lower(:needle)
           OR lower(c.conventional_formula) = lower(:needle)
           OR lower(c.id) = lower(:needle)
    """
    with _connect(database_path) as connection:
        rows = connection.execute(query, {"needle": compound}).fetchall()

    seen: set[tuple[str, float]] = set()
    results: list[MeasuredLine] = []
    for row in rows:
        if isotope is not None and row["isotope"] != isotope:
            continue
        if row["frequency_khz"] is None:
            continue
        frequency_hz = float(row["frequency_khz"]) * 1.0e3
        key = (row["isotope"], frequency_hz)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            MeasuredLine(
                compound=compound,
                isotope=row["isotope"],
                site_label=row["site_label"],
                frequency_hz=frequency_hz,
                qcc_hz=(
                    float(row["qcc_khz"]) * 1.0e3
                    if row["qcc_khz"] is not None
                    else None
                ),
                eta=float(row["eta"]) if row["eta"] is not None else None,
                temperature_k=(
                    float(row["temperature_k"])
                    if row["temperature_k"] is not None
                    else None
                ),
                transition_label=row["transition_label"],
            )
        )
    results.sort(key=lambda line: (line.isotope, line.frequency_hz))
    return results


def sites_with_parameters(
    *,
    isotope: str | None = None,
    database_path: str | Path | None = None,
) -> list[SiteRecord]:
    """Return every site that has stored ``qcc`` and ``eta`` and >= 1 line.

    These are the rows whose internal consistency can be checked by simulating
    the lines implied by ``(qcc, eta)`` and comparing to the stored lines.
    """

    query = """
        SELECT s.id              AS site_id,
               c.canonical_name  AS compound,
               s.isotope         AS isotope,
               s.site_label      AS site_label,
               s.qcc_khz         AS qcc_khz,
               s.eta             AS eta,
               sa.temperature_k  AS temperature_k,
               group_concat(l.frequency_khz) AS freqs
        FROM sites s
        JOIN lines l   ON l.site_id = s.id
        JOIN samples sa ON s.sample_id = sa.id
        JOIN compounds c ON sa.compound_id = c.id
        WHERE s.qcc_khz IS NOT NULL
          AND s.eta IS NOT NULL
          AND s.isotope IS NOT NULL
          AND l.frequency_khz IS NOT NULL
        GROUP BY s.id
    """
    with _connect(database_path) as connection:
        rows = connection.execute(query).fetchall()

    records: list[SiteRecord] = []
    for row in rows:
        if isotope is not None and row["isotope"] != isotope:
            continue
        measured = tuple(
            sorted(
                float(token) * 1.0e3
                for token in str(row["freqs"]).split(",")
                if token
            )
        )
        if not measured:
            continue
        records.append(
            SiteRecord(
                site_id=str(row["site_id"]),
                compound=str(row["compound"]),
                isotope=str(row["isotope"]),
                site_label=row["site_label"],
                qcc_hz=float(row["qcc_khz"]) * 1.0e3,
                eta=float(row["eta"]),
                temperature_k=(
                    float(row["temperature_k"])
                    if row["temperature_k"] is not None
                    else None
                ),
                measured_hz=measured,
            )
        )
    records.sort(key=lambda record: (record.isotope, record.compound))
    return records
