"""Cross-check Landolt-Bornstein entries: do tabulated lines match QCC/eta?

Unlike the canonical ``sites`` table (one site carries both parameters and
lines), the Landolt import stores, *per measurement set*, an independent list of
frequencies and an independent list of ``(QCC, eta)`` pairs.  Those two lists
describe the same nuclei, so the parameters must reproduce the lines.

For each measurement set that has both lists and a supported nucleus, this
module predicts the two strong zero-field lines (``nu_+`` and ``nu_-``) for each
``(QCC, eta)`` pair via the simulator, and checks that every predicted strong
line has a nearby measured line.  The weak ``nu_0 = nu_+ - nu_-`` line is
intentionally ignored: it is frequently too weak to tabulate, so requiring it
would flag almost everything.  Extra measured lines (other sites, or a stray
temperature that leaked into the list) are harmless under this direction.

The flags this produces are routed into ``landolt_review_queue`` by
``landolt_review_export`` so OCR/transcription errors get re-examined against
their source crops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

import numpy as np

from spin_dynamics.nqr import diagonalize_site

from .conversions import ISOTOPE_SPINS, quadrupolar_site_from_cq
from .cross_validation import match_lines

#: Landolt frequencies are quoted to ~1 kHz; the bulk of consistent sets agree
#: to < 1 kHz, so a much looser default cleanly isolates real discrepancies.
DEFAULT_LANDOLT_THRESHOLD_HZ = 50.0e3


@dataclass(frozen=True)
class LandoltSetRecord:
    """One Landolt measurement set with both lines and coupling parameters."""

    measurement_set_id: str
    entry_id: str
    nucleus: str
    isotope: str
    spin: float
    substance_name: str | None
    frequencies_hz: tuple[float, ...]
    qcc_eta_pairs: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class LandoltConsistencyReport:
    """Agreement between a set's tabulated lines and its QCC/eta pairs."""

    record: LandoltSetRecord
    predicted_strong_hz: np.ndarray
    #: ``(predicted_hz, nearest_measured_hz, signed_difference_hz)`` per strong line.
    matches: tuple[tuple[float, float, float], ...]
    max_gap_hz: float

    def flagged(self, threshold_hz: float = DEFAULT_LANDOLT_THRESHOLD_HZ) -> bool:
        return np.isfinite(self.max_gap_hz) and self.max_gap_hz > threshold_hz


def parse_nucleus(nucleus: str | None) -> tuple[str, float] | None:
    """Parse a Landolt nucleus string (e.g. ``"N-14"``) to ``(isotope, spin)``.

    Returns ``None`` when the nucleus is missing or its spin is not supported by
    the simulator's quadrupole-frequency scale.
    """

    if not nucleus:
        return None
    match = re.fullmatch(r"\s*([A-Za-z]{1,2})-?(\d{1,3})\s*", str(nucleus))
    if not match:
        return None
    element, mass = match.group(1), match.group(2)
    isotope = f"{mass}{element[0].upper()}{element[1:].lower()}"
    spin = ISOTOPE_SPINS.get(isotope)
    if spin is None:
        return None
    return isotope, spin


def _parse_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def landolt_sets_with_parameters(
    connection: sqlite3.Connection,
) -> list[LandoltSetRecord]:
    """Return Landolt measurement sets that have both lines and QCC/eta pairs."""

    connection.row_factory = sqlite3.Row
    sets = connection.execute(
        """
        SELECT ms.id AS set_id, ms.entry_id AS entry_id,
               e.nucleus AS nucleus, e.substance_name AS substance_name
        FROM landolt_measurement_sets ms
        JOIN landolt_compound_entries e ON e.id = ms.entry_id
        """
    ).fetchall()

    records: list[LandoltSetRecord] = []
    for row in sets:
        parsed = parse_nucleus(row["nucleus"])
        if parsed is None:
            continue
        isotope, spin = parsed
        freqs = [
            _parse_float(r["frequency_original"])
            for r in connection.execute(
                "SELECT frequency_original FROM landolt_frequency_records "
                "WHERE measurement_set_id = ? ORDER BY sequence_index",
                [row["set_id"]],
            )
        ]
        pairs = [
            (_parse_float(r["qcc_original"]), _parse_float(r["eta_original"]))
            for r in connection.execute(
                "SELECT qcc_original, eta_original FROM landolt_qcc_eta_records "
                "WHERE measurement_set_id = ? ORDER BY sequence_index",
                [row["set_id"]],
            )
        ]
        freqs_hz = tuple(f * 1.0e6 for f in freqs if f is not None and f > 0)
        qe = tuple(
            (q, e)
            for q, e in pairs
            if q is not None and e is not None and 0.0 <= e <= 1.0
        )
        if not freqs_hz or not qe:
            continue
        records.append(
            LandoltSetRecord(
                measurement_set_id=str(row["set_id"]),
                entry_id=str(row["entry_id"]),
                nucleus=str(row["nucleus"]),
                isotope=isotope,
                spin=spin,
                substance_name=row["substance_name"],
                frequencies_hz=freqs_hz,
                qcc_eta_pairs=qe,
            )
        )
    return records


def check_landolt_set(record: LandoltSetRecord) -> LandoltConsistencyReport:
    """Predict strong lines from each QCC/eta pair and match them to the lines."""

    strong: list[float] = []
    for qcc_mhz, eta in record.qcc_eta_pairs:
        site = quadrupolar_site_from_cq(
            cq_hz=abs(qcc_mhz) * 1.0e6,
            eta=eta,
            spin=record.spin,
            isotope=record.isotope,
        )
        ascending = sorted(t.frequency_hz for t in diagonalize_site(site).transitions)
        # The two highest transitions are the strong, tabulated nu_- and nu_+.
        strong.extend(ascending[-2:])

    predicted = np.sort(np.asarray(strong, dtype=float))
    measured = np.asarray(record.frequencies_hz, dtype=float)
    # Each predicted strong line must have a nearby measured line.
    matches = tuple(match_lines(measured, predicted))
    gaps = [abs(d) for _, _, d in matches if np.isfinite(d)]
    max_gap = max(gaps) if gaps else float("inf")
    return LandoltConsistencyReport(
        record=record,
        predicted_strong_hz=predicted,
        matches=matches,
        max_gap_hz=max_gap,
    )


def validate_landolt_sets(
    *,
    database_path: str | Path,
    threshold_hz: float = DEFAULT_LANDOLT_THRESHOLD_HZ,
) -> list[LandoltConsistencyReport]:
    """Check every supported Landolt set, sorted worst-discrepancy first."""

    connection = sqlite3.connect(f"file:{Path(database_path)}?mode=ro", uri=True)
    try:
        records = landolt_sets_with_parameters(connection)
    finally:
        connection.close()
    reports = [check_landolt_set(record) for record in records]
    reports.sort(
        key=lambda r: r.max_gap_hz if np.isfinite(r.max_gap_hz) else -1.0,
        reverse=True,
    )
    return reports


def describe_landolt(
    report: LandoltConsistencyReport,
    *,
    threshold_hz: float = DEFAULT_LANDOLT_THRESHOLD_HZ,
) -> str:
    """Return a short human-readable verdict for a Landolt set report."""

    gap_khz = report.max_gap_hz / 1e3
    pairs = ", ".join(
        f"(QCC {q:g} MHz, eta {e:g})" for q, e in report.record.qcc_eta_pairs
    )
    measured = ", ".join(f"{f / 1e6:g}" for f in report.record.frequencies_hz)
    if not report.flagged(threshold_hz):
        return (
            "Tabulated lines are consistent with the reported QCC/eta "
            f"(within {gap_khz:.1f} kHz)."
        )
    return (
        f"Reported {pairs} predict strong lines that miss the tabulated "
        f"frequencies [{measured}] MHz by up to {gap_khz:.0f} kHz; "
        "check the OCR-derived frequencies and coupling parameters."
    )
