"""Validate the NQR database against the spin-dynamics simulator at scale.

Each curated site stores both quadrupolar parameters ``(qcc, eta)`` and the
measured line frequencies.  Those two are physically linked: diagonalizing the
quadrupole Hamiltonian built from ``(qcc, eta)`` must reproduce the stored
lines.  When it does not, either the parameters or the lines were transcribed
incorrectly (a common failure mode for OCR-derived Landolt-Bornstein tables),
or they were drawn from different measurements.

This module runs that check over every supported site and returns reports
sorted by discrepancy, so the worst offenders surface first.  For spin-1 sites
it additionally back-solves the parameters *implied by the lines*, which
localizes whether ``qcc`` or ``eta`` is the inconsistent field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spin_dynamics.nqr import diagonalize_site

from .conversions import (
    ISOTOPE_SPINS,
    quadrupolar_site_from_cq,
    spin1_parameters_from_lines,
)
from .cross_validation import _unique_within, match_lines
from .database import SiteRecord, sites_with_parameters

#: Default line-agreement threshold; above this a site is "flagged".
DEFAULT_FLAG_THRESHOLD_HZ = 10.0e3


@dataclass(frozen=True)
class SiteConsistencyReport:
    """Agreement between a site's stored parameters and its stored lines."""

    site: SiteRecord
    spin: float
    predicted_hz: np.ndarray
    #: ``(measured_hz, predicted_hz, signed_difference_hz)`` per measured line.
    matches: tuple[tuple[float, float, float], ...]
    max_abs_diff_hz: float
    rms_diff_hz: float
    #: Parameters implied by the lines (spin-1, 3-line sites only).
    implied_qcc_hz: float | None
    implied_eta: float | None

    def flagged(self, threshold_hz: float = DEFAULT_FLAG_THRESHOLD_HZ) -> bool:
        """True when the worst line difference exceeds ``threshold_hz``."""

        return self.max_abs_diff_hz > threshold_hz

    @property
    def qcc_error_hz(self) -> float | None:
        """Stored minus line-implied ``C_Q`` (spin-1 only)."""

        if self.implied_qcc_hz is None:
            return None
        return self.site.qcc_hz - self.implied_qcc_hz

    @property
    def eta_error(self) -> float | None:
        """Stored minus line-implied ``eta`` (spin-1 only)."""

        if self.implied_eta is None:
            return None
        return self.site.eta - self.implied_eta


def supported_isotope(isotope: str) -> bool:
    """True when the simulator can model this isotope's spin."""

    return isotope in ISOTOPE_SPINS


def describe(
    report: SiteConsistencyReport,
    *,
    threshold_hz: float = DEFAULT_FLAG_THRESHOLD_HZ,
) -> str:
    """Return a short human-readable verdict for a consistency report."""

    max_khz = report.max_abs_diff_hz / 1e3
    if not report.flagged(threshold_hz):
        return (
            "Stored quadrupolar parameters reproduce the measured lines "
            f"within {max_khz:.1f} kHz (simulator-verified)."
        )

    # Flagged. For spin-1 we can localize the inconsistent field.
    if report.implied_eta is not None and report.implied_qcc_hz is not None:
        qcc_off_khz = (report.qcc_error_hz or 0.0) / 1e3
        eta_off = report.eta_error or 0.0
        if abs(qcc_off_khz) < 1.0 and abs(eta_off) >= 0.01:
            return (
                f"Stored eta = {report.site.eta:.3f} is inconsistent with the "
                f"measured lines, which imply eta = {report.implied_eta:.3f} "
                f"(C_Q agrees within {abs(qcc_off_khz):.1f} kHz). "
                f"Worst line mismatch {max_khz:.1f} kHz."
            )
        return (
            "Stored parameters disagree with the measured lines: they imply "
            f"C_Q = {report.implied_qcc_hz / 1e6:.4f} MHz, "
            f"eta = {report.implied_eta:.3f} versus stored "
            f"C_Q = {report.site.qcc_hz / 1e6:.4f} MHz, "
            f"eta = {report.site.eta:.3f}. "
            f"Worst line mismatch {max_khz:.1f} kHz."
        )
    return (
        "Stored quadrupolar parameters do not reproduce the measured lines "
        f"(worst mismatch {max_khz:.1f} kHz)."
    )


def check_site(record: SiteRecord) -> SiteConsistencyReport | None:
    """Check one site; return ``None`` if its isotope/spin is unsupported."""

    spin = ISOTOPE_SPINS.get(record.isotope)
    if spin is None:
        return None

    site = quadrupolar_site_from_cq(
        cq_hz=record.qcc_hz,
        eta=record.eta,
        spin=spin,
        isotope=record.isotope,
    )
    predicted = _unique_within(
        np.asarray(
            [t.frequency_hz for t in diagonalize_site(site).transitions], dtype=float
        ),
        1.0,
    )
    matches = tuple(match_lines(predicted, record.measured_hz))
    diffs = [abs(d) for _, _, d in matches if np.isfinite(d)]
    max_abs = max(diffs) if diffs else float("nan")
    rms = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else float("nan")

    implied_qcc: float | None = None
    implied_eta: float | None = None
    if np.isclose(spin, 1.0) and len(record.measured_hz) == 3:
        try:
            implied_qcc, implied_eta = spin1_parameters_from_lines(record.measured_hz)
        except ValueError:
            implied_qcc = implied_eta = None

    return SiteConsistencyReport(
        site=record,
        spin=spin,
        predicted_hz=predicted,
        matches=matches,
        max_abs_diff_hz=max_abs,
        rms_diff_hz=rms,
        implied_qcc_hz=implied_qcc,
        implied_eta=implied_eta,
    )


def validate_database(
    *,
    isotope: str | None = None,
    database_path: str | Path | None = None,
) -> list[SiteConsistencyReport]:
    """Check every supported site, sorted worst-discrepancy first."""

    reports = [
        report
        for record in sites_with_parameters(
            isotope=isotope, database_path=database_path
        )
        if (report := check_site(record)) is not None
    ]
    reports.sort(
        key=lambda report: (
            report.max_abs_diff_hz
            if np.isfinite(report.max_abs_diff_hz)
            else -1.0
        ),
        reverse=True,
    )
    return reports


def summarize(
    reports: list[SiteConsistencyReport],
    *,
    threshold_hz: float = DEFAULT_FLAG_THRESHOLD_HZ,
    worst: int = 10,
) -> str:
    """Return a compact text summary with the worst offenders."""

    total = len(reports)
    flagged = [r for r in reports if r.flagged(threshold_hz)]
    finite = [r.max_abs_diff_hz for r in reports if np.isfinite(r.max_abs_diff_hz)]
    lines = [
        f"Checked {total} sites; "
        f"{total - len(flagged)} consistent, {len(flagged)} flagged "
        f"(> {threshold_hz / 1e3:g} kHz).",
    ]
    if finite:
        lines.append(
            f"max line difference: median={np.median(finite) / 1e3:.1f} kHz  "
            f"max={np.max(finite) / 1e3:.1f} kHz"
        )
    if flagged:
        lines.append("")
        lines.append(f"Worst {min(worst, len(flagged))} flagged sites:")
        header = (
            f"  {'maxdiff':>9}  {'compound':28}  {'stored qcc/eta':>16}  "
            f"{'implied qcc/eta':>16}"
        )
        lines.append(header)
        for report in flagged[:worst]:
            stored = f"{report.site.qcc_hz / 1e3:.0f}/{report.site.eta:.3f}"
            if report.implied_qcc_hz is not None:
                implied = (
                    f"{report.implied_qcc_hz / 1e3:.0f}/{report.implied_eta:.3f}"
                )
            else:
                implied = "-"
            lines.append(
                f"  {report.max_abs_diff_hz / 1e3:8.1f}k  "
                f"{report.site.compound[:28]:28}  {stored:>16}  {implied:>16}"
            )
    return "\n".join(lines)
