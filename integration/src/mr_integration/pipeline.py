"""End-to-end predicted-vs-measured NQR comparison.

Ties the three subprojects together:

    ab initio (C_Q, eta)            measured database lines
            |                                |
            v                                |
   spin-dynamics simulator                   |
            |                                |
            +------------> compare <---------+
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_validation import PredictedLines, match_lines, predicted_lines
from .database import MeasuredLine, measured_lines


@dataclass(frozen=True)
class ComparisonReport:
    """Predicted lines (from DFT params) versus measured database lines."""

    compound: str
    isotope: str
    spin: float
    predicted: PredictedLines
    measured: tuple[MeasuredLine, ...]
    #: ``(measured_hz, predicted_hz, signed_difference_hz)`` per measured line.
    matches: tuple[tuple[float, float, float], ...]

    @property
    def measured_hz(self) -> np.ndarray:
        return np.asarray([m.frequency_hz for m in self.measured], dtype=float)

    @property
    def max_abs_difference_hz(self) -> float:
        diffs = [abs(d) for _, _, d in self.matches if np.isfinite(d)]
        return max(diffs) if diffs else float("nan")

    @property
    def rms_difference_hz(self) -> float:
        diffs = [d for _, _, d in self.matches if np.isfinite(d)]
        if not diffs:
            return float("nan")
        return float(np.sqrt(np.mean(np.square(diffs))))

    def format_table(self) -> str:
        """Return a compact human-readable comparison table (MHz)."""

        lines = [
            f"{self.compound} {self.isotope} (spin {self.spin:g})",
            f"  C_Q = {self.predicted.cq_hz / 1e6:+.4f} MHz   "
            f"eta = {self.predicted.eta:.4f}   "
            f"nu_Q = {self.predicted.nu_q_hz / 1e6:.4f} MHz",
            "  measured(MHz)  predicted(MHz)   diff(kHz)",
        ]
        for measured, predicted, diff in self.matches:
            predicted_str = "      -   " if np.isnan(predicted) else f"{predicted / 1e6:9.4f}"
            diff_str = "    -   " if np.isnan(diff) else f"{diff / 1e3:+8.1f}"
            lines.append(f"  {measured / 1e6:11.4f}  {predicted_str}   {diff_str}")
        if np.isfinite(self.rms_difference_hz):
            lines.append(f"  RMS difference: {self.rms_difference_hz / 1e3:.1f} kHz")
        return "\n".join(lines)


def compare_dft_to_measured(
    *,
    compound: str,
    cq_hz: float,
    eta: float,
    spin: float,
    isotope: str,
    database_path: str | Path | None = None,
) -> ComparisonReport:
    """Predict lines from ``(C_Q, eta)`` and compare to measured database lines.

    The same convention/Hamiltonian used by the simulator is applied, then the
    result is paired against measured database lines for ``compound``/``isotope``.
    """

    predicted = predicted_lines(cq_hz=cq_hz, eta=eta, spin=spin, isotope=isotope)
    measured = tuple(
        measured_lines(compound, isotope=isotope, database_path=database_path)
    )
    matches = tuple(
        match_lines(predicted.simulator_hz, [m.frequency_hz for m in measured])
    )
    return ComparisonReport(
        compound=compound,
        isotope=isotope,
        spin=float(spin),
        predicted=predicted,
        measured=measured,
        matches=matches,
    )
