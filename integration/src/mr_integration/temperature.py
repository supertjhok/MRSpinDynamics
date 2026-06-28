"""Compare predicted NQR temperature coefficients against the database.

The database stores a per-line temperature coefficient ``dnu/dT``
(``dnu_dt_khz_per_c``).  A finite-temperature EFG calculation
(``quadrupolar_dft.efg_temperature_sweep``) or an analytic Bayer fit predicts
``nu(T)`` and hence ``dnu/dT`` for the same lines.  This module pulls the
measured coefficients and matches predicted lines to them by frequency, so a
DFT (or model) temperature dependence can be validated against experiment.

The comparison is backend-neutral: a prediction is just a list of
``(frequency_hz, dnu_dt_hz_per_k)`` pairs.  Convenience adapters turn a
``quadrupolar_dft`` temperature sweep into that form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .database import _connect


@dataclass(frozen=True)
class MeasuredTemperatureCoefficient:
    """A measured line frequency and its temperature coefficient."""

    compound: str
    isotope: str
    frequency_hz: float
    dnu_dt_hz_per_k: float
    temperature_k: float | None


def measured_temperature_coefficients(
    compound: str,
    *,
    isotope: str | None = None,
    temperature_k: float | None = None,
    temperature_tolerance_k: float = 5.0,
    database_path: str | Path | None = None,
) -> list[MeasuredTemperatureCoefficient]:
    """Return measured ``(frequency, dnu/dT)`` pairs for a compound.

    ``dnu_dt_khz_per_c`` is kHz per degree (kHz/K); converted to Hz/K. Rows with
    no coefficient are skipped and duplicate ``(isotope, frequency, slope)``
    rows are collapsed. Pass ``temperature_k`` to keep only rows measured near
    that temperature (the database lists a line at several temperatures, each
    carrying the same coefficient), giving one row per physical line.
    """

    query = """
        SELECT s.isotope            AS isotope,
               l.frequency_khz       AS frequency_khz,
               l.dnu_dt_khz_per_c     AS dnu_dt_khz_per_c,
               l.temperature_k        AS temperature_k
        FROM lines l
        JOIN sites s    ON l.site_id = s.id
        JOIN samples sa ON s.sample_id = sa.id
        JOIN compounds c ON sa.compound_id = c.id
        WHERE l.dnu_dt_khz_per_c IS NOT NULL
          AND l.frequency_khz IS NOT NULL
          AND (lower(c.canonical_name) = lower(:needle)
               OR lower(c.formula) = lower(:needle)
               OR lower(c.conventional_formula) = lower(:needle)
               OR lower(c.id) = lower(:needle))
    """
    with _connect(database_path) as connection:
        rows = connection.execute(query, {"needle": compound}).fetchall()

    seen: set[tuple] = set()
    results: list[MeasuredTemperatureCoefficient] = []
    for row in rows:
        if isotope is not None and row["isotope"] != isotope:
            continue
        if temperature_k is not None:
            row_temperature = row["temperature_k"]
            if (
                row_temperature is None
                or abs(float(row_temperature) - temperature_k)
                > temperature_tolerance_k
            ):
                continue
        frequency_hz = float(row["frequency_khz"]) * 1.0e3
        slope = float(row["dnu_dt_khz_per_c"]) * 1.0e3
        key = (row["isotope"], round(frequency_hz, 3), round(slope, 6))
        if key in seen:
            continue
        seen.add(key)
        results.append(
            MeasuredTemperatureCoefficient(
                compound=compound,
                isotope=row["isotope"],
                frequency_hz=frequency_hz,
                dnu_dt_hz_per_k=slope,
                temperature_k=(
                    float(row["temperature_k"])
                    if row["temperature_k"] is not None
                    else None
                ),
            )
        )
    results.sort(key=lambda item: (item.isotope, item.frequency_hz))
    return results


def slopes_from_temperature_points(
    points: Sequence,
) -> list[tuple[float, float]]:
    """Turn a ``quadrupolar_dft`` temperature sweep into ``(freq, dnu/dT)`` pairs.

    ``points`` are ``ThermalEFGPoint`` objects (each with ``temperature_k`` and a
    sorted ``frequencies_hz`` array of equal length).  Uses the lowest and
    highest temperature to estimate each line's slope, reported at the line's
    frequency at the lower temperature.
    """

    ordered = sorted(points, key=lambda p: p.temperature_k)
    if len(ordered) < 2:
        raise ValueError("need at least two temperature points")
    low, high = ordered[0], ordered[-1]
    low_freqs = np.sort(np.asarray(low.frequencies_hz, dtype=float))
    high_freqs = np.sort(np.asarray(high.frequencies_hz, dtype=float))
    if low_freqs.shape != high_freqs.shape:
        raise ValueError("temperature points have differing line counts")
    span = high.temperature_k - low.temperature_k
    if span == 0:
        raise ValueError("temperature points must span a non-zero range")
    return [
        (float(low_freqs[i]), float((high_freqs[i] - low_freqs[i]) / span))
        for i in range(low_freqs.size)
    ]


@dataclass(frozen=True)
class TemperatureCoefficientMatch:
    """A measured coefficient paired with its nearest predicted line."""

    measured: MeasuredTemperatureCoefficient
    predicted_frequency_hz: float
    predicted_dnu_dt_hz_per_k: float

    @property
    def difference_hz_per_k(self) -> float:
        return self.predicted_dnu_dt_hz_per_k - self.measured.dnu_dt_hz_per_k

    @property
    def frequency_offset_hz(self) -> float:
        return self.predicted_frequency_hz - self.measured.frequency_hz


@dataclass(frozen=True)
class TemperatureCoefficientComparison:
    """Predicted vs measured temperature coefficients for one compound."""

    compound: str
    isotope: str | None
    matches: tuple[TemperatureCoefficientMatch, ...]

    @property
    def signs_agree(self) -> bool:
        return all(
            np.sign(m.predicted_dnu_dt_hz_per_k) == np.sign(m.measured.dnu_dt_hz_per_k)
            for m in self.matches
        )

    def format_table(self) -> str:
        lines = [
            f"{self.compound} {self.isotope or ''} temperature coefficients",
            "  line(MHz)  measured(Hz/K)  predicted(Hz/K)   diff",
        ]
        for match in self.matches:
            lines.append(
                f"  {match.measured.frequency_hz / 1e6:8.4f}  "
                f"{match.measured.dnu_dt_hz_per_k:14.1f}  "
                f"{match.predicted_dnu_dt_hz_per_k:14.1f}  "
                f"{match.difference_hz_per_k:+8.1f}"
            )
        return "\n".join(lines)


def compare_temperature_coefficients(
    *,
    compound: str,
    predicted: Sequence[tuple[float, float]],
    isotope: str | None = None,
    temperature_k: float | None = None,
    database_path: str | Path | None = None,
) -> TemperatureCoefficientComparison:
    """Match predicted ``(freq, dnu/dT)`` pairs to measured ones by frequency."""

    measured = measured_temperature_coefficients(
        compound,
        isotope=isotope,
        temperature_k=temperature_k,
        database_path=database_path,
    )
    predicted_freqs = np.asarray([p[0] for p in predicted], dtype=float)
    matches: list[TemperatureCoefficientMatch] = []
    for item in measured:
        if predicted_freqs.size == 0:
            break
        nearest = int(np.argmin(np.abs(predicted_freqs - item.frequency_hz)))
        matches.append(
            TemperatureCoefficientMatch(
                measured=item,
                predicted_frequency_hz=float(predicted[nearest][0]),
                predicted_dnu_dt_hz_per_k=float(predicted[nearest][1]),
            )
        )
    return TemperatureCoefficientComparison(
        compound=compound, isotope=isotope, matches=tuple(matches)
    )
