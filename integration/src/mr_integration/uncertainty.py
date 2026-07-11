"""Propagate DFT parameter uncertainty into simulated NQR line intervals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cross_validation import predicted_lines
from .database import MeasuredLine, measured_lines


@dataclass(frozen=True)
class QuadrupolarParameterDistribution:
    """Correlated normal model for a DFT ``(C_Q, eta)`` estimate."""

    cq_mean_hz: float
    cq_std_hz: float
    eta_mean: float
    eta_std: float
    correlation: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.cq_mean_hz,
            self.cq_std_hz,
            self.eta_mean,
            self.eta_std,
            self.correlation,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("quadrupolar distribution parameters must be finite")
        if self.cq_std_hz < 0.0 or self.eta_std < 0.0:
            raise ValueError("parameter standard deviations must be non-negative")
        if not 0.0 <= self.eta_mean <= 1.0:
            raise ValueError("eta_mean must lie in [0, 1]")
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must lie in [-1, 1]")


@dataclass(frozen=True)
class PredictedLineInterval:
    """Central interval for one ordered simulated transition frequency."""

    lower_hz: float
    median_hz: float
    upper_hz: float


@dataclass(frozen=True)
class UncertainLinePrediction:
    """Monte-Carlo line intervals induced by a parameter distribution."""

    parameters: QuadrupolarParameterDistribution
    isotope: str
    spin: float
    confidence: float
    sample_count: int
    intervals: tuple[PredictedLineInterval, ...]
    max_cross_implementation_discrepancy_hz: float

    @property
    def median_hz(self) -> np.ndarray:
        return np.asarray([interval.median_hz for interval in self.intervals])


@dataclass(frozen=True)
class UncertainLineMatch:
    """One measured line paired with its nearest predicted median and interval."""

    measured_hz: float
    predicted: PredictedLineInterval

    @property
    def residual_hz(self) -> float:
        return self.measured_hz - self.predicted.median_hz

    @property
    def covered(self) -> bool:
        return self.predicted.lower_hz <= self.measured_hz <= self.predicted.upper_hz


@dataclass(frozen=True)
class UncertainComparisonReport:
    """Measured lines compared with uncertainty-aware simulated predictions."""

    compound: str
    isotope: str
    prediction: UncertainLinePrediction
    measured: tuple[MeasuredLine, ...]
    matches: tuple[UncertainLineMatch, ...]

    @property
    def coverage_fraction(self) -> float:
        if not self.matches:
            return float("nan")
        return sum(match.covered for match in self.matches) / len(self.matches)

    def format_table(self) -> str:
        """Return measured, median, interval, and residual values in MHz/kHz."""

        level = 100.0 * self.prediction.confidence
        lines = [
            f"{self.compound} {self.isotope}: {level:g}% prediction intervals",
            "  measured  predicted     lower     upper  residual  covered",
            "      MHz        MHz       MHz       MHz       kHz",
        ]
        for match in self.matches:
            interval = match.predicted
            lines.append(
                f"  {match.measured_hz / 1e6:8.4f}  "
                f"{interval.median_hz / 1e6:9.4f}  "
                f"{interval.lower_hz / 1e6:8.4f}  "
                f"{interval.upper_hz / 1e6:8.4f}  "
                f"{match.residual_hz / 1e3:+8.1f}  "
                f"{'yes' if match.covered else 'no'}"
            )
        return "\n".join(lines)


def _draw_valid_parameters(
    distribution: QuadrupolarParameterDistribution,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    covariance = np.asarray(
        [
            [
                distribution.cq_std_hz**2,
                distribution.correlation
                * distribution.cq_std_hz
                * distribution.eta_std,
            ],
            [
                distribution.correlation
                * distribution.cq_std_hz
                * distribution.eta_std,
                distribution.eta_std**2,
            ],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    remaining = sample_count
    for _ in range(100):
        batch = rng.multivariate_normal(
            [distribution.cq_mean_hz, distribution.eta_mean],
            covariance,
            size=max(remaining * 2, 32),
            check_valid="raise",
        )
        valid = batch[(batch[:, 1] >= 0.0) & (batch[:, 1] <= 1.0)]
        if valid.size:
            accepted.append(valid[:remaining])
            remaining -= min(remaining, valid.shape[0])
        if remaining == 0:
            return np.concatenate(accepted, axis=0)
    raise ValueError(
        "could not draw enough eta samples in [0, 1]; use a narrower distribution"
    )


def propagate_parameter_uncertainty(
    distribution: QuadrupolarParameterDistribution,
    *,
    spin: float,
    isotope: str,
    sample_count: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> UncertainLinePrediction:
    """Sample DFT parameters and return ordered simulated line intervals."""

    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    parameters = _draw_valid_parameters(
        distribution,
        sample_count=int(sample_count),
        seed=int(seed),
    )
    line_rows: list[np.ndarray] = []
    discrepancies: list[float] = []
    expected_count: int | None = None
    for cq_hz, eta in parameters:
        prediction = predicted_lines(
            cq_hz=float(cq_hz),
            eta=float(eta),
            spin=float(spin),
            isotope=isotope,
        )
        if expected_count is None:
            expected_count = prediction.simulator_hz.size
        elif prediction.simulator_hz.size != expected_count:
            raise ValueError("transition count changed across parameter samples")
        line_rows.append(prediction.simulator_hz)
        discrepancies.append(prediction.max_abs_discrepancy_hz)

    samples = np.asarray(line_rows, dtype=float)
    tail = 0.5 * (1.0 - confidence)
    quantiles = np.quantile(samples, [tail, 0.5, 1.0 - tail], axis=0)
    intervals = tuple(
        PredictedLineInterval(
            lower_hz=float(quantiles[0, index]),
            median_hz=float(quantiles[1, index]),
            upper_hz=float(quantiles[2, index]),
        )
        for index in range(samples.shape[1])
    )
    return UncertainLinePrediction(
        parameters=distribution,
        isotope=str(isotope),
        spin=float(spin),
        confidence=float(confidence),
        sample_count=int(sample_count),
        intervals=intervals,
        max_cross_implementation_discrepancy_hz=max(discrepancies, default=0.0),
    )


def compare_uncertain_dft_to_measured(
    *,
    compound: str,
    distribution: QuadrupolarParameterDistribution,
    spin: float,
    isotope: str,
    database_path: str | Path | None = None,
    sample_count: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> UncertainComparisonReport:
    """Propagate parameter uncertainty and compare intervals with measurements."""

    prediction = propagate_parameter_uncertainty(
        distribution,
        spin=spin,
        isotope=isotope,
        sample_count=sample_count,
        confidence=confidence,
        seed=seed,
    )
    measured = tuple(
        measured_lines(compound, isotope=isotope, database_path=database_path)
    )
    matches: list[UncertainLineMatch] = []
    for line in measured:
        if not prediction.intervals:
            break
        index = int(np.argmin(np.abs(prediction.median_hz - line.frequency_hz)))
        matches.append(
            UncertainLineMatch(
                measured_hz=line.frequency_hz,
                predicted=prediction.intervals[index],
            )
        )
    return UncertainComparisonReport(
        compound=compound,
        isotope=isotope,
        prediction=prediction,
        measured=measured,
        matches=tuple(matches),
    )
