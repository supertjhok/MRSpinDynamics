"""Analytic heteronuclear J-editing models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JEditingFitResult:
    """Known-J least-squares fit of a J-modulation curve."""

    couplings_hz: np.ndarray
    amplitudes: np.ndarray
    background: float
    fitted: np.ndarray
    residual: np.ndarray
    rank: int

    @property
    def residual_norm(self) -> float:
        """Euclidean norm of the residual vector."""

        return float(np.linalg.norm(self.residual))


def _as_1d(value: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def j_modulation_curve(
    encoding_times: Iterable[float] | np.ndarray,
    couplings_hz: Iterable[float] | np.ndarray,
    amplitudes: Iterable[float] | np.ndarray | None = None,
    *,
    cycles: int = 1,
    background: float = 0.0,
    powers: Iterable[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return a superposition of J-modulated cosine components."""

    times = _as_1d(encoding_times, "encoding_times")
    couplings = _as_1d(couplings_hz, "couplings_hz")
    cycles = int(cycles)
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if amplitudes is None:
        amps = np.ones_like(couplings)
    else:
        amps = _as_1d(amplitudes, "amplitudes")
    if amps.size != couplings.size:
        raise ValueError("amplitudes must match couplings_hz")
    if powers is None:
        exponents = np.ones(couplings.size, dtype=np.int64)
    else:
        exponents = np.asarray(powers, dtype=np.int64).reshape(-1)
        if exponents.size != couplings.size:
            raise ValueError("powers must match couplings_hz")
        if np.any(exponents < 0):
            raise ValueError("powers must be non-negative")
    phase = 2.0 * np.pi * cycles * times[:, np.newaxis] * couplings[np.newaxis, :]
    basis = np.cos(phase) ** exponents[np.newaxis, :]
    return float(background) + basis @ amps


def carbon_detected_j_modulation(
    encoding_times: Iterable[float] | np.ndarray,
    couplings_hz: Iterable[float] | np.ndarray,
    abundances: Iterable[float] | np.ndarray,
    proton_counts: Iterable[int] | np.ndarray,
    *,
    cycles: int = 1,
    scale: float = 1.0,
) -> np.ndarray:
    """Return the carbon-detected low-field J-editing model."""

    return float(scale) * j_modulation_curve(
        encoding_times,
        couplings_hz,
        abundances,
        cycles=cycles,
        background=0.0,
        powers=proton_counts,
    )


def proton_detected_j_modulation(
    encoding_times: Iterable[float] | np.ndarray,
    couplings_hz: Iterable[float] | np.ndarray,
    amplitudes: Iterable[float] | np.ndarray,
    *,
    cycles: int = 1,
    background: float = 0.0,
) -> np.ndarray:
    """Return the proton-detected J-editing model."""

    return j_modulation_curve(
        encoding_times,
        couplings_hz,
        amplitudes,
        cycles=cycles,
        background=background,
    )


def tango_b_filter(
    couplings_hz: Iterable[float] | np.ndarray,
    *,
    delay_seconds: float | None = None,
    target_coupling_hz: float | None = None,
    order: int = 1,
) -> np.ndarray:
    """Return the ideal TANGO-B coupled-spin transverse filter amplitude."""

    couplings = _as_1d(couplings_hz, "couplings_hz")
    order = int(order)
    if order <= 0 or order % 2 == 0:
        raise ValueError("order must be a positive odd integer")
    if delay_seconds is None:
        if target_coupling_hz is None:
            raise ValueError("provide delay_seconds or target_coupling_hz")
        target = float(target_coupling_hz)
        if target <= 0:
            raise ValueError("target_coupling_hz must be positive")
        delay_seconds = order / (2.0 * target)
    delay_seconds = float(delay_seconds)
    if delay_seconds <= 0:
        raise ValueError("delay_seconds must be positive")
    return np.sin(np.pi * couplings * delay_seconds) ** 2


def fit_known_j_spectrum(
    encoding_times: Iterable[float] | np.ndarray,
    signal: Iterable[float] | np.ndarray,
    couplings_hz: Iterable[float] | np.ndarray,
    *,
    cycles: int = 1,
    powers: Iterable[int] | np.ndarray | None = None,
    include_background: bool = True,
) -> JEditingFitResult:
    """Fit amplitudes for a known set of J-coupling frequencies."""

    times = _as_1d(encoding_times, "encoding_times")
    y = _as_1d(signal, "signal")
    if y.size != times.size:
        raise ValueError("signal must match encoding_times")
    couplings = _as_1d(couplings_hz, "couplings_hz")
    if powers is None:
        exponents = np.ones(couplings.size, dtype=np.int64)
    else:
        exponents = np.asarray(powers, dtype=np.int64).reshape(-1)
        if exponents.size != couplings.size:
            raise ValueError("powers must match couplings_hz")
    phase = 2.0 * np.pi * int(cycles) * times[:, np.newaxis] * couplings[np.newaxis, :]
    columns = [np.cos(phase) ** exponents[np.newaxis, :]]
    if include_background:
        columns.insert(0, np.ones((times.size, 1), dtype=np.float64))
    design = np.column_stack(columns)
    coeffs, _residuals, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    if include_background:
        background = float(coeffs[0])
        amplitudes = coeffs[1:]
    else:
        background = 0.0
        amplitudes = coeffs
    fitted = design @ coeffs
    return JEditingFitResult(
        couplings_hz=couplings,
        amplitudes=np.asarray(amplitudes, dtype=np.float64),
        background=background,
        fitted=fitted,
        residual=y - fitted,
        rank=int(rank),
    )
