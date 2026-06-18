"""SNR-informed regularization selection for inverse Laplace transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from spin_dynamics.analysis.ilt import (
    ILTResult1D,
    ILTResult2D,
    KernelName,
    Regularization,
    invert_laplace_1d,
    invert_laplace_2d,
)


@dataclass(frozen=True)
class RegularizationCandidate1D:
    """A trial regularization strength and its 1D inversion result."""

    strength: float
    target_residual_norm: float
    residual_norm: float
    residual_ratio: float
    score: float
    result: ILTResult1D


@dataclass(frozen=True)
class RegularizationCandidate2D:
    """A trial regularization strength and its 2D inversion result."""

    strength: float
    axis_strengths: tuple[float, float]
    target_residual_norm: float
    residual_norm: float
    residual_ratio: float
    score: float
    result: ILTResult2D


@dataclass(frozen=True)
class RegularizationSelection1D:
    """Selected 1D regularization result plus the full candidate trace."""

    selected_strength: float
    selected_regularization: Regularization
    target_residual_norm: float
    estimated_noise_rms: float
    snr: float
    result: ILTResult1D
    candidates: tuple[RegularizationCandidate1D, ...]


@dataclass(frozen=True)
class RegularizationSelection2D:
    """Selected 2D regularization result plus the full candidate trace."""

    selected_strength: float
    selected_regularization: tuple[Regularization, Regularization]
    target_residual_norm: float
    estimated_noise_rms: float
    snr: float
    result: ILTResult2D
    candidates: tuple[RegularizationCandidate2D, ...]


def default_regularization_strengths(
    minimum: float = 1e-8,
    maximum: float = 1e1,
    count: int = 37,
) -> np.ndarray:
    """Return a logarithmic regularization-strength grid."""

    if minimum <= 0.0 or maximum <= 0.0:
        raise ValueError("regularization strength bounds must be positive")
    if minimum >= maximum:
        raise ValueError("minimum must be smaller than maximum")
    if count < 2:
        raise ValueError("count must be at least 2")
    return np.logspace(np.log10(minimum), np.log10(maximum), int(count))


def estimate_noise_rms_from_snr(data: np.ndarray, snr: float) -> float:
    """Estimate noise RMS from observed real data and an RMS SNR estimate.

    The convention is `snr = clean_signal_rms / noise_rms`. Because clean RMS is
    usually not available for measured data, this estimates noise from the
    observed RMS using `observed_rms^2 ~= clean_rms^2 + noise_rms^2`.
    """

    if snr <= 0.0:
        raise ValueError("snr must be positive")
    values = _real_values(data, "data")
    observed_rms = float(np.sqrt(np.mean(values.ravel() ** 2)))
    return observed_rms / np.sqrt(float(snr) ** 2 + 1.0)


def expected_residual_norm_from_snr(
    data: np.ndarray,
    snr: float,
    *,
    target_multiplier: float = 1.0,
) -> float:
    """Return the discrepancy-principle residual norm target for an SNR."""

    if target_multiplier <= 0.0:
        raise ValueError("target_multiplier must be positive")
    values = _real_values(data, "data")
    noise_rms = estimate_noise_rms_from_snr(values, snr)
    return float(target_multiplier * noise_rms * np.sqrt(values.size))


def select_regularization_1d(
    signal: np.ndarray,
    sample_axis: np.ndarray,
    distribution_axis: np.ndarray,
    *,
    snr: float,
    kernel: KernelName | np.ndarray = "t2",
    strengths: Sequence[float] | None = None,
    regularization_order: int = 2,
    nonnegative: bool = True,
    target_multiplier: float = 1.0,
) -> RegularizationSelection1D:
    """Select a 1D regularization strength from an SNR estimate.

    The selected strength is the strongest smoothing that keeps the fitted
    residual norm at or below the expected noise norm. If every candidate
    exceeds the target residual, the closest candidate is used instead. This is
    a practical discrepancy-principle selector for cases where SNR is known or
    estimated.
    """

    grid = _strength_grid(strengths)
    target = expected_residual_norm_from_snr(
        signal,
        snr,
        target_multiplier=target_multiplier,
    )
    candidates: list[RegularizationCandidate1D] = []
    for strength in grid:
        result = invert_laplace_1d(
            signal,
            sample_axis,
            distribution_axis,
            kernel=kernel,
            regularization=Regularization(float(strength), regularization_order),
            nonnegative=nonnegative,
        )
        ratio = _residual_ratio(result.residual_norm, target)
        candidates.append(
            RegularizationCandidate1D(
                strength=float(strength),
                target_residual_norm=target,
                residual_norm=result.residual_norm,
                residual_ratio=ratio,
                score=abs(np.log(ratio)),
                result=result,
            )
        )

    selected = _select_candidate(candidates, target)
    return RegularizationSelection1D(
        selected_strength=selected.strength,
        selected_regularization=selected.result.regularization,
        target_residual_norm=target,
        estimated_noise_rms=estimate_noise_rms_from_snr(signal, snr),
        snr=float(snr),
        result=selected.result,
        candidates=tuple(candidates),
    )


def select_regularization_2d(
    data: np.ndarray,
    sample_axis1: np.ndarray,
    sample_axis2: np.ndarray,
    distribution_axis1: np.ndarray,
    distribution_axis2: np.ndarray,
    *,
    snr: float,
    kernel1: KernelName | np.ndarray,
    kernel2: KernelName | np.ndarray,
    strengths: Sequence[float] | None = None,
    axis_strength_ratio: tuple[float, float] = (1.0, 1.0),
    regularization_order: int | tuple[int, int] = 2,
    nonnegative: bool = True,
    target_multiplier: float = 1.0,
) -> RegularizationSelection2D:
    """Select a shared 2D regularization scale from an SNR estimate.

    Each trial strength is expanded to per-axis strengths by multiplying by
    `axis_strength_ratio`. This keeps the selector one-dimensional while still
    allowing, for example, stronger smoothing along the T2 axis than the T1 or
    diffusion axis.
    """

    if axis_strength_ratio[0] <= 0.0 or axis_strength_ratio[1] <= 0.0:
        raise ValueError("axis_strength_ratio values must be positive")
    grid = _strength_grid(strengths)
    target = expected_residual_norm_from_snr(
        data,
        snr,
        target_multiplier=target_multiplier,
    )
    candidates: list[RegularizationCandidate2D] = []
    for strength in grid:
        axis_strengths = (
            float(strength * axis_strength_ratio[0]),
            float(strength * axis_strength_ratio[1]),
        )
        result = invert_laplace_2d(
            data,
            sample_axis1,
            sample_axis2,
            distribution_axis1,
            distribution_axis2,
            kernel1=kernel1,
            kernel2=kernel2,
            regularization=axis_strengths,
            regularization_order=regularization_order,
            nonnegative=nonnegative,
        )
        ratio = _residual_ratio(result.residual_norm, target)
        candidates.append(
            RegularizationCandidate2D(
                strength=float(strength),
                axis_strengths=axis_strengths,
                target_residual_norm=target,
                residual_norm=result.residual_norm,
                residual_ratio=ratio,
                score=abs(np.log(ratio)),
                result=result,
            )
        )

    selected = _select_candidate(candidates, target)
    return RegularizationSelection2D(
        selected_strength=selected.strength,
        selected_regularization=selected.result.regularization,
        target_residual_norm=target,
        estimated_noise_rms=estimate_noise_rms_from_snr(data, snr),
        snr=float(snr),
        result=selected.result,
        candidates=tuple(candidates),
    )


def _strength_grid(strengths: Sequence[float] | None) -> np.ndarray:
    grid = (
        default_regularization_strengths()
        if strengths is None
        else np.asarray(strengths, dtype=np.float64)
    )
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("strengths must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError("strengths must contain only finite positive values")
    return grid


def _residual_ratio(residual_norm: float, target: float) -> float:
    eps = np.finfo(float).tiny
    return max(float(residual_norm), eps) / max(float(target), eps)


def _select_candidate(candidates, target: float):
    below_target = [
        candidate
        for candidate in candidates
        if candidate.residual_norm <= target
    ]
    if below_target:
        return max(below_target, key=lambda item: item.strength)
    return min(candidates, key=lambda item: (item.score, item.strength))


def _real_values(data: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(data)
    if np.iscomplexobj(values):
        if not np.allclose(np.imag(values), 0.0, atol=1e-12, rtol=1e-12):
            raise ValueError(
                f"{name} must be real, or complex with negligible imaginary part"
            )
        values = np.real(values)
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values
