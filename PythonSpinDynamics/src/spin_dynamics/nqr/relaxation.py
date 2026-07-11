"""NQR relaxation helpers and backward-compatible shared-model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spin_dynamics.relaxation import (
    DipolarRelaxationSource,
    IsotropicLiquidMotionalAveraging,
    MotionalAveragingModel,
    NQRRelaxationLike,
    NQRRelaxationModel,
    NQRRelaxationSuperoperator,
    PhenomenologicalRelaxationModel,
    RedfieldDipolarRelaxationModel,
    RedfieldEFGRelaxationModel,
    RelaxationModelLike,
    RelaxationSuperoperator,
    RigidSolidMotionalAveraging,
    VibrationalMotionalAveraging,
    cycle_superoperator,
    dipolar_coupling_hz,
    dipolar_coupling_tensor,
    effective_decay_time,
    liouville_hamiltonian,
    liouville_superoperator,
    matrix_exponential,
    propagate_density_liouville,
    relaxation_superoperator,
    quadrupolar_tesseral_operators,
)


@dataclass(frozen=True)
class SpectralOverlapRelaxationFit:
    """Linear fit of a background rate plus spectral-overlap relaxation.

    The model is ``R2 = R_floor + R_cross * S``, where ``S`` is a
    dimensionless spectral-overlap factor normalized to one at the reference
    condition. It is useful when a Hamiltonian calculation predicts how an
    applied field separates otherwise resonant homonuclear transitions, while
    the absolute dipolar strength and other relaxation channels remain nuisance
    parameters.
    """

    floor_rate_per_second: float
    cross_relaxation_rate_per_second: float
    overlap_factors: np.ndarray
    predicted_rates_per_second: np.ndarray
    residuals_per_second: np.ndarray

    @property
    def rms_residual_per_second(self) -> float:
        """Return the unweighted RMS rate residual."""

        return float(np.sqrt(np.mean(self.residuals_per_second**2)))

    @property
    def predicted_t2_seconds(self) -> np.ndarray:
        """Return lifetimes corresponding to the fitted rates."""

        return 1.0 / self.predicted_rates_per_second


def transition_rms_linewidth_hz(
    frequency_offsets_hz: Iterable[float] | np.ndarray,
    intensities: Iterable[float] | np.ndarray,
    *,
    intrinsic_sigma_hz: float = 0.0,
) -> float:
    """Return the intensity-weighted RMS transition width.

    ``intrinsic_sigma_hz`` is an independent Gaussian standard deviation and is
    therefore added in quadrature. The frequency offsets may be measured from
    any common reference because the weighted mean is removed.
    """

    offsets = np.asarray(frequency_offsets_hz, dtype=np.float64)
    weights = np.asarray(intensities, dtype=np.float64)
    if offsets.ndim != 1 or weights.ndim != 1 or offsets.shape != weights.shape:
        raise ValueError("frequency_offsets_hz and intensities must be matching 1-D arrays")
    if offsets.size == 0:
        raise ValueError("at least one transition is required")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(weights)):
        raise ValueError("frequency offsets and intensities must be finite")
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("intensities must be non-negative with a positive sum")
    sigma0 = float(intrinsic_sigma_hz)
    if not np.isfinite(sigma0) or sigma0 < 0.0:
        raise ValueError("intrinsic_sigma_hz must be non-negative and finite")

    normalized = weights / np.sum(weights)
    mean = float(np.sum(normalized * offsets))
    variance = float(np.sum(normalized * (offsets - mean) ** 2))
    return float(np.sqrt(max(variance, 0.0) + sigma0**2))


def spectral_overlap_factors(
    linewidths_hz: Iterable[float] | np.ndarray,
    *,
    reference_index: int = 0,
    exponent: float = 1.0,
) -> np.ndarray:
    """Return normalized overlap factors from transition linewidths.

    For Lorentzian flip-flop overlap, the zero-frequency overlap scales as the
    inverse width, giving the default ``S = (width_ref / width)**1``. The
    exponent is explicit so alternative independently justified overlap laws
    can be tested without changing the fitting routine.
    """

    linewidths = np.asarray(linewidths_hz, dtype=np.float64)
    if linewidths.ndim != 1 or linewidths.size == 0:
        raise ValueError("linewidths_hz must be a non-empty 1-D array")
    if not np.all(np.isfinite(linewidths)) or np.any(linewidths <= 0.0):
        raise ValueError("linewidths_hz must be positive and finite")
    index = int(reference_index)
    if index < 0 or index >= linewidths.size:
        raise ValueError("reference_index is outside linewidths_hz")
    power = float(exponent)
    if not np.isfinite(power) or power <= 0.0:
        raise ValueError("exponent must be positive and finite")
    return (linewidths[index] / linewidths) ** power


def fit_spectral_overlap_relaxation(
    t2_seconds: Iterable[float] | np.ndarray,
    overlap_factors: Iterable[float] | np.ndarray,
    *,
    rate_standard_errors_per_second: Iterable[float] | np.ndarray | None = None,
) -> SpectralOverlapRelaxationFit:
    """Fit ``R2 = R_floor + R_cross * S`` by linear least squares.

    Optional rate standard errors produce an inverse-variance weighted fit.
    The returned residuals always use physical, unweighted rate units.
    """

    lifetimes = np.asarray(t2_seconds, dtype=np.float64)
    overlap = np.asarray(overlap_factors, dtype=np.float64)
    if lifetimes.ndim != 1 or overlap.ndim != 1 or lifetimes.shape != overlap.shape:
        raise ValueError("t2_seconds and overlap_factors must be matching 1-D arrays")
    if lifetimes.size < 2:
        raise ValueError("at least two measurements are required")
    if not np.all(np.isfinite(lifetimes)) or np.any(lifetimes <= 0.0):
        raise ValueError("t2_seconds must be positive and finite")
    if not np.all(np.isfinite(overlap)) or np.any(overlap < 0.0):
        raise ValueError("overlap_factors must be non-negative and finite")
    if np.ptp(overlap) == 0.0:
        raise ValueError("overlap_factors must span at least two values")

    rates = 1.0 / lifetimes
    design = np.column_stack((np.ones_like(overlap), overlap))
    if rate_standard_errors_per_second is not None:
        errors = np.asarray(rate_standard_errors_per_second, dtype=np.float64)
        if errors.shape != rates.shape or not np.all(np.isfinite(errors)):
            raise ValueError("rate standard errors must match measurements and be finite")
        if np.any(errors <= 0.0):
            raise ValueError("rate standard errors must be positive")
        scale = 1.0 / errors
        fit_design = design * scale[:, None]
        fit_rates = rates * scale
    else:
        fit_design = design
        fit_rates = rates

    coefficients, _, rank, _ = np.linalg.lstsq(fit_design, fit_rates, rcond=None)
    if rank != 2:
        raise ValueError("spectral-overlap fit is rank deficient")
    predicted = design @ coefficients
    if np.any(predicted <= 0.0):
        raise ValueError("fit predicts a non-positive relaxation rate")
    return SpectralOverlapRelaxationFit(
        floor_rate_per_second=float(coefficients[0]),
        cross_relaxation_rate_per_second=float(coefficients[1]),
        overlap_factors=overlap.copy(),
        predicted_rates_per_second=predicted,
        residuals_per_second=rates - predicted,
    )

__all__ = [
    "DipolarRelaxationSource",
    "IsotropicLiquidMotionalAveraging",
    "MotionalAveragingModel",
    "NQRRelaxationLike",
    "NQRRelaxationModel",
    "NQRRelaxationSuperoperator",
    "PhenomenologicalRelaxationModel",
    "RedfieldDipolarRelaxationModel",
    "RedfieldEFGRelaxationModel",
    "RelaxationModelLike",
    "RelaxationSuperoperator",
    "RigidSolidMotionalAveraging",
    "VibrationalMotionalAveraging",
    "SpectralOverlapRelaxationFit",
    "cycle_superoperator",
    "dipolar_coupling_hz",
    "dipolar_coupling_tensor",
    "effective_decay_time",
    "fit_spectral_overlap_relaxation",
    "liouville_hamiltonian",
    "liouville_superoperator",
    "matrix_exponential",
    "propagate_density_liouville",
    "relaxation_superoperator",
    "quadrupolar_tesseral_operators",
    "spectral_overlap_factors",
    "transition_rms_linewidth_hz",
]
