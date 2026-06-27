"""Scalar NMR relaxation models based on spectral densities.

The helpers here provide a compact Bloembergen-Purcell-Pound (BPP)-style model
for estimating ``T1`` and ``T2`` from a rotational correlation time. The default
rate coefficients use the common dipolar ratios
``R1 ~ J(w0) + 4 J(2 w0)`` and
``R2 ~ 1.5 J(0) + 2.5 J(w0) + J(2 w0)``; the overall scale absorbs the
spin-pair constants and any convention-specific prefactors.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


GAS_CONSTANT_J_PER_MOL_K = 8.31446261815324


@dataclass(frozen=True)
class BPPRelaxationRates:
    """Temperature-dependent BPP rates, times, and spectral densities."""

    temperature_kelvin: np.ndarray
    correlation_time_seconds: np.ndarray
    j0_seconds: np.ndarray
    jw_seconds: np.ndarray
    j2w_seconds: np.ndarray
    r1_per_second: np.ndarray
    r2_per_second: np.ndarray
    t1_seconds: np.ndarray
    t2_seconds: np.ndarray

    def as_parameters(self) -> dict[str, np.ndarray]:
        """Return ``T1`` and ``T2`` fields suitable for workflow parameters."""

        return {"T1": self.t1_seconds.copy(), "T2": self.t2_seconds.copy()}


@dataclass(frozen=True)
class BPPRelaxationModel:
    """Configurable BPP relaxation model with Arrhenius correlation time."""

    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray
    tau_ref_seconds: float
    coupling_scale_per_second2: float
    reference_temperature_kelvin: float = 298.15
    activation_energy_j_per_mol: float = 0.0
    r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0)
    r2_coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0)
    baseline_r1_per_second: float = 0.0
    baseline_r2_per_second: float = 0.0

    def __post_init__(self) -> None:
        _require_finite_array(
            np.asarray(self.angular_frequency_rad_per_s, dtype=np.float64),
            "angular_frequency_rad_per_s",
        )
        _require_positive(self.tau_ref_seconds, "tau_ref_seconds")
        _require_positive(self.reference_temperature_kelvin, "reference_temperature_kelvin")
        _require_finite_scalar(
            self.activation_energy_j_per_mol,
            "activation_energy_j_per_mol",
        )
        _require_nonnegative(self.coupling_scale_per_second2, "coupling_scale_per_second2")
        _require_nonnegative(self.baseline_r1_per_second, "baseline_r1_per_second")
        _require_nonnegative(self.baseline_r2_per_second, "baseline_r2_per_second")
        _validate_coefficients(self.r1_coefficients, "r1_coefficients")
        _validate_coefficients(self.r2_coefficients, "r2_coefficients")

    def correlation_time(
        self,
        temperature_kelvin: float | Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        """Return Arrhenius rotational correlation times for temperatures."""

        return arrhenius_correlation_time(
            temperature_kelvin,
            tau_ref_seconds=self.tau_ref_seconds,
            reference_temperature_kelvin=self.reference_temperature_kelvin,
            activation_energy_j_per_mol=self.activation_energy_j_per_mol,
        )

    def rates(
        self,
        temperature_kelvin: float | Iterable[float] | np.ndarray,
    ) -> BPPRelaxationRates:
        """Return BPP ``R1``/``R2`` rates and ``T1``/``T2`` times."""

        return bpp_relaxation_rates(
            angular_frequency_rad_per_s=self.angular_frequency_rad_per_s,
            correlation_time_seconds=self.correlation_time(temperature_kelvin),
            temperature_kelvin=temperature_kelvin,
            coupling_scale_per_second2=self.coupling_scale_per_second2,
            r1_coefficients=self.r1_coefficients,
            r2_coefficients=self.r2_coefficients,
            baseline_r1_per_second=self.baseline_r1_per_second,
            baseline_r2_per_second=self.baseline_r2_per_second,
        )


def spectral_density_lorentzian(
    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return the isotropic rotational spectral density ``2 tau/(1+w^2 tau^2)``."""

    omega, tau = np.broadcast_arrays(
        np.asarray(angular_frequency_rad_per_s, dtype=np.float64),
        np.asarray(correlation_time_seconds, dtype=np.float64),
    )
    _require_finite_array(omega, "angular_frequency_rad_per_s")
    _require_finite_array(tau, "correlation_time_seconds")
    if np.any(tau <= 0.0):
        raise ValueError("correlation_time_seconds must be positive")
    return 2.0 * tau / (1.0 + (omega * tau) ** 2)


def arrhenius_correlation_time(
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    *,
    tau_ref_seconds: float,
    reference_temperature_kelvin: float = 298.15,
    activation_energy_j_per_mol: float = 0.0,
) -> np.ndarray:
    """Return ``tau_c(T)`` using an Arrhenius activation energy."""

    temperature = np.asarray(temperature_kelvin, dtype=np.float64)
    _require_finite_array(temperature, "temperature_kelvin")
    if np.any(temperature <= 0.0):
        raise ValueError("temperature_kelvin must be positive")
    _require_positive(tau_ref_seconds, "tau_ref_seconds")
    _require_positive(reference_temperature_kelvin, "reference_temperature_kelvin")
    _require_finite_scalar(activation_energy_j_per_mol, "activation_energy_j_per_mol")
    exponent = (
        float(activation_energy_j_per_mol)
        / GAS_CONSTANT_J_PER_MOL_K
        * (1.0 / temperature - 1.0 / float(reference_temperature_kelvin))
    )
    tau = float(tau_ref_seconds) * np.exp(exponent)
    _require_finite_array(tau, "correlation_time_seconds")
    return tau


def bpp_relaxation_rates(
    *,
    angular_frequency_rad_per_s: float | Iterable[float] | np.ndarray,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
    temperature_kelvin: float | Iterable[float] | np.ndarray | None = None,
    coupling_scale_per_second2: float = 1.0,
    r1_coefficients: tuple[float, float, float] = (0.0, 1.0, 4.0),
    r2_coefficients: tuple[float, float, float] = (1.5, 2.5, 1.0),
    baseline_r1_per_second: float = 0.0,
    baseline_r2_per_second: float = 0.0,
) -> BPPRelaxationRates:
    """Return BPP relaxation rates from ``J(0)``, ``J(w0)``, and ``J(2w0)``."""

    _require_nonnegative(coupling_scale_per_second2, "coupling_scale_per_second2")
    _require_nonnegative(baseline_r1_per_second, "baseline_r1_per_second")
    _require_nonnegative(baseline_r2_per_second, "baseline_r2_per_second")
    r1c = _validate_coefficients(r1_coefficients, "r1_coefficients")
    r2c = _validate_coefficients(r2_coefficients, "r2_coefficients")

    omega, tau = np.broadcast_arrays(
        np.asarray(angular_frequency_rad_per_s, dtype=np.float64),
        np.asarray(correlation_time_seconds, dtype=np.float64),
    )
    if temperature_kelvin is None:
        temperature = np.full_like(tau, np.nan, dtype=np.float64)
    else:
        temperature = np.broadcast_to(
            np.asarray(temperature_kelvin, dtype=np.float64),
            tau.shape,
        ).copy()
        _require_finite_array(temperature, "temperature_kelvin")
        if np.any(temperature <= 0.0):
            raise ValueError("temperature_kelvin must be positive")

    j0 = spectral_density_lorentzian(0.0, tau)
    jw = spectral_density_lorentzian(omega, tau)
    j2w = spectral_density_lorentzian(2.0 * omega, tau)
    scale = float(coupling_scale_per_second2)
    r1 = scale * (r1c[0] * j0 + r1c[1] * jw + r1c[2] * j2w)
    r2 = scale * (r2c[0] * j0 + r2c[1] * jw + r2c[2] * j2w)
    r1 = r1 + float(baseline_r1_per_second)
    r2 = r2 + float(baseline_r2_per_second)
    return BPPRelaxationRates(
        temperature_kelvin=temperature,
        correlation_time_seconds=tau.copy(),
        j0_seconds=j0,
        jw_seconds=jw,
        j2w_seconds=j2w,
        r1_per_second=r1,
        r2_per_second=r2,
        t1_seconds=_rate_to_time(r1),
        t2_seconds=_rate_to_time(r2),
    )


def apply_relaxation_to_parameters(
    params: Mapping[str, Any] | Any,
    rates: BPPRelaxationRates,
) -> dict[str, Any]:
    """Return a shallow parameter copy with ``T1`` and ``T2`` replaced."""

    if isinstance(params, Mapping):
        updated = dict(params)
    else:
        updated = {
            name: getattr(params, name)
            for name in dir(params)
            if not name.startswith("_") and not callable(getattr(params, name))
        }
    updated["T1"] = rates.t1_seconds.copy()
    updated["T2"] = rates.t2_seconds.copy()
    return updated


def _rate_to_time(rate: np.ndarray) -> np.ndarray:
    return np.divide(
        1.0,
        rate,
        out=np.full_like(rate, np.inf, dtype=np.float64),
        where=rate > 0.0,
    )


def _validate_coefficients(
    coefficients: tuple[float, float, float],
    name: str,
) -> tuple[float, float, float]:
    if len(coefficients) != 3:
        raise ValueError(f"{name} must contain three coefficients")
    values = tuple(float(value) for value in coefficients)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    if any(value < 0.0 for value in values):
        raise ValueError(f"{name} must be non-negative")
    return values


def _require_finite_array(values: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")


def _require_finite_scalar(value: float, name: str) -> None:
    if not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(value: float, name: str) -> None:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative")
