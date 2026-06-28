"""Thermal phonon occupation and harmonic vibrational amplitudes.

These are the temperature-dependent ingredients shared by the harmonic EFG
averaging (:mod:`quadrupolar_dft.vibrational`) and the analytic Bayer model.
A harmonic mode of angular frequency ``omega`` has a mean-square (mass-weighted)
normal coordinate

    <Q^2>(T) = (hbar / 2 omega) * coth(hbar omega / 2 k_B T)

The ``coth`` factor is ``2 n(omega, T) + 1`` with ``n`` the Bose-Einstein
occupation; it goes to ``1`` as ``T -> 0`` (the zero-point amplitude, which is
why even a static-geometry EFG is never exactly what is measured) and to
``2 k_B T / hbar omega`` in the classical high-temperature limit.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    ANGULAR_FREQUENCY_PER_WAVENUMBER_CM,
    BOLTZMANN_CONSTANT_J_PER_K,
    REDUCED_PLANCK_CONSTANT_J_S,
)


def wavenumber_to_angular_frequency(wavenumber_cm_inv: float) -> float:
    """Convert a wavenumber in cm^-1 to angular frequency in rad/s."""

    return float(wavenumber_cm_inv) * ANGULAR_FREQUENCY_PER_WAVENUMBER_CM


def _reduced_frequency(omega_rad_s: float, temperature_k: float) -> float:
    """Return ``hbar omega / 2 k_B T`` (dimensionless)."""

    omega_rad_s = float(omega_rad_s)
    temperature_k = float(temperature_k)
    if omega_rad_s <= 0.0:
        raise ValueError("angular frequency must be positive")
    if temperature_k < 0.0:
        raise ValueError("temperature must be non-negative")
    if temperature_k == 0.0:
        return np.inf
    return (
        REDUCED_PLANCK_CONSTANT_J_S
        * omega_rad_s
        / (2.0 * BOLTZMANN_CONSTANT_J_PER_K * temperature_k)
    )


def thermal_quantum_factor(omega_rad_s: float, temperature_k: float) -> float:
    """Return ``coth(hbar omega / 2 k_B T) = 2 n + 1``.

    Equals 1 at ``T = 0`` (pure zero-point motion) and grows like
    ``2 k_B T / hbar omega`` at high temperature.
    """

    x = _reduced_frequency(omega_rad_s, temperature_k)
    if not np.isfinite(x):
        return 1.0
    return float(1.0 / np.tanh(x))


def bose_occupation(omega_rad_s: float, temperature_k: float) -> float:
    """Return the Bose-Einstein occupation ``n(omega, T)``."""

    x = _reduced_frequency(omega_rad_s, temperature_k)
    # n = 1/(exp(2x) - 1); 2x = hbar omega / k_B T. For large 2x the mode is
    # frozen out (n -> 0); guard against exp overflow.
    if not np.isfinite(x) or 2.0 * x > 700.0:
        return 0.0
    return float(1.0 / np.expm1(2.0 * x))


def mean_square_normal_coordinate(
    omega_rad_s: float, temperature_k: float
) -> float:
    """Return ``<Q^2>(T)`` for a mass-weighted normal coordinate (SI, kg m^2).

    ``<Q^2> = (hbar / 2 omega) coth(hbar omega / 2 k_B T)``.
    """

    omega_rad_s = float(omega_rad_s)
    zero_point = REDUCED_PLANCK_CONSTANT_J_S / (2.0 * omega_rad_s)
    return zero_point * thermal_quantum_factor(omega_rad_s, temperature_k)
