"""Electron-electron dipolar coupling helpers for pulsed dipolar ESR.

The point-dipole secular coupling between two electron spins is the physical
basis of DEER/PELDOR distance measurements. For two spins separated by ``r`` with
the inter-spin vector at angle ``theta`` to the static field, the dipolar angular
frequency is

    omega_dd(r, theta) = omega_perp(r) * (1 - 3 cos^2 theta),

where the perpendicular (``theta = 90 deg``) frequency is

    nu_perp(r) = (mu_0 / 4 pi) * g_a g_b mu_B^2 / (h r^3).

For two free-electron ``g`` values this evaluates to the canonical DEER constant
``nu_perp ~= 52.04 MHz`` at ``r = 1 nm``. This module derives that constant from
fundamental constants (so it tracks the supplied ``g`` values), converts between
distance and dipolar frequency, and builds the secular two-electron dipolar
Hamiltonian used by the density-matrix DEER validator in ``esr.deer``.
"""

from __future__ import annotations

import numpy as np

from spin_dynamics.esr.hamiltonians import TAU
from spin_dynamics.esr.systems import BOHR_MAGNETON_HZ_PER_T
from spin_dynamics.nqr.operators import spin_matrices


FREE_ELECTRON_G = 2.00231930436256
"""Free-electron ``g`` factor (CODATA)."""

# mu_0 / (4 pi) in T^2 m^3 / J. With mu_0 = 4 pi x 1e-7, this is exactly 1e-7.
_MU0_OVER_4PI = 1.0e-7
# Planck constant in J s.
_PLANCK_J_S = 6.62607015e-34
# Bohr magneton in J/T, consistent with BOHR_MAGNETON_HZ_PER_T = mu_B / h.
_BOHR_MAGNETON_J_PER_T = BOHR_MAGNETON_HZ_PER_T * _PLANCK_J_S


def dipolar_constant_hz_nm3(
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> float:
    """Return the perpendicular dipolar constant ``nu_perp * r^3`` in Hz nm^3.

    This is the proportionality constant in ``nu_perp(r) = constant / r_nm^3``.
    For two free-electron ``g`` values it is approximately ``52.04e6`` Hz nm^3.
    """

    g_a = float(g_a)
    g_b = float(g_b)
    if g_a <= 0 or g_b <= 0 or not np.isfinite(g_a) or not np.isfinite(g_b):
        raise ValueError("g values must be positive and finite")
    constant_hz_m3 = (
        _MU0_OVER_4PI * g_a * g_b * _BOHR_MAGNETON_J_PER_T**2 / _PLANCK_J_S
    )
    # nu_perp = constant_hz_m3 / r_m^3. Re-express with r in nm: dividing by
    # r_nm^3 = (r_m / 1e-9)^3 means multiplying the constant by (1e-9)^-3.
    return constant_hz_m3 / (1.0e-9) ** 3


def dipolar_frequency_hz(
    distance_nm: float | np.ndarray,
    *,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> float | np.ndarray:
    """Return the perpendicular dipolar frequency ``nu_perp(r)`` in Hz.

    ``nu_perp`` is the dipolar frequency at ``theta = 90 deg`` and sets the
    position of the Pake-pattern singularities in a DEER dipolar spectrum.
    """

    distance = np.asarray(distance_nm, dtype=np.float64)
    if np.any(distance <= 0) or not np.all(np.isfinite(distance)):
        raise ValueError("distance_nm must be positive and finite")
    constant = dipolar_constant_hz_nm3(g_a, g_b)
    out = constant / distance**3
    return float(out) if np.ndim(distance_nm) == 0 else out


def distance_from_dipolar_frequency_nm(
    frequency_hz: float | np.ndarray,
    *,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> float | np.ndarray:
    """Return the distance (nm) for a perpendicular dipolar frequency in Hz."""

    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if np.any(frequency <= 0) or not np.all(np.isfinite(frequency)):
        raise ValueError("frequency_hz must be positive and finite")
    constant = dipolar_constant_hz_nm3(g_a, g_b)
    out = (constant / frequency) ** (1.0 / 3.0)
    return float(out) if np.ndim(frequency_hz) == 0 else out


def dipolar_angular_frequency_hz(
    distance_nm: float | np.ndarray,
    theta_rad: float | np.ndarray,
    *,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> float | np.ndarray:
    """Return the orientation-dependent dipolar frequency in Hz.

    ``nu_dd(r, theta) = nu_perp(r) * (1 - 3 cos^2 theta)``. This is the
    frequency that modulates the observer echo in a DEER experiment and equals
    the splitting of the observer transition between the two pump-spin states.
    """

    nu_perp = dipolar_frequency_hz(distance_nm, g_a=g_a, g_b=g_b)
    cos_theta = np.cos(np.asarray(theta_rad, dtype=np.float64))
    return nu_perp * (1.0 - 3.0 * cos_theta**2)


def secular_dipolar_hamiltonian(
    distance_nm: float,
    theta_rad: float,
    *,
    g_a: float = FREE_ELECTRON_G,
    g_b: float = FREE_ELECTRON_G,
) -> np.ndarray:
    """Return the secular two-electron dipolar Hamiltonian in radians per second.

    For two spectrally distinct electron spins (the DEER observer/pump regime)
    the secular dipolar interaction keeps only the ``S_zA S_zB`` term:

        H_dd = 2 pi * nu_dd(r, theta) * (S_zA (x) S_zB),

    using spin-1/2 operators with eigenvalues +/- 1/2. With this normalization
    the observer transition is split by exactly ``nu_dd(r, theta)`` between the
    two pump-spin states, so a pump inversion modulates the observer echo at
    ``nu_dd``.
    """

    nu_dd = float(dipolar_angular_frequency_hz(distance_nm, theta_rad, g_a=g_a, g_b=g_b))
    sz = spin_matrices(0.5).iz
    return TAU * nu_dd * np.kron(sz, sz)
