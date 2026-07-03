"""Dense angular-momentum operators for quadrupolar spins."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def validate_spin(spin: float) -> float:
    """Return a validated integer or half-integer spin quantum number."""

    spin = float(spin)
    if spin <= 0:
        raise ValueError("spin must be positive")
    two_spin = round(2.0 * spin)
    if not np.isclose(2.0 * spin, two_spin):
        raise ValueError("spin must be an integer or half-integer")
    return spin


def spin_dimension(spin: float) -> int:
    """Return the Hilbert-space dimension for one spin."""

    spin = validate_spin(spin)
    return int(round(2.0 * spin + 1.0))


def is_half_integer_spin(spin: float) -> bool:
    """Return ``True`` for half-integer spin (1/2, 3/2, 5/2, ...)."""

    return round(2.0 * validate_spin(spin)) % 2 == 1


def fundamental_operator_gap(spin: float) -> float:
    """Return the ``eta=0`` fundamental-transition gap ``d`` of the quadrupole operator.

    In the units of ``3 Iz^2 - I(I+1) + eta(Ix^2 - Iy^2)`` the lowest transition
    (``0 <-> 1`` for integer spin, ``1/2 <-> 3/2`` for half-integer spin) spans
    ``d = 3`` or ``d = 6`` respectively. This links the public
    ``quadrupole_frequency_hz`` (nu_Q, the fundamental line) to the operator
    prefactor ``nu_Q / d`` and to ``C_Q`` via ``nu_Q = C_Q d / (4 I (2I-1))``.
    """

    return 6.0 if is_half_integer_spin(spin) else 3.0


@dataclass(frozen=True)
class SpinMatrices:
    """Dense single-spin angular momentum matrices."""

    spin: float
    m_values: np.ndarray
    identity: np.ndarray
    ix: np.ndarray
    iy: np.ndarray
    iz: np.ndarray
    i_plus: np.ndarray
    i_minus: np.ndarray


@lru_cache(maxsize=None)
def spin_matrices(spin: float) -> SpinMatrices:
    """Return dense angular-momentum matrices for one spin."""

    spin = validate_spin(spin)
    dimension = spin_dimension(spin)
    m_values = np.array([spin - idx for idx in range(dimension)], dtype=np.float64)
    i_plus = np.zeros((dimension, dimension), dtype=np.complex128)
    i_minus = np.zeros_like(i_plus)

    by_m = {round(float(m), 12): idx for idx, m in enumerate(m_values)}
    for col, m_value in enumerate(m_values):
        raised = m_value + 1.0
        lowered = m_value - 1.0
        if raised <= spin:
            row = by_m[round(float(raised), 12)]
            i_plus[row, col] = np.sqrt(spin * (spin + 1.0) - m_value * raised)
        if lowered >= -spin:
            row = by_m[round(float(lowered), 12)]
            i_minus[row, col] = np.sqrt(spin * (spin + 1.0) - m_value * lowered)

    ix = 0.5 * (i_plus + i_minus)
    iy = (i_plus - i_minus) / (2.0j)
    iz = np.diag(m_values).astype(np.complex128)
    return SpinMatrices(
        spin=spin,
        m_values=m_values,
        identity=np.eye(dimension, dtype=np.complex128),
        ix=ix,
        iy=iy,
        iz=iz,
        i_plus=i_plus,
        i_minus=i_minus,
    )
