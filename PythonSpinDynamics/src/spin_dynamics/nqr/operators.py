"""Dense angular-momentum operators for quadrupolar spins."""

from __future__ import annotations

from dataclasses import dataclass

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
