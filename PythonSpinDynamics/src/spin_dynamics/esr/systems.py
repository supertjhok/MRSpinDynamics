"""Data containers for single-electron ESR simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.operators import spin_dimension, validate_spin


BOHR_MAGNETON_HZ_PER_T = 13.99624555e9
"""Bohr magneton divided by Planck's constant, in Hz/T."""


def as_g_tensor(
    g_tensor: float | np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return a validated 3 by 3 electron ``g`` tensor."""

    values = np.asarray(g_tensor, dtype=np.float64)
    if values.shape == ():
        scalar = float(values)
        out = np.eye(3, dtype=np.float64) * scalar
    elif values.shape == (3,):
        out = np.diag(values)
    elif values.shape == (3, 3):
        out = values.copy()
    else:
        raise ValueError("g_tensor must be a scalar, length-3 vector, or 3x3 matrix")
    if not np.all(np.isfinite(out)):
        raise ValueError("g_tensor must be finite")
    if np.linalg.norm(out) <= 0:
        raise ValueError("g_tensor must be non-zero")
    return out


@dataclass(frozen=True)
class ESRSpinSystem:
    """Single-electron ESR spin system with an isotropic or anisotropic ``g`` tensor."""

    g_tensor: float | np.ndarray | list[float] | tuple[float, ...] = 2.00231930436256
    spin: float = 0.5
    label: str = "e"

    def __post_init__(self) -> None:
        spin = validate_spin(self.spin)
        if not np.isclose(spin, 0.5):
            raise NotImplementedError("ESRSpinSystem currently supports spin=1/2 only")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "g_tensor", as_g_tensor(self.g_tensor))
        object.__setattr__(self, "label", str(self.label))

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension for the electron spin."""

        return spin_dimension(self.spin)


@dataclass(frozen=True)
class ESRTransition:
    """One ESR transition between electron-spin energy eigenstates."""

    label: str
    lower: int
    upper: int
    frequency_hz: float
    dipole_vector: np.ndarray
    strength: float

    def __post_init__(self) -> None:
        dipole_vector = np.asarray(self.dipole_vector, dtype=np.complex128).reshape(3)
        strength = float(self.strength)
        if self.lower < 0 or self.upper < 0 or self.lower == self.upper:
            raise ValueError("transition levels must be distinct non-negative indices")
        if not np.isfinite(self.frequency_hz) or self.frequency_hz < 0:
            raise ValueError("frequency_hz must be non-negative and finite")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("strength must be non-negative and finite")
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "lower", int(self.lower))
        object.__setattr__(self, "upper", int(self.upper))
        object.__setattr__(self, "frequency_hz", float(self.frequency_hz))
        object.__setattr__(self, "dipole_vector", dipole_vector)
        object.__setattr__(self, "strength", strength)


@dataclass(frozen=True)
class ESREigensystem:
    """Energy levels, eigenvectors, and allowed ESR transitions for one field."""

    system: ESRSpinSystem
    b0_vector_tesla_g: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    transitions: tuple[ESRTransition, ...]

    def transition(self, label: str) -> ESRTransition:
        """Return a transition by label."""

        for item in self.transitions:
            if item.label == label:
                return item
        raise KeyError(f"unknown transition label: {label}")
