"""Data containers for pulsed NQR simulations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.operators import spin_dimension, validate_spin


@dataclass(frozen=True)
class QuadrupolarSite:
    """One quadrupolar nucleus in its EFG principal-axis system."""

    spin: float = 1.0
    quadrupole_frequency_hz: float = 1.0e6
    eta: float = 0.0
    gamma_hz_per_t: float = 0.0
    isotope: str = "14N"
    label: str = "N1"

    def __post_init__(self) -> None:
        spin = validate_spin(self.spin)
        quadrupole_frequency_hz = float(self.quadrupole_frequency_hz)
        eta = float(self.eta)
        gamma_hz_per_t = float(self.gamma_hz_per_t)
        if spin <= 0.5:
            raise ValueError("NQR requires spin > 1/2")
        if not np.isfinite(quadrupole_frequency_hz) or quadrupole_frequency_hz <= 0:
            raise ValueError("quadrupole_frequency_hz must be positive and finite")
        if not np.isfinite(eta) or eta < 0.0 or eta > 1.0:
            raise ValueError("eta must be finite and in the range [0, 1]")
        if not np.isfinite(gamma_hz_per_t):
            raise ValueError("gamma_hz_per_t must be finite")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "quadrupole_frequency_hz", quadrupole_frequency_hz)
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "gamma_hz_per_t", gamma_hz_per_t)
        object.__setattr__(self, "isotope", str(self.isotope))
        object.__setattr__(self, "label", str(self.label))

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension for this quadrupolar nucleus."""

        return spin_dimension(self.spin)


@dataclass(frozen=True)
class NQRTransition:
    """One transition between quadrupolar energy eigenstates."""

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
class NQREigensystem:
    """Energy levels, eigenvectors, and allowed transitions for one site."""

    site: QuadrupolarSite
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    transitions: tuple[NQRTransition, ...]

    def transition(self, label: str) -> NQRTransition:
        """Return a transition by label."""

        for item in self.transitions:
            if item.label == label:
                return item
        raise KeyError(f"unknown transition label: {label}")
