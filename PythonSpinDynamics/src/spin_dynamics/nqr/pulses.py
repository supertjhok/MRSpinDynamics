"""Selective RF pulses for NQR transitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.nqr.hamiltonians import TAU
from spin_dynamics.nqr.systems import NQRTransition


@dataclass(frozen=True)
class SelectivePulse:
    """A rectangular selective RF pulse applied to one NQR transition.

    ``nutation_hz`` is the *effective two-level Rabi frequency* of the addressed
    transition at full RF coupling, i.e. the on-resonance Rabi rate ``Omega / (2
    pi)`` for the embedded |lower>-|upper> two-level system. It already includes
    the transition's dipole matrix element, so it is **not** the bare ``gamma *
    B1 / (2 pi)``: for spin-1 NQR the transition matrix element carries a
    sqrt(2)-type enhancement that this convention folds in. With this
    convention the flip angle is ``theta = 2 pi * nutation_hz * duration_seconds``
    on resonance, so a 90-degree pulse satisfies ``nutation_hz *
    duration_seconds = 0.25`` and a 180-degree pulse ``= 0.5``. The actual drive
    is further scaled by the B1 orientation through ``transition_drive_scale``.
    """

    transition_label: str
    duration_seconds: float
    nutation_hz: float
    phase: float = 0.0
    rf_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        duration_seconds = float(self.duration_seconds)
        nutation_hz = float(self.nutation_hz)
        phase = float(self.phase)
        if not np.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative and finite")
        if not np.isfinite(nutation_hz) or nutation_hz < 0:
            raise ValueError("nutation_hz must be non-negative and finite")
        if not np.isfinite(phase):
            raise ValueError("phase must be finite")
        rf_frequency_hz = self.rf_frequency_hz
        if rf_frequency_hz is not None:
            rf_frequency_hz = float(rf_frequency_hz)
            if not np.isfinite(rf_frequency_hz):
                raise ValueError("rf_frequency_hz must be finite")
        object.__setattr__(self, "transition_label", str(self.transition_label))
        object.__setattr__(self, "duration_seconds", duration_seconds)
        object.__setattr__(self, "nutation_hz", nutation_hz)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "rf_frequency_hz", rf_frequency_hz)


def transition_drive_scale(
    transition: NQRTransition,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float],
) -> float:
    """Return the relative RF coupling for a transition and B1 orientation."""

    direction = np.asarray(b1_direction_pas, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("b1_direction_pas must be a finite non-zero vector")
    direction = direction / norm
    if transition.strength <= 0:
        return 0.0
    return float(abs(np.vdot(direction, transition.dipole_vector)) / transition.strength)


def selective_pulse_hamiltonian(
    dimension: int,
    transition: NQRTransition,
    *,
    nutation_hz: float,
    phase: float = 0.0,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0),
    detuning_hz: float = 0.0,
) -> np.ndarray:
    """Return an embedded two-level RF Hamiltonian in radians per second.

    ``nutation_hz`` is the effective two-level Rabi frequency of the transition
    at full coupling (see :class:`SelectivePulse`), not ``gamma * B1 / (2 pi)``.
    The off-diagonal element is ``pi * nutation_hz * drive * exp(-i phase)`` where
    ``drive`` (magnitude <= 1) is the orientation-dependent coupling from
    ``transition_drive_scale``; on resonance with ``|drive| = 1`` the Rabi rate
    is exactly ``2 pi * nutation_hz``.
    """

    dimension = int(dimension)
    if dimension <= max(transition.lower, transition.upper):
        raise ValueError("dimension does not include the selected transition")
    direction = np.asarray(b1_direction_pas, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("b1_direction_pas must be a finite non-zero vector")
    direction = direction / norm
    if transition.strength <= 0:
        drive = 0.0 + 0.0j
    else:
        drive = np.vdot(direction, transition.dipole_vector) / transition.strength
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    lower = transition.lower
    upper = transition.upper
    offdiag = 0.5 * TAU * float(nutation_hz) * drive * np.exp(-1j * float(phase))
    hamiltonian[lower, upper] = offdiag
    hamiltonian[upper, lower] = np.conj(offdiag)
    if detuning_hz:
        detuning = 0.5 * TAU * float(detuning_hz)
        hamiltonian[upper, upper] += detuning
        hamiltonian[lower, lower] -= detuning
    return hamiltonian


def apply_selective_pulse(
    density: np.ndarray,
    transition: NQRTransition,
    pulse: SelectivePulse,
    *,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> np.ndarray:
    """Apply a selective pulse to a density matrix in the energy basis."""

    density = np.asarray(density, dtype=np.complex128)
    rf_frequency_hz = (
        transition.frequency_hz
        if pulse.rf_frequency_hz is None
        else pulse.rf_frequency_hz
    )
    hamiltonian = selective_pulse_hamiltonian(
        density.shape[0],
        transition,
        nutation_hz=pulse.nutation_hz,
        phase=pulse.phase,
        b1_direction_pas=b1_direction_pas,
        detuning_hz=rf_frequency_hz - transition.frequency_hz,
    )
    return evolve_density(density, hamiltonian, pulse.duration_seconds)
