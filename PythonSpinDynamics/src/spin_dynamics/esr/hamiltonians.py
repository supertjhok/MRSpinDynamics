"""Hamiltonian builders and transition analysis for ESR."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spin_dynamics.esr.systems import (
    BOHR_MAGNETON_HZ_PER_T,
    ESREigensystem,
    ESRSpinSystem,
    ESRTransition,
)
from spin_dynamics.nqr.operators import spin_matrices


TAU = 2.0 * np.pi


def effective_g_vector(
    system: ESRSpinSystem,
    b0_direction_g: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return ``g^T n`` for a unit static-field direction in the ``g`` frame."""

    direction = np.asarray(b0_direction_g, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(direction)):
        raise ValueError("b0_direction_g must be finite")
    norm = float(np.linalg.norm(direction))
    if norm <= 0:
        raise ValueError("b0_direction_g must be non-zero")
    return system.g_tensor.T @ (direction / norm)


def effective_g_value(
    system: ESRSpinSystem,
    b0_direction_g: np.ndarray | Sequence[float],
) -> float:
    """Return the ESR effective ``g`` value for a static-field direction."""

    return float(np.linalg.norm(effective_g_vector(system, b0_direction_g)))


def resonance_frequency_hz(
    system: ESRSpinSystem,
    b0_vector_tesla_g: float | np.ndarray | Sequence[float],
) -> float:
    """Return the spin-1/2 ESR transition frequency in hertz."""

    b0 = np.asarray(b0_vector_tesla_g, dtype=np.float64)
    if b0.shape == ():
        magnitude = abs(float(b0))
        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        vector = b0.reshape(3)
        if not np.all(np.isfinite(vector)):
            raise ValueError("b0_vector_tesla_g must be finite")
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0:
            return 0.0
        direction = vector / magnitude
    if not np.isfinite(magnitude):
        raise ValueError("b0_vector_tesla_g must be finite")
    return BOHR_MAGNETON_HZ_PER_T * magnitude * effective_g_value(system, direction)


def resonance_field_tesla(
    system: ESRSpinSystem,
    microwave_frequency_hz: float,
    b0_direction_g: np.ndarray | Sequence[float] = (0.0, 0.0, 1.0),
) -> float:
    """Return the resonant static-field magnitude for one ESR orientation."""

    frequency = float(microwave_frequency_hz)
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError("microwave_frequency_hz must be positive and finite")
    g_eff = effective_g_value(system, b0_direction_g)
    if g_eff <= 0:
        raise ValueError("effective g value must be positive")
    return frequency / (BOHR_MAGNETON_HZ_PER_T * g_eff)


def zeeman_hamiltonian(
    system: ESRSpinSystem,
    b0_vector_tesla_g: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the electron Zeeman Hamiltonian in radians per second."""

    b0 = np.asarray(b0_vector_tesla_g, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_g must be finite")
    ops = spin_matrices(system.spin)
    effective_field = system.g_tensor.T @ b0
    spin_operator = (
        effective_field[0] * ops.ix
        + effective_field[1] * ops.iy
        + effective_field[2] * ops.iz
    )
    return TAU * BOHR_MAGNETON_HZ_PER_T * spin_operator


def diagonalize_system(
    system: ESRSpinSystem,
    b0_vector_tesla_g: np.ndarray | Sequence[float],
    *,
    strength_tolerance: float = 1e-12,
    frequency_tolerance_hz: float = 1e-9,
) -> ESREigensystem:
    """Diagonalize the ESR Zeeman Hamiltonian and return transition metadata."""

    b0 = np.asarray(b0_vector_tesla_g, dtype=np.float64).reshape(3)
    hamiltonian = zeeman_hamiltonian(system, b0)
    values, vectors = np.linalg.eigh(hamiltonian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    levels_hz = values / TAU

    ops = spin_matrices(system.spin)
    operator_components = (ops.ix, ops.iy, ops.iz)
    transitions: list[ESRTransition] = []
    for lower in range(system.dimension):
        for upper in range(lower + 1, system.dimension):
            frequency_hz = float(levels_hz[upper] - levels_hz[lower])
            if frequency_hz <= frequency_tolerance_hz:
                continue
            dipole = np.array(
                [
                    vectors[:, lower].conj().T @ op @ vectors[:, upper]
                    for op in operator_components
                ],
                dtype=np.complex128,
            )
            strength = float(np.linalg.norm(dipole))
            if strength <= strength_tolerance:
                continue
            transitions.append(
                ESRTransition(
                    label=system.label,
                    lower=lower,
                    upper=upper,
                    frequency_hz=frequency_hz,
                    dipole_vector=dipole,
                    strength=strength,
                )
            )

    return ESREigensystem(
        system=system,
        b0_vector_tesla_g=b0,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=tuple(transitions),
    )
