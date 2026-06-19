"""Dense density-matrix evolution for small coupled spin systems."""

from __future__ import annotations

import numpy as np

from spin_dynamics.coupling.operators import total_operator
from spin_dynamics.coupling.systems import CoupledSpinSystem


def propagator(hamiltonian: np.ndarray, duration: float) -> np.ndarray:
    """Return ``exp(-i H duration)`` for a Hermitian Hamiltonian."""

    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    if not np.allclose(hamiltonian, hamiltonian.conj().T):
        raise ValueError("hamiltonian must be Hermitian")
    values, vectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1j * values * float(duration))
    return (vectors * phases[np.newaxis, :]) @ vectors.conj().T


def evolve_density(density: np.ndarray, hamiltonian: np.ndarray, duration: float) -> np.ndarray:
    """Evolve a density operator under a time-independent Hamiltonian."""

    density = np.asarray(density, dtype=np.complex128)
    unitary = propagator(hamiltonian, duration)
    return unitary @ density @ unitary.conj().T


def propagate_density(
    density: np.ndarray,
    steps: list[tuple[np.ndarray, float]] | tuple[tuple[np.ndarray, float], ...],
) -> np.ndarray:
    """Evolve a density operator through a sequence of Hamiltonian steps."""

    out = np.asarray(density, dtype=np.complex128)
    for hamiltonian, duration in steps:
        out = evolve_density(out, hamiltonian, duration)
    return out


def equilibrium_density(system: CoupledSpinSystem, axis: str = "z") -> np.ndarray:
    """Return a high-temperature equilibrium density operator."""

    return total_operator(system.nspin, axis)
