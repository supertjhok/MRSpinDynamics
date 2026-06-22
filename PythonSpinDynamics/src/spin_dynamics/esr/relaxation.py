"""Liouville-space relaxation helpers for pulsed ESR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ESRRelaxationModel:
    """Phenomenological ESR relaxation model in the energy eigenbasis.

    ``t1_seconds`` damps population differences while preserving trace.
    ``t2_seconds`` damps coherences. Both act on the trace-zero
    high-temperature density-matrix deviation used by the ESR pulse helpers.
    """

    t1_seconds: float = np.inf
    t2_seconds: float = np.inf

    def __post_init__(self) -> None:
        t1_seconds = float(self.t1_seconds)
        t2_seconds = float(self.t2_seconds)
        if not np.isfinite(t1_seconds) and not np.isinf(t1_seconds):
            raise ValueError("t1_seconds must be positive or infinite")
        if not np.isfinite(t2_seconds) and not np.isinf(t2_seconds):
            raise ValueError("t2_seconds must be positive or infinite")
        if t1_seconds <= 0:
            raise ValueError("t1_seconds must be positive or infinite")
        if t2_seconds <= 0:
            raise ValueError("t2_seconds must be positive or infinite")
        object.__setattr__(self, "t1_seconds", t1_seconds)
        object.__setattr__(self, "t2_seconds", t2_seconds)


def matrix_exponential(matrix: np.ndarray, duration: float = 1.0) -> np.ndarray:
    """Return ``exp(matrix * duration)`` for a small dense matrix."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    duration = float(duration)
    if duration < 0 or not np.isfinite(duration):
        raise ValueError("duration must be non-negative and finite")
    if duration == 0:
        return np.eye(matrix.shape[0], dtype=np.complex128)
    values, vectors = np.linalg.eig(matrix)
    return (vectors * np.exp(values * duration)[np.newaxis, :]) @ np.linalg.inv(vectors)


def liouville_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    """Return the commutator Liouvillian for column-stacked density matrices."""

    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    dim = hamiltonian.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    return -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))


def relaxation_superoperator(
    dimension: int,
    model: ESRRelaxationModel,
) -> np.ndarray:
    """Return a trace-preserving ESR relaxation superoperator."""

    dimension = int(dimension)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    size = dimension * dimension
    out = np.zeros((size, size), dtype=np.complex128)

    if np.isfinite(model.t1_seconds):
        rate = 1.0 / model.t1_seconds
        for row in range(dimension):
            row_index = row + row * dimension
            for col in range(dimension):
                col_index = col + col * dimension
                out[row_index, col_index] += rate / dimension
            out[row_index, row_index] -= rate

    if np.isfinite(model.t2_seconds):
        rate = 1.0 / model.t2_seconds
        for row in range(dimension):
            for col in range(dimension):
                if row == col:
                    continue
                out[row + col * dimension, row + col * dimension] -= rate

    return out


def liouville_superoperator(
    hamiltonian: np.ndarray,
    model: ESRRelaxationModel | None = None,
) -> np.ndarray:
    """Return Hamiltonian plus optional ESR relaxation Liouvillian."""

    out = liouville_hamiltonian(hamiltonian)
    if model is not None:
        out = out + relaxation_superoperator(hamiltonian.shape[0], model)
    return out


def propagate_density_liouville(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    duration: float,
    *,
    relaxation: ESRRelaxationModel | None = None,
) -> np.ndarray:
    """Propagate a density matrix with Hamiltonian and ESR relaxation."""

    density = np.asarray(density, dtype=np.complex128)
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density must be square")
    superoperator = matrix_exponential(
        liouville_superoperator(hamiltonian, relaxation),
        duration,
    )
    vector = density.reshape(-1, order="F")
    return (superoperator @ vector).reshape(density.shape, order="F")
