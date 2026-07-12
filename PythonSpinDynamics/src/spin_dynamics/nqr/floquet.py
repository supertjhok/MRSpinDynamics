"""Floquet propagation for monochromatic RF across the NQR--NMR crossover."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.nqr.crossover import boltzmann_populations
from spin_dynamics.nqr.hamiltonians import TAU, nqr_hamiltonian
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import QuadrupolarSite


@dataclass(frozen=True)
class FloquetRFResult:
    """Final state from a finite-sideband monochromatic Floquet calculation."""

    density_matrix: np.ndarray
    physical_propagator: np.ndarray
    floquet_hamiltonian: np.ndarray
    unitarity_error: float
    sidebands: int
    rf_frequency_hz: float
    nutation_hz: float
    pulse_duration_seconds: float
    b0_vector_tesla_pas: np.ndarray
    b1_direction_pas: np.ndarray
    site: QuadrupolarSite


def _unit_real(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(out))
    if not np.all(np.isfinite(out)) or norm <= 0.0:
        raise ValueError(f"{name} must be finite and non-zero")
    return out / norm


def linear_rf_floquet_hamiltonian(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    *,
    nutation_hz: float,
    rf_frequency_hz: float,
    phase_radians: float = 0.0,
    b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
    sidebands: int = 3,
) -> np.ndarray:
    """Return the finite Sambe-space Hamiltonian for a linear cosine field.

    ``nutation_hz`` follows the full NQR model convention: it is half the peak
    laboratory-frame Zeeman modulation, so ``B1_peak = 2*nutation/|gamma|``.
    Blocks are ordered from Fourier index ``-sidebands`` to ``+sidebands``.
    """

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    b1 = _unit_real(b1_direction_pas, "b1_direction_pas")
    nutation = float(nutation_hz)
    carrier = float(rf_frequency_hz)
    phase = float(phase_radians)
    order = int(sidebands)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    if nutation < 0.0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be non-negative and finite")
    if carrier <= 0.0 or not np.isfinite(carrier):
        raise ValueError("rf_frequency_hz must be positive and finite")
    if not np.isfinite(phase):
        raise ValueError("phase_radians must be finite")
    if order < 1:
        raise ValueError("sidebands must be at least one")

    dimension = site.dimension
    indices = np.arange(-order, order + 1)
    static = nqr_hamiltonian(site, b0)
    ops = spin_matrices(site.spin)
    rf_operator = b1[0] * ops.ix + b1[1] * ops.iy + b1[2] * ops.iz
    positive = -TAU * nutation * np.exp(1.0j * phase) * rf_operator
    negative = positive.conj().T
    floquet = np.zeros(
        (indices.size * dimension, indices.size * dimension),
        dtype=np.complex128,
    )
    identity = np.eye(dimension)
    for row, fourier_index in enumerate(indices):
        row_slice = slice(row * dimension, (row + 1) * dimension)
        floquet[row_slice, row_slice] = (
            static + TAU * carrier * fourier_index * identity
        )
        if row > 0:
            column_slice = slice((row - 1) * dimension, row * dimension)
            floquet[row_slice, column_slice] = positive
            floquet[column_slice, row_slice] = negative
    return floquet


def simulate_floquet_rf(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    *,
    nutation_hz: float,
    rf_frequency_hz: float,
    pulse_duration_seconds: float,
    phase_radians: float = 0.0,
    b1_direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
    sidebands: int = 3,
    initial_density: np.ndarray | None = None,
    temperature_kelvin: float = 300.0,
) -> FloquetRFResult:
    """Propagate one constant-envelope linear RF pulse with Floquet sidebands."""

    duration = float(pulse_duration_seconds)
    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError("pulse_duration_seconds must be positive and finite")
    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    b1 = _unit_real(b1_direction_pas, "b1_direction_pas")
    floquet = linear_rf_floquet_hamiltonian(
        site,
        b0,
        nutation_hz=nutation_hz,
        rf_frequency_hz=rf_frequency_hz,
        phase_radians=phase_radians,
        b1_direction_pas=b1,
        sidebands=sidebands,
    )
    static = nqr_hamiltonian(site, b0)
    if initial_density is None:
        values, vectors = np.linalg.eigh(static)
        populations = boltzmann_populations(values / TAU, temperature_kelvin)
        density = (vectors * populations[np.newaxis, :]) @ vectors.conj().T
    else:
        density = np.asarray(initial_density, dtype=np.complex128)
        if density.shape != (site.dimension, site.dimension):
            raise ValueError("initial_density has the wrong shape")
        if not np.allclose(density, density.conj().T):
            raise ValueError("initial_density must be Hermitian")

    extended = propagator(floquet, duration)
    dimension = site.dimension
    order = int(sidebands)
    zero_column = slice(order * dimension, (order + 1) * dimension)
    physical = np.zeros((dimension, dimension), dtype=np.complex128)
    for row, fourier_index in enumerate(range(-order, order + 1)):
        row_slice = slice(row * dimension, (row + 1) * dimension)
        physical += (
            np.exp(1.0j * TAU * rf_frequency_hz * fourier_index * duration)
            * extended[row_slice, zero_column]
        )
    final_density = physical @ density @ physical.conj().T
    unitarity_error = float(
        np.linalg.norm(physical.conj().T @ physical - np.eye(dimension))
    )
    return FloquetRFResult(
        density_matrix=final_density,
        physical_propagator=physical,
        floquet_hamiltonian=floquet,
        unitarity_error=unitarity_error,
        sidebands=order,
        rf_frequency_hz=float(rf_frequency_hz),
        nutation_hz=float(nutation_hz),
        pulse_duration_seconds=duration,
        b0_vector_tesla_pas=b0,
        b1_direction_pas=b1,
        site=site,
    )
