"""Laboratory-frame RF propagation for quadrupolar crossover systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.nqr.crossover import boltzmann_populations
from spin_dynamics.nqr.hamiltonians import TAU, nqr_hamiltonian, zeeman_hamiltonian
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.systems import QuadrupolarSite


@dataclass(frozen=True)
class LabFrameRFResult:
    """Density trajectory and detected signal after each RF waveform segment."""

    times_seconds: np.ndarray
    density_matrices: np.ndarray
    signal: np.ndarray
    static_hamiltonian: np.ndarray
    segment_durations_seconds: np.ndarray
    rf_fields_tesla_pas: np.ndarray
    receive_direction_pas: np.ndarray
    site: QuadrupolarSite


def sample_linear_rf_pulse(
    duration_seconds: float,
    time_step_seconds: float,
    amplitude_tesla: float,
    carrier_hz: float,
    *,
    phase_radians: float = 0.0,
    direction_pas: Sequence[float] | np.ndarray = (1.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a linearly polarized cosine RF pulse at segment midpoints."""

    duration = float(duration_seconds)
    time_step = float(time_step_seconds)
    amplitude = float(amplitude_tesla)
    carrier = float(carrier_hz)
    phase = float(phase_radians)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_seconds must be positive and finite")
    if not np.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step_seconds must be positive and finite")
    if not np.isfinite(amplitude) or not np.isfinite(carrier) or not np.isfinite(phase):
        raise ValueError("RF amplitude, carrier, and phase must be finite")
    direction = np.asarray(direction_pas, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or norm <= 0.0:
        raise ValueError("direction_pas must be finite and non-zero")
    direction /= norm

    count = int(np.ceil(duration / time_step))
    edges = np.linspace(0.0, duration, count + 1)
    durations = np.diff(edges)
    midpoints = edges[:-1] + 0.5 * durations
    envelope = amplitude * np.cos(TAU * carrier * midpoints + phase)
    return durations, envelope[:, np.newaxis] * direction[np.newaxis, :]


def _equilibrium_density_pas(
    hamiltonian: np.ndarray,
    temperature_kelvin: float,
) -> np.ndarray:
    values, vectors = np.linalg.eigh(hamiltonian)
    populations = boltzmann_populations(values / TAU, temperature_kelvin)
    return (vectors * populations[np.newaxis, :]) @ vectors.conj().T


def simulate_lab_frame_rf(
    site: QuadrupolarSite,
    b0_vector_tesla_pas: Sequence[float] | np.ndarray,
    segment_durations_seconds: Sequence[float] | np.ndarray,
    rf_fields_tesla_pas: Sequence[Sequence[float]] | np.ndarray,
    *,
    initial_density: np.ndarray | None = None,
    temperature_kelvin: float = 300.0,
    receive_direction_pas: Sequence[complex] | np.ndarray = (1.0, 0.0, 0.0),
) -> LabFrameRFResult:
    """Propagate an arbitrary sampled RF waveform without an RWA.

    ``rf_fields_tesla_pas[k]`` is the instantaneous laboratory-frame magnetic
    field during segment ``k``. The waveform must resolve the carrier and any
    envelope variation; :func:`sample_linear_rf_pulse` provides midpoint
    sampling for a simple cosine pulse.
    """

    b0 = np.asarray(b0_vector_tesla_pas, dtype=np.float64).reshape(3)
    durations = np.asarray(segment_durations_seconds, dtype=np.float64).reshape(-1)
    rf_fields = np.asarray(rf_fields_tesla_pas, dtype=np.float64).reshape(-1, 3)
    if not np.all(np.isfinite(b0)):
        raise ValueError("b0_vector_tesla_pas must be finite")
    if durations.size == 0 or rf_fields.shape[0] != durations.size:
        raise ValueError("RF fields must provide one vector per non-empty duration")
    if not np.all(np.isfinite(durations)) or np.any(durations <= 0.0):
        raise ValueError("segment durations must be positive and finite")
    if not np.all(np.isfinite(rf_fields)):
        raise ValueError("rf_fields_tesla_pas must be finite")

    static_hamiltonian = nqr_hamiltonian(site, b0)
    if initial_density is None:
        density = _equilibrium_density_pas(static_hamiltonian, temperature_kelvin)
    else:
        density = np.asarray(initial_density, dtype=np.complex128)
        if density.shape != (site.dimension, site.dimension):
            raise ValueError("initial_density has the wrong shape")
        if not np.allclose(density, density.conj().T):
            raise ValueError("initial_density must be Hermitian")

    receive = np.asarray(receive_direction_pas, dtype=np.complex128).reshape(3)
    receive_norm = float(np.linalg.norm(receive))
    if not np.all(np.isfinite(receive)) or receive_norm <= 0.0:
        raise ValueError("receive_direction_pas must be finite and non-zero")
    receive /= receive_norm
    ops = spin_matrices(site.spin)
    receive_operator = sum(
        np.conj(component) * operator
        for component, operator in zip(receive, (ops.ix, ops.iy, ops.iz))
    )

    densities = np.empty(
        (durations.size + 1, site.dimension, site.dimension),
        dtype=np.complex128,
    )
    signal = np.empty(durations.size + 1, dtype=np.complex128)
    densities[0] = density
    signal[0] = np.trace(density @ receive_operator)
    for index, (segment_duration, rf_field) in enumerate(zip(durations, rf_fields)):
        hamiltonian = static_hamiltonian + zeeman_hamiltonian(site, rf_field)
        density = evolve_density(density, hamiltonian, float(segment_duration))
        densities[index + 1] = density
        signal[index + 1] = np.trace(density @ receive_operator)

    times = np.concatenate(([0.0], np.cumsum(durations)))
    return LabFrameRFResult(
        times_seconds=times,
        density_matrices=densities,
        signal=signal,
        static_hamiltonian=static_hamiltonian,
        segment_durations_seconds=durations,
        rf_fields_tesla_pas=rf_fields,
        receive_direction_pas=receive,
        site=site,
    )
