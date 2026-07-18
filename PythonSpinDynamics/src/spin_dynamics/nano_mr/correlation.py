"""Exact ideal-pulse correlation spectroscopy for resolved nano-MR clusters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.coupling.evolution import propagator
from spin_dynamics.nano_mr.exact import (
    ResolvedSpinCluster,
    nuclear_rf_hamiltonian,
    resolved_cluster_hamiltonian,
)
from spin_dynamics.nano_mr.hamiltonians import diagonalize_sensor


def _validate_times(values, *, name: str) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64).reshape(-1)
    if times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError(f"{name} must contain at least two finite points")
    if times[0] < 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing and non-negative")
    return times


def _embed_sensor_matrix(
    cluster: ResolvedSpinCluster,
    sensor_matrix: np.ndarray,
) -> np.ndarray:
    return np.kron(sensor_matrix, np.eye(cluster.nuclear_dimension))


def sensor_transition_rotation(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab,
    lower: int,
    upper: int,
    flip_angle_rad: float,
    *,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Return an ideal selective rotation on two bare sensor eigenstates."""

    lower = int(lower)
    upper = int(upper)
    angle = float(flip_angle_rad)
    phase = float(phase_rad)
    if lower < 0 or upper < 0 or lower == upper:
        raise ValueError("lower and upper must select two distinct sensor levels")
    if max(lower, upper) >= cluster.sensor.dimension:
        raise ValueError("sensor transition index is outside the sensor Hilbert space")
    if not np.isfinite(angle) or not np.isfinite(phase):
        raise ValueError("flip_angle_rad and phase_rad must be finite")
    sensor_eigensystem = diagonalize_sensor(cluster.sensor, b0_vector_tesla_lab)
    rotation_eigen = np.eye(cluster.sensor.dimension, dtype=np.complex128)
    cosine = np.cos(0.5 * angle)
    sine = np.sin(0.5 * angle)
    rotation_eigen[lower, lower] = cosine
    rotation_eigen[upper, upper] = cosine
    rotation_eigen[lower, upper] = -1j * sine * np.exp(-1j * phase)
    rotation_eigen[upper, lower] = -1j * sine * np.exp(1j * phase)
    vectors = sensor_eigensystem.eigenvectors
    rotation_sensor = vectors @ rotation_eigen @ vectors.conj().T
    return _embed_sensor_matrix(cluster, rotation_sensor)


def ideal_nuclear_rotation(
    cluster: ResolvedSpinCluster,
    flip_angle_rad: float,
    *,
    phase_rad: float = 0.0,
    nuclear_indices: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """Return an ideal resonant nuclear rotation for selected resolved spins."""

    angle = float(flip_angle_rad)
    if not np.isfinite(angle):
        raise ValueError("flip_angle_rad must be finite")
    generator = nuclear_rf_hamiltonian(
        cluster,
        1.0 / (2.0 * np.pi),
        phase_rad=phase_rad,
        nuclear_indices=nuclear_indices,
    )
    return propagator(generator, angle)


def _default_transition(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab,
) -> tuple[int, int]:
    eigensystem = diagonalize_sensor(cluster.sensor, b0_vector_tesla_lab)
    if not eigensystem.transitions:
        raise ValueError("sensor has no microwave-active transition at this field")
    transition = max(eigensystem.transitions, key=lambda item: item.strength)
    return transition.lower, transition.upper


@dataclass(frozen=True)
class TwoBlockCorrelationResult:
    """Population-detected two-block Hahn correlation surface."""

    block1_times_seconds: np.ndarray
    block2_times_seconds: np.ndarray
    bright_probability: np.ndarray
    mixing_seconds: float
    sensor_transition: tuple[int, int]
    nuclear_rf_flip_angle_rad: float


def simulate_two_block_correlation(
    cluster: ResolvedSpinCluster,
    b0_vector_tesla_lab,
    block1_times_seconds,
    block2_times_seconds,
    *,
    mixing_seconds: float = 0.0,
    sensor_transition: tuple[int, int] | None = None,
    nuclear_rf_flip_angle_rad: float = 0.0,
    nuclear_rf_phase_rad: float = 0.0,
    nuclear_rf_indices: tuple[int, ...] | list[int] | None = None,
) -> TwoBlockCorrelationResult:
    """Simulate two ideal Hahn sensing blocks separated by coherent mixing.

    Each block applies ``pi/2(y) - t/2 - pi(x) - t/2 - pi/2(-y)`` on one
    bare sensor transition.  The sensor begins in the lower addressed state,
    nuclei begin maximally mixed, and the returned observable is the lower
    sensor-state probability.  An optional ideal nuclear-RF rotation is placed
    at the midpoint of the mixing interval.
    """

    times1 = _validate_times(block1_times_seconds, name="block1_times_seconds")
    times2 = _validate_times(block2_times_seconds, name="block2_times_seconds")
    mixing = float(mixing_seconds)
    rf_angle = float(nuclear_rf_flip_angle_rad)
    if mixing < 0.0 or not np.isfinite(mixing):
        raise ValueError("mixing_seconds must be finite and non-negative")
    if not np.isfinite(rf_angle):
        raise ValueError("nuclear_rf_flip_angle_rad must be finite")
    transition = (
        _default_transition(cluster, b0_vector_tesla_lab)
        if sensor_transition is None
        else (int(sensor_transition[0]), int(sensor_transition[1]))
    )
    lower, upper = transition
    prepare = sensor_transition_rotation(
        cluster,
        b0_vector_tesla_lab,
        lower,
        upper,
        np.pi / 2.0,
        phase_rad=np.pi / 2.0,
    )
    analyze = prepare.conj().T
    refocus = sensor_transition_rotation(
        cluster,
        b0_vector_tesla_lab,
        lower,
        upper,
        np.pi,
        phase_rad=0.0,
    )

    sensor_eigensystem = diagonalize_sensor(cluster.sensor, b0_vector_tesla_lab)
    lower_state = sensor_eigensystem.eigenvectors[:, lower]
    lower_projector_sensor = np.outer(lower_state, lower_state.conj())
    lower_projector = _embed_sensor_matrix(cluster, lower_projector_sensor)
    density0 = lower_projector / cluster.nuclear_dimension

    hamiltonian = resolved_cluster_hamiltonian(cluster, b0_vector_tesla_lab)
    values, vectors = np.linalg.eigh(hamiltonian)

    def free_unitary(duration: float) -> np.ndarray:
        phases = np.exp(-1j * values * duration)
        return (vectors * phases[np.newaxis, :]) @ vectors.conj().T

    def block_unitary(duration: float) -> np.ndarray:
        free_half = free_unitary(0.5 * duration)
        return analyze @ free_half @ refocus @ free_half @ prepare

    block1 = tuple(block_unitary(float(duration)) for duration in times1)
    block2 = tuple(block_unitary(float(duration)) for duration in times2)
    free_mix_half = free_unitary(0.5 * mixing)
    rf_rotation = ideal_nuclear_rotation(
        cluster,
        rf_angle,
        phase_rad=nuclear_rf_phase_rad,
        nuclear_indices=nuclear_rf_indices,
    )
    mixing_unitary = free_mix_half @ rf_rotation @ free_mix_half

    signal = np.empty((times1.size, times2.size), dtype=np.float64)
    for row, unitary1 in enumerate(block1):
        density1 = unitary1 @ density0 @ unitary1.conj().T
        mixed = mixing_unitary @ density1 @ mixing_unitary.conj().T
        for column, unitary2 in enumerate(block2):
            density2 = unitary2 @ mixed @ unitary2.conj().T
            signal[row, column] = float(
                np.real(np.trace(lower_projector @ density2))
            )
    signal = np.clip(signal, 0.0, 1.0)
    return TwoBlockCorrelationResult(
        block1_times_seconds=times1,
        block2_times_seconds=times2,
        bright_probability=signal,
        mixing_seconds=mixing,
        sensor_transition=transition,
        nuclear_rf_flip_angle_rad=rf_angle,
    )


@dataclass(frozen=True)
class CorrelationSpectrum2D:
    """Two-dimensional Fourier magnitude of a correlation surface."""

    frequencies1_hz: np.ndarray
    frequencies2_hz: np.ndarray
    amplitude: np.ndarray


def correlation_spectrum_2d(
    result: TwoBlockCorrelationResult,
    *,
    window: bool = True,
    remove_mean: bool = True,
) -> CorrelationSpectrum2D:
    """Fourier transform a uniformly sampled two-block correlation surface."""

    times1 = result.block1_times_seconds
    times2 = result.block2_times_seconds
    spacing1 = np.diff(times1)
    spacing2 = np.diff(times2)
    if not np.allclose(spacing1, spacing1[0], rtol=1.0e-8, atol=1.0e-15):
        raise ValueError("block1_times_seconds must be uniformly sampled")
    if not np.allclose(spacing2, spacing2[0], rtol=1.0e-8, atol=1.0e-15):
        raise ValueError("block2_times_seconds must be uniformly sampled")
    values = np.asarray(result.bright_probability, dtype=np.float64).copy()
    if remove_mean:
        values -= np.mean(values)
    if window:
        values *= np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transform = np.fft.fftshift(np.fft.fft2(values))
    frequency1 = np.fft.fftshift(
        np.fft.fftfreq(times1.size, d=float(spacing1[0]))
    )
    frequency2 = np.fft.fftshift(
        np.fft.fftfreq(times2.size, d=float(spacing2[0]))
    )
    return CorrelationSpectrum2D(
        frequencies1_hz=frequency1,
        frequencies2_hz=frequency2,
        amplitude=np.abs(transform),
    )


__all__ = [
    "CorrelationSpectrum2D",
    "TwoBlockCorrelationResult",
    "correlation_spectrum_2d",
    "ideal_nuclear_rotation",
    "sensor_transition_rotation",
    "simulate_two_block_correlation",
]
