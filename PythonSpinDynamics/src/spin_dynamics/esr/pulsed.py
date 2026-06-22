"""Pulsed ESR density-matrix helpers for single-electron systems."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from spin_dynamics.coupling.evolution import evolve_density
from spin_dynamics.esr.hamiltonians import TAU, diagonalize_system
from spin_dynamics.esr.relaxation import (
    ESRRelaxationModel,
    propagate_density_liouville,
)
from spin_dynamics.esr.systems import ESREigensystem, ESRSpinSystem
from spin_dynamics.nqr.operators import spin_matrices


def _unit(direction) -> np.ndarray:
    vec = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("direction must be a finite non-zero vector")
    return vec / norm


def equilibrium_density(levels_hz: np.ndarray) -> np.ndarray:
    """Return a trace-zero high-temperature ESR density matrix."""

    levels = np.asarray(levels_hz, dtype=np.float64).reshape(-1)
    populations = -(levels - np.mean(levels))
    scale = float(np.max(np.abs(populations))) if populations.size else 0.0
    if scale > 0:
        populations = populations / scale
    return np.diag(populations.astype(np.complex128))


def flip_angle_duration(flip_angle_rad: float, nutation_hz: float) -> float:
    """Return the rectangular-pulse duration for a spin-1/2 flip angle."""

    flip_angle = float(flip_angle_rad)
    nutation = float(nutation_hz)
    if flip_angle < 0 or not np.isfinite(flip_angle):
        raise ValueError("flip_angle_rad must be non-negative and finite")
    if nutation <= 0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be positive and finite")
    return flip_angle / (TAU * nutation)


def rf_operator_eigenbasis(
    eigensystem: ESREigensystem,
    direction_g=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return ``e1 . S`` for a unit microwave-field direction in the eigenbasis."""

    ops = spin_matrices(eigensystem.system.spin)
    e1 = _unit(direction_g)
    lab = e1[0] * ops.ix + e1[1] * ops.iy + e1[2] * ops.iz
    vectors = eigensystem.eigenvectors
    return vectors.conj().T @ lab @ vectors


def rotating_indices(levels_hz: np.ndarray, rf_frequency_hz: float) -> np.ndarray:
    """Return two-level RWA winding numbers for a carrier frequency."""

    levels_hz = np.asarray(levels_hz, dtype=np.float64).reshape(-1)
    rf_frequency_hz = float(rf_frequency_hz)
    if rf_frequency_hz <= 0 or not np.isfinite(rf_frequency_hz):
        raise ValueError("rf_frequency_hz must be positive and finite")
    return np.round((levels_hz - levels_hz.min()) / rf_frequency_hz).astype(np.int64)


def static_hamiltonian_rotating(
    eigensystem: ESREigensystem,
    rf_frequency_hz: float,
) -> np.ndarray:
    """Return the rotating-frame static Hamiltonian in radians per second."""

    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    diagonal = TAU * (eigensystem.levels_hz - float(rf_frequency_hz) * indices)
    return np.diag(diagonal).astype(np.complex128)


def pulse_hamiltonian(
    eigensystem: ESREigensystem,
    *,
    nutation_hz: float,
    rf_frequency_hz: float,
    phase: float = 0.0,
    b1_direction_g=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return a rectangular microwave-pulse Hamiltonian in the rotating frame.

    ``nutation_hz`` is the on-resonance spin-1/2 Rabi rate for the selected
    microwave-field direction. A 90-degree pulse therefore has duration
    ``1 / (4 * nutation_hz)``.
    """

    nutation = float(nutation_hz)
    if nutation < 0 or not np.isfinite(nutation):
        raise ValueError("nutation_hz must be finite and non-negative")
    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    hamiltonian = static_hamiltonian_rotating(eigensystem, rf_frequency_hz)
    rf_operator = rf_operator_eigenbasis(eigensystem, b1_direction_g)
    amplitude = TAU * nutation
    phase = float(phase)
    delta = indices[:, None] - indices[None, :]
    coupling = np.zeros_like(rf_operator)
    upper = delta == 1
    lower = delta == -1
    coupling[upper] = -amplitude * rf_operator[upper] * np.exp(-1j * phase)
    coupling[lower] = -amplitude * rf_operator[lower] * np.exp(1j * phase)
    return hamiltonian + coupling


def detection_operator(
    eigensystem: ESREigensystem,
    rf_frequency_hz: float,
    rx_direction_g=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the baseband receive observable for the addressed ESR line."""

    indices = rotating_indices(eigensystem.levels_hz, rf_frequency_hz)
    rx_operator = rf_operator_eigenbasis(eigensystem, rx_direction_g)
    detector = np.zeros_like(rx_operator)
    delta = indices[:, None] - indices[None, :]
    raising = delta == 1
    detector.T[raising] = rx_operator.T[raising]
    return detector


@dataclass(frozen=True)
class ESRFIDResult:
    """Complex baseband ESR FID from one rectangular excitation pulse."""

    times_seconds: np.ndarray
    signal: np.ndarray
    rf_frequency_hz: float
    eigensystem: ESREigensystem


@dataclass(frozen=True)
class ESRHahnEchoResult:
    """Complex baseband ESR Hahn echo from one isochromat."""

    times_seconds: np.ndarray
    signal: np.ndarray
    rf_frequency_hz: float
    eigensystem: ESREigensystem
    echo_center_seconds: float


def _validate_times(times_seconds) -> np.ndarray:
    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("times_seconds must not be empty")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_seconds must be finite")
    if np.any(np.diff(times) < 0):
        raise ValueError("times_seconds must be non-decreasing")
    if times[0] < 0:
        raise ValueError("times_seconds must be non-negative")
    return times


def _default_carrier_hz(eigensystem: ESREigensystem) -> float:
    if not eigensystem.transitions:
        raise ValueError("system has no RF-active ESR transitions")
    return float(max(eigensystem.transitions, key=lambda t: t.strength).frequency_hz)


def _propagate(
    density: np.ndarray,
    hamiltonian: np.ndarray,
    duration: float,
    relaxation: ESRRelaxationModel | None = None,
) -> np.ndarray:
    if duration <= 0:
        return density
    if relaxation is not None:
        return propagate_density_liouville(
            density,
            hamiltonian,
            duration,
            relaxation=relaxation,
        )
    return evolve_density(density, hamiltonian, duration)


def _sample_signal(
    density: np.ndarray,
    free_hamiltonian: np.ndarray,
    detector: np.ndarray,
    times: np.ndarray,
    *,
    t2_seconds: float,
    relaxation: ESRRelaxationModel | None,
) -> np.ndarray:
    signal = np.empty(times.size, dtype=np.complex128)
    current = 0.0
    rho = density
    for idx, sample_time in enumerate(times):
        rho = _propagate(
            rho,
            free_hamiltonian,
            float(sample_time) - current,
            relaxation,
        )
        current = float(sample_time)
        signal[idx] = np.trace(rho @ detector)
    if np.isfinite(t2_seconds):
        signal = signal * np.exp(-times / t2_seconds)
    return signal


def simulate_fid(
    system: ESRSpinSystem,
    b0_vector_tesla_g,
    *,
    nutation_hz: float,
    pulse_duration_seconds: float,
    times_seconds,
    rf_frequency_hz: float | None = None,
    phase: float = 0.0,
    b1_direction_g=(1.0, 0.0, 0.0),
    rx_direction_g=None,
    t2_seconds: float = np.inf,
    relaxation: ESRRelaxationModel | None = None,
    initial_density: np.ndarray | None = None,
) -> ESRFIDResult:
    """Simulate a pulsed ESR free-induction decay in the rotating frame."""

    times = _validate_times(times_seconds)
    t2_seconds = float(t2_seconds)
    if t2_seconds <= 0:
        raise ValueError("t2_seconds must be positive or infinite")
    _warn_double_t2(t2_seconds, relaxation)
    eigensystem = diagonalize_system(system, b0_vector_tesla_g)
    carrier = _default_carrier_hz(eigensystem) if rf_frequency_hz is None else float(
        rf_frequency_hz
    )
    rho = (
        equilibrium_density(eigensystem.levels_hz)
        if initial_density is None
        else np.asarray(initial_density, dtype=np.complex128).copy()
    )
    pulse = pulse_hamiltonian(
        eigensystem,
        nutation_hz=nutation_hz,
        rf_frequency_hz=carrier,
        phase=phase,
        b1_direction_g=b1_direction_g,
    )
    rho = _propagate(rho, pulse, float(pulse_duration_seconds), relaxation)
    free = static_hamiltonian_rotating(eigensystem, carrier)
    detector = detection_operator(
        eigensystem,
        carrier,
        b1_direction_g if rx_direction_g is None else rx_direction_g,
    )
    return ESRFIDResult(
        times_seconds=times,
        signal=_sample_signal(
            rho,
            free,
            detector,
            times,
            t2_seconds=t2_seconds,
            relaxation=relaxation,
        ),
        rf_frequency_hz=carrier,
        eigensystem=eigensystem,
    )


def simulate_hahn_echo(
    system: ESRSpinSystem,
    b0_vector_tesla_g,
    *,
    nutation_hz: float,
    excitation_duration_seconds: float,
    refocus_duration_seconds: float,
    tau_seconds: float,
    times_seconds,
    rf_frequency_hz: float | None = None,
    excitation_phase: float = 0.0,
    refocus_phase: float = np.pi / 2.0,
    b1_direction_g=(1.0, 0.0, 0.0),
    rx_direction_g=None,
    t2_seconds: float = np.inf,
    relaxation: ESRRelaxationModel | None = None,
    initial_density: np.ndarray | None = None,
) -> ESRHahnEchoResult:
    """Simulate a two-pulse ESR Hahn echo for one isochromat.

    ``times_seconds`` are measured from the end of the refocusing pulse, so the
    echo center is near ``tau_seconds``.
    """

    tau = float(tau_seconds)
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be positive and finite")
    times = _validate_times(times_seconds)
    t2_seconds = float(t2_seconds)
    if t2_seconds <= 0:
        raise ValueError("t2_seconds must be positive or infinite")
    _warn_double_t2(t2_seconds, relaxation)
    eigensystem = diagonalize_system(system, b0_vector_tesla_g)
    carrier = _default_carrier_hz(eigensystem) if rf_frequency_hz is None else float(
        rf_frequency_hz
    )
    rho = (
        equilibrium_density(eigensystem.levels_hz)
        if initial_density is None
        else np.asarray(initial_density, dtype=np.complex128).copy()
    )
    free = static_hamiltonian_rotating(eigensystem, carrier)
    excite = pulse_hamiltonian(
        eigensystem,
        nutation_hz=nutation_hz,
        rf_frequency_hz=carrier,
        phase=excitation_phase,
        b1_direction_g=b1_direction_g,
    )
    refocus = pulse_hamiltonian(
        eigensystem,
        nutation_hz=nutation_hz,
        rf_frequency_hz=carrier,
        phase=refocus_phase,
        b1_direction_g=b1_direction_g,
    )
    rho = _propagate(rho, excite, float(excitation_duration_seconds), relaxation)
    rho = _propagate(rho, free, tau, relaxation)
    rho = _propagate(rho, refocus, float(refocus_duration_seconds), relaxation)
    detector = detection_operator(
        eigensystem,
        carrier,
        b1_direction_g if rx_direction_g is None else rx_direction_g,
    )
    signal = _sample_signal(
        rho,
        free,
        detector,
        times,
        t2_seconds=t2_seconds,
        relaxation=relaxation,
    )
    if np.isfinite(t2_seconds):
        signal = signal * np.exp(-tau / t2_seconds)
    return ESRHahnEchoResult(
        times_seconds=times,
        signal=signal,
        rf_frequency_hz=carrier,
        eigensystem=eigensystem,
        echo_center_seconds=tau,
    )


def _warn_double_t2(
    t2_seconds: float,
    relaxation: ESRRelaxationModel | None,
) -> None:
    if relaxation is not None and np.isfinite(t2_seconds):
        warnings.warn(
            "both a finite t2_seconds envelope and an ESRRelaxationModel were "
            "given; their coherence damping composes multiplicatively. Pass "
            "t2_seconds=inf when using relaxation to avoid double-counting T2.",
            RuntimeWarning,
            stacklevel=3,
        )
