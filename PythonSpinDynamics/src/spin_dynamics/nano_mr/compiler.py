"""Effective addressed-qubit compilation and propagation for sensor control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.nano_mr.sequences import SensingSequence, TimedControlPulse


_IDENTITY = np.eye(2, dtype=np.complex128)
_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


@dataclass(frozen=True)
class CompiledQubitStep:
    """One free, finite-pulse, or instantaneous qubit operation."""

    start_seconds: float
    duration_seconds: float
    control_hamiltonian_rad_s: np.ndarray
    instantaneous_unitary: np.ndarray | None
    label: str

    @property
    def kind(self) -> str:
        """Operation kind: ``free``, ``pulse``, or ``instantaneous``."""

        if self.instantaneous_unitary is not None:
            return "instantaneous"
        if np.any(self.control_hamiltonian_rad_s):
            return "pulse"
        return "free"


@dataclass(frozen=True)
class CompiledQubitSequence:
    """Piecewise addressed-qubit representation of a sensing sequence."""

    source: SensingSequence
    steps: tuple[CompiledQubitStep, ...]


@dataclass(frozen=True)
class QubitPropagationResult:
    """Final addressed-qubit density matrix and Bloch vector."""

    density: np.ndarray
    bloch_vector: np.ndarray
    coherence: complex
    elapsed_seconds: float


def rotation_unitary(flip_angle_rad: float, phase_rad: float) -> np.ndarray:
    """Return an ideal transverse qubit rotation."""

    axis = np.cos(phase_rad) * _SIGMA_X + np.sin(phase_rad) * _SIGMA_Y
    half = 0.5 * float(flip_angle_rad)
    return np.cos(half) * _IDENTITY - 1.0j * np.sin(half) * axis


def compile_qubit_sequence(sequence: SensingSequence) -> CompiledQubitSequence:
    """Compile microwave events into free and addressed-qubit control steps.

    Nuclear-RF events are preserved in the source schedule but do not act on
    the addressed sensor qubit in this addressed-qubit compiler.
    """

    steps: list[CompiledQubitStep] = []
    cursor = 0.0
    for pulse in sequence.electron_pulses:
        if pulse.start_seconds > cursor:
            steps.append(_free_step(cursor, pulse.start_seconds - cursor))
        if pulse.is_instantaneous:
            steps.append(
                CompiledQubitStep(
                    start_seconds=pulse.center_seconds,
                    duration_seconds=0.0,
                    control_hamiltonian_rad_s=np.zeros((2, 2), dtype=np.complex128),
                    instantaneous_unitary=rotation_unitary(
                        pulse.flip_angle_rad, pulse.phase_rad
                    ),
                    label=pulse.label or "instantaneous pulse",
                )
            )
            cursor = pulse.center_seconds
        else:
            steps.append(_finite_pulse_step(pulse))
            cursor = pulse.end_seconds
    if cursor < sequence.total_duration_seconds:
        steps.append(_free_step(cursor, sequence.total_duration_seconds - cursor))
    return CompiledQubitSequence(source=sequence, steps=tuple(steps))


def propagate_controlled_qubit(
    sequence: SensingSequence,
    detuning_rad_per_s: float | Callable[[float], float] = 0.0,
    *,
    initial_density: np.ndarray | None = None,
    max_step_seconds: float | None = None,
) -> QubitPropagationResult:
    """Propagate an addressed sensor qubit through control and detuning.

    The rotating-frame Hamiltonian is
    ``H = detuning(t)*sigma_z/2 + H_control``. A callable detuning is sampled
    at substep midpoints. ``max_step_seconds`` controls time discretization;
    constant detuning and control need only one substep per compiled interval.
    When ``initial_density`` is omitted, ``sequence.preparation_phase_rad``
    sets the initial equatorial Bloch-vector phase. Returned density,
    Bloch-vector, and coherence values are expressed in the transverse frame
    selected by ``sequence.readout_phase_rad``.
    """

    density = (
        _equatorial_density(sequence.preparation_phase_rad)
        if initial_density is None
        else _validated_density(initial_density)
    )
    if max_step_seconds is not None:
        max_step_seconds = float(max_step_seconds)
        if max_step_seconds <= 0.0 or not np.isfinite(max_step_seconds):
            raise ValueError("max_step_seconds must be positive and finite")
    compiled = compile_qubit_sequence(sequence)
    for step in compiled.steps:
        if step.instantaneous_unitary is not None:
            unitary = step.instantaneous_unitary
            density = unitary @ density @ unitary.conj().T
            continue
        count = _substep_count(
            step.duration_seconds,
            max_step_seconds=max_step_seconds,
            time_dependent=callable(detuning_rad_per_s),
        )
        dt = step.duration_seconds / count
        for index in range(count):
            midpoint = step.start_seconds + (index + 0.5) * dt
            detuning = _detuning_value(detuning_rad_per_s, midpoint)
            hamiltonian = step.control_hamiltonian_rad_s + 0.5 * detuning * _SIGMA_Z
            unitary = _unitary_from_hermitian(hamiltonian, dt)
            density = unitary @ density @ unitary.conj().T
    readout_frame = _z_rotation_unitary(-sequence.readout_phase_rad)
    density = readout_frame @ density @ readout_frame.conj().T
    bloch = np.array(
        [
            np.real(np.trace(density @ _SIGMA_X)),
            np.real(np.trace(density @ _SIGMA_Y)),
            np.real(np.trace(density @ _SIGMA_Z)),
        ],
        dtype=np.float64,
    )
    return QubitPropagationResult(
        density=density,
        bloch_vector=bloch,
        coherence=complex(bloch[0] + 1.0j * bloch[1]),
        elapsed_seconds=sequence.total_duration_seconds,
    )


def _equatorial_density(phase_rad: float) -> np.ndarray:
    return 0.5 * (
        _IDENTITY + np.cos(phase_rad) * _SIGMA_X + np.sin(phase_rad) * _SIGMA_Y
    )


def _z_rotation_unitary(angle_rad: float) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    return np.cos(half) * _IDENTITY - 1.0j * np.sin(half) * _SIGMA_Z


def _free_step(start: float, duration: float) -> CompiledQubitStep:
    return CompiledQubitStep(
        start_seconds=float(start),
        duration_seconds=float(duration),
        control_hamiltonian_rad_s=np.zeros((2, 2), dtype=np.complex128),
        instantaneous_unitary=None,
        label="free",
    )


def _finite_pulse_step(pulse: TimedControlPulse) -> CompiledQubitStep:
    angular_rate = pulse.flip_angle_rad / pulse.duration_seconds
    axis = np.cos(pulse.phase_rad) * _SIGMA_X + np.sin(pulse.phase_rad) * _SIGMA_Y
    return CompiledQubitStep(
        start_seconds=pulse.start_seconds,
        duration_seconds=pulse.duration_seconds,
        control_hamiltonian_rad_s=0.5 * angular_rate * axis,
        instantaneous_unitary=None,
        label=pulse.label or "finite pulse",
    )


def _substep_count(
    duration: float,
    *,
    max_step_seconds: float | None,
    time_dependent: bool,
) -> int:
    if max_step_seconds is None:
        return 64 if time_dependent else 1
    return max(1, int(np.ceil(duration / max_step_seconds)))


def _detuning_value(
    detuning: float | Callable[[float], float],
    time_seconds: float,
) -> float:
    value = float(detuning(time_seconds) if callable(detuning) else detuning)
    if not np.isfinite(value):
        raise ValueError("detuning_rad_per_s must return finite values")
    return value


def _unitary_from_hermitian(hamiltonian: np.ndarray, duration: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(hamiltonian)
    phases = np.exp(-1.0j * values * duration)
    return (vectors * phases[np.newaxis, :]) @ vectors.conj().T


def _validated_density(values) -> np.ndarray:
    density = np.asarray(values, dtype=np.complex128)
    if density.shape != (2, 2):
        raise ValueError("initial_density must be a 2x2 matrix")
    if not np.allclose(density, density.conj().T, atol=1.0e-12):
        raise ValueError("initial_density must be Hermitian")
    if not np.isclose(np.trace(density), 1.0, atol=1.0e-12):
        raise ValueError("initial_density must have unit trace")
    if np.min(np.linalg.eigvalsh(density)) < -1.0e-12:
        raise ValueError("initial_density must be positive semidefinite")
    return density.copy()


__all__ = [
    "CompiledQubitSequence",
    "CompiledQubitStep",
    "QubitPropagationResult",
    "compile_qubit_sequence",
    "propagate_controlled_qubit",
    "rotation_unitary",
]
