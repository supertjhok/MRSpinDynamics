"""Ground-state defect Hamiltonians and ODMR transition analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.esr.systems import BOHR_MAGNETON_HZ_PER_T
from spin_dynamics.nano_mr.sensors import DefectSpinSensor
from spin_dynamics.nqr.operators import spin_matrices


TAU = 2.0 * np.pi


def _vector(values, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def sensor_spin_operators(sensor: DefectSpinSensor) -> tuple[np.ndarray, ...]:
    """Return local ``(Sx, Sy, Sz)`` operators for the defect ground state."""

    operators = spin_matrices(sensor.spin)
    return operators.ix, operators.iy, operators.iz


def zfs_hamiltonian(sensor: DefectSpinSensor) -> np.ndarray:
    """Return the zero-field-splitting Hamiltonian in radians per second."""

    operators = sensor_spin_operators(sensor)
    hamiltonian_hz = np.zeros(
        (sensor.dimension, sensor.dimension), dtype=np.complex128
    )
    for row, operator_row in enumerate(operators):
        for column, operator_column in enumerate(operators):
            coefficient = sensor.zfs_tensor_hz[row, column]
            if coefficient:
                product = 0.5 * (
                    operator_row @ operator_column
                    + operator_column @ operator_row
                )
                hamiltonian_hz = hamiltonian_hz + coefficient * product
    return TAU * hamiltonian_hz


def zeeman_hamiltonian(
    sensor: DefectSpinSensor,
    b0_vector_tesla_lab,
) -> np.ndarray:
    """Return the electron Zeeman Hamiltonian in radians per second."""

    b0_lab = _vector(b0_vector_tesla_lab, name="b0_vector_tesla_lab")
    b0_local = sensor.frame.vector_to_local(b0_lab)
    effective_field = sensor.g_tensor.T @ b0_local
    operators = sensor_spin_operators(sensor)
    frequency_operator = sum(
        component * operator
        for component, operator in zip(effective_field, operators)
    )
    return TAU * BOHR_MAGNETON_HZ_PER_T * frequency_operator


def sensor_hamiltonian(
    sensor: DefectSpinSensor,
    b0_vector_tesla_lab=(0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the ground-state ZFS plus Zeeman Hamiltonian."""

    return zfs_hamiltonian(sensor) + zeeman_hamiltonian(
        sensor, b0_vector_tesla_lab
    )


@dataclass(frozen=True)
class DefectTransition:
    """One microwave-active transition of a defect ground state."""

    lower: int
    upper: int
    frequency_hz: float
    strength: float


@dataclass(frozen=True)
class DefectEigensystem:
    """Defect energy levels and microwave-active transitions at one field."""

    sensor: DefectSpinSensor
    b0_vector_tesla_lab: np.ndarray
    levels_hz: np.ndarray
    eigenvectors: np.ndarray
    transitions: tuple[DefectTransition, ...]


def drive_operator(
    sensor: DefectSpinSensor,
    drive_direction_lab=(1.0, 0.0, 0.0),
) -> np.ndarray:
    """Return the dimensionless microwave magnetic-dipole drive operator."""

    direction_lab = _vector(drive_direction_lab, name="drive_direction_lab")
    norm = float(np.linalg.norm(direction_lab))
    if norm <= 0.0:
        raise ValueError("drive_direction_lab must be non-zero")
    direction_local = sensor.frame.vector_to_local(direction_lab / norm)
    effective = sensor.g_tensor.T @ direction_local
    return sum(
        component * operator
        for component, operator in zip(effective, sensor_spin_operators(sensor))
    )


def diagonalize_sensor(
    sensor: DefectSpinSensor,
    b0_vector_tesla_lab=(0.0, 0.0, 0.0),
    *,
    drive_direction_lab=(1.0, 0.0, 0.0),
    frequency_tolerance_hz: float = 1.0e-9,
    strength_tolerance: float = 1.0e-14,
) -> DefectEigensystem:
    """Diagonalize a defect sensor and report microwave-active transitions."""

    b0 = _vector(b0_vector_tesla_lab, name="b0_vector_tesla_lab")
    values, vectors = np.linalg.eigh(sensor_hamiltonian(sensor, b0))
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    levels_hz = values / TAU
    drive = drive_operator(sensor, drive_direction_lab)
    drive_eigen = vectors.conj().T @ drive @ vectors

    transitions = []
    for lower in range(sensor.dimension):
        for upper in range(lower + 1, sensor.dimension):
            frequency = float(levels_hz[upper] - levels_hz[lower])
            strength = float(abs(drive_eigen[lower, upper]) ** 2)
            if (
                frequency > frequency_tolerance_hz
                and strength > strength_tolerance
            ):
                transitions.append(
                    DefectTransition(
                        lower=lower,
                        upper=upper,
                        frequency_hz=frequency,
                        strength=strength,
                    )
                )
    return DefectEigensystem(
        sensor=sensor,
        b0_vector_tesla_lab=b0,
        levels_hz=levels_hz,
        eigenvectors=vectors,
        transitions=tuple(transitions),
    )


__all__ = [
    "DefectEigensystem",
    "DefectTransition",
    "diagonalize_sensor",
    "drive_operator",
    "sensor_hamiltonian",
    "sensor_spin_operators",
    "zeeman_hamiltonian",
    "zfs_hamiltonian",
]
