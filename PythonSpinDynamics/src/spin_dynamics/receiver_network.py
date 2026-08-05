"""Coupled multiport receiver loading, sensitivity, and thermal-noise model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

BOLTZMANN_J_PER_K = 1.380649e-23


def _as_impedance_matrix(
    value: complex | Iterable[complex] | np.ndarray,
    n_channels: int,
    name: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        matrix = complex(array) * np.eye(n_channels, dtype=np.complex128)
    elif array.ndim == 1:
        if array.shape != (n_channels,):
            raise ValueError(f"{name} vector must have length {n_channels}")
        matrix = np.diag(array)
    elif array.shape == (n_channels, n_channels):
        matrix = array
    else:
        raise ValueError(
            f"{name} must be scalar, length {n_channels}, or "
            f"shape ({n_channels}, {n_channels})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix


def _dissipative_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.conj().T)


def _validate_passive_reciprocal(matrix: np.ndarray, name: str) -> np.ndarray:
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-12 * scale
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=1e-12):
        raise ValueError(f"{name} must be reciprocal (complex symmetric)")
    dissipation = _dissipative_part(matrix)
    if float(np.min(np.linalg.eigvalsh(dissipation))) < -tolerance:
        raise ValueError(f"{name} must be passive")
    return matrix


def _validate_covariance(
    value: Iterable[complex] | np.ndarray | None,
    n_channels: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.zeros((n_channels, n_channels), dtype=np.complex128)
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (n_channels, n_channels):
        raise ValueError(f"{name} must have shape ({n_channels}, {n_channels})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-12 * scale
    if not np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=1e-12):
        raise ValueError(f"{name} must be Hermitian")
    matrix = _dissipative_part(matrix)
    if float(np.min(np.linalg.eigvalsh(matrix))) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return matrix


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    """Return the complex correlation matrix of a channel covariance."""

    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    variance = np.clip(np.real(np.diag(matrix)), 0.0, None)
    scale = np.sqrt(variance[:, np.newaxis] * variance[np.newaxis, :])
    return np.divide(
        matrix,
        scale,
        out=np.zeros_like(matrix),
        where=scale > 0.0,
    )


def scale_noise_covariance(
    covariance: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    """Scale a covariance shape to a requested mean per-channel RMS."""

    std = float(noise_std)
    if not np.isfinite(std) or std < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    matrix = np.asarray(covariance, dtype=np.complex128)
    mean_variance = float(np.mean(np.real(np.diag(matrix))))
    if mean_variance <= 0.0:
        raise ValueError("covariance must contain positive channel variance")
    return matrix * (std**2 / mean_variance)


@dataclass(frozen=True)
class ReceiverNetworkSolution:
    """Loaded sensitivity maps and output-noise diagnostics."""

    geometric_sensitivities: np.ndarray
    effective_sensitivities: np.ndarray
    transfer_matrix: np.ndarray
    source_impedance_ohm: np.ndarray
    total_impedance_ohm: np.ndarray
    output_impedance_ohm: np.ndarray
    passive_noise_covariance_v2: np.ndarray
    preamp_voltage_noise_covariance_v2: np.ndarray
    preamp_current_noise_covariance_v2: np.ndarray
    noise_covariance_v2: np.ndarray
    noise_correlation: np.ndarray
    frequency_hz: float


@dataclass(frozen=True, eq=False)
class ReceiverNetwork:
    """Reciprocal passive coil/load network at one receive frequency.

    ``coil_impedance_ohm`` is the geometric multiport coil impedance. Optional
    ``series_impedance_ohm`` represents tuning, decoupling, cable, and matching
    elements in series with the coil ports. ``load_impedance_ohm`` is the
    preamplifier-side termination. Scalars and vectors expand to diagonal
    matrices.
    """

    frequency_hz: float
    coil_impedance_ohm: np.ndarray
    load_impedance_ohm: complex | np.ndarray
    series_impedance_ohm: complex | np.ndarray | None = None
    temperature_k: float = 293.15
    noise_bandwidth_hz: float = 1.0
    preamp_voltage_noise_covariance_v2_per_hz: np.ndarray | None = None
    preamp_current_noise_covariance_a2_per_hz: np.ndarray | None = None

    def __post_init__(self) -> None:
        frequency = float(self.frequency_hz)
        temperature = float(self.temperature_k)
        bandwidth = float(self.noise_bandwidth_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature_k must be finite and positive")
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            raise ValueError("noise_bandwidth_hz must be finite and positive")

        coil = np.asarray(self.coil_impedance_ohm, dtype=np.complex128)
        if coil.ndim != 2 or coil.shape[0] == 0 or coil.shape[0] != coil.shape[1]:
            raise ValueError("coil_impedance_ohm must be a non-empty square matrix")
        n_channels = int(coil.shape[0])
        coil = _validate_passive_reciprocal(coil, "coil_impedance_ohm")
        load = _validate_passive_reciprocal(
            _as_impedance_matrix(
                self.load_impedance_ohm,
                n_channels,
                "load_impedance_ohm",
            ),
            "load_impedance_ohm",
        )
        series = (
            np.zeros_like(coil)
            if self.series_impedance_ohm is None
            else _validate_passive_reciprocal(
                _as_impedance_matrix(
                    self.series_impedance_ohm,
                    n_channels,
                    "series_impedance_ohm",
                ),
                "series_impedance_ohm",
            )
        )
        voltage_noise = _validate_covariance(
            self.preamp_voltage_noise_covariance_v2_per_hz,
            n_channels,
            "preamp_voltage_noise_covariance_v2_per_hz",
        )
        current_noise = _validate_covariance(
            self.preamp_current_noise_covariance_a2_per_hz,
            n_channels,
            "preamp_current_noise_covariance_a2_per_hz",
        )
        total = coil + series + load
        if np.linalg.matrix_rank(total) < n_channels:
            raise ValueError("the loaded receiver impedance matrix must be nonsingular")

        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "noise_bandwidth_hz", bandwidth)
        object.__setattr__(self, "coil_impedance_ohm", coil)
        object.__setattr__(self, "load_impedance_ohm", load)
        object.__setattr__(self, "series_impedance_ohm", series)
        object.__setattr__(
            self,
            "preamp_voltage_noise_covariance_v2_per_hz",
            voltage_noise,
        )
        object.__setattr__(
            self,
            "preamp_current_noise_covariance_a2_per_hz",
            current_noise,
        )

    @property
    def n_channels(self) -> int:
        return int(self.coil_impedance_ohm.shape[0])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReceiverNetwork):
            return False
        scalar_fields = (
            "frequency_hz",
            "temperature_k",
            "noise_bandwidth_hz",
        )
        array_fields = (
            "coil_impedance_ohm",
            "load_impedance_ohm",
            "series_impedance_ohm",
            "preamp_voltage_noise_covariance_v2_per_hz",
            "preamp_current_noise_covariance_a2_per_hz",
        )
        return all(
            getattr(self, name) == getattr(other, name) for name in scalar_fields
        ) and all(
            np.array_equal(getattr(self, name), getattr(other, name))
            for name in array_fields
        )

    @property
    def source_impedance_ohm(self) -> np.ndarray:
        return self.coil_impedance_ohm + self.series_impedance_ohm

    @property
    def total_impedance_ohm(self) -> np.ndarray:
        return self.source_impedance_ohm + self.load_impedance_ohm

    @property
    def transfer_matrix(self) -> np.ndarray:
        """Map open-circuit coil voltages to loaded output voltages."""

        return self.load_impedance_ohm @ np.linalg.inv(self.total_impedance_ohm)

    @property
    def output_impedance_ohm(self) -> np.ndarray:
        """Multiport source/load parallel impedance seen by preamp current noise."""

        source_admittance = np.linalg.pinv(
            self.source_impedance_ohm,
            hermitian=False,
        )
        load_admittance = np.linalg.pinv(
            self.load_impedance_ohm,
            hermitian=False,
        )
        return np.linalg.pinv(
            source_admittance + load_admittance,
            hermitian=False,
        )

    def solve(
        self,
        geometric_sensitivities: Iterable[complex] | np.ndarray,
    ) -> ReceiverNetworkSolution:
        """Load geometric channel maps and derive the output noise covariance."""

        sensitivities = np.asarray(
            geometric_sensitivities,
            dtype=np.complex128,
        )
        if sensitivities.ndim < 2 or sensitivities.shape[0] != self.n_channels:
            raise ValueError(
                "geometric_sensitivities must be channel-leading with one "
                "map per network port"
            )
        if not np.all(np.isfinite(sensitivities)):
            raise ValueError("geometric_sensitivities must be finite")

        transfer = self.transfer_matrix
        flat = sensitivities.reshape(self.n_channels, -1)
        effective = (transfer @ flat).reshape(sensitivities.shape)
        bandwidth = self.noise_bandwidth_hz
        output_impedance = self.output_impedance_ohm
        passive = (
            4.0
            * BOLTZMANN_J_PER_K
            * self.temperature_k
            * bandwidth
            * _dissipative_part(output_impedance)
        )
        voltage = (
            bandwidth
            * self.preamp_voltage_noise_covariance_v2_per_hz
        )
        current = (
            bandwidth
            * output_impedance
            @ self.preamp_current_noise_covariance_a2_per_hz
            @ output_impedance.conj().T
        )
        covariance = _dissipative_part(passive + voltage + current)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        covariance = (
            eigenvectors * np.clip(eigenvalues, 0.0, None)[np.newaxis, :]
        ) @ eigenvectors.conj().T
        return ReceiverNetworkSolution(
            geometric_sensitivities=sensitivities,
            effective_sensitivities=effective,
            transfer_matrix=transfer,
            source_impedance_ohm=self.source_impedance_ohm,
            total_impedance_ohm=self.total_impedance_ohm,
            output_impedance_ohm=output_impedance,
            passive_noise_covariance_v2=_dissipative_part(passive),
            preamp_voltage_noise_covariance_v2=voltage,
            preamp_current_noise_covariance_v2=_dissipative_part(current),
            noise_covariance_v2=covariance,
            noise_correlation=covariance_to_correlation(covariance),
            frequency_hz=self.frequency_hz,
        )


def coupled_resonant_modes(
    inductance_h: Iterable[float] | np.ndarray,
    capacitance_f: float | Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lossless coupled-series-resonator frequencies and current modes."""

    inductance = np.asarray(inductance_h, dtype=np.float64)
    if (
        inductance.ndim != 2
        or inductance.shape[0] == 0
        or inductance.shape[0] != inductance.shape[1]
    ):
        raise ValueError("inductance_h must be a non-empty square matrix")
    n_channels = inductance.shape[0]
    capacitance = np.asarray(capacitance_f, dtype=np.float64)
    if capacitance.ndim == 0:
        capacitance = float(capacitance) * np.eye(n_channels)
    elif capacitance.ndim == 1:
        if capacitance.shape != (n_channels,):
            raise ValueError("capacitance_f vector must match inductance_h")
        capacitance = np.diag(capacitance)
    elif capacitance.shape != inductance.shape:
        raise ValueError("capacitance_f matrix must match inductance_h")
    if not np.all(np.isfinite(inductance)) or not np.all(np.isfinite(capacitance)):
        raise ValueError("inductance_h and capacitance_f must be finite")
    if not np.allclose(inductance, inductance.T, rtol=1e-12, atol=1e-15):
        raise ValueError("inductance_h must be symmetric")
    if not np.allclose(capacitance, capacitance.T, rtol=1e-12, atol=1e-15):
        raise ValueError("capacitance_f must be symmetric")
    if np.min(np.linalg.eigvalsh(inductance)) <= 0.0:
        raise ValueError("inductance_h must be positive definite")
    if np.min(np.linalg.eigvalsh(capacitance)) <= 0.0:
        raise ValueError("capacitance_f must be positive definite")

    omega_squared, modes = np.linalg.eig(
        np.linalg.solve(inductance, np.linalg.inv(capacitance))
    )
    if np.max(np.abs(np.imag(omega_squared))) > 1e-10 * np.max(
        np.abs(omega_squared)
    ):
        raise np.linalg.LinAlgError("coupled resonant frequencies are not real")
    omega_squared = np.real(omega_squared)
    if np.min(omega_squared) <= 0.0:
        raise np.linalg.LinAlgError("coupled resonant frequencies are not positive")
    order = np.argsort(omega_squared)
    frequencies = np.sqrt(omega_squared[order]) / (2.0 * np.pi)
    ordered_modes = np.real_if_close(modes[:, order])
    ordered_modes /= np.linalg.norm(ordered_modes, axis=0, keepdims=True)
    return frequencies, ordered_modes
