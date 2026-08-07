"""Lumped-element resonant circuit model for cylindrical birdcage coils.

The solver uses the rung currents as independent mesh coordinates.  End-ring
currents are completed by KCL exactly as in :mod:`spin_dynamics.fields.birdcage`.
Series branch R, L, and C values then form compact effective matrices for
lossless modal analysis and finite-loss frequency-domain drive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from spin_dynamics.fields.birdcage import (
    BirdcageCurrentMode,
    BirdcageGeometry,
    birdcage_current_mode,
)

RealArrayInput = float | Sequence[float] | np.ndarray


def _positive_array(
    value: RealArrayInput,
    size: int,
    name: str,
    *,
    allow_zero: bool = False,
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=np.float64)
    else:
        result = np.array(result, dtype=np.float64, copy=True)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if result.shape != (size,) or not np.all(np.isfinite(result)) or not np.all(lower_ok):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} scalar or length-{size} array")
    result.setflags(write=False)
    return result


def _optional_capacitance(
    value: RealArrayInput | None,
    size: int,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    return _positive_array(value, size, name)


def _zero_sum_basis(size: int) -> np.ndarray:
    projector = np.eye(size) - np.ones((size, size)) / size
    basis, _ = np.linalg.qr(projector[:, :-1], mode="reduced")
    return basis


def _end_ring_completion_matrix(size: int) -> np.ndarray:
    cumulative = np.tril(np.ones((size, size), dtype=np.float64))
    return cumulative - np.mean(cumulative, axis=0, keepdims=True)


def _is_uniform(value: np.ndarray) -> bool:
    return bool(np.allclose(value, value[0], rtol=1.0e-12, atol=0.0))


def _normalize_mode(rung_currents: np.ndarray) -> np.ndarray:
    result = np.asarray(rung_currents, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(result)))
    scale = abs(result[pivot])
    if scale == 0.0:
        raise ValueError("mode current cannot be identically zero")
    result = result / scale
    phase = np.angle(result[pivot])
    result *= np.exp(-1.0j * phase)
    return np.real_if_close(result, tol=1000).astype(np.complex128)


def _dominant_azimuthal_index(rung_currents: np.ndarray) -> int:
    spectrum = np.abs(np.fft.fft(rung_currents))
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return min(index, rung_currents.size - index)


@dataclass(frozen=True)
class BirdcageCircuitMode:
    """One lossless resonant mode with a series-loss Q estimate."""

    frequency_hz: float
    quality_factor: float
    dominant_azimuthal_index: int
    currents: BirdcageCurrentMode


@dataclass(frozen=True)
class BirdcageModalAnalysis:
    """Complete rung-current modal solution, ordered by frequency."""

    modes: tuple[BirdcageCircuitMode, ...]

    @property
    def frequencies_hz(self) -> np.ndarray:
        """Resonant frequencies in ascending order."""

        return np.asarray([mode.frequency_hz for mode in self.modes])

    @property
    def quality_factors(self) -> np.ndarray:
        """Series-loss Q estimates in ascending-frequency order."""

        return np.asarray([mode.quality_factor for mode in self.modes])

    def azimuthal_modes(self, mode_index: int) -> tuple[BirdcageCircuitMode, ...]:
        """Return modes whose strongest discrete Fourier component is ``mode_index``."""

        requested = int(mode_index)
        matches = tuple(
            mode
            for mode in self.modes
            if mode.dominant_azimuthal_index == requested
        )
        if not matches:
            raise ValueError(f"no mode has dominant azimuthal index {requested}")
        return matches

    def splitting_hz(self, mode_index: int) -> float:
        """Frequency span of one nominally degenerate azimuthal mode family."""

        family = self.azimuthal_modes(mode_index)
        frequencies = [mode.frequency_hz for mode in family]
        return float(max(frequencies) - min(frequencies))


@dataclass(frozen=True)
class BirdcageDriveSolution:
    """Finite-loss response to series voltage sources placed in rung branches."""

    frequency_hz: float
    source_voltages_v: np.ndarray
    currents: BirdcageCurrentMode
    supplied_power_w: float
    dissipated_power_w: float

    def __post_init__(self) -> None:
        source = np.array(self.source_voltages_v, dtype=np.complex128, copy=True)
        source.setflags(write=False)
        object.__setattr__(self, "source_voltages_v", source)

    @property
    def input_impedance_ohm(self) -> complex:
        """Single-source branch impedance, or NaN for a multi-source drive."""

        active = np.flatnonzero(np.abs(self.source_voltages_v) > 0.0)
        if active.size != 1:
            return complex(np.nan, np.nan)
        index = int(active[0])
        current = self.currents.rung_currents_a[index]
        if abs(current) == 0.0:
            return complex(np.inf, 0.0)
        return complex(self.source_voltages_v[index] / current)


@dataclass(frozen=True)
class BirdcageCircuit:
    """Series-branch RLC model of a symmetric two-end-ring birdcage.

    Component arguments may be scalars or one value per azimuthal section.
    End-ring values apply to the corresponding section on both end rings.
    A low-pass cage has rung capacitors only, a high-pass cage has end-ring
    capacitors only, and a band-pass cage has both.
    """

    geometry: BirdcageGeometry
    rung_inductance_h: RealArrayInput
    end_ring_inductance_h: RealArrayInput
    rung_capacitance_f: RealArrayInput | None = None
    end_ring_capacitance_f: RealArrayInput | None = None
    rung_resistance_ohm: RealArrayInput = 0.0
    end_ring_resistance_ohm: RealArrayInput = 0.0

    def __post_init__(self) -> None:
        size = self.geometry.n_rungs
        rung_inductance = _positive_array(
            self.rung_inductance_h,
            size,
            "rung_inductance_h",
        )
        ring_inductance = _positive_array(
            self.end_ring_inductance_h,
            size,
            "end_ring_inductance_h",
        )
        rung_capacitance = _optional_capacitance(
            self.rung_capacitance_f,
            size,
            "rung_capacitance_f",
        )
        ring_capacitance = _optional_capacitance(
            self.end_ring_capacitance_f,
            size,
            "end_ring_capacitance_f",
        )
        if rung_capacitance is None and ring_capacitance is None:
            raise ValueError("at least one branch family must contain capacitance")
        rung_resistance = _positive_array(
            self.rung_resistance_ohm,
            size,
            "rung_resistance_ohm",
            allow_zero=True,
        )
        ring_resistance = _positive_array(
            self.end_ring_resistance_ohm,
            size,
            "end_ring_resistance_ohm",
            allow_zero=True,
        )
        object.__setattr__(self, "rung_inductance_h", rung_inductance)
        object.__setattr__(self, "end_ring_inductance_h", ring_inductance)
        object.__setattr__(self, "rung_capacitance_f", rung_capacitance)
        object.__setattr__(self, "end_ring_capacitance_f", ring_capacitance)
        object.__setattr__(self, "rung_resistance_ohm", rung_resistance)
        object.__setattr__(self, "end_ring_resistance_ohm", ring_resistance)

    @property
    def architecture(self) -> str:
        """Return ``"low_pass"``, ``"high_pass"``, or ``"band_pass"``."""

        if self.rung_capacitance_f is not None and self.end_ring_capacitance_f is None:
            return "low_pass"
        if self.rung_capacitance_f is None and self.end_ring_capacitance_f is not None:
            return "high_pass"
        return "band_pass"

    @property
    def end_ring_completion_matrix(self) -> np.ndarray:
        """Map zero-sum rung currents to minimum-norm positive-ring currents."""

        return _end_ring_completion_matrix(self.geometry.n_rungs)

    @property
    def zero_sum_basis(self) -> np.ndarray:
        """Orthonormal basis for physically closed rung-current patterns."""

        return _zero_sum_basis(self.geometry.n_rungs)

    def _effective_branch_matrix(
        self,
        rung_values: np.ndarray,
        ring_values: np.ndarray,
    ) -> np.ndarray:
        completion = self.end_ring_completion_matrix
        return np.diag(rung_values) + 2.0 * (
            completion.T @ np.diag(ring_values) @ completion
        )

    @property
    def effective_inductance_h(self) -> np.ndarray:
        """Rung-coordinate inductance matrix, including both end rings."""

        return self._effective_branch_matrix(
            self.rung_inductance_h,
            self.end_ring_inductance_h,
        )

    @property
    def effective_resistance_ohm(self) -> np.ndarray:
        """Rung-coordinate series resistance matrix, including both end rings."""

        return self._effective_branch_matrix(
            self.rung_resistance_ohm,
            self.end_ring_resistance_ohm,
        )

    @property
    def effective_inverse_capacitance_f_inv(self) -> np.ndarray:
        """Rung-coordinate inverse-capacitance matrix."""

        rung_inverse = (
            np.zeros(self.geometry.n_rungs)
            if self.rung_capacitance_f is None
            else 1.0 / self.rung_capacitance_f
        )
        ring_inverse = (
            np.zeros(self.geometry.n_rungs)
            if self.end_ring_capacitance_f is None
            else 1.0 / self.end_ring_capacitance_f
        )
        return self._effective_branch_matrix(rung_inverse, ring_inverse)

    def modal_analysis(self) -> BirdcageModalAnalysis:
        """Solve ``C^-1 i = omega^2 L i`` in the zero-sum rung subspace."""

        basis = self.zero_sum_basis
        inductance = basis.T @ self.effective_inductance_h @ basis
        inverse_capacitance = (
            basis.T @ self.effective_inverse_capacitance_f_inv @ basis
        )
        resistance = basis.T @ self.effective_resistance_ohm @ basis

        cholesky = np.linalg.cholesky(inductance)
        left = np.linalg.solve(cholesky, inverse_capacitance)
        transformed = np.linalg.solve(cholesky, left.T).T
        transformed = 0.5 * (transformed + transformed.T)
        omega_squared, transformed_vectors = np.linalg.eigh(transformed)
        if np.any(omega_squared <= 0.0):
            raise ValueError("circuit contains a non-resonant zero-frequency mode")
        coordinates = np.linalg.solve(cholesky.T, transformed_vectors)

        modes: list[BirdcageCircuitMode] = []
        for index, value in enumerate(omega_squared):
            reduced_current = coordinates[:, index]
            rung_current = _normalize_mode(basis @ reduced_current)
            reduced_current = basis.T @ rung_current
            omega = float(np.sqrt(value))
            magnetic = float(
                np.real(
                    np.vdot(
                        reduced_current,
                        inductance @ reduced_current,
                    )
                )
            )
            loss = float(
                np.real(
                    np.vdot(
                        reduced_current,
                        resistance @ reduced_current,
                    )
                )
            )
            quality_factor = np.inf if loss == 0.0 else omega * magnetic / loss
            azimuthal_index = _dominant_azimuthal_index(rung_current)
            modes.append(
                BirdcageCircuitMode(
                    frequency_hz=omega / (2.0 * np.pi),
                    quality_factor=float(quality_factor),
                    dominant_azimuthal_index=azimuthal_index,
                    currents=birdcage_current_mode(
                        rung_current,
                        label=(
                            f"{self.architecture} circuit mode "
                            f"m={azimuthal_index}"
                        ),
                    ),
                )
            )
        return BirdcageModalAnalysis(tuple(modes))

    def uniform_mode_frequency_hz(self, mode_index: int) -> float:
        """Analytical frequency for a spatially uniform component set."""

        mode = int(mode_index)
        size = self.geometry.n_rungs
        if mode < 1 or mode > size // 2:
            raise ValueError("mode_index must satisfy 1 <= mode_index <= n_rungs/2")
        component_arrays = (
            self.rung_inductance_h,
            self.end_ring_inductance_h,
            self.rung_resistance_ohm,
            self.end_ring_resistance_ohm,
        )
        optional_arrays = (
            self.rung_capacitance_f,
            self.end_ring_capacitance_f,
        )
        if any(not _is_uniform(value) for value in component_arrays) or any(
            value is not None and not _is_uniform(value)
            for value in optional_arrays
        ):
            raise ValueError("uniform_mode_frequency_hz requires uniform components")
        sine_squared = np.sin(np.pi * mode / size) ** 2
        effective_inductance = (
            self.rung_inductance_h[0]
            + self.end_ring_inductance_h[0] / (2.0 * sine_squared)
        )
        inverse_capacitance = 0.0
        if self.rung_capacitance_f is not None:
            inverse_capacitance += 1.0 / self.rung_capacitance_f[0]
        if self.end_ring_capacitance_f is not None:
            inverse_capacitance += (
                1.0
                / (
                    2.0
                    * sine_squared
                    * self.end_ring_capacitance_f[0]
                )
            )
        return float(
            np.sqrt(inverse_capacitance / effective_inductance)
            / (2.0 * np.pi)
        )

    def solve_drive(
        self,
        frequency_hz: float,
        rung_source_voltages_v: Iterable[complex] | np.ndarray,
        *,
        label: str = "driven circuit",
    ) -> BirdcageDriveSolution:
        """Solve the finite-loss response to series sources in rung branches."""

        frequency = float(frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        source = np.asarray(
            tuple(rung_source_voltages_v),
            dtype=np.complex128,
        )
        size = self.geometry.n_rungs
        if source.shape != (size,) or not np.all(np.isfinite(source)):
            raise ValueError(
                f"rung_source_voltages_v must be a finite length-{size} array"
            )
        basis = self.zero_sum_basis
        omega = 2.0 * np.pi * frequency
        impedance = (
            self.effective_resistance_ohm
            + 1.0j
            * (
                omega * self.effective_inductance_h
                - self.effective_inverse_capacitance_f_inv / omega
            )
        )
        reduced_impedance = basis.T @ impedance @ basis
        coordinates = np.linalg.solve(
            reduced_impedance,
            basis.T @ source,
        )
        rung_current = basis @ coordinates
        currents = birdcage_current_mode(rung_current, label=label)
        supplied_power = 0.5 * float(np.real(np.vdot(source, rung_current)))
        dissipated_power = 0.5 * float(
            np.real(
                np.vdot(
                    rung_current,
                    self.effective_resistance_ohm @ rung_current,
                )
            )
        )
        return BirdcageDriveSolution(
            frequency_hz=frequency,
            source_voltages_v=source,
            currents=currents,
            supplied_power_w=supplied_power,
            dissipated_power_w=dissipated_power,
        )


def tuned_low_pass_birdcage(
    geometry: BirdcageGeometry,
    resonance_frequency_hz: float,
    *,
    rung_inductance_h: RealArrayInput,
    end_ring_inductance_h: RealArrayInput,
    mode_index: int = 1,
    rung_resistance_ohm: RealArrayInput = 0.0,
    end_ring_resistance_ohm: RealArrayInput = 0.0,
) -> BirdcageCircuit:
    """Create a uniform low-pass cage tuned at one azimuthal mode."""

    frequency = float(resonance_frequency_hz)
    mode = int(mode_index)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("resonance_frequency_hz must be finite and positive")
    if mode < 1 or mode > geometry.n_rungs // 2:
        raise ValueError("mode_index must satisfy 1 <= mode_index <= n_rungs/2")
    rung_l = _positive_array(
        rung_inductance_h,
        geometry.n_rungs,
        "rung_inductance_h",
    )
    ring_l = _positive_array(
        end_ring_inductance_h,
        geometry.n_rungs,
        "end_ring_inductance_h",
    )
    if not _is_uniform(rung_l) or not _is_uniform(ring_l):
        raise ValueError("tuned factory requires uniform inductances")
    sine_squared = np.sin(np.pi * mode / geometry.n_rungs) ** 2
    effective_l = rung_l[0] + ring_l[0] / (2.0 * sine_squared)
    capacitance = 1.0 / ((2.0 * np.pi * frequency) ** 2 * effective_l)
    return BirdcageCircuit(
        geometry=geometry,
        rung_inductance_h=rung_l,
        end_ring_inductance_h=ring_l,
        rung_capacitance_f=capacitance,
        rung_resistance_ohm=rung_resistance_ohm,
        end_ring_resistance_ohm=end_ring_resistance_ohm,
    )


def tuned_high_pass_birdcage(
    geometry: BirdcageGeometry,
    resonance_frequency_hz: float,
    *,
    rung_inductance_h: RealArrayInput,
    end_ring_inductance_h: RealArrayInput,
    mode_index: int = 1,
    rung_resistance_ohm: RealArrayInput = 0.0,
    end_ring_resistance_ohm: RealArrayInput = 0.0,
) -> BirdcageCircuit:
    """Create a uniform high-pass cage tuned at one azimuthal mode."""

    frequency = float(resonance_frequency_hz)
    mode = int(mode_index)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("resonance_frequency_hz must be finite and positive")
    if mode < 1 or mode > geometry.n_rungs // 2:
        raise ValueError("mode_index must satisfy 1 <= mode_index <= n_rungs/2")
    rung_l = _positive_array(
        rung_inductance_h,
        geometry.n_rungs,
        "rung_inductance_h",
    )
    ring_l = _positive_array(
        end_ring_inductance_h,
        geometry.n_rungs,
        "end_ring_inductance_h",
    )
    if not _is_uniform(rung_l) or not _is_uniform(ring_l):
        raise ValueError("tuned factory requires uniform inductances")
    sine_squared = np.sin(np.pi * mode / geometry.n_rungs) ** 2
    effective_l = ring_l[0] + 2.0 * sine_squared * rung_l[0]
    capacitance = 1.0 / ((2.0 * np.pi * frequency) ** 2 * effective_l)
    return BirdcageCircuit(
        geometry=geometry,
        rung_inductance_h=rung_l,
        end_ring_inductance_h=ring_l,
        end_ring_capacitance_f=capacitance,
        rung_resistance_ohm=rung_resistance_ohm,
        end_ring_resistance_ohm=end_ring_resistance_ohm,
    )


def birdcage_quadrature_port_voltages(
    geometry: BirdcageGeometry,
    *,
    voltage_v: complex = 1.0,
    first_rung: int = 0,
    handedness: int = 1,
) -> np.ndarray:
    """Two equal rung sources separated by 90 degrees in spatial and RF phase."""

    if geometry.n_rungs % 4 != 0:
        raise ValueError("quadrature ports require n_rungs divisible by four")
    if handedness not in (-1, 1):
        raise ValueError("handedness must be +1 or -1")
    voltage = complex(voltage_v)
    if not np.isfinite(voltage):
        raise ValueError("voltage_v must be finite")
    first = int(first_rung) % geometry.n_rungs
    second = (first + geometry.n_rungs // 4) % geometry.n_rungs
    result = np.zeros(geometry.n_rungs, dtype=np.complex128)
    result[first] = voltage
    result[second] = -1.0j * handedness * voltage
    return result
