"""Loaded, matched, multiport birdcage circuits and receive sensitivities.

This module completes the lumped Phase 5 birdcage path:

* exact filament mutual partial inductances between explicit branches;
* additive positive-semidefinite conductor/sample loss matrices;
* loaded modal frequencies and Q;
* physical rung-port impedance and scattering matrices;
* lossless per-port L matching;
* reciprocal B1- maps and equilibrium thermal-noise covariance.

The conductive loading is a first-order impressed-E/Born model.  It does not
include charge correction, skin-effect shielding, or full-wave propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, TypeAlias

import numpy as np

from spin_dynamics.fields.birdcage import (
    BirdcageFieldSolution,
    BirdcageGeometry,
    birdcage_current_mode,
    solve_birdcage_field,
)
from spin_dynamics.fields.birdcage_circuit import (
    BirdcageCircuit,
    BirdcageCircuitMode,
    BirdcageDriveSolution,
    BirdcageModalAnalysis,
)
from spin_dynamics.fields.coil_peec import _mutualfil_matrix
from spin_dynamics.fields.quasistatic import vector_potential

_BOLTZMANN_J_PER_K = 1.380649e-23


def _freeze(value: np.ndarray, *, dtype: np.dtype | type | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _real_symmetric_matrix(
    value: np.ndarray | None,
    size: int,
    name: str,
    *,
    positive_semidefinite: bool,
) -> np.ndarray:
    if value is None:
        result = np.zeros((size, size), dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite ({size}, {size}) matrix")
    scale = max(float(np.max(np.abs(result))), float(np.finfo(np.float64).tiny))
    if not np.allclose(result, result.T, atol=1.0e-12 * scale, rtol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    result = 0.5 * (result + result.T)
    if positive_semidefinite:
        tolerance = 1.0e-11 * scale
        if float(np.min(np.linalg.eigvalsh(result))) < -tolerance:
            raise ValueError(f"{name} must be positive semidefinite")
    return _freeze(result)


def _branch_paths(
    geometry: BirdcageGeometry,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], ...]:
    rungs = tuple((segment,) for segment in geometry.rung_segments())
    return (
        *rungs,
        *geometry.positive_end_ring_sections,
        *geometry.negative_end_ring_sections,
    )


def _path_endpoint_arrays(
    path: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.asarray([start for start, _ in path], dtype=np.float64)
    ends = np.asarray([end for _, end in path], dtype=np.float64)
    return starts, ends


def birdcage_branch_mutual_inductance_matrix(
    geometry: BirdcageGeometry,
) -> np.ndarray:
    """Return the full off-diagonal branch mutual-partial-inductance matrix.

    Branch order is all rungs, positive end-ring sections, then negative
    end-ring sections.  The diagonal is zero because branch self inductances
    remain the explicit values in :class:`BirdcageCircuit`.
    """

    paths = _branch_paths(geometry)
    endpoints = tuple(_path_endpoint_arrays(path) for path in paths)
    result = np.zeros((len(paths), len(paths)), dtype=np.float64)
    for first in range(len(paths)):
        starts_first, ends_first = endpoints[first]
        for second in range(first + 1, len(paths)):
            starts_second, ends_second = endpoints[second]
            mutual = float(
                np.sum(
                    _mutualfil_matrix(
                        starts_first,
                        ends_first,
                        starts_second,
                        ends_second,
                    )
                )
            )
            result[first, second] = mutual
            result[second, first] = mutual
    return result


def birdcage_conductive_loading_resistance(
    geometry: BirdcageGeometry,
    frequency_hz: float,
    sample_points_m: np.ndarray,
    *,
    conductivity_s_per_m: float | np.ndarray,
    cell_volume_m3: float,
) -> np.ndarray:
    """Return the first-order conductive-loss resistance matrix.

    For branch unit-current vector potentials ``A_i``, this evaluates

    ``R_ij = omega^2 integral sigma A_i . A_j dV``.

    It is the multi-branch analogue of ``reflected_resistance`` and is
    positive semidefinite by construction.  ``sample_points_m`` may have any
    leading spatial shape ending in three; conductivity is scalar or matches
    that leading shape.
    """

    frequency = float(frequency_hz)
    volume = float(cell_volume_m3)
    points = np.asarray(sample_points_m, dtype=np.float64)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("cell_volume_m3 must be finite and positive")
    if points.ndim < 2 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("sample_points_m must be finite with shape (..., 3)")
    spatial_shape = points.shape[:-1]
    conductivity = np.asarray(conductivity_s_per_m, dtype=np.float64)
    if conductivity.ndim == 0:
        conductivity = np.full(spatial_shape, float(conductivity))
    if (
        conductivity.shape != spatial_shape
        or not np.all(np.isfinite(conductivity))
        or np.any(conductivity < 0.0)
    ):
        raise ValueError(
            "conductivity_s_per_m must be a finite non-negative scalar "
            "or match sample_points_m.shape[:-1]"
        )
    flat_points = points.reshape(-1, 3)
    flat_conductivity = conductivity.reshape(-1)
    potentials = np.asarray(
        [
            vector_potential(flat_points, path, current=1.0)
            for path in _branch_paths(geometry)
        ]
    )
    omega = 2.0 * np.pi * frequency
    resistance = (omega**2 * volume) * np.einsum(
        "ipk,p,jpk->ij",
        potentials,
        flat_conductivity,
        potentials,
        optimize=True,
    )
    resistance = np.real(0.5 * (resistance + resistance.T))
    return resistance


@dataclass(frozen=True)
class BirdcageBranchLoading:
    """Additive branch coupling inductance and dissipative resistance."""

    n_branches: int
    inductance_coupling_h: np.ndarray | None = None
    resistance_ohm: np.ndarray | None = None

    def __post_init__(self) -> None:
        size = int(self.n_branches)
        if size < 12 or size % 3:
            raise ValueError("n_branches must be three times at least four rungs")
        inductance = _real_symmetric_matrix(
            self.inductance_coupling_h,
            size,
            "inductance_coupling_h",
            positive_semidefinite=False,
        )
        resistance = _real_symmetric_matrix(
            self.resistance_ohm,
            size,
            "resistance_ohm",
            positive_semidefinite=True,
        )
        object.__setattr__(self, "n_branches", size)
        object.__setattr__(self, "inductance_coupling_h", inductance)
        object.__setattr__(self, "resistance_ohm", resistance)


def _normalize_mode(rung_currents: np.ndarray) -> np.ndarray:
    result = np.asarray(rung_currents, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(result)))
    scale = abs(result[pivot])
    if scale == 0.0:
        raise ValueError("mode current cannot be identically zero")
    result = result / scale
    result *= np.exp(-1.0j * np.angle(result[pivot]))
    return np.real_if_close(result, tol=1000).astype(np.complex128)


def _dominant_azimuthal_index(rung_currents: np.ndarray) -> int:
    spectrum = np.abs(np.fft.fft(rung_currents))
    spectrum[0] = 0.0
    index = int(np.argmax(spectrum))
    return min(index, rung_currents.size - index)


@dataclass(frozen=True)
class BirdcageLoadedCircuit:
    """Birdcage RLC circuit augmented by mutual inductance and loading loss."""

    circuit: BirdcageCircuit
    loading: BirdcageBranchLoading

    def __post_init__(self) -> None:
        expected = 3 * self.circuit.geometry.n_rungs
        if self.loading.n_branches != expected:
            raise ValueError(
                f"loading must contain {expected} branches for this geometry"
            )
        basis = self.zero_sum_basis
        reduced_inductance = basis.T @ self.effective_inductance_h @ basis
        try:
            np.linalg.cholesky(reduced_inductance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "total inductance must be positive definite in the closed-cage subspace"
            ) from exc

    @property
    def geometry(self) -> BirdcageGeometry:
        return self.circuit.geometry

    @property
    def architecture(self) -> str:
        return self.circuit.architecture

    @property
    def zero_sum_basis(self) -> np.ndarray:
        return self.circuit.zero_sum_basis

    @property
    def branch_current_transform(self) -> np.ndarray:
        """Map rung currents to rungs, positive ring, then negative ring."""

        size = self.geometry.n_rungs
        completion = self.circuit.end_ring_completion_matrix
        return np.vstack((np.eye(size), completion, -completion))

    @property
    def effective_inductance_h(self) -> np.ndarray:
        transform = self.branch_current_transform
        return (
            self.circuit.effective_inductance_h
            + transform.T @ self.loading.inductance_coupling_h @ transform
        )

    @property
    def effective_resistance_ohm(self) -> np.ndarray:
        transform = self.branch_current_transform
        return (
            self.circuit.effective_resistance_ohm
            + transform.T @ self.loading.resistance_ohm @ transform
        )

    @property
    def effective_inverse_capacitance_f_inv(self) -> np.ndarray:
        return self.circuit.effective_inverse_capacitance_f_inv

    def impedance_matrix_ohm(self, frequency_hz: float) -> np.ndarray:
        """Return the rung-coordinate loaded impedance matrix."""

        frequency = float(frequency_hz)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        omega = 2.0 * np.pi * frequency
        return self.effective_resistance_ohm + 1.0j * (
            omega * self.effective_inductance_h
            - self.effective_inverse_capacitance_f_inv / omega
        )

    def modal_analysis(self) -> BirdcageModalAnalysis:
        """Solve loaded lossless modes and estimate Q from total resistance."""

        basis = self.zero_sum_basis
        inductance = basis.T @ self.effective_inductance_h @ basis
        inverse_capacitance = basis.T @ self.effective_inverse_capacitance_f_inv @ basis
        resistance = basis.T @ self.effective_resistance_ohm @ basis
        cholesky = np.linalg.cholesky(inductance)
        left = np.linalg.solve(cholesky, inverse_capacitance)
        transformed = np.linalg.solve(cholesky, left.T).T
        transformed = 0.5 * (transformed + transformed.T)
        omega_squared, transformed_vectors = np.linalg.eigh(transformed)
        if np.any(omega_squared <= 0.0):
            raise ValueError("loaded circuit contains a zero-frequency mode")
        coordinates = np.linalg.solve(cholesky.T, transformed_vectors)

        modes: list[BirdcageCircuitMode] = []
        for index, value in enumerate(omega_squared):
            rung_current = _normalize_mode(basis @ coordinates[:, index])
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
                        label=f"loaded circuit mode m={azimuthal_index}",
                    ),
                )
            )
        return BirdcageModalAnalysis(tuple(modes))

    def solve_drive(
        self,
        frequency_hz: float,
        rung_source_voltages_v: Iterable[complex] | np.ndarray,
        *,
        label: str = "loaded driven circuit",
    ) -> BirdcageDriveSolution:
        """Solve loaded finite-loss response to rung series sources."""

        source = np.asarray(tuple(rung_source_voltages_v), dtype=np.complex128)
        size = self.geometry.n_rungs
        if source.shape != (size,) or not np.all(np.isfinite(source)):
            raise ValueError(
                f"rung_source_voltages_v must be a finite length-{size} array"
            )
        basis = self.zero_sum_basis
        impedance = self.impedance_matrix_ohm(frequency_hz)
        reduced_impedance = basis.T @ impedance @ basis
        coordinates = np.linalg.solve(reduced_impedance, basis.T @ source)
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
            frequency_hz=float(frequency_hz),
            source_voltages_v=source,
            currents=currents,
            supplied_power_w=supplied_power,
            dissipated_power_w=dissipated_power,
        )


def retune_loaded_birdcage(
    circuit: BirdcageCircuit,
    loading: BirdcageBranchLoading,
    resonance_frequency_hz: float,
    *,
    mode_index: int = 1,
) -> BirdcageLoadedCircuit:
    """Scale all installed capacitors to retune one loaded mode family.

    Uniform scaling preserves the capacitance pattern and mode vectors while
    scaling every modal frequency by the inverse square root. This is exact
    for the frequency-independent lumped loading represented here.
    """

    target = float(resonance_frequency_hz)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("resonance_frequency_hz must be finite and positive")
    preliminary = BirdcageLoadedCircuit(circuit, loading)
    family = preliminary.modal_analysis().azimuthal_modes(mode_index)
    current = float(np.mean([mode.frequency_hz for mode in family]))
    capacitance_scale = (current / target) ** 2
    tuned = BirdcageCircuit(
        geometry=circuit.geometry,
        rung_inductance_h=circuit.rung_inductance_h,
        end_ring_inductance_h=circuit.end_ring_inductance_h,
        rung_capacitance_f=(
            None
            if circuit.rung_capacitance_f is None
            else circuit.rung_capacitance_f * capacitance_scale
        ),
        end_ring_capacitance_f=(
            None
            if circuit.end_ring_capacitance_f is None
            else circuit.end_ring_capacitance_f * capacitance_scale
        ),
        rung_resistance_ohm=circuit.rung_resistance_ohm,
        end_ring_resistance_ohm=circuit.end_ring_resistance_ohm,
    )
    return BirdcageLoadedCircuit(tuned, loading)


def calibrate_birdcage_conductor_quality_factor(
    circuit: BirdcageLoadedCircuit,
    quality_factor: float,
    *,
    mode_index: int = 1,
) -> BirdcageLoadedCircuit:
    """Scale explicit branch resistances to a selected loaded-mode Q.

    Mutual inductance and any additive loading resistance are retained. Only
    the rung and end-ring resistances owned by the underlying
    :class:`BirdcageCircuit` are scaled. This makes an assumed or measured
    unloaded Q explicit without misrepresenting it as a geometry-derived loss.
    """

    target = float(quality_factor)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("quality_factor must be finite and positive")
    family = circuit.modal_analysis().azimuthal_modes(mode_index)
    if not family:
        raise ValueError(f"no mode with azimuthal index {mode_index}")
    mode = family[0]
    current = mode.currents.rung_currents_a
    omega = 2.0 * np.pi * mode.frequency_hz
    magnetic = float(
        np.real(np.vdot(current, circuit.effective_inductance_h @ current))
    )
    conductor_resistance = circuit.circuit.effective_resistance_ohm
    conductor_loss = float(np.real(np.vdot(current, conductor_resistance @ current)))
    loading_resistance = circuit.effective_resistance_ohm - conductor_resistance
    loading_loss = float(np.real(np.vdot(current, loading_resistance @ current)))
    if conductor_loss <= 0.0:
        raise ValueError("explicit branch resistances must dissipate positive power")
    required_total_loss = omega * magnetic / target
    scale = (required_total_loss - loading_loss) / conductor_loss
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "quality_factor cannot be reached by scaling conductor loss: "
            "fixed loading loss already exceeds the requested total loss"
        )
    base = circuit.circuit
    calibrated = BirdcageCircuit(
        geometry=base.geometry,
        rung_inductance_h=base.rung_inductance_h,
        end_ring_inductance_h=base.end_ring_inductance_h,
        rung_capacitance_f=base.rung_capacitance_f,
        end_ring_capacitance_f=base.end_ring_capacitance_f,
        rung_resistance_ohm=base.rung_resistance_ohm * scale,
        end_ring_resistance_ohm=base.end_ring_resistance_ohm * scale,
    )
    return BirdcageLoadedCircuit(calibrated, circuit.loading)


CircuitModel: TypeAlias = BirdcageCircuit | BirdcageLoadedCircuit


def _circuit_impedance(circuit: CircuitModel, frequency_hz: float) -> np.ndarray:
    if isinstance(circuit, BirdcageLoadedCircuit):
        return circuit.impedance_matrix_ohm(frequency_hz)
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    omega = 2.0 * np.pi * frequency
    return circuit.effective_resistance_ohm + 1.0j * (
        omega * circuit.effective_inductance_h
        - circuit.effective_inverse_capacitance_f_inv / omega
    )


def _circuit_solve_drive(
    circuit: CircuitModel,
    frequency_hz: float,
    source: np.ndarray,
    label: str,
) -> BirdcageDriveSolution:
    return circuit.solve_drive(frequency_hz, source, label=label)


@dataclass(frozen=True)
class BirdcageMultiport:
    """Physical series-feed ports placed in selected rung branches."""

    circuit: CircuitModel
    rung_indices: Sequence[int]
    labels: Sequence[str] = ()

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.rung_indices)
        size = self.circuit.geometry.n_rungs
        if not indices or len(indices) >= size:
            raise ValueError(
                "rung_indices must contain between one and n_rungs-1 ports"
            )
        if len(set(indices)) != len(indices) or any(
            index < 0 or index >= size for index in indices
        ):
            raise ValueError("rung_indices must be unique valid rung indices")
        labels = tuple(str(label) for label in self.labels)
        if not labels:
            labels = tuple(f"rung{index}" for index in indices)
        if len(labels) != len(indices):
            raise ValueError("labels must contain one entry per port")
        reduced = self.circuit.zero_sum_basis.T @ self._selector(indices, size)
        if np.linalg.matrix_rank(reduced) != len(indices):
            raise ValueError("selected ports are not independent")
        object.__setattr__(self, "rung_indices", indices)
        object.__setattr__(self, "labels", labels)

    @staticmethod
    def _selector(indices: Sequence[int], size: int) -> np.ndarray:
        result = np.zeros((size, len(indices)), dtype=np.float64)
        result[np.asarray(indices), np.arange(len(indices))] = 1.0
        return result

    @property
    def n_ports(self) -> int:
        return len(self.rung_indices)

    @property
    def selector(self) -> np.ndarray:
        return self._selector(self.rung_indices, self.circuit.geometry.n_rungs)

    def admittance_matrix_siemens(self, frequency_hz: float) -> np.ndarray:
        """Return port current per applied series port voltage."""

        basis = self.circuit.zero_sum_basis
        reduced_ports = basis.T @ self.selector
        reduced_impedance = (
            basis.T
            @ _circuit_impedance(
                self.circuit,
                frequency_hz,
            )
            @ basis
        )
        response = np.linalg.solve(reduced_impedance, reduced_ports)
        return reduced_ports.T @ response

    def impedance_matrix_ohm(self, frequency_hz: float) -> np.ndarray:
        """Return the reciprocal physical-port impedance matrix."""

        return np.linalg.inv(self.admittance_matrix_siemens(frequency_hz))

    def scattering_matrix(
        self,
        frequency_hz: float,
        *,
        reference_impedance_ohm: float = 50.0,
    ) -> np.ndarray:
        """Return the port S matrix for equal real reference impedances."""

        impedance = self.impedance_matrix_ohm(frequency_hz)
        return impedance_to_scattering(
            impedance,
            reference_impedance_ohm=reference_impedance_ohm,
        )

    def solve_port_currents(
        self,
        frequency_hz: float,
        port_currents_a: Iterable[complex] | np.ndarray,
        *,
        label: str = "port-current drive",
    ) -> BirdcageDriveSolution:
        """Return branch currents that realize specified physical port currents."""

        currents = np.asarray(tuple(port_currents_a), dtype=np.complex128)
        if currents.shape != (self.n_ports,) or not np.all(np.isfinite(currents)):
            raise ValueError(
                f"port_currents_a must be a finite length-{self.n_ports} array"
            )
        voltages = self.impedance_matrix_ohm(frequency_hz) @ currents
        source = self.selector @ voltages
        return _circuit_solve_drive(
            self.circuit,
            frequency_hz,
            source,
            label,
        )


def impedance_to_scattering(
    impedance_ohm: np.ndarray,
    *,
    reference_impedance_ohm: float = 50.0,
) -> np.ndarray:
    """Convert a square impedance matrix to equal-reference S parameters."""

    impedance = np.asarray(impedance_ohm, dtype=np.complex128)
    reference = float(reference_impedance_ohm)
    if (
        impedance.ndim != 2
        or impedance.shape[0] == 0
        or impedance.shape[0] != impedance.shape[1]
        or not np.all(np.isfinite(impedance))
    ):
        raise ValueError("impedance_ohm must be a finite non-empty square matrix")
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_impedance_ohm must be finite and positive")
    identity = np.eye(impedance.shape[0], dtype=np.complex128)
    return (impedance - reference * identity) @ np.linalg.inv(
        impedance + reference * identity
    )


@dataclass(frozen=True)
class BirdcageLMatch:
    """Lossless series-then-shunt L match for equal-reference input ports."""

    frequency_hz: float
    reference_impedance_ohm: float
    series_reactance_ohm: np.ndarray
    shunt_susceptance_siemens: np.ndarray

    def __post_init__(self) -> None:
        frequency = float(self.frequency_hz)
        reference = float(self.reference_impedance_ohm)
        series = np.asarray(self.series_reactance_ohm, dtype=np.float64)
        shunt = np.asarray(self.shunt_susceptance_siemens, dtype=np.float64)
        if not np.isfinite(frequency) or frequency <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_impedance_ohm must be finite and positive")
        if (
            series.ndim != 1
            or shunt.shape != series.shape
            or series.size == 0
            or not np.all(np.isfinite(series))
            or not np.all(np.isfinite(shunt))
        ):
            raise ValueError("matching arrays must be finite equal-length vectors")
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "reference_impedance_ohm", reference)
        object.__setattr__(self, "series_reactance_ohm", _freeze(series))
        object.__setattr__(self, "shunt_susceptance_siemens", _freeze(shunt))

    @property
    def n_ports(self) -> int:
        return int(self.series_reactance_ohm.size)

    def input_impedance_ohm(self, coil_impedance_ohm: np.ndarray) -> np.ndarray:
        """Transform the coupled coil-port impedance through the L match."""

        coil = np.asarray(coil_impedance_ohm, dtype=np.complex128)
        if coil.shape != (self.n_ports, self.n_ports):
            raise ValueError("coil_impedance_ohm shape must match the matching network")
        series_loaded = coil + 1.0j * np.diag(self.series_reactance_ohm)
        input_admittance = np.linalg.inv(series_loaded) + 1.0j * np.diag(
            self.shunt_susceptance_siemens
        )
        return np.linalg.inv(input_admittance)

    def scattering_matrix(self, coil_impedance_ohm: np.ndarray) -> np.ndarray:
        """Return matched-network S parameters."""

        return impedance_to_scattering(
            self.input_impedance_ohm(coil_impedance_ohm),
            reference_impedance_ohm=self.reference_impedance_ohm,
        )

    def input_to_coil_current_matrix(
        self,
        coil_impedance_ohm: np.ndarray,
    ) -> np.ndarray:
        """Map imposed input currents to physical coil-port currents."""

        coil = np.asarray(coil_impedance_ohm, dtype=np.complex128)
        if coil.shape != (self.n_ports, self.n_ports):
            raise ValueError("coil_impedance_ohm shape must match the matching network")
        series_loaded = coil + 1.0j * np.diag(self.series_reactance_ohm)
        input_impedance = self.input_impedance_ohm(coil)
        return np.linalg.solve(series_loaded, input_impedance)

    def component_summary(self) -> tuple[str, ...]:
        """Return compact per-port series and shunt L/C descriptions."""

        omega = 2.0 * np.pi * self.frequency_hz
        summaries: list[str] = []
        for series, shunt in zip(
            self.series_reactance_ohm,
            self.shunt_susceptance_siemens,
        ):
            if series >= 0.0:
                series_text = f"series L={series / omega * 1e9:.3f} nH"
            else:
                series_text = f"series C={-1.0 / (omega * series) * 1e12:.3f} pF"
            if shunt >= 0.0:
                shunt_text = f"shunt C={shunt / omega * 1e12:.3f} pF"
            else:
                shunt_text = f"shunt L={-1.0 / (omega * shunt) * 1e9:.3f} nH"
            summaries.append(f"{series_text}, {shunt_text}")
        return tuple(summaries)


def design_independent_l_match(
    port_impedance_ohm: np.ndarray,
    frequency_hz: float,
    *,
    reference_impedance_ohm: float = 50.0,
    solution_sign: int = 1,
) -> BirdcageLMatch:
    """Match each uncoupled port diagonal with a series-then-shunt L network.

    Off-diagonal coupling is retained when the returned network is evaluated,
    but the closed-form component design uses each diagonal independently.
    This makes residual S21/S12 an explicit diagnostic rather than silently
    absorbing coupling into an idealized decoupling network.
    """

    impedance = np.asarray(port_impedance_ohm, dtype=np.complex128)
    reference = float(reference_impedance_ohm)
    if (
        impedance.ndim != 2
        or impedance.shape[0] == 0
        or impedance.shape[0] != impedance.shape[1]
        or not np.all(np.isfinite(impedance))
    ):
        raise ValueError("port_impedance_ohm must be a finite square matrix")
    if solution_sign not in (-1, 1):
        raise ValueError("solution_sign must be +1 or -1")
    diagonal = np.diag(impedance)
    resistance = np.real(diagonal)
    if np.any(resistance <= 0.0) or np.any(resistance > reference):
        raise ValueError(
            "series-then-shunt match requires every port to satisfy "
            "0 < Re(Z_port) <= reference_impedance_ohm"
        )
    target_reactance = solution_sign * np.sqrt(resistance * (reference - resistance))
    series = target_reactance - np.imag(diagonal)
    shunt = np.divide(
        target_reactance,
        resistance * reference,
        out=np.zeros_like(target_reactance),
        where=resistance < reference,
    )
    return BirdcageLMatch(
        frequency_hz=frequency_hz,
        reference_impedance_ohm=reference,
        series_reactance_ohm=series,
        shunt_susceptance_siemens=shunt,
    )


@dataclass(frozen=True)
class BirdcageReceiveSensitivityMaps:
    """Per-input-current reciprocal fields and passive noise covariance."""

    frequency_hz: float
    b1_vector_t_per_a: np.ndarray
    b1_minus_t_per_a: np.ndarray
    normalization_t_per_a: np.ndarray
    normalized_magnitude: np.ndarray
    channel_labels: tuple[str, ...]
    input_impedance_ohm: np.ndarray
    noise_covariance_v2_per_hz: np.ndarray

    @property
    def n_channels(self) -> int:
        return int(self.b1_minus_t_per_a.shape[0])

    @property
    def normalized_complex(self) -> np.ndarray:
        shape = (self.n_channels,) + (1,) * (self.b1_minus_t_per_a.ndim - 1)
        return self.b1_minus_t_per_a / self.normalization_t_per_a.reshape(shape)

    @property
    def noise_correlation(self) -> np.ndarray:
        diagonal = np.real(np.diag(self.noise_covariance_v2_per_hz))
        scale = np.sqrt(np.outer(diagonal, diagonal))
        return np.divide(
            self.noise_covariance_v2_per_hz,
            scale,
            out=np.zeros_like(self.noise_covariance_v2_per_hz),
            where=scale > 0.0,
        )


def solve_birdcage_receive_sensitivities(
    multiport: BirdcageMultiport,
    frequency_hz: float,
    points_m: np.ndarray | Sequence[float],
    *,
    matching: BirdcageLMatch | None = None,
    normalization_weights: np.ndarray | None = None,
    temperature_k: float = 290.0,
) -> BirdcageReceiveSensitivityMaps:
    """Solve B1- per unit input current and equilibrium voltage-noise PSD."""

    frequency = float(frequency_hz)
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature_k must be finite and positive")
    coil_impedance = multiport.impedance_matrix_ohm(frequency)
    if matching is None:
        input_impedance = coil_impedance
        input_to_coil = np.eye(multiport.n_ports, dtype=np.complex128)
    else:
        if matching.n_ports != multiport.n_ports:
            raise ValueError("matching network port count must match multiport")
        if not np.isclose(matching.frequency_hz, frequency, rtol=1.0e-12):
            raise ValueError("matching network frequency must equal frequency_hz")
        input_impedance = matching.input_impedance_ohm(coil_impedance)
        input_to_coil = matching.input_to_coil_current_matrix(coil_impedance)

    field_solutions: list[BirdcageFieldSolution] = []
    for channel in range(multiport.n_ports):
        driven = multiport.solve_port_currents(
            frequency,
            input_to_coil[:, channel],
            label=f"receive unit input current {channel}",
        )
        field_solutions.append(
            solve_birdcage_field(
                multiport.circuit.geometry,
                driven.currents,
                points_m,
            )
        )
    vector = np.stack([solution.field_t for solution in field_solutions], axis=0)
    b1_minus = np.stack(
        [solution.b1_minus_t for solution in field_solutions],
        axis=0,
    )
    spatial_shape = b1_minus.shape[1:]
    if normalization_weights is None:
        weights = np.ones(spatial_shape, dtype=np.float64)
    else:
        weights = np.asarray(normalization_weights, dtype=np.float64)
    if (
        weights.shape != spatial_shape
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
    ):
        raise ValueError(
            "normalization_weights must be finite, non-negative, "
            "match the spatial field shape, and contain positive weight"
        )
    normalizations = np.sum(
        np.abs(b1_minus) * weights[np.newaxis, ...],
        axis=tuple(range(1, b1_minus.ndim)),
    ) / float(np.sum(weights))
    if np.any(normalizations <= 0.0):
        raise ValueError("every receive port must produce nonzero B1-")
    normalized_magnitude = np.abs(b1_minus) / normalizations.reshape(
        (multiport.n_ports,) + (1,) * len(spatial_shape)
    )
    dissipative = 0.5 * (input_impedance + input_impedance.conj().T)
    noise = 4.0 * _BOLTZMANN_J_PER_K * temperature * dissipative
    noise = 0.5 * (noise + noise.conj().T)
    return BirdcageReceiveSensitivityMaps(
        frequency_hz=frequency,
        b1_vector_t_per_a=_freeze(vector),
        b1_minus_t_per_a=_freeze(b1_minus),
        normalization_t_per_a=_freeze(normalizations),
        normalized_magnitude=_freeze(normalized_magnitude),
        channel_labels=tuple(multiport.labels),
        input_impedance_ohm=_freeze(input_impedance),
        noise_covariance_v2_per_hz=_freeze(noise),
    )
