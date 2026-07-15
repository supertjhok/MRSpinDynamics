"""Transient cross-talk during multi-element EPM programming.

The isolated programming driver is sufficient when neighboring windings and
magnetic states are fixed.  An array needs two additional interactions:

* mutual inductance creates induced current and voltage in inactive windings;
* winding leakage plus retained-magnet fields can move every element along its
  own return-point trajectory during a pulse.

This module integrates the mutually coupled winding circuit, then advances all
weighted-play states sample by sample.  State-dependent self inductance is
closed with an outer fixed-point iteration.  Mutual inductance and off-axis
winding-field fractions are explicit, evidence-tagged inputs because the local
archive does not contain a complete measured coupling matrix.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.fields.electropermanent import (
    AlNiCoMaterial,
    ElectropermanentRod,
    ElectropermanentSource,
    EvidenceRecord,
    RemanenceState,
)
from spin_dynamics.fields.electropermanent_hysteresis import (
    PlayHysteresisModel,
    ReturnPointState,
)
from spin_dynamics.fields.electropermanent_pulses import (
    ProgrammingPulse,
    PulsePowerDriver,
    PulseThermalState,
)


def _integral(values: np.ndarray, times_s: np.ndarray, *, axis: int = 0) -> np.ndarray:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return np.asarray(integrator(values, times_s, axis=axis))


def _source_material(source: ElectropermanentSource) -> AlNiCoMaterial:
    if isinstance(source, ElectropermanentRod):
        return source.material
    material = source.rods[0].material
    if any(rod.material != material for rod in source.rods):
        raise ValueError("transient programming requires one material per bundle")
    return material


def _source_state(source: ElectropermanentSource) -> RemanenceState:
    if isinstance(source, ElectropermanentRod):
        return source.state
    state = source.rods[0].state
    if any(rod.state != state for rod in source.rods):
        raise ValueError("transient programming requires one shared state per bundle")
    return state


def _validated_coupling_coefficients(
    values: Sequence[Sequence[float]] | np.ndarray,
    count: int,
) -> np.ndarray:
    coefficients = np.asarray(values, dtype=np.float64)
    if coefficients.shape != (count, count) or np.any(~np.isfinite(coefficients)):
        raise ValueError("coupling_coefficients must be a finite square matrix")
    if not np.allclose(coefficients, coefficients.T, rtol=1e-12, atol=1e-15):
        raise ValueError("coupling_coefficients must be symmetric")
    if not np.allclose(np.diag(coefficients), 0.0, atol=1e-15):
        raise ValueError("coupling_coefficients must have a zero diagonal")
    normalized = np.eye(count) + coefficients
    if np.min(np.linalg.eigvalsh(normalized)) <= 0.0:
        raise ValueError("coupling_coefficients produce a non-positive inductance matrix")
    return coefficients


@dataclass(frozen=True)
class ArrayPulseWaveform:
    """Electrical, magnetic, and thermal histories for all EPM windings."""

    circuit: MutualProgrammingCircuit
    target_index: int
    pulse: ProgrammingPulse
    times_s: np.ndarray
    currents_a: np.ndarray
    capacitor_voltage_v: np.ndarray
    applied_voltage_v: np.ndarray
    mutual_induced_voltage_v: np.ndarray
    winding_h_a_per_m: np.ndarray
    static_bias_h_a_per_m: np.ndarray
    internal_h_a_per_m: np.ndarray
    coil_temperature_k: np.ndarray
    driver_temperature_k: np.ndarray
    coil_resistance_ohm: np.ndarray
    coil_power_w: np.ndarray
    driver_power_w: np.ndarray
    switch_state: np.ndarray
    inductance_matrix_h: np.ndarray
    initial_currents_a: np.ndarray
    initial_thermal_states: tuple[PulseThermalState, ...]

    @property
    def peak_current_a(self) -> np.ndarray:
        """Per-winding peak absolute current."""

        return np.max(np.abs(self.currents_a), axis=0)

    @property
    def peak_internal_h_a_per_m(self) -> np.ndarray:
        """Per-element peak absolute internal field."""

        return np.max(np.abs(self.internal_h_a_per_m), axis=0)

    @property
    def peak_mutual_induced_voltage_v(self) -> np.ndarray:
        """Per-winding peak absolute mutual-induction voltage."""

        return np.max(np.abs(self.mutual_induced_voltage_v), axis=0)

    @property
    def coil_energy_j(self) -> np.ndarray:
        """Integrated Joule loss in every winding."""

        return _integral(self.coil_power_w, self.times_s)

    @property
    def driver_conduction_energy_j(self) -> np.ndarray:
        """Integrated series/recovery-path loss for every winding."""

        return _integral(self.driver_power_w, self.times_s)

    @property
    def switching_energy_j(self) -> float:
        """Discrete turn-on and turn-off loss for the commanded bridge."""

        return 2.0 * self.circuit.drivers[self.target_index].switching_energy_per_transition_j

    @property
    def initial_electrical_energy_j(self) -> float:
        """Initial capacitor plus coupled-inductor energy."""

        driver = self.circuit.drivers[self.target_index]
        magnetic = 0.5 * self.initial_currents_a @ self.inductance_matrix_h @ self.initial_currents_a
        return float(0.5 * driver.capacitance_f * self.pulse.capacitor_voltage_v**2 + magnetic)

    @property
    def final_electrical_energy_j(self) -> float:
        """Final capacitor plus coupled-inductor energy."""

        driver = self.circuit.drivers[self.target_index]
        current = self.currents_a[-1]
        magnetic = 0.5 * current @ self.inductance_matrix_h @ current
        return float(0.5 * driver.capacitance_f * self.capacitor_voltage_v[-1] ** 2 + magnetic)

    @property
    def electrical_energy_balance_error_j(self) -> float:
        """Energy residual excluding discrete switching-loss metadata."""

        losses = np.sum(self.coil_energy_j) + np.sum(self.driver_conduction_energy_j)
        return float(self.initial_electrical_energy_j - self.final_electrical_energy_j - losses)

    @property
    def final_thermal_states(self) -> tuple[PulseThermalState, ...]:
        """Per-channel thermal states ready for a later array command."""

        coil_energy = self.coil_energy_j
        driver_energy = self.driver_conduction_energy_j
        result = []
        for index, initial in enumerate(self.initial_thermal_states):
            switching = self.switching_energy_j if index == self.target_index else 0.0
            result.append(
                PulseThermalState(
                    coil_temperature_k=float(self.coil_temperature_k[-1, index]),
                    driver_temperature_k=float(self.driver_temperature_k[-1, index]),
                    cumulative_coil_energy_j=(
                        initial.cumulative_coil_energy_j + float(coil_energy[index])
                    ),
                    cumulative_driver_energy_j=(
                        initial.cumulative_driver_energy_j
                        + float(driver_energy[index])
                        + switching
                    ),
                )
            )
        return tuple(result)


@dataclass(frozen=True)
class MutualProgrammingCircuit:
    """Mutually coupled EPM windings with one commanded bridge per pulse.

    ``mutual_inductance_h`` contains off-diagonal signed mutual inductances and
    must have a zero diagonal.  ``field_coupling_a_per_m_per_a[i, j]`` is the
    programming field at magnet ``i`` per ampere in winding ``j``.  Inactive
    windings are closed through their recovery resistors, which represents the
    worst-case induced-current path rather than an open-circuit terminal.
    """

    drivers: tuple[PulsePowerDriver, ...]
    mutual_inductance_h: np.ndarray
    field_coupling_a_per_m_per_a: np.ndarray
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        drivers = tuple(self.drivers)
        if not drivers:
            raise ValueError("drivers must not be empty")
        count = len(drivers)
        mutual = np.asarray(self.mutual_inductance_h, dtype=np.float64)
        if mutual.shape != (count, count) or np.any(~np.isfinite(mutual)):
            raise ValueError("mutual_inductance_h must be a finite square matrix")
        if not np.allclose(mutual, mutual.T, rtol=1e-12, atol=1e-15):
            raise ValueError("mutual_inductance_h must be symmetric")
        if not np.allclose(np.diag(mutual), 0.0, atol=1e-15):
            raise ValueError("mutual_inductance_h must have a zero diagonal")
        field = np.asarray(self.field_coupling_a_per_m_per_a, dtype=np.float64)
        if field.shape != (count, count) or np.any(~np.isfinite(field)):
            raise ValueError("field_coupling_a_per_m_per_a must be a finite square matrix")
        nominal = np.diag([driver.inductance_h for driver in drivers]) + mutual
        if np.min(np.linalg.eigvalsh(nominal)) <= 0.0:
            raise ValueError("nominal coupled inductance matrix must be positive definite")
        object.__setattr__(self, "drivers", drivers)
        object.__setattr__(self, "mutual_inductance_h", mutual)
        object.__setattr__(self, "field_coupling_a_per_m_per_a", field)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @classmethod
    def from_coupling_coefficients(
        cls,
        drivers: Sequence[PulsePowerDriver],
        coupling_coefficients: Sequence[Sequence[float]] | np.ndarray,
        *,
        field_coupling_fractions: Sequence[Sequence[float]] | np.ndarray | None = None,
        label: str = "",
        evidence: Sequence[EvidenceRecord] = (),
    ) -> MutualProgrammingCircuit:
        """Build mutual inductance from dimensionless winding coefficients.

        Off-diagonal ``k[i, j]`` gives ``M[i, j] / sqrt(L_i L_j)``.  Field
        fractions are referenced to the source winding: fraction ``f[i, j]``
        multiplies winding ``j``'s own ``H/I`` efficiency at magnet ``i``.
        """

        channels = tuple(drivers)
        count = len(channels)
        if not count:
            raise ValueError("drivers must not be empty")
        coefficients = _validated_coupling_coefficients(coupling_coefficients, count)
        nominal_l = np.asarray([driver.inductance_h for driver in channels])
        mutual = coefficients * np.sqrt(np.outer(nominal_l, nominal_l))
        if field_coupling_fractions is None:
            fractions = np.eye(count)
        else:
            fractions = np.asarray(field_coupling_fractions, dtype=np.float64)
            if fractions.shape != (count, count) or np.any(~np.isfinite(fractions)):
                raise ValueError("field_coupling_fractions must be a finite square matrix")
            if not np.allclose(np.diag(fractions), 1.0, rtol=1e-12, atol=1e-15):
                raise ValueError("field_coupling_fractions must have a unit diagonal")
        own_efficiency = np.asarray([driver.programming_field(1.0) for driver in channels])
        field = fractions * own_efficiency[None, :]
        return cls(channels, mutual, field, label=label, evidence=tuple(evidence))

    def inductance_matrix(
        self,
        states: Sequence[RemanenceState],
        materials: Sequence[AlNiCoMaterial],
    ) -> np.ndarray:
        """Return the state-dependent positive-definite inductance matrix."""

        if len(states) != len(self.drivers) or len(materials) != len(self.drivers):
            raise ValueError("states and materials must match drivers")
        diagonal = [
            driver.inductance_for_state(state, material)
            for driver, state, material in zip(self.drivers, states, materials)
        ]
        matrix = np.diag(diagonal) + self.mutual_inductance_h
        if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
            raise ValueError("state-dependent coupled inductance matrix is not positive definite")
        return matrix

    def simulate(
        self,
        target_index: int,
        times_s: Sequence[float] | np.ndarray,
        pulse: ProgrammingPulse,
        *,
        states: Sequence[RemanenceState],
        materials: Sequence[AlNiCoMaterial],
        static_bias_h_a_per_m: Sequence[float] | np.ndarray | None = None,
        initial_currents_a: Sequence[float] | np.ndarray | None = None,
        initial_thermal_states: Sequence[PulseThermalState] | None = None,
        max_step_s: float | None = None,
    ) -> ArrayPulseWaveform:
        """Integrate one commanded channel and all shorted neighbor windings."""

        count = len(self.drivers)
        index = int(target_index)
        if index != target_index or not 0 <= index < count:
            raise ValueError("target_index is out of range")
        times = np.asarray(times_s, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size < 2
            or np.any(~np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times_s must be finite, increasing, 1-D, and non-negative")
        if pulse.capacitor_voltage_v > self.drivers[index].voltage_limit_v:
            raise ValueError("pulse capacitor voltage exceeds the target driver limit")
        public_states = tuple(states)
        material_records = tuple(materials)
        inductance = self.inductance_matrix(public_states, material_records)
        if static_bias_h_a_per_m is None:
            static_bias = np.zeros(count)
        else:
            static_bias = np.asarray(static_bias_h_a_per_m, dtype=np.float64)
            if static_bias.shape != (count,) or np.any(~np.isfinite(static_bias)):
                raise ValueError("static_bias_h_a_per_m must match drivers")
        if initial_currents_a is None:
            initial_current = np.zeros(count)
        else:
            initial_current = np.asarray(initial_currents_a, dtype=np.float64)
            if initial_current.shape != (count,) or np.any(~np.isfinite(initial_current)):
                raise ValueError("initial_currents_a must match drivers")
        if initial_thermal_states is None:
            thermal = tuple(
                PulseThermalState(driver.ambient_temperature_k, driver.ambient_temperature_k)
                for driver in self.drivers
            )
        else:
            thermal = tuple(initial_thermal_states)
            if len(thermal) != count:
                raise ValueError("initial_thermal_states must match drivers")
        default_step = min(pulse.duration_s / 100.0, times[-1] / 2000.0)
        step_limit = default_step if max_step_s is None else float(max_step_s)
        if not np.isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("max_step_s must be finite and positive")

        initial = np.concatenate(
            (
                [pulse.capacitor_voltage_v],
                initial_current,
                [state.coil_temperature_k for state in thermal],
                [state.driver_temperature_k for state in thermal],
            )
        ).astype(np.float64)

        def derivative(time_s: float, values: np.ndarray) -> np.ndarray:
            capacitor_v = values[0]
            currents = values[1 : 1 + count]
            coil_temperature = values[1 + count : 1 + 2 * count]
            driver_temperature = values[1 + 2 * count :]
            coil_r = np.asarray(
                [
                    driver.coil_resistance(coil_temperature[channel])
                    for channel, driver in enumerate(self.drivers)
                ]
            )
            driving = time_s < pulse.duration_s
            external_r = np.asarray(
                [driver.recovery_resistance_ohm for driver in self.drivers],
                dtype=np.float64,
            )
            voltage = np.zeros(count)
            if driving:
                target = self.drivers[index]
                external_r[index] = target.series_resistance_ohm
                voltage[index] = pulse.polarity * max(capacitor_v - target.bridge_drop_v, 0.0)
            current_derivative = np.linalg.solve(
                inductance,
                voltage - (coil_r + external_r) * currents,
            )
            capacitor_derivative = (
                -pulse.polarity * currents[index] / self.drivers[index].capacitance_f
                if driving
                else 0.0
            )
            coil_power = currents**2 * coil_r
            driver_power = currents**2 * external_r
            coil_temperature_derivative = np.asarray(
                [
                    (
                        coil_power[channel]
                        - driver.coil_thermal_conductance_w_per_k
                        * (coil_temperature[channel] - driver.ambient_temperature_k)
                    )
                    / driver.coil_heat_capacity_j_per_k
                    for channel, driver in enumerate(self.drivers)
                ]
            )
            driver_temperature_derivative = np.asarray(
                [
                    (
                        driver_power[channel]
                        - driver.driver_thermal_conductance_w_per_k
                        * (driver_temperature[channel] - driver.ambient_temperature_k)
                    )
                    / driver.driver_heat_capacity_j_per_k
                    for channel, driver in enumerate(self.drivers)
                ]
            )
            return np.concatenate(
                (
                    [capacitor_derivative],
                    current_derivative,
                    coil_temperature_derivative,
                    driver_temperature_derivative,
                )
            )

        def rk4(time_s: float, values: np.ndarray, step_s: float) -> np.ndarray:
            first = derivative(time_s, values)
            second = derivative(time_s + 0.5 * step_s, values + 0.5 * step_s * first)
            third = derivative(time_s + 0.5 * step_s, values + 0.5 * step_s * second)
            fourth = derivative(time_s + step_s, values + step_s * third)
            advanced = values + step_s * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
            advanced[0] = max(advanced[0], 0.0)
            limits = np.asarray([driver.current_limit_a for driver in self.drivers])
            advanced[1 : 1 + count] = np.clip(advanced[1 : 1 + count], -limits, limits)
            return advanced

        values = initial.copy()
        current_time = 0.0
        integrated = np.empty((times.size, initial.size), dtype=np.float64)
        for sample, target_time in enumerate(times):
            boundaries = [target_time]
            if current_time < pulse.duration_s < target_time:
                boundaries.insert(0, pulse.duration_s)
            for boundary in boundaries:
                interval = boundary - current_time
                steps = max(1, int(np.ceil(interval / step_limit))) if interval else 0
                if steps:
                    step = interval / steps
                    for _ in range(steps):
                        values = rk4(current_time, values, step)
                        current_time += step
            integrated[sample] = values

        capacitor = integrated[:, 0]
        currents = integrated[:, 1 : 1 + count]
        coil_temperature = integrated[:, 1 + count : 1 + 2 * count]
        driver_temperature = integrated[:, 1 + 2 * count :]
        coil_r = np.column_stack(
            [
                driver.coil_resistance(coil_temperature[:, channel])
                for channel, driver in enumerate(self.drivers)
            ]
        )
        applied = np.zeros_like(currents)
        external_r = np.empty_like(currents)
        derivatives = np.empty_like(currents)
        driving_samples = times < pulse.duration_s
        for sample, driving in enumerate(driving_samples):
            external_r[sample] = [driver.recovery_resistance_ohm for driver in self.drivers]
            if driving:
                target = self.drivers[index]
                external_r[sample, index] = target.series_resistance_ohm
                applied[sample, index] = pulse.polarity * max(
                    capacitor[sample] - target.bridge_drop_v,
                    0.0,
                )
            derivatives[sample] = np.linalg.solve(
                inductance,
                applied[sample] - (coil_r[sample] + external_r[sample]) * currents[sample],
            )
        induced = -(self.mutual_inductance_h @ derivatives.T).T
        winding_h = currents @ self.field_coupling_a_per_m_per_a.T
        static_history = np.broadcast_to(static_bias, winding_h.shape).copy()
        internal_h = winding_h + static_history
        return ArrayPulseWaveform(
            circuit=self,
            target_index=index,
            pulse=pulse,
            times_s=times,
            currents_a=currents,
            capacitor_voltage_v=capacitor,
            applied_voltage_v=applied,
            mutual_induced_voltage_v=induced,
            winding_h_a_per_m=winding_h,
            static_bias_h_a_per_m=static_history,
            internal_h_a_per_m=internal_h,
            coil_temperature_k=coil_temperature,
            driver_temperature_k=driver_temperature,
            coil_resistance_ohm=coil_r,
            coil_power_w=currents**2 * coil_r,
            driver_power_w=currents**2 * external_r,
            switch_state=np.where(driving_samples, "drive", "recovery"),
            inductance_matrix_h=inductance,
            initial_currents_a=initial_current,
            initial_thermal_states=thermal,
        )


@dataclass(frozen=True)
class TransientProgrammingResult:
    """Converged multi-element state update for one array programming pulse."""

    target_index: int
    initial_states: tuple[ReturnPointState, ...]
    final_states: tuple[ReturnPointState, ...]
    final_sources: tuple[ElectropermanentSource, ...]
    waveform: ArrayPulseWaveform
    remanence_t: np.ndarray
    iterations: int
    residual_t: float
    converged: bool

    @property
    def remanence_change_t(self) -> np.ndarray:
        """Per-element retained-remanence change."""

        initial = np.asarray([state.remanence.remanence_t for state in self.initial_states])
        final = np.asarray([state.remanence.remanence_t for state in self.final_states])
        return final - initial

    @property
    def disturbed_indices(self) -> tuple[int, ...]:
        """Non-target elements whose retained remanence changed numerically."""

        return tuple(
            index
            for index, change in enumerate(self.remanence_change_t)
            if index != self.target_index and abs(change) > 1e-12
        )


@dataclass(frozen=True)
class TransientCoupledEPMProgrammer:
    """Pulse, mutual-inductance, and return-point update for an EPM array."""

    sources: tuple[ElectropermanentSource, ...]
    hysteresis_models: tuple[PlayHysteresisModel, ...]
    remanence_coupling_a_per_m_per_t: np.ndarray
    circuit: MutualProgrammingCircuit

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        models = tuple(self.hysteresis_models)
        count = len(sources)
        if not count or len(models) != count or len(self.circuit.drivers) != count:
            raise ValueError("sources, models, and circuit channels must have equal nonzero length")
        coupling = np.asarray(self.remanence_coupling_a_per_m_per_t, dtype=np.float64)
        if coupling.shape != (count, count) or np.any(~np.isfinite(coupling)):
            raise ValueError("remanence_coupling_a_per_m_per_t must be a finite square matrix")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "hysteresis_models", models)
        object.__setattr__(self, "remanence_coupling_a_per_m_per_t", coupling)

    def initial_states(self) -> tuple[ReturnPointState, ...]:
        """Initialize hidden return-point coordinates from all source states."""

        return tuple(
            model.initialize(_source_state(source))
            for source, model in zip(self.sources, self.hysteresis_models)
        )

    def _propagate_states(
        self,
        initial: tuple[ReturnPointState, ...],
        waveform: ArrayPulseWaveform,
        *,
        field_tolerance_a_per_m: float,
        max_field_iterations: int,
        field_relaxation: float,
    ) -> tuple[tuple[ReturnPointState, ...], np.ndarray, np.ndarray]:
        current = initial
        field_history = np.empty_like(waveform.winding_h_a_per_m)
        remanence_history = np.empty_like(waveform.winding_h_a_per_m)
        for sample in range(waveform.times_s.size):
            guess = np.asarray([state.remanence.remanence_t for state in current])
            candidate = current
            field = waveform.winding_h_a_per_m[sample].copy()
            for _ in range(max_field_iterations):
                field = (
                    waveform.winding_h_a_per_m[sample]
                    + self.remanence_coupling_a_per_m_per_t @ guess
                )
                candidate = tuple(
                    model.propagate(
                        state,
                        [field[index]],
                        temperatures_k=[waveform.coil_temperature_k[sample, index]],
                    ).final_state
                    for index, (model, state) in enumerate(
                        zip(self.hysteresis_models, current)
                    )
                )
                updated = np.asarray(
                    [state.remanence.remanence_t for state in candidate]
                )
                field_error = np.max(
                    np.abs(
                        self.remanence_coupling_a_per_m_per_t @ (updated - guess)
                    )
                )
                if field_error <= field_tolerance_a_per_m:
                    break
                guess = (1.0 - field_relaxation) * guess + field_relaxation * updated
            else:
                raise RuntimeError("transient magnetostatic state iteration did not converge")
            current = candidate
            field_history[sample] = field
            remanence_history[sample] = [
                state.remanence.remanence_t for state in current
            ]
        return current, field_history, remanence_history

    def program(
        self,
        target_index: int,
        times_s: Sequence[float] | np.ndarray,
        pulse: ProgrammingPulse,
        *,
        states: Sequence[ReturnPointState] | None = None,
        initial_thermal_states: Sequence[PulseThermalState] | None = None,
        tolerance_t: float = 1e-7,
        max_iterations: int = 30,
        relaxation: float = 0.7,
        field_tolerance_a_per_m: float = 1e-3,
        max_field_iterations: int = 50,
        field_relaxation: float = 0.7,
        max_step_s: float | None = None,
        raise_on_nonconvergence: bool = True,
    ) -> TransientProgrammingResult:
        """Program one element while every winding and magnetic state responds."""

        count = len(self.sources)
        index = int(target_index)
        if index != target_index or not 0 <= index < count:
            raise ValueError("target_index is out of range")
        initial = self.initial_states() if states is None else tuple(states)
        if len(initial) != count:
            raise ValueError("states must match sources")
        if not np.isfinite(tolerance_t) or tolerance_t <= 0.0:
            raise ValueError("tolerance_t must be finite and positive")
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")
        if not np.isfinite(field_tolerance_a_per_m) or field_tolerance_a_per_m <= 0.0:
            raise ValueError("field_tolerance_a_per_m must be finite and positive")
        if int(max_field_iterations) != max_field_iterations or max_field_iterations < 1:
            raise ValueError("max_field_iterations must be a positive integer")
        if not np.isfinite(field_relaxation) or not 0.0 < field_relaxation <= 1.0:
            raise ValueError("field_relaxation must lie in (0, 1]")

        materials = tuple(_source_material(source) for source in self.sources)
        guess = np.asarray([state.remanence.remanence_t for state in initial])
        final_states = initial
        waveform: ArrayPulseWaveform | None = None
        residual = np.inf
        converged = False
        for iteration in range(1, max_iterations + 1):
            guessed_public = tuple(
                RemanenceState(
                    float(value),
                    branch="partial",
                    temperature_k=initial[channel].remanence.temperature_k,
                    calibration_id=initial[channel].remanence.calibration_id,
                    uncertainty_t=initial[channel].remanence.uncertainty_t,
                    evidence=initial[channel].remanence.evidence,
                )
                for channel, value in enumerate(guess)
            )
            waveform = self.circuit.simulate(
                index,
                times_s,
                pulse,
                states=guessed_public,
                materials=materials,
                initial_thermal_states=initial_thermal_states,
                max_step_s=max_step_s,
            )
            final_states, field_history, remanence_history = self._propagate_states(
                initial,
                waveform,
                field_tolerance_a_per_m=field_tolerance_a_per_m,
                max_field_iterations=max_field_iterations,
                field_relaxation=field_relaxation,
            )
            updated = np.asarray(
                [state.remanence.remanence_t for state in final_states]
            )
            residual = float(np.max(np.abs(updated - guess)))
            static_history = field_history - waveform.winding_h_a_per_m
            waveform = replace(
                waveform,
                static_bias_h_a_per_m=static_history,
                internal_h_a_per_m=field_history,
            )
            if residual <= tolerance_t:
                converged = True
                break
            guess = (1.0 - relaxation) * guess + relaxation * updated
        if waveform is None:  # pragma: no cover - protected by validation
            raise RuntimeError("transient programming iteration did not start")
        if not converged and raise_on_nonconvergence:
            raise RuntimeError(
                "transient coupled EPM programming did not converge: "
                f"residual={residual:.3g} T after {max_iterations} iterations"
            )
        final_sources = tuple(
            source.with_state(state.remanence)
            for source, state in zip(self.sources, final_states)
        )
        return TransientProgrammingResult(
            target_index=index,
            initial_states=initial,
            final_states=final_states,
            final_sources=final_sources,
            waveform=waveform,
            remanence_t=remanence_history,
            iterations=iteration,
            residual_t=residual,
            converged=converged,
        )


__all__ = [
    "ArrayPulseWaveform",
    "MutualProgrammingCircuit",
    "TransientProgrammingResult",
    "TransientCoupledEPMProgrammer",
]
