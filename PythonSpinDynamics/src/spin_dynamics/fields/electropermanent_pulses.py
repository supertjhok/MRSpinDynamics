"""Programming-pulse and empirical state models for AlNiCo EPM sources.

This module is deliberately split from the static magnet model.  A capacitor
bank and H-bridge determine the realized current and internal programming
field, while a protocol-specific calibration maps a measured waveform metric
to a retained remanence.  A material B-H table is not treated as a hysteresis
history model.

The electrical integrator uses only NumPy.  During the commanded interval it
solves a series RLC discharge; after turn-off it follows a resistive recovery
path.  Coil and driver temperatures are lumped states with independent heat
capacities and ambient conductances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.electropermanent import (
    AlNiCoMaterial,
    EvidenceRecord,
    ProgrammingCoil,
    RemanenceBranch,
    RemanenceState,
)

ProgrammingMetric = Literal[
    "duty_fraction",
    "peak_internal_h",
    "absolute_h_impulse",
    "signed_h_impulse",
]
ExtrapolationMode = Literal["clip", "raise"]


def _integral(values: np.ndarray, times_s: np.ndarray) -> float:
    integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrator(values, times_s))


@dataclass(frozen=True)
class PulseThermalState:
    """Temperatures and cumulative losses carried between programming pulses."""

    coil_temperature_k: float = 293.15
    driver_temperature_k: float = 293.15
    cumulative_coil_energy_j: float = 0.0
    cumulative_driver_energy_j: float = 0.0

    def __post_init__(self) -> None:
        for name in ("coil_temperature_k", "driver_temperature_k"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("cumulative_coil_energy_j", "cumulative_driver_energy_j"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ProgrammingPulse:
    """One capacitor-bank command applied through an H-bridge.

    ``capacitor_voltage_v`` is the charged-bank magnitude and ``polarity`` is
    the signed bridge direction.  ``command_fraction`` carries a duty-cycle or
    normalized command only when a calibration explicitly uses that protocol;
    it does not alter the circuit waveform.
    """

    capacitor_voltage_v: float
    duration_s: float
    polarity: int = 1
    command_fraction: float | None = None
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.capacitor_voltage_v) or self.capacitor_voltage_v <= 0:
            raise ValueError("capacitor_voltage_v must be finite and positive")
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if self.polarity not in {-1, 1}:
            raise ValueError("polarity must be -1 or +1")
        if self.command_fraction is not None and (
            not np.isfinite(self.command_fraction)
            or not 0.0 <= self.command_fraction <= 1.0
        ):
            raise ValueError("command_fraction must lie in [0, 1] when supplied")
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True)
class PulsePowerDriver:
    """Capacitor/H-bridge/series-RLC model for an EPM programming winding.

    The nominal programming field is

    ``H = field_efficiency * turns * I / magnetic_path_length``.

    This is a winding-field estimate, not a solved internal AlNiCo field.  A
    static neighbor or demagnetizing contribution can be supplied to
    :meth:`simulate` through ``bias_field_a_per_m``.
    """

    capacitance_f: float
    inductance_h: float
    coil_resistance_ohm: float
    turns: int
    magnetic_path_length_m: float
    series_resistance_ohm: float = 0.0
    recovery_resistance_ohm: float = 1.0
    bridge_drop_v: float = 0.0
    current_limit_a: float = np.inf
    voltage_limit_v: float = np.inf
    field_efficiency: float = 1.0
    reference_temperature_k: float = 293.15
    resistance_temperature_coefficient_per_k: float = 0.00393
    state_inductance_fraction_at_saturation: float = 0.0
    ambient_temperature_k: float = 293.15
    coil_heat_capacity_j_per_k: float = 100.0
    coil_thermal_conductance_w_per_k: float = 1.0
    driver_heat_capacity_j_per_k: float = 50.0
    driver_thermal_conductance_w_per_k: float = 1.0
    switching_energy_per_transition_j: float = 0.0
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "capacitance_f",
            "inductance_h",
            "coil_resistance_ohm",
            "magnetic_path_length_m",
            "field_efficiency",
            "reference_temperature_k",
            "ambient_temperature_k",
            "coil_heat_capacity_j_per_k",
            "driver_heat_capacity_j_per_k",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.turns) != self.turns or self.turns <= 0:
            raise ValueError("turns must be a positive integer")
        for name in (
            "series_resistance_ohm",
            "recovery_resistance_ohm",
            "bridge_drop_v",
            "coil_thermal_conductance_w_per_k",
            "driver_thermal_conductance_w_per_k",
            "switching_energy_per_transition_j",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("current_limit_a", "voltage_limit_v"):
            value = float(getattr(self, name))
            if np.isnan(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and not NaN")
        alpha = float(self.resistance_temperature_coefficient_per_k)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError(
                "resistance_temperature_coefficient_per_k must be finite and non-negative"
            )
        fraction = float(self.state_inductance_fraction_at_saturation)
        if not np.isfinite(fraction) or fraction <= -1.0:
            raise ValueError(
                "state_inductance_fraction_at_saturation must be finite and greater than -1"
            )
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @classmethod
    def from_programming_coil(
        cls,
        coil: ProgrammingCoil,
        *,
        capacitance_f: float,
        magnetic_path_length_m: float | None = None,
        **kwargs,
    ) -> PulsePowerDriver:
        """Build a driver from a fully specified static ``ProgrammingCoil``."""

        if coil.inductance_h is None or coil.resistance_ohm is None:
            raise ValueError("coil inductance_h and resistance_ohm are required")
        path_length = (
            coil.winding_length_m
            if magnetic_path_length_m is None
            else magnetic_path_length_m
        )
        if path_length is None:
            raise ValueError(
                "magnetic_path_length_m or coil.winding_length_m is required"
            )
        return cls(
            capacitance_f=capacitance_f,
            inductance_h=coil.inductance_h,
            coil_resistance_ohm=coil.resistance_ohm,
            turns=coil.turns,
            magnetic_path_length_m=path_length,
            **kwargs,
        )

    def inductance_for_state(
        self,
        state: RemanenceState | None = None,
        material: AlNiCoMaterial | None = None,
    ) -> float:
        """Return the fixed inductance used for one pulse integration."""

        fraction = 0.0
        if state is not None:
            if material is None:
                raise ValueError("material is required when state is supplied")
            fraction = abs(state.fraction_of(material))
        return float(
            self.inductance_h
            * (1.0 + self.state_inductance_fraction_at_saturation * fraction)
        )

    def coil_resistance(self, temperature_k: float | np.ndarray) -> float | np.ndarray:
        """Return winding resistance at one or more temperatures."""

        temperature = np.asarray(temperature_k, dtype=np.float64)
        result = self.coil_resistance_ohm * (
            1.0
            + self.resistance_temperature_coefficient_per_k
            * (temperature - self.reference_temperature_k)
        )
        if np.any(result <= 0.0) or not np.all(np.isfinite(result)):
            raise ValueError("temperature produces a non-positive coil resistance")
        return float(result) if result.ndim == 0 else result

    def programming_field(self, current_a: float | np.ndarray) -> float | np.ndarray:
        """Return the nominal winding field H in A/m."""

        result = (
            self.field_efficiency
            * self.turns
            * np.asarray(current_a, dtype=np.float64)
            / self.magnetic_path_length_m
        )
        return float(result) if result.ndim == 0 else result

    def cool_thermal_state(
        self,
        state: PulseThermalState,
        duration_s: float,
    ) -> PulseThermalState:
        """Apply exact zero-current lumped cooling between pulses."""

        if not np.isfinite(duration_s) or duration_s < 0.0:
            raise ValueError("duration_s must be finite and non-negative")

        def cool(value: float, conductance: float, capacity: float) -> float:
            if conductance == 0.0:
                return value
            factor = np.exp(-conductance * duration_s / capacity)
            return float(self.ambient_temperature_k + (value - self.ambient_temperature_k) * factor)

        return PulseThermalState(
            coil_temperature_k=cool(
                state.coil_temperature_k,
                self.coil_thermal_conductance_w_per_k,
                self.coil_heat_capacity_j_per_k,
            ),
            driver_temperature_k=cool(
                state.driver_temperature_k,
                self.driver_thermal_conductance_w_per_k,
                self.driver_heat_capacity_j_per_k,
            ),
            cumulative_coil_energy_j=state.cumulative_coil_energy_j,
            cumulative_driver_energy_j=state.cumulative_driver_energy_j,
        )

    def simulate(
        self,
        times_s: np.ndarray,
        pulse: ProgrammingPulse,
        *,
        initial_thermal_state: PulseThermalState | None = None,
        initial_current_a: float = 0.0,
        state: RemanenceState | None = None,
        material: AlNiCoMaterial | None = None,
        bias_field_a_per_m: float = 0.0,
        max_step_s: float | None = None,
    ) -> PulseWaveform:
        """Integrate one drive interval followed by passive current recovery."""

        times = np.asarray(times_s, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.all(np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError("times_s must be finite, increasing, one-dimensional, and non-negative")
        if pulse.capacitor_voltage_v > self.voltage_limit_v:
            raise ValueError("pulse capacitor voltage exceeds driver voltage_limit_v")
        if not np.isfinite(initial_current_a):
            raise ValueError("initial_current_a must be finite")
        if not np.isfinite(bias_field_a_per_m):
            raise ValueError("bias_field_a_per_m must be finite")
        thermal = initial_thermal_state or PulseThermalState(
            self.ambient_temperature_k,
            self.ambient_temperature_k,
        )
        inductance = self.inductance_for_state(state, material)
        default_step = min(pulse.duration_s / 100.0, times[-1] / 2000.0)
        step_limit = default_step if max_step_s is None else float(max_step_s)
        if not np.isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("max_step_s must be finite and positive")

        initial = np.asarray(
            [
                pulse.capacitor_voltage_v,
                initial_current_a,
                thermal.coil_temperature_k,
                thermal.driver_temperature_k,
            ],
            dtype=np.float64,
        )

        def derivative(time_s: float, values: np.ndarray) -> np.ndarray:
            capacitor_v, current, coil_temperature, driver_temperature = values
            coil_r = float(self.coil_resistance(coil_temperature))
            driving = time_s < pulse.duration_s
            if driving:
                available = max(capacitor_v - self.bridge_drop_v, 0.0)
                applied = pulse.polarity * available
                current_derivative = (
                    applied - current * (coil_r + self.series_resistance_ohm)
                ) / inductance
                at_limit = (
                    pulse.polarity * current >= self.current_limit_a
                    and pulse.polarity * current_derivative > 0.0
                )
                if at_limit:
                    current_derivative = 0.0
                    capacitor_derivative = -(
                        current**2 * (coil_r + self.series_resistance_ohm)
                    ) / max(self.capacitance_f * capacitor_v, 1e-30)
                else:
                    capacitor_derivative = -pulse.polarity * current / self.capacitance_f
                driver_power = current**2 * self.series_resistance_ohm
            else:
                capacitor_derivative = 0.0
                current_derivative = -(
                    coil_r + self.recovery_resistance_ohm
                ) * current / inductance
                driver_power = current**2 * self.recovery_resistance_ohm
            coil_power = current**2 * coil_r
            coil_temperature_derivative = (
                coil_power
                - self.coil_thermal_conductance_w_per_k
                * (coil_temperature - self.ambient_temperature_k)
            ) / self.coil_heat_capacity_j_per_k
            driver_temperature_derivative = (
                driver_power
                - self.driver_thermal_conductance_w_per_k
                * (driver_temperature - self.ambient_temperature_k)
            ) / self.driver_heat_capacity_j_per_k
            return np.asarray(
                [
                    capacitor_derivative,
                    current_derivative,
                    coil_temperature_derivative,
                    driver_temperature_derivative,
                ]
            )

        def rk4(time_s: float, values: np.ndarray, step_s: float) -> np.ndarray:
            first = derivative(time_s, values)
            second = derivative(time_s + 0.5 * step_s, values + 0.5 * step_s * first)
            third = derivative(time_s + 0.5 * step_s, values + 0.5 * step_s * second)
            fourth = derivative(time_s + step_s, values + step_s * third)
            advanced = values + step_s * (first + 2 * second + 2 * third + fourth) / 6.0
            advanced[0] = max(advanced[0], 0.0)
            advanced[1] = float(
                np.clip(advanced[1], -self.current_limit_a, self.current_limit_a)
            )
            return advanced

        states = np.empty((times.size, 4), dtype=np.float64)
        values = initial.copy()
        current_time = 0.0
        for index, target_time in enumerate(times):
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
            states[index] = values

        capacitor_voltage = states[:, 0]
        current = states[:, 1]
        coil_temperature = states[:, 2]
        driver_temperature = states[:, 3]
        coil_resistance = np.asarray(self.coil_resistance(coil_temperature))
        driving = times < pulse.duration_s
        bridge_voltage = np.where(
            driving,
            pulse.polarity * np.maximum(capacitor_voltage - self.bridge_drop_v, 0.0),
            -self.recovery_resistance_ohm * current,
        )
        coil_power = current**2 * coil_resistance
        driver_power = current**2 * np.where(
            driving,
            self.series_resistance_ohm,
            self.recovery_resistance_ohm,
        )
        internal_h = np.asarray(self.programming_field(current)) + bias_field_a_per_m
        switch_state = np.where(driving, "drive", "recovery")
        return PulseWaveform(
            driver=self,
            pulse=pulse,
            times_s=times,
            current_a=current,
            capacitor_voltage_v=capacitor_voltage,
            bridge_voltage_v=bridge_voltage,
            internal_h_a_per_m=internal_h,
            coil_temperature_k=coil_temperature,
            driver_temperature_k=driver_temperature,
            coil_resistance_ohm=coil_resistance,
            coil_power_w=coil_power,
            driver_power_w=driver_power,
            switch_state=switch_state,
            inductance_h=inductance,
            initial_current_a=float(initial_current_a),
            initial_thermal_state=thermal,
            bias_field_a_per_m=float(bias_field_a_per_m),
        )


@dataclass(frozen=True)
class PulseWaveform:
    """Realized electrical, field, loss, and temperature response of one pulse."""

    driver: PulsePowerDriver
    pulse: ProgrammingPulse
    times_s: np.ndarray
    current_a: np.ndarray
    capacitor_voltage_v: np.ndarray
    bridge_voltage_v: np.ndarray
    internal_h_a_per_m: np.ndarray
    coil_temperature_k: np.ndarray
    driver_temperature_k: np.ndarray
    coil_resistance_ohm: np.ndarray
    coil_power_w: np.ndarray
    driver_power_w: np.ndarray
    switch_state: np.ndarray
    inductance_h: float
    initial_current_a: float
    initial_thermal_state: PulseThermalState
    bias_field_a_per_m: float = 0.0

    @property
    def peak_current_a(self) -> float:
        """Largest absolute current in amperes."""

        return float(np.max(np.abs(self.current_a)))

    @property
    def signed_peak_current_a(self) -> float:
        """Current sample having the largest magnitude."""

        index = int(np.argmax(np.abs(self.current_a)))
        return float(self.current_a[index])

    @property
    def peak_internal_h_a_per_m(self) -> float:
        """Largest absolute internal-field estimate in A/m."""

        return float(np.max(np.abs(self.internal_h_a_per_m)))

    @property
    def signed_peak_internal_h_a_per_m(self) -> float:
        """Internal-field sample having the largest magnitude."""

        index = int(np.argmax(np.abs(self.internal_h_a_per_m)))
        return float(self.internal_h_a_per_m[index])

    @property
    def absolute_h_impulse_a_s_per_m(self) -> float:
        """Integral of ``abs(H)`` in A s/m."""

        return _integral(np.abs(self.internal_h_a_per_m), self.times_s)

    @property
    def signed_h_impulse_a_s_per_m(self) -> float:
        """Signed integral of H in A s/m."""

        return _integral(self.internal_h_a_per_m, self.times_s)

    @property
    def coil_energy_j(self) -> float:
        """Integrated winding Joule loss."""

        return _integral(self.coil_power_w, self.times_s)

    @property
    def driver_conduction_energy_j(self) -> float:
        """Integrated H-bridge and recovery-path conduction loss."""

        return _integral(self.driver_power_w, self.times_s)

    @property
    def switching_energy_j(self) -> float:
        """Turn-on plus turn-off switching loss."""

        return 2.0 * self.driver.switching_energy_per_transition_j

    @property
    def driver_energy_j(self) -> float:
        """Total driver conduction plus discrete switching energy."""

        return self.driver_conduction_energy_j + self.switching_energy_j

    @property
    def initial_electrical_energy_j(self) -> float:
        """Initial capacitor plus inductor energy."""

        return float(
            0.5 * self.driver.capacitance_f * self.pulse.capacitor_voltage_v**2
            + 0.5 * self.inductance_h * self.initial_current_a**2
        )

    @property
    def final_electrical_energy_j(self) -> float:
        """Final capacitor plus inductor energy."""

        return float(
            0.5 * self.driver.capacitance_f * self.capacitor_voltage_v[-1] ** 2
            + 0.5 * self.inductance_h * self.current_a[-1] ** 2
        )

    @property
    def electrical_energy_balance_error_j(self) -> float:
        """Circuit energy residual excluding discrete switching-loss metadata."""

        return float(
            self.initial_electrical_energy_j
            - self.final_electrical_energy_j
            - self.coil_energy_j
            - self.driver_conduction_energy_j
        )

    @property
    def final_thermal_state(self) -> PulseThermalState:
        """Temperatures and cumulative losses ready for a later pulse."""

        return PulseThermalState(
            coil_temperature_k=float(self.coil_temperature_k[-1]),
            driver_temperature_k=float(self.driver_temperature_k[-1]),
            cumulative_coil_energy_j=(
                self.initial_thermal_state.cumulative_coil_energy_j
                + self.coil_energy_j
            ),
            cumulative_driver_energy_j=(
                self.initial_thermal_state.cumulative_driver_energy_j
                + self.driver_energy_j
            ),
        )


@dataclass(frozen=True)
class PulseValidationCase:
    """One configuration-specific archived peak-current comparison."""

    driver: PulsePowerDriver
    pulse: ProgrammingPulse
    measured_peak_current_a: float
    relative_tolerance: float
    evidence: tuple[EvidenceRecord, ...]

    def __post_init__(self) -> None:
        if not np.isfinite(self.measured_peak_current_a) or self.measured_peak_current_a <= 0:
            raise ValueError("measured_peak_current_a must be finite and positive")
        if not np.isfinite(self.relative_tolerance) or not 0 < self.relative_tolerance < 1:
            raise ValueError("relative_tolerance must lie between zero and one")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def run(self, *, samples: int = 1001) -> PulseWaveform:
        """Simulate through eight recovery time constants."""

        recovery_tau = self.driver.inductance_h / (
            self.driver.coil_resistance_ohm
            + self.driver.recovery_resistance_ohm
        )
        end = self.pulse.duration_s + 8.0 * recovery_tau
        times = np.linspace(0.0, end, int(samples))
        return self.driver.simulate(times, self.pulse)


def archived_igbt_pulse_cases() -> tuple[PulseValidationCase, ...]:
    """Return the separately calibrated 220/400/600-V archive comparisons.

    Only voltage, gate duration, and measured peak current are treated as
    measured.  The effective inductances below are inferred configuration
    parameters; the archive does not prove that all three traces used the same
    bundle or wiring.
    """

    measured = EvidenceRecord(
        source="Weinberg archive: MRI/IGBT_board/experimental results.pdf",
        classification="measured",
        detail="Peak-current examples at 220/400/600 V with 50/20/10 us gates",
    )
    inferred = EvidenceRecord(
        source="PythonSpinDynamics archived IGBT trace fit",
        classification="inferred",
        detail=(
            "Configuration-specific effective L values fitted to peak current; "
            "C=20 mF and total drive resistance=0.24 ohm are model assumptions"
        ),
    )
    specifications = (
        (220.0, 50.0e-6, 330.0, 26.9e-6),
        (400.0, 20.0e-6, 170.0, 44.6e-6),
        (600.0, 10.0e-6, 147.0, 39.6e-6),
    )
    cases = []
    for voltage, duration, peak, inductance in specifications:
        label = f"archive {voltage:.0f} V / {duration * 1e6:.0f} us"
        driver = PulsePowerDriver(
            capacitance_f=20.0e-3,
            inductance_h=inductance,
            coil_resistance_ohm=0.12,
            series_resistance_ohm=0.12,
            recovery_resistance_ohm=2.0,
            turns=60,
            magnetic_path_length_m=0.1016,
            voltage_limit_v=650.0,
            current_limit_a=400.0,
            label=label,
            evidence=(inferred,),
        )
        pulse = ProgrammingPulse(
            voltage,
            duration,
            label=label,
            evidence=(measured,),
        )
        cases.append(
            PulseValidationCase(
                driver,
                pulse,
                measured_peak_current_a=peak,
                relative_tolerance=0.04,
                evidence=(measured, inferred),
            )
        )
    return tuple(cases)


@dataclass(frozen=True)
class EmpiricalProgrammingCalibration:
    """Protocol-scoped interpolation from a pulse metric to retained remanence."""

    command_values: tuple[float, ...]
    remanence_t: tuple[float, ...]
    metric: ProgrammingMetric
    calibration_id: str
    uncertainty_t: float | tuple[float, ...]
    required_initial_branch: RemanenceBranch | None = None
    required_polarity: int | None = None
    extrapolation: ExtrapolationMode = "raise"
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        command = tuple(float(value) for value in self.command_values)
        remanence = tuple(float(value) for value in self.remanence_t)
        if len(command) < 2 or len(command) != len(remanence):
            raise ValueError("command_values and remanence_t must have equal length >= 2")
        if any(not np.isfinite(value) for value in (*command, *remanence)):
            raise ValueError("calibration values must be finite")
        if any(command[index + 1] <= command[index] for index in range(len(command) - 1)):
            raise ValueError("command_values must be strictly increasing")
        if self.metric not in {
            "duty_fraction",
            "peak_internal_h",
            "absolute_h_impulse",
            "signed_h_impulse",
        }:
            raise ValueError("invalid programming metric")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty")
        uncertainty = np.asarray(self.uncertainty_t, dtype=np.float64)
        if uncertainty.ndim > 1 or uncertainty.size not in {1, len(command)}:
            raise ValueError("uncertainty_t must be scalar or match command_values")
        if np.any(~np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
            raise ValueError("uncertainty_t must be finite and non-negative")
        if self.required_polarity not in {None, -1, 1}:
            raise ValueError("required_polarity must be None, -1, or +1")
        if self.extrapolation not in {"clip", "raise"}:
            raise ValueError("extrapolation must be 'clip' or 'raise'")
        object.__setattr__(self, "command_values", command)
        object.__setattr__(self, "remanence_t", remanence)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def command_from(
        self,
        waveform: PulseWaveform,
        pulse: ProgrammingPulse,
    ) -> float:
        """Extract this calibration's independent variable from a waveform."""

        if self.metric == "duty_fraction":
            if pulse.command_fraction is None:
                raise ValueError("pulse.command_fraction is required by this calibration")
            return float(pulse.command_fraction)
        if self.metric == "peak_internal_h":
            return waveform.peak_internal_h_a_per_m
        if self.metric == "absolute_h_impulse":
            return waveform.absolute_h_impulse_a_s_per_m
        return waveform.signed_h_impulse_a_s_per_m

    def apply(
        self,
        initial_state: RemanenceState,
        waveform: PulseWaveform,
        pulse: ProgrammingPulse | None = None,
    ) -> ProgrammingResult:
        """Apply the calibrated protocol and return a traceable state update."""

        active_pulse = waveform.pulse if pulse is None else pulse
        if (
            self.required_initial_branch is not None
            and initial_state.branch != self.required_initial_branch
        ):
            raise ValueError(
                "initial remanence branch is outside this calibration protocol"
            )
        if self.required_polarity is not None and active_pulse.polarity != self.required_polarity:
            raise ValueError("pulse polarity is outside this calibration protocol")
        command = self.command_from(waveform, active_pulse)
        lower, upper = self.command_values[0], self.command_values[-1]
        if self.extrapolation == "raise" and not lower <= command <= upper:
            raise ValueError("pulse command is outside the calibrated range")
        command_used = float(np.clip(command, lower, upper))
        retained = float(np.interp(command_used, self.command_values, self.remanence_t))
        uncertainty_values = np.asarray(self.uncertainty_t, dtype=np.float64)
        uncertainty = (
            float(uncertainty_values.reshape(-1)[0])
            if uncertainty_values.size == 1
            else float(
                np.interp(command_used, self.command_values, uncertainty_values)
            )
        )
        maximum = max(abs(value) for value in self.remanence_t)
        tolerance = max(1e-12, 1e-9 * maximum)
        if retained >= maximum - tolerance:
            branch: RemanenceBranch = "positive_saturation"
        elif retained <= -maximum + tolerance:
            branch = "negative_saturation"
        else:
            branch = "partial"
        reversal = initial_state.reversal_fields_a_per_m
        signed_peak = waveform.signed_peak_internal_h_a_per_m
        if retained * initial_state.remanence_t < 0.0 or (
            initial_state.remanence_t != 0.0
            and np.sign(signed_peak) != np.sign(initial_state.remanence_t)
        ):
            reversal = (*reversal, signed_peak)
        final_state = RemanenceState(
            retained,
            branch=branch,
            reversal_fields_a_per_m=reversal,
            temperature_k=waveform.final_thermal_state.coil_temperature_k,
            calibration_id=self.calibration_id,
            uncertainty_t=uncertainty,
            evidence=self.evidence,
        )
        return ProgrammingResult(
            calibration=self,
            waveform=waveform,
            initial_state=initial_state,
            final_state=final_state,
            command_value=command,
            command_value_used=command_used,
        )


@dataclass(frozen=True)
class ProgrammingResult:
    """One waveform-driven, empirically calibrated retained-state transition."""

    calibration: EmpiricalProgrammingCalibration
    waveform: PulseWaveform
    initial_state: RemanenceState
    final_state: RemanenceState
    command_value: float
    command_value_used: float


def published_demagnetization_calibration() -> EmpiricalProgrammingCalibration:
    """Return a conservative three-anchor interpretation of the published plot.

    The cropped archive figure establishes positive and negative endpoints and
    a zero crossing near 17 percent duty.  It does not provide raw samples, so
    this preset intentionally does not invent a densely digitized curve.
    """

    evidence = EvidenceRecord(
        source="Weinberg archive: NQR_proposal_2019/figures/demag_data",
        classification="inferred",
        detail=(
            "Coarse anchors from a published cropped figure: +/-0.33 T effective "
            "rod states and a zero crossing near 17% duty; uncertainty includes "
            "graphical extraction and missing raw data"
        ),
    )
    return EmpiricalProgrammingCalibration(
        command_values=(0.0, 0.17, 1.0),
        remanence_t=(0.33, 0.0, -0.33),
        metric="duty_fraction",
        calibration_id="published-demag-envelope-v1",
        uncertainty_t=(0.03, 0.04, 0.03),
        required_initial_branch="positive_saturation",
        required_polarity=-1,
        extrapolation="raise",
        evidence=(evidence,),
    )


__all__ = [
    "PulseThermalState",
    "ProgrammingPulse",
    "PulsePowerDriver",
    "PulseWaveform",
    "PulseValidationCase",
    "EmpiricalProgrammingCalibration",
    "ProgrammingResult",
    "archived_igbt_pulse_cases",
    "published_demagnetization_calibration",
]
