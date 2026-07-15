"""Coupled electrothermal electromagnets used as time-dependent B0 sources.

The model follows the feedback diagram in Section 11.2 of the measurements
textbook.  A coil obeys ``L dI/dt = V - I R(T)``, its field is
``B0 = (B/I) I``, Joule power ``I**2 R(T)`` heats a lumped thermal pole, and
the conductor temperature coefficient closes the loop through ``R(T)``.

Four drive modes expose the feedback choices discussed there: direct voltage,
temperature-compensated voltage, current feedback, and direct field lock.  A
realized coil path can be supplied so each simulated state also becomes a
spatial B0 map for the existing imaging and motion workflows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.coil_properties import (
    ANNEALED_COPPER,
    ConductorMaterial,
)
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, biot_savart
from spin_dynamics.thermal.coupling import resistance_at_temperature

Segment = tuple[np.ndarray, np.ndarray]
ElectromagnetControlMode = Literal[
    "voltage",
    "temperature_compensated",
    "current",
    "field",
]
Waveform = float | Sequence[float] | np.ndarray | Callable[[float], float]


@dataclass(frozen=True)
class ElectromagnetControl:
    """Power-supply and feedback configuration for an electromagnet.

    ``command`` passed to :meth:`ElectrothermalElectromagnet.simulate` is in
    volts for ``voltage``, amperes for ``temperature_compensated`` and
    ``current``, and tesla for ``field``.  The two closed-loop modes use a PI
    controller whose zero cancels the nominal coil L/R pole; therefore
    ``response_time_s`` directly sets the remaining nominal closed-loop pole.

    ``temperature_compensated`` represents the textbook thermal-feedback path:
    it commands ``V = I_set R(T)`` without sensing current or field.  Its
    electrical turn-on still follows the physical L/R dynamics.
    """

    mode: ElectromagnetControlMode = "voltage"
    response_time_s: float = 0.05
    voltage_limits_v: tuple[float, float] = (-np.inf, np.inf)

    def __post_init__(self) -> None:
        if self.mode not in {
            "voltage",
            "temperature_compensated",
            "current",
            "field",
        }:
            raise ValueError(
                "mode must be 'voltage', 'temperature_compensated', "
                "'current', or 'field'"
            )
        if not np.isfinite(self.response_time_s) or self.response_time_s <= 0.0:
            raise ValueError("response_time_s must be finite and positive")
        if len(self.voltage_limits_v) != 2:
            raise ValueError("voltage_limits_v must contain (minimum, maximum)")
        lower, upper = (float(value) for value in self.voltage_limits_v)
        if np.isnan(lower) or np.isnan(upper) or lower >= upper:
            raise ValueError("voltage_limits_v must be ordered and not NaN")
        object.__setattr__(self, "voltage_limits_v", (lower, upper))


@dataclass(frozen=True)
class ElectrothermalElectromagnetResult:
    """Time-domain electrical, thermal, and B0 response of an electromagnet."""

    model: ElectrothermalElectromagnet
    control: ElectromagnetControl
    times_s: np.ndarray
    command: np.ndarray
    current_a: np.ndarray
    temperature_k: np.ndarray
    resistance_ohm: np.ndarray
    voltage_v: np.ndarray
    power_w: np.ndarray
    field_t: np.ndarray

    @property
    def deposited_energy_j(self) -> float:
        """Integrated coil plus user-supplied loss energy in joules."""

        integrator = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        return float(integrator(self.power_w, self.times_s))

    def field_vectors(
        self,
        points: np.ndarray,
        *,
        time_index: int = -1,
    ) -> np.ndarray:
        """Return the vector B0 field at ``points`` for one simulated state."""

        return self.model.field_vectors(points, self.current_a[time_index])

    def uniform_b0(self, *, time_index: int = -1):
        """Return an existing :class:`experiment.UniformB0` hardware object."""

        from spin_dynamics.experiment.hardware import UniformB0

        field = float(self.field_t[time_index])
        if field == 0.0:
            raise ValueError("the selected state has zero B0 field")
        direction = self.model.field_direction
        if field < 0.0:
            direction = tuple(-value for value in direction)
        return UniformB0(direction=direction, field_tesla=abs(field))

    def projected_field_map(
        self,
        axes: Sequence[Sequence[float] | np.ndarray],
        *,
        cartesian_axes: Sequence[int] | None = None,
        time_index: int = -1,
    ) -> np.ndarray:
        """Sample the projected B0 field on a 1-D, 2-D, or 3-D Cartesian grid."""

        coordinate_axes, cartesian = _spatial_axes(axes, cartesian_axes)
        grids = np.meshgrid(*coordinate_axes, indexing="ij")
        points = np.zeros(grids[0].shape + (3,), dtype=np.float64)
        for grid, axis in zip(grids, cartesian):
            points[..., axis] = grid
        vectors = self.field_vectors(points, time_index=time_index)
        return vectors @ np.asarray(self.model.field_direction)

    def to_motion_field_maps(
        self,
        axes: Sequence[Sequence[float] | np.ndarray],
        *,
        cartesian_axes: Sequence[int] | None = None,
        time_index: int = -1,
        gyromagnetic_ratio: float = GAMMA_PROTON,
        reference_field_t: float | None = None,
        b1_tx_map: np.ndarray | None = None,
        b1_rx_map: np.ndarray | None = None,
    ):
        """Return the existing motion/imaging map with B0 in angular units.

        By default the simulated field at the model's reference point is
        removed, leaving the spatial off-resonance map expected by imaging
        workflows.  Pass ``reference_field_t=0`` to retain absolute angular
        frequency instead.
        """

        from spin_dynamics.motion import make_motion_field_maps

        coordinate_axes, cartesian = _spatial_axes(axes, cartesian_axes)
        projected = self.projected_field_map(
            coordinate_axes,
            cartesian_axes=cartesian,
            time_index=time_index,
        )
        reference = (
            float(self.field_t[time_index])
            if reference_field_t is None
            else float(reference_field_t)
        )
        off_resonance = float(gyromagnetic_ratio) * (projected - reference)
        return make_motion_field_maps(
            coordinate_axes,
            b0_map=off_resonance,
            b1_tx_map=b1_tx_map,
            b1_rx_map=b1_rx_map,
        )


@dataclass(frozen=True)
class ElectrothermalElectromagnet:
    """Lumped electrothermal model of an electromagnet B0 source.

    The thermal transfer function is one pole,
    ``H_T(s) = 1 / (C_th s + G_th)``, where ``C_th`` is
    ``heat_capacity_j_per_k`` and ``G_th`` is
    ``thermal_conductance_w_per_k``.  ``field_sensitivity_t_per_a`` is the
    projected field at the reference point per ampere.  When ``segments`` are
    supplied, spatial fields use the existing Biot-Savart solver; otherwise a
    uniform field along ``field_direction`` is returned.

    The model assumes linear magnetic response.  Ferromagnetic-core
    hysteresis, saturation, and frequency-dependent core loss are not inferred;
    measured or separately calculated loss can be supplied to :meth:`simulate`.
    """

    inductance_h: float
    reference_resistance_ohm: float
    field_sensitivity_t_per_a: float
    heat_capacity_j_per_k: float
    thermal_conductance_w_per_k: float
    ambient_temperature_k: float = 293.15
    reference_temperature_k: float = 293.15
    material: ConductorMaterial = ANNEALED_COPPER
    field_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    segments: tuple[Segment, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "inductance_h",
            "reference_resistance_ohm",
            "heat_capacity_j_per_k",
            "thermal_conductance_w_per_k",
            "ambient_temperature_k",
            "reference_temperature_k",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.field_sensitivity_t_per_a)
            or self.field_sensitivity_t_per_a == 0.0
        ):
            raise ValueError("field_sensitivity_t_per_a must be finite and nonzero")
        direction = np.asarray(self.field_direction, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("field_direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("field_direction must be nonzero")
        object.__setattr__(self, "field_direction", tuple(direction / norm))
        normalized_segments = tuple(
            (
                np.asarray(start, dtype=np.float64),
                np.asarray(end, dtype=np.float64),
            )
            for start, end in self.segments
        )
        if any(
            start.shape != (3,)
            or end.shape != (3,)
            or not np.all(np.isfinite(start))
            or not np.all(np.isfinite(end))
            for start, end in normalized_segments
        ):
            raise ValueError("each segment endpoint must be a 3-vector")
        object.__setattr__(self, "segments", normalized_segments)

    @classmethod
    def from_segments(
        cls,
        segments: Sequence[Segment],
        *,
        sample_point: Sequence[float] = (0.0, 0.0, 0.0),
        field_direction: Sequence[float] = (0.0, 0.0, 1.0),
        **kwargs,
    ) -> ElectrothermalElectromagnet:
        """Build a model and calculate ``(B/I)_coil`` from coil geometry."""

        paths = tuple(segments)
        point = np.asarray(sample_point, dtype=np.float64)
        direction = np.asarray(field_direction, dtype=np.float64)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("sample_point must be a finite 3-vector")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("field_direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("field_direction must be nonzero")
        unit = direction / norm
        field = biot_savart(point[None, :], paths, current=1.0)[0]
        sensitivity = float(field @ unit)
        return cls(
            field_sensitivity_t_per_a=sensitivity,
            field_direction=tuple(unit),
            segments=paths,
            **kwargs,
        )

    @property
    def electrical_time_constant_s(self) -> float:
        """Nominal ``L/R`` time constant at the reference temperature."""

        return self.inductance_h / self.reference_resistance_ohm

    @property
    def thermal_time_constant_s(self) -> float:
        """Lumped ``C_th/G_th`` thermal time constant."""

        return self.heat_capacity_j_per_k / self.thermal_conductance_w_per_k

    def resistance(self, temperature_k: float | np.ndarray) -> float | np.ndarray:
        """Coil resistance at one or more temperatures."""

        values = np.asarray(temperature_k, dtype=np.float64)
        scaled = np.asarray(
            [
                resistance_at_temperature(
                    self.material,
                    self.reference_resistance_ohm,
                    self.reference_temperature_k,
                    float(value),
                    exponent=1.0,
                )
                for value in values.reshape(-1)
            ]
        ).reshape(values.shape)
        return float(scaled) if scaled.ndim == 0 else scaled

    def field_vectors(self, points: np.ndarray, current_a: float) -> np.ndarray:
        """Evaluate the B0 vector field at ``points`` for ``current_a``."""

        positions = np.asarray(points, dtype=np.float64)
        if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("points must be finite with shape (..., 3)")
        if not np.isfinite(current_a):
            raise ValueError("current_a must be finite")
        if self.segments:
            return biot_savart(positions, self.segments, current=float(current_a))
        direction = np.asarray(self.field_direction)
        return np.broadcast_to(
            float(current_a) * self.field_sensitivity_t_per_a * direction,
            positions.shape,
        ).copy()

    def simulate(
        self,
        times_s: Sequence[float] | np.ndarray,
        command: Waveform,
        *,
        control: ElectromagnetControl = ElectromagnetControl(),
        initial_current_a: float = 0.0,
        initial_temperature_k: float | None = None,
        additional_power_w: Waveform = 0.0,
        max_step_s: float | None = None,
    ) -> ElectrothermalElectromagnetResult:
        """Simulate coupled current, temperature, resistance, power, and B0.

        ``additional_power_w`` can represent measured core, eddy-current, or
        nearby-structure losses.  It is deposited in the same lumped thermal
        node and must remain non-negative.
        """

        times = np.asarray(times_s, dtype=np.float64).reshape(-1)
        if times.size < 2 or not np.all(np.isfinite(times)):
            raise ValueError("times_s must contain at least two finite samples")
        if not np.all(np.diff(times) > 0.0):
            raise ValueError("times_s must be strictly increasing")
        command_values = _sample_waveform(times, command, "command")
        extra_values = _sample_waveform(times, additional_power_w, "additional_power_w")
        if np.any(extra_values < 0.0):
            raise ValueError("additional_power_w must be non-negative")
        initial_temperature = (
            self.ambient_temperature_k
            if initial_temperature_k is None
            else float(initial_temperature_k)
        )
        if not np.isfinite(initial_current_a):
            raise ValueError("initial_current_a must be finite")
        if not np.isfinite(initial_temperature) or initial_temperature <= 0.0:
            raise ValueError("initial_temperature_k must be finite and positive")
        if max_step_s is not None and (
            not np.isfinite(max_step_s) or max_step_s <= 0.0
        ):
            raise ValueError("max_step_s must be finite and positive")

        def input_at(values: np.ndarray, time: float) -> float:
            return float(np.interp(time, times, values))

        def drive_voltage(
            current: float,
            temperature: float,
            integral_error: float,
            time: float,
        ) -> tuple[float, float]:
            target = input_at(command_values, time)
            resistance = float(self.resistance(temperature))
            integral_derivative = 0.0
            if control.mode == "voltage":
                raw_voltage = target
            elif control.mode == "temperature_compensated":
                raw_voltage = target * resistance
            else:
                if control.mode == "current":
                    error = target - current
                    scale = 1.0
                else:
                    error = target - self.field_sensitivity_t_per_a * current
                    scale = self.field_sensitivity_t_per_a
                kp = self.inductance_h / (control.response_time_s * scale)
                ki = self.reference_resistance_ohm / (
                    control.response_time_s * scale
                )
                raw_voltage = kp * error + ki * integral_error
                integral_derivative = error
            lower, upper = control.voltage_limits_v
            voltage = float(np.clip(raw_voltage, lower, upper))
            if voltage != raw_voltage and control.mode in {"current", "field"}:
                if (raw_voltage > upper and ki * integral_derivative > 0.0) or (
                    raw_voltage < lower and ki * integral_derivative < 0.0
                ):
                    integral_derivative = 0.0
            return voltage, integral_derivative

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            current, temperature, integral_error = state
            resistance = float(self.resistance(temperature))
            voltage, integral_derivative = drive_voltage(
                current,
                temperature,
                integral_error,
                time,
            )
            electrical = (voltage - current * resistance) / self.inductance_h
            power = current * current * resistance + input_at(extra_values, time)
            thermal = (
                power
                - self.thermal_conductance_w_per_k
                * (temperature - self.ambient_temperature_k)
            ) / self.heat_capacity_j_per_k
            return np.asarray([electrical, thermal, integral_derivative])

        initial = np.asarray([initial_current_a, initial_temperature, 0.0])
        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            states = _rk4_states(times, initial, rhs, self, control, max_step_s)
        else:
            solution = solve_ivp(
                rhs,
                (float(times[0]), float(times[-1])),
                initial,
                method="LSODA",
                t_eval=times,
                max_step=np.inf if max_step_s is None else float(max_step_s),
                rtol=1.0e-8,
                atol=(1.0e-10, 1.0e-7, 1.0e-10),
            )
            if not solution.success:
                raise RuntimeError(
                    f"electrothermal integration failed: {solution.message}"
                )
            states = solution.y.T

        current = states[:, 0]
        temperature = states[:, 1]
        resistance = np.asarray(self.resistance(temperature))
        voltage = np.asarray(
            [
                drive_voltage(i, temp, integ, time)[0]
                for i, temp, integ, time in zip(
                    current,
                    temperature,
                    states[:, 2],
                    times,
                )
            ]
        )
        power = current * current * resistance + extra_values
        field = self.field_sensitivity_t_per_a * current
        return ElectrothermalElectromagnetResult(
            model=self,
            control=control,
            times_s=times,
            command=command_values,
            current_a=current,
            temperature_k=temperature,
            resistance_ohm=resistance,
            voltage_v=voltage,
            power_w=power,
            field_t=field,
        )


def _sample_waveform(times: np.ndarray, waveform: Waveform, name: str) -> np.ndarray:
    if callable(waveform):
        values = np.asarray([waveform(float(time)) for time in times], dtype=np.float64)
    else:
        values = np.asarray(waveform, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(times.shape, float(values))
        else:
            values = values.reshape(-1)
    if values.shape != times.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite and scalar, callable, or match times_s")
    return values


def _rk4_states(
    times: np.ndarray,
    initial: np.ndarray,
    rhs,
    model: ElectrothermalElectromagnet,
    control: ElectromagnetControl,
    max_step_s: float | None,
) -> np.ndarray:
    limits = [model.electrical_time_constant_s / 8.0]
    if control.mode in {"current", "field"}:
        limits.append(control.response_time_s / 8.0)
    if max_step_s is not None:
        limits.append(float(max_step_s))
    limit = min(limits)
    output = np.empty((times.size, initial.size), dtype=np.float64)
    output[0] = initial
    state = initial.copy()
    for index in range(times.size - 1):
        span = float(times[index + 1] - times[index])
        substeps = max(1, int(np.ceil(span / limit)))
        step = span / substeps
        time = float(times[index])
        for _ in range(substeps):
            k1 = rhs(time, state)
            k2 = rhs(time + 0.5 * step, state + 0.5 * step * k1)
            k3 = rhs(time + 0.5 * step, state + 0.5 * step * k2)
            k4 = rhs(time + step, state + step * k3)
            state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            time += step
        output[index + 1] = state
    return output


def _spatial_axes(
    axes: Sequence[Sequence[float] | np.ndarray],
    cartesian_axes: Sequence[int] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    coordinate_axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in axes)
    if not 1 <= len(coordinate_axes) <= 3:
        raise ValueError("axes must contain one, two, or three coordinate axes")
    if any(
        axis.size < 2
        or not np.all(np.isfinite(axis))
        or not np.all(np.diff(axis) > 0.0)
        for axis in coordinate_axes
    ):
        raise ValueError("each coordinate axis must be finite and strictly increasing")
    if cartesian_axes is None:
        defaults = {1: (2,), 2: (0, 2), 3: (0, 1, 2)}
        cartesian = defaults[len(coordinate_axes)]
    else:
        cartesian = tuple(int(axis) for axis in cartesian_axes)
    if (
        len(cartesian) != len(coordinate_axes)
        or len(set(cartesian)) != len(cartesian)
        or any(axis not in (0, 1, 2) for axis in cartesian)
    ):
        raise ValueError("cartesian_axes must select one distinct xyz index per axis")
    return coordinate_axes, cartesian


__all__ = [
    "ElectromagnetControlMode",
    "ElectromagnetControl",
    "ElectrothermalElectromagnet",
    "ElectrothermalElectromagnetResult",
]
