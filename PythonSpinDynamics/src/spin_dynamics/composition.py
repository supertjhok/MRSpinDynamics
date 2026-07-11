"""Typed adapters for composing fields, flow, thermal, hardware, and sequences.

The numerical engines in :mod:`spin_dynamics` intentionally keep their native
result types.  This module supplies the small interchange layer between them:
spatial axes are named and expressed in metres, time is absolute and expressed
in seconds, and every channel carries an explicit unit.  Adapters resample at
the boundary; solvers therefore do not need to know about one another.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.interpolate import dlinear_sample

__all__ = [
    "convert_units",
    "SpatialGrid",
    "TimeAxis",
    "FieldSolutionAdapter",
    "ThermalStateAdapter",
    "FlowFieldAdapter",
    "HardwareResponseAdapter",
    "SequenceTimelineAdapter",
]

Interpolation = Literal["linear", "previous"]

# ``base_value = value * scale + offset``.  Deliberately small: these are the
# units crossing current package boundaries, not a general unit system.
_UNIT_DEFINITIONS: dict[str, tuple[str, float, float]] = {
    "1": ("dimensionless", 1.0, 0.0),
    "s": ("time", 1.0, 0.0),
    "ms": ("time", 1e-3, 0.0),
    "us": ("time", 1e-6, 0.0),
    "m": ("length", 1.0, 0.0),
    "cm": ("length", 1e-2, 0.0),
    "mm": ("length", 1e-3, 0.0),
    "Hz": ("frequency", 1.0, 0.0),
    "rad/s": ("angular_frequency", 1.0, 0.0),
    "Hz/m": ("frequency_gradient", 1.0, 0.0),
    "rad/s/m": ("angular_frequency_gradient", 1.0, 0.0),
    "T": ("field", 1.0, 0.0),
    "mT": ("field", 1e-3, 0.0),
    "uT": ("field", 1e-6, 0.0),
    "T/m": ("field_gradient", 1.0, 0.0),
    "mT/m": ("field_gradient", 1e-3, 0.0),
    "T/A": ("field_per_current", 1.0, 0.0),
    "m/s": ("velocity", 1.0, 0.0),
    "mm/s": ("velocity", 1e-3, 0.0),
    "m^2/s": ("diffusivity", 1.0, 0.0),
    "mm^2/s": ("diffusivity", 1e-6, 0.0),
    "K": ("temperature", 1.0, 0.0),
    "degC": ("temperature", 1.0, 273.15),
}


def convert_units(values: Any, source_unit: str, target_unit: str) -> np.ndarray:
    """Convert values between units used at package component boundaries.

    Cyclic and angular frequency are supported explicitly even though their
    dimensions differ by convention.  Unknown or dimensionally incompatible
    units raise instead of silently relabelling a channel.
    """

    data = np.asarray(values)
    if source_unit == target_unit:
        return data.copy()
    angular = {
        ("Hz", "rad/s"): 2.0 * np.pi,
        ("rad/s", "Hz"): 1.0 / (2.0 * np.pi),
        ("Hz/m", "rad/s/m"): 2.0 * np.pi,
        ("rad/s/m", "Hz/m"): 1.0 / (2.0 * np.pi),
    }
    if (source_unit, target_unit) in angular:
        return data * angular[(source_unit, target_unit)]
    try:
        source_dimension, source_scale, source_offset = _UNIT_DEFINITIONS[source_unit]
        target_dimension, target_scale, target_offset = _UNIT_DEFINITIONS[target_unit]
    except KeyError as exc:
        raise ValueError(f"unsupported unit {exc.args[0]!r}") from None
    if source_dimension != target_dimension:
        raise ValueError(f"cannot convert {source_unit!r} to {target_unit!r}")
    base = data * source_scale + source_offset
    return (base - target_offset) / target_scale


def _readonly_1d(values: Any, label: str, *, minimum_size: int = 1) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if out.size < minimum_size:
        raise ValueError(f"{label} must contain at least {minimum_size} value(s)")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{label} must contain finite values")
    if out.size > 1 and np.any(np.diff(out) <= 0.0):
        raise ValueError(f"{label} must be strictly increasing")
    out.setflags(write=False)
    return out


def _readonly(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    out = np.asarray(values, dtype=dtype).copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("channel values must be finite")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class SpatialGrid:
    """Named rectilinear spatial axes in metres.

    Axis order is also array dimension order.  Naming the axes prevents the
    common ``(x, z)`` versus ``(x, y)`` ambiguity while :class:`SpatialDomain`
    remains the lightweight representation used by existing kernels.
    """

    axes_m: tuple[np.ndarray, ...]
    axis_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        axes = tuple(_readonly_1d(a, f"axis {i}") for i, a in enumerate(self.axes_m))
        if not 1 <= len(axes) <= 3:
            raise ValueError("SpatialGrid supports 1, 2, or 3 axes")
        names = self.axis_names or tuple("xyz"[: len(axes)])
        names = tuple(str(name) for name in names)
        if len(names) != len(axes):
            raise ValueError("axis_names must contain one name per axis")
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("axis_names must be non-empty and unique")
        object.__setattr__(self, "axes_m", axes)
        object.__setattr__(self, "axis_names", names)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(axis.size) for axis in self.axes_m)

    @property
    def ndim(self) -> int:
        return len(self.axes_m)

    @property
    def domain(self) -> SpatialDomain:
        return SpatialDomain(self.axes_m)

    @classmethod
    def from_domain(
        cls, domain: SpatialDomain, *, axis_names: Sequence[str] | None = None
    ) -> "SpatialGrid":
        """Adapt an existing domain whose coordinates are in metres."""

        return cls(domain.axes, () if axis_names is None else tuple(axis_names))

    def positions(self) -> np.ndarray:
        """Return all grid positions as ``(number_of_cells, ndim)``."""

        return np.column_stack([grid.ravel() for grid in self.domain.meshgrid("ij")])


@dataclass(frozen=True)
class TimeAxis:
    """A strictly increasing absolute time axis in seconds."""

    times_seconds: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "times_seconds", _readonly_1d(self.times_seconds, "times_seconds")
        )

    @classmethod
    def uniform(
        cls, count: int, dwell_seconds: float, *, start_seconds: float = 0.0
    ) -> "TimeAxis":
        if count < 1:
            raise ValueError("count must be positive")
        if dwell_seconds <= 0.0 or not np.isfinite(dwell_seconds):
            raise ValueError("dwell_seconds must be finite and positive")
        return cls(start_seconds + np.arange(count, dtype=np.float64) * dwell_seconds)

    @property
    def is_uniform(self) -> bool:
        if self.times_seconds.size < 3:
            return True
        delta = np.diff(self.times_seconds)
        return bool(np.allclose(delta, delta[0], rtol=1e-10, atol=1e-15))

    @property
    def dwell_seconds(self) -> float:
        if self.times_seconds.size < 2:
            raise ValueError("a dwell requires at least two time samples")
        if not self.is_uniform:
            raise ValueError("time axis is not uniformly sampled")
        return float(self.times_seconds[1] - self.times_seconds[0])

    def resample(
        self,
        values: np.ndarray,
        target: "TimeAxis",
        *,
        axis: int = 0,
        method: Interpolation = "linear",
    ) -> np.ndarray:
        """Resample an array from this axis to ``target``.

        Values outside this axis are rejected instead of silently extrapolated.
        ``previous`` is the appropriate zero-order hold for sequence commands;
        ``linear`` is the default for physical state trajectories.
        """

        source = self.times_seconds
        query = target.times_seconds
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(source[-1]))
        if query[0] < source[0] - tolerance or query[-1] > source[-1] + tolerance:
            raise ValueError("target time axis lies outside the source time axis")
        data = np.moveaxis(np.asarray(values), axis, 0)
        if data.shape[0] != source.size:
            raise ValueError("values axis length must match the source time axis")
        flat = data.reshape(source.size, -1)
        if method == "previous":
            index = np.searchsorted(source, query, side="right") - 1
            result = flat[np.clip(index, 0, source.size - 1)]
        elif method == "linear":
            result = np.empty((query.size, flat.shape[1]), dtype=np.result_type(data, float))
            if np.iscomplexobj(data):
                for column in range(flat.shape[1]):
                    result[:, column] = np.interp(query, source, flat[:, column].real)
                    result[:, column] += 1j * np.interp(
                        query, source, flat[:, column].imag
                    )
            else:
                for column in range(flat.shape[1]):
                    result[:, column] = np.interp(query, source, flat[:, column])
        else:
            raise ValueError("method must be 'linear' or 'previous'")
        result = result.reshape((query.size, *data.shape[1:]))
        return np.moveaxis(result, 0, axis)


def _validate_units(values: Mapping[str, np.ndarray], units: Mapping[str, str]) -> None:
    missing = set(values) - set(units)
    extra = set(units) - set(values)
    if missing or extra:
        raise ValueError("units must contain exactly one entry per channel")
    if any(not unit for unit in units.values()):
        raise ValueError("channel units must be non-empty")


def _spatial_resample(
    values: np.ndarray, source: SpatialGrid, target: SpatialGrid
) -> np.ndarray:
    if source.axis_names != target.axis_names:
        raise ValueError("source and target spatial axis names must match in order")
    data = np.asarray(values)
    if data.shape[: source.ndim] != source.shape:
        raise ValueError("channel leading dimensions must match the spatial grid")
    positions = target.positions()
    trailing = data.shape[source.ndim :]
    flat_components = data.reshape((*source.shape, -1))
    sampled = np.empty((positions.shape[0], flat_components.shape[-1]), dtype=data.dtype)
    for component in range(flat_components.shape[-1]):
        sampled[:, component] = dlinear_sample(
            flat_components[..., component], source.axes_m, positions
        )
    return sampled.reshape((*target.shape, *trailing))


@dataclass(frozen=True)
class FieldSolutionAdapter:
    """Named field channels sampled on a shared spatial grid.

    Channel arrays have the grid shape as their leading dimensions and may
    have trailing component dimensions (for example ``b0_vector_t`` has a
    final length-three dimension).
    """

    grid: SpatialGrid
    channels: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channels = {name: _readonly(value) for name, value in self.channels.items()}
        if not channels:
            raise ValueError("at least one field channel is required")
        for name, value in channels.items():
            if value.shape[: self.grid.ndim] != self.grid.shape:
                raise ValueError(f"channel {name!r} does not match the spatial grid")
        units = dict(self.units)
        _validate_units(channels, units)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def resample(self, grid: SpatialGrid) -> "FieldSolutionAdapter":
        return FieldSolutionAdapter(
            grid,
            {name: _spatial_resample(value, self.grid, grid) for name, value in self.channels.items()},
            self.units,
            self.metadata,
        )

    def converted(self, channel: str, target_unit: str) -> "FieldSolutionAdapter":
        """Return a copy with one channel converted and relabelled."""

        if channel not in self.channels:
            raise KeyError(channel)
        channels = dict(self.channels)
        channels[channel] = convert_units(
            channels[channel], self.units[channel], target_unit
        )
        units = dict(self.units)
        units[channel] = target_unit
        return FieldSolutionAdapter(self.grid, channels, units, self.metadata)

    @classmethod
    def from_magnet_field_maps(cls, result: Any) -> "FieldSolutionAdapter":
        """Adapt :class:`fields.MagnetFieldMaps` without changing its values."""

        channels = {
            "b0_vector": result.b0_vector,
            "b0_magnitude": result.b0_magnitude,
            "b0_gradient": result.b0_gradient,
            "larmor_frequency": result.larmor_hz,
        }
        units = {
            "b0_vector": "T",
            "b0_magnitude": "T",
            "b0_gradient": "T/m",
            "larmor_frequency": "Hz",
        }
        if result.b1_vector is not None:
            channels["b1_vector"] = result.b1_vector
            units["b1_vector"] = "T/A"
        if result.b1_transverse is not None:
            channels["b1_transverse"] = result.b1_transverse
            units["b1_transverse"] = "T/A"
        return cls(SpatialGrid((result.x_axis, result.y_axis), ("x", "y")), channels, units)

    @classmethod
    def from_spatial_field_maps(
        cls,
        result: Any,
        *,
        axis_names: Sequence[str] | None = None,
        frequency_unit: str = "rad/s",
    ) -> "FieldSolutionAdapter":
        """Adapt :class:`fields.SpatialFieldMaps` with explicit conventions."""

        grid = SpatialGrid.from_domain(result.domain, axis_names=axis_names)
        channels = {
            "density": result.rho,
            "t1": result.t1_map,
            "t2": result.t2_map,
            "b0_offset": result.b0_map,
            "b1_transmit": result.b1_tx_map,
            "b1_receive": result.b1_rx_map,
        }
        units = {
            "density": "1",
            "t1": "s",
            "t2": "s",
            "b0_offset": frequency_unit,
            "b1_transmit": frequency_unit,
            "b1_receive": frequency_unit,
        }
        if result.diffusion_map is not None:
            channels["diffusion"] = result.diffusion_map
            units["diffusion"] = "m^2/s"
        return cls(grid, channels, units)


@dataclass(frozen=True)
class ThermalStateAdapter:
    """Temperature channels in kelvin on an explicit time base.

    A steady state is represented by a one-sample time axis.  Named channels
    cover lumped thermal nodes; a single ``temperature`` channel may carry a
    trailing spatial dimension for conduction results.
    """

    time: TimeAxis
    temperatures_kelvin: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {name: _readonly(value, dtype=np.float64) for name, value in self.temperatures_kelvin.items()}
        if not values:
            raise ValueError("at least one temperature channel is required")
        for name, value in values.items():
            if value.ndim == 0 or value.shape[0] != self.time.times_seconds.size:
                raise ValueError(f"temperature channel {name!r} must start with the time axis")
        object.__setattr__(self, "temperatures_kelvin", values)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def at(self, time: TimeAxis) -> "ThermalStateAdapter":
        return ThermalStateAdapter(
            time,
            {name: self.time.resample(value, time) for name, value in self.temperatures_kelvin.items()},
            self.metadata,
        )

    @classmethod
    def from_transient(cls, result: Any) -> "ThermalStateAdapter":
        """Adapt a ``ThermalTransientResult``."""

        return cls(TimeAxis(result.times), result.temperatures)

    @classmethod
    def from_conduction(cls, result: Any, *, time_seconds: float = 0.0) -> "ThermalStateAdapter":
        """Adapt a steady or transient ``ConductionResult``."""

        temperature = np.asarray(result.temperature)
        if result.times is None:
            temperature = temperature.reshape((1, *temperature.shape))
            time = TimeAxis(np.asarray([time_seconds]))
        else:
            time = TimeAxis(result.times)
        return cls(time, {"temperature": temperature}, {"radius_m": np.asarray(result.r)})

    @classmethod
    def from_coupling(
        cls, result: Any, *, time_seconds: float | None = None
    ) -> "ThermalStateAdapter":
        """Adapt a steady or marched ``ThermalCouplingResult``."""

        if result.transient is not None:
            return cls.from_transient(result.transient)
        instant = 0.0 if time_seconds is None else float(time_seconds)
        values = {name: np.asarray([value]) for name, value in result.temperatures.items()}
        return cls(TimeAxis(np.asarray([instant])), values)


VelocityCallable = Callable[[np.ndarray, float], np.ndarray]


@dataclass(frozen=True)
class FlowFieldAdapter:
    """Eulerian velocity in m/s, optionally varying over an absolute time axis."""

    grid: SpatialGrid | None
    velocity_m_per_s: np.ndarray | VelocityCallable
    time: TimeAxis | None = None

    def __post_init__(self) -> None:
        if callable(self.velocity_m_per_s):
            if self.grid is not None or self.time is not None:
                raise ValueError("callable flow fields do not take grid or time axes")
            return
        velocity = _readonly(self.velocity_m_per_s, dtype=np.float64)
        if self.grid is None:
            if velocity.ndim != 1 or not 1 <= velocity.size <= 3:
                raise ValueError("uniform velocity must be a vector with 1 to 3 components")
        else:
            prefix = (() if self.time is None else (self.time.times_seconds.size,)) + self.grid.shape
            if (
                velocity.ndim != len(prefix) + 1
                or velocity.shape[: len(prefix)] != prefix
                or velocity.shape[-1] != self.grid.ndim
            ):
                raise ValueError("velocity shape must be [time,] * grid.shape + (grid.ndim,)")
        object.__setattr__(self, "velocity_m_per_s", velocity)

    @classmethod
    def uniform(cls, velocity_m_per_s: Sequence[float]) -> "FlowFieldAdapter":
        return cls(None, np.asarray(velocity_m_per_s, dtype=np.float64))

    @classmethod
    def from_flow_model(
        cls, flow: Any, *, direction: Sequence[float] = (1.0, 0.0, 0.0)
    ) -> "FlowFieldAdapter":
        """Adapt a pipe ``FlowModel`` to its bulk uniform velocity vector."""

        vector = np.asarray(direction, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not 1 <= vector.size <= 3 or norm == 0.0 or not np.isfinite(norm):
            raise ValueError("direction must be a finite non-zero 1D/2D/3D vector")
        return cls.uniform(flow.mean_velocity * vector / norm)

    def sample(self, positions_m: np.ndarray, time_seconds: float = 0.0) -> np.ndarray:
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2:
            raise ValueError("positions_m must have shape (number_of_positions, ndim)")
        if callable(self.velocity_m_per_s):
            result = np.asarray(self.velocity_m_per_s(positions, time_seconds), dtype=np.float64)
            if result.ndim == 1:
                result = np.broadcast_to(result, positions.shape)
            if result.shape != positions.shape or not np.all(np.isfinite(result)):
                raise ValueError("flow callable must return a finite velocity matching positions")
            return result
        velocity = self.velocity_m_per_s
        if self.grid is None:
            if positions.shape[1] != velocity.size:
                raise ValueError("position and velocity dimensions must match")
            return np.broadcast_to(velocity, positions.shape).copy()
        if positions.shape[1] != self.grid.ndim:
            raise ValueError("position and grid dimensions must match")
        if self.time is not None:
            query = TimeAxis(np.asarray([time_seconds]))
            velocity = self.time.resample(velocity, query)[0]
        out = np.empty_like(positions)
        for component in range(self.grid.ndim):
            out[:, component] = dlinear_sample(
                velocity[..., component], self.grid.axes_m, positions
            )
        return out


@runtime_checkable
class _HardwareResponse(Protocol):
    def apply(self, waveform: np.ndarray, dt: float, *, xp: Any = np) -> np.ndarray: ...


@dataclass(frozen=True)
class HardwareResponseAdapter:
    """Typed bridge from a uniformly sampled channel to an LTI response."""

    response: _HardwareResponse
    input_unit: str
    output_unit: str
    channel_kind: Literal["rf", "gradient", "receive"]

    def __post_init__(self) -> None:
        if not isinstance(self.response, _HardwareResponse):
            raise TypeError("response must provide apply(waveform, dt, *, xp=...)")
        if not self.input_unit or not self.output_unit:
            raise ValueError("input_unit and output_unit must be non-empty")

    def apply(self, values: np.ndarray, time: TimeAxis) -> np.ndarray:
        """Apply the response, checking the otherwise implicit uniform dwell."""

        data = np.asarray(values)
        if data.shape[0] != time.times_seconds.size:
            raise ValueError("waveform leading axis must match time")
        return np.asarray(self.response.apply(data, time.dwell_seconds, xp=np))


@dataclass(frozen=True)
class SequenceTimelineAdapter:
    """Sequence channels on one absolute, interval-centered time axis."""

    time: TimeAxis
    channels: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    durations_seconds: np.ndarray | None = None
    adc_time: TimeAxis | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        channels = {name: _readonly(value) for name, value in self.channels.items()}
        for name, value in channels.items():
            if value.ndim == 0 or value.shape[0] != self.time.times_seconds.size:
                raise ValueError(f"sequence channel {name!r} must start with the time axis")
        units = dict(self.units)
        _validate_units(channels, units)
        durations = self.durations_seconds
        if durations is not None:
            durations = _readonly(durations, dtype=np.float64).reshape(-1)
            if durations.size != self.time.times_seconds.size or np.any(durations <= 0.0):
                raise ValueError("durations_seconds must be positive and match time")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "durations_seconds", durations)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def at(self, time: TimeAxis) -> "SequenceTimelineAdapter":
        """Zero-order-hold all commands onto another absolute time axis."""

        return SequenceTimelineAdapter(
            time,
            {name: self.time.resample(value, time, method="previous") for name, value in self.channels.items()},
            self.units,
            adc_time=self.adc_time,
            metadata=self.metadata,
        )

    def converted(self, channel: str, target_unit: str) -> "SequenceTimelineAdapter":
        """Return a timeline with one channel converted to ``target_unit``."""

        if channel not in self.channels:
            raise KeyError(channel)
        channels = dict(self.channels)
        channels[channel] = convert_units(
            channels[channel], self.units[channel], target_unit
        )
        units = dict(self.units)
        units[channel] = target_unit
        return SequenceTimelineAdapter(
            self.time,
            channels,
            units,
            self.durations_seconds,
            self.adc_time,
            self.metadata,
        )

    def with_hardware_response(
        self, channel: str, response: HardwareResponseAdapter
    ) -> "SequenceTimelineAdapter":
        """Filter one channel after checking its declared physical unit."""

        if channel not in self.channels:
            raise KeyError(channel)
        if self.units[channel] != response.input_unit:
            raise ValueError(
                f"channel {channel!r} has unit {self.units[channel]!r}, "
                f"expected {response.input_unit!r}"
            )
        channels = dict(self.channels)
        channels[channel] = response.apply(channels[channel], self.time)
        units = dict(self.units)
        units[channel] = response.output_unit
        return SequenceTimelineAdapter(
            self.time,
            channels,
            units,
            self.durations_seconds,
            self.adc_time,
            self.metadata,
        )

    @classmethod
    def from_compiled(cls, compiled: Any) -> "SequenceTimelineAdapter":
        """Adapt a :class:`sequences.CompiledSequence` in its native SI units."""

        centers = compiled.start_times_seconds + 0.5 * compiled.durations_seconds
        adc = None
        if compiled.adc.times_seconds.size:
            adc = TimeAxis(compiled.adc.times_seconds)
        return cls(
            TimeAxis(centers),
            {"rf": compiled.rf_hz, "gradient": compiled.gradients_hz_per_m},
            {"rf": "Hz", "gradient": "Hz/m"},
            compiled.durations_seconds,
            adc,
            {
                "source_format": compiled.source_format,
                "source_version": compiled.source_version,
                "block_indices": compiled.block_indices,
            },
        )
