"""Absolute RF phase helpers for pulse-sequence simulations.

The helpers in this module keep laboratory-frame RF phase bookkeeping separate
from individual workflow implementations.  They intentionally do not assume a
specific pulse sequence; workflows supply pulse start times and rotating-frame
phases, then optionally ask a transient model to perturb the pulse shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


TAU = 2.0 * np.pi


@dataclass(frozen=True)
class PulseShape:
    """Piecewise-constant rotating-frame pulse shape."""

    duration: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray

    def __post_init__(self) -> None:
        duration = np.asarray(self.duration, dtype=np.float64).reshape(-1)
        phase = np.asarray(self.phase, dtype=np.float64).reshape(-1)
        amplitude = np.asarray(self.amplitude, dtype=np.float64).reshape(-1)
        if not (duration.size == phase.size == amplitude.size):
            raise ValueError("duration, phase, and amplitude must have the same length")
        if duration.size == 0:
            raise ValueError("pulse shape must contain at least one segment")
        for name, values in {
            "duration": duration,
            "phase": phase,
            "amplitude": amplitude,
        }.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} values must be finite")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "amplitude", amplitude)


@dataclass(frozen=True)
class SinusoidalTransientPerturbation:
    """Simple absolute-phase-dependent pulse perturbation.

    This is a compact phenomenological bridge for early absolute-phase studies.
    ``phase_amplitude_rad`` applies a uniform rotating-frame phase perturbation
    to the pulse and ``amplitude_fraction`` applies a fractional amplitude
    perturbation.  ``periodicity`` controls whether the dependence repeats every
    RF period (``"full"``) or half RF period (``"half"``), matching the main
    signatures discussed for non-resonant and tuned probes, respectively.
    """

    phase_amplitude_rad: float = 0.0
    amplitude_fraction: float = 0.0
    phase_offset_rad: float = 0.0
    periodicity: str = "full"
    applies_to: str = "refocusing"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.phase_amplitude_rad)):
            raise ValueError("phase_amplitude_rad must be finite")
        if not np.isfinite(float(self.amplitude_fraction)):
            raise ValueError("amplitude_fraction must be finite")
        if abs(float(self.amplitude_fraction)) >= 1.0:
            raise ValueError("amplitude_fraction magnitude must be less than 1")
        if self.periodicity not in {"full", "half"}:
            raise ValueError("periodicity must be 'full' or 'half'")
        if self.applies_to not in {"all", "excitation", "refocusing"}:
            raise ValueError("applies_to must be 'all', 'excitation', or 'refocusing'")

    @property
    def phase_multiplier(self) -> float:
        """Multiplier for the absolute phase argument."""

        return 2.0 if self.periodicity == "half" else 1.0

    def perturbation(self, absolute_phase_rad: float) -> float:
        """Return the signed scalar perturbation for an absolute pulse phase."""

        arg = self.phase_multiplier * float(absolute_phase_rad) + float(
            self.phase_offset_rad
        )
        return float(np.sin(arg))

    def applies(self, pulse_kind: str) -> bool:
        """Return whether this model applies to a pulse kind."""

        return self.applies_to == "all" or self.applies_to == pulse_kind

    def apply(self, shape: PulseShape, absolute_phase_rad: float, pulse_kind: str) -> PulseShape:
        """Return a perturbed copy of ``shape``."""

        if not self.applies(pulse_kind):
            return shape
        scale = self.perturbation(absolute_phase_rad)
        phase = np.asarray(shape.phase, dtype=np.float64) + (
            float(self.phase_amplitude_rad) * scale
        )
        amplitude = np.asarray(shape.amplitude, dtype=np.float64) * (
            1.0 + float(self.amplitude_fraction) * scale
        )
        return PulseShape(
            duration=np.asarray(shape.duration, dtype=np.float64),
            phase=phase,
            amplitude=amplitude,
        )


@dataclass(frozen=True)
class LongitudinalPhaseKick:
    """Absolute-phase-dependent z phase shift after an RF pulse.

    This model represents the simplified non-resonant/DC transient mechanism
    from Mandal 2015, where a small ``B1_z`` transient rotates transverse
    magnetization by ``epsilon * sin(absolute_phase)``.
    """

    phase_amplitude_rad: float
    phase_offset_rad: float = 0.0
    periodicity: str = "full"
    applies_to: str = "refocusing"

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.phase_amplitude_rad)):
            raise ValueError("phase_amplitude_rad must be finite")
        if not np.isfinite(float(self.phase_offset_rad)):
            raise ValueError("phase_offset_rad must be finite")
        if self.periodicity not in {"full", "half"}:
            raise ValueError("periodicity must be 'full' or 'half'")
        if self.applies_to not in {"all", "excitation", "refocusing"}:
            raise ValueError("applies_to must be 'all', 'excitation', or 'refocusing'")

    @property
    def phase_multiplier(self) -> float:
        """Multiplier for the absolute phase argument."""

        return 2.0 if self.periodicity == "half" else 1.0

    def applies(self, pulse_kind: str) -> bool:
        """Return whether this model applies to a pulse kind."""

        return self.applies_to == "all" or self.applies_to == pulse_kind

    def phase_kick(self, absolute_phase_rad: float, pulse_kind: str) -> float:
        """Return the z-rotation phase kick for a pulse."""

        if not self.applies(pulse_kind):
            return 0.0
        arg = self.phase_multiplier * float(absolute_phase_rad) + float(
            self.phase_offset_rad
        )
        return float(self.phase_amplitude_rad) * float(np.sin(arg))

    def apply(
        self,
        shape: PulseShape,
        absolute_phase_rad: float,
        pulse_kind: str,
    ) -> PulseShape:
        """Pulse shape is unchanged; workflows apply the z kick separately."""

        return shape


@dataclass(frozen=True)
class PulseShapeLibrary:
    """Absolute-phase-indexed library of rotating-frame pulse shapes.

    The library is periodic in absolute phase.  Interpolation is performed on
    the complex RF drive ``amplitude * exp(1j * phase)`` so phase wrapping does
    not create artificial jumps.
    """

    absolute_phase_rad: np.ndarray
    shapes: Mapping[str, Sequence[PulseShape]]
    period_rad: float = TAU

    def __post_init__(self) -> None:
        period = float(self.period_rad)
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period_rad must be finite and positive")
        phase_grid = np.asarray(self.absolute_phase_rad, dtype=np.float64).reshape(-1)
        if phase_grid.size < 2:
            raise ValueError("absolute_phase_rad must contain at least two samples")
        if not np.all(np.isfinite(phase_grid)):
            raise ValueError("absolute_phase_rad values must be finite")
        phase_grid = np.mod(phase_grid, period)
        order = np.argsort(phase_grid)
        phase_grid = phase_grid[order]
        if np.any(np.diff(phase_grid) <= 0):
            raise ValueError("absolute_phase_rad samples must be unique modulo period")

        normalized: dict[str, tuple[PulseShape, ...]] = {}
        for pulse_kind, raw_shapes in self.shapes.items():
            shape_tuple = tuple(raw_shapes)
            if len(shape_tuple) != phase_grid.size:
                raise ValueError(
                    f"shape count for {pulse_kind!r} must match absolute_phase_rad"
                )
            shape_tuple = tuple(shape_tuple[int(idx)] for idx in order)
            first_duration = shape_tuple[0].duration
            for shape in shape_tuple[1:]:
                if not np.allclose(shape.duration, first_duration, rtol=0.0, atol=0.0):
                    raise ValueError(
                        f"all {pulse_kind!r} library shapes must share durations"
                    )
            normalized[str(pulse_kind)] = shape_tuple
        if not normalized:
            raise ValueError("shapes must contain at least one pulse kind")
        object.__setattr__(self, "absolute_phase_rad", phase_grid)
        object.__setattr__(self, "shapes", normalized)
        object.__setattr__(self, "period_rad", period)

    def has_pulse_kind(self, pulse_kind: str) -> bool:
        """Return whether the library contains a pulse kind."""

        return pulse_kind in self.shapes

    def shape(self, pulse_kind: str, absolute_phase_rad: float) -> PulseShape:
        """Return an interpolated pulse shape for a pulse kind and phase."""

        if pulse_kind not in self.shapes:
            raise KeyError(f"pulse kind {pulse_kind!r} is not in the library")
        phase = float(np.mod(float(absolute_phase_rad), self.period_rad))
        grid = self.absolute_phase_rad
        idx = int(np.searchsorted(grid, phase, side="right") - 1)
        if idx < 0:
            idx = grid.size - 1
            next_idx = 0
            left = grid[-1] - self.period_rad
            right = grid[0]
        else:
            next_idx = (idx + 1) % grid.size
            left = grid[idx]
            right = grid[next_idx]
            if next_idx == 0:
                right += self.period_rad
        frac = 0.0 if right == left else (phase - left) / (right - left)

        shapes = self.shapes[pulse_kind]
        left_shape = shapes[idx]
        right_shape = shapes[next_idx]
        left_drive = left_shape.amplitude * np.exp(1j * left_shape.phase)
        right_drive = right_shape.amplitude * np.exp(1j * right_shape.phase)
        drive = (1.0 - frac) * left_drive + frac * right_drive
        return PulseShape(
            duration=left_shape.duration,
            phase=np.angle(drive),
            amplitude=np.abs(drive),
        )


@dataclass(frozen=True)
class InterpolatedPulseShapeModel:
    """Absolute-phase model backed by a pulse-shape library."""

    library: PulseShapeLibrary
    missing: str = "base"

    def __post_init__(self) -> None:
        if self.missing not in {"base", "error"}:
            raise ValueError("missing must be 'base' or 'error'")

    def apply(
        self,
        shape: PulseShape,
        absolute_phase_rad: float,
        pulse_kind: str,
    ) -> PulseShape:
        """Return the library shape for ``pulse_kind`` or the base shape."""

        if not self.library.has_pulse_kind(pulse_kind):
            if self.missing == "base":
                return shape
            raise KeyError(f"pulse kind {pulse_kind!r} is not in the library")
        return self.library.shape(pulse_kind, absolute_phase_rad)


TransientModel = (
    SinusoidalTransientPerturbation
    | LongitudinalPhaseKick
    | InterpolatedPulseShapeModel
)


@dataclass(frozen=True)
class AbsolutePhaseSpec:
    """Laboratory-frame RF phase configuration."""

    rf_frequency_hz: float
    rf_phase_at_zero_rad: float = 0.0
    transient_model: TransientModel | None = None
    receiver_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.rf_frequency_hz)) or self.rf_frequency_hz <= 0:
            raise ValueError("rf_frequency_hz must be finite and positive")
        if not np.isfinite(float(self.rf_phase_at_zero_rad)):
            raise ValueError("rf_phase_at_zero_rad must be finite")
        if not np.isfinite(float(self.receiver_phase_rad)):
            raise ValueError("receiver_phase_rad must be finite")

    @property
    def rf_angular_frequency_rad_s(self) -> float:
        """RF angular frequency in rad/s."""

        return TAU * float(self.rf_frequency_hz)

    def pulse_phase(self, start_seconds: float, rotating_phase_rad: float) -> float:
        """Return the laboratory-frame absolute phase at pulse start."""

        phase = (
            float(self.rf_phase_at_zero_rad)
            + self.rf_angular_frequency_rad_s * float(start_seconds)
            + float(rotating_phase_rad)
        )
        return wrap_phase(phase)


@dataclass(frozen=True)
class AbsolutePhaseMetadata:
    """Absolute phase values used by a sequence simulation."""

    rf_frequency_hz: float
    refocus_start_seconds: np.ndarray
    refocus_absolute_phase_rad: np.ndarray
    excitation_absolute_phase_rad: np.ndarray
    echo_spacing_seconds: float
    delta_refocus_phase_rad: float
    receiver_phase_rad: float = 0.0
    transient_model: str | None = None
    pulse_matrix_count: int = 0
    pulse_start_seconds: np.ndarray | None = None
    pulse_kind: tuple[str, ...] = ()
    pulse_rotating_phase_rad: np.ndarray | None = None
    pulse_absolute_phase_rad: np.ndarray | None = None
    excitation_matrix_indices: np.ndarray | None = None
    refocus_matrix_indices: np.ndarray | None = None

    @property
    def delta_refocus_phase_cycles(self) -> float:
        """Refocusing phase increment in RF cycles."""

        return float(self.delta_refocus_phase_rad / TAU)


@dataclass(frozen=True)
class FiniteCPMGPhaseSchedule:
    """Absolute phase schedule for a finite CPMG-like echo train."""

    rf_frequency_hz: float
    excitation_start_seconds: float
    excitation_duration_seconds: float
    correction_delay_seconds: float
    pre_refocus_delay_seconds: float
    echo_spacing_seconds: float
    excitation_rotating_phase_rad: np.ndarray
    excitation_absolute_phase_rad: np.ndarray
    refocus_start_seconds: np.ndarray
    refocus_rotating_phase_rad: float
    refocus_absolute_phase_rad: np.ndarray
    pulse_start_seconds: np.ndarray
    pulse_kind: tuple[str, ...]
    pulse_rotating_phase_rad: np.ndarray
    pulse_absolute_phase_rad: np.ndarray
    receiver_phase_rad: float = 0.0

    @property
    def delta_refocus_phase_rad(self) -> float:
        """Absolute phase increment between adjacent refocusing pulses."""

        return float(wrap_phase(TAU * self.rf_frequency_hz * self.echo_spacing_seconds))

    @property
    def delta_refocus_phase_cycles(self) -> float:
        """Absolute phase increment between refocusing pulses in RF cycles."""

        return float(self.delta_refocus_phase_rad / TAU)


@dataclass(frozen=True)
class FiniteCPMGPulsePlan:
    """Pulse-matrix indices for finite CPMG phase cycling."""

    excitation_cycle_one: np.ndarray
    excitation_cycle_two: np.ndarray
    refocus_cycle: np.ndarray
    pulse_matrix_count: int

    def pulse_indices(self, cycle: int = 1) -> np.ndarray:
        """Return complete pulse-index vector for a phase-cycle branch."""

        if int(cycle) == 1:
            excitation = self.excitation_cycle_one
        elif int(cycle) == 2:
            excitation = self.excitation_cycle_two
        else:
            raise ValueError("cycle must be 1 or 2")
        return np.concatenate([excitation, self.refocus_cycle])


def wrap_phase(phase_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap phase into the interval [0, 2*pi)."""

    return np.mod(phase_rad, TAU)


def sinusoidal_transient_from_mapping(
    value: Mapping[str, Any],
) -> SinusoidalTransientPerturbation:
    """Build a sinusoidal perturbation model from a mapping."""

    return SinusoidalTransientPerturbation(
        phase_amplitude_rad=float(value.get("phase_amplitude_rad", 0.0)),
        amplitude_fraction=float(value.get("amplitude_fraction", 0.0)),
        phase_offset_rad=float(value.get("phase_offset_rad", 0.0)),
        periodicity=str(value.get("periodicity", "full")),
        applies_to=str(value.get("applies_to", "refocusing")),
    )


def longitudinal_phase_kick_from_mapping(value: Mapping[str, Any]) -> LongitudinalPhaseKick:
    """Build a longitudinal phase-kick model from a mapping."""

    return LongitudinalPhaseKick(
        phase_amplitude_rad=float(value["phase_amplitude_rad"]),
        phase_offset_rad=float(value.get("phase_offset_rad", 0.0)),
        periodicity=str(value.get("periodicity", "full")),
        applies_to=str(value.get("applies_to", "refocusing")),
    )


def _pulse_shapes_from_mapping(value: Mapping[str, Any]) -> tuple[PulseShape, ...]:
    duration = np.asarray(value["duration"], dtype=np.float64)
    phase = np.asarray(value["phase"], dtype=np.float64)
    amplitude = np.asarray(value["amplitude"], dtype=np.float64)
    if phase.ndim != 2 or amplitude.ndim != 2:
        raise ValueError("library phase and amplitude arrays must be two-dimensional")
    if phase.shape != amplitude.shape:
        raise ValueError("library phase and amplitude arrays must have the same shape")
    if duration.ndim == 1:
        durations = np.broadcast_to(duration.reshape(1, -1), phase.shape)
    elif duration.ndim == 2:
        durations = duration
    else:
        raise ValueError("library duration array must be one- or two-dimensional")
    if durations.shape != phase.shape:
        raise ValueError("library duration array must match phase/amplitude shape")
    return tuple(
        PulseShape(
            duration=durations[idx, :],
            phase=phase[idx, :],
            amplitude=amplitude[idx, :],
        )
        for idx in range(phase.shape[0])
    )


def pulse_shape_library_from_mapping(value: Mapping[str, Any]) -> PulseShapeLibrary:
    """Build a pulse-shape library from plain arrays."""

    raw_shapes = value["shapes"]
    if not isinstance(raw_shapes, Mapping):
        raise TypeError("library shapes must be a mapping from pulse kind to shapes")
    shapes: dict[str, tuple[PulseShape, ...]] = {}
    for pulse_kind, raw in raw_shapes.items():
        if isinstance(raw, Mapping):
            shapes[str(pulse_kind)] = _pulse_shapes_from_mapping(raw)
        else:
            shapes[str(pulse_kind)] = tuple(raw)
    return PulseShapeLibrary(
        absolute_phase_rad=np.asarray(value["absolute_phase_rad"], dtype=np.float64),
        shapes=shapes,
        period_rad=float(value.get("period_rad", TAU)),
    )


def interpolated_pulse_model_from_mapping(
    value: Mapping[str, Any],
) -> InterpolatedPulseShapeModel:
    """Build an interpolated pulse-shape model from a mapping."""

    return InterpolatedPulseShapeModel(
        library=pulse_shape_library_from_mapping(value),
        missing=str(value.get("missing", "base")),
    )


def _segment_sample_times(
    duration_seconds: float,
    segment_duration_seconds: float,
    samples_per_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if segment_duration_seconds <= 0:
        raise ValueError("segment_duration_seconds must be positive")
    if samples_per_segment <= 0:
        raise ValueError("samples_per_segment must be positive")
    edges = np.arange(
        0.0,
        duration_seconds + segment_duration_seconds,
        segment_duration_seconds,
        dtype=np.float64,
    )
    if edges[-1] < duration_seconds:
        edges = np.concatenate([edges, [duration_seconds]])
    edges[-1] = duration_seconds
    left = edges[:-1]
    right = edges[1:]
    durations = right - left
    offsets = (
        np.arange(int(samples_per_segment), dtype=np.float64) + 0.5
    ) / int(samples_per_segment)
    sample_times = (
        left[:, np.newaxis] + durations[:, np.newaxis] * offsets[np.newaxis, :]
    )
    return durations, sample_times.reshape(-1), sample_times


def _rk4_scalar_step(
    state: np.ndarray,
    t0: float,
    dt: float,
    rhs: Any,
) -> np.ndarray:
    k1 = rhs(t0, state)
    k2 = rhs(t0 + dt / 2.0, state + dt * k1 / 2.0)
    k3 = rhs(t0 + dt / 2.0, state + dt * k2 / 2.0)
    k4 = rhs(t0 + dt, state + dt * k3)
    return state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _sample_response(
    sample_times: np.ndarray,
    rhs: Any,
    initial_state: np.ndarray,
    max_step_seconds: float,
) -> np.ndarray:
    values = np.zeros(sample_times.size, dtype=np.float64)
    state = np.asarray(initial_state, dtype=np.float64)
    t_curr = 0.0
    for idx, t_next in enumerate(sample_times):
        dt_total = float(t_next - t_curr)
        steps = max(1, int(np.ceil(abs(dt_total) / max_step_seconds)))
        dt = dt_total / steps
        for _ in range(steps):
            state = _rk4_scalar_step(state, t_curr, dt, rhs)
            t_curr += dt
        values[idx] = state[0]
    return values


def _shape_from_real_response(
    *,
    response: np.ndarray,
    flat_sample_times: np.ndarray,
    sample_times_by_segment: np.ndarray,
    segment_duration_seconds: np.ndarray,
    rf_angular_frequency_rad_s: float,
    oscillator_phase_at_start_rad: float,
    time_scale_rad_per_s: float,
    steady_state_response: complex,
) -> PulseShape:
    demod = np.exp(
        -1j
        * (
            float(oscillator_phase_at_start_rad)
            + float(rf_angular_frequency_rad_s) * flat_sample_times
        )
    )
    drive_samples = 2.0 * response * demod
    drive = drive_samples.reshape(sample_times_by_segment.shape).mean(axis=1)
    if steady_state_response != 0:
        drive = drive / steady_state_response
    return PulseShape(
        duration=segment_duration_seconds * float(time_scale_rad_per_s),
        phase=np.angle(drive),
        amplitude=np.abs(drive),
    )


def build_nonresonant_circuit_pulse_library(
    *,
    absolute_phase_rad: Sequence[float] | np.ndarray,
    rf_frequency_hz: float,
    pulse_duration_seconds: float,
    time_scale_rad_per_s: float,
    tau_seconds: float,
    rotating_phase_rad: float = 0.0,
    post_delay_seconds: float = 0.0,
    pulse_kind: str = "refocusing",
    segment_duration_seconds: float | None = None,
    samples_per_segment: int = 16,
    max_step_seconds: float | None = None,
) -> PulseShapeLibrary:
    """Build a first-order non-resonant transmit pulse-shape library.

    The model integrates ``di/dt = (vin - i) / tau`` after transmitter turn-on
    and optional turn-off.  The real response is demodulated over half-RF-cycle
    segments by default and normalized by the steady-state complex response.
    """

    rf_omega = TAU * float(rf_frequency_hz)
    if rf_omega <= 0 or not np.isfinite(rf_omega):
        raise ValueError("rf_frequency_hz must be finite and positive")
    tau = float(tau_seconds)
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be finite and positive")
    total_duration = float(pulse_duration_seconds) + float(post_delay_seconds)
    segment_duration = (
        np.pi / rf_omega
        if segment_duration_seconds is None
        else float(segment_duration_seconds)
    )
    durations, flat_times, times_by_segment = _segment_sample_times(
        total_duration,
        segment_duration,
        int(samples_per_segment),
    )
    max_step = (
        segment_duration / max(4, int(samples_per_segment))
        if max_step_seconds is None
        else float(max_step_seconds)
    )
    steady = 1.0 / (1.0 + 1j * rf_omega * tau)
    shapes = []
    for absolute_phase in np.asarray(absolute_phase_rad, dtype=np.float64).reshape(-1):
        oscillator_phase = float(absolute_phase) - float(rotating_phase_rad)

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            if t <= float(pulse_duration_seconds):
                vin = np.cos(float(absolute_phase) + rf_omega * t)
            else:
                vin = 0.0
            return np.array([(vin - state[0]) / tau], dtype=np.float64)

        response = _sample_response(flat_times, rhs, np.array([0.0]), max_step)
        shapes.append(
            _shape_from_real_response(
                response=response,
                flat_sample_times=flat_times,
                sample_times_by_segment=times_by_segment,
                segment_duration_seconds=durations,
                rf_angular_frequency_rad_s=rf_omega,
                oscillator_phase_at_start_rad=oscillator_phase,
                time_scale_rad_per_s=time_scale_rad_per_s,
                steady_state_response=steady,
            )
        )
    return PulseShapeLibrary(
        absolute_phase_rad=np.asarray(absolute_phase_rad, dtype=np.float64),
        shapes={pulse_kind: tuple(shapes)},
    )


def build_tuned_resonator_pulse_library(
    *,
    absolute_phase_rad: Sequence[float] | np.ndarray,
    rf_frequency_hz: float,
    pulse_duration_seconds: float,
    time_scale_rad_per_s: float,
    resonant_frequency_hz: float,
    quality_factor: float,
    rotating_phase_rad: float = 0.0,
    post_delay_seconds: float = 0.0,
    pulse_kind: str = "refocusing",
    segment_duration_seconds: float | None = None,
    samples_per_segment: int = 16,
    max_step_seconds: float | None = None,
) -> PulseShapeLibrary:
    """Build a second-order tuned-resonator transmit pulse-shape library."""

    rf_omega = TAU * float(rf_frequency_hz)
    omega0 = TAU * float(resonant_frequency_hz)
    q_value = float(quality_factor)
    if rf_omega <= 0 or omega0 <= 0:
        raise ValueError("frequencies must be positive")
    if not np.isfinite(q_value) or q_value <= 0:
        raise ValueError("quality_factor must be finite and positive")
    total_duration = float(pulse_duration_seconds) + float(post_delay_seconds)
    segment_duration = (
        np.pi / rf_omega
        if segment_duration_seconds is None
        else float(segment_duration_seconds)
    )
    durations, flat_times, times_by_segment = _segment_sample_times(
        total_duration,
        segment_duration,
        int(samples_per_segment),
    )
    max_step = (
        segment_duration / max(4, int(samples_per_segment))
        if max_step_seconds is None
        else float(max_step_seconds)
    )
    damping = omega0 / q_value
    steady = omega0**2 / (omega0**2 - rf_omega**2 + 1j * damping * rf_omega)
    shapes = []
    for absolute_phase in np.asarray(absolute_phase_rad, dtype=np.float64).reshape(-1):
        oscillator_phase = float(absolute_phase) - float(rotating_phase_rad)

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            if t <= float(pulse_duration_seconds):
                vin = np.cos(float(absolute_phase) + rf_omega * t)
            else:
                vin = 0.0
            return np.array(
                [
                    state[1],
                    omega0**2 * vin - damping * state[1] - omega0**2 * state[0],
                ],
                dtype=np.float64,
            )

        response = _sample_response(flat_times, rhs, np.array([0.0, 0.0]), max_step)
        shapes.append(
            _shape_from_real_response(
                response=response,
                flat_sample_times=flat_times,
                sample_times_by_segment=times_by_segment,
                segment_duration_seconds=durations,
                rf_angular_frequency_rad_s=rf_omega,
                oscillator_phase_at_start_rad=oscillator_phase,
                time_scale_rad_per_s=time_scale_rad_per_s,
                steady_state_response=steady,
            )
        )
    return PulseShapeLibrary(
        absolute_phase_rad=np.asarray(absolute_phase_rad, dtype=np.float64),
        shapes={pulse_kind: tuple(shapes)},
        period_rad=np.pi,
    )


def nonresonant_dc_phase_perturbation(
    *,
    nutation_rate_rad_s: float,
    tau_seconds: float,
    longitudinal_fraction: float,
    phase_offset_rad: float = 0.0,
    applies_to: str = "refocusing",
) -> LongitudinalPhaseKick:
    """Return the simple non-resonant DC transient phase-shift model.

    The perturbation amplitude follows the Mandal 2015 estimate
    ``epsilon = 2 * omega_1 * tau * B1_z / |B1|``.
    """

    epsilon = (
        2.0
        * float(nutation_rate_rad_s)
        * float(tau_seconds)
        * float(longitudinal_fraction)
    )
    return LongitudinalPhaseKick(
        phase_amplitude_rad=float(epsilon),
        phase_offset_rad=float(phase_offset_rad),
        periodicity="full",
        applies_to=applies_to,
    )


def as_absolute_phase_spec(
    value: AbsolutePhaseSpec | Mapping[str, Any] | None,
) -> AbsolutePhaseSpec | None:
    """Normalize a user absolute-phase specification."""

    if value is None:
        return None
    if isinstance(value, AbsolutePhaseSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("absolute_phase must be an AbsolutePhaseSpec, mapping, or None")
    transient_raw = value.get("transient_model")
    transient_model: TransientModel | None
    if transient_raw is None:
        transient_model = None
    elif isinstance(
        transient_raw,
        (
            SinusoidalTransientPerturbation,
            LongitudinalPhaseKick,
            InterpolatedPulseShapeModel,
        ),
    ):
        transient_model = transient_raw
    elif isinstance(transient_raw, Mapping):
        kind = str(transient_raw.get("kind", "sinusoidal"))
        if kind == "sinusoidal":
            transient_model = sinusoidal_transient_from_mapping(transient_raw)
        elif kind in {"longitudinal_phase_kick", "nonresonant_dc_phase_kick"}:
            transient_model = longitudinal_phase_kick_from_mapping(transient_raw)
        elif kind == "library":
            transient_model = interpolated_pulse_model_from_mapping(transient_raw)
        else:
            raise ValueError(
                "transient_model kind must be 'sinusoidal', "
                "'longitudinal_phase_kick', or 'library'"
            )
    else:
        raise TypeError(
            "transient_model must be an absolute-phase model, mapping, or None"
        )
    return AbsolutePhaseSpec(
        rf_frequency_hz=float(value["rf_frequency_hz"]),
        rf_phase_at_zero_rad=float(value.get("rf_phase_at_zero_rad", 0.0)),
        transient_model=transient_model,
        receiver_phase_rad=float(value.get("receiver_phase_rad", 0.0)),
    )


def apply_absolute_phase_model(
    shape: PulseShape,
    spec: AbsolutePhaseSpec | None,
    absolute_phase_rad: float,
    pulse_kind: str,
) -> PulseShape:
    """Apply the optional absolute-phase transient model to a pulse shape."""

    if spec is None or spec.transient_model is None:
        return shape
    return spec.transient_model.apply(shape, absolute_phase_rad, pulse_kind)


def cpmg_refocus_start_times(
    *,
    excitation_start_seconds: float,
    excitation_duration_seconds: float,
    correction_delay_seconds: float,
    pre_refocus_delay_seconds: float,
    echo_spacing_seconds: float,
    num_echoes: int,
) -> np.ndarray:
    """Return refocusing-pulse start times for a CPMG-like train."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    first = (
        float(excitation_start_seconds)
        + float(excitation_duration_seconds)
        + float(correction_delay_seconds)
        + float(pre_refocus_delay_seconds)
    )
    return first + float(echo_spacing_seconds) * np.arange(
        int(num_echoes), dtype=np.float64
    )


def build_finite_cpmg_phase_schedule(
    *,
    spec: AbsolutePhaseSpec,
    excitation_start_seconds: float,
    excitation_duration_seconds: float,
    correction_delay_seconds: float,
    pre_refocus_delay_seconds: float,
    echo_spacing_seconds: float,
    num_echoes: int,
    excitation_phases_rad: Sequence[float] | np.ndarray = (np.pi / 2, 3 * np.pi / 2),
    refocus_rotating_phase_rad: float = 0.0,
) -> FiniteCPMGPhaseSchedule:
    """Build absolute RF phase schedule for a finite CPMG-like train."""

    if int(num_echoes) <= 0:
        raise ValueError("num_echoes must be positive")
    excitation_phases = np.asarray(excitation_phases_rad, dtype=np.float64).reshape(-1)
    if excitation_phases.size == 0:
        raise ValueError("excitation_phases_rad must not be empty")
    refocus_starts = cpmg_refocus_start_times(
        excitation_start_seconds=float(excitation_start_seconds),
        excitation_duration_seconds=float(excitation_duration_seconds),
        correction_delay_seconds=float(correction_delay_seconds),
        pre_refocus_delay_seconds=float(pre_refocus_delay_seconds),
        echo_spacing_seconds=float(echo_spacing_seconds),
        num_echoes=int(num_echoes),
    )
    excitation_absolute = np.array(
        [
            spec.pulse_phase(float(excitation_start_seconds), float(phase))
            for phase in excitation_phases
        ],
        dtype=np.float64,
    )
    refocus_absolute = np.array(
        [
            spec.pulse_phase(float(start), float(refocus_rotating_phase_rad))
            for start in refocus_starts
        ],
        dtype=np.float64,
    )
    pulse_start = np.concatenate(
        [
            np.full(excitation_phases.size, float(excitation_start_seconds)),
            refocus_starts,
        ]
    )
    pulse_kind = ("excitation",) * excitation_phases.size + ("refocusing",) * int(
        num_echoes
    )
    pulse_rotating = np.concatenate(
        [
            excitation_phases,
            np.full(int(num_echoes), float(refocus_rotating_phase_rad)),
        ]
    )
    pulse_absolute = np.concatenate([excitation_absolute, refocus_absolute])
    return FiniteCPMGPhaseSchedule(
        rf_frequency_hz=float(spec.rf_frequency_hz),
        excitation_start_seconds=float(excitation_start_seconds),
        excitation_duration_seconds=float(excitation_duration_seconds),
        correction_delay_seconds=float(correction_delay_seconds),
        pre_refocus_delay_seconds=float(pre_refocus_delay_seconds),
        echo_spacing_seconds=float(echo_spacing_seconds),
        excitation_rotating_phase_rad=excitation_phases,
        excitation_absolute_phase_rad=excitation_absolute,
        refocus_start_seconds=refocus_starts,
        refocus_rotating_phase_rad=float(refocus_rotating_phase_rad),
        refocus_absolute_phase_rad=refocus_absolute,
        pulse_start_seconds=pulse_start,
        pulse_kind=pulse_kind,
        pulse_rotating_phase_rad=pulse_rotating,
        pulse_absolute_phase_rad=pulse_absolute,
        receiver_phase_rad=float(spec.receiver_phase_rad),
    )


def build_finite_cpmg_pulse_plan(
    num_echoes: int,
    *,
    per_echo_refocusing: bool,
) -> FiniteCPMGPulsePlan:
    """Return pulse-matrix index vectors for finite CPMG phase cycling.

    Matrix indices follow the MATLAB-style convention used by the arbitrary
    pulse kernels: zero selects free precession, and positive indices are
    one-based entries into the precomputed pulse-matrix list.
    """

    if int(num_echoes) <= 0:
        raise ValueError("num_echoes must be positive")
    excitation_one = np.array([1, 0], dtype=np.int64)
    excitation_two = np.array([2, 0], dtype=np.int64)
    if per_echo_refocusing:
        refocus = np.asarray(
            [
                value
                for echo in range(int(num_echoes))
                for value in (0, 3 + echo, 0)
            ],
            dtype=np.int64,
        )
        matrix_count = 2 + int(num_echoes)
    else:
        refocus = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
        matrix_count = 3
    return FiniteCPMGPulsePlan(
        excitation_cycle_one=excitation_one,
        excitation_cycle_two=excitation_two,
        refocus_cycle=refocus,
        pulse_matrix_count=matrix_count,
    )


def build_cpmg_absolute_phase_metadata(
    *,
    spec: AbsolutePhaseSpec,
    excitation_start_seconds: float,
    excitation_phases_rad: np.ndarray,
    refocus_start_seconds: np.ndarray,
    refocus_rotating_phase_rad: float,
    echo_spacing_seconds: float,
    pulse_matrix_count: int,
    schedule: FiniteCPMGPhaseSchedule | None = None,
    pulse_plan: FiniteCPMGPulsePlan | None = None,
) -> AbsolutePhaseMetadata:
    """Build metadata for a CPMG-like absolute-phase schedule."""

    if schedule is None:
        excitation_phases = np.asarray(
            excitation_phases_rad, dtype=np.float64
        ).reshape(-1)
        refocus_times = np.asarray(refocus_start_seconds, dtype=np.float64).reshape(-1)
        excitation_absolute = np.array(
            [
                spec.pulse_phase(float(excitation_start_seconds), float(phase))
                for phase in excitation_phases
            ],
            dtype=np.float64,
        )
        refocus_absolute = np.array(
            [
                spec.pulse_phase(float(start), float(refocus_rotating_phase_rad))
                for start in refocus_times
            ],
            dtype=np.float64,
        )
        pulse_start = np.concatenate(
            [
                np.full(excitation_phases.size, float(excitation_start_seconds)),
                refocus_times,
            ]
        )
        pulse_kind = ("excitation",) * excitation_phases.size + (
            "refocusing",
        ) * refocus_times.size
        pulse_rotating = np.concatenate(
            [
                excitation_phases,
                np.full(refocus_times.size, float(refocus_rotating_phase_rad)),
            ]
        )
        pulse_absolute = np.concatenate([excitation_absolute, refocus_absolute])
        delta_refocus = float(
            wrap_phase(spec.rf_angular_frequency_rad_s * float(echo_spacing_seconds))
        )
    else:
        refocus_times = schedule.refocus_start_seconds
        excitation_absolute = schedule.excitation_absolute_phase_rad
        refocus_absolute = schedule.refocus_absolute_phase_rad
        pulse_start = schedule.pulse_start_seconds
        pulse_kind = schedule.pulse_kind
        pulse_rotating = schedule.pulse_rotating_phase_rad
        pulse_absolute = schedule.pulse_absolute_phase_rad
        delta_refocus = schedule.delta_refocus_phase_rad
    return AbsolutePhaseMetadata(
        rf_frequency_hz=float(spec.rf_frequency_hz),
        refocus_start_seconds=refocus_times,
        refocus_absolute_phase_rad=refocus_absolute,
        excitation_absolute_phase_rad=excitation_absolute,
        echo_spacing_seconds=float(echo_spacing_seconds),
        delta_refocus_phase_rad=delta_refocus,
        receiver_phase_rad=float(spec.receiver_phase_rad),
        transient_model=(
            type(spec.transient_model).__name__ if spec.transient_model else None
        ),
        pulse_matrix_count=int(pulse_matrix_count),
        pulse_start_seconds=pulse_start,
        pulse_kind=pulse_kind,
        pulse_rotating_phase_rad=pulse_rotating,
        pulse_absolute_phase_rad=pulse_absolute,
        excitation_matrix_indices=(
            pulse_plan.excitation_cycle_one.copy() if pulse_plan is not None else None
        ),
        refocus_matrix_indices=(
            pulse_plan.refocus_cycle.copy() if pulse_plan is not None else None
        ),
    )
