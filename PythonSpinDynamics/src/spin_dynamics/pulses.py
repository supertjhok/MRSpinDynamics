"""Pulse-shape utilities and compact probe pulse responses.

MATLAB reference folders:
    MATLABSpinDynamics/Version_3/code/Pulse Shape
    MATLABSpinDynamics/Version_3/code/opt_pulse
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.parameters import (
    set_params_matched_jmr,
    set_params_tuned_jmr,
    set_params_untuned_jmr,
)
from spin_dynamics.probes.matched import (
    find_coil_current,
    find_coil_current_wurst,
    matching_network_design2,
)
from spin_dynamics.probes.tuned import tuned_probe_lp, tuned_probe_rx_tf
from spin_dynamics.probes.untuned import untuned_probe_lp, untuned_probe_rx_tf


@dataclass(frozen=True)
class ProbePulseResponse:
    """Transmit pulse response and receiver transfer function arrays."""

    probe: str
    rotating_time: np.ndarray
    rotating_current: np.ndarray
    raw_time: np.ndarray
    raw_current: np.ndarray
    receiver_tf: np.ndarray
    receiver_tf_signal: np.ndarray | None = None
    scale: float = 1.0


@dataclass(frozen=True)
class WURSTPulse:
    """Piecewise-constant WURST pulse representation.

    `frequency_offset` is an angular offset in rad/s relative to the RF
    carrier. `phase` is the integrated rotating-frame phase suitable for ideal
    spin propagation; matched-probe transmit simulations may instead use a
    constant drive phase with the explicit `frequency_offset` vector.
    """

    duration: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    frequency_offset: np.ndarray
    acquisition: np.ndarray
    order: int
    sweep_width_rad_per_s: float
    total_duration: float
    initial_phase: float


@dataclass(frozen=True)
class UntunedPulseAdjustment:
    """Quantized phase and segment-length adjustment for an untuned pulse."""

    segment_lengths: np.ndarray
    phases: np.ndarray
    phase_rotation: float
    clock_period: float
    steady_state_phase: float


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        if name == "in_":
            return obj["in"]
    return getattr(obj, name)


def _with_fields(obj: Mapping[str, Any] | Any, **updates: Any) -> Any:
    if isinstance(obj, Mapping):
        out = dict(obj)
        out.update(updates)
        return out
    out = dict(vars(obj))
    out.update(updates)
    return out


def _as_vector(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def quantize_phase(phi: np.ndarray | list[float], num_phases: int) -> np.ndarray:
    """Quantize phases to the nearest evenly spaced phase state.

    Mirrors MATLAB `opt_pulse/quantize_phase.m`.
    """

    if int(num_phases) <= 0:
        raise ValueError("num_phases must be positive")
    phi_arr = _as_vector(phi)
    states = (2 * np.pi / int(num_phases)) * np.arange(int(num_phases), dtype=np.float64)
    indices = np.argmin(np.abs(states[:, np.newaxis] - phi_arr[np.newaxis, :]), axis=0)
    return states[indices]


def create_wurst_pulse(
    *,
    duration_seconds: float,
    sweep_width_rad_per_s: float,
    num_steps: int = 2000,
    order: int = 20,
    amplitude: float = 1.0,
    initial_phase: float = np.pi / 2,
    center_frequency_offset: float = 0.0,
) -> WURSTPulse:
    """Create a WURST amplitude and frequency-sweep pulse.

    The envelope follows ``1 - |cos(pi t / T)|**order`` and the frequency
    offset is swept linearly across `sweep_width_rad_per_s`. Segment phases are
    integrated from the angular frequency offsets at segment centers.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sweep_width_rad_per_s < 0:
        raise ValueError("sweep_width_rad_per_s must be non-negative")
    if int(num_steps) < 2:
        raise ValueError("num_steps must be at least 2")
    if int(order) <= 0:
        raise ValueError("order must be positive")
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")

    steps = int(num_steps)
    u = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    amp = float(amplitude) * (1.0 - np.abs(np.cos(np.pi * u)) ** int(order))
    freq = (
        float(center_frequency_offset)
        + np.linspace(
            -0.5 * float(sweep_width_rad_per_s),
            0.5 * float(sweep_width_rad_per_s),
            steps,
            dtype=np.float64,
        )
    )
    duration = np.full(steps, float(duration_seconds) / steps, dtype=np.float64)
    centers_phase_increment = freq * duration
    phase = float(initial_phase) + np.cumsum(centers_phase_increment) - 0.5 * centers_phase_increment
    return WURSTPulse(
        duration=duration,
        amplitude=amp,
        phase=phase,
        frequency_offset=freq,
        acquisition=np.zeros(steps, dtype=np.int64),
        order=int(order),
        sweep_width_rad_per_s=float(sweep_width_rad_per_s),
        total_duration=float(duration_seconds),
        initial_phase=float(initial_phase),
    )


def adjust_untuned_segment_lengths(
    segment_lengths: np.ndarray | list[float],
    phases: np.ndarray | list[float],
    sp: Mapping[str, Any] | Any | None = None,
    pp: Mapping[str, Any] | Any | None = None,
    *,
    num_phases: int | None = None,
) -> UntunedPulseAdjustment:
    """Adjust untuned-probe segment lengths to reduce switching transients.

    This ports the timing core of MATLAB `opt_pulse/untuned_pulse_adjust.m`
    without file loading or plotting. The input pulse is represented by segment
    lengths and phases; amplitudes and probe-current plotting are left to the
    caller.
    """

    if sp is None or pp is None:
        sp, pp = set_params_untuned_jmr()
    lengths = _as_vector(segment_lengths).copy()
    phase_input = _as_vector(phases)
    if lengths.size != phase_input.size:
        raise ValueError("segment_lengths and phases must have the same length")
    if lengths.size == 0:
        raise ValueError("segment_lengths and phases must not be empty")

    phase_count = int(num_phases) if num_phases is not None else int(_field(pp, "N")) // 2
    pvec = quantize_phase(phase_input, phase_count)

    w = float(_field(pp, "w"))
    n_clock = int(_field(pp, "N"))
    tclk = 2 * np.pi / (w * n_clock)
    rsref = _as_vector(_field(pp, "Rsref"))
    tau = float(_field(sp, "L")) / (float(_field(sp, "R")) + float(rsref[1]))
    theta = -np.arctan2(w * tau, 1)

    phase_rotation = (np.pi / 2 - theta) - pvec[0]
    pvec = pvec + phase_rotation
    adjusted = lengths.copy()

    for idx in range(lengths.size - 1):
        alpha = np.mod(-(pvec[idx] + pvec[idx + 1]) / 2 - theta, np.pi)
        if alpha <= np.pi / 2:
            tadj = alpha / w
        else:
            tadj = -(np.pi - alpha) / w
        tadj = np.round(tadj / tclk) * tclk
        adjusted[idx] += tadj
        adjusted[idx + 1] -= tadj

    end_options = np.array(
        [
            np.pi / 2 - pvec[-1] - theta,
            3 * np.pi / 2 - pvec[-1] - theta,
        ],
        dtype=np.float64,
    )
    tadj = end_options[np.argmin(np.abs(end_options))] / w
    tadj = np.round(tadj / tclk) * tclk
    adjusted[-1] += tadj

    return UntunedPulseAdjustment(
        segment_lengths=adjusted,
        phases=pvec,
        phase_rotation=float(phase_rotation),
        clock_period=float(tclk),
        steady_state_phase=float(theta),
    )


def tuned_rectangular_pulse_response(
    *,
    voltage_scale: float = 62.5,
    numpts: int = 10_000,
) -> ProbePulseResponse:
    """Return the JMR tuned-probe rectangular-pulse response.

    Mirrors the array-producing parts of MATLAB `Pulse Shape/tunedPulse.m`.
    """

    sp, pp = set_params_tuned_jmr(numpts=numpts)
    rotating_time, rotating_current, raw_time, raw_current = tuned_probe_lp(sp, pp)
    return ProbePulseResponse(
        probe="tuned",
        rotating_time=rotating_time,
        rotating_current=float(voltage_scale) * rotating_current,
        raw_time=raw_time,
        raw_current=float(voltage_scale) * raw_current,
        receiver_tf=tuned_probe_rx_tf(sp, pp),
        scale=float(voltage_scale),
    )


def untuned_rectangular_pulse_response(
    *,
    voltage_scale: float = 62.5,
    numpts: int = 2000,
) -> ProbePulseResponse:
    """Return the JMR untuned-probe rectangular-pulse response.

    Mirrors the array-producing parts of MATLAB `Pulse Shape/untunedPulse.m`.
    """

    sp, pp = set_params_untuned_jmr(numpts=numpts)
    rotating_time, rotating_current, raw_time, raw_current = untuned_probe_lp(sp, pp)
    return ProbePulseResponse(
        probe="untuned",
        rotating_time=rotating_time,
        rotating_current=float(voltage_scale) * rotating_current,
        raw_time=raw_time,
        raw_current=float(voltage_scale) * raw_current,
        receiver_tf=untuned_probe_rx_tf(sp, pp),
        scale=float(voltage_scale),
    )


def matched_rectangular_pulse_response(
    *,
    numpts: int = 2000,
) -> ProbePulseResponse:
    """Return the JMR matched-probe rectangular-pulse response.

    Mirrors the non-plotting portion of MATLAB `Pulse Shape/matchedPulse.m`.
    """

    sp, pp = set_params_matched_jmr(numpts=numpts)
    c1, c2 = matching_network_design2(sp.L, sp.Q, sp.f0, sp.Rs)
    sp_match = _with_fields(sp, C1=c1, C2=c2)
    pp_curr = _with_fields(pp, tp=pp.tref, phi=pp.pref, amp=pp.aref)
    rotating_time, rotating_current, tf_noise, tf_signal = find_coil_current(sp_match, pp_curr)
    return ProbePulseResponse(
        probe="matched",
        rotating_time=rotating_time,
        rotating_current=rotating_current,
        raw_time=np.array([], dtype=np.float64),
        raw_current=np.array([], dtype=np.complex128),
        receiver_tf=tf_noise,
        receiver_tf_signal=tf_signal,
        scale=1.0,
    )


def matched_wurst_pulse_response(
    pulse: WURSTPulse,
    *,
    numpts: int = 2000,
    q_value: float | None = None,
    drive_phase: float | None = None,
) -> ProbePulseResponse:
    """Return matched-probe transmit response to a WURST RF block."""

    sp, pp = set_params_matched_jmr(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp = _with_fields(
            sp,
            Q=float(q_value),
            R=2 * np.pi * _field(sp, "f0") * _field(sp, "L") / float(q_value),
        )
    c1, c2 = matching_network_design2(
        _field(sp, "L"),
        _field(sp, "Q"),
        _field(sp, "f0"),
        _field(sp, "Rs"),
    )
    sp_match = _with_fields(sp, C1=c1, C2=c2, plt_tx=0, plt_rx=0)
    phase = pulse.initial_phase if drive_phase is None else float(drive_phase)
    pp_curr = _with_fields(
        pp,
        tp=pulse.duration,
        phi=np.full(pulse.duration.size, phase, dtype=np.float64),
        amp=pulse.amplitude,
        freq=pulse.frequency_offset,
    )
    rotating_time, rotating_current, tf_noise, tf_signal = find_coil_current_wurst(
        sp_match,
        pp_curr,
    )
    return ProbePulseResponse(
        probe="matched_wurst",
        rotating_time=rotating_time,
        rotating_current=rotating_current,
        raw_time=np.array([], dtype=np.float64),
        raw_current=np.array([], dtype=np.complex128),
        receiver_tf=tf_noise,
        receiver_tf_signal=tf_signal,
        scale=1.0,
    )
