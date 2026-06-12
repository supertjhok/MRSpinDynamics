"""Pulse-shape utilities and compact probe pulse responses.

MATLAB reference folders:
    SpinDynamicsUpdated/Version_2/code/Pulse Shape
    SpinDynamicsUpdated/Version_2/code/opt_pulse
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
from spin_dynamics.probes.matched import find_coil_current, matching_network_design2
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
