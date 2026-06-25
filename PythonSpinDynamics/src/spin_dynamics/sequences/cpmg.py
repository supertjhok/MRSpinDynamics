"""CPMG and UDD timing helpers for ideal refocusing-pulse trains.

The utilities in this module describe ideal instantaneous pi pulses within a
fixed evolution window. They are useful for building pulse schedules and for
comparing dynamical-decoupling filter functions before routing a sequence into
one of the heavier spin-dynamics workflow layers.
"""

from __future__ import annotations

import numpy as np


def cpmg_pulse_times(num_pulses: int, total_duration: float) -> np.ndarray:
    """Return ideal CPMG refocusing pulse centers in ``[0, total_duration]``.

    For ``n`` pulses over an evolution window ``T``, CPMG places the pi pulses
    at ``(j - 1/2) T / n`` for ``j = 1, ..., n``. The first and last free
    intervals are half the duration of the interior intervals.
    """

    count = _validate_num_pulses(num_pulses)
    duration = _validate_duration(total_duration)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    pulse_index = np.arange(1, count + 1, dtype=np.float64)
    return duration * (pulse_index - 0.5) / count


def udd_pulse_times(num_pulses: int, total_duration: float) -> np.ndarray:
    """Return Uhrig dynamical decoupling pulse centers.

    For ``n`` pulses over an evolution window ``T``, UDD places the ``j``th pi
    pulse at ``T sin^2(j pi / (2 n + 2))`` for ``j = 1, ..., n``.
    """

    count = _validate_num_pulses(num_pulses)
    duration = _validate_duration(total_duration)
    if count == 0:
        return np.empty(0, dtype=np.float64)
    pulse_index = np.arange(1, count + 1, dtype=np.float64)
    angles = np.pi * pulse_index / (2 * count + 2)
    return duration * np.sin(angles) ** 2


def interval_durations(
    pulse_times: np.ndarray | list[float] | tuple[float, ...],
    total_duration: float,
) -> np.ndarray:
    """Return free-precession intervals between sequence endpoints and pulses."""

    duration = _validate_duration(total_duration)
    times = _validate_pulse_times(pulse_times, duration)
    edges = np.concatenate(([0.0], times, [duration]))
    return np.diff(edges)


def toggling_frame_integral(
    angular_frequencies: float | np.ndarray,
    pulse_times: np.ndarray | list[float] | tuple[float, ...],
    total_duration: float,
) -> complex | np.ndarray:
    """Return ``int y(t) exp(i omega t) dt`` for an ideal pi-pulse train.

    ``y(t)`` starts at ``+1`` and changes sign at each pulse. The returned
    complex response gives the phase accumulated from a sinusoidal detuning at
    angular frequency ``omega``. Zero frequency is handled by the exact interval
    sum, avoiding the removable singularity in the exponential formula.
    """

    duration = _validate_duration(total_duration)
    times = _validate_pulse_times(pulse_times, duration)
    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    scalar_input = frequencies.ndim == 0
    omega = np.atleast_1d(frequencies).astype(np.float64, copy=False)
    response = np.zeros(omega.shape, dtype=np.complex128)

    edges = np.concatenate(([0.0], times, [duration]))
    signs = np.where(np.arange(edges.size - 1) % 2 == 0, 1.0, -1.0)
    nonzero = omega != 0.0
    if np.any(nonzero):
        omega_nz = omega[nonzero]
        segment_response = np.zeros(omega_nz.shape, dtype=np.complex128)
        for sign, start, stop in zip(signs, edges[:-1], edges[1:]):
            segment_response += sign * (
                np.exp(1j * omega_nz * stop) - np.exp(1j * omega_nz * start)
            )
        response[nonzero] = segment_response / (1j * omega_nz)
    if np.any(~nonzero):
        response[~nonzero] = np.sum(signs * np.diff(edges))

    if scalar_input:
        return complex(response[0])
    return response.reshape(frequencies.shape)


def dephasing_filter_function(
    angular_frequencies: float | np.ndarray,
    pulse_times: np.ndarray | list[float] | tuple[float, ...],
    total_duration: float,
) -> float | np.ndarray:
    """Return the dimensionless pure-dephasing filter ``omega^2 |Y|^2``.

    Here ``Y(omega)`` is the toggling-frame integral returned by
    :func:`toggling_frame_integral`. This is the common dynamical-decoupling
    filter-function convention for ideal instantaneous pi pulses.
    """

    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    response = toggling_frame_integral(frequencies, pulse_times, total_duration)
    values = frequencies**2 * np.abs(response) ** 2
    if frequencies.ndim == 0:
        return float(np.asarray(values))
    return values


def _validate_num_pulses(num_pulses: int) -> int:
    if isinstance(num_pulses, bool):
        raise ValueError("num_pulses must be a non-negative integer")
    count = int(num_pulses)
    if count != num_pulses or count < 0:
        raise ValueError("num_pulses must be a non-negative integer")
    return count


def _validate_duration(total_duration: float) -> float:
    duration = float(total_duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("total_duration must be positive and finite")
    return duration


def _validate_pulse_times(
    pulse_times: np.ndarray | list[float] | tuple[float, ...],
    total_duration: float,
) -> np.ndarray:
    times = np.asarray(pulse_times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("pulse_times must be one-dimensional")
    if times.size == 0:
        return times
    if not np.all(np.isfinite(times)):
        raise ValueError("pulse_times must be finite")
    if np.any(times <= 0.0) or np.any(times >= total_duration):
        raise ValueError("pulse_times must lie strictly inside total_duration")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("pulse_times must be strictly increasing")
    return times.astype(np.float64, copy=False)
