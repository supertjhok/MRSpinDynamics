"""Ideal and finite-pulse dephasing filter functions for defect sensors."""

from __future__ import annotations

from typing import Literal

import numpy as np

from spin_dynamics.nano_mr.sequences import SensingSequence
from spin_dynamics.sequences.cpmg import (
    dephasing_filter_function as _ideal_filter,
)
from spin_dynamics.sequences.cpmg import (
    toggling_frame_integral as _ideal_integral,
)


PulseModel = Literal["ideal", "finite"]


def _refocusing_pulses(sequence: SensingSequence):
    pulses = sequence.electron_pulses
    for pulse in pulses:
        if not np.isclose(pulse.flip_angle_rad, np.pi):
            raise ValueError("scalar dephasing modulation requires electron pi pulses")
    return pulses


def modulation_function(
    sequence: SensingSequence,
    times_seconds,
    *,
    pulse_model: PulseModel = "ideal",
) -> np.ndarray:
    """Return the longitudinal toggling coefficient ``y(t)``.

    In the ideal model, each electron pi pulse changes the sign instantly at
    its center. In the finite model, a rectangular resonant pi pulse rotates
    the longitudinal operator continuously, giving a cosine transition between
    the adjacent ``+/-1`` free-evolution intervals.
    """

    times = np.asarray(times_seconds, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ValueError("times_seconds must be finite")
    if np.any(times < 0.0) or np.any(times > sequence.total_duration_seconds):
        raise ValueError("times_seconds must lie within the sensing window")
    pulses = _refocusing_pulses(sequence)
    if pulse_model not in ("ideal", "finite"):
        raise ValueError("pulse_model must be 'ideal' or 'finite'")

    values = np.ones(times.shape, dtype=np.float64)
    sign = 1.0
    for pulse in pulses:
        if pulse_model == "ideal" or pulse.is_instantaneous:
            values[times >= pulse.center_seconds] = -sign
            sign = -sign
            continue
        within = (times >= pulse.start_seconds) & (times <= pulse.end_seconds)
        after = times > pulse.end_seconds
        fraction = (
            times[within] - pulse.start_seconds
        ) / pulse.duration_seconds
        values[within] = sign * np.cos(pulse.flip_angle_rad * fraction)
        values[after] = -sign
        sign = -sign
    return values


def toggling_integral(
    sequence: SensingSequence,
    angular_frequencies,
    *,
    pulse_model: PulseModel = "ideal",
    samples_per_pulse: int = 64,
) -> complex | np.ndarray:
    """Return ``integral y(t) exp(i*omega*t) dt`` for a sensing sequence."""

    pulses = _refocusing_pulses(sequence)
    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("angular_frequencies must be finite")
    if pulse_model == "ideal" or all(item.is_instantaneous for item in pulses):
        result = _ideal_integral(
            frequencies,
            [item.center_seconds for item in pulses],
            sequence.total_duration_seconds,
        )
        return result
    if pulse_model != "finite":
        raise ValueError("pulse_model must be 'ideal' or 'finite'")

    grid = modulation_time_grid(
        sequence,
        samples_per_pulse=samples_per_pulse,
    )
    modulation = modulation_function(sequence, grid, pulse_model="finite")
    scalar_input = frequencies.ndim == 0
    omega = np.atleast_1d(frequencies)
    phase = np.exp(1j * omega[:, None] * grid[None, :])
    integrand = phase * modulation[None, :]
    response = np.sum(
        0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(grid),
        axis=1,
    )
    if scalar_input:
        return complex(response[0])
    return response.reshape(frequencies.shape)


def dephasing_filter_function(
    sequence: SensingSequence,
    angular_frequencies,
    *,
    pulse_model: PulseModel = "ideal",
    samples_per_pulse: int = 64,
) -> float | np.ndarray:
    """Return the dimensionless pure-dephasing filter ``omega^2 |Y|^2``."""

    frequencies = np.asarray(angular_frequencies, dtype=np.float64)
    pulses = _refocusing_pulses(sequence)
    if pulse_model == "ideal" or all(item.is_instantaneous for item in pulses):
        values = _ideal_filter(
            frequencies,
            [item.center_seconds for item in pulses],
            sequence.total_duration_seconds,
        )
    else:
        response = toggling_integral(
            sequence,
            frequencies,
            pulse_model=pulse_model,
            samples_per_pulse=samples_per_pulse,
        )
        values = frequencies**2 * np.abs(response) ** 2
    if frequencies.ndim == 0:
        return float(np.asarray(values))
    return np.asarray(values)


def modulation_time_grid(
    sequence: SensingSequence,
    *,
    samples_per_pulse: int = 64,
    min_free_samples: int = 2049,
) -> np.ndarray:
    """Return an integration grid resolving every finite electron pulse."""

    if samples_per_pulse < 4:
        raise ValueError("samples_per_pulse must be at least 4")
    if min_free_samples < 2:
        raise ValueError("min_free_samples must be at least 2")
    grids = [
        np.linspace(
            0.0,
            sequence.total_duration_seconds,
            min_free_samples,
            dtype=np.float64,
        )
    ]
    for pulse in _refocusing_pulses(sequence):
        if pulse.duration_seconds > 0.0:
            grids.append(
                np.linspace(
                    pulse.start_seconds,
                    pulse.end_seconds,
                    samples_per_pulse + 1,
                    dtype=np.float64,
                )
            )
    return np.unique(np.concatenate(grids))


__all__ = [
    "PulseModel",
    "dephasing_filter_function",
    "modulation_function",
    "modulation_time_grid",
    "toggling_integral",
]
