"""Bloch-Siegert phase workflows for broadband and multi-frequency NMR.

The conventions follow Mandal et al., JMR 242 (2014) 113-125. ``nutation_hz``
is the co-rotating RWA nutation frequency. A real linear laboratory field is
therefore ``2 * nutation_hz`` in the transverse Bloch equation. The exact
solver retains that field in the laboratory frame and consequently includes
the counter-rotating component omitted by the RWA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlochSiegertPhaseSweep:
    """Exact and perturbative paired-offset Bloch-Siegert phases."""

    durations_s: np.ndarray
    lower_offset_phase_rad: np.ndarray
    upper_offset_phase_rad: np.ndarray
    differential_phase_rad: np.ndarray
    common_mode_phase_rad: np.ndarray
    rwa_differential_phase_rad: np.ndarray
    second_order_differential_phase_rad: np.ndarray
    second_order_common_mode_phase_rad: np.ndarray


@dataclass(frozen=True)
class MultisliceBlochSiegertCorrection:
    """Mandal-2014 phase and timing corrections for interleaved slices."""

    slice_number: np.ndarray
    excitation_phase_error_rad: np.ndarray
    excitation_phase_correction_rad: np.ndarray
    excitation_timing_shift_s: np.ndarray


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _validated_pair(
    larmor_hz: float,
    offset_hz: float,
    nutation_hz: float,
) -> tuple[float, float, float]:
    f0 = _positive_finite(larmor_hz, "larmor_hz")
    offset = _positive_finite(offset_hz, "offset_hz")
    f1 = _positive_finite(nutation_hz, "nutation_hz")
    if offset >= f0:
        raise ValueError("offset_hz must be smaller than larmor_hz")
    return f0, offset, f1


def bloch_siegert_pair_second_order(
    durations_s: np.ndarray | float,
    *,
    larmor_hz: float,
    offset_hz: float,
    nutation_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return second-order lower/upper, differential, and common phases.

    The lower carrier is ``f0 - offset`` and the upper carrier is
    ``f0 + offset``. The co-rotating contribution gives the Mandal-2014
    differential phase ``T * omega1**2 / delta``. The common-mode sum cancels
    in the RWA and contains only the counter-rotating contribution.
    """

    f0, offset, f1 = _validated_pair(larmor_hz, offset_hz, nutation_hz)
    durations = np.asarray(durations_s, dtype=np.float64)
    if np.any(~np.isfinite(durations)) or np.any(durations < 0.0):
        raise ValueError("durations_s must be finite and nonnegative")
    omega0 = 2.0 * np.pi * f0
    delta = 2.0 * np.pi * offset
    omega1 = 2.0 * np.pi * f1
    lower = 0.5 * durations * omega1**2 * (
        1.0 / delta + 1.0 / (2.0 * omega0 - delta)
    )
    upper = 0.5 * durations * omega1**2 * (
        -1.0 / delta + 1.0 / (2.0 * omega0 + delta)
    )
    return lower, upper, lower - upper, lower + upper


def _rotate_bloch(magnetization: np.ndarray, omega: np.ndarray, dt: float) -> None:
    magnitude = float(np.linalg.norm(omega))
    axis = omega / magnitude
    angle = magnitude * dt
    initial = magnetization.copy()
    magnetization[:] = (
        initial * np.cos(angle)
        + np.cross(axis, initial) * np.sin(angle)
        + axis * np.dot(axis, initial) * (1.0 - np.cos(angle))
    )


def _lab_frame_phase_sweep(
    durations_s: np.ndarray,
    *,
    larmor_hz: float,
    rf_hz: float,
    nutation_hz: float,
    steps_per_cycle: int,
) -> np.ndarray:
    omega0 = 2.0 * np.pi * larmor_hz
    omega_rf = 2.0 * np.pi * rf_hz
    omega1 = 2.0 * np.pi * nutation_hz
    dt_nominal = 1.0 / (steps_per_cycle * max(larmor_hz, rf_hz))
    magnetization = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    phases = np.empty_like(durations_s)
    time_s = 0.0
    for index, target_s in enumerate(durations_s):
        while time_s < target_s:
            dt = min(dt_nominal, target_s - time_s)
            midpoint = time_s + 0.5 * dt
            omega = np.array(
                [2.0 * omega1 * np.cos(omega_rf * midpoint), 0.0, omega0]
            )
            _rotate_bloch(magnetization, omega, dt)
            time_s += dt
        coherence = (magnetization[0] + 1.0j * magnetization[1]) * np.exp(
            -1.0j * omega0 * target_s
        )
        phases[index] = np.angle(coherence)
    return np.unwrap(phases)


def simulate_bloch_siegert_phase_sweep(
    durations_s: np.ndarray,
    *,
    larmor_hz: float,
    offset_hz: float,
    nutation_hz: float,
    steps_per_cycle: int = 48,
) -> BlochSiegertPhaseSweep:
    """Simulate paired off-resonant pulses with the exact lab-frame field."""

    f0, offset, f1 = _validated_pair(larmor_hz, offset_hz, nutation_hz)
    durations = np.asarray(durations_s, dtype=np.float64).reshape(-1)
    if (
        durations.size == 0
        or np.any(~np.isfinite(durations))
        or np.any(durations < 0.0)
        or np.any(np.diff(durations) < 0.0)
    ):
        raise ValueError("durations_s must be a nonempty, sorted, nonnegative array")
    steps = int(steps_per_cycle)
    if steps < 12:
        raise ValueError("steps_per_cycle must be at least 12")
    lower = _lab_frame_phase_sweep(
        durations,
        larmor_hz=f0,
        rf_hz=f0 - offset,
        nutation_hz=f1,
        steps_per_cycle=steps,
    )
    upper = _lab_frame_phase_sweep(
        durations,
        larmor_hz=f0,
        rf_hz=f0 + offset,
        nutation_hz=f1,
        steps_per_cycle=steps,
    )
    _, _, second_difference, second_common = bloch_siegert_pair_second_order(
        durations,
        larmor_hz=f0,
        offset_hz=offset,
        nutation_hz=f1,
    )
    omega1 = 2.0 * np.pi * f1
    delta = 2.0 * np.pi * offset
    return BlochSiegertPhaseSweep(
        durations_s=durations,
        lower_offset_phase_rad=lower,
        upper_offset_phase_rad=upper,
        differential_phase_rad=np.unwrap(lower - upper),
        common_mode_phase_rad=np.unwrap(lower + upper),
        rwa_differential_phase_rad=durations * omega1**2 / delta,
        second_order_differential_phase_rad=second_difference,
        second_order_common_mode_phase_rad=second_common,
    )


def estimate_nutation_hz(
    differential_phase_rad: np.ndarray | float,
    durations_s: np.ndarray | float,
    *,
    offset_hz: float,
) -> np.ndarray:
    """Estimate co-rotating nutation from the Mandal-2014 differential phase."""

    phase = np.asarray(differential_phase_rad, dtype=np.float64)
    duration = np.asarray(durations_s, dtype=np.float64)
    offset = _positive_finite(offset_hz, "offset_hz")
    if np.any(~np.isfinite(phase)) or np.any(phase < 0.0):
        raise ValueError("differential_phase_rad must be finite and nonnegative")
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0.0):
        raise ValueError("durations_s must be positive and finite")
    return np.sqrt(phase * 2.0 * np.pi * offset / duration) / (2.0 * np.pi)


def estimate_larmor_hz_from_counter_rotating_phase(
    common_mode_phase_rad: np.ndarray | float,
    durations_s: np.ndarray | float,
    *,
    offset_hz: float,
    nutation_hz: float,
) -> np.ndarray:
    """Invert the paired-pulse common phase to estimate low-frequency B0.

    This is the second-order inversion of the counter-rotating term. The RWA
    predicts zero common phase and therefore cannot provide this estimate.
    """

    phase = np.asarray(common_mode_phase_rad, dtype=np.float64)
    duration = np.asarray(durations_s, dtype=np.float64)
    offset = _positive_finite(offset_hz, "offset_hz")
    f1 = _positive_finite(nutation_hz, "nutation_hz")
    if np.any(~np.isfinite(phase)) or np.any(phase <= 0.0):
        raise ValueError("common_mode_phase_rad must be positive and finite")
    if np.any(~np.isfinite(duration)) or np.any(duration <= 0.0):
        raise ValueError("durations_s must be positive and finite")
    delta = 2.0 * np.pi * offset
    omega1 = 2.0 * np.pi * f1
    numerator = duration * omega1**2 + np.sqrt(
        duration**2 * omega1**4 + 4.0 * phase**2 * delta**2
    )
    return numerator / (4.0 * phase * 2.0 * np.pi)


def counter_rotating_common_phase(
    larmor_hz: np.ndarray | float,
    durations_s: np.ndarray | float,
    *,
    offset_hz: float,
    nutation_hz: float,
) -> np.ndarray:
    """Return the vectorized second-order common phase of a paired-offset scan."""

    larmor = np.asarray(larmor_hz, dtype=np.float64)
    duration = np.asarray(durations_s, dtype=np.float64)
    offset = _positive_finite(offset_hz, "offset_hz")
    f1 = _positive_finite(nutation_hz, "nutation_hz")
    if np.any(~np.isfinite(larmor)) or np.any(larmor <= offset):
        raise ValueError("larmor_hz must be finite and greater than offset_hz")
    if np.any(~np.isfinite(duration)) or np.any(duration < 0.0):
        raise ValueError("durations_s must be finite and nonnegative")
    omega0 = 2.0 * np.pi * larmor
    delta = 2.0 * np.pi * offset
    omega1 = 2.0 * np.pi * f1
    return duration * omega1**2 * 2.0 * omega0 / (4.0 * omega0**2 - delta**2)


def mandal_multislice_correction(
    num_slices: int,
    *,
    nutation_hz: float,
    slice_spacing_hz: float,
    t90_s: float,
) -> MultisliceBlochSiegertCorrection:
    """Return Eqs. 14-16 of Mandal et al. for interleaved CPMG slices."""

    count = int(num_slices)
    if count < 1:
        raise ValueError("num_slices must be at least 1")
    f1 = _positive_finite(nutation_hz, "nutation_hz")
    spacing = _positive_finite(slice_spacing_hz, "slice_spacing_hz")
    t90 = _positive_finite(t90_s, "t90_s")
    slices = np.arange(1, count + 1)
    phase = np.zeros(count, dtype=np.float64)
    timing = np.zeros(count, dtype=np.float64)
    for index in range(1, count):
        integers = np.arange(1, index + 1, dtype=np.float64)
        phase[index] = -0.25 * np.pi * (f1 / spacing) * np.sum(1.0 / integers)
        timing[index] = t90 * np.sum(1.0 / integers**2)
    return MultisliceBlochSiegertCorrection(
        slice_number=slices,
        excitation_phase_error_rad=phase,
        excitation_phase_correction_rad=-phase,
        excitation_timing_shift_s=timing,
    )


__all__ = [
    "BlochSiegertPhaseSweep",
    "MultisliceBlochSiegertCorrection",
    "bloch_siegert_pair_second_order",
    "counter_rotating_common_phase",
    "estimate_larmor_hz_from_counter_rotating_phase",
    "estimate_nutation_hz",
    "mandal_multislice_correction",
    "simulate_bloch_siegert_phase_sweep",
]
