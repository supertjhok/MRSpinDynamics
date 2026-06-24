"""Pulsed-gradient spin-echo diffusion workflows."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.motion import (
    Boundary,
    MotionFieldMaps2D,
    ParticleEnsemble,
    Velocity,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import (
    MotionSequenceResult,
    MotionSequenceStep,
    run_motion_sequence,
)


PGSEBackend = Literal["moment", "walkers"]
PGSEAxis = Literal["x", "z"]


@dataclass(frozen=True)
class PGSEMomentResult:
    """Deterministic PGSE result from gradient-moment diffusion attenuation."""

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    diffusion_attenuation: float
    t2_attenuation: np.ndarray
    gradient_amplitude: float
    gradient_duration: float
    diffusion_time: float
    diffusion_coefficient: float
    gamma: float
    backend: str = "moment"


@dataclass(frozen=True)
class PGSEWalkerResult:
    """Random-walker PGSE result from explicit diffusive displacement."""

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    sequence: MotionSequenceResult
    initial_ensemble: ParticleEnsemble
    gradient_amplitude: float
    gradient_duration: float
    diffusion_time: float
    diffusion_coefficient: float
    gamma: float
    backend: str = "walkers"


def pgse_b_value(
    gradient_amplitude: float,
    gradient_duration: float,
    diffusion_time: float,
    *,
    gamma: float = 2.675e8,
) -> float:
    """Return the rectangular Stejskal-Tanner PGSE b-value.

    ``diffusion_time`` is the separation between the leading edges of the two
    gradient lobes. The physical gradient lobes have the same polarity across
    the refocusing pulse; in the effective coherence frame their signs are
    opposite, giving ``b = (gamma * G * delta)^2 * (Delta - delta / 3)``.
    """

    gradient = float(gradient_amplitude)
    delta = float(gradient_duration)
    delta_big = float(diffusion_time)
    gamma_value = float(gamma)
    if delta < 0.0 or delta_big < 0.0:
        raise ValueError("gradient_duration and diffusion_time must be non-negative")
    if delta_big < delta:
        raise ValueError("diffusion_time must be at least gradient_duration")
    return float((gamma_value * gradient * delta) ** 2 * (delta_big - delta / 3.0))


def gradient_moment_b_value(
    segments: Iterable[tuple[float, float]],
    *,
    gamma: float = 2.675e8,
) -> float:
    """Integrate ``q(t)^2`` for a piecewise-constant effective gradient.

    Each segment is ``(duration_seconds, effective_gradient_t_per_m)``. The
    effective gradient should already include RF-coherence sign changes. This
    helper is useful for validating rectangular PGSE and for future arbitrary
    diffusion-gradient waveforms.
    """

    q = 0.0
    integral = 0.0
    gamma_value = float(gamma)
    for duration, gradient in segments:
        dt = float(duration)
        slope = gamma_value * float(gradient)
        if dt < 0.0:
            raise ValueError("segment durations must be non-negative")
        integral += q * q * dt + q * slope * dt**2 + (slope * slope * dt**3) / 3.0
        q += slope * dt
    if abs(q) > 1e-9 * max(1.0, abs(gamma_value)):
        # A residual moment means stationary spins are not fully refocused.
        # The integral is still meaningful, but it is not a pure diffusion
        # weighting b-value in the usual Stejskal-Tanner sense.
        pass
    return float(integral)


def run_pgse_moment(
    *,
    num_echoes: int = 1,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    diffusion_time: float = 20.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    t2_seconds: float = np.inf,
    first_echo_time_seconds: float | None = None,
    echo_spacing_seconds: float | None = None,
    initial_signal: complex = 1.0 + 0.0j,
    gamma: float = 2.675e8,
) -> PGSEMomentResult:
    """Run a fast ideal PGSE or PGSE-prepared CPMG calculation.

    The diffusion weighting is computed from the effective gradient moment. For
    ``num_echoes > 1`` the same PGSE preparation attenuates each CPMG echo, and
    the echo-to-echo decay is governed by ``t2_seconds``.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if t2_seconds <= 0.0 and not np.isinf(t2_seconds):
        raise ValueError("t2_seconds must be positive or infinite")
    delta = float(gradient_duration)
    delta_big = float(diffusion_time)
    b_value = gradient_moment_b_value(
        [
            (delta, float(gradient_amplitude)),
            (delta_big - delta, 0.0),
            (delta, -float(gradient_amplitude)),
        ],
        gamma=gamma,
    )
    first_echo = (
        2.0 * delta_big
        if first_echo_time_seconds is None
        else float(first_echo_time_seconds)
    )
    if first_echo <= 0.0:
        raise ValueError("first_echo_time_seconds must be positive")
    if num_echoes > 1:
        if echo_spacing_seconds is None:
            echo_spacing = first_echo
        else:
            echo_spacing = float(echo_spacing_seconds)
        if echo_spacing <= 0.0:
            raise ValueError("echo_spacing_seconds must be positive")
    else:
        echo_spacing = 0.0
    echo_times = first_echo + echo_spacing * np.arange(int(num_echoes), dtype=np.float64)
    diffusion_attenuation = float(np.exp(-b_value * float(diffusion_coefficient)))
    t2_attenuation = np.exp(-echo_times / float(t2_seconds))
    signal = complex(initial_signal) * diffusion_attenuation * t2_attenuation
    return PGSEMomentResult(
        signal=np.asarray(signal, dtype=np.complex128),
        echo_times=echo_times,
        b_value=b_value,
        diffusion_attenuation=diffusion_attenuation,
        t2_attenuation=t2_attenuation,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=delta,
        diffusion_time=delta_big,
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=float(gamma),
    )


def run_pgse_walkers(
    *,
    rho: Iterable[float] | np.ndarray | None = None,
    x_axis: Iterable[float] | np.ndarray | None = None,
    z_axis: Iterable[float] | np.ndarray | None = None,
    fields: MotionFieldMaps2D | None = None,
    num_echoes: int = 1,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    diffusion_time: float = 20.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    gamma: float = 2.675e8,
    gradient_axis: PGSEAxis = "x",
    walkers_per_cell: int = 128,
    seed: int | None = None,
    jitter: bool = False,
    excitation_duration: float = 100.0e-6,
    refocusing_duration: float = 200.0e-6,
    echo_spacing_seconds: float | None = None,
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 8,
) -> PGSEWalkerResult:
    """Run PGSE with explicit random-walker diffusion.

    The physical gradient lobes have the same polarity before and after the
    refocusing pulse. The 180-degree RF pulse flips the coherence sign, so this
    is equivalent to a sign-reversed effective gradient moment for stationary
    spins.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("RF pulse durations must be positive")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")
    delta = float(gradient_duration)
    delta_big = float(diffusion_time)
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    if delta_big <= delta + refocusing_duration:
        raise ValueError(
            "diffusion_time must exceed gradient_duration + refocusing_duration"
        )

    rho_arr = np.ones((1, 1), dtype=np.float64) if rho is None else _map2d(rho, "rho")
    x = (
        np.array([0.0], dtype=np.float64)
        if x_axis is None
        else _axis(x_axis, "x_axis", rho_arr.shape[0])
    )
    z = (
        np.array([0.0], dtype=np.float64)
        if z_axis is None
        else _axis(z_axis, "z_axis", rho_arr.shape[1])
    )
    ensemble = initialize_ensemble_from_density(
        rho_arr,
        x,
        z,
        walkers_per_cell=int(walkers_per_cell),
        diffusion_coefficient=float(diffusion_coefficient),
        seed=seed,
        jitter=jitter,
    )
    if fields is None:
        fields = _default_motion_fields(x, z, diffusion_time, diffusion_coefficient)

    gradient = _gradient_tuple(float(gamma) * float(gradient_amplitude), gradient_axis)
    gap = 0.5 * (delta_big - delta - float(refocusing_duration))
    steps = _make_pgse_steps(
        num_echoes=int(num_echoes),
        gradient_duration=delta,
        gradient=gradient,
        gap=gap,
        excitation_duration=float(excitation_duration),
        refocusing_duration=float(refocusing_duration),
        echo_spacing_seconds=echo_spacing_seconds,
        substeps_per_interval=int(substeps_per_interval),
    )
    sequence = run_motion_sequence(
        ensemble,
        fields,
        steps,
        velocity=velocity,
        rng=np.random.default_rng(seed),
        t1=t1_seconds,
        t2=t2_seconds,
        boundary=boundary,
        default_substeps=int(substeps_per_interval),
    )
    return PGSEWalkerResult(
        signal=sequence.signal,
        echo_times=sequence.sample_times,
        b_value=pgse_b_value(
            gradient_amplitude,
            gradient_duration,
            diffusion_time,
            gamma=gamma,
        ),
        sequence=sequence,
        initial_ensemble=ensemble,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=delta,
        diffusion_time=delta_big,
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=float(gamma),
    )


def run_pgse(
    *,
    backend: PGSEBackend = "moment",
    **kwargs: object,
) -> PGSEMomentResult | PGSEWalkerResult:
    """Dispatch to the moment or random-walker PGSE backend."""

    if backend == "moment":
        return run_pgse_moment(**kwargs)
    if backend == "walkers":
        return run_pgse_walkers(**kwargs)
    raise ValueError("backend must be 'moment' or 'walkers'")


def _make_pgse_steps(
    *,
    num_echoes: int,
    gradient_duration: float,
    gradient: tuple[float, float],
    gap: float,
    excitation_duration: float,
    refocusing_duration: float,
    echo_spacing_seconds: float | None,
    substeps_per_interval: int,
) -> tuple[MotionSequenceStep, ...]:
    steps = [
        MotionSequenceStep(
            duration=excitation_duration,
            rf_amplitude=(0.5 * np.pi) / excitation_duration,
            rf_phase=np.pi / 2,
            substeps=max(1, substeps_per_interval),
            label="excitation_90",
        ),
        MotionSequenceStep(
            duration=gradient_duration,
            gradient=gradient,
            substeps=substeps_per_interval,
            label="pgse_lobe_1",
        ),
        MotionSequenceStep(
            duration=gap,
            substeps=substeps_per_interval,
            label="pgse_gap_1",
        ),
        MotionSequenceStep(
            duration=refocusing_duration,
            rf_amplitude=np.pi / refocusing_duration,
            rf_phase=0.0,
            substeps=max(1, substeps_per_interval),
            label="pgse_180",
        ),
        MotionSequenceStep(
            duration=gap,
            substeps=substeps_per_interval,
            label="pgse_gap_2",
        ),
        MotionSequenceStep(
            duration=gradient_duration,
            gradient=gradient,
            acquire=True,
            num_samples=1,
            substeps=substeps_per_interval,
            label="echo_1",
        ),
    ]
    if num_echoes == 1:
        return tuple(steps)
    if echo_spacing_seconds is None:
        raise ValueError("echo_spacing_seconds is required when num_echoes > 1")
    echo_spacing = float(echo_spacing_seconds)
    if echo_spacing <= refocusing_duration:
        raise ValueError("echo_spacing_seconds must exceed refocusing_duration")
    half_free = 0.5 * (echo_spacing - refocusing_duration)
    for echo_index in range(2, num_echoes + 1):
        steps.extend(
            [
                MotionSequenceStep(
                    duration=half_free,
                    substeps=substeps_per_interval,
                    label=f"echo_{echo_index}_pre",
                ),
                MotionSequenceStep(
                    duration=refocusing_duration,
                    rf_amplitude=np.pi / refocusing_duration,
                    rf_phase=0.0,
                    substeps=max(1, substeps_per_interval),
                    label=f"echo_{echo_index}_180",
                ),
                MotionSequenceStep(
                    duration=half_free,
                    acquire=True,
                    num_samples=1,
                    substeps=substeps_per_interval,
                    label=f"echo_{echo_index}",
                ),
            ]
        )
    return tuple(steps)


def _gradient_tuple(value: float, axis: PGSEAxis) -> tuple[float, float]:
    if axis == "x":
        return (float(value), 0.0)
    if axis == "z":
        return (0.0, float(value))
    raise ValueError("gradient_axis must be 'x' or 'z'")


def _default_motion_fields(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    diffusion_time: float,
    diffusion_coefficient: float,
) -> MotionFieldMaps2D:
    total_time = max(float(diffusion_time), 0.0)
    sigma = np.sqrt(max(0.0, 2.0 * float(diffusion_coefficient) * total_time))
    margin = max(10.0 * sigma, 1.0e-6)
    x_min = float(np.min(x_axis)) - margin
    x_max = float(np.max(x_axis)) + margin
    z_min = float(np.min(z_axis)) - margin
    z_max = float(np.max(z_axis)) + margin
    if x_min == x_max:
        x_min -= margin
        x_max += margin
    if z_min == z_max:
        z_min -= margin
        z_max += margin
    return make_motion_field_maps_2d([x_min, x_max], [z_min, z_max])


def _map2d(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _axis(
    values: Iterable[float] | np.ndarray,
    name: str,
    expected_size: int,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != expected_size:
        raise ValueError(f"{name} length must match rho shape")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr
