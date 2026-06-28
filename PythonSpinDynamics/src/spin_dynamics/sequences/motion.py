"""Sequence drivers for moving-isochromat simulations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.motion import (
    Boundary,
    MotionFieldMaps,
    MotionFieldMaps2D,
    ParticleEnsemble,
    Velocity,
    apply_free_precession,
    apply_rf_rotation,
    move_ensemble,
    receive_signal,
)
from spin_dynamics.sequences.cpmg import udd_pulse_times

# Either field-map container works: the engine only calls ``sample`` and reads
# ``bounds``. ``MotionFieldMaps2D`` is the 2-D form; ``MotionFieldMaps`` is n-D.
MotionFields = MotionFieldMaps2D | MotionFieldMaps


DetuningWaveform = (
    float | Iterable[float] | np.ndarray | Callable[..., float | np.ndarray] | None
)


@dataclass(frozen=True)
class MotionSequenceStep:
    """One interval in a moving-isochromat pulse sequence.

    The interval can include free precession, a rectangular RF field, gradients,
    deterministic velocity, diffusion from the ensemble, and receive sampling.
    Long intervals are split into `substeps` so positions and fields are sampled
    along the particle trajectories.
    """

    duration: float
    gradient: tuple[float, float] = (0.0, 0.0)
    rf_amplitude: float = 0.0
    rf_phase: float = 0.0
    acquire: bool = False
    num_samples: int = 0
    substeps: int | None = None
    label: str = ""


@dataclass(frozen=True)
class MotionSequenceResult:
    """Result from a moving-isochromat sequence simulation."""

    final_ensemble: ParticleEnsemble
    signal: np.ndarray
    sample_times: np.ndarray
    sample_labels: tuple[str, ...]
    step_end_times: np.ndarray
    step_labels: tuple[str, ...]


def run_motion_sequence(
    ensemble: ParticleEnsemble,
    fields: MotionFields,
    steps: Sequence[MotionSequenceStep],
    *,
    velocity: Velocity = None,
    rng: np.random.Generator | None = None,
    t1: float | Iterable[float] | np.ndarray = np.inf,
    t2: float | Iterable[float] | np.ndarray = np.inf,
    mth: float | Iterable[float] | np.ndarray = 1.0,
    boundary: Boundary = "reflect",
    default_substeps: int = 1,
    detuning_waveform: DetuningWaveform = None,
) -> MotionSequenceResult:
    """Run a sequence while moving particles through sampled field maps.

    This is the sequence-level companion to `spin_dynamics.motion`: it advances
    particles, samples B0/B1 maps at their current positions, applies RF/free
    precession updates, and records receive samples at requested acquisition
    times.
    """

    if default_substeps <= 0:
        raise ValueError("default_substeps must be positive")
    current = ensemble
    time = 0.0
    signals: list[complex] = []
    sample_times: list[float] = []
    sample_labels: list[str] = []
    step_end_times: list[float] = []
    step_labels: list[str] = []
    generator = np.random.default_rng() if rng is None else rng

    for index, step in enumerate(steps):
        _validate_step(step)
        label = step.label or f"step_{index}"
        current, time, new_signals, new_times, new_labels = _run_step(
            current,
            fields,
            step,
            time=time,
            velocity=velocity,
            rng=generator,
            t1=t1,
            t2=t2,
            mth=mth,
            boundary=boundary,
            default_substeps=int(default_substeps),
            detuning_waveform=detuning_waveform,
            label=label,
        )
        signals.extend(new_signals)
        sample_times.extend(new_times)
        sample_labels.extend(new_labels)
        step_end_times.append(time)
        step_labels.append(label)

    return MotionSequenceResult(
        final_ensemble=current,
        signal=np.asarray(signals, dtype=np.complex128),
        sample_times=np.asarray(sample_times, dtype=np.float64),
        sample_labels=tuple(sample_labels),
        step_end_times=np.asarray(step_end_times, dtype=np.float64),
        step_labels=tuple(step_labels),
    )


def make_motion_cpmg_sequence(
    num_echoes: int,
    echo_spacing: float,
    *,
    excitation_duration: float,
    refocusing_duration: float,
    excitation_phase: float = np.pi / 2,
    refocusing_phase: float = 0.0,
    gradient: tuple[float, float] = (0.0, 0.0),
    substeps_per_interval: int = 1,
) -> tuple[MotionSequenceStep, ...]:
    """Build a rectangular-pulse CPMG sequence for moving isochromats.

    Each echo records one receive sample at the echo center. Gradients are
    applied during the free-precession windows, while RF intervals sample local
    B1 transmit and B0 at the moving particle positions.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing <= 0.0:
        raise ValueError("echo_spacing must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("pulse durations must be positive")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")
    half_free = 0.5 * float(echo_spacing) - 0.5 * float(refocusing_duration)
    if half_free < 0.0:
        raise ValueError("echo_spacing must be at least refocusing_duration")
    zero_gradient = tuple(0.0 for _ in tuple(gradient))

    steps: list[MotionSequenceStep] = [
        MotionSequenceStep(
            duration=float(excitation_duration),
            gradient=zero_gradient,
            rf_amplitude=(0.5 * np.pi) / float(excitation_duration),
            rf_phase=float(excitation_phase),
            substeps=max(1, int(substeps_per_interval)),
            label="excitation_90",
        )
    ]
    for echo_index in range(int(num_echoes)):
        echo_num = echo_index + 1
        steps.extend(
            [
                MotionSequenceStep(
                    duration=half_free,
                    gradient=gradient,
                    substeps=int(substeps_per_interval),
                    label=f"echo_{echo_num}_pre",
                ),
                MotionSequenceStep(
                    duration=float(refocusing_duration),
                    gradient=zero_gradient,
                    rf_amplitude=np.pi / float(refocusing_duration),
                    rf_phase=float(refocusing_phase),
                    substeps=max(1, int(substeps_per_interval)),
                    label=f"echo_{echo_num}_180",
                ),
                MotionSequenceStep(
                    duration=half_free,
                    gradient=gradient,
                    acquire=True,
                    num_samples=1,
                    substeps=int(substeps_per_interval),
                    label=f"echo_{echo_num}",
                ),
            ]
        )
    return tuple(steps)


def make_motion_udd_sequence(
    num_pulses: int,
    total_duration: float,
    *,
    excitation_duration: float,
    refocusing_duration: float,
    excitation_phase: float = np.pi / 2,
    refocusing_phase: float = 0.0,
    gradient: tuple[float, float] = (0.0, 0.0),
    substeps_per_interval: int = 1,
) -> tuple[MotionSequenceStep, ...]:
    """Build a rectangular-pulse UDD sequence for moving isochromats.

    The UDD pulse centers are placed inside the evolution window after the
    excitation pulse using ``t_j = T sin^2(j*pi/(2*n + 2))``. A single receive
    sample is recorded at the end of the evolution window. Gradients are
    applied during free-precession windows, matching the CPMG motion helper's
    finite-pulse convention.
    """

    if num_pulses < 0:
        raise ValueError("num_pulses must be non-negative")
    if total_duration <= 0.0:
        raise ValueError("total_duration must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("pulse durations must be positive")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")

    centers = udd_pulse_times(int(num_pulses), float(total_duration))
    half_pulse = 0.5 * float(refocusing_duration)
    if centers.size > 0:
        starts = centers - half_pulse
        stops = centers + half_pulse
        if starts[0] < 0.0 or stops[-1] > total_duration:
            raise ValueError("refocusing pulses do not fit within total_duration")
        if np.any(starts[1:] < stops[:-1]):
            raise ValueError("refocusing pulses overlap; increase total_duration")
    else:
        starts = np.empty(0, dtype=np.float64)
        stops = np.empty(0, dtype=np.float64)
    zero_gradient = tuple(0.0 for _ in tuple(gradient))

    steps: list[MotionSequenceStep] = [
        MotionSequenceStep(
            duration=float(excitation_duration),
            gradient=zero_gradient,
            rf_amplitude=(0.5 * np.pi) / float(excitation_duration),
            rf_phase=float(excitation_phase),
            substeps=max(1, int(substeps_per_interval)),
            label="excitation_90",
        )
    ]
    cursor = 0.0
    for pulse_index, (start, stop) in enumerate(zip(starts, stops), start=1):
        free_duration = float(start - cursor)
        if free_duration > 0.0:
            steps.append(
                MotionSequenceStep(
                    duration=free_duration,
                    gradient=gradient,
                    substeps=int(substeps_per_interval),
                    label=f"udd_{pulse_index}_pre",
                )
            )
        steps.append(
            MotionSequenceStep(
                duration=float(refocusing_duration),
                gradient=zero_gradient,
                rf_amplitude=np.pi / float(refocusing_duration),
                rf_phase=float(refocusing_phase),
                substeps=max(1, int(substeps_per_interval)),
                label=f"udd_{pulse_index}_180",
            )
        )
        cursor = float(stop)

    final_free = float(total_duration - cursor)
    steps.append(
        MotionSequenceStep(
            duration=final_free,
            gradient=gradient,
            acquire=True,
            num_samples=1,
            substeps=int(substeps_per_interval),
            label="udd_echo",
        )
    )
    return tuple(steps)


def run_motion_cpmg_sequence(
    ensemble: ParticleEnsemble,
    fields: MotionFields,
    *,
    num_echoes: int,
    echo_spacing: float,
    excitation_duration: float,
    refocusing_duration: float,
    gradient: tuple[float, float] = (0.0, 0.0),
    velocity: Velocity = None,
    rng: np.random.Generator | None = None,
    t1: float | Iterable[float] | np.ndarray = np.inf,
    t2: float | Iterable[float] | np.ndarray = np.inf,
    mth: float | Iterable[float] | np.ndarray = 1.0,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 1,
    detuning_waveform: DetuningWaveform = None,
) -> MotionSequenceResult:
    """Run a rectangular-pulse CPMG sequence with moving isochromats."""

    steps = make_motion_cpmg_sequence(
        num_echoes,
        echo_spacing,
        excitation_duration=excitation_duration,
        refocusing_duration=refocusing_duration,
        gradient=gradient,
        substeps_per_interval=substeps_per_interval,
    )
    return run_motion_sequence(
        ensemble,
        fields,
        steps,
        velocity=velocity,
        rng=rng,
        t1=t1,
        t2=t2,
        mth=mth,
        boundary=boundary,
        default_substeps=substeps_per_interval,
        detuning_waveform=detuning_waveform,
    )


def run_motion_udd_sequence(
    ensemble: ParticleEnsemble,
    fields: MotionFields,
    *,
    num_pulses: int,
    total_duration: float,
    excitation_duration: float,
    refocusing_duration: float,
    gradient: tuple[float, float] = (0.0, 0.0),
    velocity: Velocity = None,
    rng: np.random.Generator | None = None,
    t1: float | Iterable[float] | np.ndarray = np.inf,
    t2: float | Iterable[float] | np.ndarray = np.inf,
    mth: float | Iterable[float] | np.ndarray = 1.0,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 1,
    detuning_waveform: DetuningWaveform = None,
) -> MotionSequenceResult:
    """Run a rectangular-pulse UDD sequence with moving isochromats."""

    steps = make_motion_udd_sequence(
        num_pulses,
        total_duration,
        excitation_duration=excitation_duration,
        refocusing_duration=refocusing_duration,
        gradient=gradient,
        substeps_per_interval=substeps_per_interval,
    )
    return run_motion_sequence(
        ensemble,
        fields,
        steps,
        velocity=velocity,
        rng=rng,
        t1=t1,
        t2=t2,
        mth=mth,
        boundary=boundary,
        default_substeps=substeps_per_interval,
        detuning_waveform=detuning_waveform,
    )


def _run_step(
    ensemble: ParticleEnsemble,
    fields: MotionFields,
    step: MotionSequenceStep,
    *,
    time: float,
    velocity: Velocity,
    rng: np.random.Generator,
    t1: float | Iterable[float] | np.ndarray,
    t2: float | Iterable[float] | np.ndarray,
    mth: float | Iterable[float] | np.ndarray,
    boundary: Boundary,
    default_substeps: int,
    detuning_waveform: DetuningWaveform,
    label: str,
) -> tuple[ParticleEnsemble, float, list[complex], list[float], list[str]]:
    duration = float(step.duration)
    substeps = int(default_substeps if step.substeps is None else step.substeps)
    sample_count = int(step.num_samples)
    if step.acquire and sample_count == 0:
        sample_count = 1
    breakpoints = _step_breakpoints(duration, substeps, sample_count)
    sample_points: set[float] = set()
    if sample_count > 0:
        sample_points = set(
            np.round(
                np.linspace(
                    duration / sample_count,
                    duration,
                    sample_count,
                    dtype=np.float64,
                ),
                decimals=14,
            )
        )

    current = ensemble
    signals: list[complex] = []
    sample_times: list[float] = []
    sample_labels: list[str] = []
    local_previous = 0.0
    current_time = float(time)
    for local_time in breakpoints:
        dt = float(local_time - local_previous)
        if dt > 0.0:
            current = _propagate_segment(
                current,
                fields,
                dt,
                absolute_time=current_time,
                velocity=velocity,
                rng=rng,
                gradient=step.gradient,
                rf_amplitude=step.rf_amplitude,
                rf_phase=step.rf_phase,
                t1=t1,
                t2=t2,
                mth=mth,
                boundary=boundary,
                detuning_waveform=detuning_waveform,
            )
            current_time += dt
        if sample_count > 0 and np.round(local_time, decimals=14) in sample_points:
            signals.append(receive_signal(current, fields))
            sample_times.append(current_time)
            sample_labels.append(label)
        local_previous = float(local_time)
    return current, current_time, signals, sample_times, sample_labels


def _propagate_segment(
    ensemble: ParticleEnsemble,
    fields: MotionFields,
    dt: float,
    *,
    absolute_time: float,
    velocity: Velocity,
    rng: np.random.Generator,
    gradient: tuple[float, float],
    rf_amplitude: float,
    rf_phase: float,
    t1: float | Iterable[float] | np.ndarray,
    t2: float | Iterable[float] | np.ndarray,
    mth: float | Iterable[float] | np.ndarray,
    boundary: Boundary,
    detuning_waveform: DetuningWaveform,
) -> ParticleEnsemble:
    moved = move_ensemble(
        ensemble,
        dt,
        velocity=velocity,
        rng=rng,
        time=absolute_time,
        bounds=fields.bounds,
        boundary=boundary,
    )
    sampled = fields.sample(moved.positions)
    grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if grad.size != moved.positions.shape[1]:
        raise ValueError("step gradient length must match the spatial dimension")
    detuning = _detuning_values(
        detuning_waveform,
        absolute_time + 0.5 * dt,
        moved.positions,
    )
    off_resonance = sampled["b0"] + moved.positions @ grad + detuning
    if rf_amplitude == 0.0:
        return apply_free_precession(
            moved,
            dt,
            off_resonance,
            t1=t1,
            t2=t2,
            mth=mth,
        )
    return apply_rf_rotation(
        moved,
        dt,
        float(rf_phase),
        float(rf_amplitude),
        off_resonance,
        b1_tx=sampled["b1_tx"],
    )


def _step_breakpoints(
    duration: float,
    substeps: int,
    sample_count: int,
) -> np.ndarray:
    if substeps <= 0:
        raise ValueError("substeps must be positive")
    points = [0.0, float(duration)]
    if duration > 0.0:
        points.extend(np.linspace(0.0, duration, int(substeps) + 1)[1:])
        if sample_count > 0:
            points.extend(
                np.linspace(
                    duration / int(sample_count),
                    duration,
                    int(sample_count),
                    dtype=np.float64,
                )
            )
    return np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=14))[1:]


def _validate_step(step: MotionSequenceStep) -> None:
    if step.duration < 0.0:
        raise ValueError("step duration must be non-negative")
    if step.substeps is not None and step.substeps <= 0:
        raise ValueError("step substeps must be positive")
    if step.num_samples < 0:
        raise ValueError("step num_samples must be non-negative")
    gradient = np.asarray(step.gradient, dtype=np.float64)
    if gradient.ndim != 1 or gradient.size < 1 or not np.all(np.isfinite(gradient)):
        raise ValueError("step gradient must be a 1-D vector of finite values")
    if not np.isfinite(step.rf_amplitude) or not np.isfinite(step.rf_phase):
        raise ValueError("RF amplitude and phase must be finite")


def _detuning_values(
    detuning_waveform: DetuningWaveform,
    time: float,
    positions: np.ndarray,
) -> float | np.ndarray:
    if detuning_waveform is None:
        return 0.0
    if callable(detuning_waveform):
        try:
            value = detuning_waveform(float(time), positions)
        except TypeError:
            value = detuning_waveform(float(time))
    else:
        value = detuning_waveform
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        detuning = float(arr)
    else:
        detuning = arr.reshape(-1)
        if detuning.size != positions.shape[0]:
            raise ValueError(
                "detuning_waveform must return a scalar or one value per particle"
            )
    if not np.all(np.isfinite(detuning)):
        raise ValueError("detuning_waveform must return finite values")
    return detuning
