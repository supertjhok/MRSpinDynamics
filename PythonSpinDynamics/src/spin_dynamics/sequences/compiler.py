"""Compile sequence IR blocks into piecewise-constant backend inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.sequences.ir import (
    GradientWaveform,
    HardwareEffectsPolicy,
    RFPulse,
    SequenceIR,
)


@dataclass(frozen=True)
class CompiledADC:
    """Receive samples on the absolute sequence timeline."""

    times_seconds: np.ndarray
    frequency_offsets_hz: np.ndarray
    phase_offsets_rad: np.ndarray
    block_indices: np.ndarray


@dataclass(frozen=True)
class CompiledSequence:
    """Piecewise-constant RF/gradient timeline plus exact ADC sample times."""

    start_times_seconds: np.ndarray
    durations_seconds: np.ndarray
    rf_hz: np.ndarray
    gradients_hz_per_m: np.ndarray
    block_indices: np.ndarray
    block_start_times_seconds: np.ndarray
    block_durations_seconds: np.ndarray
    block_labels: tuple[str, ...]
    adc: CompiledADC
    source_format: str
    source_version: tuple[int, int, int] | None
    hardware_effects: HardwareEffectsPolicy

    @property
    def duration_seconds(self) -> float:
        if self.durations_seconds.size == 0:
            return 0.0
        return float(self.start_times_seconds[-1] + self.durations_seconds[-1])

    def plot(self, **kwargs):
        """Plot aligned RF, gradient, and ADC lanes."""

        from spin_dynamics.sequences.plotting import plot_sequence

        return plot_sequence(self, **kwargs)


def compile_sequence(
    sequence: SequenceIR,
    *,
    system_frequency_hz: float | None = None,
) -> CompiledSequence:
    """Compile an IR into piecewise-constant intervals.

    Event raster edges and ADC sample centers become interval boundaries.  RF
    and gradient values are sampled at each interval midpoint, matching
    Pulseq's default center-of-raster sampling convention.
    """

    starts: list[float] = []
    durations: list[float] = []
    rf_values: list[complex] = []
    gradient_values: list[tuple[float, float, float]] = []
    block_indices: list[int] = []
    adc_times: list[float] = []
    adc_frequencies: list[float] = []
    adc_phases: list[float] = []
    adc_blocks: list[int] = []
    block_starts: list[float] = []
    block_durations: list[float] = []
    block_labels: list[str] = []

    block_start = 0.0
    for block_index, block in enumerate(sequence.blocks):
        block_starts.append(block_start)
        block_durations.append(float(block.duration_seconds))
        block_labels.append(block.label or f"block_{block_index + 1}")
        boundaries = [0.0, float(block.duration_seconds)]
        if block.rf is not None:
            boundaries.extend(_event_edges(block.rf))
        for gradient in block.gradients:
            if gradient is not None:
                boundaries.extend(_event_edges(gradient))
        if block.adc is not None:
            relative_adc = block.adc.sample_times_seconds()
            boundaries.extend(relative_adc.tolist())
            phase_shape = block.adc.phase_offsets_rad
            for sample_index, relative_time in enumerate(relative_adc):
                adc_times.append(block_start + float(relative_time))
                adc_frequencies.append(
                    _combined_frequency_offset(block.adc, system_frequency_hz)
                )
                phase = _combined_phase_offset(block.adc, system_frequency_hz)
                phase += 2.0 * np.pi * adc_frequencies[-1] * float(relative_time)
                if phase_shape is not None:
                    phase += float(phase_shape[sample_index])
                adc_phases.append(phase)
                adc_blocks.append(block_index)

        edges = np.unique(np.round(boundaries, decimals=15))
        for left, right in zip(edges[:-1], edges[1:]):
            duration = float(right - left)
            if duration <= 0.0:
                continue
            midpoint = 0.5 * float(left + right)
            starts.append(block_start + float(left))
            durations.append(duration)
            rf_values.append(_sample_rf(block.rf, midpoint, system_frequency_hz))
            gradient_values.append(
                tuple(_sample_gradient(event, midpoint) for event in block.gradients)
            )
            block_indices.append(block_index)
        block_start += float(block.duration_seconds)

    return CompiledSequence(
        start_times_seconds=np.asarray(starts, dtype=np.float64),
        durations_seconds=np.asarray(durations, dtype=np.float64),
        rf_hz=np.asarray(rf_values, dtype=np.complex128),
        gradients_hz_per_m=np.asarray(gradient_values, dtype=np.float64).reshape(-1, 3),
        block_indices=np.asarray(block_indices, dtype=np.int64),
        block_start_times_seconds=np.asarray(block_starts, dtype=np.float64),
        block_durations_seconds=np.asarray(block_durations, dtype=np.float64),
        block_labels=tuple(block_labels),
        adc=CompiledADC(
            times_seconds=np.asarray(adc_times, dtype=np.float64),
            frequency_offsets_hz=np.asarray(adc_frequencies, dtype=np.float64),
            phase_offsets_rad=np.asarray(adc_phases, dtype=np.float64),
            block_indices=np.asarray(adc_blocks, dtype=np.int64),
        ),
        source_format=sequence.source_format,
        source_version=sequence.source_version,
        hardware_effects=sequence.hardware_effects,
    )


def compiled_to_motion_steps(
    compiled: CompiledSequence,
    *,
    spatial_dimensions: int = 2,
    gradient_axes: tuple[int, ...] | None = None,
):
    """Adapt compiled intervals to the existing moving-isochromat engine.

    Pulseq frequencies are converted from cycles/s to angular units.  The
    motion engine currently accepts one receive sample at an interval end, so
    the compiler's ADC-centered boundaries map directly onto those samples.
    """

    policy = compiled.hardware_effects
    requested = [
        name for name in ("transmit", "receive") if getattr(policy, name) == "apply"
    ]
    if requested:
        raise NotImplementedError(
            "the motion adapter cannot yet apply probe hardware effects for "
            + " and ".join(requested)
            + "; compile with HardwareEffectsPolicy(...='ignore') or use a "
            "probe-aware backend"
        )

    from spin_dynamics.sequences.motion import MotionSequenceStep

    if spatial_dimensions not in (1, 2, 3):
        raise ValueError("spatial_dimensions must be 1, 2, or 3")
    axes = tuple(range(spatial_dimensions)) if gradient_axes is None else gradient_axes
    if len(axes) != spatial_dimensions or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("gradient_axes must select one physical channel per dimension")
    if len(set(axes)) != len(axes):
        raise ValueError("gradient_axes must not contain duplicates")
    adc_counts = _adc_counts_at_interval_ends(compiled)
    steps = []
    for index, duration in enumerate(compiled.durations_seconds):
        rf = complex(compiled.rf_hz[index])
        count = int(adc_counts[index])
        block_label = compiled.block_labels[int(compiled.block_indices[index])]
        steps.append(
            MotionSequenceStep(
                duration=float(duration),
                gradient=tuple(
                    2.0
                    * np.pi
                    * compiled.gradients_hz_per_m[index, list(axes)]
                ),
                rf_amplitude=2.0 * np.pi * abs(rf),
                rf_phase=float(np.angle(rf)) if rf != 0.0 else 0.0,
                acquire=count > 0,
                num_samples=count,
                label=f"{block_label}:{index}",
            )
        )
    return tuple(steps)


def _event_edges(event: RFPulse | GradientWaveform) -> list[float]:
    start = float(event.delay_seconds)
    dwell = float(event.dwell_seconds)
    return (start + dwell * np.arange(event.samples_hz.size + 1)).tolist() if isinstance(
        event, RFPulse
    ) else (start + dwell * np.arange(event.samples_hz_per_m.size + 1)).tolist()


def _sample_rf(
    event: RFPulse | None,
    time_seconds: float,
    system_frequency_hz: float | None,
) -> complex:
    if event is None:
        return 0.0j
    index = _sample_index(
        time_seconds,
        event.delay_seconds,
        event.dwell_seconds,
        event.samples_hz.size,
    )
    if index is None:
        return 0.0j
    local_time = time_seconds - float(event.delay_seconds)
    offset_phase = _combined_phase_offset(event, system_frequency_hz) + 2.0 * np.pi * (
        _combined_frequency_offset(event, system_frequency_hz) * local_time
    )
    return complex(event.samples_hz[index] * np.exp(1j * offset_phase))


def _sample_gradient(event: GradientWaveform | None, time_seconds: float) -> float:
    if event is None:
        return 0.0
    index = _sample_index(
        time_seconds,
        event.delay_seconds,
        event.dwell_seconds,
        event.samples_hz_per_m.size,
    )
    return 0.0 if index is None else float(event.samples_hz_per_m[index])


def _sample_index(
    time_seconds: float,
    delay_seconds: float,
    dwell_seconds: float,
    size: int,
) -> int | None:
    local = float(time_seconds) - float(delay_seconds)
    if local < 0.0 or local >= float(size) * float(dwell_seconds):
        return None
    return min(int(np.floor(local / float(dwell_seconds))), size - 1)


def _adc_counts_at_interval_ends(compiled: CompiledSequence) -> np.ndarray:
    ends = compiled.start_times_seconds + compiled.durations_seconds
    counts = np.zeros(ends.size, dtype=np.int64)
    for sample_time in compiled.adc.times_seconds:
        matches = np.flatnonzero(np.isclose(ends, sample_time, rtol=0.0, atol=1e-13))
        if matches.size == 0:
            raise ValueError("ADC sample does not align with a compiled interval boundary")
        counts[int(matches[0])] += 1
    return counts


def _combined_frequency_offset(event, system_frequency_hz: float | None) -> float:
    ppm = float(event.frequency_offset_ppm)
    if ppm != 0.0 and system_frequency_hz is None:
        raise ValueError("system_frequency_hz is required for a nonzero PPM offset")
    system_mhz = 0.0 if system_frequency_hz is None else system_frequency_hz / 1e6
    return float(event.frequency_offset_hz) + ppm * system_mhz


def _combined_phase_offset(event, system_frequency_hz: float | None) -> float:
    per_mhz = float(event.phase_offset_rad_per_mhz)
    if per_mhz != 0.0 and system_frequency_hz is None:
        raise ValueError("system_frequency_hz is required for a frequency-scaled phase")
    system_mhz = 0.0 if system_frequency_hz is None else system_frequency_hz / 1e6
    return float(event.phase_offset_rad) + per_mhz * system_mhz


__all__ = [
    "CompiledADC",
    "CompiledSequence",
    "compile_sequence",
    "compiled_to_motion_steps",
]
