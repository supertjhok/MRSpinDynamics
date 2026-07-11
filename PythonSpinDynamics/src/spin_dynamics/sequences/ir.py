"""Backend-neutral pulse-sequence intermediate representation.

The IR follows Pulseq's block model: blocks execute sequentially, while RF,
three physical gradient channels, and ADC events may overlap inside a block.
RF amplitudes are stored in hertz and gradients in hertz per meter.  Backends
own conversions to angular-frequency or field units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


HARDWARE_EFFECT_MODES = ("ignore", "apply")


@dataclass(frozen=True)
class HardwareEffectsPolicy:
    """Declare whether execution must realize transmit and receive hardware.

    The sequence waveforms remain nominal in either mode. ``"ignore"`` asks a
    backend to use the nominal RF waveform or ideal receive signal directly;
    ``"apply"`` requires it to pass transmit RF or acquired samples through a
    separately supplied probe/hardware model. Keeping the policy separate from
    event samples prevents double-filtering and lets one IR support ideal and
    probe-aware comparisons.

    Compilation preserves this policy but does not itself implement a hardware
    transfer function. A backend must reject ``"apply"`` when it cannot honor
    the requested path rather than silently falling back to ideal behavior.
    """

    transmit: str = "ignore"
    receive: str = "ignore"

    def __post_init__(self) -> None:
        for name in ("transmit", "receive"):
            value = getattr(self, name)
            if value not in HARDWARE_EFFECT_MODES:
                raise ValueError(
                    f"{name} must be one of {HARDWARE_EFFECT_MODES}, got {value!r}"
                )


def _finite_1d(values: Any, name: str, *, complex_values: bool = False) -> np.ndarray:
    dtype = np.complex128 if complex_values else np.float64
    array = np.asarray(values, dtype=dtype)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


@dataclass(frozen=True)
class RFPulse:
    """Sampled complex RF envelope in hertz.

    Samples are centered in consecutive ``dwell_seconds`` raster cells.  The
    frequency and phase offsets are kept separate so importers do not discard
    source-format intent.
    """

    samples_hz: np.ndarray
    dwell_seconds: float
    delay_seconds: float = 0.0
    frequency_offset_hz: float = 0.0
    frequency_offset_ppm: float = 0.0
    phase_offset_rad: float = 0.0
    phase_offset_rad_per_mhz: float = 0.0
    use: str = "undefined"
    center_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples_hz",
            _finite_1d(self.samples_hz, "samples_hz", complex_values=True),
        )
        _validate_timing(self.dwell_seconds, self.delay_seconds)
        if not np.isfinite(self.frequency_offset_hz):
            raise ValueError("frequency_offset_hz must be finite")
        if not np.isfinite(self.frequency_offset_ppm):
            raise ValueError("frequency_offset_ppm must be finite")
        if not np.isfinite(self.phase_offset_rad):
            raise ValueError("phase_offset_rad must be finite")
        if not np.isfinite(self.phase_offset_rad_per_mhz):
            raise ValueError("phase_offset_rad_per_mhz must be finite")
        if self.center_seconds is not None and (
            not np.isfinite(self.center_seconds)
            or self.center_seconds < 0.0
            or self.center_seconds > self.duration_seconds
        ):
            raise ValueError("center_seconds must lie within the RF waveform")

    @property
    def duration_seconds(self) -> float:
        return float(self.samples_hz.size) * float(self.dwell_seconds)

    @property
    def end_seconds(self) -> float:
        return float(self.delay_seconds) + self.duration_seconds

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RFPulse):
            return NotImplemented
        return _dataclass_values_equal(self, other)


@dataclass(frozen=True)
class GradientWaveform:
    """Sampled gradient waveform in hertz per meter."""

    samples_hz_per_m: np.ndarray
    dwell_seconds: float
    delay_seconds: float = 0.0
    first_hz_per_m: float = 0.0
    last_hz_per_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples_hz_per_m",
            _finite_1d(self.samples_hz_per_m, "samples_hz_per_m"),
        )
        _validate_timing(self.dwell_seconds, self.delay_seconds)
        if not np.isfinite(self.first_hz_per_m) or not np.isfinite(self.last_hz_per_m):
            raise ValueError("gradient boundary amplitudes must be finite")

    @property
    def duration_seconds(self) -> float:
        return float(self.samples_hz_per_m.size) * float(self.dwell_seconds)

    @property
    def end_seconds(self) -> float:
        return float(self.delay_seconds) + self.duration_seconds

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GradientWaveform):
            return NotImplemented
        return _dataclass_values_equal(self, other)


@dataclass(frozen=True)
class ADCEvent:
    """Uniform receive-sampling event."""

    num_samples: int
    dwell_seconds: float
    delay_seconds: float = 0.0
    frequency_offset_hz: float = 0.0
    frequency_offset_ppm: float = 0.0
    phase_offset_rad: float = 0.0
    phase_offset_rad_per_mhz: float = 0.0
    phase_offsets_rad: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        _validate_timing(self.dwell_seconds, self.delay_seconds)
        if not np.isfinite(self.frequency_offset_hz):
            raise ValueError("frequency_offset_hz must be finite")
        if not np.isfinite(self.frequency_offset_ppm):
            raise ValueError("frequency_offset_ppm must be finite")
        if not np.isfinite(self.phase_offset_rad):
            raise ValueError("phase_offset_rad must be finite")
        if not np.isfinite(self.phase_offset_rad_per_mhz):
            raise ValueError("phase_offset_rad_per_mhz must be finite")
        if self.phase_offsets_rad is not None:
            offsets = _finite_1d(self.phase_offsets_rad, "phase_offsets_rad")
            if offsets.size != self.num_samples:
                raise ValueError("phase_offsets_rad must have num_samples entries")
            object.__setattr__(self, "phase_offsets_rad", offsets)

    @property
    def duration_seconds(self) -> float:
        return float(self.num_samples) * float(self.dwell_seconds)

    @property
    def end_seconds(self) -> float:
        return float(self.delay_seconds) + self.duration_seconds

    def sample_times_seconds(self) -> np.ndarray:
        """Return sample centers relative to the containing block."""

        return float(self.delay_seconds) + float(self.dwell_seconds) * (
            np.arange(self.num_samples, dtype=np.float64) + 0.5
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ADCEvent):
            return NotImplemented
        return _dataclass_values_equal(self, other)


@dataclass(frozen=True)
class SequenceBlock:
    """Concurrent events with an explicit duration."""

    duration_seconds: float
    rf: RFPulse | None = None
    gradients: tuple[
        GradientWaveform | None,
        GradientWaveform | None,
        GradientWaveform | None,
    ] = (None, None, None)
    adc: ADCEvent | None = None
    label: str = ""
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.duration_seconds) or self.duration_seconds < 0.0:
            raise ValueError("duration_seconds must be finite and non-negative")
        if len(self.gradients) != 3:
            raise ValueError("gradients must contain the x, y, and z channels")
        object.__setattr__(self, "gradients", tuple(self.gradients))
        object.__setattr__(self, "extensions", tuple(self.extensions))
        events = [self.rf, *self.gradients, self.adc]
        for event in events:
            if event is not None and event.end_seconds > self.duration_seconds + 1e-12:
                raise ValueError("an event extends beyond the containing block")


@dataclass(frozen=True)
class SequenceIR:
    """A complete backend-neutral pulse sequence."""

    blocks: tuple[SequenceBlock, ...]
    definitions: Mapping[str, Any] = field(default_factory=dict)
    source_format: str = "native"
    source_version: tuple[int, int, int] | None = None
    hardware_effects: HardwareEffectsPolicy = field(
        default_factory=HardwareEffectsPolicy
    )

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a sequence must contain at least one block")
        if not isinstance(self.hardware_effects, HardwareEffectsPolicy):
            raise TypeError("hardware_effects must be a HardwareEffectsPolicy")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "definitions", dict(self.definitions))

    @property
    def duration_seconds(self) -> float:
        return float(sum(block.duration_seconds for block in self.blocks))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceIR):
            return NotImplemented
        return _dataclass_values_equal(self, other)

    def plot(self, **kwargs):
        """Plot aligned RF, gradient, and ADC lanes.

        Matplotlib is imported lazily. Keyword arguments are forwarded to
        :func:`spin_dynamics.sequences.plotting.plot_sequence`.
        """

        from spin_dynamics.sequences.plotting import plot_sequence

        return plot_sequence(self, **kwargs)

    def phase_cycle(self, cycle, *, pulse_blocks=None):
        """Return independently executable branches for an arbitrary phase cycle."""

        from spin_dynamics.phase_cycling import phase_cycle_sequence_ir

        return phase_cycle_sequence_ir(self, cycle, pulse_blocks=pulse_blocks)


def _validate_timing(dwell_seconds: float, delay_seconds: float) -> None:
    if not np.isfinite(dwell_seconds) or dwell_seconds <= 0.0:
        raise ValueError("dwell_seconds must be finite and positive")
    if not np.isfinite(delay_seconds) or delay_seconds < 0.0:
        raise ValueError("delay_seconds must be finite and non-negative")


def _dataclass_values_equal(left: Any, right: Any) -> bool:
    """Compare frozen IR records without NumPy's ambiguous array truth value."""

    for name in left.__dataclass_fields__:
        left_value = getattr(left, name)
        right_value = getattr(right, name)
        if isinstance(left_value, np.ndarray) or isinstance(right_value, np.ndarray):
            if not np.array_equal(left_value, right_value):
                return False
        elif isinstance(left_value, Mapping) and isinstance(right_value, Mapping):
            if left_value.keys() != right_value.keys() or any(
                not _values_equal(left_value[key], right_value[key])
                for key in left_value
            ):
                return False
        elif not _values_equal(left_value, right_value):
            return False
    return True


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return bool(np.array_equal(left, right))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    return bool(left == right)


__all__ = [
    "ADCEvent",
    "GradientWaveform",
    "HardwareEffectsPolicy",
    "RFPulse",
    "SequenceBlock",
    "SequenceIR",
]
