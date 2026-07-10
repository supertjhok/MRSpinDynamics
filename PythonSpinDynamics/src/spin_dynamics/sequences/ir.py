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

    @property
    def duration_seconds(self) -> float:
        return float(self.samples_hz.size) * float(self.dwell_seconds)

    @property
    def end_seconds(self) -> float:
        return float(self.delay_seconds) + self.duration_seconds


@dataclass(frozen=True)
class GradientWaveform:
    """Sampled gradient waveform in hertz per meter."""

    samples_hz_per_m: np.ndarray
    dwell_seconds: float
    delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "samples_hz_per_m",
            _finite_1d(self.samples_hz_per_m, "samples_hz_per_m"),
        )
        _validate_timing(self.dwell_seconds, self.delay_seconds)

    @property
    def duration_seconds(self) -> float:
        return float(self.samples_hz_per_m.size) * float(self.dwell_seconds)

    @property
    def end_seconds(self) -> float:
        return float(self.delay_seconds) + self.duration_seconds


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

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a sequence must contain at least one block")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "definitions", dict(self.definitions))

    @property
    def duration_seconds(self) -> float:
        return float(sum(block.duration_seconds for block in self.blocks))


def _validate_timing(dwell_seconds: float, delay_seconds: float) -> None:
    if not np.isfinite(dwell_seconds) or dwell_seconds <= 0.0:
        raise ValueError("dwell_seconds must be finite and positive")
    if not np.isfinite(delay_seconds) or delay_seconds < 0.0:
        raise ValueError("delay_seconds must be finite and non-negative")


__all__ = [
    "ADCEvent",
    "GradientWaveform",
    "RFPulse",
    "SequenceBlock",
    "SequenceIR",
]
