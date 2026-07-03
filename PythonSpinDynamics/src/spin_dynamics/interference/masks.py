"""Sample-clock and receive-valid masks for gated RFI cancellation.

In pulsed NQR (SLSE, SORC, ...) the primary receive channel is blanked during
transmit pulses and the subsequent ringdown, and only carries useful NQR signal
in the gaps. Reference detectors, by contrast, run continuously on the same shot
clock. The cancellation problem is therefore a *masked* estimation problem: a
canceller is trained on samples believed to contain no NQR response (baselines,
late-time tails, off-resonance shots) and applied on the signal gaps.

``AcquisitionMask`` labels every sample on a single shot clock as exactly one of

* ``TRANSMIT``  -- RF pulse on; primary receiver blanked,
* ``RINGDOWN``  -- probe/preamp recovery after a pulse; primary unusable,
* ``SIGNAL``    -- valid receive gap where NQR response is expected (set G),
* ``BASELINE``  -- valid receive but no NQR expected; canceller training (set B).

The two derived boolean views the cancellers consume are :attr:`signal_mask`
(set G, where the cleaned estimate is reported) and :attr:`baseline_mask`
(set B, where a fixed canceller is fit). :attr:`receive_mask` is the union of the
two -- every sample where the primary receiver is open.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from spin_dynamics.interference._signal import sample_times


class SampleLabel(IntEnum):
    """Per-sample acquisition state on the shot clock."""

    TRANSMIT = 0
    RINGDOWN = 1
    SIGNAL = 2
    BASELINE = 3


# Interval lists resolve onto samples in this priority order (highest first), so
# an overlapping transmit interval always wins over a ringdown or baseline one.
_INTERVAL_PRIORITY = (
    SampleLabel.TRANSMIT,
    SampleLabel.RINGDOWN,
    SampleLabel.BASELINE,
)

Interval = tuple[float, float]


@dataclass(frozen=True)
class AcquisitionMask:
    """Per-sample receive-state labels sharing one shot clock."""

    sample_rate_hz: float
    labels: np.ndarray

    def __post_init__(self) -> None:
        sample_rate_hz = float(self.sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if labels.size == 0:
            raise ValueError("labels must contain at least one sample")
        valid = {int(item) for item in SampleLabel}
        if not set(np.unique(labels)).issubset(valid):
            raise ValueError("labels must be SampleLabel values")
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "labels", labels)

    @property
    def num_samples(self) -> int:
        """Number of samples on the shot clock."""

        return int(self.labels.size)

    @property
    def duration_seconds(self) -> float:
        """Total record duration in seconds."""

        return self.num_samples / self.sample_rate_hz

    @property
    def times(self) -> np.ndarray:
        """Sample time grid in seconds."""

        return sample_times(self.num_samples, self.sample_rate_hz)

    @property
    def transmit_mask(self) -> np.ndarray:
        """Boolean mask of transmit-pulse samples."""

        return self.labels == SampleLabel.TRANSMIT

    @property
    def ringdown_mask(self) -> np.ndarray:
        """Boolean mask of ringdown/recovery samples."""

        return self.labels == SampleLabel.RINGDOWN

    @property
    def signal_mask(self) -> np.ndarray:
        """Boolean mask of valid NQR-signal gaps (set G)."""

        return self.labels == SampleLabel.SIGNAL

    @property
    def baseline_mask(self) -> np.ndarray:
        """Boolean mask of baseline / no-NQR windows (set B)."""

        return self.labels == SampleLabel.BASELINE

    @property
    def receive_mask(self) -> np.ndarray:
        """Boolean mask of every sample where the receiver is open."""

        return self.signal_mask | self.baseline_mask

    @property
    def masked_fraction(self) -> float:
        """Fraction of samples blanked by transmit or ringdown."""

        return float(np.mean(self.transmit_mask | self.ringdown_mask))


def blank_mask(
    sample_rate_hz: float,
    num_samples: int,
    *,
    fill: SampleLabel = SampleLabel.SIGNAL,
) -> AcquisitionMask:
    """Return a mask with every sample set to ``fill``."""

    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    labels = np.full(num_samples, int(fill), dtype=np.int64)
    return AcquisitionMask(sample_rate_hz=sample_rate_hz, labels=labels)


def _apply_intervals(
    labels: np.ndarray,
    sample_rate_hz: float,
    intervals: Iterable[Interval] | None,
    label: SampleLabel,
) -> None:
    if intervals is None:
        return
    n = labels.size
    for start, stop in intervals:
        start = float(start)
        stop = float(stop)
        if not (np.isfinite(start) and np.isfinite(stop)):
            raise ValueError("interval bounds must be finite")
        if stop < start:
            raise ValueError("interval stop must be >= start")
        # Half-open [start, stop): round to the nearest sample edges.
        i0 = int(np.ceil(start * sample_rate_hz - 1e-9))
        i1 = int(np.ceil(stop * sample_rate_hz - 1e-9))
        i0 = max(0, min(n, i0))
        i1 = max(0, min(n, i1))
        if i1 > i0:
            labels[i0:i1] = int(label)


def mask_from_intervals(
    sample_rate_hz: float,
    duration_seconds: float,
    *,
    transmit: Sequence[Interval] | None = None,
    ringdown: Sequence[Interval] | None = None,
    baseline: Sequence[Interval] | None = None,
    fill: SampleLabel = SampleLabel.SIGNAL,
) -> AcquisitionMask:
    """Build a mask from second-valued intervals on a shot clock.

    Every sample defaults to ``fill`` (``SIGNAL``); ``transmit``, ``ringdown``,
    and ``baseline`` intervals then overwrite it. Overlaps resolve by priority
    (transmit > ringdown > baseline), so a transmit interval always wins.
    Intervals are treated as half-open ``[start, stop)`` in seconds.
    """

    sample_rate_hz = float(sample_rate_hz)
    duration_seconds = float(duration_seconds)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive and finite")
    num_samples = int(round(duration_seconds * sample_rate_hz))
    if num_samples <= 0:
        raise ValueError("duration_seconds is too short for the sample rate")

    labels = np.full(num_samples, int(fill), dtype=np.int64)
    supplied = {
        SampleLabel.TRANSMIT: transmit,
        SampleLabel.RINGDOWN: ringdown,
        SampleLabel.BASELINE: baseline,
    }
    # Apply low-to-high priority so higher-priority labels overwrite lower ones.
    for label in reversed(_INTERVAL_PRIORITY):
        _apply_intervals(labels, sample_rate_hz, supplied[label], label)
    return AcquisitionMask(sample_rate_hz=sample_rate_hz, labels=labels)
