"""Acquisition masks for pulsed NQR RFI-cancellation workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spin_dynamics.interference.masks import AcquisitionMask, Interval, mask_from_intervals
from spin_dynamics.nqr.sequences import SLSESequence, SORCSequence


def _nonnegative_seconds(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def _intervals_with_baselines(
    *,
    pre_baseline_seconds: float,
    sequence_end_seconds: float,
    post_baseline_seconds: float,
    baseline_intervals: Sequence[Interval] | None,
) -> list[Interval]:
    intervals: list[Interval] = []
    if pre_baseline_seconds > 0.0:
        intervals.append((0.0, pre_baseline_seconds))
    if post_baseline_seconds > 0.0:
        intervals.append((sequence_end_seconds, sequence_end_seconds + post_baseline_seconds))
    if baseline_intervals is not None:
        intervals.extend(tuple(pair) for pair in baseline_intervals)
    return intervals


def slse_acquisition_mask(
    sequence: SLSESequence,
    sample_rate_hz: float,
    *,
    ringdown_seconds: float = 0.0,
    pre_baseline_seconds: float = 0.0,
    post_baseline_seconds: float = 0.0,
    baseline_intervals: Sequence[Interval] | None = None,
) -> AcquisitionMask:
    """Return a transmit/ringdown/signal/baseline mask for an SLSE train.

    The shot clock starts with an optional pre-shot baseline. SLSE transmit
    pulses then begin every ``echo_spacing_seconds``; all non-blanked sequence
    samples are labelled as signal gaps, and optional baseline windows are
    interpreted as absolute intervals on the returned shot clock.
    """

    ringdown = _nonnegative_seconds(ringdown_seconds, "ringdown_seconds")
    pre = _nonnegative_seconds(pre_baseline_seconds, "pre_baseline_seconds")
    post = _nonnegative_seconds(post_baseline_seconds, "post_baseline_seconds")
    pulse_duration = sequence.detection.duration_seconds
    spacing = sequence.echo_spacing_seconds
    if pulse_duration > spacing:
        raise ValueError("SLSE pulse duration must not exceed echo_spacing_seconds")

    sequence_start = pre
    sequence_duration = sequence.num_echoes * spacing
    sequence_end = sequence_start + sequence_duration
    duration = sequence_end + post
    transmit = [
        (sequence_start + idx * spacing, sequence_start + idx * spacing + pulse_duration)
        for idx in range(sequence.num_echoes)
    ]
    ringdown_intervals = [
        (stop, min(stop + ringdown, sequence_start + (idx + 1) * spacing))
        for idx, (_, stop) in enumerate(transmit)
        if ringdown > 0.0
    ]
    baseline = _intervals_with_baselines(
        pre_baseline_seconds=pre,
        sequence_end_seconds=sequence_end,
        post_baseline_seconds=post,
        baseline_intervals=baseline_intervals,
    )
    return mask_from_intervals(
        sample_rate_hz,
        duration,
        transmit=transmit,
        ringdown=ringdown_intervals,
        baseline=baseline,
    )


def sorc_acquisition_mask(
    sequence: SORCSequence,
    sample_rate_hz: float,
    *,
    ringdown_seconds: float = 0.0,
    pre_baseline_seconds: float = 0.0,
    post_baseline_seconds: float = 0.0,
    baseline_intervals: Sequence[Interval] | None = None,
    initial_gap_is_baseline: bool = True,
) -> AcquisitionMask:
    """Return a transmit/ringdown/signal/baseline mask for a SORC train.

    The timing mirrors ``simulate_sorc``: each cycle is
    ``tau - pulse - tau`` and has duration
    ``2 * half_spacing_seconds + pulse_duration_seconds``. The first pre-pulse
    ``tau`` interval is labelled as baseline by default because no SORC response
    has yet been generated.
    """

    ringdown = _nonnegative_seconds(ringdown_seconds, "ringdown_seconds")
    pre = _nonnegative_seconds(pre_baseline_seconds, "pre_baseline_seconds")
    post = _nonnegative_seconds(post_baseline_seconds, "post_baseline_seconds")
    tau = sequence.half_spacing_seconds
    pulse_duration = sequence.detection.duration_seconds
    cycle_duration = 2.0 * tau + pulse_duration
    if cycle_duration <= 0.0:
        raise ValueError("SORC cycle duration must be positive")

    sequence_start = pre
    sequence_end = sequence_start + sequence.num_pulses * cycle_duration
    duration = sequence_end + post
    transmit = []
    ringdown_intervals = []
    for idx in range(sequence.num_pulses):
        cycle_start = sequence_start + idx * cycle_duration
        pulse_start = cycle_start + tau
        pulse_stop = pulse_start + pulse_duration
        transmit.append((pulse_start, pulse_stop))
        if ringdown > 0.0:
            cycle_stop = cycle_start + cycle_duration
            ringdown_intervals.append((pulse_stop, min(pulse_stop + ringdown, cycle_stop)))

    baseline = _intervals_with_baselines(
        pre_baseline_seconds=pre,
        sequence_end_seconds=sequence_end,
        post_baseline_seconds=post,
        baseline_intervals=baseline_intervals,
    )
    if initial_gap_is_baseline and tau > 0.0:
        baseline.append((sequence_start, sequence_start + tau))

    return mask_from_intervals(
        sample_rate_hz,
        duration,
        transmit=transmit,
        ringdown=ringdown_intervals,
        baseline=baseline,
    )
