"""Acquisition masks for pulsed NQR RFI-cancellation workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spin_dynamics.interference.masks import AcquisitionMask, Interval, mask_from_intervals
from spin_dynamics.interference.recordings import RFIRecording
from spin_dynamics.nqr.sequences import SLSESequence, SORCSequence


def _nonnegative_seconds(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return out


def _positive_seconds(value: float, name: str) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return out


def _window_duration(num_samples: int, sample_rate_hz: float) -> tuple[int, float, float]:
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    sample_rate_hz = float(sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    return num_samples, sample_rate_hz, num_samples / sample_rate_hz


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


def slse_mask_from_metadata(
    num_samples: int,
    sample_rate_hz: float,
    *,
    echo_spacing_seconds: float,
    detection_duration_seconds: float,
    num_echoes: int,
    ringdown_seconds: float = 0.0,
    start_offset_seconds: float = 0.0,
    post_baseline_seconds: float = 0.0,
    baseline_intervals: Sequence[Interval] | None = None,
) -> AcquisitionMask:
    """Reconstruct an SLSE acquisition mask over a window of ``num_samples`` ADC samples.

    Experimental records store only the sample window; the gating is rebuilt from
    the sequence timing. This mirrors :func:`slse_acquisition_mask` but is driven
    by the recorded sample count (the window duration is ``num_samples /
    sample_rate_hz``) and raw timing values rather than a sequence object.
    ``start_offset_seconds`` is the time of the first transmit pulse within the
    window; the samples before it are labelled baseline.
    """

    num_samples, sample_rate_hz, duration = _window_duration(num_samples, sample_rate_hz)
    spacing = _positive_seconds(echo_spacing_seconds, "echo_spacing_seconds")
    pulse_duration = _positive_seconds(detection_duration_seconds, "detection_duration_seconds")
    if pulse_duration > spacing:
        raise ValueError("detection_duration_seconds must not exceed echo_spacing_seconds")
    num_echoes = int(num_echoes)
    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    ringdown = _nonnegative_seconds(ringdown_seconds, "ringdown_seconds")
    start = _nonnegative_seconds(start_offset_seconds, "start_offset_seconds")
    post = _nonnegative_seconds(post_baseline_seconds, "post_baseline_seconds")

    transmit = [
        (start + idx * spacing, start + idx * spacing + pulse_duration)
        for idx in range(num_echoes)
    ]
    ringdown_intervals = [
        (stop, min(stop + ringdown, start + (idx + 1) * spacing))
        for idx, (_, stop) in enumerate(transmit)
        if ringdown > 0.0
    ]
    sequence_end = start + num_echoes * spacing
    baseline: list[Interval] = []
    if start > 0.0:
        baseline.append((0.0, start))
    if post > 0.0:
        baseline.append((sequence_end, sequence_end + post))
    if baseline_intervals is not None:
        baseline.extend(tuple(pair) for pair in baseline_intervals)

    return mask_from_intervals(
        sample_rate_hz,
        duration,
        transmit=transmit,
        ringdown=ringdown_intervals,
        baseline=baseline,
    )


def sorc_mask_from_metadata(
    num_samples: int,
    sample_rate_hz: float,
    *,
    half_spacing_seconds: float,
    detection_duration_seconds: float,
    num_pulses: int,
    ringdown_seconds: float = 0.0,
    start_offset_seconds: float = 0.0,
    post_baseline_seconds: float = 0.0,
    baseline_intervals: Sequence[Interval] | None = None,
    initial_gap_is_baseline: bool = True,
) -> AcquisitionMask:
    """Reconstruct a SORC acquisition mask over a window of ``num_samples`` ADC samples.

    The metadata counterpart of :func:`sorc_acquisition_mask`: each cycle is
    ``tau - pulse - tau`` with ``tau = half_spacing_seconds``, driven by the
    recorded sample count and raw timing.
    """

    num_samples, sample_rate_hz, duration = _window_duration(num_samples, sample_rate_hz)
    tau = _positive_seconds(half_spacing_seconds, "half_spacing_seconds")
    pulse_duration = _positive_seconds(detection_duration_seconds, "detection_duration_seconds")
    num_pulses = int(num_pulses)
    if num_pulses <= 0:
        raise ValueError("num_pulses must be positive")
    ringdown = _nonnegative_seconds(ringdown_seconds, "ringdown_seconds")
    start = _nonnegative_seconds(start_offset_seconds, "start_offset_seconds")
    post = _nonnegative_seconds(post_baseline_seconds, "post_baseline_seconds")
    cycle_duration = 2.0 * tau + pulse_duration

    transmit = []
    ringdown_intervals = []
    for idx in range(num_pulses):
        cycle_start = start + idx * cycle_duration
        pulse_start = cycle_start + tau
        pulse_stop = pulse_start + pulse_duration
        transmit.append((pulse_start, pulse_stop))
        if ringdown > 0.0:
            ringdown_intervals.append((pulse_stop, min(pulse_stop + ringdown, cycle_start + cycle_duration)))

    sequence_end = start + num_pulses * cycle_duration
    baseline: list[Interval] = []
    if start > 0.0:
        baseline.append((0.0, start))
    if post > 0.0:
        baseline.append((sequence_end, sequence_end + post))
    if baseline_intervals is not None:
        baseline.extend(tuple(pair) for pair in baseline_intervals)
    if initial_gap_is_baseline and tau > 0.0:
        baseline.append((start, start + tau))

    return mask_from_intervals(
        sample_rate_hz,
        duration,
        transmit=transmit,
        ringdown=ringdown_intervals,
        baseline=baseline,
    )


def nqr_recording_from_samples(
    primary: np.ndarray,
    references: np.ndarray | None,
    sample_rate_hz: float,
    *,
    sequence: str = "slse",
    **timing: float,
) -> RFIRecording:
    """Pair measured ADC windows with a mask reconstructed from sequence timing.

    ``primary`` is the recorded primary channel and ``references`` the stacked
    reference channels ``(K, N)`` (or ``None`` for a reference-free tracker
    record). ``sequence`` selects the mask reconstruction (``"slse"`` or
    ``"sorc"``); the remaining timing keywords are forwarded to
    :func:`slse_mask_from_metadata` / :func:`sorc_mask_from_metadata`. The result
    drops straight into the ``(primary, references, mask)`` canceller contract.
    """

    prim = np.asarray(primary).reshape(-1)
    num_samples = prim.size
    if sequence == "slse":
        mask = slse_mask_from_metadata(num_samples, sample_rate_hz, **timing)
    elif sequence == "sorc":
        mask = sorc_mask_from_metadata(num_samples, sample_rate_hz, **timing)
    else:
        raise ValueError("sequence must be 'slse' or 'sorc'")
    if references is None:
        references = np.zeros((0, num_samples), dtype=np.float64)
    metadata = {"sequence": sequence, **{key: float(value) for key, value in timing.items()}}
    return RFIRecording(
        primary=prim,
        references=references,
        sample_rate_hz=float(sample_rate_hz),
        mask=mask,
        metadata=metadata,
    )
