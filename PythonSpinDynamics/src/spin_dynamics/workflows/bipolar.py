"""Bipolar 13-interval PGSTE for background-gradient-suppressed diffusion.

A constant background gradient ``g0`` -- for example the internal gradient set up
by magnetic-susceptibility contrast in porous media (see
``spin_dynamics.susceptibility``) -- biases an ordinary pulsed-gradient
measurement through the cross-term between the applied and background gradients.
The Cotts 13-interval alternating pulsed-gradient stimulated-echo (APGSTE)
sequence cancels that cross-term. It is the sequence in the Bruker
``diff_stebp.gp`` program: a stimulated echo (three 90 degree pulses) with one
180 degree refocusing pulse in each of the two encoding periods, a gradient lobe
on each side of every 180, and the applied gradient polarity inverted between
the two encoding periods. The 180 pulses refocus the continuously present
background gradient within each period, and the polarity alternation cancels the
applied-times-background cross-term, so the apparent diffusion coefficient is
unbiased by ``g0``.

This module works in the toggling (coherence) frame. Each interval carries the
applied gradient and the coherence order ``p(t)`` selected by the
``diff_stebp`` phase cycle (``spin_dynamics.phase_cycling.diff_stebp_phase_cycle``):
``+1`` and ``-1`` on the two sides of a 180, and ``0`` during longitudinal
storage, where the dephasing wavevector is parked and no new phase accrues. The
gradient polarity carried by ``applied_gradient`` alternates between the two
encoding periods, so the effective (coherence-weighted) gradient pattern is
``[+, -, -, +]``. ``toggling_frame_moments`` integrates the wavevector to return
the applied diffusion weighting together with the background cross- and
self-terms, so the suppression is explicit:

    ln(E) = -D * (b_applied + g0 * cross_coefficient + g0**2 * background_coefficient).

For the 13-interval sequence ``cross_coefficient`` is zero; for the ordinary
monopolar stimulated echo it is not, and the apparent diffusion coefficient then
drifts with ``g0``. The moment model is the free-diffusion (Gaussian-propagator)
b-value for ideal rectangular lobes; restricted-geometry and ramp effects need
the walker pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.phase_cycling import (
    PhaseCycle,
    diff_stebp_phase_cycle,
    pgste_stimulated_echo_phase_cycle,
)


__all__ = [
    "ToggleInterval",
    "GradientMoments",
    "BipolarPGSTEResult",
    "toggling_frame_moments",
    "cotts_thirteen_interval_intervals",
    "monopolar_pgste_intervals",
    "run_cotts_thirteen_interval_moment",
    "run_monopolar_pgste_moment",
]

# Proton gyromagnetic ratio (rad/s/T), matching the package PGSE default.
GAMMA = 2.675e8


@dataclass(frozen=True)
class ToggleInterval:
    """One piecewise-constant interval in the toggling (coherence) frame.

    ``applied_gradient`` is the physical applied gradient (T/m) during this
    interval. ``sign`` is the coherence sign: ``+1`` or ``-1`` for transverse
    evolution (already folding in 180 inversions and the stimulated-echo
    conjugation across storage), or ``0`` for longitudinal storage, during which
    the dephasing wavevector is held and no new phase accrues.
    """

    duration: float
    applied_gradient: float = 0.0
    sign: int = 1

    def __post_init__(self) -> None:
        duration = float(self.duration)
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("duration must be non-negative and finite")
        if self.sign not in (-1, 0, 1):
            raise ValueError("sign must be -1, 0, or +1")
        if not np.isfinite(self.applied_gradient):
            raise ValueError("applied_gradient must be finite")
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "applied_gradient", float(self.applied_gradient))
        object.__setattr__(self, "sign", int(self.sign))


@dataclass(frozen=True)
class GradientMoments:
    """Toggling-frame diffusion moments for a gradient waveform.

    The diffusion attenuation in a constant background gradient ``g0`` is
    ``ln(E) = -D * (b_applied + g0 * cross_coefficient + g0**2 *
    background_coefficient)``. ``residual_applied`` and ``residual_background``
    are the end-of-sequence wavevectors (per unit background gradient for the
    latter); both must be zero for a stationary-spin echo.
    """

    b_applied: float
    cross_coefficient: float
    background_coefficient: float
    residual_applied: float
    residual_background: float
    total_duration: float

    @property
    def refocuses_static_spins(self) -> bool:
        scale = max(abs(self.residual_applied), abs(self.residual_background), 1.0)
        tol = 1e-6 * scale
        return abs(self.residual_applied) <= tol and abs(self.residual_background) <= tol


@dataclass(frozen=True)
class BipolarPGSTEResult:
    """Moment-model result for a stimulated-echo diffusion sequence.

    ``cross_term_bias`` is the fractional slope bias the background gradient
    imposes on a diffusion measurement made with this applied gradient,
    ``g0 * cross_coefficient / b_applied``. It is ~0 for the 13-interval
    sequence and non-zero for the monopolar one. The background *self*-term
    (``g0**2 * background_coefficient``) is a gradient-independent attenuation
    that offsets the echo but does not bias the diffusion coefficient recovered
    from a proper b-value sweep, so it is reported through ``moments`` but not
    folded into ``cross_term_bias``.
    """

    signal: complex
    b_value: float
    diffusion_attenuation: float
    cross_term_bias: float
    moments: GradientMoments
    diffusion_coefficient: float
    background_gradient: float
    gradient_amplitude: float
    gradient_duration: float
    storage_time: float
    gamma: float
    label: str
    phase_cycle: PhaseCycle


def toggling_frame_moments(
    intervals: Iterable[ToggleInterval],
    *,
    gamma: float = GAMMA,
) -> GradientMoments:
    """Integrate the toggling-frame wavevector moments for a gradient waveform.

    Returns the applied diffusion weighting and the background cross- and
    self-terms by integrating the piecewise-linear wavevector across the supplied
    intervals. The applied wavevector accrues only during gradient lobes; the
    background wavevector (per unit gradient) accrues whenever the spins are
    transverse, so it is sensitive to the placement of the 180 pulses.
    """

    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma == 0.0:
        raise ValueError("gamma must be finite and non-zero")

    q_applied = 0.0
    q_background = 0.0
    b_applied = 0.0
    cross = 0.0
    background = 0.0
    total = 0.0

    for interval in intervals:
        h = interval.duration
        total += h
        if interval.sign == 0:
            # Longitudinal storage: wavevector parked, no new phase. The parked
            # values still accrue their squared/cross contributions over h.
            b_applied += q_applied * q_applied * h
            background += q_background * q_background * h
            cross += 2.0 * q_applied * q_background * h
            continue
        slope_applied = gamma * interval.sign * interval.applied_gradient
        slope_background = gamma * interval.sign  # per unit background gradient
        b_applied += (
            q_applied * q_applied * h
            + q_applied * slope_applied * h**2
            + slope_applied * slope_applied * h**3 / 3.0
        )
        background += (
            q_background * q_background * h
            + q_background * slope_background * h**2
            + slope_background * slope_background * h**3 / 3.0
        )
        cross += 2.0 * (
            q_applied * q_background * h
            + 0.5 * (q_applied * slope_background + q_background * slope_applied) * h**2
            + slope_applied * slope_background * h**3 / 3.0
        )
        q_applied += slope_applied * h
        q_background += slope_background * h

    return GradientMoments(
        b_applied=float(b_applied),
        cross_coefficient=float(cross),
        background_coefficient=float(background),
        residual_applied=float(q_applied),
        residual_background=float(q_background),
        total_duration=float(total),
    )


def cotts_thirteen_interval_intervals(
    *,
    gradient_amplitude: float,
    gradient_duration: float,
    half_echo_time: float,
    storage_time: float,
) -> tuple[ToggleInterval, ...]:
    """Build the 13-interval bipolar APGSTE toggling-frame intervals.

    Each stimulated-echo half holds two gradient lobes of duration
    ``gradient_duration`` straddling a 180 degree pulse; ``half_echo_time`` is
    the spacing from the start of a lobe to the centre of its 180 pulse, so the
    free spacing on each side of the lobe is ``half_echo_time - gradient_duration``
    and must be non-negative. The applied polarity is ``+`` in the first encode
    period and ``-`` in the second; the coherence signs fold in the two 180
    inversions and the stimulated-echo conjugation across ``storage_time``.
    """

    g = float(gradient_amplitude)
    delta = float(gradient_duration)
    tau = float(half_echo_time)
    storage = float(storage_time)
    rest = tau - delta
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    if rest < 0.0:
        raise ValueError("half_echo_time must be at least gradient_duration")
    if storage < 0.0:
        raise ValueError("storage_time must be non-negative")

    return (
        ToggleInterval(delta, +g, +1),
        ToggleInterval(rest, 0.0, +1),
        ToggleInterval(rest, 0.0, -1),
        ToggleInterval(delta, +g, -1),
        ToggleInterval(storage, 0.0, 0),
        ToggleInterval(delta, -g, +1),
        ToggleInterval(rest, 0.0, +1),
        ToggleInterval(rest, 0.0, -1),
        ToggleInterval(delta, -g, -1),
    )


def monopolar_pgste_intervals(
    *,
    gradient_amplitude: float,
    gradient_duration: float,
    half_echo_time: float,
    storage_time: float,
) -> tuple[ToggleInterval, ...]:
    """Build an ordinary monopolar PGSTE for comparison.

    One gradient lobe per stimulated-echo half, no 180 refocusing pulses. This
    sequence does not suppress the background gradient, so its cross-term is
    non-zero and its apparent diffusion coefficient drifts with ``g0``.
    """

    g = float(gradient_amplitude)
    delta = float(gradient_duration)
    tau = float(half_echo_time)
    storage = float(storage_time)
    rest = tau - delta
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    if rest < 0.0:
        raise ValueError("half_echo_time must be at least gradient_duration")
    if storage < 0.0:
        raise ValueError("storage_time must be non-negative")

    return (
        ToggleInterval(delta, +g, +1),
        ToggleInterval(rest, 0.0, +1),
        ToggleInterval(storage, 0.0, 0),
        ToggleInterval(rest, 0.0, -1),
        ToggleInterval(delta, +g, -1),
    )


def _run_moment(
    intervals: Sequence[ToggleInterval],
    *,
    label: str,
    phase_cycle: PhaseCycle,
    diffusion_coefficient: float,
    background_gradient: float,
    gradient_amplitude: float,
    gradient_duration: float,
    storage_time: float,
    initial_signal: complex,
    gamma: float,
) -> BipolarPGSTEResult:
    diffusion = float(diffusion_coefficient)
    if diffusion < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    g0 = float(background_gradient)
    moments = toggling_frame_moments(intervals, gamma=gamma)
    b_value = float(moments.b_applied)
    total_weighting = (
        moments.b_applied
        + g0 * moments.cross_coefficient
        + g0 * g0 * moments.background_coefficient
    )
    attenuation = float(np.exp(-diffusion * total_weighting))
    # the cross-term is the only g0-dependent part that biases the diffusion
    # slope; the g0^2 self-term is a gradient-independent offset (see docstring)
    if b_value > 0.0:
        cross_bias = g0 * moments.cross_coefficient / b_value
    else:
        cross_bias = float("nan")
    signal = complex(initial_signal) * attenuation
    return BipolarPGSTEResult(
        signal=signal,
        b_value=b_value,
        diffusion_attenuation=attenuation,
        cross_term_bias=float(cross_bias),
        moments=moments,
        diffusion_coefficient=diffusion,
        background_gradient=g0,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=float(gradient_duration),
        storage_time=float(storage_time),
        gamma=float(gamma),
        label=label,
        phase_cycle=phase_cycle,
    )


def run_cotts_thirteen_interval_moment(
    *,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    half_echo_time: float = 6.0e-3,
    storage_time: float = 40.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    background_gradient: float = 0.0,
    initial_signal: complex = 1.0 + 0.0j,
    gamma: float = GAMMA,
) -> BipolarPGSTEResult:
    """Run the 13-interval bipolar APGSTE moment model.

    The applied-times-background cross-term cancels, so the recovered apparent
    diffusion coefficient is unbiased by ``background_gradient``.
    """

    intervals = cotts_thirteen_interval_intervals(
        gradient_amplitude=gradient_amplitude,
        gradient_duration=gradient_duration,
        half_echo_time=half_echo_time,
        storage_time=storage_time,
    )
    return _run_moment(
        intervals,
        label="cotts_13_interval",
        phase_cycle=diff_stebp_phase_cycle(),
        diffusion_coefficient=diffusion_coefficient,
        background_gradient=background_gradient,
        gradient_amplitude=gradient_amplitude,
        gradient_duration=gradient_duration,
        storage_time=storage_time,
        initial_signal=initial_signal,
        gamma=gamma,
    )


def run_monopolar_pgste_moment(
    *,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    half_echo_time: float = 6.0e-3,
    storage_time: float = 40.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    background_gradient: float = 0.0,
    initial_signal: complex = 1.0 + 0.0j,
    gamma: float = GAMMA,
) -> BipolarPGSTEResult:
    """Run an ordinary monopolar PGSTE moment model for comparison.

    The cross-term is non-zero, so the recovered apparent diffusion coefficient
    drifts with ``background_gradient``.
    """

    intervals = monopolar_pgste_intervals(
        gradient_amplitude=gradient_amplitude,
        gradient_duration=gradient_duration,
        half_echo_time=half_echo_time,
        storage_time=storage_time,
    )
    return _run_moment(
        intervals,
        label="monopolar_pgste",
        phase_cycle=pgste_stimulated_echo_phase_cycle(),
        diffusion_coefficient=diffusion_coefficient,
        background_gradient=background_gradient,
        gradient_amplitude=gradient_amplitude,
        gradient_duration=gradient_duration,
        storage_time=storage_time,
        initial_signal=initial_signal,
        gamma=gamma,
    )
