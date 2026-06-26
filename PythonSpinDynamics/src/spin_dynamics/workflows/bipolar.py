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
from spin_dynamics.phase_cycling import (
    PhaseCycle,
    diff_stebp_phase_cycle,
    pgste_stimulated_echo_phase_cycle,
)
from spin_dynamics.sequences.motion import (
    MotionSequenceResult,
    MotionSequenceStep,
    run_motion_sequence,
)


GradientAxis = Literal["x", "z"]


__all__ = [
    "ToggleInterval",
    "GradientMoments",
    "BipolarPGSTEResult",
    "BipolarPGSTEWalkerResult",
    "toggling_frame_moments",
    "cotts_thirteen_interval_intervals",
    "monopolar_pgste_intervals",
    "run_cotts_thirteen_interval_moment",
    "run_monopolar_pgste_moment",
    "run_cotts_thirteen_interval_walkers",
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


@dataclass(frozen=True)
class BipolarPGSTEWalkerResult:
    """Random-walker result for the bipolar 13-interval PGSTE.

    The explicit moving-walker simulation runs the real five-pulse sequence with
    the two 180 refocusing pulses and the four gradient lobes, so it captures
    restricted-geometry effects and finite-pulse timing that the toggling-frame
    moment model omits. ``b_value`` is the free-diffusion moment-model b-value
    for reference; ``signal`` is the acquired stimulated echo.
    """

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    sequence: MotionSequenceResult
    initial_ensemble: ParticleEnsemble
    gradient_amplitude: float
    gradient_duration: float
    half_echo_time: float
    storage_time: float
    background_gradient: float
    diffusion_coefficient: float
    gamma: float
    phase_cycle: PhaseCycle
    label: str = "cotts_13_interval_walkers"


def _gradient_tuple(value: float, axis: GradientAxis) -> tuple[float, float]:
    if axis == "x":
        return (float(value), 0.0)
    if axis == "z":
        return (0.0, float(value))
    raise ValueError("axis must be 'x' or 'z'")


def _bipolar_background_fields(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    *,
    gamma: float,
    background_gradient: float,
    gradient_axis: GradientAxis,
    total_time: float,
    diffusion_coefficient: float,
) -> MotionFieldMaps2D:
    """Build motion field maps with an optional constant background gradient.

    A constant background gradient ``g0`` along the gradient axis is a linear
    off-resonance map ``omega(r) = gamma * g0 * r``, which the walkers feel
    during every free-precession interval (and which the longitudinal storage
    period does not dephase). A two-point grid represents the linear ramp
    exactly under bilinear sampling.
    """

    sigma = np.sqrt(max(0.0, 2.0 * float(diffusion_coefficient) * float(total_time)))
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
    xs = np.array([x_min, x_max], dtype=np.float64)
    zs = np.array([z_min, z_max], dtype=np.float64)
    if float(background_gradient) == 0.0:
        return make_motion_field_maps_2d(xs, zs)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    coordinate = grid_x if gradient_axis == "x" else grid_z
    b0_map = float(gamma) * float(background_gradient) * coordinate
    return make_motion_field_maps_2d(xs, zs, b0_map=b0_map)


def _make_thirteen_interval_steps(
    *,
    gradient_duration: float,
    half_echo_time: float,
    storage_time: float,
    gradient: tuple[float, float],
    spoiler: tuple[float, float],
    excitation_duration: float,
    refocusing_duration: float,
    substeps_per_interval: int,
) -> tuple[MotionSequenceStep, ...]:
    rest = half_echo_time - gradient_duration
    sub = substeps_per_interval
    negative = (-gradient[0], -gradient[1])

    def pulse(flip: float, label: str) -> MotionSequenceStep:
        duration = excitation_duration if flip < np.pi * 0.75 else refocusing_duration
        return MotionSequenceStep(
            duration=duration,
            rf_amplitude=flip / duration,
            rf_phase=(np.pi / 2.0 if flip < np.pi * 0.75 else 0.0),
            substeps=max(1, sub),
            label=label,
        )

    def lobe(value: tuple[float, float], label: str, *, acquire: bool = False) -> MotionSequenceStep:
        return MotionSequenceStep(
            duration=gradient_duration,
            gradient=value,
            acquire=acquire,
            num_samples=1 if acquire else 0,
            substeps=sub,
            label=label,
        )

    def gap(label: str) -> list[MotionSequenceStep]:
        if rest <= 0.0:
            return []
        return [MotionSequenceStep(duration=rest, substeps=sub, label=label)]

    def encode(value: tuple[float, float], index: int, *, acquire_last: bool):
        # lobe - gap - 180 - gap - lobe, the 180 centred so a constant
        # background gradient refocuses within the encoding period
        return [
            lobe(value, f"enc{index}_lobe1"),
            *gap(f"enc{index}_gap1"),
            pulse(np.pi, f"refocus_180_{index}"),
            *gap(f"enc{index}_gap2"),
            lobe(
                value,
                "stimulated_echo" if acquire_last else f"enc{index}_lobe2",
                acquire=acquire_last,
            ),
        ]

    steps = [pulse(0.5 * np.pi, "excitation_90")]
    steps += encode(gradient, 1, acquire_last=False)
    steps.append(pulse(0.5 * np.pi, "store_90"))
    steps.append(
        MotionSequenceStep(
            duration=storage_time,
            gradient=spoiler,
            substeps=sub,
            label="storage",
        )
    )
    steps.append(pulse(0.5 * np.pi, "read_90"))
    steps += encode(negative, 2, acquire_last=True)
    return tuple(steps)


def run_cotts_thirteen_interval_walkers(
    *,
    rho: Iterable[float] | np.ndarray | None = None,
    x_axis: Iterable[float] | np.ndarray | None = None,
    z_axis: Iterable[float] | np.ndarray | None = None,
    fields: MotionFieldMaps2D | None = None,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    half_echo_time: float = 6.0e-3,
    storage_time: float = 40.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    gamma: float = GAMMA,
    gradient_axis: GradientAxis = "x",
    background_gradient: float = 0.0,
    walkers_per_cell: int = 128,
    seed: int | None = None,
    jitter: bool = False,
    excitation_duration: float = 100.0e-6,
    refocusing_duration: float = 200.0e-6,
    spoiler_gradient: float = 0.2,
    spoiler_axis: GradientAxis = "x",
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 8,
) -> BipolarPGSTEWalkerResult:
    """Run the bipolar 13-interval PGSTE with explicit random-walker diffusion.

    The five-pulse stimulated echo is built with its two 180 refocusing pulses
    and four gradient lobes; the applied gradient polarity is positive in the
    first encoding period and negative in the second, the alternation that
    cancels the background-gradient cross-term. A constant ``background_gradient``
    (T/m) along ``gradient_axis`` is applied as a linear off-resonance map, so the
    13-interval suppression and restricted-geometry effects appear directly in the
    simulated signal. The storage period uses the spoiler-plus-``mth=0`` PGSTE
    pathway-selection convention. Unlike a monopolar stimulated echo, the bipolar
    pair refocuses the encoding phase before storage, so a fully refocused
    component is stored and the diffusion attenuation matches ``exp(-b D)`` for
    free diffusion.
    """

    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("RF pulse durations must be positive")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")
    delta = float(gradient_duration)
    tau = float(half_echo_time)
    storage = float(storage_time)
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    if tau < delta:
        raise ValueError("half_echo_time must be at least gradient_duration")
    if storage <= 0.0:
        raise ValueError("storage_time must be positive")

    rho_arr = (
        np.ones((1, 1), dtype=np.float64)
        if rho is None
        else _walker_map2d(rho, "rho")
    )
    x = (
        np.array([0.0], dtype=np.float64)
        if x_axis is None
        else _walker_axis(x_axis, "x_axis", rho_arr.shape[0])
    )
    z = (
        np.array([0.0], dtype=np.float64)
        if z_axis is None
        else _walker_axis(z_axis, "z_axis", rho_arr.shape[1])
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

    total_time = (
        storage
        + 4.0 * tau
        + 2.0 * float(refocusing_duration)
        + 3.0 * float(excitation_duration)
    )
    if fields is None:
        fields = _bipolar_background_fields(
            x,
            z,
            gamma=gamma,
            background_gradient=background_gradient,
            gradient_axis=gradient_axis,
            total_time=total_time,
            diffusion_coefficient=diffusion_coefficient,
        )

    gradient = _gradient_tuple(float(gamma) * float(gradient_amplitude), gradient_axis)
    spoiler = _gradient_tuple(float(gamma) * float(spoiler_gradient), spoiler_axis)
    steps = _make_thirteen_interval_steps(
        gradient_duration=delta,
        half_echo_time=tau,
        storage_time=storage,
        gradient=gradient,
        spoiler=spoiler,
        excitation_duration=float(excitation_duration),
        refocusing_duration=float(refocusing_duration),
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
        # mth=0 keeps equilibrium magnetization from regrowing into a
        # contaminating FID during storage while T1 still decays the stored
        # signal; the diff_stebp phase cycle records the selected pathway.
        mth=0.0,
        boundary=boundary,
        default_substeps=int(substeps_per_interval),
    )
    b_value = toggling_frame_moments(
        cotts_thirteen_interval_intervals(
            gradient_amplitude=gradient_amplitude,
            gradient_duration=delta,
            half_echo_time=tau,
            storage_time=storage,
        ),
        gamma=gamma,
    ).b_applied
    return BipolarPGSTEWalkerResult(
        signal=sequence.signal,
        echo_times=sequence.sample_times,
        b_value=float(b_value),
        sequence=sequence,
        initial_ensemble=ensemble,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=delta,
        half_echo_time=tau,
        storage_time=storage,
        background_gradient=float(background_gradient),
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=float(gamma),
        phase_cycle=diff_stebp_phase_cycle(),
    )


def _walker_map2d(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _walker_axis(
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
