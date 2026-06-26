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
from spin_dynamics.phase_cycling import (
    PhaseCycle,
    pgste_stimulated_echo_phase_cycle,
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


@dataclass(frozen=True)
class PGSTEWalkerResult:
    """Random-walker stimulated-echo PGSE (PGSTE) result.

    The diffusion encoding is split across a storage interval during which the
    magnetization is longitudinal, so the reachable diffusion time is bounded by
    ``T1`` rather than ``T2``. The stimulated echo carries half of the
    spin-echo amplitude (the other coherence pathway is spoiled away).
    """

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    storage_time: float
    sequence: MotionSequenceResult
    initial_ensemble: ParticleEnsemble
    gradient_amplitude: float
    gradient_duration: float
    diffusion_time: float
    diffusion_coefficient: float
    gamma: float
    phase_cycle: PhaseCycle | None = None
    backend: str = "walkers_ste"


@dataclass(frozen=True)
class DDEWalkerResult:
    """Random-walker double diffusion encoding (DDE / double-PGSE) result.

    Two PGSE blocks separated by a mixing time encode displacement along two
    directions at a relative angle ``psi = angle2 - angle1``. The dependence of
    the echo on ``psi`` reports microscopic anisotropy of the local geometry,
    which survives orientational averaging even when single-PGSE diffusion is
    macroscopically isotropic. ``b_value`` is the per-block Stejskal-Tanner value.
    """

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    mixing_time: float
    angle1: float
    angle2: float
    sequence: MotionSequenceResult
    initial_ensemble: ParticleEnsemble
    gradient_amplitude: float
    gradient_duration: float
    diffusion_time: float
    diffusion_coefficient: float
    gamma: float
    backend: str = "walkers_dde"


@dataclass(frozen=True)
class OGSEWalkerResult:
    """Random-walker oscillating-gradient spin-echo (OGSE) result.

    The two diffusion-encoding lobes are cosine-modulated gradient waveforms, so
    the encoding spectrum is concentrated at the angular frequency
    ``omega = 2*pi*oscillation_frequency``. Sweeping the frequency maps the
    diffusion spectrum ``D(omega)``: in restricted geometry the apparent
    diffusion coefficient rises from the long-time (tortuosity) value toward the
    bulk value as the frequency increases.
    """

    signal: np.ndarray
    echo_times: np.ndarray
    b_value: float
    oscillation_frequency: float
    num_periods: int
    encoding_time: float
    sequence: MotionSequenceResult
    initial_ensemble: ParticleEnsemble
    gradient_amplitude: float
    diffusion_coefficient: float
    gamma: float
    backend: str = "walkers_ogse"


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


def run_pgste_walkers(
    *,
    rho: Iterable[float] | np.ndarray | None = None,
    x_axis: Iterable[float] | np.ndarray | None = None,
    z_axis: Iterable[float] | np.ndarray | None = None,
    fields: MotionFieldMaps2D | None = None,
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
    encode_delay: float = 0.0,
    spoiler_gradient: float = 0.2,
    spoiler_axis: PGSEAxis = "x",
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 8,
) -> PGSTEWalkerResult:
    """Run a pulsed-gradient stimulated-echo (PGSTE) walker simulation.

    The sequence is ``90 - G(delta) - 90 - [storage] - 90 - G(delta) - echo``.
    The first storage pulse parks one quadrature of the encoded magnetization
    along the longitudinal axis, so during the (long) storage interval it decays
    with ``T1`` instead of ``T2``. A spoiler gradient applied during storage
    dephases the residual transverse coherences; the surviving stimulated-echo
    pathway carries half of the corresponding spin-echo amplitude.

    ``diffusion_time`` is the leading-edge separation of the two gradient lobes,
    so the rectangular Stejskal-Tanner ``b = (gamma G delta)^2 (Delta - delta/3)``
    still applies. The storage interval is
    ``Delta - delta - 2*encode_delay - 2*excitation_duration``.
    """

    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    if excitation_duration <= 0.0:
        raise ValueError("excitation_duration must be positive")
    if encode_delay < 0.0:
        raise ValueError("encode_delay must be non-negative")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")
    delta = float(gradient_duration)
    delta_big = float(diffusion_time)
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    overhead = delta + 2.0 * float(encode_delay) + 2.0 * float(excitation_duration)
    storage_time = delta_big - overhead
    if storage_time <= 0.0:
        raise ValueError(
            "diffusion_time must exceed gradient_duration + 2*encode_delay "
            "+ 2*excitation_duration"
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
    spoiler = _gradient_tuple(float(gamma) * float(spoiler_gradient), spoiler_axis)
    phase_cycle = pgste_stimulated_echo_phase_cycle()
    steps = _make_pgste_steps(
        gradient_duration=delta,
        gradient=gradient,
        spoiler=spoiler,
        storage_time=storage_time,
        encode_delay=float(encode_delay),
        excitation_duration=float(excitation_duration),
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
        # The phase_cycle records the selected stimulated-echo pathway. mth=0
        # keeps equilibrium magnetization from regrowing into a contaminating
        # FID during storage while T1 still decays the stored signal.
        mth=0.0,
        boundary=boundary,
        default_substeps=int(substeps_per_interval),
    )
    return PGSTEWalkerResult(
        signal=sequence.signal,
        echo_times=sequence.sample_times,
        b_value=pgse_b_value(
            gradient_amplitude,
            gradient_duration,
            diffusion_time,
            gamma=gamma,
        ),
        storage_time=storage_time,
        sequence=sequence,
        initial_ensemble=ensemble,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=delta,
        diffusion_time=delta_big,
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=float(gamma),
        phase_cycle=phase_cycle,
    )


def run_dde_walkers(
    *,
    rho: Iterable[float] | np.ndarray | None = None,
    x_axis: Iterable[float] | np.ndarray | None = None,
    z_axis: Iterable[float] | np.ndarray | None = None,
    fields: MotionFieldMaps2D | None = None,
    gradient_amplitude: float = 0.05,
    gradient_duration: float = 2.0e-3,
    diffusion_time: float = 20.0e-3,
    mixing_time: float = 0.0,
    angle1: float = 0.0,
    angle2: float = 0.0,
    diffusion_coefficient: float = 2.3e-9,
    gamma: float = 2.675e8,
    walkers_per_cell: int = 128,
    seed: int | None = None,
    jitter: bool = False,
    excitation_duration: float = 100.0e-6,
    refocusing_duration: float = 200.0e-6,
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 8,
) -> DDEWalkerResult:
    """Run a double diffusion encoding (DDE / double-PGSE) walker simulation.

    The sequence is two refocused PGSE blocks separated by a mixing time:
    ``90 - [G1 block] - mixing - [G2 block] - echo``. The two blocks encode
    displacement along ``angle1`` and ``angle2`` (radians in the x-z plane) with
    equal gradient magnitude. Sweeping the relative angle ``psi = angle2 - angle1``
    probes microscopic anisotropy: in a restricted anisotropic pore the echo
    modulates with ``psi`` (a ``cos 2*psi`` term at leading order), whereas an
    isotropic pore gives an angle-independent echo.
    """

    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("RF pulse durations must be positive")
    if mixing_time < 0.0:
        raise ValueError("mixing_time must be non-negative")
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

    moment = float(gamma) * float(gradient_amplitude)
    gradient_1 = (moment * float(np.cos(angle1)), moment * float(np.sin(angle1)))
    gradient_2 = (moment * float(np.cos(angle2)), moment * float(np.sin(angle2)))
    gap = 0.5 * (delta_big - delta - float(refocusing_duration))
    steps = _make_dde_steps(
        gradient_duration=delta,
        gradient_1=gradient_1,
        gradient_2=gradient_2,
        gap=gap,
        mixing_time=float(mixing_time),
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
        boundary=boundary,
        default_substeps=int(substeps_per_interval),
    )
    return DDEWalkerResult(
        signal=sequence.signal,
        echo_times=sequence.sample_times,
        b_value=pgse_b_value(
            gradient_amplitude,
            gradient_duration,
            diffusion_time,
            gamma=gamma,
        ),
        mixing_time=float(mixing_time),
        angle1=float(angle1),
        angle2=float(angle2),
        sequence=sequence,
        initial_ensemble=ensemble,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=delta,
        diffusion_time=delta_big,
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=float(gamma),
    )


def run_ogse_walkers(
    *,
    rho: Iterable[float] | np.ndarray | None = None,
    x_axis: Iterable[float] | np.ndarray | None = None,
    z_axis: Iterable[float] | np.ndarray | None = None,
    fields: MotionFieldMaps2D | None = None,
    gradient_amplitude: float = 0.05,
    oscillation_frequency: float = 100.0,
    num_periods: int = 2,
    samples_per_period: int = 16,
    diffusion_coefficient: float = 2.3e-9,
    gamma: float = 2.675e8,
    gradient_axis: PGSEAxis = "x",
    walkers_per_cell: int = 128,
    seed: int | None = None,
    jitter: bool = False,
    excitation_duration: float = 100.0e-6,
    refocusing_duration: float = 200.0e-6,
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 4,
) -> OGSEWalkerResult:
    """Run an oscillating-gradient spin-echo (OGSE) walker simulation.

    Each diffusion-encoding lobe is a cosine waveform ``G cos(2*pi*f*t)`` of
    ``num_periods`` whole periods, applied symmetrically around a refocusing
    pulse: ``90 - cos lobe - 180 - cos lobe - echo``. Because each lobe spans an
    integer number of periods, its zeroth gradient moment is zero and stationary
    spins refocus. The encoding power sits at ``omega = 2*pi*f``, so sweeping
    ``oscillation_frequency`` probes the diffusion spectrum ``D(omega)`` -- the
    short-diffusion-time regime that ordinary PGSE cannot reach.
    """

    if diffusion_coefficient < 0.0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    if excitation_duration <= 0.0 or refocusing_duration <= 0.0:
        raise ValueError("RF pulse durations must be positive")
    if oscillation_frequency <= 0.0:
        raise ValueError("oscillation_frequency must be positive")
    if num_periods < 1:
        raise ValueError("num_periods must be at least 1")
    if samples_per_period < 4:
        raise ValueError("samples_per_period must be at least 4")
    if substeps_per_interval <= 0:
        raise ValueError("substeps_per_interval must be positive")

    frequency = float(oscillation_frequency)
    periods = int(num_periods)
    per_period = int(samples_per_period)
    step_dt = 1.0 / frequency / per_period
    num_steps = periods * per_period
    midpoints = (np.arange(num_steps) + 0.5) * step_dt
    cosine = np.cos(2.0 * np.pi * frequency * midpoints)
    amplitude = float(gradient_amplitude)
    encoding_time = periods / frequency

    # Effective gradient segments for the b-value: the 180 flips the coherence
    # sign, so the second lobe contributes with opposite effective sign.
    segments: list[tuple[float, float]] = [
        (step_dt, amplitude * float(c)) for c in cosine
    ]
    segments.append((float(refocusing_duration), 0.0))
    segments.extend((step_dt, -amplitude * float(c)) for c in cosine)
    b_value = gradient_moment_b_value(segments, gamma=gamma)

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
        fields = _default_motion_fields(x, z, encoding_time, diffusion_coefficient)

    steps = _make_ogse_steps(
        cosine=cosine,
        step_dt=step_dt,
        moment=float(gamma) * amplitude,
        gradient_axis=gradient_axis,
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
        boundary=boundary,
        default_substeps=int(substeps_per_interval),
    )
    return OGSEWalkerResult(
        signal=sequence.signal,
        echo_times=sequence.sample_times,
        b_value=b_value,
        oscillation_frequency=frequency,
        num_periods=periods,
        encoding_time=encoding_time,
        sequence=sequence,
        initial_ensemble=ensemble,
        gradient_amplitude=amplitude,
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


def _make_pgste_steps(
    *,
    gradient_duration: float,
    gradient: tuple[float, float],
    spoiler: tuple[float, float],
    storage_time: float,
    encode_delay: float,
    excitation_duration: float,
    substeps_per_interval: int,
) -> tuple[MotionSequenceStep, ...]:
    pulse_amplitude = (0.5 * np.pi) / excitation_duration
    pulse_substeps = max(1, substeps_per_interval)

    def _pulse(label: str) -> MotionSequenceStep:
        return MotionSequenceStep(
            duration=excitation_duration,
            rf_amplitude=pulse_amplitude,
            rf_phase=np.pi / 2,
            substeps=pulse_substeps,
            label=label,
        )

    steps = [
        _pulse("excitation_90"),
        MotionSequenceStep(
            duration=gradient_duration,
            gradient=gradient,
            substeps=substeps_per_interval,
            label="ste_lobe_1",
        ),
    ]
    if encode_delay > 0.0:
        steps.append(
            MotionSequenceStep(
                duration=encode_delay,
                substeps=substeps_per_interval,
                label="ste_encode_delay_1",
            )
        )
    steps.append(_pulse("store_90"))
    # During storage the encoded magnetization is longitudinal (decays with T1);
    # the spoiler gradient dephases the residual transverse coherences.
    steps.append(
        MotionSequenceStep(
            duration=storage_time,
            gradient=spoiler,
            substeps=substeps_per_interval,
            label="storage",
        )
    )
    steps.append(_pulse("read_90"))
    if encode_delay > 0.0:
        steps.append(
            MotionSequenceStep(
                duration=encode_delay,
                substeps=substeps_per_interval,
                label="ste_encode_delay_2",
            )
        )
    steps.append(
        MotionSequenceStep(
            duration=gradient_duration,
            gradient=gradient,
            acquire=True,
            num_samples=1,
            substeps=substeps_per_interval,
            label="stimulated_echo",
        )
    )
    return tuple(steps)


def _make_dde_steps(
    *,
    gradient_duration: float,
    gradient_1: tuple[float, float],
    gradient_2: tuple[float, float],
    gap: float,
    mixing_time: float,
    excitation_duration: float,
    refocusing_duration: float,
    substeps_per_interval: int,
) -> tuple[MotionSequenceStep, ...]:
    sub = substeps_per_interval

    def _block(gradient: tuple[float, float], index: int, acquire: bool):
        return [
            MotionSequenceStep(
                duration=gradient_duration,
                gradient=gradient,
                substeps=sub,
                label=f"dde{index}_lobe_1",
            ),
            MotionSequenceStep(duration=gap, substeps=sub, label=f"dde{index}_gap_1"),
            MotionSequenceStep(
                duration=refocusing_duration,
                rf_amplitude=np.pi / refocusing_duration,
                rf_phase=0.0,
                substeps=max(1, sub),
                label=f"dde{index}_180",
            ),
            MotionSequenceStep(duration=gap, substeps=sub, label=f"dde{index}_gap_2"),
            MotionSequenceStep(
                duration=gradient_duration,
                gradient=gradient,
                acquire=acquire,
                num_samples=1 if acquire else 0,
                substeps=sub,
                label="dde_echo" if acquire else f"dde{index}_lobe_2",
            ),
        ]

    steps = [
        MotionSequenceStep(
            duration=excitation_duration,
            rf_amplitude=(0.5 * np.pi) / excitation_duration,
            rf_phase=np.pi / 2,
            substeps=max(1, sub),
            label="excitation_90",
        ),
    ]
    steps.extend(_block(gradient_1, 1, acquire=False))
    if mixing_time > 0.0:
        steps.append(
            MotionSequenceStep(duration=mixing_time, substeps=sub, label="mixing")
        )
    steps.extend(_block(gradient_2, 2, acquire=True))
    return tuple(steps)


def _make_ogse_steps(
    *,
    cosine: np.ndarray,
    step_dt: float,
    moment: float,
    gradient_axis: PGSEAxis,
    excitation_duration: float,
    refocusing_duration: float,
    substeps_per_interval: int,
) -> tuple[MotionSequenceStep, ...]:
    sub = substeps_per_interval

    def _lobe(acquire_last: bool) -> list[MotionSequenceStep]:
        last_index = cosine.size - 1
        lobe: list[MotionSequenceStep] = []
        for index, value in enumerate(cosine):
            acquire = acquire_last and index == last_index
            lobe.append(
                MotionSequenceStep(
                    duration=step_dt,
                    gradient=_gradient_tuple(moment * float(value), gradient_axis),
                    acquire=acquire,
                    num_samples=1 if acquire else 0,
                    substeps=sub,
                    label="ogse_echo" if acquire else "ogse_wave",
                )
            )
        return lobe

    steps = [
        MotionSequenceStep(
            duration=excitation_duration,
            rf_amplitude=(0.5 * np.pi) / excitation_duration,
            rf_phase=np.pi / 2,
            substeps=max(1, sub),
            label="excitation_90",
        ),
        *_lobe(acquire_last=False),
        MotionSequenceStep(
            duration=refocusing_duration,
            rf_amplitude=np.pi / refocusing_duration,
            rf_phase=0.0,
            substeps=max(1, sub),
            label="ogse_180",
        ),
        *_lobe(acquire_last=True),
    ]
    return tuple(steps)


def _gradient_vector(value: float, axis_index: int, ndim: int) -> tuple[float, ...]:
    """Return a length-``ndim`` gradient with ``value`` on ``axis_index``.

    The dimension-agnostic generalization of ``_gradient_tuple``: a unit gradient
    along one spatial axis, with zeros elsewhere. The motion engine couples it to
    moving spins as ``positions @ gradient``.
    """

    index = int(axis_index)
    if not 0 <= index < int(ndim):
        raise ValueError("axis_index must be in range(ndim)")
    vector = [0.0] * int(ndim)
    vector[index] = float(value)
    return tuple(vector)


def _gradient_tuple(value: float, axis: PGSEAxis) -> tuple[float, float]:
    # 2-D specialization over the (x, z) plane: "x" -> axis 0, "z" -> axis 1.
    if axis == "x":
        return _gradient_vector(value, 0, 2)  # type: ignore[return-value]
    if axis == "z":
        return _gradient_vector(value, 1, 2)  # type: ignore[return-value]
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
