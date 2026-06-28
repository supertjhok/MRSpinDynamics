"""Continuous refocusing-pulse phase optimization helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_tuned.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_untuned.m
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_ref_pulse_matched.m
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    calc_rotation_matrix,
    calc_rot_axis_arba4,
    calc_v0crit,
    sim_spin_dynamics_exc,
)
from spin_dynamics.optimization._bounded import maximize_bounded, validate_bounds
from spin_dynamics.optimization.spa import (
    MatchedRefocusingEvaluation,
    TunedRefocusingEvaluation,
    UntunedRefocusingEvaluation,
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
)
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.acquisition import calc_macq_ideal_probe_relax4
from spin_dynamics.workflows.time_varying import sinusoidal_field_waveform


@dataclass(frozen=True)
class IdealV0CritRefocusingEvaluation:
    """Ideal-probe refocusing evaluation for the v0crit objective."""

    del_w: np.ndarray
    neff: np.ndarray
    alpha: np.ndarray
    v0crit: np.ndarray
    masy: np.ndarray
    axis_rms: float
    v0crit_average: float
    score: float
    pulse_length_t180: float
    phases: np.ndarray

    @property
    def snr(self) -> float:
        """Compatibility alias for optimizer result ranking."""

        return self.score


@dataclass(frozen=True)
class IdealTimeVaryingRefocusingEvaluation:
    """Ideal-probe refocusing evaluation for time-varying B0 fields."""

    del_w: np.ndarray
    field_offsets: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    reference_echo: np.ndarray
    tvect: np.ndarray
    matched_signal: complex
    score: float
    pulse_length_t180: float
    phases: np.ndarray

    @property
    def snr(self) -> float:
        """Compatibility alias for optimizer result ranking."""

        return self.score


RefocusingEvaluation = (
    TunedRefocusingEvaluation
    | UntunedRefocusingEvaluation
    | MatchedRefocusingEvaluation
    | IdealV0CritRefocusingEvaluation
    | IdealTimeVaryingRefocusingEvaluation
)
RefocusingEvaluator = Callable[..., RefocusingEvaluation]


@dataclass(frozen=True)
class RefocusingOptimizationResult:
    """Result of bounded fixed-amplitude refocusing phase optimization."""

    probe: str
    initial_phases: np.ndarray
    best_phases: np.ndarray
    best_score: float
    initial_score: float
    best_evaluation: RefocusingEvaluation
    history_scores: np.ndarray
    history_phases: tuple[np.ndarray, ...]
    iterations: int
    improved: bool
    final_step: float
    bounds: tuple[float, float]
    optimizer_method: str
    optimizer_success: bool
    optimizer_message: str


def _evaluate_score(
    evaluator: RefocusingEvaluator,
    phases: np.ndarray,
    *,
    segment_fraction: float,
    numpts: int,
    excitation_amplitude: float,
) -> tuple[float, RefocusingEvaluation]:
    evaluation = evaluator(
        phases,
        segment_fraction=segment_fraction,
        numpts=numpts,
        excitation_amplitude=excitation_amplitude,
    )
    score = float(evaluation.snr)
    if not np.isfinite(score):
        score = -np.inf
    return score, evaluation


def _normalized_sinc_window(del_w: np.ndarray, tacq: float) -> np.ndarray:
    window = np.sinc(del_w * float(tacq) / (2 * np.pi))
    total = np.sum(window)
    if total == 0:
        raise ValueError("acquisition window normalization is zero")
    return window / total


def ideal_time_varying_excitation_vector(
    *,
    numpts: int = 101,
    maxoffs: float = 4.0,
    pulse_times: np.ndarray | list[float] | None = None,
    pulse_phases: np.ndarray | list[float] | None = None,
    pulse_amplitudes: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """Return the ideal excitation vector used by v0crit-excitation searches.

    This is the non-plotting excitation-preparation part of MATLAB
    `opt_ref_pulse_ideal_v0crit_exc_repeat.m`, using the excitation pulse from
    `Params/set_params_ideal_tv_exc.m`.
    """

    if numpts < 2:
        raise ValueError("numpts must be at least 2")
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    tp = (
        np.array([np.pi / 4, -0.5], dtype=np.float64)
        if pulse_times is None
        else np.asarray(pulse_times, dtype=np.float64).reshape(-1)
    )
    phi = (
        np.array([np.pi / 2, 0.0], dtype=np.float64)
        if pulse_phases is None
        else np.asarray(pulse_phases, dtype=np.float64).reshape(-1)
    )
    amp = (
        np.array([2.0, 0.0], dtype=np.float64)
        if pulse_amplitudes is None
        else np.asarray(pulse_amplitudes, dtype=np.float64).reshape(-1)
    )
    return sim_spin_dynamics_exc(tp, phi, amp, del_w)


def evaluate_ideal_v0crit_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    free_precession_t180: float = 1.5,
    numpts: int = 101,
    maxoffs: float | None = None,
    acquisition_time_normalized: float | None = None,
    excitation_vector: np.ndarray | list[list[complex]] | None = None,
    v0crit_weight: float = 100.0,
) -> IdealV0CritRefocusingEvaluation:
    """Evaluate the ideal-probe refocusing objective used by v0crit workflows.

    This ports the array-returning objective core of MATLAB
    `opt_pulse/opt_ref_pulse_ideal_v0crit.m` and
    `opt_pulse/opt_ref_pulse_ideal_v0crit_exc.m`. Segment durations and free
    precession are expressed in units of an ideal rectangular 180-degree pulse.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0 or free_precession_t180 < 0:
        raise ValueError(
            "segment_fraction must be positive and free precession non-negative"
        )
    if numpts < 2:
        raise ValueError("numpts must be at least 2")
    if v0crit_weight <= 0:
        raise ValueError("v0crit_weight must be positive")

    sp, pp = set_params_ideal(numpts=numpts)
    del_w = np.linspace(
        -float(sp.maxoffs if maxoffs is None else maxoffs),
        float(sp.maxoffs if maxoffs is None else maxoffs),
        int(numpts),
    )
    t180_normalized = np.pi
    tfp = float(free_precession_t180) * t180_normalized
    segment_length = float(segment_fraction) * t180_normalized
    tp = np.concatenate(
        [
            [tfp],
            segment_length * np.ones(phase_arr.size, dtype=np.float64),
            [tfp],
        ]
    )
    phi = np.concatenate([[0.0], phase_arr, [0.0]])
    amp = np.concatenate([[0.0], np.ones(phase_arr.size, dtype=np.float64), [0.0]])
    neff, alpha = calc_rot_axis_arba4(tp, phi, amp, del_w)
    v0crit = calc_v0crit(del_w, neff, alpha)

    if acquisition_time_normalized is None:
        w_1n = (np.pi / 2) / pp.T_90
        tacq = float(w_1n * pp.tacq[0])
    else:
        tacq = float(acquisition_time_normalized)
    window = _normalized_sinc_window(del_w, tacq)

    transverse_axis = neff[0, :] + 1j * neff[1, :]
    if excitation_vector is None:
        masy_raw = transverse_axis
    else:
        mexc = np.asarray(excitation_vector, dtype=np.complex128)
        if mexc.shape != neff.shape:
            raise ValueError("excitation_vector must have shape (3, numpts)")
        masy_raw = np.sum(np.conj(mexc) * neff, axis=0) * transverse_axis
    masy = np.convolve(masy_raw, window, mode="same")
    axis_rms = float(np.real(trapezoid(np.abs(masy) ** 2, del_w)))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_v0crit = 1.0 / v0crit
    v0crit_denominator = float(np.real(trapezoid(inv_v0crit, del_w)))
    if v0crit_denominator == 0 or not np.isfinite(v0crit_denominator):
        v0crit_average = 0.0
    else:
        v0crit_average = float(v0crit_weight / v0crit_denominator)
    score = axis_rms + v0crit_average

    return IdealV0CritRefocusingEvaluation(
        del_w=del_w,
        neff=neff,
        alpha=alpha,
        v0crit=v0crit,
        masy=masy,
        axis_rms=axis_rms,
        v0crit_average=v0crit_average,
        score=score,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
    )


def evaluate_ideal_v0crit_excited_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    free_precession_t180: float = 1.5,
    numpts: int = 101,
    maxoffs: float = 4.0,
    acquisition_time_normalized: float | None = None,
    excitation_vector: np.ndarray | list[list[complex]] | None = None,
    v0crit_weight: float = 100.0,
) -> IdealV0CritRefocusingEvaluation:
    """Evaluate ideal v0crit refocusing with a supplied excitation spectrum.

    This mirrors the objective core in MATLAB
    `opt_pulse/opt_ref_pulse_ideal_v0crit_exc.m`. When `excitation_vector` is
    omitted, the default excitation pulse from `set_params_ideal_tv_exc.m` is
    simulated on the same offset grid.
    """

    mexc = (
        ideal_time_varying_excitation_vector(numpts=numpts, maxoffs=maxoffs)
        if excitation_vector is None
        else excitation_vector
    )
    return evaluate_ideal_v0crit_refocusing_pulse(
        phases,
        segment_fraction=segment_fraction,
        free_precession_t180=free_precession_t180,
        numpts=numpts,
        maxoffs=maxoffs,
        acquisition_time_normalized=acquisition_time_normalized,
        excitation_vector=mexc,
        v0crit_weight=v0crit_weight,
    )


def _simulate_ideal_time_varying_refocusing(
    phases: np.ndarray,
    field_offsets: np.ndarray,
    *,
    segment_fraction: float,
    echo_spacing_t180: float,
    numpts: int,
    maxoffs: float,
    t1_seconds: float,
    t2_seconds: float,
    num_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]:
    sp0, pp0 = set_params_ideal(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    w1n = (np.pi / 2) / pp0.T_90
    t180_seconds = pp0.T_180
    pulse_length_t180 = float(segment_fraction) * phases.size
    echo_spacing_seconds = float(echo_spacing_t180) * t180_seconds
    ref_duration_seconds = pulse_length_t180 * t180_seconds
    if ref_duration_seconds > echo_spacing_seconds:
        raise ValueError("refocusing pulse is longer than the echo spacing")
    ref_pre = 0.5 * (echo_spacing_seconds - ref_duration_seconds)
    ref_post = echo_spacing_seconds - ref_duration_seconds - ref_pre
    ref_tp_seconds = (
        float(segment_fraction) * t180_seconds * np.ones(phases.size, dtype=np.float64)
    )
    ref_tp_norm = w1n * ref_tp_seconds
    ref_amp = np.ones(phases.size, dtype=np.float64)

    rtot = [
        calc_rotation_matrix(del_w, np.ones_like(del_w), w1n * pp0.texc, pp0.pexc, pp0.aexc),
        calc_rotation_matrix(
            del_w,
            np.ones_like(del_w),
            w1n * pp0.texc,
            pp0.pexc + np.pi,
            pp0.aexc,
        ),
    ]
    for offset in field_offsets:
        rtot.append(
            calc_rotation_matrix(
                del_w + offset,
                np.ones_like(del_w),
                ref_tp_norm,
                phases,
                ref_amp,
            )
        )

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    num_echoes = field_offsets.size
    tref = np.empty(3 * num_echoes, dtype=np.float64)
    pref = np.empty(3 * num_echoes, dtype=np.int64)
    aref = np.empty(3 * num_echoes, dtype=np.float64)
    acq_ref = np.zeros(3 * num_echoes, dtype=np.int64)
    grad_ref = np.empty(3 * num_echoes, dtype=np.float64)
    for idx, offset in enumerate(field_offsets):
        base = 3 * idx
        tref[base : base + 3] = [w1n * ref_pre, np.pi, w1n * ref_post]
        pref[base : base + 3] = [0, idx + 3, 0]
        aref[base : base + 3] = [0.0, 1.0, 0.0]
        grad_ref[base : base + 3] = offset
    acq_ref[-1] = 1

    sp = {
        "del_w": del_w,
        "del_wg": np.ones_like(del_w),
        "w_1": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
    }
    pp_common = {
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, pref])}
    mrx1 = calc_macq_ideal_probe_relax4(sp, pp1, num_workers=num_workers)
    mrx2 = calc_macq_ideal_probe_relax4(sp, pp2, num_workers=num_workers)
    mrx = ((mrx1 - mrx2) / 2)[0]

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(
        1j * tvect[:, np.newaxis] * (del_w + field_offsets[-1])[np.newaxis, :]
    )
    echo = isoc @ mrx
    echo_integral = complex(trapezoid(echo, tvect))
    return del_w, mrx, echo, tvect, echo_integral


def evaluate_ideal_time_varying_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    echo_spacing_t180: float = 4.0,
    field_offsets: np.ndarray | list[float] | None = None,
    fluctuation_amplitude: float = 1.5,
    num_echoes: int = 16,
    numpts: int = 101,
    maxoffs: float = 10.0,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    score_scale: float = 1e4,
    num_workers: int | None = 1,
) -> IdealTimeVaryingRefocusingEvaluation:
    """Evaluate ideal refocusing phases for time-varying-field robustness.

    This ports the non-plotting objective shape of MATLAB
    `opt_pulse/opt_ref_pulse_ideal_tv.m`: simulate the final echo under a
    time-varying B0 waveform and score it with a zero-fluctuation matched
    filter built from the same refocusing pulse.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")
    if echo_spacing_t180 <= 0:
        raise ValueError("echo_spacing_t180 must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")
    if score_scale <= 0:
        raise ValueError("score_scale must be positive")
    if field_offsets is None:
        offsets = float(fluctuation_amplitude) * sinusoidal_field_waveform(num_echoes)
    else:
        offsets = np.asarray(field_offsets, dtype=np.float64).reshape(-1)
    if offsets.size == 0:
        raise ValueError("field_offsets must not be empty")

    del_w, mrx, echo, tvect, _echo_integral = _simulate_ideal_time_varying_refocusing(
        phase_arr,
        offsets,
        segment_fraction=segment_fraction,
        echo_spacing_t180=echo_spacing_t180,
        numpts=numpts,
        maxoffs=maxoffs,
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
        num_workers=num_workers,
    )
    _ref_del_w, _ref_mrx, reference_echo, ref_tvect, _ref_integral = (
        _simulate_ideal_time_varying_refocusing(
            phase_arr,
            np.zeros_like(offsets),
            segment_fraction=segment_fraction,
            echo_spacing_t180=echo_spacing_t180,
            numpts=numpts,
            maxoffs=maxoffs,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            num_workers=num_workers,
        )
    )
    if ref_tvect.shape != tvect.shape or np.any(ref_tvect != tvect):
        raise RuntimeError("time-varying reference echo grid does not match candidate")
    norm = np.sqrt(trapezoid(np.abs(reference_echo) ** 2, tvect))
    if norm == 0 or not np.isfinite(norm):
        matched_signal = complex(np.nan)
        score = -np.inf
    else:
        matched_filter = np.conj(reference_echo) / norm
        matched_signal = complex(trapezoid(echo * matched_filter, tvect))
        score = float(np.real(matched_signal) / float(score_scale))
        if not np.isfinite(score):
            score = -np.inf

    return IdealTimeVaryingRefocusingEvaluation(
        del_w=del_w,
        field_offsets=offsets,
        mrx=mrx,
        echo=echo,
        reference_echo=reference_echo,
        tvect=tvect,
        matched_signal=matched_signal,
        score=score,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
    )


def _optimize_refocusing_phase_program(
    probe: str,
    initial_phases: np.ndarray | list[float],
    evaluator: RefocusingEvaluator,
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    excitation_amplitude: float = 6.0,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> RefocusingOptimizationResult:
    """Optimize fixed-amplitude refocusing phases with a bounded optimizer."""

    lower, upper = validate_bounds(bounds)
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    def score_fn(phases: np.ndarray) -> float:
        score, _evaluation = _evaluate_score(
            evaluator,
            phases,
            segment_fraction=segment_fraction,
            numpts=numpts,
            excitation_amplitude=excitation_amplitude,
        )
        return score

    run = maximize_bounded(
        score_fn,
        initial,
        bounds=(lower, upper),
        optimizer=optimizer,
        initial_step=initial_step,
        step_decay=step_decay,
        min_step=min_step,
        max_passes=max_passes,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
    )
    _best_score, best_evaluation = _evaluate_score(
        evaluator,
        run.best_x,
        segment_fraction=segment_fraction,
        numpts=numpts,
        excitation_amplitude=excitation_amplitude,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )

    return RefocusingOptimizationResult(
        probe=probe,
        initial_phases=initial,
        best_phases=run.best_x,
        best_score=run.best_score,
        initial_score=initial_score,
        best_evaluation=best_evaluation,
        history_scores=run.history_scores,
        history_phases=run.history_x,
        iterations=run.iterations,
        improved=run.improved,
        final_step=run.final_step,
        bounds=(lower, upper),
        optimizer_method=run.method,
        optimizer_success=run.success,
        optimizer_message=run.message,
    )


def optimize_tuned_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize tuned-probe fixed-amplitude refocusing phases."""

    return _optimize_refocusing_phase_program(
        "tuned",
        initial_phases,
        evaluate_tuned_refocusing_pulse,
        **kwargs,
    )


def optimize_untuned_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize untuned-probe fixed-amplitude refocusing phases."""

    return _optimize_refocusing_phase_program(
        "untuned",
        initial_phases,
        evaluate_untuned_refocusing_pulse,
        **kwargs,
    )


def optimize_matched_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    **kwargs: object,
) -> RefocusingOptimizationResult:
    """Optimize matched-probe fixed-amplitude refocusing phases.

    The matched transient solver is slower than the tuned and untuned paths, so
    the default grid and pass count are intentionally conservative.
    """

    options = {"numpts": 21, "max_passes": 3}
    options.update(kwargs)
    return _optimize_refocusing_phase_program(
        "matched",
        initial_phases,
        evaluate_matched_refocusing_pulse,
        **options,
    )


def optimize_ideal_v0crit_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    free_precession_t180: float = 1.5,
    numpts: int = 101,
    maxoffs: float | None = None,
    acquisition_time_normalized: float | None = None,
    excitation_vector: np.ndarray | list[list[complex]] | None = None,
    v0crit_weight: float = 100.0,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> RefocusingOptimizationResult:
    """Optimize ideal-probe phases for the v0crit refocusing objective."""

    lower, upper = validate_bounds(bounds)
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    if optimizer == "jax":
        if excitation_vector is not None:
            raise ValueError(
                "optimizer='jax' currently supports excitation_vector=None"
            )
        from spin_dynamics.optimization._bounded import scipy_maximize_with_grad
        from spin_dynamics.optimization._jax_objectives import (
            make_ideal_v0crit_objective,
        )

        value_and_grad = make_ideal_v0crit_objective(
            initial.size,
            segment_fraction=segment_fraction,
            free_precession_t180=free_precession_t180,
            numpts=numpts,
            maxoffs=maxoffs,
            acquisition_time_normalized=acquisition_time_normalized,
            v0crit_weight=v0crit_weight,
        )
        run = scipy_maximize_with_grad(
            value_and_grad,
            initial,
            bounds=(lower, upper),
            scipy_method=scipy_method,
            options=scipy_options,
        )
    else:
        def score_fn(phases: np.ndarray) -> float:
            evaluation = evaluate_ideal_v0crit_refocusing_pulse(
                phases,
                segment_fraction=segment_fraction,
                free_precession_t180=free_precession_t180,
                numpts=numpts,
                maxoffs=maxoffs,
                acquisition_time_normalized=acquisition_time_normalized,
                excitation_vector=excitation_vector,
                v0crit_weight=v0crit_weight,
            )
            score = float(evaluation.score)
            return score if np.isfinite(score) else -np.inf

        run = maximize_bounded(
            score_fn,
            initial,
            bounds=(lower, upper),
            optimizer=optimizer,
            initial_step=initial_step,
            step_decay=step_decay,
            min_step=min_step,
            max_passes=max_passes,
            scipy_method=scipy_method,
            scipy_options=scipy_options,
        )
    best_evaluation = evaluate_ideal_v0crit_refocusing_pulse(
        run.best_x,
        segment_fraction=segment_fraction,
        free_precession_t180=free_precession_t180,
        numpts=numpts,
        maxoffs=maxoffs,
        acquisition_time_normalized=acquisition_time_normalized,
        excitation_vector=excitation_vector,
        v0crit_weight=v0crit_weight,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )
    return RefocusingOptimizationResult(
        probe="ideal_v0crit",
        initial_phases=initial,
        best_phases=run.best_x,
        best_score=run.best_score,
        initial_score=initial_score,
        best_evaluation=best_evaluation,
        history_scores=run.history_scores,
        history_phases=run.history_x,
        iterations=run.iterations,
        improved=run.improved,
        final_step=run.final_step,
        bounds=(lower, upper),
        optimizer_method=run.method,
        optimizer_success=run.success,
        optimizer_message=run.message,
    )


def optimize_ideal_v0crit_excited_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    free_precession_t180: float = 1.5,
    numpts: int = 101,
    maxoffs: float = 4.0,
    acquisition_time_normalized: float | None = None,
    excitation_vector: np.ndarray | list[list[complex]] | None = None,
    v0crit_weight: float = 100.0,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> RefocusingOptimizationResult:
    """Optimize ideal v0crit refocusing phases after a fixed excitation pulse."""

    mexc = (
        ideal_time_varying_excitation_vector(numpts=numpts, maxoffs=maxoffs)
        if excitation_vector is None
        else np.asarray(excitation_vector, dtype=np.complex128)
    )
    result = optimize_ideal_v0crit_refocusing_phases(
        initial_phases,
        segment_fraction=segment_fraction,
        free_precession_t180=free_precession_t180,
        numpts=numpts,
        maxoffs=maxoffs,
        acquisition_time_normalized=acquisition_time_normalized,
        excitation_vector=mexc,
        v0crit_weight=v0crit_weight,
        bounds=bounds,
        initial_step=initial_step,
        step_decay=step_decay,
        min_step=min_step,
        max_passes=max_passes,
        optimizer=optimizer,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
    )
    return replace(result, probe="ideal_v0crit_excited")


def optimize_ideal_time_varying_refocusing_phases(
    initial_phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    echo_spacing_t180: float = 4.0,
    field_offsets: np.ndarray | list[float] | None = None,
    fluctuation_amplitude: float = 1.5,
    num_echoes: int = 16,
    numpts: int = 101,
    maxoffs: float = 10.0,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    score_scale: float = 1e4,
    num_workers: int | None = 1,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> RefocusingOptimizationResult:
    """Optimize ideal refocusing phases for time-varying-field robustness."""

    lower, upper = validate_bounds(bounds)
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    def score_fn(phases: np.ndarray) -> float:
        evaluation = evaluate_ideal_time_varying_refocusing_pulse(
            phases,
            segment_fraction=segment_fraction,
            echo_spacing_t180=echo_spacing_t180,
            field_offsets=field_offsets,
            fluctuation_amplitude=fluctuation_amplitude,
            num_echoes=num_echoes,
            numpts=numpts,
            maxoffs=maxoffs,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            score_scale=score_scale,
            num_workers=num_workers,
        )
        score = float(evaluation.score)
        return score if np.isfinite(score) else -np.inf

    run = maximize_bounded(
        score_fn,
        initial,
        bounds=(lower, upper),
        optimizer=optimizer,
        initial_step=initial_step,
        step_decay=step_decay,
        min_step=min_step,
        max_passes=max_passes,
        scipy_method=scipy_method,
        scipy_options=scipy_options,
    )
    best_evaluation = evaluate_ideal_time_varying_refocusing_pulse(
        run.best_x,
        segment_fraction=segment_fraction,
        echo_spacing_t180=echo_spacing_t180,
        field_offsets=field_offsets,
        fluctuation_amplitude=fluctuation_amplitude,
        num_echoes=num_echoes,
        numpts=numpts,
        maxoffs=maxoffs,
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
        score_scale=score_scale,
        num_workers=num_workers,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )
    return RefocusingOptimizationResult(
        probe="ideal_time_varying",
        initial_phases=initial,
        best_phases=run.best_x,
        best_score=run.best_score,
        initial_score=initial_score,
        best_evaluation=best_evaluation,
        history_scores=run.history_scores,
        history_phases=run.history_x,
        iterations=run.iterations,
        improved=run.improved,
        final_step=run.final_step,
        bounds=(lower, upper),
        optimizer_method=run.method,
        optimizer_success=run.success,
        optimizer_message=run.message,
    )
