"""Excitation-pulse phase optimization helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/opt_pulse/opt_exc_pulse_tuned.m
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import sim_spin_dynamics_asymp_mag3
from spin_dynamics.optimization._bounded import maximize_bounded, validate_bounds
from spin_dynamics.parameters import set_params_tuned_spa
from spin_dynamics.probes.tuned import tuned_probe_lp_orig, tuned_probe_rx


@dataclass(frozen=True)
class TunedExcitationEvaluation:
    """Non-plotting tuned-probe excitation-pulse evaluation."""

    del_w: np.ndarray
    mrx: np.ndarray
    masy: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float
    pulse_length_t180: float
    phases: np.ndarray
    neff: np.ndarray


@dataclass(frozen=True)
class TunedInverseExcitationEvaluation:
    """Evaluation of a tuned excitation pulse against an inversion target."""

    excitation: TunedExcitationEvaluation
    target_mrx: np.ndarray
    target_snr: float
    mismatch: float
    phases: np.ndarray
    neff: np.ndarray
    snr: float


ExcitationEvaluation = TunedExcitationEvaluation | TunedInverseExcitationEvaluation


@dataclass(frozen=True)
class ExcitationOptimizationResult:
    """Result of bounded fixed-amplitude excitation phase optimization."""

    probe: str
    initial_phases: np.ndarray
    best_phases: np.ndarray
    best_score: float
    initial_score: float
    best_evaluation: ExcitationEvaluation
    history_scores: np.ndarray
    history_phases: tuple[np.ndarray, ...]
    iterations: int
    improved: bool
    final_step: float
    bounds: tuple[float, float]
    optimizer_method: str
    optimizer_success: bool
    optimizer_message: str


ExcitationEvaluator = Callable[..., ExcitationEvaluation]


def _validate_neff(neff: np.ndarray | list[list[float]], numpts: int) -> np.ndarray:
    neff_arr = np.asarray(neff, dtype=np.complex128)
    if neff_arr.shape != (3, int(numpts)):
        raise ValueError("neff must have shape (3, numpts)")
    return neff_arr


def evaluate_tuned_excitation_pulse(
    phases: np.ndarray | list[float],
    neff: np.ndarray | list[list[float]],
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
) -> TunedExcitationEvaluation:
    """Evaluate a fixed-amplitude tuned-probe excitation phase program.

    This ports the non-plotting objective calculation in MATLAB
    `opt_pulse/opt_exc_pulse_tuned.m`. The refocusing rotation axis is supplied
    by the caller, matching MATLAB's `params.neff` contract.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")
    neff_arr = _validate_neff(neff, int(numpts))

    params, sp, pp = set_params_tuned_spa(numpts=numpts)
    texc_params = pp.T_180 * float(segment_fraction) * np.ones(
        phase_arr.size,
        dtype=np.float64,
    )
    aexc_params = np.ones(phase_arr.size, dtype=np.float64)
    params = params.__class__(
        **{
            **params.__dict__,
            "texc": texc_params,
            "pexc": phase_arr,
            "aexc": aexc_params,
        }
    )
    sp = sp.__class__(**{**sp.__dict__, "plt_axis": 0, "plt_tx": 0, "plt_rx": 0})

    pp_exc = pp.__class__(
        **{
            **pp.__dict__,
            "tref": np.concatenate([params.texc, [params.tqs, params.trd]]),
            "pref": np.concatenate([params.pexc, [0.0, 0.0]]),
            "aref": np.concatenate([params.aexc, [0.0, 0.0]]),
            "Rsref": np.concatenate(
                [
                    params.Rs[1] * np.ones(params.texc.size, dtype=np.float64),
                    [params.Rs[2], params.Rs[0]],
                ]
            ),
        }
    )
    tvect_probe, icr, _tvect_raw, _ic = tuned_probe_lp_orig(sp, pp_exc)
    if tvect_probe.size < 2:
        raise ValueError("excitation pulse is too short for probe response sampling")

    t_90 = pp.T_90
    b1max = (np.pi / 2) / (t_90 * sp.gamma)
    delt = (np.pi / 2) * (tvect_probe[1] - tvect_probe[0]) / t_90
    texc = delt * np.ones(tvect_probe.size, dtype=np.float64)
    pexc = np.arctan2(np.imag(icr), np.real(icr))
    aexc = np.abs(icr) * sp.sens / b1max
    aexc[aexc < pp.amp_zero] = 0
    pexc[aexc == 0] = 0

    tail = -(params.tqs + params.trd) * (np.pi / 2) / t_90
    texc = np.concatenate([texc, [tail]])
    pexc = np.concatenate([pexc, [0.0]])
    aexc = np.concatenate([aexc, [0.0]])

    tacq = float((np.pi / 2) * pp.tacq[0] / t_90)
    masy = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff_arr, sp.del_w, tacq)
    mrx, snr, _vsig = tuned_probe_rx(sp, pp, masy)
    echo, tvect = calc_time_domain_echo(mrx, sp.del_w)

    return TunedExcitationEvaluation(
        del_w=sp.del_w,
        mrx=mrx,
        masy=masy,
        echo=echo,
        tvect=tvect,
        snr=snr / 1e8,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
        neff=neff_arr,
    )


def evaluate_tuned_inverse_excitation_pulse(
    phases: np.ndarray | list[float],
    neff: np.ndarray | list[list[float]],
    target_mrx: np.ndarray | list[complex],
    target_snr: float,
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
) -> TunedInverseExcitationEvaluation:
    """Evaluate a tuned excitation pulse as an inverse phase-cycle partner.

    This mirrors the objective in MATLAB `opt_exc_pulse_tuned_inv.m`: minimize
    residual received magnetization after adding the target and candidate
    spectra, while keeping the candidate SNR close to the target SNR.
    """

    excitation = evaluate_tuned_excitation_pulse(
        phases,
        neff,
        segment_fraction=segment_fraction,
        numpts=numpts,
    )
    target = np.asarray(target_mrx, dtype=np.complex128).reshape(-1)
    if target.shape != excitation.mrx.shape:
        raise ValueError("target_mrx must have shape (numpts,)")
    if not np.isfinite(float(target_snr)):
        raise ValueError("target_snr must be finite")
    mismatch = float(
        trapezoid(np.abs(target + excitation.mrx), excitation.del_w)
        + 0.8 * abs(excitation.snr - float(target_snr))
    )
    return TunedInverseExcitationEvaluation(
        excitation=excitation,
        target_mrx=target,
        target_snr=float(target_snr),
        mismatch=mismatch,
        phases=excitation.phases,
        neff=excitation.neff,
        snr=excitation.snr,
    )


def _evaluate_score(
    evaluator: ExcitationEvaluator,
    phases: np.ndarray,
    neff: np.ndarray,
    *,
    segment_fraction: float,
    numpts: int,
) -> tuple[float, TunedExcitationEvaluation]:
    evaluation = evaluator(
        phases,
        neff,
        segment_fraction=segment_fraction,
        numpts=numpts,
    )
    score = float(evaluation.snr)
    if not np.isfinite(score):
        score = -np.inf
    return score, evaluation


def _evaluate_inverse_score(
    phases: np.ndarray,
    neff: np.ndarray,
    target_mrx: np.ndarray,
    target_snr: float,
    *,
    segment_fraction: float,
    numpts: int,
) -> tuple[float, TunedInverseExcitationEvaluation]:
    evaluation = evaluate_tuned_inverse_excitation_pulse(
        phases,
        neff,
        target_mrx,
        target_snr,
        segment_fraction=segment_fraction,
        numpts=numpts,
    )
    score = -float(evaluation.mismatch)
    if not np.isfinite(score):
        score = -np.inf
    return score, evaluation


def optimize_tuned_excitation_phases(
    initial_phases: np.ndarray | list[float],
    neff: np.ndarray | list[list[float]],
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> ExcitationOptimizationResult:
    """Optimize tuned-probe fixed-amplitude excitation phases."""

    lower, upper = validate_bounds(bounds)
    neff_arr = _validate_neff(neff, int(numpts))
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    def score_fn(phases: np.ndarray) -> float:
        score, _evaluation = _evaluate_score(
            evaluate_tuned_excitation_pulse,
            phases,
            neff_arr,
            segment_fraction=segment_fraction,
            numpts=numpts,
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
        evaluate_tuned_excitation_pulse,
        run.best_x,
        neff_arr,
        segment_fraction=segment_fraction,
        numpts=numpts,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )

    return ExcitationOptimizationResult(
        probe="tuned",
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


def optimize_tuned_inverse_excitation_phases(
    initial_phases: np.ndarray | list[float],
    neff: np.ndarray | list[list[float]],
    target_mrx: np.ndarray | list[complex],
    target_snr: float,
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    initial_step: float = np.pi / 2,
    step_decay: float = 0.5,
    min_step: float = 1e-3,
    max_passes: int = 8,
    optimizer: str = "auto",
    scipy_method: str = "L-BFGS-B",
    scipy_options: dict[str, object] | None = None,
) -> ExcitationOptimizationResult:
    """Optimize a tuned excitation pulse to invert a target received spectrum."""

    lower, upper = validate_bounds(bounds)
    neff_arr = _validate_neff(neff, int(numpts))
    target = np.asarray(target_mrx, dtype=np.complex128).reshape(-1)
    if target.shape != (int(numpts),):
        raise ValueError("target_mrx must have shape (numpts,)")
    if not np.isfinite(float(target_snr)):
        raise ValueError("target_snr must be finite")
    initial = np.asarray(initial_phases, dtype=np.float64).reshape(-1)
    if initial.size == 0:
        raise ValueError("initial_phases must not be empty")

    def score_fn(phases: np.ndarray) -> float:
        score, _evaluation = _evaluate_inverse_score(
            phases,
            neff_arr,
            target,
            float(target_snr),
            segment_fraction=segment_fraction,
            numpts=numpts,
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
    _best_score, best_evaluation = _evaluate_inverse_score(
        run.best_x,
        neff_arr,
        target,
        float(target_snr),
        segment_fraction=segment_fraction,
        numpts=numpts,
    )
    initial_score = (
        float(run.history_scores[0])
        if run.history_scores.size
        else float(run.best_score)
    )

    return ExcitationOptimizationResult(
        probe="tuned",
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
