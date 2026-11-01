"""End-to-end helpers for OCT-style optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.numerics import trapezoid
import spin_dynamics.optimization.drivers as drivers
from spin_dynamics.optimization.results import (
    PulseProgram,
    SelectedOptimizationProgram,
    select_matlab_result_program,
)
from spin_dynamics.parameters import set_params_tuned_spa
from spin_dynamics.probes.tuned import calc_rot_axis_tuned_probe_lp_orig2


@dataclass(frozen=True)
class InverseExcitationValidation:
    """Quantitative cancellation evidence for one excitation/inverse pair."""

    target_integral: float
    residual_integral: float
    residual_ratio: float
    peak_residual_ratio: float
    inverse_coherence: float
    snr_relative_error: float | None


def validate_inverse_excitation_pair(
    target_mrx: np.ndarray,
    inverse_mrx: np.ndarray,
    del_w: np.ndarray,
    *,
    target_snr: float | None = None,
    inverse_snr: float | None = None,
) -> InverseExcitationValidation:
    """Measure broadband cancellation, peak residual, phase, and SNR balance.

    ``inverse_coherence`` is one for a spectrum proportional to ``-target`` and
    zero for an orthogonal spectrum. This is analytical/internal validation;
    callers should attach external fixture provenance separately when available.
    """

    target = np.asarray(target_mrx, dtype=np.complex128).reshape(-1)
    inverse = np.asarray(inverse_mrx, dtype=np.complex128).reshape(-1)
    offsets = np.asarray(del_w, dtype=np.float64).reshape(-1)
    if target.shape != inverse.shape or target.shape != offsets.shape:
        raise ValueError("target_mrx, inverse_mrx, and del_w must have matching shapes")
    if target.size < 2 or not np.all(np.diff(offsets) > 0.0):
        raise ValueError("del_w must be strictly increasing with at least two points")
    target_integral = float(trapezoid(np.abs(target), offsets))
    if target_integral <= 0.0:
        raise ValueError("target_mrx must have non-zero integrated magnitude")
    residual = target + inverse
    residual_integral = float(trapezoid(np.abs(residual), offsets))
    target_peak = float(np.max(np.abs(target)))
    target_norm = float(np.linalg.norm(target))
    inverse_norm = float(np.linalg.norm(inverse))
    coherence = (
        0.0
        if inverse_norm == 0.0
        else float(np.real(np.vdot(-target, inverse)) / (target_norm * inverse_norm))
    )
    snr_error = None
    if target_snr is not None or inverse_snr is not None:
        if target_snr is None or inverse_snr is None:
            raise ValueError("target_snr and inverse_snr must be supplied together")
        target_snr_value = float(target_snr)
        inverse_snr_value = float(inverse_snr)
        if not np.isfinite(target_snr_value) or not np.isfinite(inverse_snr_value):
            raise ValueError("SNR values must be finite")
        snr_error = abs(inverse_snr_value - target_snr_value) / max(
            abs(target_snr_value), np.finfo(float).tiny
        )
    return InverseExcitationValidation(
        target_integral=target_integral,
        residual_integral=residual_integral,
        residual_ratio=residual_integral / target_integral,
        peak_residual_ratio=float(np.max(np.abs(residual))) / target_peak,
        inverse_coherence=coherence,
        snr_relative_error=snr_error,
    )


@dataclass(frozen=True)
class TunedExcitationInversePipelineResult:
    """Selected-refocusing to excitation/inverse-excitation pipeline result."""

    del_w: np.ndarray
    neff: np.ndarray
    selected_refocusing: SelectedOptimizationProgram | None
    excitation: drivers.MultiStartOptimizationResult
    inverse: drivers.MultiStartOptimizationResult
    inverse_validations: tuple[InverseExcitationValidation, ...]
    inverse_residual_ratios: np.ndarray
    residual_best_index: int

    @property
    def residual_best_result(self) -> Any:
        return self.inverse.results[self.residual_best_index]

    @property
    def residual_best_ratio(self) -> float:
        return float(self.inverse_residual_ratios[self.residual_best_index])


def _result_at_pulse_number(refocusing: Any, pulse_number: int | None) -> Any:
    if pulse_number is None:
        return getattr(refocusing, "best_result")
    results = tuple(getattr(refocusing, "results"))
    index = int(pulse_number) - 1
    if index < 0 or index >= len(results):
        raise ValueError("pulse_number is outside the available result range")
    return results[index]


def _tuned_refocusing_axis(
    times_t180: np.ndarray,
    phases: np.ndarray,
    amplitudes: np.ndarray,
    *,
    numpts: int,
    free_precession_t180: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tuned-probe refocusing-cycle axis, as MATLAB `opt_exc_pulse_tuned_repeat.m`.

    The refocusing segments are played through the tuned SPA probe model with
    the ``tfp`` free precession before and after the pulse
    (:func:`spin_dynamics.probes.tuned.calc_rot_axis_tuned_probe_lp_orig2`), so
    the axis includes the probe transient and the free precession exactly as
    the tuned refocusing evaluator and the tuned excitation evaluator see it.
    ``free_precession_t180=None`` uses the tuned SPA default ``tfp``.
    """

    params, sp, pp = set_params_tuned_spa(numpts=int(numpts))
    tfp = (
        float(params.tfp)
        if free_precession_t180 is None
        else float(free_precession_t180) * float(pp.T_180)
    )
    if not np.isfinite(tfp) or tfp < float(params.tqs):
        raise ValueError(
            "refocusing free precession must be finite and at least the Q-switch delay"
        )
    params = params.__class__(
        **{
            **params.__dict__,
            "tref": np.asarray(times_t180, dtype=np.float64) * float(pp.T_180),
            "pref": np.asarray(phases, dtype=np.float64),
            "aref": np.asarray(amplitudes, dtype=np.float64),
            "tfp": tfp,
        }
    )
    neff = calc_rot_axis_tuned_probe_lp_orig2(params, sp, pp)
    return np.asarray(sp.del_w, dtype=np.float64).copy(), np.asarray(neff)


def _axis_from_program(
    program: PulseProgram,
    *,
    numpts: int,
    times_are_t180: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(program.times, dtype=np.float64).reshape(-1)
    phases = np.asarray(program.phases, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(program.amplitudes, dtype=np.float64).reshape(-1)
    if not (times.size == phases.size == amplitudes.size):
        raise ValueError("refocusing pulse program arrays must have matching lengths")
    if times.size == 0:
        raise ValueError("refocusing pulse program must not be empty")
    times_t180 = times if times_are_t180 else times / np.pi
    # MATLAB-style refocusing cells store the cycle as [tfp, pulse..., tfp]
    # with zero amplitude on the free-precession padding. Strip that padding
    # and pass it as the tuned-probe tfp; cells without padding use the
    # tuned SPA default free precession.
    free_precession_t180 = None
    if times.size >= 3 and amplitudes[0] == 0.0 and amplitudes[-1] == 0.0:
        if not np.isclose(times_t180[0], times_t180[-1]):
            raise ValueError(
                "refocusing cell free-precession padding must be symmetric"
            )
        free_precession_t180 = float(times_t180[0])
        times_t180 = times_t180[1:-1]
        phases = phases[1:-1]
        amplitudes = amplitudes[1:-1]
    return _tuned_refocusing_axis(
        times_t180,
        phases,
        amplitudes,
        numpts=numpts,
        free_precession_t180=free_precession_t180,
    )


def _resolve_refocusing_axis(
    refocusing: Any,
    *,
    pulse_number: int | None,
    numpts: int,
    result_times_are_t180: bool,
) -> tuple[np.ndarray, np.ndarray, SelectedOptimizationProgram | None]:
    if hasattr(refocusing, "best_result") and hasattr(refocusing, "results"):
        result = _result_at_pulse_number(refocusing, pulse_number)
        phases = np.asarray(getattr(result, "best_phases"), dtype=np.float64).reshape(-1)
        if phases.size == 0:
            raise ValueError("refocusing result phases must not be empty")
        evaluation = getattr(result, "best_evaluation", None)
        pulse_length = getattr(evaluation, "pulse_length_t180", None)
        segment_fraction = (
            0.1 if pulse_length is None else float(pulse_length) / phases.size
        )
        # A stored evaluation axis (e.g. an ideal-probe v0crit ``neff``) is not
        # the axis the tuned excitation evaluator sees; always recompute it
        # through the tuned probe with the tuned SPA free precession.
        del_w, neff = _tuned_refocusing_axis(
            segment_fraction * np.ones(phases.size, dtype=np.float64),
            phases,
            np.ones(phases.size, dtype=np.float64),
            numpts=numpts,
            free_precession_t180=None,
        )
        return del_w, neff, None

    selected = select_matlab_result_program(refocusing, pulse_number=pulse_number)
    if selected.refocusing is None:
        raise ValueError("selected optimization result does not contain refocusing data")
    del_w, neff = _axis_from_program(
        selected.refocusing,
        numpts=numpts,
        times_are_t180=result_times_are_t180,
    )
    return del_w, neff, selected


def _validate_pipeline_pair(target: Any, inverse_result: Any) -> InverseExcitationValidation:
    inverse_eval = inverse_result.best_evaluation.excitation
    has_snr = hasattr(target, "snr") and hasattr(inverse_eval, "snr")
    return validate_inverse_excitation_pair(
        target.mrx,
        inverse_eval.mrx,
        target.del_w,
        target_snr=target.snr if has_snr else None,
        inverse_snr=inverse_eval.snr if has_snr else None,
    )


def run_tuned_excitation_inverse_pipeline(
    refocusing: Any,
    *,
    pulse_number: int | None = None,
    excitation_segments: int = 3,
    excitation_starts: int = 4,
    inverse_starts: int = 4,
    seed: int | None = None,
    numpts: int = 21,
    maxoffs: float | None = None,
    result_times_are_t180: bool = True,
    random_fraction: float = 0.3,
    excitation_kwargs: dict[str, Any] | None = None,
    inverse_kwargs: dict[str, Any] | None = None,
) -> TunedExcitationInversePipelineResult:
    """Run excitation and inverse-excitation searches from a refocusing result.

    `refocusing` may be a Python multi-start result or MATLAB-style result
    cells. This is the plotting-free workflow handoff used by MATLAB's
    `opt_exc_pulse_tuned_repeat.m`: select a refocusing pulse, derive its
    effective axis, optimize a target excitation pulse, then run the
    phase-flipped inverse-excitation multi-start search.

    The refocusing axis ``neff`` is always the tuned-probe axis of the selected
    refocusing cycle (:func:`~spin_dynamics.probes.tuned.calc_rot_axis_tuned_probe_lp_orig2`):
    the pulse segments are played through the tuned SPA probe model with the
    ``tfp`` free precession before and after, as MATLAB does. MATLAB-style
    cells supply ``tfp`` from their zero-amplitude ``[tfp, pulse, tfp]``
    padding (in T180 units when ``result_times_are_t180``, otherwise in
    ``w1 = 1`` normalized time); Python results and unpadded cells use the
    tuned SPA default. A stored evaluation ``neff`` (e.g. an ideal-probe
    v0crit axis) is not reused.

    The axis and the tuned excitation evaluators share the fixed tuned SPA
    offset grid (``maxoffs = 10``). ``maxoffs`` is accepted only as a
    consistency check against that grid; any other value raises
    ``ValueError`` rather than silently misaligning the axis and the spectra.
    """

    if maxoffs is not None:
        _params, tuned_sp, _pp = set_params_tuned_spa(numpts=int(numpts))
        if not np.isclose(float(maxoffs), float(tuned_sp.maxoffs)):
            raise ValueError(
                "maxoffs must match the tuned excitation evaluator offset grid "
                f"(maxoffs={float(tuned_sp.maxoffs):g}); got {float(maxoffs):g}"
            )
    del_w, neff, selected = _resolve_refocusing_axis(
        refocusing,
        pulse_number=pulse_number,
        numpts=numpts,
        result_times_are_t180=result_times_are_t180,
    )
    actual_numpts = int(neff.shape[1])
    excitation_options = dict(excitation_kwargs or {})
    inverse_options = dict(inverse_kwargs or {})
    excitation = drivers.run_tuned_excitation_multistart(
        excitation_segments,
        neff,
        num_starts=excitation_starts,
        seed=None if seed is None else int(seed),
        numpts=actual_numpts,
        **excitation_options,
    )
    target = excitation.best_result.best_evaluation
    inverse = drivers.run_tuned_inverse_excitation_multistart(
        excitation_segments,
        neff,
        target.mrx,
        target.snr,
        target.phases,
        num_starts=inverse_starts,
        seed=None if seed is None else int(seed) + 1,
        random_fraction=random_fraction,
        numpts=actual_numpts,
        **inverse_options,
    )
    validations = tuple(
        _validate_pipeline_pair(target, result) for result in inverse.results
    )
    residual_ratios = np.asarray(
        [validation.residual_ratio for validation in validations], dtype=np.float64
    )
    residual_best_index = int(np.nanargmin(residual_ratios))
    return TunedExcitationInversePipelineResult(
        del_w=del_w,
        neff=neff,
        selected_refocusing=selected,
        excitation=excitation,
        inverse=inverse,
        inverse_validations=validations,
        inverse_residual_ratios=residual_ratios,
        residual_best_index=residual_best_index,
    )
