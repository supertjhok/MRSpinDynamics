"""End-to-end helpers for OCT-style optimization workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rot_axis_arba4
import spin_dynamics.optimization.drivers as drivers
from spin_dynamics.optimization.results import (
    PulseProgram,
    SelectedOptimizationProgram,
    select_matlab_result_program,
)


@dataclass(frozen=True)
class TunedExcitationInversePipelineResult:
    """Selected-refocusing to excitation/inverse-excitation pipeline result."""

    del_w: np.ndarray
    neff: np.ndarray
    selected_refocusing: SelectedOptimizationProgram | None
    excitation: drivers.MultiStartOptimizationResult
    inverse: drivers.MultiStartOptimizationResult
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


def _axis_from_evaluation(
    evaluation: Any,
    *,
    numpts: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    neff = getattr(evaluation, "neff", None)
    if neff is None:
        return None
    neff_arr = np.asarray(neff, dtype=np.complex128)
    if neff_arr.ndim != 2 or neff_arr.shape[0] != 3:
        raise ValueError("refocusing evaluation neff must have shape (3, numpts)")
    if numpts is not None and neff_arr.shape[1] != int(numpts):
        raise ValueError("refocusing evaluation neff does not match requested numpts")
    del_w = getattr(evaluation, "del_w", None)
    if del_w is None:
        del_w_arr = np.linspace(-10.0, 10.0, neff_arr.shape[1])
    else:
        del_w_arr = np.asarray(del_w, dtype=np.float64).reshape(-1)
    if del_w_arr.shape != (neff_arr.shape[1],):
        raise ValueError("refocusing evaluation del_w must have shape (numpts,)")
    return del_w_arr, neff_arr


def _axis_from_program(
    program: PulseProgram,
    *,
    numpts: int,
    maxoffs: float,
    times_are_t180: bool,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(program.times, dtype=np.float64).reshape(-1)
    phases = np.asarray(program.phases, dtype=np.float64).reshape(-1)
    amplitudes = np.asarray(program.amplitudes, dtype=np.float64).reshape(-1)
    if not (times.size == phases.size == amplitudes.size):
        raise ValueError("refocusing pulse program arrays must have matching lengths")
    if times.size == 0:
        raise ValueError("refocusing pulse program must not be empty")
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    normalized_times = np.pi * times if times_are_t180 else times
    neff, _alpha = calc_rot_axis_arba4(normalized_times, phases, amplitudes, del_w)
    return del_w, neff


def _resolve_refocusing_axis(
    refocusing: Any,
    *,
    pulse_number: int | None,
    numpts: int,
    maxoffs: float,
    result_times_are_t180: bool,
) -> tuple[np.ndarray, np.ndarray, SelectedOptimizationProgram | None]:
    if hasattr(refocusing, "best_result") and hasattr(refocusing, "results"):
        result = _result_at_pulse_number(refocusing, pulse_number)
        evaluation_axis = _axis_from_evaluation(
            getattr(result, "best_evaluation", None),
            numpts=numpts,
        )
        if evaluation_axis is not None:
            del_w, neff = evaluation_axis
            return del_w, neff, None
        phases = np.asarray(getattr(result, "best_phases"), dtype=np.float64).reshape(-1)
        evaluation = getattr(result, "best_evaluation", None)
        pulse_length = getattr(evaluation, "pulse_length_t180", None)
        segment_fraction = (
            0.1 if pulse_length is None else float(pulse_length) / phases.size
        )
        program = PulseProgram(
            times=segment_fraction * np.ones(phases.size, dtype=np.float64),
            phases=phases,
            amplitudes=np.ones(phases.size, dtype=np.float64),
        )
        del_w, neff = _axis_from_program(
            program,
            numpts=numpts,
            maxoffs=maxoffs,
            times_are_t180=True,
        )
        return del_w, neff, None

    selected = select_matlab_result_program(refocusing, pulse_number=pulse_number)
    if selected.refocusing is None:
        raise ValueError("selected optimization result does not contain refocusing data")
    del_w, neff = _axis_from_program(
        selected.refocusing,
        numpts=numpts,
        maxoffs=maxoffs,
        times_are_t180=result_times_are_t180,
    )
    return del_w, neff, selected


def _residual_ratio(target: Any, inverse_result: Any) -> float:
    inverse_eval = inverse_result.best_evaluation.excitation
    target_mrx = np.asarray(target.mrx, dtype=np.complex128)
    inverse_mrx = np.asarray(inverse_eval.mrx, dtype=np.complex128)
    del_w = np.asarray(target.del_w, dtype=np.float64)
    target_norm = trapezoid(np.abs(target_mrx), del_w)
    if target_norm == 0:
        return np.inf
    return float(trapezoid(np.abs(target_mrx + inverse_mrx), del_w) / target_norm)


def run_tuned_excitation_inverse_pipeline(
    refocusing: Any,
    *,
    pulse_number: int | None = None,
    excitation_segments: int = 3,
    excitation_starts: int = 4,
    inverse_starts: int = 4,
    seed: int | None = None,
    numpts: int = 21,
    maxoffs: float = 10.0,
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
    """

    del_w, neff, selected = _resolve_refocusing_axis(
        refocusing,
        pulse_number=pulse_number,
        numpts=numpts,
        maxoffs=maxoffs,
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
    residual_ratios = np.asarray(
        [_residual_ratio(target, result) for result in inverse.results],
        dtype=np.float64,
    )
    residual_best_index = int(np.nanargmin(residual_ratios))
    return TunedExcitationInversePipelineResult(
        del_w=del_w,
        neff=neff,
        selected_refocusing=selected,
        excitation=excitation,
        inverse=inverse,
        inverse_residual_ratios=residual_ratios,
        residual_best_index=residual_best_index,
    )
