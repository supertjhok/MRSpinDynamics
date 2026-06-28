"""Batched (vmap) sweep workflows over the JAX arb10 kernel.

Phase 2b of the acceleration plan wires the batched primitive
(`spin_dynamics.core.kernels.sim_spin_dynamics_arb10_batched`) into a concrete
end-user workflow. ``run_ideal_cpmg_relaxation_sweep`` evaluates the ideal finite
CPMG echo train over a grid of (T1, T2) values in a single batched call instead
of a Python/thread loop — the common shape for building T2 / T1-T2 dictionaries
for inverse-Laplace analysis.

All cases share one pulse program (identical excitation/refocusing matrices and
timing); only the per-isochromat relaxation fields differ, so the batch maps
directly onto the batched kernel. The PAP two-step phase cycle is handled by
batching each branch and combining per case. Requires the optional ``jax`` extra.

See ``docs/performance.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.core.isochromats import check_rephasing
from spin_dynamics.core.kernels import sim_spin_dynamics_arb10_batched
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.cpmg import (
    _build_absolute_phase_rtot_and_pul,
    _cpmg_absolute_phase_plan,
    _cpmg_branch_prefixes,
    _cpmg_excitation_offsets,
    _default_cpmg_phase_cycle,
    _echo_phase_matrix,
    _echo_train_from_spectra,
    _maybe_refine_numpts,
    _offset_grid,
    _pulse_shape,
)


@dataclass(frozen=True)
class CPMGRelaxationSweepResult:
    """Array-returning result for a batched ideal CPMG (T1, T2) sweep."""

    t1_seconds: np.ndarray
    t2_seconds: np.ndarray
    del_w: np.ndarray
    mrx: np.ndarray  # (cases, num_echoes, numpts)
    echo: np.ndarray  # (cases, num_echoes, ntime)
    tvect: np.ndarray
    echo_integrals: np.ndarray  # (cases, num_echoes)
    sequence_time: np.ndarray


def run_ideal_cpmg_relaxation_sweep(
    t1_seconds: np.ndarray | list[float],
    t2_seconds: np.ndarray | list[float],
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGRelaxationSweepResult:
    """Evaluate the ideal finite CPMG echo train over a (T1, T2) grid in one batch.

    ``t1_seconds`` and ``t2_seconds`` are paired element-wise (one case each).
    Results match looping :func:`run_ideal_cpmg_train` per pair, but run as a
    single vmapped JAX program. Requires the optional ``jax`` extra.
    """

    t1 = np.asarray(t1_seconds, dtype=np.float64).reshape(-1)
    t2 = np.asarray(t2_seconds, dtype=np.float64).reshape(-1)
    if t1.size == 0:
        raise ValueError("t1_seconds must not be empty")
    if t1.shape != t2.shape:
        raise ValueError("t1_seconds and t2_seconds must have the same shape")
    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if np.any(t1 <= 0) or np.any(t2 <= 0):
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_ideal(numpts=numpts)
    w1n = (np.pi / 2) / pp0.T_90
    max_time = float(
        np.pi / 2
        + w1n * pp0.tcorr
        + int(num_echoes) * np.sum(w1n * np.asarray(pp0.tref, dtype=np.float64))
    )
    numpts = _maybe_refine_numpts(
        numpts, maxoffs, max_time, rephase_safety_factor, auto_refine_grid
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w, max_time, safety_factor=rephase_safety_factor, action=rephase_action
        )

    # Shared pulse program (T1/T2-independent), mirroring run_ideal_cpmg_train.
    phase_cycle = _default_cpmg_phase_cycle()
    excitation_offsets = _cpmg_excitation_offsets(phase_cycle)
    ap_spec, ap_schedule = _cpmg_absolute_phase_plan(
        None, pp0, num_echoes=int(num_echoes), phase_cycle=phase_cycle
    )
    exc_y = _pulse_shape(w1n * pp0.texc, pp0.pexc + excitation_offsets[0], pp0.aexc)
    exc_minus_y = _pulse_shape(
        w1n * pp0.texc, pp0.pexc + excitation_offsets[1], pp0.aexc
    )
    ref_x = _pulse_shape(w1n * pp0.tref[1:-1], pp0.pref[1:-1], pp0.aref[1:-1])
    rtot, pref, _pref2, _ap_metadata = _build_absolute_phase_rtot_and_pul(
        del_w=del_w,
        w_1=np.ones_like(del_w),
        spec=ap_spec,
        pp=pp0,
        schedule=ap_schedule,
        num_echoes=int(num_echoes),
        exc_y=exc_y,
        exc_minus_y=exc_minus_y,
        ref_shape_factory=lambda _absolute_start: ref_x,
        phase_cycle=phase_cycle,
    )

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)
    tref = np.tile(
        w1n * np.array([pp0.tref[0], pp0.tref[1], pp0.tref[2]], dtype=np.float64),
        int(num_echoes),
    )
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    tp = np.concatenate([texc, tref])
    amp = np.concatenate([aexc, aref])
    acq = np.concatenate([acq_exc, acq_ref])
    grad = np.concatenate([grad_exc, grad_ref])
    del_wg = np.zeros_like(del_w)
    m0 = sp0.m0 * np.ones_like(del_w)
    mth = sp0.mth * np.ones_like(del_w)

    # T1/T2 normalization matches calc_macq_ideal_probe_relax4.
    scale = (np.pi / 2) / pp0.T_90
    ones = np.ones_like(del_w)

    def case_params(branch_pul: np.ndarray, idx: int) -> dict:
        return {
            "tp": tp,
            "pul": branch_pul,
            "amp": amp,
            "acq": acq,
            "grad": grad,
            "Rtot": rtot,
            "del_w": del_w,
            "del_wg": del_wg,
            "T1n": scale * t1[idx] * ones,
            "T2n": scale * t2[idx] * ones,
            "m0": m0,
            "mth": mth,
        }

    # Batch each phase-cycle branch over all cases, then combine per case.
    branch_macq = []
    for prefix in _cpmg_branch_prefixes(phase_cycle):
        branch_pul = np.concatenate([prefix, pref])
        params_list = [case_params(branch_pul, idx) for idx in range(t1.size)]
        branch_macq.append(sim_spin_dynamics_arb10_batched(params_list))

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)

    mrx_cases, echo_cases, integral_cases = [], [], []
    for idx in range(t1.size):
        branch_signals = [branch_macq[b][idx] for b in range(len(branch_macq))]
        mrx_i = phase_cycle.combine(branch_signals)
        echo_i, _tvect, integrals_i = _echo_train_from_spectra(
            mrx_i, del_w, tacq, tdw, tvect=tvect, isoc=isoc
        )
        mrx_cases.append(mrx_i)
        echo_cases.append(echo_i)
        integral_cases.append(integrals_i)

    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )
    return CPMGRelaxationSweepResult(
        t1_seconds=t1,
        t2_seconds=t2,
        del_w=del_w,
        mrx=np.stack(mrx_cases, axis=0),
        echo=np.stack(echo_cases, axis=0),
        tvect=tvect,
        echo_integrals=np.stack(integral_cases, axis=0),
        sequence_time=sequence_time,
    )
