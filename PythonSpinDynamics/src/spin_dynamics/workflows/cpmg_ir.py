"""CPMG inversion-recovery workflows."""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.core.isochromats import check_rephasing
from spin_dynamics.core.rotations import calc_rotation_matrix
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.probes.matched import matching_network_design2
from spin_dynamics.probes.tuned import tuned_probe_rx_tf
from spin_dynamics.probes.untuned import untuned_probe_rx_tf
from spin_dynamics.workflows.acquisition import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
)
from spin_dynamics.workflows.cpmg import (
    _calc_tuned_pulse_shape,
    _calc_untuned_pulse_shape,
    _calc_matched_pulse_shape,
    _echo_train_from_spectra,
    _maybe_refine_numpts,
    _offset_grid,
)


@dataclass(frozen=True)
class CPMGIRTrainResult:
    """Finite CPMG-IR echo train over inversion delays."""

    tauvect: np.ndarray
    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str


@dataclass(frozen=True)
class MatchedCPMGIRTrainResult(CPMGIRTrainResult):
    """Finite matched-probe CPMG-IR echo train over inversion delays."""


def _as_tauvect(tauvect: Iterable[float] | np.ndarray | None) -> np.ndarray:
    if tauvect is None:
        return np.linspace(0.5e-3, 10e-3, 20)
    tau = np.asarray(tauvect, dtype=np.float64).reshape(-1)
    if tau.size == 0:
        raise ValueError("tauvect must not be empty")
    if np.any(tau < 0):
        raise ValueError("tauvect entries must be non-negative")
    return tau


def _validate_ir_inputs(
    num_echoes: int,
    echo_spacing_seconds: float,
    t1_seconds: float,
    t2_seconds: float,
) -> None:
    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing_seconds <= 0:
        raise ValueError("echo_spacing_seconds must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")


def _stack_ir_rows(
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([row[0] for row in rows], axis=0),
        np.stack([row[1] for row in rows], axis=0),
        np.stack([row[2] for row in rows], axis=0),
    )


def run_ideal_cpmg_ir_train(
    num_echoes: int = 10,
    echo_spacing_seconds: float = 0.5e-3,
    tauvect: Iterable[float] | np.ndarray | None = None,
    t1_seconds: float = 5e-3,
    t2_seconds: float = 5e-3,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
    tau_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGIRTrainResult:
    """Run a compact ideal-probe CPMG-IR finite echo train."""

    _validate_ir_inputs(num_echoes, echo_spacing_seconds, t1_seconds, t2_seconds)
    tau = _as_tauvect(tauvect)
    sp0, pp0 = set_params_ideal(numpts=numpts)
    if echo_spacing_seconds <= pp0.T_180:
        raise ValueError("echo_spacing_seconds must be longer than the refocusing pulse")

    t_90 = pp0.T_90
    w1n = (np.pi / 2) / t_90
    tfp = (np.pi / 2) * (echo_spacing_seconds - pp0.T_180) / (2 * t_90)
    tau_norm_max = (np.pi / 2) * float(np.max(tau)) / t_90
    max_time = float(np.pi + tau_norm_max + np.pi / 2 + int(num_echoes) * (tfp + np.pi + tfp))
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(del_w, max_time, safety_factor=rephase_safety_factor, action=rephase_action)

    sp = {
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
    }
    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], w1n * pp0.texc, pp0.pexc, pp0.aexc),
        calc_rotation_matrix(del_w, sp["w_1"], w1n * pp0.texc, pp0.pexc + np.pi, pp0.aexc),
        calc_rotation_matrix(
            del_w,
            sp["w_1"],
            w1n * pp0.tref[1:-1],
            pp0.pref[1:-1],
            pp0.aref[1:-1],
        ),
    ]

    tenc = np.array([np.pi, (np.pi / 2) * tau[0] / t_90], dtype=np.float64)
    penc = np.array([3, 0], dtype=np.int64)
    aenc = np.array([1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0], dtype=np.int64)
    grad_enc = np.array([0.0, 0.0], dtype=np.float64)
    ind_enc = 1

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t_90,
        "tp": np.concatenate([tenc, texc, tref]),
        "amp": np.concatenate([aenc, aexc, aref]),
        "acq": np.concatenate([acq_enc, acq_exc, acq_ref]),
        "grad": np.concatenate([grad_enc, grad_exc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([penc, pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([penc, pexc2, pref])}
    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / t_90)
    tdw = float((np.pi / 2) * pp0.tdw / t_90)

    def run_tau(tau_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ppc1 = {**pp1, "tp": np.array(pp1["tp"], copy=True)}
        ppc2 = {**pp2, "tp": np.array(pp2["tp"], copy=True)}
        tau_norm = (np.pi / 2) * float(tau_seconds) / t_90
        ppc1["tp"][ind_enc] = tau_norm
        ppc2["tp"][ind_enc] = tau_norm
        mrx1 = calc_macq_ideal_probe_relax4(sp, ppc1, num_workers=num_workers)
        mrx2 = calc_macq_ideal_probe_relax4(sp, ppc2, num_workers=num_workers)
        mrx = mrx2 - mrx1
        echo, _tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
        return mrx, echo, echo_integrals

    workers = 1 if tau_workers is None else int(tau_workers)
    if workers <= 1:
        rows = [run_tau(float(value)) for value in tau]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run_tau, [float(value) for value in tau]))

    mrx, echo, echo_integrals = _stack_ir_rows(rows)
    tvect = _echo_train_from_spectra(rows[0][0], del_w, tacq, tdw)[1]
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1.0)
    return CPMGIRTrainResult(
        tauvect=tau,
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="ideal",
    )


def _run_probe_cpmg_ir_train(
    probe: str,
    num_echoes: int,
    echo_spacing_seconds: float,
    tauvect: Iterable[float] | np.ndarray | None,
    t1_seconds: float,
    t2_seconds: float,
    *,
    numpts: int,
    maxoffs: float,
    num_workers: int | None,
    tau_workers: int | None,
    auto_refine_grid: bool,
    rephase_safety_factor: float,
    rephase_action: str,
) -> CPMGIRTrainResult:
    _validate_ir_inputs(num_echoes, echo_spacing_seconds, t1_seconds, t2_seconds)
    tau = _as_tauvect(tauvect)
    if probe == "tuned":
        _params, sp0, pp0 = set_params_tuned_orig(numpts=numpts)
        pulse_shape = _calc_tuned_pulse_shape
        receiver_tf = tuned_probe_rx_tf
        calc_macq = calc_macq_tuned_probe_relax4
        sp0 = replace(sp0, R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q)
    elif probe == "untuned":
        _params, sp0, pp0 = set_params_untuned_orig(numpts=numpts)
        pulse_shape = _calc_untuned_pulse_shape
        receiver_tf = untuned_probe_rx_tf
        calc_macq = calc_macq_untuned_probe_relax4
        sp0 = replace(
            sp0,
            R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            C=1 / ((2 * np.pi * 10 * sp0.f0) ** 2 * sp0.L),
        )
    else:
        raise ValueError("probe must be 'tuned' or 'untuned'")
    if echo_spacing_seconds <= pp0.T_180:
        raise ValueError("echo_spacing_seconds must be longer than the refocusing pulse")

    t_90 = pp0.T_90
    tfp = (np.pi / 2) * (echo_spacing_seconds - pp0.T_180) / (2 * t_90)
    tau_norm_max = (np.pi / 2) * float(np.max(tau)) / t_90
    max_time = float(np.pi + tau_norm_max + np.pi / 2 + int(num_echoes) * (tfp + np.pi + tfp))
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(del_w, max_time, safety_factor=rephase_safety_factor, action=rephase_action)

    sp = {
        **sp0.__dict__,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    exc_y = pulse_shape(sp, pp0, pp0.T_90, np.pi / 2, 1.0, 2 * pp0.T_90)
    exc_minus_y = pulse_shape(sp, pp0, pp0.T_90, 3 * np.pi / 2, 1.0, 2 * pp0.T_90)
    ref_x = pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, 2 * pp0.T_90)
    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], *exc_y),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]
    sp["tf"] = receiver_tf(sp, pp0)

    tenc = np.array([np.pi, (np.pi / 2) * tau[0] / t_90], dtype=np.float64)
    penc = np.array([3, 0], dtype=np.int64)
    aenc = np.array([1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0], dtype=np.int64)
    grad_enc = np.array([0.0, 0.0], dtype=np.float64)
    ind_enc = 1

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / t_90], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t_90,
        "tp": np.concatenate([tenc, texc, tref]),
        "amp": np.concatenate([aenc, aexc, aref]),
        "acq": np.concatenate([acq_enc, acq_exc, acq_ref]),
        "grad": np.concatenate([grad_enc, grad_exc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([penc, pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([penc, pexc2, pref])}
    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / t_90)
    tdw = float((np.pi / 2) * pp0.tdw / t_90)

    def run_tau(tau_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ppc1 = {**pp1, "tp": np.array(pp1["tp"], copy=True)}
        ppc2 = {**pp2, "tp": np.array(pp2["tp"], copy=True)}
        tau_norm = (np.pi / 2) * float(tau_seconds) / t_90
        ppc1["tp"][ind_enc] = tau_norm
        ppc2["tp"][ind_enc] = tau_norm
        _macq1, mrx1 = calc_macq(sp, ppc1, num_workers=num_workers)
        _macq2, mrx2 = calc_macq(sp, ppc2, num_workers=num_workers)
        mrx = mrx2 - mrx1
        echo, _tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
        return mrx, echo, echo_integrals

    workers = 1 if tau_workers is None else int(tau_workers)
    if workers <= 1:
        rows = [run_tau(float(value)) for value in tau]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run_tau, [float(value) for value in tau]))

    mrx, echo, echo_integrals = _stack_ir_rows(rows)
    tvect = _echo_train_from_spectra(rows[0][0], del_w, tacq, tdw)[1]
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1.0)
    return CPMGIRTrainResult(
        tauvect=tau,
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe=probe,
    )


def run_tuned_cpmg_ir_train(
    num_echoes: int = 10,
    echo_spacing_seconds: float = 0.5e-3,
    tauvect: Iterable[float] | np.ndarray | None = None,
    t1_seconds: float = 5e-3,
    t2_seconds: float = 5e-3,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
    tau_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGIRTrainResult:
    """Run a compact tuned-probe CPMG-IR finite echo train."""

    return _run_probe_cpmg_ir_train(
        "tuned",
        num_echoes,
        echo_spacing_seconds,
        tauvect,
        t1_seconds,
        t2_seconds,
        numpts=numpts,
        maxoffs=maxoffs,
        num_workers=num_workers,
        tau_workers=tau_workers,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )


def run_untuned_cpmg_ir_train(
    num_echoes: int = 10,
    echo_spacing_seconds: float = 0.5e-3,
    tauvect: Iterable[float] | np.ndarray | None = None,
    t1_seconds: float = 5e-3,
    t2_seconds: float = 5e-3,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
    tau_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGIRTrainResult:
    """Run a compact untuned-probe CPMG-IR finite echo train."""

    return _run_probe_cpmg_ir_train(
        "untuned",
        num_echoes,
        echo_spacing_seconds,
        tauvect,
        t1_seconds,
        t2_seconds,
        numpts=numpts,
        maxoffs=maxoffs,
        num_workers=num_workers,
        tau_workers=tau_workers,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )


def run_matched_cpmg_ir_train(
    num_echoes: int = 10,
    echo_spacing_seconds: float = 0.5e-3,
    tauvect: Iterable[float] | np.ndarray | None = None,
    t1_seconds: float = 5e-3,
    t2_seconds: float = 5e-3,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
    tau_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> MatchedCPMGIRTrainResult:
    """Run a compact matched-probe CPMG-IR finite echo train.

    This mirrors the optimized MATLAB workflow
    `Sim_CPMG_IR/sim_cpmg_ir_matched_probe_relax4.m`: matched-probe RF pulse
    shapes, rotation matrices, receiver transfer functions, and time-domain
    isochromats are precomputed once, then the inversion delay is swept over
    `tauvect`.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing_seconds <= 0:
        raise ValueError("echo_spacing_seconds must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    tau = _as_tauvect(tauvect)
    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    if echo_spacing_seconds <= pp0.T_180:
        raise ValueError("echo_spacing_seconds must be longer than the refocusing pulse")

    t_90 = pp0.T_90
    tfp = (np.pi / 2) * (echo_spacing_seconds - pp0.T_180) / (2 * t_90)
    tau_norm_max = (np.pi / 2) * float(np.max(tau)) / t_90
    max_time = float(np.pi + tau_norm_max + np.pi / 2 + int(num_echoes) * (tfp + np.pi + tfp))
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )

    c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
    sp = {
        **sp0.__dict__,
        "C1": c1,
        "C2": c2,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        np.pi / 2,
        1.0,
        pp0.trd,
    )
    exc_minus_y = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        3 * np.pi / 2,
        1.0,
        pp0.trd,
    )[:3]
    ref_x = _calc_matched_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)[:3]

    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], exc_y_tp, exc_y_phi, exc_y_amp),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]

    tenc = np.array([np.pi, (np.pi / 2) * tau[0] / t_90], dtype=np.float64)
    penc = np.array([3, 0], dtype=np.int64)
    aenc = np.array([1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0], dtype=np.int64)
    grad_enc = np.array([0.0, 0.0], dtype=np.float64)
    ind_enc = 1

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t_90,
        "tp": np.concatenate([tenc, texc, tref]),
        "amp": np.concatenate([aenc, aexc, aref]),
        "acq": np.concatenate([acq_enc, acq_exc, acq_ref]),
        "grad": np.concatenate([grad_enc, grad_exc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([penc, pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([penc, pexc2, pref])}
    sp["tf1"] = tf1
    sp["tf2"] = tf2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / t_90)
    tdw = float((np.pi / 2) * pp0.tdw / t_90)

    def run_tau(tau_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ppc1 = {**pp1, "tp": np.array(pp1["tp"], copy=True)}
        ppc2 = {**pp2, "tp": np.array(pp2["tp"], copy=True)}
        tau_norm = (np.pi / 2) * float(tau_seconds) / t_90
        ppc1["tp"][ind_enc] = tau_norm
        ppc2["tp"][ind_enc] = tau_norm
        _macq1, mrx1 = calc_macq_matched_probe_relax4(
            sp,
            ppc1,
            num_workers=num_workers,
        )
        _macq2, mrx2 = calc_macq_matched_probe_relax4(
            sp,
            ppc2,
            num_workers=num_workers,
        )
        mrx = mrx2 - mrx1
        echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
        return mrx, echo, echo_integrals

    workers = 1 if tau_workers is None else int(tau_workers)
    if workers <= 1:
        rows = [run_tau(float(value)) for value in tau]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(run_tau, [float(value) for value in tau]))

    mrx = np.stack([row[0] for row in rows], axis=0)
    echo = np.stack([row[1] for row in rows], axis=0)
    echo_integrals = np.stack([row[2] for row in rows], axis=0)
    sequence_time = echo_spacing_seconds * (
        np.arange(int(num_echoes), dtype=np.float64) + 1.0
    )
    tvect = _echo_train_from_spectra(rows[0][0], del_w, tacq, tdw)[1]
    return MatchedCPMGIRTrainResult(
        tauvect=tau,
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="matched",
    )
