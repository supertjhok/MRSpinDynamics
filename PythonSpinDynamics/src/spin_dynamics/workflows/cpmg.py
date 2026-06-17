"""CPMG workflow entry points.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.core.isochromats import (
    check_rephasing,
    recommended_numpts_for_rephasing,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    calc_rot_axis_arba3,
    calc_rotation_matrix,
    sim_spin_dynamics_asymp_mag3,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.probes.matched import calc_masy_matched_probe_orig
from spin_dynamics.probes.matched import find_coil_current, matching_network_design2
from spin_dynamics.probes.tuned import (
    calc_masy_tuned_probe_lp_orig,
    tuned_probe_lp,
    tuned_probe_rx_tf,
)
from spin_dynamics.probes.untuned import (
    calc_masy_untuned_probe_lp,
    untuned_probe_lp,
    untuned_probe_rx_tf,
)
from spin_dynamics.workflows.acquisition import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
)


@dataclass(frozen=True)
class CPMGResult:
    """Common result object for ideal and probe-aware CPMG workflows."""

    del_w: np.ndarray
    masy: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float | None
    probe: str


@dataclass(frozen=True)
class CPMGTrainResult:
    """Finite ideal CPMG acquisition result."""

    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def calc_masy_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> np.ndarray:
    """Calculate ideal CPMG asymptotic magnetization.

    Mirrors MATLAB `calc_masy/calc_masy_ideal.m`, with plotting removed.
    """

    T_90 = _field(pp, "T_90")
    del_w = np.asarray(_field(sp, "del_w"), dtype=np.float64).reshape(-1)

    tacq = (np.pi / 2) * np.asarray(_field(pp, "tacq"), dtype=np.float64) / T_90
    tacq_scalar = float(np.ravel(tacq)[0])

    tref = (np.pi / 2) * np.asarray(_field(pp, "tref"), dtype=np.float64) / T_90
    pref = np.asarray(_field(pp, "pref"), dtype=np.float64)
    aref = np.asarray(_field(pp, "aref"), dtype=np.float64)
    neff = calc_rot_axis_arba3(tref, pref, aref, del_w)

    texc = (np.pi / 2) * np.asarray(_field(pp, "texc"), dtype=np.float64) / T_90
    texc = np.concatenate([texc.reshape(-1), np.array([-1.0])])
    pexc = np.concatenate([
        np.asarray(_field(pp, "pexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])
    aexc = np.concatenate([
        np.asarray(_field(pp, "aexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])

    masy1 = sim_spin_dynamics_asymp_mag3(texc, pexc, aexc, neff, del_w, tacq_scalar)
    masy2 = sim_spin_dynamics_asymp_mag3(texc, pexc + np.pi, aexc, neff, del_w, tacq_scalar)
    return (masy1 - masy2) / 2


def _offset_grid(numpts: int, maxoffs: float) -> np.ndarray:
    return np.linspace(-float(maxoffs), float(maxoffs), int(numpts))


def _maybe_refine_numpts(
    numpts: int,
    maxoffs: float,
    max_time: float,
    safety_factor: float,
    auto_refine_grid: bool,
) -> int:
    if not auto_refine_grid:
        return int(numpts)
    recommended = recommended_numpts_for_rephasing(maxoffs, max_time, safety_factor)
    return max(int(numpts), recommended)


def _echo_train_from_spectra(
    mrx: np.ndarray,
    del_w: np.ndarray,
    tacq: float,
    tdw: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * del_w[np.newaxis, :])
    echo = (isoc @ mrx.T).T
    echo_integrals = trapezoid(echo, tvect, axis=1)
    return echo, tvect, echo_integrals


def run_ideal_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGTrainResult:
    """Run a finite ideal CPMG echo train with relaxation.

    This assembles the same no-probe PAP phase-cycled acquisition pattern used
    by MATLAB finite CPMG examples around `calc_macq_ideal_probe_relax4`.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_ideal(numpts=numpts)
    w1n = (np.pi / 2) / pp0.T_90
    max_time = float(
        np.pi / 2
        + w1n * pp0.tcorr
        + int(num_echoes) * np.sum(w1n * np.asarray(pp0.tref, dtype=np.float64))
    )
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
        calc_rotation_matrix(
            del_w,
            sp["w_1"],
            w1n * pp0.texc,
            pp0.pexc,
            pp0.aexc,
        ),
        calc_rotation_matrix(
            del_w,
            sp["w_1"],
            w1n * pp0.texc,
            pp0.pexc + np.pi,
            pp0.aexc,
        ),
        calc_rotation_matrix(
            del_w,
            sp["w_1"],
            w1n * pp0.tref[1:-1],
            pp0.pref[1:-1],
            pp0.aref[1:-1],
        ),
    ]

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(
        w1n * np.array([pp0.tref[0], pp0.tref[1], pp0.tref[2]], dtype=np.float64),
        int(num_echoes),
    )
    pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

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
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="ideal",
    )


def _calc_tuned_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tref": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "pref": np.array([pulse_phase, 0.0], dtype=np.float64),
        "aref": np.array([pulse_amplitude, 0.0], dtype=np.float64),
    }
    tvect, icr, _tvect_raw, _ic = tuned_probe_lp(sp, pp_curr)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0

    amp_range = float(np.max(amp) - np.min(amp))
    if amp_range > 0:
        amp = (amp - np.min(amp)) / amp_range
    amp[amp < amp_zero] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
    )


def _calc_untuned_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tref": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "pref": np.array([pulse_phase, 0.0], dtype=np.float64),
        "aref": np.array([pulse_amplitude, 0.0], dtype=np.float64),
    }
    tvect, icr, _tvect_raw, _ic = untuned_probe_lp(sp, pp_curr)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    B1max = (np.pi / 2) / (T_90 * float(_field(sp, "gamma")))
    amp = np.abs(icr) * float(_field(sp, "sens")) / B1max
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0
    phi[amp == 0] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
    )


def _calc_matched_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tp": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "phi": np.array([pulse_phase, 0.0], dtype=np.float64),
        "amp": np.array([pulse_amplitude, 0.0], dtype=np.float64),
    }
    tvect, icr, tf1, tf2 = find_coil_current(sp, pp_curr)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
        tf1,
        tf2,
    )


def run_tuned_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGTrainResult:
    """Run a finite tuned-probe CPMG echo train with relaxation.

    This is the homogeneous-sample finite-acquisition analogue of the tuned
    Version 2 CPMG imaging assembly, without phase-encoding gradients.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    _params, sp0, pp0 = set_params_tuned_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(
        sp0,
        R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
        C=1 / ((2 * np.pi * sp0.f0) ** 2 * sp0.L),
    )
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
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

    exc_y = _calc_tuned_pulse_shape(sp, pp0, pp0.T_90, np.pi / 2, 1.0, 2 * pp0.T_90)
    exc_minus_y = _calc_tuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        3 * np.pi / 2,
        1.0,
        2 * pp0.T_90,
    )
    ref_x = _calc_tuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, 2 * pp0.T_90)

    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], *exc_y),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
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
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf"] = tuned_probe_rx_tf(sp, pp0)
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, pref])}
    _macq1, mrx1 = calc_macq_tuned_probe_relax4(sp, pp1, num_workers=num_workers)
    _macq2, mrx2 = calc_macq_tuned_probe_relax4(sp, pp2, num_workers=num_workers)
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="tuned",
    )


def run_untuned_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGTrainResult:
    """Run a finite untuned-probe CPMG echo train with relaxation."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    _params, sp0, pp0 = set_params_untuned_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(
        sp0,
        R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
        C=1 / ((2 * np.pi * 10 * sp0.f0) ** 2 * sp0.L),
    )
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
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

    exc_y = _calc_untuned_pulse_shape(sp, pp0, pp0.T_90, np.pi / 2, 1.0, pp0.trd)
    exc_minus_y = _calc_untuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        3 * np.pi / 2,
        1.0,
        pp0.trd,
    )
    ref_x = _calc_untuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)

    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], *exc_y),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
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
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf"] = untuned_probe_rx_tf(sp, pp0)
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, pref])}
    _macq1, mrx1 = calc_macq_untuned_probe_relax4(sp, pp1, num_workers=num_workers)
    _macq2, mrx2 = calc_macq_untuned_probe_relax4(sp, pp2, num_workers=num_workers)
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="untuned",
    )


def run_matched_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGTrainResult:
    """Run a finite matched-probe CPMG echo train with relaxation."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(sp0, R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q)
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
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

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
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
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf1"] = tf1
    sp["tf2"] = tf2
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, pref])}
    _macq1, mrx1 = calc_macq_matched_probe_relax4(sp, pp1, num_workers=num_workers)
    _macq2, mrx2 = calc_macq_matched_probe_relax4(sp, pp2, num_workers=num_workers)
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="matched",
    )


def run_ideal_cpmg(numpts: int = 101, maxoffs: float = 10.0) -> CPMGResult:
    """Run the validated ideal no-probe CPMG workflow."""

    del_w = _offset_grid(numpts, maxoffs)
    sp, pp = set_params_ideal(numpts=numpts)
    sp = replace(sp, maxoffs=float(maxoffs), del_w=del_w)
    masy = calc_masy_ideal(sp, pp)
    echo, tvect = calc_time_domain_echo(masy, del_w)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=masy,
        echo=echo,
        tvect=tvect,
        snr=None,
        probe="ideal",
    )


def run_tuned_cpmg(numpts: int = 101, maxoffs: float = 10.0) -> CPMGResult:
    """Run the original/reference tuned-probe CPMG workflow."""

    del_w = _offset_grid(numpts, maxoffs)
    params, sp, pp = set_params_tuned_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
    )
    mrx, masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="tuned",
    )


def run_untuned_cpmg(numpts: int = 101, maxoffs: float = 10.0) -> CPMGResult:
    """Run the original/reference untuned-probe CPMG workflow."""

    del_w = _offset_grid(numpts, maxoffs)
    params, sp, pp = set_params_untuned_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
        plt_axis=0,
    )
    mrx, masy, snr = calc_masy_untuned_probe_lp(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="untuned",
    )


def run_matched_cpmg(numpts: int = 101, maxoffs: float = 10.0) -> CPMGResult:
    """Run the original/reference matched-probe CPMG workflow."""

    del_w = _offset_grid(numpts, maxoffs)
    sp, pp = set_params_matched_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
        plt_axis=0,
        plt_mn=0,
    )
    mrx, masy, snr = calc_masy_matched_probe_orig(sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="matched",
    )
