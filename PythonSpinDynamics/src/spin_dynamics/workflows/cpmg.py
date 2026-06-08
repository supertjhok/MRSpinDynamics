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
from spin_dynamics.probes.tuned import calc_masy_tuned_probe_lp_orig
from spin_dynamics.probes.untuned import calc_masy_untuned_probe_lp
from spin_dynamics.workflows.acquisition import calc_macq_ideal_probe_relax4


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


def run_ideal_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
) -> CPMGTrainResult:
    """Run a finite ideal CPMG echo train with relaxation.

    This assembles the same no-probe PAP phase-cycled acquisition pattern used
    by MATLAB finite CPMG examples around `calc_macq_ideal_probe_relax4`.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    del_w = _offset_grid(numpts, maxoffs)
    sp0, pp0 = set_params_ideal(numpts=numpts)
    w1n = (np.pi / 2) / pp0.T_90

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
    mrx1 = calc_macq_ideal_probe_relax4(sp, pp1)
    mrx2 = calc_macq_ideal_probe_relax4(sp, pp2)
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * del_w[np.newaxis, :])
    echo = (isoc @ mrx.T).T
    echo_integrals = trapezoid(echo, tvect, axis=1)
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
