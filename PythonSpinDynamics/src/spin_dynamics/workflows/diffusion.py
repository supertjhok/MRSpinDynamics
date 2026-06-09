"""Diffusion-aware matched-probe CPMG workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb10_diffusion,
    sim_spin_dynamics_arb10_diffusion_chunked,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rotation_matrix
from spin_dynamics.parameters import set_params_matched_orig
from spin_dynamics.probes.matched import matching_network_design2
from spin_dynamics.workflows.acquisition import _apply_receiver, _as_vector, _field
from spin_dynamics.workflows.cpmg import _calc_matched_pulse_shape, _echo_train_from_spectra


@dataclass(frozen=True)
class MatchedDiffusionCPMGResult:
    """Matched-probe diffusion-aware finite CPMG result."""

    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    q_value: float
    diffusion_coefficient: float
    diffusion_time: float
    gradient: float
    dz: float
    probe: str


@dataclass(frozen=True)
class MatchedDiffusionQSweepResult:
    """Q sweep result for matched-probe diffusion-aware CPMG."""

    values: np.ndarray
    value_label: str
    del_w: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    diffusion_coefficient: float
    diffusion_time: float
    gradient: float
    dz: float
    probe: str
    sweep: str


def calc_macq_matched_probe_relax_diffusion(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    *,
    apply_receiver: bool = True,
    num_workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffusion-aware matched-probe finite acquisition.

    This is an `arb10`-style Python analogue of MATLAB
    `calc_macq_diff/calc_macq_matched_probe_relax_diff*.m`. It expects
    precomputed pulse matrices in `pp.Rtot` and returns acquired spectra
    without the older acquisition-window convolution.
    """

    t_90 = float(_field(pp, "T_90"))
    params = {
        "tp": _field(pp, "tp"),
        "pul": _field(pp, "pul"),
        "amp": _field(pp, "amp"),
        "acq": _field(pp, "acq"),
        "grad": _field(pp, "grad"),
        "Rtot": _field(pp, "Rtot"),
        "del_w": _field(sp, "del_w"),
        "del_wg": _field(sp, "del_wg"),
        "w_1": _field(sp, "w_1"),
        "T1n": (np.pi / 2) * _as_vector(_field(sp, "T1"), np.float64) / t_90,
        "T2n": (np.pi / 2) * _as_vector(_field(sp, "T2"), np.float64) / t_90,
        "m0": _field(sp, "m0"),
        "mth": _field(sp, "mth"),
        "gamma": _field(sp, "gamma"),
        "gradient": _field(sp, "grad_physical"),
        "diffusion_coefficient": _field(sp, "D"),
        "diffusion_time": _field(sp, "Delta"),
    }
    if num_workers is None or int(num_workers) > 1:
        macq = sim_spin_dynamics_arb10_diffusion_chunked(params, num_workers=num_workers)
    else:
        macq = sim_spin_dynamics_arb10_diffusion(params)
    if not apply_receiver:
        return macq, macq
    return macq, _apply_receiver(macq, _field(sp, "tf2"), _field(sp, "w_1r"))


def run_matched_diffusion_cpmg(
    num_echoes: int = 5,
    echo_spacing_seconds: float = 1000e-6,
    t1_seconds: float = 100e-3,
    t2_seconds: float = 100e-3,
    dz: float = 0.001,
    diffusion_time: float = 1000e-6,
    t90_seconds: float = 100e-6,
    q_value: float = 50.0,
    *,
    numpts: int = 101,
    apply_receiver: bool = False,
    num_workers: int | None = 1,
) -> MatchedDiffusionCPMGResult:
    """Run a compact matched-probe diffusion-aware CPMG train.

    Mirrors the useful core of MATLAB
    `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`, with precomputed RF matrices
    and an `arb10`-style diffusion kernel.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing_seconds <= 2 * t90_seconds:
        raise ValueError("echo_spacing_seconds must be longer than T_180")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")
    if dz <= 0 or diffusion_time < 0 or t90_seconds <= 0:
        raise ValueError("dz and t90_seconds must be positive; diffusion_time must be non-negative")
    if q_value <= 0:
        raise ValueError("q_value must be positive")

    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    pp0 = pp0.__class__(
        **{
            **pp0.__dict__,
            "T_90": float(t90_seconds),
            "T_180": 2 * float(t90_seconds),
            "tacq": np.array([echo_spacing_seconds / 2], dtype=np.float64),
        }
    )
    gradient = float(sp0.grad)
    w1 = np.pi / (2 * t90_seconds)
    maxoffs = sp0.gamma * gradient * dz / w1
    del_w = np.linspace(-maxoffs, maxoffs, int(numpts))
    c1, c2 = matching_network_design2(sp0.L, q_value, sp0.f0, sp0.Rs)
    sp = {
        **sp0.__dict__,
        "C1": c1,
        "C2": c2,
        "Q": float(q_value),
        "R": 2 * np.pi * sp0.f0 * sp0.L / float(q_value),
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "D": sp0.D,
        "Delta": float(diffusion_time),
        "grad_physical": gradient,
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
        2 * pp0.T_90,
    )
    exc_minus_y = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        3 * np.pi / 2,
        1.0,
        2 * pp0.T_90,
    )[:3]
    ref_x = _calc_matched_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, 2 * pp0.T_90)[:3]
    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], exc_y_tp, exc_y_phi, exc_y_amp),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]
    sp["tf1"] = tf1
    sp["tf2"] = tf2

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    t_180 = 2 * t90_seconds
    tenc = np.array(
        [
            (np.pi / 2) * (diffusion_time - 0.5 * t90_seconds - 0.5 * t_180) / t90_seconds,
            np.pi,
            (np.pi / 2) * (diffusion_time - 0.5 * t90_seconds - 0.5 * t_180) / t90_seconds,
        ],
        dtype=np.float64,
    )
    if np.any(tenc[[0, 2]] < 0):
        raise ValueError("diffusion_time is too short for the encoding pulse block")
    penc = np.array([0, 3, 0], dtype=np.int64)
    aenc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0, 0], dtype=np.int64)
    grad_enc = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    tfp = (np.pi / 2) * (echo_spacing_seconds - t_180) / (2 * t90_seconds)
    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t90_seconds,
        "tp": np.concatenate([texc, tenc, tref]),
        "amp": np.concatenate([aexc, aenc, aref]),
        "acq": np.concatenate([acq_exc, acq_enc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_enc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, penc, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, penc, pref])}
    _macq1, mrx1 = calc_macq_matched_probe_relax_diffusion(
        sp,
        pp1,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    _macq2, mrx2 = calc_macq_matched_probe_relax_diffusion(
        sp,
        pp2,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    mrx = mrx1 - mrx2

    tacq = float((np.pi / 2) * (echo_spacing_seconds / 2) / t90_seconds)
    tdw = float((np.pi / 2) * pp0.tdw / t90_seconds)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1)
    return MatchedDiffusionCPMGResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        q_value=float(q_value),
        diffusion_coefficient=float(sp0.D),
        diffusion_time=float(diffusion_time),
        gradient=gradient,
        dz=float(dz),
        probe="matched",
    )


def run_matched_diffusion_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    num_echoes: int = 5,
    echo_spacing_seconds: float = 1000e-6,
    numpts: int = 101,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
) -> MatchedDiffusionQSweepResult:
    """Sweep matched-probe Q for the compact diffusion CPMG workflow."""

    values = np.asarray([20, 50, 80] if q_values is None else q_values)
    values = values.astype(np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("q_values must not be empty")

    def case_runner(q_value: float) -> MatchedDiffusionCPMGResult:
        return run_matched_diffusion_cpmg(
            num_echoes=num_echoes,
            echo_spacing_seconds=echo_spacing_seconds,
            q_value=q_value,
            numpts=numpts,
            apply_receiver=False,
            num_workers=num_workers,
        )

    workers = 1 if sweep_workers is None else int(sweep_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in values]))

    return MatchedDiffusionQSweepResult(
        values=values,
        value_label="coil Q",
        del_w=rows[0].del_w,
        echo=np.stack([row.echo for row in rows], axis=0),
        tvect=rows[0].tvect,
        echo_integrals=np.stack([row.echo_integrals for row in rows], axis=0),
        sequence_time=rows[0].sequence_time,
        diffusion_coefficient=rows[0].diffusion_coefficient,
        diffusion_time=rows[0].diffusion_time,
        gradient=rows[0].gradient,
        dz=rows[0].dz,
        probe="matched",
        sweep="diffusion_q",
    )
