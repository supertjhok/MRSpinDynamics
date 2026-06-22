"""WURST pulse and matched-probe inversion/CPMG workflows.

The WURST waveform is built by :func:`spin_dynamics.pulses.create_wurst_pulse`:
the amplitude envelope is ``1 - |cos(pi t / T)|**order`` (zero at both ends, flat
top for large ``order``) and the frequency offset is swept *linearly* and
symmetrically about the carrier across ``sweep_width_rad_per_s``, which by
``cumsum`` integration gives the characteristic quadratic phase. ``order`` and
``sweep_width`` therefore set the offset bandwidth covered.

These workflows assume the sweep is *adiabatic*: the adiabaticity factor
``Q = omega_1**2 / |d(omega_offset)/dt|`` must stay well above 1 across the band
for clean inversion. That condition is the caller's responsibility -- it is
**not** computed or enforced here -- so reduce the sweep rate (longer duration or
narrower sweep) or raise ``omega_1`` if inversion is incomplete.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spin_dynamics.core.isochromats import check_rephasing
from spin_dynamics.core.rotations import calc_rotation_matrix, sim_spin_dynamics_exc
from spin_dynamics.parameters import set_params_matched_orig
from spin_dynamics.probes.matched import find_coil_current_wurst, matching_network_design2
from spin_dynamics.pulses import WURSTPulse, create_wurst_pulse
from spin_dynamics.workflows.acquisition import calc_macq_matched_probe_relax4
from spin_dynamics.workflows.cpmg import (
    _calc_matched_pulse_shape,
    _echo_train_from_spectra,
    _maybe_refine_numpts,
)


@dataclass(frozen=True)
class WURSTInversionResult:
    """Isochromat magnetization after a WURST inversion pulse."""

    del_w: np.ndarray
    pulse: WURSTPulse
    magnetization: np.ndarray
    mz: np.ndarray
    transverse: np.ndarray
    probe: str
    rotating_time: np.ndarray | None = None
    rotating_current: np.ndarray | None = None
    receiver_tf_signal: np.ndarray | None = None


@dataclass(frozen=True)
class MatchedWURSTCPMGResult:
    """Matched-probe WURST excitation followed by a finite CPMG train."""

    del_w: np.ndarray
    pulse: WURSTPulse
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    rotating_time: np.ndarray
    rotating_current: np.ndarray
    probe: str
    q_value: float


def _offset_grid(numpts: int, maxoffs: float) -> np.ndarray:
    return np.linspace(-float(maxoffs), float(maxoffs), int(numpts))


def _default_wurst_duration(t90_seconds: float) -> float:
    return 20.0 * float(t90_seconds)


def _prepare_wurst_pulse(
    *,
    t90_seconds: float,
    duration_seconds: float | None,
    sweep_width_normalized: float,
    num_steps: int,
    order: int,
    amplitude: float,
    initial_phase: float,
) -> WURSTPulse:
    if t90_seconds <= 0:
        raise ValueError("t90_seconds must be positive")
    if sweep_width_normalized < 0:
        raise ValueError("sweep_width_normalized must be non-negative")
    w1n = (np.pi / 2) / float(t90_seconds)
    duration = _default_wurst_duration(t90_seconds) if duration_seconds is None else duration_seconds
    return create_wurst_pulse(
        duration_seconds=float(duration),
        sweep_width_rad_per_s=float(sweep_width_normalized) * w1n,
        num_steps=num_steps,
        order=order,
        amplitude=amplitude,
        initial_phase=initial_phase,
    )


def _simulate_ideal_pulse(
    pulse: WURSTPulse,
    del_w: np.ndarray,
    t90_seconds: float,
) -> np.ndarray:
    tp_norm = ((np.pi / 2) / float(t90_seconds)) * pulse.duration
    return sim_spin_dynamics_exc(tp_norm, pulse.phase, pulse.amplitude, del_w)


def run_ideal_wurst_inversion(
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    t90_seconds: float = 25e-6,
    duration_seconds: float | None = None,
    sweep_width_normalized: float = 20.0,
    num_steps: int = 256,
    order: int = 20,
    amplitude: float = 1.0,
    initial_phase: float = np.pi / 2,
) -> WURSTInversionResult:
    """Run an ideal-probe WURST inversion pulse over a uniform offset grid."""

    pulse = _prepare_wurst_pulse(
        t90_seconds=t90_seconds,
        duration_seconds=duration_seconds,
        sweep_width_normalized=sweep_width_normalized,
        num_steps=num_steps,
        order=order,
        amplitude=amplitude,
        initial_phase=initial_phase,
    )
    del_w = _offset_grid(numpts, maxoffs)
    magnetization = _simulate_ideal_pulse(pulse, del_w, t90_seconds)
    transverse = magnetization[0, :] + 1j * magnetization[1, :]
    return WURSTInversionResult(
        del_w=del_w,
        pulse=pulse,
        magnetization=magnetization,
        mz=np.real(magnetization[2, :]),
        transverse=transverse,
        probe="ideal",
    )


def _prepare_matched_system(
    *,
    numpts: int,
    maxoffs: float,
    q_value: float | None,
    t1_seconds: float,
    t2_seconds: float,
) -> tuple[dict[str, Any], Any]:
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")
    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    sp0 = replace(sp0, R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q)
    del_w = _offset_grid(numpts, maxoffs)
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
    return sp, pp0


def _matched_wurst_shape(
    sp: Mapping[str, Any],
    pp: Any,
    pulse: WURSTPulse,
    *,
    drive_phase: float,
    delay_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t90 = float(pp.T_90)
    delay_normalized = (np.pi / 2) * float(delay_seconds) / t90
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tp": np.concatenate([pulse.duration, [float(delay_seconds)]]),
        "phi": np.concatenate(
            [np.full(pulse.duration.size, float(drive_phase), dtype=np.float64), [0.0]]
        ),
        "amp": np.concatenate([pulse.amplitude, [0.0]]),
        "freq": np.concatenate([pulse.frequency_offset, [0.0]]),
    }
    tvect, icr, tf1, tf2 = find_coil_current_wurst(sp, pp_curr)
    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / t90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp[amp < float(pp.amp_zero)] = 0
    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
        tf1,
        tf2,
        tvect,
        icr,
    )


def run_matched_wurst_inversion(
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    q_value: float | None = None,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    duration_seconds: float | None = None,
    sweep_width_normalized: float = 20.0,
    num_steps: int = 128,
    order: int = 20,
    amplitude: float = 1.0,
    initial_phase: float = np.pi / 2,
) -> WURSTInversionResult:
    """Run a matched-probe WURST inversion pulse over a uniform offset grid."""

    sp, pp0 = _prepare_matched_system(
        numpts=numpts,
        maxoffs=maxoffs,
        q_value=q_value,
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
    )
    pulse = _prepare_wurst_pulse(
        t90_seconds=pp0.T_90,
        duration_seconds=duration_seconds,
        sweep_width_normalized=sweep_width_normalized,
        num_steps=num_steps,
        order=order,
        amplitude=amplitude,
        initial_phase=initial_phase,
    )
    tp, phi, amp, _tf1, tf2, rotating_time, rotating_current = _matched_wurst_shape(
        sp,
        pp0,
        pulse,
        drive_phase=initial_phase,
        delay_seconds=2 * pp0.T_90,
    )
    magnetization = sim_spin_dynamics_exc(tp, phi, amp, sp["del_w"])
    transverse = magnetization[0, :] + 1j * magnetization[1, :]
    return WURSTInversionResult(
        del_w=sp["del_w"],
        pulse=pulse,
        magnetization=magnetization,
        mz=np.real(magnetization[2, :]),
        transverse=transverse,
        probe="matched",
        rotating_time=rotating_time,
        rotating_current=rotating_current,
        receiver_tf_signal=tf2,
    )


def run_matched_wurst_cpmg(
    *,
    num_echoes: int = 4,
    numpts: int = 101,
    maxoffs: float = 10.0,
    q_value: float | None = None,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    duration_seconds: float | None = None,
    sweep_width_normalized: float = 20.0,
    num_steps: int = 128,
    order: int = 20,
    amplitude: float = 1.0,
    initial_phase: float = np.pi / 2,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> MatchedWURSTCPMGResult:
    """Run matched-probe WURST excitation followed by rectangular CPMG echoes."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    t90 = float(pp0.T_90)
    pulse_duration = _default_wurst_duration(t90) if duration_seconds is None else duration_seconds
    wurst_time = (np.pi / 2) * float(pulse_duration) / t90
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * t90)
    max_time = float(wurst_time + int(num_echoes) * (tfp + np.pi + tfp))
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )

    sp, pp0 = _prepare_matched_system(
        numpts=numpts,
        maxoffs=maxoffs,
        q_value=q_value,
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
    )
    if rephase_action != "ignore":
        check_rephasing(
            sp["del_w"],
            max_time=max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    pulse = _prepare_wurst_pulse(
        t90_seconds=pp0.T_90,
        duration_seconds=pulse_duration,
        sweep_width_normalized=sweep_width_normalized,
        num_steps=num_steps,
        order=order,
        amplitude=amplitude,
        initial_phase=initial_phase,
    )
    wurst_tp, wurst_phi, wurst_amp, _tf1, tf2, rotating_time, rotating_current = (
        _matched_wurst_shape(
            sp,
            pp0,
            pulse,
            drive_phase=initial_phase,
            delay_seconds=2 * pp0.T_90,
        )
    )
    ref_x = _calc_matched_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)[:3]
    del_w = sp["del_w"]
    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], wurst_tp, wurst_phi, wurst_amp),
        calc_rotation_matrix(del_w, sp["w_1"], wurst_tp, wurst_phi + np.pi, wurst_amp),
        calc_rotation_matrix(del_w, sp["w_1"], *ref_x),
    ]

    texc = np.array([float(np.sum(wurst_tp))], dtype=np.float64)
    aexc = np.array([1.0], dtype=np.float64)
    pexc1 = np.array([1], dtype=np.int64)
    pexc2 = np.array([2], dtype=np.int64)
    acq_exc = np.array([0], dtype=np.int64)
    grad_exc = np.array([0.0], dtype=np.float64)
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
    sp["tf2"] = tf2
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, pref])}
    _macq1, mrx1 = calc_macq_matched_probe_relax4(sp, pp1, num_workers=num_workers)
    _macq2, mrx2 = calc_macq_matched_probe_relax4(sp, pp2, num_workers=num_workers)
    mrx = (mrx1 - mrx2) / 2
    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = float(pulse_duration) + np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )
    return MatchedWURSTCPMGResult(
        del_w=del_w,
        pulse=pulse,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        rotating_time=rotating_time,
        rotating_current=rotating_current,
        probe="matched_wurst",
        q_value=float(sp["Q"]),
    )
