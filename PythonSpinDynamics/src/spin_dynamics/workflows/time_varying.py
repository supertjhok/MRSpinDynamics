"""Ideal CPMG workflows with time-varying field offsets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spin_dynamics.core.isochromats import check_rephasing
from spin_dynamics.core.numerics import trapezoid
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
    _calc_matched_pulse_shape,
    _calc_tuned_pulse_shape,
    _calc_untuned_pulse_shape,
    _maybe_refine_numpts,
)


@dataclass(frozen=True)
class IdealTimeVaryingCPMGResult:
    """Final-echo result for ideal CPMG with time-varying B0 offsets."""

    del_w: np.ndarray
    field_offsets: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integral: complex
    pulse_name: str


@dataclass(frozen=True)
class IdealTimeVaryingSweepResult:
    """Amplitude sweep result for ideal time-varying-field CPMG."""

    amplitudes: np.ndarray
    waveform: np.ndarray
    del_w: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    matched_filter: np.ndarray
    matched_signal: np.ndarray
    pulse_name: str


@dataclass(frozen=True)
class ProbeTimeVaryingCPMGResult:
    """Final-echo result for probe-aware CPMG with time-varying B0 offsets."""

    probe: str
    del_w: np.ndarray
    field_offsets: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integral: complex
    q_value: float | None
    mistuning_offset: float | None


@dataclass(frozen=True)
class ProbeTimeVaryingSweepResult:
    """Amplitude sweep result for probe-aware time-varying-field CPMG."""

    probe: str
    amplitudes: np.ndarray
    waveform: np.ndarray
    del_w: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    matched_filter: np.ndarray
    matched_signal: np.ndarray
    q_value: float | None
    mistuning_offset: float | None


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _offset_grid(numpts: int, maxoffs: float) -> np.ndarray:
    return np.linspace(-float(maxoffs), float(maxoffs), int(numpts))


def _checked_offset_grid(
    numpts: int,
    maxoffs: float,
    max_time: float,
    *,
    auto_refine_grid: bool,
    rephase_safety_factor: float,
    rephase_action: str,
) -> np.ndarray:
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
            max_time=max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    return del_w


def _pulse_definition(name: str, t_180: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if name == "rect180":
        return (
            np.array([t_180], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    if name == "rect135":
        return (
            np.array([0.75 * t_180], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )
    if name == "rp2":
        return (
            t_180 * np.array([0.14, 0.72, 0.14], dtype=np.float64),
            np.pi * np.array([1.0, 0.0, 1.0], dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )
    raise ValueError("pulse_name must be 'rect180', 'rect135', or 'rp2'")


def _as_waveform(field_offsets: Iterable[float] | np.ndarray, num_echoes: int) -> np.ndarray:
    waveform = np.asarray(field_offsets, dtype=np.float64).reshape(-1)
    if waveform.size != int(num_echoes):
        raise ValueError("field_offsets must have length num_echoes")
    return waveform


def _echo_from_spectrum(
    mrx: np.ndarray,
    del_w: np.ndarray,
    final_offset: float,
    tacq: float,
    tdw: float,
) -> tuple[np.ndarray, np.ndarray, complex]:
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * (del_w + final_offset)[np.newaxis, :])
    echo = isoc @ mrx
    return echo, tvect, complex(trapezoid(echo, tvect))


def run_ideal_time_varying_cpmg_final(
    field_offsets: Iterable[float] | np.ndarray,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    pulse_name: str = "rect180",
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> IdealTimeVaryingCPMGResult:
    """Run the final echo of an ideal CPMG train with per-echo B0 offsets.

    `field_offsets` are normalized angular offsets, matching MATLAB's
    `w_0t = gamma * B_0t / w_1n` convention in
    `time_varying_field/sim_cpmg_ideal_tv_final.m`.
    """

    field_offsets = np.asarray(field_offsets, dtype=np.float64).reshape(-1)
    num_echoes = field_offsets.size
    if num_echoes <= 0:
        raise ValueError("field_offsets must not be empty")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_ideal(numpts=numpts)
    w1n = (np.pi / 2) / pp0.T_90
    ref_tp_seconds, ref_phi, ref_amp = _pulse_definition(pulse_name, pp0.T_180)
    ref_duration = float(np.sum(ref_tp_seconds))
    echo_period = float(np.sum(pp0.tref))
    if ref_duration > echo_period:
        raise ValueError("refocusing pulse is longer than the default echo period")

    ref_pre = (echo_period - ref_duration) / 2
    ref_post = echo_period - ref_duration - ref_pre
    ref_tp_norm = w1n * ref_tp_seconds
    max_time = float(
        np.pi / 2
        + w1n * pp0.tcorr
        + num_echoes * (w1n * ref_pre + float(np.sum(ref_tp_norm)) + w1n * ref_post)
    )
    del_w = _checked_offset_grid(
        numpts,
        maxoffs,
        max_time,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )

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
                ref_phi,
                ref_amp,
            )
        )

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

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
    echo, tvect, echo_integral = _echo_from_spectrum(mrx, del_w, field_offsets[-1], tacq, tdw)
    return IdealTimeVaryingCPMGResult(
        del_w=del_w,
        field_offsets=field_offsets,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integral=echo_integral,
        pulse_name=pulse_name,
    )


def _prepare_probe_parameters(
    probe: str,
    numpts: int,
    maxoffs: float,
    q_value: float | None,
    mistuning_offset: float | None,
    *,
    num_echoes_for_rephase: int | None = None,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> tuple[dict[str, Any], Any]:
    if probe == "tuned":
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
    elif probe == "untuned":
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
    elif probe == "matched":
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
    else:
        raise ValueError("probe must be 'tuned', 'untuned', or 'matched'")

    if num_echoes_for_rephase is None:
        del_w = _offset_grid(numpts, maxoffs)
    else:
        tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
        max_time = float(
            np.pi / 2
            + (np.pi / 2) * pp0.tcorr / pp0.T_90
            + int(num_echoes_for_rephase) * (tfp + np.pi + tfp)
        )
        del_w = _checked_offset_grid(
            numpts,
            maxoffs,
            max_time,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )
        numpts = int(del_w.size)
    sp = {
        **sp0.__dict__,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }
    if probe == "matched":
        c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
        sp["C1"] = c1
        sp["C2"] = c2
    return sp, pp0


def _probe_pulses(
    probe: str,
    sp: dict[str, Any],
    pp0: Any,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]:
    if probe == "tuned":
        sp["tf"] = tuned_probe_rx_tf(sp, pp0)
        return (
            _calc_tuned_pulse_shape(sp, pp0, pp0.T_90, np.pi / 2, 1.0, 2 * pp0.T_90),
            _calc_tuned_pulse_shape(sp, pp0, pp0.T_90, 3 * np.pi / 2, 1.0, 2 * pp0.T_90),
            _calc_tuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, 2 * pp0.T_90),
        )
    if probe == "untuned":
        sp["tf"] = untuned_probe_rx_tf(sp, pp0)
        return (
            _calc_untuned_pulse_shape(sp, pp0, pp0.T_90, np.pi / 2, 1.0, pp0.trd),
            _calc_untuned_pulse_shape(sp, pp0, pp0.T_90, 3 * np.pi / 2, 1.0, pp0.trd),
            _calc_untuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd),
        )

    exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        np.pi / 2,
        1.0,
        pp0.trd,
    )
    sp["tf1"] = tf1
    sp["tf2"] = tf2
    return (
        (exc_y_tp, exc_y_phi, exc_y_amp),
        _calc_matched_pulse_shape(sp, pp0, pp0.T_90, 3 * np.pi / 2, 1.0, pp0.trd)[:3],
        _calc_matched_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)[:3],
    )


def _run_probe_time_varying_cpmg_final(
    probe: str,
    field_offsets: Iterable[float] | np.ndarray,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> ProbeTimeVaryingCPMGResult:
    field_offsets = np.asarray(field_offsets, dtype=np.float64).reshape(-1)
    num_echoes = field_offsets.size
    if num_echoes <= 0:
        raise ValueError("field_offsets must not be empty")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp, pp0 = _prepare_probe_parameters(
        probe,
        numpts,
        maxoffs,
        q_value,
        mistuning_offset,
        num_echoes_for_rephase=num_echoes,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )
    del_w = sp["del_w"]
    sp["T1"] = t1_seconds * np.ones_like(del_w)
    sp["T2"] = t2_seconds * np.ones_like(del_w)
    exc_y, exc_minus_y, ref_x = _probe_pulses(probe, sp, pp0)

    rtot = [
        calc_rotation_matrix(del_w, sp["w_1"], *exc_y),
        calc_rotation_matrix(del_w, sp["w_1"], *exc_minus_y),
    ]
    for offset in field_offsets:
        rtot.append(calc_rotation_matrix(del_w + offset, sp["w_1"], *ref_x))

    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.empty(3 * num_echoes, dtype=np.float64)
    pref = np.empty(3 * num_echoes, dtype=np.int64)
    aref = np.empty(3 * num_echoes, dtype=np.float64)
    acq_ref = np.zeros(3 * num_echoes, dtype=np.int64)
    grad_ref = np.empty(3 * num_echoes, dtype=np.float64)
    for idx, offset in enumerate(field_offsets):
        base = 3 * idx
        tref[base : base + 3] = [tfp, np.pi, tfp]
        pref[base : base + 3] = [0, idx + 3, 0]
        aref[base : base + 3] = [0.0, 1.0, 0.0]
        grad_ref[base : base + 3] = offset
    acq_ref[-1] = 1

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

    if probe == "tuned":
        _macq1, mrx1 = calc_macq_tuned_probe_relax4(sp, pp1, num_workers=num_workers)
        _macq2, mrx2 = calc_macq_tuned_probe_relax4(sp, pp2, num_workers=num_workers)
    elif probe == "untuned":
        _macq1, mrx1 = calc_macq_untuned_probe_relax4(sp, pp1, num_workers=num_workers)
        _macq2, mrx2 = calc_macq_untuned_probe_relax4(sp, pp2, num_workers=num_workers)
    else:
        _macq1, mrx1 = calc_macq_matched_probe_relax4(sp, pp1, num_workers=num_workers)
        _macq2, mrx2 = calc_macq_matched_probe_relax4(sp, pp2, num_workers=num_workers)

    mrx = ((mrx1 - mrx2) / 2)[0]
    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    echo, tvect, echo_integral = _echo_from_spectrum(mrx, del_w, field_offsets[-1], tacq, tdw)
    return ProbeTimeVaryingCPMGResult(
        probe=probe,
        del_w=del_w,
        field_offsets=field_offsets,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integral=echo_integral,
        q_value=q_value,
        mistuning_offset=mistuning_offset,
    )


def run_tuned_time_varying_cpmg_final(
    field_offsets: Iterable[float] | np.ndarray,
    **kwargs: Any,
) -> ProbeTimeVaryingCPMGResult:
    """Run the final echo of a tuned-probe CPMG train with per-echo B0 offsets."""

    return _run_probe_time_varying_cpmg_final("tuned", field_offsets, **kwargs)


def run_untuned_time_varying_cpmg_final(
    field_offsets: Iterable[float] | np.ndarray,
    **kwargs: Any,
) -> ProbeTimeVaryingCPMGResult:
    """Run the final echo of an untuned-probe CPMG train with per-echo B0 offsets."""

    return _run_probe_time_varying_cpmg_final("untuned", field_offsets, **kwargs)


def run_matched_time_varying_cpmg_final(
    field_offsets: Iterable[float] | np.ndarray,
    **kwargs: Any,
) -> ProbeTimeVaryingCPMGResult:
    """Run the final echo of a matched-probe CPMG train with per-echo B0 offsets."""

    return _run_probe_time_varying_cpmg_final("matched", field_offsets, **kwargs)


def sinusoidal_field_waveform(num_echoes: int, cycles: float = 0.5) -> np.ndarray:
    """Return the default sinusoidal normalized B0 waveform used by v0crit."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    return np.sin(2 * np.pi * float(cycles) * np.linspace(0, 1, int(num_echoes)))


def run_ideal_time_varying_amplitude_sweep(
    amplitudes: Iterable[float] | np.ndarray | None = None,
    *,
    waveform: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 16,
    numpts: int = 101,
    maxoffs: float = 10.0,
    pulse_name: str = "rect180",
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> IdealTimeVaryingSweepResult:
    """Sweep normalized B0 fluctuation amplitude for ideal CPMG final echoes."""

    amp_values = np.asarray(
        np.linspace(0, 3, 16) if amplitudes is None else amplitudes,
        dtype=np.float64,
    ).reshape(-1)
    if amp_values.size == 0:
        raise ValueError("amplitudes must not be empty")
    if waveform is None:
        base_waveform = sinusoidal_field_waveform(num_echoes)
    else:
        base_waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
        if base_waveform.size == 0:
            raise ValueError("waveform must not be empty")
        num_echoes = int(base_waveform.size)

    def case_runner(amplitude: float) -> IdealTimeVaryingCPMGResult:
        return run_ideal_time_varying_cpmg_final(
            amplitude * base_waveform,
            numpts=numpts,
            maxoffs=maxoffs,
            pulse_name=pulse_name,
            num_workers=1,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    workers = 1 if num_workers is None else int(num_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in amp_values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in amp_values]))

    reference = run_ideal_time_varying_cpmg_final(
        np.zeros_like(base_waveform),
        numpts=numpts,
        maxoffs=maxoffs,
        pulse_name=pulse_name,
        num_workers=1,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )
    norm = np.sqrt(trapezoid(np.abs(reference.echo) ** 2, reference.tvect))
    matched_filter = np.conj(reference.echo) / norm
    echo = np.stack([row.echo for row in rows], axis=0)
    echo_integrals = np.asarray([row.echo_integral for row in rows], dtype=np.complex128)
    matched_signal = trapezoid(echo * matched_filter[np.newaxis, :], reference.tvect, axis=1)
    return IdealTimeVaryingSweepResult(
        amplitudes=amp_values,
        waveform=base_waveform,
        del_w=reference.del_w,
        echo=echo,
        tvect=reference.tvect,
        echo_integrals=echo_integrals,
        matched_filter=matched_filter,
        matched_signal=matched_signal / norm,
        pulse_name=pulse_name,
    )


def _run_probe_time_varying_amplitude_sweep(
    probe: str,
    amplitudes: Iterable[float] | np.ndarray | None = None,
    *,
    waveform: Iterable[float] | np.ndarray | None = None,
    num_echoes: int = 16,
    numpts: int = 101,
    maxoffs: float = 10.0,
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> ProbeTimeVaryingSweepResult:
    amp_values = np.asarray(
        np.linspace(0, 3, 16) if amplitudes is None else amplitudes,
        dtype=np.float64,
    ).reshape(-1)
    if amp_values.size == 0:
        raise ValueError("amplitudes must not be empty")
    if waveform is None:
        base_waveform = sinusoidal_field_waveform(num_echoes)
    else:
        base_waveform = np.asarray(waveform, dtype=np.float64).reshape(-1)
        if base_waveform.size == 0:
            raise ValueError("waveform must not be empty")
        num_echoes = int(base_waveform.size)

    def case_runner(amplitude: float) -> ProbeTimeVaryingCPMGResult:
        return _run_probe_time_varying_cpmg_final(
            probe,
            amplitude * base_waveform,
            numpts=numpts,
            maxoffs=maxoffs,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            q_value=q_value,
            mistuning_offset=mistuning_offset,
            num_workers=1,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    workers = 1 if num_workers is None else int(num_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in amp_values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in amp_values]))

    reference = _run_probe_time_varying_cpmg_final(
        probe,
        np.zeros_like(base_waveform),
        numpts=numpts,
        maxoffs=maxoffs,
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
        q_value=q_value,
        mistuning_offset=mistuning_offset,
        num_workers=1,
        auto_refine_grid=auto_refine_grid,
        rephase_safety_factor=rephase_safety_factor,
        rephase_action=rephase_action,
    )
    norm = np.sqrt(trapezoid(np.abs(reference.echo) ** 2, reference.tvect))
    matched_filter = np.conj(reference.echo) / norm
    echo = np.stack([row.echo for row in rows], axis=0)
    echo_integrals = np.asarray([row.echo_integral for row in rows], dtype=np.complex128)
    matched_signal = trapezoid(echo * matched_filter[np.newaxis, :], reference.tvect, axis=1)
    return ProbeTimeVaryingSweepResult(
        probe=probe,
        amplitudes=amp_values,
        waveform=base_waveform,
        del_w=reference.del_w,
        echo=echo,
        tvect=reference.tvect,
        echo_integrals=echo_integrals,
        matched_filter=matched_filter,
        matched_signal=matched_signal / norm,
        q_value=q_value,
        mistuning_offset=mistuning_offset,
    )


def run_tuned_time_varying_amplitude_sweep(
    amplitudes: Iterable[float] | np.ndarray | None = None,
    **kwargs: Any,
) -> ProbeTimeVaryingSweepResult:
    """Sweep normalized B0 fluctuation amplitude for tuned-probe CPMG."""

    return _run_probe_time_varying_amplitude_sweep("tuned", amplitudes, **kwargs)


def run_untuned_time_varying_amplitude_sweep(
    amplitudes: Iterable[float] | np.ndarray | None = None,
    **kwargs: Any,
) -> ProbeTimeVaryingSweepResult:
    """Sweep normalized B0 fluctuation amplitude for untuned-probe CPMG."""

    return _run_probe_time_varying_amplitude_sweep("untuned", amplitudes, **kwargs)


def run_matched_time_varying_amplitude_sweep(
    amplitudes: Iterable[float] | np.ndarray | None = None,
    **kwargs: Any,
) -> ProbeTimeVaryingSweepResult:
    """Sweep normalized B0 fluctuation amplitude for matched-probe CPMG."""

    return _run_probe_time_varying_amplitude_sweep("matched", amplitudes, **kwargs)
