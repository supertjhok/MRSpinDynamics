"""Ideal CPMG workflows with time-varying field offsets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import calc_rotation_matrix
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.acquisition import calc_macq_ideal_probe_relax4


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


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _offset_grid(numpts: int, maxoffs: float) -> np.ndarray:
    return np.linspace(-float(maxoffs), float(maxoffs), int(numpts))


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


def run_ideal_time_varying_cpmg_final(
    field_offsets: Iterable[float] | np.ndarray,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    pulse_name: str = "rect180",
    t1_seconds: float = 1e8,
    t2_seconds: float = 1e8,
    num_workers: int | None = 1,
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
    del_w = _offset_grid(numpts, maxoffs)
    w1n = (np.pi / 2) / pp0.T_90
    ref_tp_seconds, ref_phi, ref_amp = _pulse_definition(pulse_name, pp0.T_180)
    ref_duration = float(np.sum(ref_tp_seconds))
    echo_period = float(np.sum(pp0.tref))
    if ref_duration > echo_period:
        raise ValueError("refocusing pulse is longer than the default echo period")

    ref_pre = (echo_period - ref_duration) / 2
    ref_post = echo_period - ref_duration - ref_pre
    ref_tp_norm = w1n * ref_tp_seconds

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
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * (del_w + field_offsets[-1])[np.newaxis, :])
    echo = isoc @ mrx
    echo_integral = complex(trapezoid(echo, tvect))
    return IdealTimeVaryingCPMGResult(
        del_w=del_w,
        field_offsets=field_offsets,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integral=echo_integral,
        pulse_name=pulse_name,
    )


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
