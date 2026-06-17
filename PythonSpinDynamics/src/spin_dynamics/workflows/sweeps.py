"""Probe-parameter sweep workflows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.probes.matched import calc_masy_matched_nut, calc_masy_matched_probe_orig
from spin_dynamics.probes.tuned import calc_masy_tuned_probe_lp_orig
from spin_dynamics.workflows.cpmg import (
    CPMGTrainResult,
    run_matched_cpmg_train,
    run_tuned_cpmg_train,
    run_untuned_cpmg_train,
)


@dataclass(frozen=True)
class CPMGParameterSweepResult:
    """Result for probe-parameter CPMG sweeps."""

    values: np.ndarray
    value_label: str
    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: np.ndarray
    probe: str
    sweep: str


@dataclass(frozen=True)
class ZMagnetizationSweepResult:
    """Result for matched-probe z-magnetization sweeps."""

    values: np.ndarray
    value_label: str
    del_w: np.ndarray
    mz: np.ndarray
    tvect: np.ndarray
    probe: str
    sweep: str


@dataclass(frozen=True)
class CPMGFiniteParameterSweepResult:
    """Result for finite-train probe-parameter sweeps."""

    values: np.ndarray
    value_label: str
    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str
    sweep: str


def _run_sweep_cases(
    values: np.ndarray,
    case_runner: Callable[[float], tuple[np.ndarray, np.ndarray, float]],
    num_workers: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    workers = 1 if num_workers is None else int(num_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in values]))

    mrx = np.stack([row[0] for row in rows], axis=0)
    echo = np.stack([row[1] for row in rows], axis=0)
    snr = np.asarray([row[2] for row in rows], dtype=np.float64)
    return mrx, echo, snr


def _run_z_sweep_cases(
    values: np.ndarray,
    case_runner: Callable[[float], tuple[np.ndarray, np.ndarray]],
    num_workers: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    workers = 1 if num_workers is None else int(num_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in values]))

    mz = np.stack([row[0] for row in rows], axis=0)
    return mz, rows[0][1]


def _run_finite_sweep_cases(
    values: np.ndarray,
    case_runner: Callable[[float], CPMGTrainResult],
    num_workers: int | None,
) -> list[CPMGTrainResult]:
    workers = 1 if num_workers is None else int(num_workers)
    if workers <= 1:
        return [case_runner(float(value)) for value in values]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(case_runner, [float(value) for value in values]))


def _finite_sweep_result(
    values: np.ndarray,
    value_label: str,
    probe: str,
    sweep: str,
    rows: list[CPMGTrainResult],
) -> CPMGFiniteParameterSweepResult:
    return CPMGFiniteParameterSweepResult(
        values=values,
        value_label=value_label,
        del_w=rows[0].del_w,
        mrx=np.stack([row.mrx for row in rows], axis=0),
        echo=np.stack([row.echo for row in rows], axis=0),
        tvect=rows[0].tvect,
        echo_integrals=np.stack([row.echo_integrals for row in rows], axis=0),
        sequence_time=rows[0].sequence_time,
        probe=probe,
        sweep=sweep,
    )




def _as_values(values: Iterable[float] | np.ndarray) -> np.ndarray:
    out = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    out = out.reshape(-1)
    if out.size == 0:
        raise ValueError("sweep values must not be empty")
    return out


def _silence_tuned_sp(sp: Any) -> Any:
    return replace(
        sp,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )


def _silence_matched_sp(sp: Any) -> Any:
    return replace(
        sp,
        plt_tx=0,
        plt_rx=0,
        plt_sequence=0,
        plt_axis=0,
        plt_mn=0,
        plt_echo=0,
    )


def run_tuned_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
) -> CPMGParameterSweepResult:
    """Sweep tuned-probe coil Q for the original/reference CPMG path."""

    values = _as_values(np.linspace(1, 100, 100) if q_values is None else q_values)
    params, sp0, pp = set_params_tuned_orig(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    sp0 = _silence_tuned_sp(replace(sp0, del_w=del_w, maxoffs=float(maxoffs)))

    def case_runner(q_value: float) -> tuple[np.ndarray, np.ndarray, float]:
        if q_value <= 0:
            raise ValueError("Q values must be positive")
        resistance = 2 * np.pi * sp0.f0 * sp0.L / q_value
        sp = replace(sp0, Q=q_value, R=resistance)
        mrx, _masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
        echo, _tvect = calc_time_domain_echo(mrx, del_w)
        return mrx, echo, snr

    mrx, echo, snr = _run_sweep_cases(values, case_runner, num_workers)
    _echo0, tvect = calc_time_domain_echo(mrx[0], del_w)
    return CPMGParameterSweepResult(
        values=values,
        value_label="coil Q",
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="tuned",
        sweep="q",
    )


def run_matched_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
) -> CPMGParameterSweepResult:
    """Sweep matched-probe coil Q for the original/reference CPMG path."""

    values = _as_values(np.linspace(10, 100, 100) if q_values is None else q_values)
    sp0, pp = set_params_matched_orig(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    sp0 = _silence_matched_sp(replace(sp0, del_w=del_w, maxoffs=float(maxoffs)))

    def case_runner(q_value: float) -> tuple[np.ndarray, np.ndarray, float]:
        if q_value <= 0:
            raise ValueError("Q values must be positive")
        resistance = 2 * np.pi * sp0.f0 * sp0.L / q_value
        sp = replace(sp0, Q=q_value, R=resistance)
        mrx, _masy, snr = calc_masy_matched_probe_orig(sp, pp)
        echo, _tvect = calc_time_domain_echo(mrx, del_w)
        return mrx, echo, snr

    mrx, echo, snr = _run_sweep_cases(values, case_runner, num_workers)
    _echo0, tvect = calc_time_domain_echo(mrx[0], del_w)
    return CPMGParameterSweepResult(
        values=values,
        value_label="coil Q",
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="matched",
        sweep="q",
    )


def run_tuned_mistuning_sweep(
    offsets: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
) -> CPMGParameterSweepResult:
    """Sweep tuned-probe frequency error in units of `fin / Q`."""

    values = _as_values(np.linspace(-5, 5, 51) if offsets is None else offsets)
    params, sp0, pp = set_params_tuned_orig(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    sp0 = _silence_tuned_sp(replace(sp0, del_w=del_w, maxoffs=float(maxoffs)))

    def case_runner(offset: float) -> tuple[np.ndarray, np.ndarray, float]:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * offset
        if f0 <= 0:
            raise ValueError("mistuning offset produced non-positive f0")
        resistance = 2 * np.pi * f0 * sp0.L / sp0.Q
        capacitance = 1 / ((2 * np.pi * f0) ** 2 * sp0.L)
        sp = replace(sp0, f0=f0, R=resistance, C=capacitance)
        mrx, _masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
        echo, _tvect = calc_time_domain_echo(mrx, del_w)
        return mrx, echo, snr

    mrx, echo, snr = _run_sweep_cases(values, case_runner, num_workers)
    _echo0, tvect = calc_time_domain_echo(mrx[0], del_w)
    return CPMGParameterSweepResult(
        values=values,
        value_label="frequency error (fin/Q)",
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="tuned",
        sweep="mistuning",
    )


def run_matched_mistuning_sweep(
    offsets: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
) -> CPMGParameterSweepResult:
    """Sweep matched-probe frequency error in units of `fin / Q`."""

    values = _as_values(np.linspace(-5, 5, 51) if offsets is None else offsets)
    sp0, pp = set_params_matched_orig(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    sp0 = _silence_matched_sp(replace(sp0, del_w=del_w, maxoffs=float(maxoffs)))

    def case_runner(offset: float) -> tuple[np.ndarray, np.ndarray, float]:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * offset
        if f0 <= 0:
            raise ValueError("mistuning offset produced non-positive f0")
        resistance = 2 * np.pi * f0 * sp0.L / sp0.Q
        sp = replace(sp0, f0=f0, R=resistance)
        mrx, _masy, snr = calc_masy_matched_probe_orig(sp, pp)
        echo, _tvect = calc_time_domain_echo(mrx, del_w)
        return mrx, echo, snr

    mrx, echo, snr = _run_sweep_cases(values, case_runner, num_workers)
    _echo0, tvect = calc_time_domain_echo(mrx[0], del_w)
    return CPMGParameterSweepResult(
        values=values,
        value_label="frequency error (fin/Q)",
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="matched",
        sweep="mistuning",
    )


def run_matched_z_magnetization_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_workers: int | None = 1,
) -> ZMagnetizationSweepResult:
    """Sweep matched-probe coil Q and return excitation z magnetization."""

    values = _as_values(np.linspace(11, 200, 100) if q_values is None else q_values)
    sp0, pp = set_params_matched_orig(numpts=numpts)
    del_w = np.linspace(-float(maxoffs), float(maxoffs), int(numpts))
    sp0 = _silence_matched_sp(replace(sp0, del_w=del_w, maxoffs=float(maxoffs)))

    def case_runner(q_value: float) -> tuple[np.ndarray, np.ndarray]:
        if q_value <= 0:
            raise ValueError("Q values must be positive")
        resistance = 2 * np.pi * sp0.f0 * sp0.L / q_value
        sp = replace(sp0, Q=q_value, R=resistance)
        return calc_masy_matched_nut(sp, pp)

    mz, tvect = _run_z_sweep_cases(values, case_runner, num_workers)
    return ZMagnetizationSweepResult(
        values=values,
        value_label="coil Q",
        del_w=del_w,
        mz=mz,
        tvect=tvect,
        probe="matched",
        sweep="z_magnetization_q",
    )


def run_tuned_finite_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep tuned-probe Q for finite CPMG echo trains."""

    values = _as_values(np.linspace(20, 80, 4) if q_values is None else q_values)

    def case_runner(q_value: float) -> CPMGTrainResult:
        return run_tuned_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            q_value=q_value,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(values, "coil Q", "tuned", "finite_q", rows)


def run_untuned_finite_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep untuned-probe Q for finite CPMG echo trains."""

    values = _as_values(np.linspace(20, 80, 4) if q_values is None else q_values)

    def case_runner(q_value: float) -> CPMGTrainResult:
        return run_untuned_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            q_value=q_value,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(values, "coil Q", "untuned", "finite_q", rows)


def run_matched_finite_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep matched-probe Q for finite CPMG echo trains."""

    values = _as_values(np.linspace(20, 80, 4) if q_values is None else q_values)

    def case_runner(q_value: float) -> CPMGTrainResult:
        return run_matched_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            q_value=q_value,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(values, "coil Q", "matched", "finite_q", rows)


def run_tuned_finite_mistuning_sweep(
    offsets: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep tuned-probe mistuning for finite CPMG echo trains."""

    values = _as_values(np.linspace(-2, 2, 5) if offsets is None else offsets)

    def case_runner(offset: float) -> CPMGTrainResult:
        return run_tuned_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            mistuning_offset=offset,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(
        values,
        "frequency error (fin/Q)",
        "tuned",
        "finite_mistuning",
        rows,
    )


def run_untuned_finite_mistuning_sweep(
    offsets: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep untuned-probe mistuning for finite CPMG echo trains."""

    values = _as_values(np.linspace(-2, 2, 5) if offsets is None else offsets)

    def case_runner(offset: float) -> CPMGTrainResult:
        return run_untuned_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            mistuning_offset=offset,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(
        values,
        "frequency error (fin/Q)",
        "untuned",
        "finite_mistuning",
        rows,
    )


def run_matched_finite_mistuning_sweep(
    offsets: Iterable[float] | np.ndarray | None = None,
    *,
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    auto_refine_grid: bool = True,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
) -> CPMGFiniteParameterSweepResult:
    """Sweep matched-probe mistuning for finite CPMG echo trains."""

    values = _as_values(np.linspace(-2, 2, 5) if offsets is None else offsets)

    def case_runner(offset: float) -> CPMGTrainResult:
        return run_matched_cpmg_train(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=t1_seconds,
            t2_seconds=t2_seconds,
            mistuning_offset=offset,
            num_workers=num_workers,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
        )

    rows = _run_finite_sweep_cases(values, case_runner, sweep_workers)
    return _finite_sweep_result(
        values,
        "frequency error (fin/Q)",
        "matched",
        "finite_mistuning",
        rows,
    )
