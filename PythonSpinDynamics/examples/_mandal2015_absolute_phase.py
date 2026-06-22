"""Shared helpers for Mandal-2015-inspired absolute-phase examples."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Any

import numpy as np

from spin_dynamics.parameters import (
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.workflows import (
    run_matched_cpmg_train,
    run_tuned_cpmg_train,
    run_untuned_cpmg_train,
)
from spin_dynamics.workflows.cpmg import (
    _calc_matched_pulse_shape,
    _calc_tuned_pulse_shape,
    _calc_untuned_pulse_shape,
    _offset_grid,
)
from spin_dynamics.probes.matched import matching_network_design2


@dataclass(frozen=True)
class PhaseResolvedProbeCase:
    """Finite CPMG train with absolute-phase-resolved probe pulse shapes."""

    result: Any
    probe: str
    phase_step_cycles: float
    rf_frequency_hz: float
    initial_refocus_phase_rad: float

    @property
    def echo_energy(self) -> np.ndarray:
        return np.trapezoid(np.abs(self.result.echo) ** 2, self.result.tvect, axis=1)

    @property
    def refocus_phase_cycles(self) -> np.ndarray:
        metadata = self.result.absolute_phase
        if metadata is None:
            return np.zeros(self.result.echo.shape[0], dtype=np.float64)
        phase = np.asarray(metadata.refocus_absolute_phase_rad, dtype=np.float64)
        return np.mod(phase / (2.0 * np.pi), 1.0)


@dataclass(frozen=True)
class PhaseResolvedPulseShape:
    """Solved rotating-frame pulse shape for one absolute RF phase."""

    probe: str
    absolute_phase_rad: float
    time_seconds: np.ndarray
    duration: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray

    @property
    def drive(self) -> np.ndarray:
        return self.amplitude * np.exp(1j * self.phase)

    @property
    def absolute_phase_cycles(self) -> float:
        return float(np.mod(self.absolute_phase_rad / (2.0 * np.pi), 1.0))


def phase_advance_frequency(
    *,
    probe: str,
    phase_step_cycles: float,
) -> float:
    """Return RF frequency that gives a phase advance per CPMG echo."""

    _sp, pp = _probe_parameters(probe, numpts=3)
    echo_spacing = float(np.sum(np.asarray(pp.tref, dtype=np.float64)))
    phase_step = float(phase_step_cycles)
    cycles_per_echo = 1.0 if phase_step == 0.0 else phase_step
    return cycles_per_echo / echo_spacing


def _probe_parameters(probe: str, *, numpts: int) -> tuple[Any, Any]:
    if probe == "tuned":
        _params, sp, pp = set_params_tuned_orig(numpts=int(numpts))
        return sp, pp
    if probe == "untuned":
        _params, sp, pp = set_params_untuned_orig(numpts=int(numpts))
        return sp, pp
    if probe == "matched":
        return set_params_matched_orig(numpts=int(numpts))
    raise ValueError("probe must be 'tuned', 'untuned', or 'matched'")


def _probe_shape_state(probe: str, *, numpts: int, maxoffs: float = 10.0) -> tuple[Any, Any]:
    del_w = _offset_grid(int(numpts), float(maxoffs))
    if probe == "tuned":
        _params, sp0, pp = set_params_tuned_orig(numpts=int(numpts))
        sp0 = replace(
            sp0,
            R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            C=1.0 / ((2.0 * np.pi * sp0.f0) ** 2 * sp0.L),
        )
        return {**sp0.__dict__, "del_w": del_w}, pp
    if probe == "untuned":
        _params, sp0, pp = set_params_untuned_orig(numpts=int(numpts))
        sp0 = replace(
            sp0,
            R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            C=1.0 / ((2.0 * np.pi * 10.0 * sp0.f0) ** 2 * sp0.L),
        )
        return {**sp0.__dict__, "del_w": del_w}, pp
    if probe == "matched":
        sp0, pp = set_params_matched_orig(numpts=int(numpts))
        sp0 = replace(sp0, R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q)
        c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
        return {**sp0.__dict__, "C1": c1, "C2": c2, "del_w": del_w}, pp
    raise ValueError("probe must be 'tuned', 'untuned', or 'matched'")


def solve_refocusing_pulse_shape(
    *,
    probe: str,
    absolute_phase_rad: float,
    numpts: int = 17,
    maxoffs: float = 10.0,
) -> PhaseResolvedPulseShape:
    """Solve one refocusing pulse shape for a requested absolute RF phase."""

    probe = str(probe)
    sp, pp = _probe_shape_state(probe, numpts=numpts, maxoffs=maxoffs)
    if probe == "tuned":
        duration, phase, amplitude = _calc_tuned_pulse_shape(
            sp,
            pp,
            pp.T_180,
            0.0,
            1.0,
            2.0 * pp.T_90,
            psi=float(absolute_phase_rad),
        )
    elif probe == "untuned":
        duration, phase, amplitude = _calc_untuned_pulse_shape(
            sp,
            pp,
            pp.T_180,
            0.0,
            1.0,
            pp.trd,
            psi=float(absolute_phase_rad),
        )
    elif probe == "matched":
        duration, phase, amplitude = _calc_matched_pulse_shape(
            sp,
            pp,
            pp.T_180,
            0.0,
            1.0,
            pp.trd,
            psi=float(absolute_phase_rad),
        )[:3]
    else:
        raise ValueError("probe must be 'tuned', 'untuned', or 'matched'")

    duration = np.asarray(duration, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    valid = duration > 0.0
    duration = duration[valid]
    phase = phase[valid]
    amplitude = amplitude[valid]
    segment_seconds = duration * float(pp.T_90) / (np.pi / 2.0)
    time_seconds = np.cumsum(segment_seconds) - 0.5 * segment_seconds
    return PhaseResolvedPulseShape(
        probe=probe,
        absolute_phase_rad=float(absolute_phase_rad),
        time_seconds=time_seconds,
        duration=duration,
        phase=phase,
        amplitude=amplitude,
    )


def _first_refocus_start_seconds(pp: Any) -> float:
    return float(
        np.ravel(pp.texc)[0] + pp.tcorr + np.ravel(pp.tref)[0]
    )


def run_phase_resolved_probe_case(
    *,
    probe: str,
    numpts: int,
    num_echoes: int,
    phase_step_cycles: float,
    initial_refocus_phase_rad: float = 0.0,
    maxoffs: float = 10.0,
) -> PhaseResolvedProbeCase:
    """Run finite CPMG with per-pulse probe waveform solves."""

    probe = str(probe)
    _sp, pp = _probe_parameters(probe, numpts=numpts)
    rf_frequency_hz = phase_advance_frequency(
        probe=probe,
        phase_step_cycles=phase_step_cycles,
    )
    first_refocus = _first_refocus_start_seconds(pp)
    rf_phase_at_zero_rad = (
        float(initial_refocus_phase_rad)
        - 2.0 * np.pi * rf_frequency_hz * first_refocus
    )
    runners = {
        "tuned": run_tuned_cpmg_train,
        "untuned": run_untuned_cpmg_train,
        "matched": run_matched_cpmg_train,
    }
    result = runners[probe](
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        num_echoes=int(num_echoes),
        t1_seconds=1.0e9,
        t2_seconds=1.0e9,
        rephase_action="ignore",
        absolute_phase={
            "rf_frequency_hz": rf_frequency_hz,
            "rf_phase_at_zero_rad": rf_phase_at_zero_rad,
        },
    )
    return PhaseResolvedProbeCase(
        result=result,
        probe=probe,
        phase_step_cycles=float(phase_step_cycles),
        rf_frequency_hz=rf_frequency_hz,
        initial_refocus_phase_rad=float(initial_refocus_phase_rad),
    )


def matched_filter_ratio(
    result: PhaseResolvedProbeCase,
    baseline: PhaseResolvedProbeCase,
) -> np.ndarray:
    denominator = np.trapezoid(
        np.abs(baseline.result.echo) ** 2,
        baseline.result.tvect,
        axis=1,
    )
    numerator = np.trapezoid(
        result.result.echo * np.conj(baseline.result.echo),
        result.result.tvect,
        axis=1,
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
