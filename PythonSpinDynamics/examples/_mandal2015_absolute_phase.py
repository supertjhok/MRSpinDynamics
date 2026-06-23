"""Shared helpers for Mandal-2015-inspired absolute-phase examples."""

from __future__ import annotations

from dataclasses import dataclass
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
from spin_dynamics.pulse_diagnostics import (
    ProbePulseShapeDiagnostics as PhaseResolvedPulseShape,
    solve_probe_pulse_shape,
)


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


def solve_refocusing_pulse_shape(
    *,
    probe: str,
    absolute_phase_rad: float,
    numpts: int = 17,
    maxoffs: float = 10.0,
) -> PhaseResolvedPulseShape:
    """Solve one refocusing pulse shape for a requested absolute RF phase."""

    return solve_probe_pulse_shape(
        probe=probe,
        absolute_phase_rad=float(absolute_phase_rad),
        pulse_kind="refocusing",
        numpts=numpts,
        maxoffs=maxoffs,
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
    phase_bins: int | None = None,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
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
        auto_refine_grid=bool(auto_refine_grid),
        rephase_safety_factor=float(rephase_safety_factor),
        rephase_action=rephase_action,
        absolute_phase={
            "rf_frequency_hz": rf_frequency_hz,
            "rf_phase_at_zero_rad": rf_phase_at_zero_rad,
            "phase_bins": phase_bins,
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
