"""Probe pulse-shape diagnostics for absolute-phase studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spin_dynamics.absolute_phase import PulseShape, PulseShapeLibrary, TAU
from spin_dynamics.parameters import (
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.probes.matched import matching_network_design2
from spin_dynamics.workflows.cpmg import (
    _calc_matched_pulse_shape,
    _calc_tuned_pulse_shape,
    _calc_untuned_pulse_shape,
    _offset_grid,
)


@dataclass(frozen=True)
class ProbePulseShapeDiagnostics:
    """Solved rotating-frame probe pulse shape and diagnostic metrics."""

    probe: str
    pulse_kind: str
    absolute_phase_rad: float
    rotating_phase_rad: float
    carrier_phase_rad: float
    pulse_duration_seconds: float
    delay_seconds: float
    time_seconds: np.ndarray
    duration: np.ndarray
    segment_duration_seconds: np.ndarray
    phase: np.ndarray
    amplitude: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "time_seconds": np.asarray(self.time_seconds, dtype=np.float64).reshape(-1),
            "duration": np.asarray(self.duration, dtype=np.float64).reshape(-1),
            "segment_duration_seconds": np.asarray(
                self.segment_duration_seconds,
                dtype=np.float64,
            ).reshape(-1),
            "phase": np.asarray(self.phase, dtype=np.float64).reshape(-1),
            "amplitude": np.asarray(self.amplitude, dtype=np.float64).reshape(-1),
        }
        sizes = {values.size for values in arrays.values()}
        if len(sizes) != 1:
            raise ValueError("pulse diagnostic arrays must have the same length")
        if not sizes or next(iter(sizes)) == 0:
            raise ValueError("pulse diagnostics must contain at least one segment")
        for name, values in arrays.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} values must be finite")
        object.__setattr__(self, "probe", str(self.probe))
        object.__setattr__(self, "pulse_kind", str(self.pulse_kind))
        object.__setattr__(self, "absolute_phase_rad", float(self.absolute_phase_rad))
        object.__setattr__(self, "rotating_phase_rad", float(self.rotating_phase_rad))
        object.__setattr__(self, "carrier_phase_rad", float(self.carrier_phase_rad))
        object.__setattr__(
            self,
            "pulse_duration_seconds",
            float(self.pulse_duration_seconds),
        )
        object.__setattr__(self, "delay_seconds", float(self.delay_seconds))
        for name, values in arrays.items():
            object.__setattr__(self, name, values)

    @property
    def drive(self) -> np.ndarray:
        """Complex rotating-frame RF drive for each segment."""

        return self.amplitude * np.exp(1j * self.phase)

    @property
    def in_phase(self) -> np.ndarray:
        """In-phase component of the rotating-frame drive."""

        return np.real(self.drive)

    @property
    def quadrature(self) -> np.ndarray:
        """Quadrature component of the rotating-frame drive."""

        return np.imag(self.drive)

    @property
    def absolute_phase_cycles(self) -> float:
        """Absolute RF phase at pulse start, in cycles modulo one."""

        return float(np.mod(self.absolute_phase_rad / TAU, 1.0))

    @property
    def total_duration_seconds(self) -> float:
        """Total simulated nonzero pulse duration."""

        return float(np.sum(self.segment_duration_seconds))

    @property
    def peak_amplitude(self) -> float:
        """Peak segment amplitude."""

        return float(np.max(np.abs(self.amplitude)))

    @property
    def rms_amplitude(self) -> float:
        """Duration-weighted RMS amplitude."""

        weights = self.segment_duration_seconds
        return float(
            np.sqrt(np.sum((np.abs(self.amplitude) ** 2) * weights) / np.sum(weights))
        )

    @property
    def quadrature_energy_fraction(self) -> float:
        """Fraction of duration-weighted drive energy in quadrature."""

        weights = self.segment_duration_seconds
        drive = self.drive
        total = float(np.sum((np.abs(drive) ** 2) * weights))
        if total == 0.0:
            return 0.0
        return float(np.sum((np.imag(drive) ** 2) * weights) / total)

    @property
    def integrated_drive(self) -> complex:
        """Duration-integrated complex drive."""

        return complex(np.sum(self.drive * self.segment_duration_seconds))

    def to_pulse_shape(self) -> PulseShape:
        """Return the shape in spin-dynamics pulse-segment units."""

        return PulseShape(
            duration=self.duration,
            phase=self.phase,
            amplitude=self.amplitude,
        )


@dataclass(frozen=True)
class ProbePulseShapeSweep:
    """Set of solved probe pulse shapes over absolute RF phase."""

    probe: str
    pulse_kind: str
    shapes: tuple[ProbePulseShapeDiagnostics, ...]
    pulse_shape_library: PulseShapeLibrary

    @property
    def absolute_phase_rad(self) -> np.ndarray:
        """Absolute phase samples in the user-requested order."""

        return np.array([shape.absolute_phase_rad for shape in self.shapes])

    @property
    def absolute_phase_cycles(self) -> np.ndarray:
        """Absolute phase samples in cycles modulo one."""

        return np.mod(self.absolute_phase_rad / TAU, 1.0)

    @property
    def quadrature_energy_fraction(self) -> np.ndarray:
        """Quadrature energy fraction for each solved shape."""

        return np.array([shape.quadrature_energy_fraction for shape in self.shapes])


def solve_probe_pulse_shape(
    *,
    probe: str,
    absolute_phase_rad: float,
    pulse_kind: str = "refocusing",
    numpts: int = 17,
    maxoffs: float = 10.0,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    rotating_phase_rad: float | None = None,
    pulse_duration_seconds: float | None = None,
    pulse_amplitude: float = 1.0,
    delay_seconds: float | None = None,
) -> ProbePulseShapeDiagnostics:
    """Solve one probe pulse shape for a requested absolute RF phase."""

    probe = _validate_probe(probe)
    pulse_kind = str(pulse_kind)
    sp, pp = _probe_shape_state(
        probe,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        q_value=q_value,
        mistuning_offset=mistuning_offset,
    )
    rotating_phase, pulse_duration, delay = _pulse_parameters(
        probe=probe,
        pp=pp,
        pulse_kind=pulse_kind,
        rotating_phase_rad=rotating_phase_rad,
        pulse_duration_seconds=pulse_duration_seconds,
        delay_seconds=delay_seconds,
    )
    carrier_phase = float(absolute_phase_rad) - rotating_phase
    raw_shape = _solve_raw_shape(
        probe=probe,
        sp=sp,
        pp=pp,
        pulse_duration_seconds=pulse_duration,
        rotating_phase_rad=rotating_phase,
        pulse_amplitude=float(pulse_amplitude),
        delay_seconds=delay,
        carrier_phase_rad=carrier_phase,
    )
    return _diagnostics_from_raw_shape(
        probe=probe,
        pulse_kind=pulse_kind,
        absolute_phase_rad=float(absolute_phase_rad),
        rotating_phase_rad=rotating_phase,
        carrier_phase_rad=carrier_phase,
        pulse_duration_seconds=pulse_duration,
        delay_seconds=delay,
        pp=pp,
        raw_shape=raw_shape,
    )


def solve_probe_pulse_shape_sweep(
    *,
    probe: str,
    absolute_phase_rad: Sequence[float] | np.ndarray,
    pulse_kind: str = "refocusing",
    numpts: int = 17,
    maxoffs: float = 10.0,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    rotating_phase_rad: float | None = None,
    pulse_duration_seconds: float | None = None,
    pulse_amplitude: float = 1.0,
    delay_seconds: float | None = None,
) -> ProbePulseShapeSweep:
    """Solve a probe pulse-shape sweep over absolute RF phase."""

    phases = np.asarray(absolute_phase_rad, dtype=np.float64).reshape(-1)
    if phases.size == 0:
        raise ValueError("absolute_phase_rad must contain at least one phase")
    shapes = tuple(
        solve_probe_pulse_shape(
            probe=probe,
            absolute_phase_rad=float(phase),
            pulse_kind=pulse_kind,
            numpts=numpts,
            maxoffs=maxoffs,
            q_value=q_value,
            mistuning_offset=mistuning_offset,
            rotating_phase_rad=rotating_phase_rad,
            pulse_duration_seconds=pulse_duration_seconds,
            pulse_amplitude=pulse_amplitude,
            delay_seconds=delay_seconds,
        )
        for phase in phases
    )
    library = PulseShapeLibrary(
        absolute_phase_rad=phases,
        shapes={str(pulse_kind): tuple(shape.to_pulse_shape() for shape in shapes)},
    )
    return ProbePulseShapeSweep(
        probe=str(shapes[0].probe),
        pulse_kind=str(pulse_kind),
        shapes=shapes,
        pulse_shape_library=library,
    )


def build_probe_pulse_shape_library(
    *,
    probe: str,
    absolute_phase_rad: Sequence[float] | np.ndarray,
    pulse_kind: str = "refocusing",
    numpts: int = 17,
    maxoffs: float = 10.0,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    rotating_phase_rad: float | None = None,
    pulse_duration_seconds: float | None = None,
    pulse_amplitude: float = 1.0,
    delay_seconds: float | None = None,
) -> PulseShapeLibrary:
    """Build a probe-solved absolute-phase pulse-shape library."""

    return solve_probe_pulse_shape_sweep(
        probe=probe,
        absolute_phase_rad=absolute_phase_rad,
        pulse_kind=pulse_kind,
        numpts=numpts,
        maxoffs=maxoffs,
        q_value=q_value,
        mistuning_offset=mistuning_offset,
        rotating_phase_rad=rotating_phase_rad,
        pulse_duration_seconds=pulse_duration_seconds,
        pulse_amplitude=pulse_amplitude,
        delay_seconds=delay_seconds,
    ).pulse_shape_library


def _validate_probe(probe: str) -> str:
    probe = str(probe)
    if probe not in {"tuned", "untuned", "matched"}:
        raise ValueError("probe must be 'tuned', 'untuned', or 'matched'")
    return probe


def _probe_shape_state(
    probe: str,
    *,
    numpts: int,
    maxoffs: float,
    q_value: float | None,
    mistuning_offset: float | None,
) -> tuple[dict[str, Any], Any]:
    del_w = _offset_grid(int(numpts), float(maxoffs))
    if probe == "tuned":
        _params, sp0, pp = set_params_tuned_orig(numpts=int(numpts))
        sp0 = _apply_probe_options(
            sp0,
            q_value=q_value,
            mistuning_offset=mistuning_offset,
        )
        sp0 = replace(
            sp0,
            R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            C=1.0 / ((2.0 * np.pi * sp0.f0) ** 2 * sp0.L),
        )
        return {**sp0.__dict__, "del_w": del_w}, pp
    if probe == "untuned":
        _params, sp0, pp = set_params_untuned_orig(numpts=int(numpts))
        sp0 = _apply_probe_options(
            sp0,
            q_value=q_value,
            mistuning_offset=mistuning_offset,
        )
        sp0 = replace(
            sp0,
            R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            C=1.0 / ((2.0 * np.pi * 10.0 * sp0.f0) ** 2 * sp0.L),
        )
        return {**sp0.__dict__, "del_w": del_w}, pp
    sp0, pp = set_params_matched_orig(numpts=int(numpts))
    sp0 = _apply_probe_options(
        sp0,
        q_value=q_value,
        mistuning_offset=mistuning_offset,
    )
    sp0 = replace(sp0, R=2.0 * np.pi * sp0.f0 * sp0.L / sp0.Q)
    c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
    return {**sp0.__dict__, "C1": c1, "C2": c2, "del_w": del_w}, pp


def _apply_probe_options(
    sp0: Any,
    *,
    q_value: float | None,
    mistuning_offset: float | None,
) -> Any:
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    return sp0


def _pulse_parameters(
    *,
    probe: str,
    pp: Any,
    pulse_kind: str,
    rotating_phase_rad: float | None,
    pulse_duration_seconds: float | None,
    delay_seconds: float | None,
) -> tuple[float, float, float]:
    if pulse_kind not in {"excitation", "refocusing", "custom"}:
        raise ValueError("pulse_kind must be 'excitation', 'refocusing', or 'custom'")
    if pulse_duration_seconds is None:
        if pulse_kind == "custom":
            raise ValueError("custom pulses require pulse_duration_seconds")
        pulse_duration_seconds = pp.T_90 if pulse_kind == "excitation" else pp.T_180
    if rotating_phase_rad is None:
        rotating_phase_rad = np.pi / 2 if pulse_kind == "excitation" else 0.0
    if delay_seconds is None:
        delay_seconds = 2.0 * pp.T_90 if probe == "tuned" else pp.trd
    if pulse_duration_seconds <= 0:
        raise ValueError("pulse_duration_seconds must be positive")
    if delay_seconds < 0:
        raise ValueError("delay_seconds must be non-negative")
    return (
        float(rotating_phase_rad),
        float(pulse_duration_seconds),
        float(delay_seconds),
    )


def _solve_raw_shape(
    *,
    probe: str,
    sp: dict[str, Any],
    pp: Any,
    pulse_duration_seconds: float,
    rotating_phase_rad: float,
    pulse_amplitude: float,
    delay_seconds: float,
    carrier_phase_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probe == "tuned":
        return _calc_tuned_pulse_shape(
            sp,
            pp,
            pulse_duration_seconds,
            rotating_phase_rad,
            pulse_amplitude,
            delay_seconds,
            psi=carrier_phase_rad,
        )
    if probe == "untuned":
        return _calc_untuned_pulse_shape(
            sp,
            pp,
            pulse_duration_seconds,
            rotating_phase_rad,
            pulse_amplitude,
            delay_seconds,
            psi=carrier_phase_rad,
        )
    return _calc_matched_pulse_shape(
        sp,
        pp,
        pulse_duration_seconds,
        rotating_phase_rad,
        pulse_amplitude,
        delay_seconds,
        psi=carrier_phase_rad,
        segment_fraction=0.5,
    )[:3]


def _diagnostics_from_raw_shape(
    *,
    probe: str,
    pulse_kind: str,
    absolute_phase_rad: float,
    rotating_phase_rad: float,
    carrier_phase_rad: float,
    pulse_duration_seconds: float,
    delay_seconds: float,
    pp: Any,
    raw_shape: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> ProbePulseShapeDiagnostics:
    duration = np.asarray(raw_shape[0], dtype=np.float64)
    phase = np.asarray(raw_shape[1], dtype=np.float64)
    amplitude = np.asarray(raw_shape[2], dtype=np.float64)
    valid = duration > 0.0
    duration = duration[valid]
    phase = phase[valid]
    amplitude = amplitude[valid]
    segment_seconds = duration * float(pp.T_90) / (np.pi / 2.0)
    time_seconds = np.cumsum(segment_seconds) - 0.5 * segment_seconds
    return ProbePulseShapeDiagnostics(
        probe=probe,
        pulse_kind=pulse_kind,
        absolute_phase_rad=absolute_phase_rad,
        rotating_phase_rad=rotating_phase_rad,
        carrier_phase_rad=carrier_phase_rad,
        pulse_duration_seconds=pulse_duration_seconds,
        delay_seconds=delay_seconds,
        time_seconds=time_seconds,
        duration=duration,
        segment_duration_seconds=segment_seconds,
        phase=phase,
        amplitude=amplitude,
    )
