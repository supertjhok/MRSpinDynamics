"""Sensor-control sequence definitions for defect-spin nano-MR.

This module describes coherent sensing windows. Optical initialization and
readout are effective preparation/measurement operations in
``nano_mr.readout`` rather than fictitious coherent Hamiltonian pulses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.sequences.cpmg import cpmg_pulse_times


ControlChannel = Literal["electron_mw", "nuclear_rf"]
_CONTROL_CHANNELS = ("electron_mw", "nuclear_rf")


@dataclass(frozen=True)
class TimedControlPulse:
    """One ideal or finite rectangular control pulse.

    ``center_seconds`` is the pulse center. A zero duration denotes an
    instantaneous ideal rotation. ``flip_angle_rad`` is the addressed-spin
    rotation angle, and ``phase_rad`` is measured from the local ``x`` axis.
    """

    center_seconds: float
    flip_angle_rad: float
    phase_rad: float = 0.0
    duration_seconds: float = 0.0
    channel: ControlChannel = "electron_mw"
    label: str = ""

    def __post_init__(self) -> None:
        center = float(self.center_seconds)
        flip = float(self.flip_angle_rad)
        phase = float(self.phase_rad)
        duration = float(self.duration_seconds)
        if center < 0.0 or not np.isfinite(center):
            raise ValueError("center_seconds must be finite and non-negative")
        if not np.isfinite(flip) or flip <= 0.0:
            raise ValueError("flip_angle_rad must be positive and finite")
        if not np.isfinite(phase):
            raise ValueError("phase_rad must be finite")
        if duration < 0.0 or not np.isfinite(duration):
            raise ValueError("duration_seconds must be finite and non-negative")
        if self.channel not in _CONTROL_CHANNELS:
            raise ValueError(f"channel must be one of {_CONTROL_CHANNELS}")
        object.__setattr__(self, "center_seconds", center)
        object.__setattr__(self, "flip_angle_rad", flip)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(self, "duration_seconds", duration)
        object.__setattr__(self, "label", str(self.label))

    @property
    def start_seconds(self) -> float:
        """Pulse start within the coherent sensing window."""

        return self.center_seconds - 0.5 * self.duration_seconds

    @property
    def end_seconds(self) -> float:
        """Pulse end within the coherent sensing window."""

        return self.center_seconds + 0.5 * self.duration_seconds

    @property
    def is_instantaneous(self) -> bool:
        """Whether this pulse is an ideal instantaneous rotation."""

        return self.duration_seconds == 0.0


@dataclass(frozen=True)
class SensingSequence:
    """A phase-aware control schedule over one coherent sensing window."""

    total_duration_seconds: float
    pulses: tuple[TimedControlPulse, ...] = ()
    name: str = "sensing"
    preparation_phase_rad: float = 0.0
    readout_phase_rad: float = 0.0

    def __post_init__(self) -> None:
        duration = float(self.total_duration_seconds)
        if duration <= 0.0 or not np.isfinite(duration):
            raise ValueError("total_duration_seconds must be positive and finite")
        pulses = tuple(sorted(self.pulses, key=lambda item: item.center_seconds))
        if not np.isfinite(self.preparation_phase_rad):
            raise ValueError("preparation_phase_rad must be finite")
        if not np.isfinite(self.readout_phase_rad):
            raise ValueError("readout_phase_rad must be finite")
        previous_end_by_channel = {
            "electron_mw": 0.0,
            "nuclear_rf": 0.0,
        }
        for pulse in pulses:
            if pulse.start_seconds < -1.0e-15:
                raise ValueError("a control pulse starts before the sensing window")
            if pulse.end_seconds > duration + 1.0e-15:
                raise ValueError("a control pulse ends after the sensing window")
            previous_end = previous_end_by_channel[pulse.channel]
            if pulse.start_seconds < previous_end:
                raise ValueError("control pulses on the same channel must not overlap")
            previous_end_by_channel[pulse.channel] = max(
                previous_end, pulse.end_seconds
            )
        object.__setattr__(self, "total_duration_seconds", duration)
        object.__setattr__(self, "pulses", pulses)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(
            self, "preparation_phase_rad", float(self.preparation_phase_rad)
        )
        object.__setattr__(self, "readout_phase_rad", float(self.readout_phase_rad))

    @property
    def electron_pulses(self) -> tuple[TimedControlPulse, ...]:
        """Microwave pulses that act on the defect-sensor qubit."""

        return tuple(item for item in self.pulses if item.channel == "electron_mw")

    @property
    def nuclear_rf_pulses(self) -> tuple[TimedControlPulse, ...]:
        """RF pulses reserved for target nuclei."""

        return tuple(item for item in self.pulses if item.channel == "nuclear_rf")

    @property
    def cycle_pulse_phases_rad(self) -> np.ndarray:
        """Electron-pulse phases as a NumPy array."""

        return np.array(
            [item.phase_rad for item in self.electron_pulses], dtype=np.float64
        )


def ramsey_sequence(
    evolution_seconds: float,
    *,
    preparation_phase_rad: float = 0.0,
    readout_phase_rad: float = 0.0,
) -> SensingSequence:
    """Return a free-evolution Ramsey sensing window."""

    return SensingSequence(
        total_duration_seconds=evolution_seconds,
        name="Ramsey",
        preparation_phase_rad=preparation_phase_rad,
        readout_phase_rad=readout_phase_rad,
    )


def hahn_echo_sequence(
    tau_seconds: float,
    *,
    pulse_duration_seconds: float = 0.0,
    refocus_phase_rad: float = 0.0,
) -> SensingSequence:
    """Return a Hahn window with a pi pulse at ``tau`` and duration ``2*tau``."""

    tau = float(tau_seconds)
    if tau <= 0.0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be positive and finite")
    return SensingSequence(
        total_duration_seconds=2.0 * tau,
        pulses=(
            TimedControlPulse(
                center_seconds=tau,
                duration_seconds=pulse_duration_seconds,
                flip_angle_rad=np.pi,
                phase_rad=refocus_phase_rad,
                label="pi",
            ),
        ),
        name="Hahn echo",
    )


def cpmg_sequence(
    num_pulses: int,
    total_duration_seconds: float,
    *,
    pulse_duration_seconds: float = 0.0,
    phase_rad: float = np.pi / 2.0,
) -> SensingSequence:
    """Return an equally spaced CPMG control sequence."""

    centers = cpmg_pulse_times(num_pulses, total_duration_seconds)
    return _sequence_from_phases(
        np.full(centers.size, float(phase_rad)),
        centers,
        total_duration_seconds,
        pulse_duration_seconds=pulse_duration_seconds,
        name=f"CPMG-{centers.size}",
    )


def xy_sequence(
    order: Literal[4, 8, 16],
    repetitions: int,
    total_duration_seconds: float,
    *,
    pulse_duration_seconds: float = 0.0,
) -> SensingSequence:
    """Return an XY4, XY8, or phase-alternated XY16 sequence."""

    if order not in (4, 8, 16):
        raise ValueError("order must be 4, 8, or 16")
    if isinstance(repetitions, bool):
        raise ValueError("repetitions must be a positive integer")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    x = 0.0
    y = np.pi / 2.0
    xy4 = np.array([x, y, x, y])
    xy8 = np.array([x, y, x, y, y, x, y, x])
    if order == 4:
        base = xy4
    elif order == 8:
        base = xy8
    else:
        base = np.concatenate((xy8, xy8 + np.pi))
    phases = np.tile(base, repetitions)
    centers = cpmg_pulse_times(phases.size, total_duration_seconds)
    return _sequence_from_phases(
        phases,
        centers,
        total_duration_seconds,
        pulse_duration_seconds=pulse_duration_seconds,
        name=f"XY{order}-{repetitions}",
    )


def kdd_sequence(
    repetitions: int,
    total_duration_seconds: float,
    *,
    pulse_duration_seconds: float = 0.0,
    base_phase_rad: float = 0.0,
) -> SensingSequence:
    """Return repeated 20-pulse Knill dynamical-decoupling cycles.

    One KDD cycle is ``[KDD_phi, KDD_(phi+pi/2)]`` repeated twice,
    where each five-pulse Knill block has phases
    ``phi + [pi/6, 0, pi/2, 0, pi/6]``. Pulse centers are equally spaced;
    the phase cycle supplies robustness to systematic pulse errors.
    """

    if isinstance(repetitions, bool):
        raise ValueError("repetitions must be a positive integer")
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    base_phase = float(base_phase_rad)
    if not np.isfinite(base_phase):
        raise ValueError("base_phase_rad must be finite")
    knill = base_phase + np.array([np.pi / 6.0, 0.0, np.pi / 2.0, 0.0, np.pi / 6.0])
    cycle = np.concatenate((knill, knill + np.pi / 2.0) * 2)
    phases = np.tile(cycle, repetitions)
    centers = cpmg_pulse_times(phases.size, total_duration_seconds)
    return _sequence_from_phases(
        phases,
        centers,
        total_duration_seconds,
        pulse_duration_seconds=pulse_duration_seconds,
        name=f"KDD-{20 * repetitions}",
    )


def phase_cycled_sequence(
    phases_rad: Sequence[float],
    total_duration_seconds: float,
    *,
    pulse_duration_seconds: float = 0.0,
    name: str = "phase-cycled",
) -> SensingSequence:
    """Return equally spaced pi pulses with caller-supplied transverse phases."""

    phases = np.asarray(phases_rad, dtype=np.float64)
    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("phases_rad must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases_rad must be finite")
    centers = cpmg_pulse_times(phases.size, total_duration_seconds)
    return _sequence_from_phases(
        phases,
        centers,
        total_duration_seconds,
        pulse_duration_seconds=pulse_duration_seconds,
        name=name,
    )


def with_nuclear_rf_pulses(
    sequence: SensingSequence,
    pulses: Sequence[TimedControlPulse],
) -> SensingSequence:
    """Return a sequence augmented with target-nuclear RF events."""

    additions = tuple(pulses)
    if any(item.channel != "nuclear_rf" for item in additions):
        raise ValueError("all added pulses must use channel='nuclear_rf'")
    return SensingSequence(
        total_duration_seconds=sequence.total_duration_seconds,
        pulses=sequence.pulses + additions,
        name=sequence.name,
        preparation_phase_rad=sequence.preparation_phase_rad,
        readout_phase_rad=sequence.readout_phase_rad,
    )


def _sequence_from_phases(
    phases: np.ndarray,
    centers: np.ndarray,
    total_duration_seconds: float,
    *,
    pulse_duration_seconds: float,
    name: str,
) -> SensingSequence:
    pulses = tuple(
        TimedControlPulse(
            center_seconds=float(center),
            duration_seconds=pulse_duration_seconds,
            flip_angle_rad=np.pi,
            phase_rad=float(phase),
            label=f"pi-{index + 1}",
        )
        for index, (center, phase) in enumerate(zip(centers, phases))
    )
    return SensingSequence(
        total_duration_seconds=total_duration_seconds,
        pulses=pulses,
        name=name,
    )


__all__ = [
    "ControlChannel",
    "SensingSequence",
    "TimedControlPulse",
    "cpmg_sequence",
    "hahn_echo_sequence",
    "kdd_sequence",
    "phase_cycled_sequence",
    "ramsey_sequence",
    "with_nuclear_rf_pulses",
    "xy_sequence",
]
