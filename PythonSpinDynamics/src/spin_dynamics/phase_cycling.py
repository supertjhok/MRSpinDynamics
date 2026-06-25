"""Reusable phase-cycle tables and signal combination helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class PhaseStep:
    """One scan branch in a phase cycle.

    ``pulse_phases`` maps logical pulse names to requested rotating-frame
    phases in radians. ``receiver_phase_rad`` is applied during branch
    combination, and ``weight`` is the branch's scalar accumulation weight.
    """

    pulse_phases: Mapping[str, float] | Iterable[tuple[str, float]] = ()
    receiver_phase_rad: float = 0.0
    weight: complex = 1.0 + 0.0j
    label: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.pulse_phases, Mapping):
            raw_items = list(self.pulse_phases.items())
        else:
            raw_items = list(self.pulse_phases)

        seen: set[str] = set()
        phases: list[tuple[str, float]] = []
        for raw_name, raw_phase in raw_items:
            name = str(raw_name)
            if not name:
                raise ValueError("pulse phase names must not be empty")
            if name in seen:
                raise ValueError(f"duplicate pulse phase name: {name!r}")
            phase = float(raw_phase)
            if not np.isfinite(phase):
                raise ValueError("pulse phases must be finite")
            seen.add(name)
            phases.append((name, phase))

        receiver_phase = float(self.receiver_phase_rad)
        if not np.isfinite(receiver_phase):
            raise ValueError("receiver_phase_rad must be finite")
        weight = complex(self.weight)
        if not np.isfinite(weight.real) or not np.isfinite(weight.imag):
            raise ValueError("weight must be finite")
        label = None if self.label is None else str(self.label)

        object.__setattr__(self, "pulse_phases", tuple(phases))
        object.__setattr__(self, "receiver_phase_rad", receiver_phase)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "label", label)

    @property
    def pulse_names(self) -> tuple[str, ...]:
        """Pulse names defined by this step."""

        return tuple(name for name, _phase in self.pulse_phases)

    def pulse_phase(self, pulse_name: str) -> float:
        """Return the phase for ``pulse_name``."""

        requested = str(pulse_name)
        for name, phase in self.pulse_phases:
            if name == requested:
                return float(phase)
        raise KeyError(f"phase step does not define pulse {requested!r}")

    @property
    def receiver_weight(self) -> complex:
        """Complex branch weight including receiver phase."""

        return complex(self.weight) * np.exp(-1j * float(self.receiver_phase_rad))


@dataclass(frozen=True)
class PhaseCycle:
    """A reusable phase-cycle scan table."""

    steps: Sequence[PhaseStep]
    pulse_names: Sequence[str] | None = None
    normalize: bool = True
    name: str = "custom"

    def __post_init__(self) -> None:
        steps = tuple(
            step if isinstance(step, PhaseStep) else PhaseStep(step)
            for step in self.steps
        )
        if len(steps) == 0:
            raise ValueError("phase cycle must contain at least one step")

        if self.pulse_names is None:
            pulse_names = steps[0].pulse_names
        else:
            pulse_names = tuple(str(name) for name in self.pulse_names)
        if any(not name for name in pulse_names):
            raise ValueError("pulse_names must not contain empty names")
        if len(set(pulse_names)) != len(pulse_names):
            raise ValueError("pulse_names must be unique")
        for step in steps:
            missing = set(pulse_names) - set(step.pulse_names)
            if missing:
                missing_names = ", ".join(sorted(missing))
                raise ValueError(f"phase step is missing pulse phases: {missing_names}")

        weights = np.asarray(
            [step.receiver_weight for step in steps],
            dtype=np.complex128,
        )
        if self.normalize:
            norm = float(np.sum(np.abs(weights)))
            if norm <= 0.0:
                raise ValueError("normalized phase cycle weights must not all be zero")
        name = str(self.name)
        if not name:
            raise ValueError("name must not be empty")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "pulse_names", pulse_names)
        object.__setattr__(self, "normalize", bool(self.normalize))
        object.__setattr__(self, "name", name)

    @property
    def num_steps(self) -> int:
        """Number of scan branches."""

        return len(self.steps)

    @property
    def branch_weights(self) -> np.ndarray:
        """Complex branch-combination weights."""

        weights = np.asarray(
            [step.receiver_weight for step in self.steps],
            dtype=np.complex128,
        )
        if not self.normalize:
            return weights
        return weights / float(np.sum(np.abs(weights)))

    @property
    def labels(self) -> tuple[str | None, ...]:
        """Optional branch labels."""

        return tuple(step.label for step in self.steps)

    def pulse_phases(self, pulse_name: str) -> np.ndarray:
        """Return one phase per branch for ``pulse_name``."""

        return np.asarray(
            [step.pulse_phase(pulse_name) for step in self.steps],
            dtype=np.float64,
        )

    def combine(self, branch_signals: Sequence[np.ndarray | complex]) -> np.ndarray:
        """Combine one signal per branch using receiver-weighted accumulation."""

        if len(branch_signals) != self.num_steps:
            raise ValueError("branch_signals length must match phase cycle steps")
        combined = None
        for weight, signal in zip(self.branch_weights, branch_signals):
            contribution = complex(weight) * np.asarray(signal)
            combined = contribution if combined is None else combined + contribution
        return np.asarray(combined)


def cpmg_two_step_phase_cycle(
    *,
    excitation_name: str = "excitation",
    excitation_phase_rad: float = np.pi / 2.0,
) -> PhaseCycle:
    """Return the default two-step CPMG/PAP excitation phase cycle."""

    name = str(excitation_name)
    first_phase = float(excitation_phase_rad)
    return PhaseCycle(
        steps=(
            PhaseStep(
                pulse_phases={name: first_phase},
                weight=1.0,
                label=f"{name}_plus",
            ),
            PhaseStep(
                pulse_phases={name: first_phase + np.pi},
                weight=-1.0,
                label=f"{name}_minus",
            ),
        ),
        pulse_names=(name,),
        normalize=True,
        name="cpmg_two_step",
    )


def pgste_stimulated_echo_phase_cycle() -> PhaseCycle:
    """Return the selected-pathway PGSTE stimulated-echo phase table."""

    return PhaseCycle(
        steps=(
            PhaseStep(
                pulse_phases={
                    "excitation_90": np.pi / 2.0,
                    "store_90": np.pi / 2.0,
                    "read_90": np.pi / 2.0,
                },
                label="stimulated_echo_pathway",
            ),
        ),
        pulse_names=("excitation_90", "store_90", "read_90"),
        normalize=False,
        name="pgste_stimulated_echo",
    )


__all__ = [
    "PhaseCycle",
    "PhaseStep",
    "cpmg_two_step_phase_cycle",
    "pgste_stimulated_echo_phase_cycle",
]
