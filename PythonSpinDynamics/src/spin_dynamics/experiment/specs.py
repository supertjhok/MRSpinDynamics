"""Declarative experiment specs for the unified workflow facade.

These frozen dataclasses describe *what* to simulate; they never implement
dynamics. ``Experiment.plan()`` resolves a spec against the workflow registry
and reports compatibility issues; ``Experiment.run()`` delegates to the
validated ``spin_dynamics.workflows`` functions. Optional fields left at
their defaults defer to the wrapped workflow's own defaults, so a default
``Experiment`` reproduces the direct workflow call bit for bit.

Fields typed as ``Mapping | None`` accept the same plain-mapping forms the
underlying workflows accept (e.g. ``noise={"sigma": 0.01}``); rich spec
objects such as :class:`~spin_dynamics.noise.NoiseSpec` also work, and
serialize as long as their contents are JSON-representable.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from spin_dynamics.experiment.serialization import decode, encode, register_serializable

PROBE_NAMES = ("ideal", "tuned", "untuned", "matched")


@register_serializable
@dataclass(frozen=True)
class Sample:
    """Homogeneous sample description.

    ``t1_seconds``/``t2_seconds`` left as ``None`` defer to the wrapped
    workflow's defaults. Asymptotic (steady-state) workflows do not consume
    relaxation times at all; ``plan()`` warns when they would be ignored.
    """

    t1_seconds: float | None = None
    t2_seconds: float | None = None
    label: str = ""


@register_serializable
@dataclass(frozen=True)
class Hardware:
    """Transmit/receive hardware description.

    PR-1 scope: probe selection plus probe-circuit perturbations. Geometry,
    field solvers, and non-inductive detectors join in later milestones.
    """

    probe: str = "ideal"
    q_value: float | None = None
    mistuning_offset: float | None = None
    radiation_damping: Mapping[str, Any] | Any | None = None
    absolute_phase: Mapping[str, Any] | Any | None = None


@register_serializable
@dataclass(frozen=True)
class Acquisition:
    """Offset grid, receiver noise, and rephasing-guard configuration."""

    numpts: int = 101
    maxoffs: float = 10.0
    noise: Mapping[str, Any] | float | Any | None = None
    auto_refine_grid: bool = False
    rephase_safety_factor: float = 1.25
    rephase_action: str = "warn"


@register_serializable
@dataclass(frozen=True)
class CPMG:
    """Asymptotic (infinite-train) CPMG echo, no relaxation."""


@register_serializable
@dataclass(frozen=True)
class CPMGTrain:
    """Finite CPMG echo train with relaxation."""

    num_echoes: int = 8


@register_serializable
@dataclass(frozen=True)
class CPMGIRTrain:
    """Finite CPMG echo train preceded by an inversion-recovery delay sweep."""

    num_echoes: int = 10
    echo_spacing_seconds: float = 0.5e-3
    tauvect: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.tauvect is not None and not isinstance(self.tauvect, tuple):
            values: Iterable[float] = np.asarray(self.tauvect, dtype=np.float64).reshape(-1)
            object.__setattr__(self, "tauvect", tuple(float(v) for v in values))


SEQUENCE_TYPES: tuple[type, ...] = (CPMG, CPMGTrain, CPMGIRTrain)


@register_serializable
@dataclass(frozen=True)
class Experiment:
    """A complete declarative experiment description."""

    sequence: Any
    sample: Sample = field(default_factory=Sample)
    hardware: Hardware = field(default_factory=Hardware)
    acquisition: Acquisition = field(default_factory=Acquisition)

    def plan(self, *, estimate: bool = True) -> "Any":
        from spin_dynamics.experiment.plan import plan_experiment

        return plan_experiment(self, estimate=estimate)

    def run(self, **execution: Any) -> "Any":
        from spin_dynamics.experiment.runner import run_experiment

        return run_experiment(self, **execution)

    def to_dict(self) -> dict[str, Any]:
        return encode(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "Experiment":
        experiment = decode(dict(data))
        if not isinstance(experiment, Experiment):
            raise TypeError("serialized payload does not describe an Experiment")
        return experiment

    @staticmethod
    def from_json(payload: str) -> "Experiment":
        return Experiment.from_dict(json.loads(payload))


def _differs(value: Any, default: Any) -> bool:
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return not np.array_equal(np.asarray(value), np.asarray(default))
    return bool(value != default)


def non_default_fields(experiment: Experiment) -> dict[str, Any]:
    """Return dotted spec-field names whose values differ from the defaults.

    The sequence spec is excluded: its type selects the workflow, so all of
    its fields are honored by construction.
    """

    out: dict[str, Any] = {}
    for group_name in ("sample", "hardware", "acquisition"):
        group = getattr(experiment, group_name)
        for spec_field in dataclasses.fields(group):
            value = getattr(group, spec_field.name)
            if spec_field.default is not dataclasses.MISSING:
                default = spec_field.default
            elif spec_field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                default = spec_field.default_factory()  # type: ignore[misc]
            else:
                continue
            if _differs(value, default):
                out[f"{group_name}.{spec_field.name}"] = value
    return out
