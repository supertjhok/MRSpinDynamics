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


def _as_optional_map(value: Any, name: str) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    return arr


@register_serializable
@dataclass(frozen=True, eq=False)
class Phantom:
    """Spatial sample description for imaging: density plus optional maps.

    ``rho`` is the 2-D proton-density map; ``t1_map``/``t2_map`` override the
    scalar ``Sample`` relaxation times per voxel when given.
    """

    rho: np.ndarray
    t1_map: np.ndarray | None = None
    t2_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        rho = _as_optional_map(self.rho, "rho")
        if rho is None:
            raise ValueError("rho is required")
        object.__setattr__(self, "rho", rho)
        for name in ("t1_map", "t2_map"):
            arr = _as_optional_map(getattr(self, name), name)
            if arr is not None and arr.shape != rho.shape:
                raise ValueError(f"{name} must have the same shape as rho")
            object.__setattr__(self, name, arr)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Phantom):
            return NotImplemented

        def same(a: np.ndarray | None, b: np.ndarray | None) -> bool:
            if a is None or b is None:
                return a is None and b is None
            return bool(np.array_equal(a, b))

        return (
            same(self.rho, other.rho)
            and same(self.t1_map, other.t1_map)
            and same(self.t2_map, other.t2_map)
        )


@register_serializable
@dataclass(frozen=True)
class Sample:
    """Sample description.

    ``t1_seconds``/``t2_seconds`` left as ``None`` defer to the wrapped
    workflow's defaults; asymptotic (steady-state) workflows do not consume
    relaxation times at all, and ``plan()`` warns when they would be ignored.
    Imaging sequences additionally require a :class:`Phantom`; scalar
    relaxation times then broadcast to uniform maps unless the phantom
    carries its own.
    """

    t1_seconds: float | None = None
    t2_seconds: float | None = None
    phantom: Phantom | None = None
    site: Any | None = None
    """Quadrupolar site (``spin_dynamics.nqr.QuadrupolarSite``) for NQR sequences."""
    label: str = ""


@register_serializable
@dataclass(frozen=True)
class Hardware:
    """Transmit/receive hardware description.

    ``probe`` plus the circuit perturbations apply to all workflows. The
    geometry fields (``b0``, ``tx_coil``, ``rx_coil``, ``plane`` — see
    :mod:`spin_dynamics.experiment.hardware`) drive an automatic Biot-Savart
    field solve for imaging sequences: the transverse-B1 maps are computed on
    the phantom grid and passed to the workflow, replacing its synthetic
    default. Non-inductive detectors join in a later milestone.
    """

    probe: str = "ideal"
    q_value: float | None = None
    mistuning_offset: float | None = None
    radiation_damping: Mapping[str, Any] | Any | None = None
    absolute_phase: Mapping[str, Any] | Any | None = None
    b0: Any | None = None
    tx_coil: Any | None = None
    rx_coil: Any | None = None
    plane: Any | None = None


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


@register_serializable
@dataclass(frozen=True)
class CPMGImaging:
    """Phase-encoded CPMG imaging (2-D spin-warp on a phantom).

    ``fov``, ``ny``, and ``maxoffs`` are the imaging workflow's own
    acquisition-geometry parameters (units per the workflow); the physical
    placement of the phantom for coil-field solving is set separately via
    ``Hardware.plane`` in meters.
    """

    num_echoes: int = 2
    echo_spacing_seconds: float = 0.2e-3
    gradient_duration_seconds: float = 0.5e-3
    fov: tuple[float, float] = (20.0, 20.0)
    ny: int = 9
    maxoffs: float = 5.0

    def __post_init__(self) -> None:
        fov = tuple(float(v) for v in self.fov)
        if len(fov) != 2:
            raise ValueError("fov must contain two values")
        object.__setattr__(self, "fov", fov)


def _validate_nqr_common(spec: Any) -> None:
    if spec.pulse_duration_seconds <= 0:
        raise ValueError("pulse_duration_seconds must be positive")
    if spec.nutation_hz <= 0:
        raise ValueError("nutation_hz must be positive")
    if spec.orientations not in ("powder", "single"):
        raise ValueError("orientations must be 'powder' or 'single'")
    if spec.t2e_seconds is not None and spec.t2e_seconds <= 0:
        raise ValueError("t2e_seconds must be positive when set")


@register_serializable
@dataclass(frozen=True)
class NQRSLSE:
    """Spin-lock spin-echo NQR detection train.

    ``nutation_hz`` uses the reduced engine's convention: the *effective*
    two-level Rabi rate of the addressed transition at full RF coupling, so
    the on-resonance flip angle is ``2*pi*nutation_hz*pulse_duration_seconds``
    (90 degrees at ``nutation_hz * duration = 0.25``). The conversion to the
    bare ``gamma*B1/(2*pi)`` the full-model engine expects happens in the
    adapter. ``model="auto"`` picks the reduced two-level or full
    density-matrix engine via ``select_nqr_model``; ``transition="auto"``
    addresses the strongest line.
    """

    pulse_duration_seconds: float
    nutation_hz: float
    echo_spacing_seconds: float
    num_echoes: int
    transition: str = "auto"
    model: str = "auto"
    phase: float = 0.0
    rf_frequency_hz: float | None = None
    orientations: str = "powder"
    b0_tesla: float = 0.0
    t2e_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_nqr_common(self)
        if self.model not in ("auto", "reduced", "full"):
            raise ValueError("model must be 'auto', 'reduced', or 'full'")
        if self.num_echoes <= 0:
            raise ValueError("num_echoes must be positive")
        if self.echo_spacing_seconds < 0:
            raise ValueError("echo_spacing_seconds must be non-negative")


@register_serializable
@dataclass(frozen=True)
class NQRSORC:
    """Strong off-resonance comb NQR train (reduced spin-1 engine only).

    Same ``nutation_hz`` convention as :class:`NQRSLSE`.
    """

    pulse_duration_seconds: float
    nutation_hz: float
    half_spacing_seconds: float
    num_pulses: int
    transition: str = "auto"
    phase: float = 0.0
    rf_frequency_hz: float | None = None
    orientations: str = "powder"
    b0_tesla: float = 0.0
    t2e_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_nqr_common(self)
        if self.num_pulses <= 0:
            raise ValueError("num_pulses must be positive")
        if self.half_spacing_seconds < 0:
            raise ValueError("half_spacing_seconds must be non-negative")


SEQUENCE_TYPES: tuple[type, ...] = (
    CPMG,
    CPMGTrain,
    CPMGIRTrain,
    CPMGImaging,
    NQRSLSE,
    NQRSORC,
)


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
