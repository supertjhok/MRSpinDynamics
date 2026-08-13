"""Optional openEMS backend for solver-neutral harmonic electromagnetic fields.

This module has no import-time dependency on openEMS, CSXCAD, or h5py.  It
serializes a validated project to a standalone Python driver that can be run by
any external Python environment containing the openEMS bindings, then imports
the resulting frequency-domain HDF5 dumps into :class:`HarmonicEMSolution`.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from xml.etree import ElementTree
from typing import Literal

import numpy as np

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.harmonic import (
    HarmonicConvergence,
    HarmonicEMSolution,
    HarmonicFieldNormalization,
    HarmonicMaterial,
    HarmonicPort,
    HarmonicSolverProvenance,
    save_harmonic_em_npz,
)


OPENEMS_PROJECT_SCHEMA = "python-spin-dynamics.openems-project/v1"
_PROJECT_FILENAME = "openems_project.json"
_DRIVER_FILENAME = "run_openems.py"
_RUN_METADATA_FILENAME = "openems_run.json"
_PORT_METADATA_FILENAME = "openems_port.json"

PortKind = Literal["curve", "lumped"]
NormalizationKind = Literal[
    "absolute",
    "per_ampere",
    "per_volt",
    "per_sqrt_watt",
]


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return result


def _vector3(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3-vector")
    result.setflags(write=False)
    return result


def _polyline(value: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] < 2:
        raise ValueError(f"{name} must have shape (num_points >= 2, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite coordinates")
    if np.any(np.linalg.norm(np.diff(result, axis=0), axis=1) <= 0.0):
        raise ValueError(f"{name} must not contain zero-length segments")
    result.setflags(write=False)
    return result


def _bounds3(value: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.shape != (3, 2) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape (3, 2)")
    if np.any(result[:, 1] <= result[:, 0]):
        raise ValueError(f"each {name} upper bound must exceed its lower bound")
    result.setflags(write=False)
    return result


def _json_mapping(value: Mapping[str, object], name: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a mapping")
    return decoded


def segments_to_polylines(
    segments: Sequence[tuple[Sequence[float], Sequence[float]]],
    *,
    tolerance_m: float = 1.0e-12,
) -> tuple[np.ndarray, ...]:
    """Join ordered, contiguous straight segments into open or closed polylines.

    A discontinuity starts a new polyline, which allows the flat segment lists
    returned by multi-loop coil generators to be converted without inventing
    electrical connections between loops.  A segment whose end, rather than
    start, matches the current endpoint is reversed automatically.
    """

    tolerance = _finite_nonnegative(tolerance_m, "tolerance_m")
    if not segments:
        raise ValueError("segments must not be empty")
    polylines: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for index, (start_value, stop_value) in enumerate(segments):
        start = _vector3(start_value, f"segments[{index}].start")
        stop = _vector3(stop_value, f"segments[{index}].stop")
        if np.linalg.norm(stop - start) <= tolerance:
            raise ValueError(f"segments[{index}] has zero length")
        if not current:
            current = [start, stop]
            continue
        endpoint = current[-1]
        if np.linalg.norm(start - endpoint) <= tolerance:
            current.append(stop)
        elif np.linalg.norm(stop - endpoint) <= tolerance:
            current.append(start)
        else:
            polylines.append(_polyline(current, f"polyline {len(polylines)}"))
            current = [start, stop]
    if current:
        polylines.append(_polyline(current, f"polyline {len(polylines)}"))
    return tuple(polylines)


@dataclass(frozen=True)
class OpenEMSWire:
    """One PEC wire property containing one or more piecewise-linear paths."""

    name: str
    polylines_m: tuple[np.ndarray, ...]
    radius_m: float
    priority: int = 10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("wire name must not be empty")
        polylines = tuple(
            _polyline(points, f"{self.name}.polylines_m[{index}]")
            for index, points in enumerate(self.polylines_m)
        )
        if not polylines:
            raise ValueError("wire polylines_m must not be empty")
        radius = _finite_positive(self.radius_m, "wire radius_m")
        priority = int(self.priority)
        if priority < 0:
            raise ValueError("wire priority must be non-negative")
        object.__setattr__(self, "polylines_m", polylines)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "priority", priority)

    @classmethod
    def from_segments(
        cls,
        name: str,
        segments: Sequence[tuple[Sequence[float], Sequence[float]]],
        *,
        radius_m: float,
        priority: int = 10,
        tolerance_m: float = 1.0e-12,
    ) -> OpenEMSWire:
        """Build a wire from the package's common straight-segment format."""

        return cls(
            name=name,
            polylines_m=segments_to_polylines(
                segments,
                tolerance_m=tolerance_m,
            ),
            radius_m=radius_m,
            priority=priority,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "polylines_m": [points.tolist() for points in self.polylines_m],
            "radius_m": self.radius_m,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSWire:
        raw = value.get("polylines_m")
        if not isinstance(raw, list):
            raise ValueError("wire polylines_m must be a list")
        return cls(
            name=str(value["name"]),
            polylines_m=tuple(np.asarray(points) for points in raw),
            radius_m=float(value["radius_m"]),
            priority=int(value.get("priority", 10)),
        )


@dataclass(frozen=True)
class OpenEMSPlanarConductor:
    """Axis-aligned planar PEC polygon with a mesh-resolved strip width."""

    name: str
    points_m: np.ndarray
    normal_axis: Literal["x", "y", "z"] = "x"
    priority: int = 10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("planar conductor name must not be empty")
        points = np.array(self.points_m, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError(
                "planar conductor points_m must have shape (num_points >= 3, 3)"
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("planar conductor points_m must contain finite values")
        if self.normal_axis not in {"x", "y", "z"}:
            raise ValueError("planar conductor normal_axis must be 'x', 'y', or 'z'")
        normal_index = "xyz".index(self.normal_axis)
        if not np.allclose(
            points[:, normal_index],
            points[0, normal_index],
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("planar conductor points_m must be coplanar")
        tangential = np.delete(points, normal_index, axis=1)
        signed_area = 0.5 * np.sum(
            tangential[:, 0] * np.roll(tangential[:, 1], -1)
            - tangential[:, 1] * np.roll(tangential[:, 0], -1)
        )
        if abs(float(signed_area)) <= np.finfo(np.float64).eps:
            raise ValueError("planar conductor polygon must have non-zero area")
        priority = int(self.priority)
        if priority < 0:
            raise ValueError("planar conductor priority must be non-negative")
        points.setflags(write=False)
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "priority", priority)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "points_m": self.points_m.tolist(),
            "normal_axis": self.normal_axis,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSPlanarConductor:
        return cls(
            name=str(value["name"]),
            points_m=np.asarray(value["points_m"]),
            normal_axis=str(value.get("normal_axis", "x")),  # type: ignore[arg-type]
            priority=int(value.get("priority", 10)),
        )


@dataclass(frozen=True)
class OpenEMSCylindricalSample:
    """Homogeneous lossy cylindrical material region."""

    name: str
    center_m: np.ndarray
    radius_m: float
    length_m: float
    axis: Literal["x", "y", "z"] = "z"
    relative_permittivity: float = 1.0
    conductivity_s_per_m: float = 0.0
    relative_permeability: float = 1.0
    priority: int = 5
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("sample name must not be empty")
        center = _vector3(self.center_m, "sample center_m")
        radius = _finite_positive(self.radius_m, "sample radius_m")
        length = _finite_positive(self.length_m, "sample length_m")
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("sample axis must be 'x', 'y', or 'z'")
        epsilon = _finite_positive(
            self.relative_permittivity,
            "sample relative_permittivity",
        )
        conductivity = _finite_nonnegative(
            self.conductivity_s_per_m,
            "sample conductivity_s_per_m",
        )
        permeability = _finite_positive(
            self.relative_permeability,
            "sample relative_permeability",
        )
        priority = int(self.priority)
        if priority < 0:
            raise ValueError("sample priority must be non-negative")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "radius_m", radius)
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "relative_permittivity", epsilon)
        object.__setattr__(self, "conductivity_s_per_m", conductivity)
        object.__setattr__(self, "relative_permeability", permeability)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "sample metadata"),
        )

    @property
    def endpoints_m(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the cylinder-axis start and stop coordinates."""

        axis_index = {"x": 0, "y": 1, "z": 2}[self.axis]
        start = np.array(self.center_m, copy=True)
        stop = np.array(self.center_m, copy=True)
        start[axis_index] -= 0.5 * self.length_m
        stop[axis_index] += 0.5 * self.length_m
        return start, stop

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center_m": self.center_m.tolist(),
            "radius_m": self.radius_m,
            "length_m": self.length_m,
            "axis": self.axis,
            "relative_permittivity": self.relative_permittivity,
            "conductivity_s_per_m": self.conductivity_s_per_m,
            "relative_permeability": self.relative_permeability,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSCylindricalSample:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("sample metadata must be a mapping")
        return cls(
            name=str(value["name"]),
            center_m=np.asarray(value["center_m"]),
            radius_m=float(value["radius_m"]),
            length_m=float(value["length_m"]),
            axis=str(value.get("axis", "z")),  # type: ignore[arg-type]
            relative_permittivity=float(value.get("relative_permittivity", 1.0)),
            conductivity_s_per_m=float(value.get("conductivity_s_per_m", 0.0)),
            relative_permeability=float(value.get("relative_permeability", 1.0)),
            priority=int(value.get("priority", 5)),
            metadata=metadata,
        )


@dataclass(frozen=True)
class OpenEMSLumpedPort:
    """Curve-based or box-based lumped excitation and measurement port."""

    number: int
    start_m: np.ndarray
    stop_m: np.ndarray
    resistance_ohm: float = 50.0
    kind: PortKind = "curve"
    direction: Literal["x", "y", "z"] | None = None
    excite: bool = True
    priority: int = 20
    name: str = "feed"

    def __post_init__(self) -> None:
        number = int(self.number)
        if number < 1:
            raise ValueError("openEMS port number must be at least 1")
        start = _vector3(self.start_m, "port start_m")
        stop = _vector3(self.stop_m, "port stop_m")
        if np.linalg.norm(stop - start) <= 0.0:
            raise ValueError("port start_m and stop_m must differ")
        resistance = _finite_nonnegative(self.resistance_ohm, "port resistance_ohm")
        if self.kind not in {"curve", "lumped"}:
            raise ValueError("port kind must be 'curve' or 'lumped'")
        direction = self.direction
        if self.kind == "lumped":
            if direction is None:
                direction = "xyz"[int(np.argmax(np.abs(stop - start)))]
            if direction not in {"x", "y", "z"}:
                raise ValueError("lumped port direction must be 'x', 'y', or 'z'")
        priority = int(self.priority)
        if priority < 0:
            raise ValueError("port priority must be non-negative")
        if not self.name:
            raise ValueError("port name must not be empty")
        object.__setattr__(self, "number", number)
        object.__setattr__(self, "start_m", start)
        object.__setattr__(self, "stop_m", stop)
        object.__setattr__(self, "resistance_ohm", resistance)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "priority", priority)

    def to_dict(self) -> dict[str, object]:
        return {
            "number": self.number,
            "start_m": self.start_m.tolist(),
            "stop_m": self.stop_m.tolist(),
            "resistance_ohm": self.resistance_ohm,
            "kind": self.kind,
            "direction": self.direction,
            "excite": self.excite,
            "priority": self.priority,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSLumpedPort:
        return cls(
            number=int(value["number"]),
            start_m=np.asarray(value["start_m"]),
            stop_m=np.asarray(value["stop_m"]),
            resistance_ohm=float(value.get("resistance_ohm", 50.0)),
            kind=str(value.get("kind", "curve")),  # type: ignore[arg-type]
            direction=(
                None if value.get("direction") is None else str(value["direction"])
            ),  # type: ignore[arg-type]
            excite=bool(value.get("excite", True)),
            priority=int(value.get("priority", 20)),
            name=str(value.get("name", "feed")),
        )


@dataclass(frozen=True)
class OpenEMSSettings:
    """FDTD excitation, termination, boundary, and mesh settings."""

    frequency_hz: float
    gaussian_bandwidth_hz: float | None = None
    max_cell_size_m: float = 5.0e-3
    mesh_growth_ratio: float = 1.4
    max_timesteps: int = 100_000
    end_criteria: float = 1.0e-5
    boundary_conditions: tuple[str, ...] = ("PML_8",) * 6

    def __post_init__(self) -> None:
        frequency = _finite_positive(self.frequency_hz, "frequency_hz")
        bandwidth = (
            0.5 * frequency
            if self.gaussian_bandwidth_hz is None
            else _finite_positive(
                self.gaussian_bandwidth_hz,
                "gaussian_bandwidth_hz",
            )
        )
        cell = _finite_positive(self.max_cell_size_m, "max_cell_size_m")
        ratio = _finite_positive(self.mesh_growth_ratio, "mesh_growth_ratio")
        if ratio <= 1.0:
            raise ValueError("mesh_growth_ratio must exceed 1")
        timesteps = int(self.max_timesteps)
        if timesteps < 1:
            raise ValueError("max_timesteps must be positive")
        end_criteria = _finite_positive(self.end_criteria, "end_criteria")
        if end_criteria >= 1.0:
            raise ValueError("end_criteria must be less than 1")
        boundaries = tuple(str(item) for item in self.boundary_conditions)
        allowed = {"PEC", "PMC", "MUR", "PML_8"}
        if len(boundaries) != 6 or any(item not in allowed for item in boundaries):
            raise ValueError(
                "boundary_conditions must contain six PEC/PMC/MUR/PML_8 entries"
            )
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "gaussian_bandwidth_hz", bandwidth)
        object.__setattr__(self, "max_cell_size_m", cell)
        object.__setattr__(self, "mesh_growth_ratio", ratio)
        object.__setattr__(self, "max_timesteps", timesteps)
        object.__setattr__(self, "end_criteria", end_criteria)
        object.__setattr__(self, "boundary_conditions", boundaries)

    def to_dict(self) -> dict[str, object]:
        return {
            "frequency_hz": self.frequency_hz,
            "gaussian_bandwidth_hz": self.gaussian_bandwidth_hz,
            "max_cell_size_m": self.max_cell_size_m,
            "mesh_growth_ratio": self.mesh_growth_ratio,
            "max_timesteps": self.max_timesteps,
            "end_criteria": self.end_criteria,
            "boundary_conditions": list(self.boundary_conditions),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSSettings:
        boundaries = value.get("boundary_conditions", ["PML_8"] * 6)
        if not isinstance(boundaries, list):
            raise ValueError("boundary_conditions must be a list")
        return cls(
            frequency_hz=float(value["frequency_hz"]),
            gaussian_bandwidth_hz=(
                None
                if value.get("gaussian_bandwidth_hz") is None
                else float(value["gaussian_bandwidth_hz"])
            ),
            max_cell_size_m=float(value.get("max_cell_size_m", 5.0e-3)),
            mesh_growth_ratio=float(value.get("mesh_growth_ratio", 1.4)),
            max_timesteps=int(value.get("max_timesteps", 100_000)),
            end_criteria=float(value.get("end_criteria", 1.0e-5)),
            boundary_conditions=tuple(str(item) for item in boundaries),
        )


@dataclass(frozen=True)
class OpenEMSProject:
    """Complete solver-neutral description consumed by the openEMS driver."""

    name: str
    settings: OpenEMSSettings
    simulation_bounds_m: np.ndarray
    field_domain: SpatialDomain
    wires: tuple[OpenEMSWire, ...]
    ports: tuple[OpenEMSLumpedPort, ...]
    samples: tuple[OpenEMSCylindricalSample, ...] = ()
    planar_conductors: tuple[OpenEMSPlanarConductor, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("project name must not be empty")
        if not isinstance(self.settings, OpenEMSSettings):
            raise TypeError("settings must be OpenEMSSettings")
        bounds = _bounds3(self.simulation_bounds_m, "simulation_bounds_m")
        if self.field_domain.ndim != 3:
            raise ValueError("openEMS field_domain must be three-dimensional")
        for index, axis in enumerate(self.field_domain.axes):
            if axis[0] < bounds[index, 0] or axis[-1] > bounds[index, 1]:
                raise ValueError("field_domain must lie inside simulation_bounds_m")
        wires = tuple(self.wires)
        ports = tuple(self.ports)
        samples = tuple(self.samples)
        conductors = tuple(self.planar_conductors)
        if not all(isinstance(item, OpenEMSWire) for item in wires):
            raise TypeError("wires must contain OpenEMSWire objects")
        if not all(isinstance(item, OpenEMSPlanarConductor) for item in conductors):
            raise TypeError(
                "planar_conductors must contain OpenEMSPlanarConductor objects"
            )
        if not wires and not conductors:
            raise ValueError("at least one wire or planar conductor is required")
        if not ports or not all(isinstance(item, OpenEMSLumpedPort) for item in ports):
            raise TypeError("ports must contain at least one OpenEMSLumpedPort")
        if not all(isinstance(item, OpenEMSCylindricalSample) for item in samples):
            raise TypeError("samples must contain OpenEMSCylindricalSample objects")
        conductor_names = [item.name for item in (*wires, *conductors)]
        if len(set(conductor_names)) != len(conductor_names):
            raise ValueError("wire and planar conductor names must be unique")
        if len({item.number for item in ports}) != len(ports):
            raise ValueError("port numbers must be unique")
        if sum(bool(item.excite) for item in ports) != 1:
            raise ValueError("exactly one port must be excited per openEMS project")
        object.__setattr__(self, "simulation_bounds_m", bounds)
        object.__setattr__(self, "wires", wires)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "planar_conductors", conductors)
        object.__setattr__(
            self,
            "metadata",
            _json_mapping(self.metadata, "project metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": OPENEMS_PROJECT_SCHEMA,
            "name": self.name,
            "settings": self.settings.to_dict(),
            "simulation_bounds_m": self.simulation_bounds_m.tolist(),
            "field_axes_m": [axis.tolist() for axis in self.field_domain.axes],
            "wires": [item.to_dict() for item in self.wires],
            "planar_conductors": [
                item.to_dict() for item in self.planar_conductors
            ],
            "ports": [item.to_dict() for item in self.ports],
            "samples": [item.to_dict() for item in self.samples],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> OpenEMSProject:
        schema = str(value.get("schema", ""))
        if schema != OPENEMS_PROJECT_SCHEMA:
            raise ValueError(
                f"unsupported openEMS project schema {schema!r}; "
                f"expected {OPENEMS_PROJECT_SCHEMA!r}"
            )
        settings = value.get("settings")
        axes = value.get("field_axes_m")
        wires = value.get("wires")
        conductors = value.get("planar_conductors", [])
        ports = value.get("ports")
        samples = value.get("samples", [])
        metadata = value.get("metadata", {})
        if not isinstance(settings, Mapping):
            raise ValueError("settings must be an object")
        if not isinstance(axes, list) or len(axes) != 3:
            raise ValueError("field_axes_m must contain three axes")
        if not isinstance(wires, list) or not all(
            isinstance(item, Mapping) for item in wires
        ):
            raise ValueError("wires must be a list of objects")
        if not isinstance(conductors, list) or not all(
            isinstance(item, Mapping) for item in conductors
        ):
            raise ValueError("planar_conductors must be a list of objects")
        if not isinstance(ports, list) or not all(
            isinstance(item, Mapping) for item in ports
        ):
            raise ValueError("ports must be a list of objects")
        if not isinstance(samples, list) or not all(
            isinstance(item, Mapping) for item in samples
        ):
            raise ValueError("samples must be a list of objects")
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be an object")
        return cls(
            name=str(value["name"]),
            settings=OpenEMSSettings.from_dict(settings),
            simulation_bounds_m=np.asarray(value["simulation_bounds_m"]),
            field_domain=SpatialDomain(tuple(np.asarray(axis) for axis in axes)),
            wires=tuple(OpenEMSWire.from_dict(item) for item in wires),
            planar_conductors=tuple(
                OpenEMSPlanarConductor.from_dict(item) for item in conductors
            ),
            ports=tuple(OpenEMSLumpedPort.from_dict(item) for item in ports),
            samples=tuple(
                OpenEMSCylindricalSample.from_dict(item) for item in samples
            ),
            metadata=metadata,
        )

    @property
    def model_hash(self) -> str:
        """Return a stable SHA-256 hash of the canonical project JSON."""

        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def loaded_loop_openems_reference(
    *,
    frequency_hz: float = 128.0e6,
    loop_radius_m: float = 0.065,
    wire_radius_m: float = 1.5e-3,
    strip_width_m: float | None = 6.0e-3,
    feed_gap_m: float = 5.0e-3,
    sample_radius_m: float = 0.045,
    sample_length_m: float = 0.10,
    relative_permittivity: float = 80.0,
    conductivity_s_per_m: float = 0.5,
    port_resistance_ohm: float = 50.0,
    n_wire_points: int = 96,
) -> OpenEMSProject:
    """Return the Phase 3 high-permittivity, lossy loaded-loop reference case.

    The loop lies in the y-z plane, so its central field is transverse to a
    conventional +z B0.  A symmetric gap at +y is bridged by a geometry-aligned
    lumped port.  The sample is a homogeneous cylinder along the loop's x
    axis, so the strip encloses the sample without intersecting it.
    """

    frequency = _finite_positive(frequency_hz, "frequency_hz")
    loop_radius = _finite_positive(loop_radius_m, "loop_radius_m")
    wire_radius = _finite_positive(wire_radius_m, "wire_radius_m")
    gap = _finite_positive(feed_gap_m, "feed_gap_m")
    if gap >= 2.0 * loop_radius:
        raise ValueError("feed_gap_m must be smaller than the loop diameter")
    point_count = int(n_wire_points)
    if point_count < 16:
        raise ValueError("n_wire_points must be at least 16")
    strip_width = (
        2.0 * wire_radius
        if strip_width_m is None
        else _finite_positive(strip_width_m, "strip_width_m")
    )
    if strip_width >= 2.0 * loop_radius:
        raise ValueError("strip_width_m must be smaller than the loop diameter")
    inner_radius = loop_radius - 0.5 * strip_width
    outer_radius = loop_radius + 0.5 * strip_width
    if gap >= 2.0 * inner_radius:
        raise ValueError("feed_gap_m must be smaller than the inner diameter")
    outer_half_angle = np.arcsin(gap / (2.0 * outer_radius))
    inner_half_angle = np.arcsin(gap / (2.0 * inner_radius))
    outer_angles = np.linspace(
        outer_half_angle,
        2.0 * np.pi - outer_half_angle,
        point_count,
    )
    inner_angles = np.linspace(
        2.0 * np.pi - inner_half_angle,
        inner_half_angle,
        point_count,
    )
    points = np.zeros((2 * point_count, 3), dtype=np.float64)
    points[:point_count, 1] = outer_radius * np.cos(outer_angles)
    points[:point_count, 2] = outer_radius * np.sin(outer_angles)
    points[point_count:, 1] = inner_radius * np.cos(inner_angles)
    points[point_count:, 2] = inner_radius * np.sin(inner_angles)
    conductor = OpenEMSPlanarConductor(
        "coil",
        points,
        normal_axis="x",
        priority=10,
    )
    port_start = np.array([0.0, points[-1, 1], -0.5 * gap])
    port_stop = np.array([0.0, points[0, 1], 0.5 * gap])
    port = OpenEMSLumpedPort(
        1,
        start_m=port_start,
        stop_m=port_stop,
        resistance_ohm=port_resistance_ohm,
        kind="lumped",
        direction="z",
        excite=True,
        priority=20,
        name="feed",
    )
    sample = OpenEMSCylindricalSample(
        "high_epsilon_sample",
        center_m=np.zeros(3),
        radius_m=sample_radius_m,
        length_m=sample_length_m,
        axis="x",
        relative_permittivity=relative_permittivity,
        conductivity_s_per_m=conductivity_s_per_m,
        priority=5,
        metadata={"role": "Phase 3 loaded-loop reference"},
    )
    padding = max(0.06, 0.75 * loop_radius)
    transverse_extent = loop_radius + padding
    axial_extent = max(loop_radius, 0.5 * sample_length_m) + padding
    bounds = np.array(
        [
            [-transverse_extent, transverse_extent],
            [-transverse_extent, transverse_extent],
            [-axial_extent, axial_extent],
        ]
    )
    axes = (
        np.linspace(-0.5 * sample_length_m, 0.5 * sample_length_m, 21),
        np.linspace(-sample_radius_m, sample_radius_m, 17),
        np.linspace(-sample_radius_m, sample_radius_m, 17),
    )
    c0 = 299_792_458.0
    wavelength_sample = c0 / (
        frequency * np.sqrt(float(relative_permittivity))
    )
    max_cell = min(5.0e-3, wavelength_sample / 20.0)
    return OpenEMSProject(
        name="loaded_loop_high_epsilon_reference",
        settings=OpenEMSSettings(
            frequency_hz=frequency,
            gaussian_bandwidth_hz=0.5 * frequency,
            max_cell_size_m=max_cell,
            max_timesteps=100_000,
            end_criteria=1.0e-5,
        ),
        simulation_bounds_m=bounds,
        field_domain=SpatialDomain(axes),
        wires=(),
        ports=(port,),
        samples=(sample,),
        planar_conductors=(conductor,),
        metadata={
            "reference_case": "phase3-loaded-loop",
            "b0_direction": [0.0, 0.0, 1.0],
            "conductor_model": "mesh-resolved planar annular strip",
            "strip_width_m": strip_width,
            "validation_status": "Phase 4 validation required",
        },
    )


def unloaded_loop_openems_reference(
    *,
    frequency_hz: float = 32.0e6,
    loop_radius_m: float = 0.065,
    wire_radius_m: float = 1.5e-3,
    strip_width_m: float | None = 6.0e-3,
    feed_gap_m: float = 5.0e-3,
    port_resistance_ohm: float = 50.0,
    n_wire_points: int = 96,
    max_cell_size_m: float = 5.0e-3,
    max_timesteps: int = 250_000,
) -> OpenEMSProject:
    """Return an unloaded loop for the Phase 4 Biot--Savart benchmark."""

    project = loaded_loop_openems_reference(
        frequency_hz=frequency_hz,
        loop_radius_m=loop_radius_m,
        wire_radius_m=wire_radius_m,
        strip_width_m=strip_width_m,
        feed_gap_m=feed_gap_m,
        port_resistance_ohm=port_resistance_ohm,
        n_wire_points=n_wire_points,
    )
    settings = replace(
        project.settings,
        max_cell_size_m=max_cell_size_m,
        max_timesteps=max_timesteps,
    )
    return replace(
        project,
        name="unloaded_loop_biot_savart_reference",
        settings=settings,
        samples=(),
        metadata={
            "reference_case": "phase4-unloaded-loop-biot-savart",
            "b0_direction": [0.0, 0.0, 1.0],
            "validation_status": "analytical low-frequency benchmark",
            "loop_radius_m": float(loop_radius_m),
            "conductor_model": project.metadata["conductor_model"],
            "strip_width_m": project.metadata["strip_width_m"],
        },
    )


_OPENEMS_DRIVER = r'''"""Standalone openEMS driver generated by PythonSpinDynamics."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

try:
    import openEMS as openems_module
    from CSXCAD import ContinuousStructure
    from openEMS import openEMS
except ImportError as exc:
    raise SystemExit(
        "This driver requires the openEMS and CSXCAD Python bindings in the "
        "selected external Python environment. Original error: " + str(exc)
    ) from exc


def _complex_pair(value):
    value = complex(value)
    return [float(value.real), float(value.imag)]


def _version():
    for distribution in ("openEMS", "openems"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return str(getattr(openems_module, "__version__", "unknown"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()

    project_path = Path(args.project_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    project = json.loads(project_path.read_text(encoding="utf-8"))
    settings = project["settings"]

    fdtd = openEMS(
        NrTS=int(settings["max_timesteps"]),
        EndCriteria=float(settings["end_criteria"]),
    )
    fdtd.SetGaussExcite(
        float(settings["frequency_hz"]),
        float(settings["gaussian_bandwidth_hz"]),
    )
    fdtd.SetBoundaryCond(list(settings["boundary_conditions"]))

    csx = ContinuousStructure()
    fdtd.SetCSX(csx)
    mesh = csx.GetGrid()
    mesh.SetDeltaUnit(1.0)

    feature_lines = [set(), set(), set()]
    for axis_index, bounds in enumerate(project["simulation_bounds_m"]):
        feature_lines[axis_index].update(float(item) for item in bounds)
    for axis_index, axis in enumerate(project["field_axes_m"]):
        feature_lines[axis_index].update(float(item) for item in axis)
    for wire in project["wires"]:
        for polyline in wire["polylines_m"]:
            points = np.asarray(polyline, dtype=float)
            for axis_index in range(3):
                coordinates = points[:, axis_index]
                feature_lines[axis_index].update(
                    (
                        float(coordinates[0]),
                        float(coordinates[-1]),
                        float(np.min(coordinates)),
                        float(np.max(coordinates)),
                    )
                )
    for conductor in project.get("planar_conductors", []):
        points = np.asarray(conductor["points_m"], dtype=float)
        for axis_index in range(3):
            coordinates = points[:, axis_index]
            feature_lines[axis_index].update(
                (
                    float(coordinates[0]),
                    float(coordinates[-1]),
                    float(np.min(coordinates)),
                    float(np.max(coordinates)),
                )
            )
    for port_data in project["ports"]:
        for point in (port_data["start_m"], port_data["stop_m"]):
            for axis_index in range(3):
                feature_lines[axis_index].add(float(point[axis_index]))
    for sample in project["samples"]:
        center = np.asarray(sample["center_m"], dtype=float)
        axis_index = "xyz".index(sample["axis"])
        for index in range(3):
            if index == axis_index:
                feature_lines[index].update(
                    [
                        center[index] - 0.5 * float(sample["length_m"]),
                        center[index] + 0.5 * float(sample["length_m"]),
                    ]
                )
            else:
                feature_lines[index].update(
                    [
                        center[index] - float(sample["radius_m"]),
                        center[index] + float(sample["radius_m"]),
                    ]
                )
    for axis_index, axis_name in enumerate("xyz"):
        mesh.AddLine(axis_name, sorted(feature_lines[axis_index]))
    mesh.SmoothMeshLines(
        "all",
        float(settings["max_cell_size_m"]),
        float(settings["mesh_growth_ratio"]),
        check_symmetry=False,
    )

    for sample in project["samples"]:
        material = csx.AddMaterial(
            sample["name"],
            epsilon=float(sample["relative_permittivity"]),
            mue=float(sample["relative_permeability"]),
            kappa=float(sample["conductivity_s_per_m"]),
        )
        center = np.asarray(sample["center_m"], dtype=float)
        axis_index = "xyz".index(sample["axis"])
        start = center.copy()
        stop = center.copy()
        start[axis_index] -= 0.5 * float(sample["length_m"])
        stop[axis_index] += 0.5 * float(sample["length_m"])
        material.AddCylinder(
            start.tolist(),
            stop.tolist(),
            float(sample["radius_m"]),
            priority=int(sample["priority"]),
        )

    for wire in project["wires"]:
        metal = csx.AddMetal(wire["name"])
        for polyline in wire["polylines_m"]:
            points = np.asarray(polyline, dtype=float)
            metal.AddWire(
                points.T.tolist(),
                float(wire["radius_m"]),
                priority=int(wire["priority"]),
            )

    for conductor in project.get("planar_conductors", []):
        metal = csx.AddMetal(conductor["name"])
        points = np.asarray(conductor["points_m"], dtype=float)
        normal_index = "xyz".index(conductor["normal_axis"])
        tangential_indices = [index for index in range(3) if index != normal_index]
        metal.AddPolygon(
            points[:, tangential_indices].T.tolist(),
            norm_dir=normal_index,
            elevation=float(points[0, normal_index]),
            priority=int(conductor["priority"]),
        )

    ports = []
    for port_data in project["ports"]:
        common = dict(
            port_nr=int(port_data["number"]),
            R=float(port_data["resistance_ohm"]),
            start=list(port_data["start_m"]),
            stop=list(port_data["stop_m"]),
            excite=bool(port_data["excite"]),
            priority=int(port_data["priority"]),
        )
        if port_data["kind"] == "curve":
            port = fdtd.AddCurvePort(**common)
        else:
            port = fdtd.AddLumpedPort(
                p_dir=port_data["direction"],
                edges2grid="all",
                **common,
            )
        ports.append((port_data, port))

    field_start = [float(axis[0]) for axis in project["field_axes_m"]]
    field_stop = [float(axis[-1]) for axis in project["field_axes_m"]]
    frequency = [float(settings["frequency_hz"])]
    e_dump = csx.AddDump(
        "E_fd",
        dump_type=10,
        frequency=frequency,
        file_type=1,
        dump_mode=1,
    )
    e_dump.AddBox(field_start, field_stop)
    j_dump = csx.AddDump(
        "J_fd",
        dump_type=12,
        frequency=frequency,
        file_type=1,
        dump_mode=1,
    )
    j_dump.AddBox(field_start, field_stop)
    b_dump = csx.AddDump(
        "B_fd",
        dump_type=15,
        frequency=frequency,
        file_type=1,
        dump_mode=1,
    )
    b_dump.AddBox(field_start, field_stop)

    csx.Write2XML(str(output_dir / "openems_model.xml"))
    fdtd.Run(
        str(output_dir),
        cleanup=True,
        setup_only=bool(args.setup_only),
        engine="multithreaded",
        numThreads=max(1, min(8, os.cpu_count() or 1)),
    )

    run_metadata = {
        "completed": not args.setup_only,
        "setup_only": bool(args.setup_only),
        "backend": "openEMS",
        "backend_version": _version(),
        "frequency_hz": float(settings["frequency_hz"]),
        "max_timesteps": int(settings["max_timesteps"]),
        "end_criteria": float(settings["end_criteria"]),
        "termination_verified": False,
    }
    (output_dir / "openems_run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.setup_only:
        return

    port_results = []
    target_frequency = np.asarray([float(settings["frequency_hz"])])
    for port_data, port in ports:
        port.CalcPort(str(output_dir), target_frequency)
        port_results.append(
            {
                "number": int(port_data["number"]),
                "name": port_data["name"],
                "voltage_v": _complex_pair(port.uf_tot[0]),
                "current_a": _complex_pair(port.if_tot[0]),
                "accepted_power_w": float(port.P_acc[0]),
                "reference_impedance_ohm": float(port_data["resistance_ohm"]),
            }
        )
    (output_dir / "openems_port.json").write_text(
        json.dumps(port_results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

if __name__ == "__main__":
    main()
'''


def write_openems_project(
    project: OpenEMSProject,
    directory: str | Path,
) -> tuple[Path, Path]:
    """Write project JSON and a standalone external openEMS Python driver."""

    if not isinstance(project, OpenEMSProject):
        raise TypeError("project must be OpenEMSProject")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    project_path = output / _PROJECT_FILENAME
    driver_path = output / _DRIVER_FILENAME
    project_path.write_text(
        json.dumps(project.to_dict(), indent=2, allow_nan=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    driver_path.write_text(_OPENEMS_DRIVER, encoding="utf-8")
    return project_path, driver_path


def load_openems_project(path: str | Path) -> OpenEMSProject:
    """Load and validate a versioned openEMS project JSON file."""

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("openEMS project is not valid JSON") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("openEMS project must be a JSON object")
    return OpenEMSProject.from_dict(raw)


@dataclass(frozen=True)
class OpenEMSAvailability:
    """Result of probing an external Python environment for openEMS."""

    available: bool
    python_executable: str
    version: str | None = None
    detail: str | None = None


def check_openems_python(
    python_executable: str | Path = sys.executable,
    *,
    timeout: float = 15.0,
) -> OpenEMSAvailability:
    """Check whether an external Python can import openEMS and CSXCAD."""

    executable = str(python_executable)
    script = (
        "import openEMS, CSXCAD; "
        "print(getattr(openEMS, '__version__', 'unknown'))"
    )
    try:
        completed = subprocess.run(
            [executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return OpenEMSAvailability(False, executable, detail=str(exc))
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return OpenEMSAvailability(False, executable, detail=detail)
    return OpenEMSAvailability(
        True,
        executable,
        version=completed.stdout.strip() or "unknown",
    )


class OpenEMSExecutionError(RuntimeError):
    """Raised when the external openEMS driver fails or times out."""


@dataclass(frozen=True)
class OpenEMSRunResult:
    """External process result and optional imported harmonic solution."""

    project_directory: Path
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    solution: HarmonicEMSolution | None = None


def _openems_mesh_spacing(path: Path) -> tuple[float | None, float | None]:
    try:
        root = ElementTree.parse(path).getroot()
    except (FileNotFoundError, ElementTree.ParseError):
        return None, None
    grid = root.find("RectilinearGrid")
    if grid is None:
        return None, None
    spacings: list[np.ndarray] = []
    for name in ("XLines", "YLines", "ZLines"):
        element = grid.find(name)
        if element is None or not element.text:
            return None, None
        lines = np.unique(np.fromstring(element.text, sep=","))
        if lines.size < 2:
            return None, None
        spacings.append(np.diff(lines))
    all_spacings = np.concatenate(spacings)
    if np.any(all_spacings <= 0.0):
        return None, None
    return float(np.min(all_spacings)), float(np.max(all_spacings))


def _parse_openems_output(
    stdout: str,
    stderr: str,
    *,
    end_criteria: float,
) -> dict[str, object]:
    """Extract auditable time-stepping and mesh metrics from openEMS logs."""

    result: dict[str, object] = {}
    size = re.search(
        r"FDTD simulation size:\s*(\d+)x(\d+)x(\d+)\s*-->\s*([0-9.eE+-]+)",
        stdout,
    )
    if size:
        result["mesh_shape"] = [int(size.group(i)) for i in range(1, 4)]
        result["mesh_cells"] = int(round(float(size.group(4))))
    timestep = re.search(r"FDTD timestep is:\s*([0-9.eE+-]+)\s*s", stdout)
    if timestep:
        result["timestep_s"] = float(timestep.group(1))
    iterations = re.findall(r"Time for\s+(\d+) iterations", stdout)
    if iterations:
        result["iterations"] = int(iterations[-1])
    energies = re.findall(
        r"Energy:.*?\(\s*([+-]?[0-9]+(?:\.[0-9]+)?)dB\)", stdout
    )
    final_energy_db = None if not energies else float(energies[-1])
    if final_energy_db is not None:
        result["final_energy_db"] = final_energy_db
        result["relative_energy"] = 10.0 ** (final_energy_db / 10.0)
    max_reached = "Max. number of timesteps was reached before" in stderr
    target_db = 10.0 * np.log10(float(end_criteria))
    terminated = bool(
        final_energy_db is not None
        and final_energy_db <= target_db
        and not max_reached
    )
    result.update(
        {
            "termination_verified": terminated,
            "max_timesteps_reached": max_reached,
            "target_energy_db": float(target_db),
        }
    )
    return result


def run_openems(
    project: OpenEMSProject,
    *,
    directory: str | Path | None = None,
    python_executable: str | Path = sys.executable,
    timeout: float = 3600.0,
    setup_only: bool = False,
    normalization: NormalizationKind = "per_ampere",
    save_interchange: bool = True,
) -> OpenEMSRunResult:
    """Generate and run an openEMS project in an external Python environment.

    No shell is used.  The selected interpreter must provide openEMS and
    CSXCAD.  Standard output and error are retained in the returned result and
    written to the project directory.  A successful full run is imported and,
    by default, saved as ``harmonic_solution.npz``.
    """

    if directory is None:
        project_directory = Path(tempfile.mkdtemp(prefix="spin_openems_"))
    else:
        project_directory = Path(directory).resolve()
    project_path, driver_path = write_openems_project(project, project_directory)
    command = [
        str(python_executable),
        str(driver_path),
        str(project_path),
        "--output-dir",
        str(project_directory),
    ]
    if setup_only:
        command.append("--setup-only")
    try:
        completed = subprocess.run(
            command,
            cwd=project_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=float(timeout),
        )
    except subprocess.TimeoutExpired as exc:
        raise OpenEMSExecutionError(
            f"openEMS run exceeded the {float(timeout):g} s timeout in "
            f"{project_directory}"
        ) from exc
    except OSError as exc:
        raise OpenEMSExecutionError(
            f"could not start external openEMS Python {python_executable!s}: {exc}"
        ) from exc
    (project_directory / "openems_stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    (project_directory / "openems_stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        last_line = detail[-1] if detail else "no error output"
        raise OpenEMSExecutionError(
            f"openEMS driver failed with exit code {completed.returncode}: {last_line}. "
            f"See {project_directory / 'openems_stderr.log'}"
        )
    solution = None
    if not setup_only:
        run_path = project_directory / _RUN_METADATA_FILENAME
        run_metadata = _read_json_object(run_path, "openEMS run metadata")
        run_metadata.update(
            _parse_openems_output(
                completed.stdout,
                completed.stderr,
                end_criteria=project.settings.end_criteria,
            )
        )
        minimum_cell, maximum_cell = _openems_mesh_spacing(
            project_directory / "openems_model.xml"
        )
        run_metadata["minimum_cell_m"] = minimum_cell
        run_metadata["maximum_cell_m"] = maximum_cell
        run_path.write_text(
            json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        solution = load_openems_solution(
            project_directory,
            normalization=normalization,
        )
        if save_interchange:
            save_harmonic_em_npz(
                project_directory / "harmonic_solution.npz",
                solution,
            )
    return OpenEMSRunResult(
        project_directory=project_directory,
        command=tuple(command),
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        solution=solution,
    )


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "importing openEMS HDF5 dumps requires h5py; install "
            "python-spin-dynamics[fullwave] or h5py directly"
        ) from exc
    return h5py


def _attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        return item.decode("utf-8") if isinstance(item, bytes) else str(item)
    return str(value)


def _complex_dataset(dataset) -> np.ndarray:
    array = np.asarray(dataset)
    if array.dtype.fields:
        names = set(array.dtype.fields)
        for real_name, imag_name in (("r", "i"), ("real", "imag")):
            if {real_name, imag_name}.issubset(names):
                return np.asarray(
                    array[real_name] + 1.0j * array[imag_name],
                    dtype=np.complex128,
                )
        raise ValueError("unsupported compound complex dtype in openEMS HDF5 dump")
    return np.asarray(array, dtype=np.complex128)


def _openems_vector_order(
    array: np.ndarray,
    order: str,
    shape: tuple[int, int, int],
) -> np.ndarray:
    normalized_order = order.upper()
    if normalized_order == "NXYZ":
        expected = (3,) + shape
        if array.shape != expected:
            raise ValueError(
                f"NXYZ openEMS field must have shape {expected}, got {array.shape}"
            )
        return np.moveaxis(array, 0, -1)
    if normalized_order == "NZYX":
        expected = (3, shape[2], shape[1], shape[0])
        if array.shape != expected:
            raise ValueError(
                f"NZYX openEMS field must have shape {expected}, got {array.shape}"
            )
        return np.transpose(np.moveaxis(array, 0, -1), (2, 1, 0, 3))
    if normalized_order == "XYZN":
        expected = shape + (3,)
        if array.shape != expected:
            raise ValueError(
                f"XYZN openEMS field must have shape {expected}, got {array.shape}"
            )
        return array
    raise ValueError(f"unsupported openEMS vector data order {order!r}")


@dataclass(frozen=True)
class OpenEMSFieldDump:
    """One complex frequency-domain vector field read from openEMS HDF5."""

    domain: SpatialDomain
    frequency_hz: float
    values: np.ndarray
    data_order: str
    legacy_format: bool


def load_openems_field_dump(
    path: str | Path,
    *,
    frequency_hz: float | None = None,
) -> OpenEMSFieldDump:
    """Read a current or legacy Cartesian openEMS frequency-domain HDF5 dump."""

    h5py = _h5py()
    with h5py.File(Path(path), "r") as handle:
        if "Mesh" not in handle or "FieldData/FD" not in handle:
            raise ValueError("openEMS HDF5 requires /Mesh and /FieldData/FD")
        mesh = handle["Mesh"]
        mesh_type = int(np.asarray(mesh.attrs.get("mesh_type", 0)).item())
        if mesh_type != 0:
            raise ValueError("only Cartesian openEMS HDF5 meshes are supported")
        missing_axes = [axis for axis in "xyz" if axis not in mesh]
        if missing_axes:
            raise ValueError(
                f"openEMS HDF5 is missing mesh axes: {', '.join(missing_axes)}"
            )
        axes = tuple(np.asarray(mesh[axis], dtype=np.float64) for axis in "xyz")
        domain = SpatialDomain(axes)
        fd_group = handle["FieldData/FD"]
        if "frequency" not in fd_group.attrs:
            raise ValueError("openEMS HDF5 frequency-domain group has no frequency")
        frequencies = np.atleast_1d(
            np.asarray(fd_group.attrs["frequency"], dtype=np.float64)
        ).reshape(-1)
        if frequencies.size < 1 or not np.all(np.isfinite(frequencies)):
            raise ValueError("openEMS HDF5 contains invalid frequencies")
        if frequency_hz is None:
            if frequencies.size != 1:
                raise ValueError("frequency_hz is required for a multi-frequency dump")
            frequency_index = 0
        else:
            target = _finite_positive(frequency_hz, "frequency_hz")
            frequency_index = int(np.argmin(np.abs(frequencies - target)))
            tolerance = max(1.0e-6 * target, 1.0e-3)
            if abs(frequencies[frequency_index] - target) > tolerance:
                raise ValueError(
                    f"requested frequency {target:g} Hz is not present in openEMS dump"
                )
        dataset_name = f"f{frequency_index}"
        legacy_default = bool(np.asarray(handle.attrs.get("legacy_fmt", False)).item())
        if dataset_name in fd_group:
            dataset = fd_group[dataset_name]
            raw = _complex_dataset(dataset)
            order = _attribute_text(dataset.attrs.get("d_order", "NXYZ"))
            legacy = legacy_default or order.upper() == "NZYX"
        else:
            real_name = dataset_name + "_real"
            imag_name = dataset_name + "_imag"
            if real_name not in fd_group or imag_name not in fd_group:
                raise ValueError(
                    f"openEMS HDF5 is missing field dataset {dataset_name!r}"
                )
            real_dataset = fd_group[real_name]
            raw = np.asarray(real_dataset, dtype=np.float64) + 1.0j * np.asarray(
                fd_group[imag_name],
                dtype=np.float64,
            )
            order = _attribute_text(real_dataset.attrs.get("d_order", "NZYX"))
            legacy = True
        values = np.asarray(
            _openems_vector_order(raw, order, domain.shape),
            dtype=np.complex128,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("openEMS HDF5 field contains non-finite values")
        values.setflags(write=False)
        return OpenEMSFieldDump(
            domain=domain,
            frequency_hz=float(frequencies[frequency_index]),
            values=values,
            data_order=order,
            legacy_format=legacy,
        )


def _complex_from_json(value: object, name: str) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a [real, imaginary] pair")
    result = complex(float(value[0]), float(value[1]))
    if not np.isfinite(result.real) or not np.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _read_json_object(path: Path, name: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {name}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a JSON object")
    return raw


def load_openems_solution(
    directory: str | Path,
    *,
    normalization: NormalizationKind = "per_ampere",
    frequency_hz: float | None = None,
    electric_filename: str = "E_fd.h5",
    magnetic_filename: str = "B_fd.h5",
) -> HarmonicEMSolution:
    """Import and normalize openEMS E/B dumps and port metadata.

    openEMS frequency-domain dumps use the package's canonical
    ``exp(+i*omega*t)`` representation.  Normalization divides both fields by
    the complex total port current or voltage, or by the positive square root
    of accepted power.  Numerical termination is not claimed as convergence;
    the requested limits are retained in provenance for Phase 4 validation.
    """

    if normalization not in {
        "absolute",
        "per_ampere",
        "per_volt",
        "per_sqrt_watt",
    }:
        raise ValueError(f"unsupported normalization {normalization!r}")
    root = Path(directory)
    project = load_openems_project(root / _PROJECT_FILENAME)
    target_frequency = (
        project.settings.frequency_hz if frequency_hz is None else frequency_hz
    )
    electric = load_openems_field_dump(
        root / electric_filename,
        frequency_hz=target_frequency,
    )
    magnetic = load_openems_field_dump(
        root / magnetic_filename,
        frequency_hz=target_frequency,
    )
    if electric.domain.shape != magnetic.domain.shape or any(
        not np.array_equal(left, right)
        for left, right in zip(electric.domain.axes, magnetic.domain.axes)
    ):
        raise ValueError("openEMS E and B dumps use different spatial meshes")
    if electric.frequency_hz != magnetic.frequency_hz:
        raise ValueError("openEMS E and B dumps use different frequencies")

    port_path = root / _PORT_METADATA_FILENAME
    try:
        raw_ports = json.loads(port_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing openEMS port metadata: {port_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("openEMS port metadata is not valid JSON") from exc
    if not isinstance(raw_ports, list) or not raw_ports:
        raise ValueError("openEMS port metadata must be a nonempty list")
    port_objects: list[HarmonicPort] = []
    port_values: dict[int, tuple[complex, complex, float]] = {}
    for raw_port in raw_ports:
        if not isinstance(raw_port, Mapping):
            raise ValueError("each openEMS port metadata entry must be an object")
        number = int(raw_port["number"])
        voltage = _complex_from_json(raw_port["voltage_v"], "voltage_v")
        current = _complex_from_json(raw_port["current_a"], "current_a")
        power = float(raw_port["accepted_power_w"])
        if not np.isfinite(power):
            raise ValueError("accepted_power_w must be finite")
        port_values[number] = (voltage, current, power)
        port_objects.append(
            HarmonicPort(
                index=number,
                name=str(raw_port.get("name", f"port_{number}")),
                voltage_v=voltage,
                current_a=current,
                accepted_power_w=power,
                metadata={
                    "reference_impedance_ohm": raw_port.get(
                        "reference_impedance_ohm"
                    )
                },
            )
        )
    excited = next(port for port in project.ports if port.excite)
    if excited.number not in port_values:
        raise ValueError("openEMS metadata does not contain the excited port")
    voltage, current, accepted_power = port_values[excited.number]
    denominator: complex | float = 1.0
    reference_value = 1.0
    if normalization == "per_ampere":
        denominator = current
        reference_value = abs(current)
    elif normalization == "per_volt":
        denominator = voltage
        reference_value = abs(voltage)
    elif normalization == "per_sqrt_watt":
        if accepted_power <= 0.0:
            raise ValueError(
                "per_sqrt_watt normalization requires positive accepted power"
            )
        denominator = np.sqrt(accepted_power)
        reference_value = float(denominator)
    if abs(denominator) <= np.finfo(np.float64).tiny:
        raise ValueError(f"cannot normalize openEMS fields by zero {normalization}")

    run_metadata = _read_json_object(
        root / _RUN_METADATA_FILENAME,
        "openEMS run metadata",
    )
    if not bool(run_metadata.get("completed", False)):
        raise ValueError("openEMS run metadata does not describe a completed solve")
    materials = tuple(
        HarmonicMaterial(
            name=sample.name,
            relative_permittivity=sample.relative_permittivity,
            conductivity_s_per_m=sample.conductivity_s_per_m,
            relative_permeability=sample.relative_permeability,
            region_id=sample.name,
            metadata=dict(sample.metadata),
        )
        for sample in project.samples
    )
    convergence = None
    if run_metadata.get("final_energy_db") is not None:
        time_terminated = bool(run_metadata.get("termination_verified", False))
        mesh_verified = bool(run_metadata.get("mesh_convergence_verified", False))
        convergence = HarmonicConvergence(
            converged=time_terminated and mesh_verified,
            relative_residual=float(run_metadata["relative_energy"]),
            iterations=(
                None
                if run_metadata.get("iterations") is None
                else int(run_metadata["iterations"])
            ),
            mesh_cells=(
                None
                if run_metadata.get("mesh_cells") is None
                else int(run_metadata["mesh_cells"])
            ),
            minimum_cell_m=(
                None
                if run_metadata.get("minimum_cell_m") is None
                else float(run_metadata["minimum_cell_m"])
            ),
            maximum_cell_m=(
                None
                if run_metadata.get("maximum_cell_m") is None
                else float(run_metadata["maximum_cell_m"])
            ),
            metadata={
                "time_domain_terminated": time_terminated,
                "mesh_convergence_verified": mesh_verified,
                "end_criteria": project.settings.end_criteria,
                "final_energy_db": run_metadata.get("final_energy_db"),
                "target_energy_db": run_metadata.get("target_energy_db"),
                "timestep_s": run_metadata.get("timestep_s"),
                "max_timesteps_reached": run_metadata.get(
                    "max_timesteps_reached", False
                ),
            },
        )
    provenance_metadata = {
        "project_schema": OPENEMS_PROJECT_SCHEMA,
        "max_timesteps": project.settings.max_timesteps,
        "end_criteria": project.settings.end_criteria,
        "termination_verified": bool(
            run_metadata.get("termination_verified", False)
        ),
        "electric_data_order": electric.data_order,
        "magnetic_data_order": magnetic.data_order,
        "legacy_hdf5": electric.legacy_format or magnetic.legacy_format,
        "project_metadata": dict(project.metadata),
    }
    return HarmonicEMSolution(
        domain=electric.domain,
        frequency_hz=electric.frequency_hz,
        phasor_convention="exp(+iwt)",
        electric_field_v_per_m=electric.values / denominator,
        magnetic_flux_density_t=magnetic.values / denominator,
        normalization=HarmonicFieldNormalization(
            kind=normalization,
            reference_value=reference_value,
            port_index=excited.number,
            description="normalized from openEMS total port quantities",
        ),
        ports=tuple(port_objects),
        materials=materials,
        provenance=HarmonicSolverProvenance(
            backend="openEMS",
            backend_version=str(run_metadata.get("backend_version", "unknown")),
            source=str(root.resolve()),
            model_hash=project.model_hash,
            metadata=provenance_metadata,
        ),
        convergence=convergence,
    )


__all__ = [
    "OPENEMS_PROJECT_SCHEMA",
    "OpenEMSAvailability",
    "OpenEMSCylindricalSample",
    "OpenEMSExecutionError",
    "OpenEMSFieldDump",
    "OpenEMSLumpedPort",
    "OpenEMSProject",
    "OpenEMSRunResult",
    "OpenEMSSettings",
    "OpenEMSPlanarConductor",
    "OpenEMSWire",
    "check_openems_python",
    "load_openems_field_dump",
    "load_openems_project",
    "load_openems_solution",
    "loaded_loop_openems_reference",
    "run_openems",
    "segments_to_polylines",
    "unloaded_loop_openems_reference",
    "write_openems_project",
]
