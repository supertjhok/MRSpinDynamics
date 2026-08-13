"""Solver-neutral complex harmonic electromagnetic field interchange.

The types in this module form the boundary between external full-wave solvers
and PythonSpinDynamics.  They deliberately describe sampled fields rather than
solver geometry or meshing APIs, so openEMS, Palace, commercial tools, and
hand-authored reference data can share one validated representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.maps import SpatialFieldMaps


MU_0 = 4.0e-7 * np.pi
HARMONIC_EM_SCHEMA = "python-spin-dynamics.harmonic-em/v1"

PhasorConvention = Literal["exp(+iwt)", "exp(-iwt)"]
NormalizationKind = Literal[
    "absolute",
    "per_ampere",
    "per_volt",
    "per_sqrt_watt",
]


def _finite_optional(value: float | None, name: str, *, positive: bool = False) -> None:
    if value is None:
        return
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite when provided")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive when provided")


def _json_metadata(value: Mapping[str, object], name: str) -> dict[str, object]:
    try:
        encoded = json.dumps(dict(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a mapping")
    return decoded


def _complex_pair(value: complex) -> list[float]:
    number = complex(value)
    if not np.isfinite(number.real) or not np.isfinite(number.imag):
        raise ValueError("complex metadata values must be finite")
    return [float(number.real), float(number.imag)]


def _from_complex_pair(value: object, name: str) -> complex:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a [real, imaginary] pair")
    number = complex(float(value[0]), float(value[1]))
    if not np.isfinite(number.real) or not np.isfinite(number.imag):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class HarmonicFieldNormalization:
    """Describe the excitation normalization applied to stored field arrays.

    ``absolute`` means the arrays are physical SI phasors for the simulated
    drive.  The other kinds mean that they have been divided by port current,
    port voltage, or the square root of accepted power. ``reference_value``
    records the corresponding drive used to form that normalization.
    """

    kind: NormalizationKind = "absolute"
    reference_value: float = 1.0
    port_index: int | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "absolute",
            "per_ampere",
            "per_volt",
            "per_sqrt_watt",
        }:
            raise ValueError(f"unsupported field normalization: {self.kind!r}")
        _finite_optional(self.reference_value, "reference_value", positive=True)
        if self.port_index is not None and self.port_index < 0:
            raise ValueError("port_index must be non-negative")

    @property
    def field_unit_suffix(self) -> str:
        """Return the denominator appended to the base SI field unit."""

        return {
            "absolute": "",
            "per_ampere": "/A",
            "per_volt": "/V",
            "per_sqrt_watt": "/sqrt(W)",
        }[self.kind]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "reference_value": self.reference_value,
            "port_index": self.port_index,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HarmonicFieldNormalization:
        return cls(
            kind=str(value.get("kind", "absolute")),  # type: ignore[arg-type]
            reference_value=float(value.get("reference_value", 1.0)),
            port_index=(
                None
                if value.get("port_index") is None
                else int(value["port_index"])
            ),
            description=(
                None if value.get("description") is None else str(value["description"])
            ),
        )


@dataclass(frozen=True)
class HarmonicPort:
    """Frequency-domain voltage, current, and accepted-power port metadata."""

    index: int
    name: str
    voltage_v: complex | None = None
    current_a: complex | None = None
    accepted_power_w: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("port index must be non-negative")
        if not self.name:
            raise ValueError("port name must not be empty")
        for value, name in (
            (self.voltage_v, "voltage_v"),
            (self.current_a, "current_a"),
        ):
            if value is not None:
                _complex_pair(value)
        _finite_optional(self.accepted_power_w, "accepted_power_w")
        object.__setattr__(self, "metadata", _json_metadata(self.metadata, "port metadata"))

    @property
    def impedance_ohm(self) -> complex | None:
        """Return ``V/I`` when a nonzero current phasor is available."""

        if self.voltage_v is None or self.current_a in (None, 0.0):
            return None
        return complex(self.voltage_v) / complex(self.current_a)

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "voltage_v": None if self.voltage_v is None else _complex_pair(self.voltage_v),
            "current_a": None if self.current_a is None else _complex_pair(self.current_a),
            "accepted_power_w": self.accepted_power_w,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HarmonicPort:
        voltage = value.get("voltage_v")
        current = value.get("current_a")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("port metadata must be a mapping")
        return cls(
            index=int(value["index"]),
            name=str(value["name"]),
            voltage_v=(
                None if voltage is None else _from_complex_pair(voltage, "voltage_v")
            ),
            current_a=(
                None if current is None else _from_complex_pair(current, "current_a")
            ),
            accepted_power_w=(
                None
                if value.get("accepted_power_w") is None
                else float(value["accepted_power_w"])
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class HarmonicMaterial:
    """Solver-neutral material properties and backend region identity."""

    name: str
    relative_permittivity: float
    conductivity_s_per_m: float = 0.0
    relative_permeability: float = 1.0
    region_id: str | int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("material name must not be empty")
        _finite_optional(
            self.relative_permittivity,
            "relative_permittivity",
            positive=True,
        )
        _finite_optional(self.conductivity_s_per_m, "conductivity_s_per_m")
        if self.conductivity_s_per_m < 0.0:
            raise ValueError("conductivity_s_per_m must be non-negative")
        _finite_optional(
            self.relative_permeability,
            "relative_permeability",
            positive=True,
        )
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata, "material metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "relative_permittivity": self.relative_permittivity,
            "conductivity_s_per_m": self.conductivity_s_per_m,
            "relative_permeability": self.relative_permeability,
            "region_id": self.region_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HarmonicMaterial:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("material metadata must be a mapping")
        region_id = value.get("region_id")
        if region_id is not None and not isinstance(region_id, (str, int)):
            raise ValueError("material region_id must be a string, integer, or null")
        return cls(
            name=str(value["name"]),
            relative_permittivity=float(value["relative_permittivity"]),
            conductivity_s_per_m=float(value.get("conductivity_s_per_m", 0.0)),
            relative_permeability=float(value.get("relative_permeability", 1.0)),
            region_id=region_id,
            metadata=metadata,
        )


@dataclass(frozen=True)
class HarmonicSolverProvenance:
    """Identify the backend and model that produced a field solution."""

    backend: str
    backend_version: str | None = None
    source: str | None = None
    model_hash: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must not be empty")
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata, "provenance metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "backend_version": self.backend_version,
            "source": self.source,
            "model_hash": self.model_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HarmonicSolverProvenance:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("provenance metadata must be a mapping")
        return cls(
            backend=str(value["backend"]),
            backend_version=(
                None
                if value.get("backend_version") is None
                else str(value["backend_version"])
            ),
            source=None if value.get("source") is None else str(value["source"]),
            model_hash=(
                None if value.get("model_hash") is None else str(value["model_hash"])
            ),
            metadata=metadata,
        )


@dataclass(frozen=True)
class HarmonicConvergence:
    """Numerical convergence and mesh provenance for a harmonic solve."""

    converged: bool
    relative_residual: float | None = None
    iterations: int | None = None
    mesh_cells: int | None = None
    minimum_cell_m: float | None = None
    maximum_cell_m: float | None = None
    energy_balance_relative_error: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, name in (
            (self.relative_residual, "relative_residual"),
            (self.minimum_cell_m, "minimum_cell_m"),
            (self.maximum_cell_m, "maximum_cell_m"),
            (self.energy_balance_relative_error, "energy_balance_relative_error"),
        ):
            _finite_optional(value, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for value, name in (
            (self.iterations, "iterations"),
            (self.mesh_cells, "mesh_cells"),
        ):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.minimum_cell_m is not None
            and self.maximum_cell_m is not None
            and self.minimum_cell_m > self.maximum_cell_m
        ):
            raise ValueError("minimum_cell_m must not exceed maximum_cell_m")
        object.__setattr__(
            self,
            "metadata",
            _json_metadata(self.metadata, "convergence metadata"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "relative_residual": self.relative_residual,
            "iterations": self.iterations,
            "mesh_cells": self.mesh_cells,
            "minimum_cell_m": self.minimum_cell_m,
            "maximum_cell_m": self.maximum_cell_m,
            "energy_balance_relative_error": self.energy_balance_relative_error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> HarmonicConvergence:
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("convergence metadata must be a mapping")

        def optional_float(name: str) -> float | None:
            item = value.get(name)
            return None if item is None else float(item)

        def optional_int(name: str) -> int | None:
            item = value.get(name)
            return None if item is None else int(item)

        return cls(
            converged=bool(value["converged"]),
            relative_residual=optional_float("relative_residual"),
            iterations=optional_int("iterations"),
            mesh_cells=optional_int("mesh_cells"),
            minimum_cell_m=optional_float("minimum_cell_m"),
            maximum_cell_m=optional_float("maximum_cell_m"),
            energy_balance_relative_error=optional_float(
                "energy_balance_relative_error"
            ),
            metadata=metadata,
        )


def _readonly_array(
    value: np.ndarray | Sequence[complex] | Sequence[float],
    *,
    dtype: np.dtype[np.complex128] | np.dtype[np.float64],
    name: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True, order="C")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HarmonicEMSolution:
    """Complex electromagnetic phasors sampled on a rectilinear spatial grid.

    At least one magnetic field representation must be supplied.  ``B`` is the
    preferred interchange quantity; an H-only solution can still compute B1 if
    ``relative_permeability_map`` is supplied.  All arrays have
    ``domain.shape + (3,)`` and use SI coordinates and field units, modified by
    the declared excitation normalization.
    """

    domain: SpatialDomain
    frequency_hz: float
    phasor_convention: PhasorConvention
    electric_field_v_per_m: np.ndarray
    magnetic_flux_density_t: np.ndarray | None = None
    magnetic_field_a_per_m: np.ndarray | None = None
    relative_permeability_map: np.ndarray | float | None = None
    normalization: HarmonicFieldNormalization = field(
        default_factory=HarmonicFieldNormalization
    )
    ports: tuple[HarmonicPort, ...] = ()
    materials: tuple[HarmonicMaterial, ...] = ()
    provenance: HarmonicSolverProvenance = field(
        default_factory=lambda: HarmonicSolverProvenance("unknown")
    )
    convergence: HarmonicConvergence | None = None

    def __post_init__(self) -> None:
        _finite_optional(self.frequency_hz, "frequency_hz", positive=True)
        if self.phasor_convention not in {"exp(+iwt)", "exp(-iwt)"}:
            raise ValueError(
                "phasor_convention must be 'exp(+iwt)' or 'exp(-iwt)'"
            )
        axes: list[np.ndarray] = []
        for index, axis in enumerate(self.domain.axes):
            axes.append(
                _readonly_array(
                    axis,
                    dtype=np.dtype(np.float64),
                    name=f"domain axis {index}",
                )
            )
        domain = SpatialDomain(tuple(axes))
        vector_shape = domain.shape + (3,)
        electric = _readonly_array(
            self.electric_field_v_per_m,
            dtype=np.dtype(np.complex128),
            name="electric_field_v_per_m",
            shape=vector_shape,
        )
        magnetic_b = None
        if self.magnetic_flux_density_t is not None:
            magnetic_b = _readonly_array(
                self.magnetic_flux_density_t,
                dtype=np.dtype(np.complex128),
                name="magnetic_flux_density_t",
                shape=vector_shape,
            )
        magnetic_h = None
        if self.magnetic_field_a_per_m is not None:
            magnetic_h = _readonly_array(
                self.magnetic_field_a_per_m,
                dtype=np.dtype(np.complex128),
                name="magnetic_field_a_per_m",
                shape=vector_shape,
            )
        if magnetic_b is None and magnetic_h is None:
            raise ValueError(
                "at least one of magnetic_flux_density_t or "
                "magnetic_field_a_per_m must be provided"
            )
        permeability = self.relative_permeability_map
        if permeability is not None:
            permeability = _readonly_array(
                np.broadcast_to(np.asarray(permeability), domain.shape),
                dtype=np.dtype(np.float64),
                name="relative_permeability_map",
                shape=domain.shape,
            )
            if np.any(permeability <= 0.0):
                raise ValueError("relative_permeability_map must be positive")
        if not isinstance(self.normalization, HarmonicFieldNormalization):
            raise TypeError("normalization must be HarmonicFieldNormalization")
        if not isinstance(self.provenance, HarmonicSolverProvenance):
            raise TypeError("provenance must be HarmonicSolverProvenance")
        if self.convergence is not None and not isinstance(
            self.convergence, HarmonicConvergence
        ):
            raise TypeError("convergence must be HarmonicConvergence or None")
        ports = tuple(self.ports)
        if not all(isinstance(port, HarmonicPort) for port in ports):
            raise TypeError("ports must contain HarmonicPort objects")
        if len({port.index for port in ports}) != len(ports):
            raise ValueError("port indices must be unique")
        materials = tuple(self.materials)
        if not all(isinstance(material, HarmonicMaterial) for material in materials):
            raise TypeError("materials must contain HarmonicMaterial objects")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "electric_field_v_per_m", electric)
        object.__setattr__(self, "magnetic_flux_density_t", magnetic_b)
        object.__setattr__(self, "magnetic_field_a_per_m", magnetic_h)
        object.__setattr__(self, "relative_permeability_map", permeability)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "materials", materials)

    @property
    def field_shape(self) -> tuple[int, ...]:
        """Return the common vector-field array shape."""

        return self.domain.shape + (3,)

    def _canonical_phasor(self, field_array: np.ndarray) -> np.ndarray:
        if self.phasor_convention == "exp(+iwt)":
            return field_array
        return np.conjugate(field_array)

    def electric_field(self) -> np.ndarray:
        """Return E using the package's canonical ``exp(+i omega t)`` phasor."""

        return self._canonical_phasor(self.electric_field_v_per_m)

    def magnetic_flux_density(
        self,
        *,
        relative_permeability: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return B using the canonical phasor convention.

        H-only data require either a stored ``relative_permeability_map`` or an
        explicit scalar/map argument.  No silent nonmagnetic-material assumption
        is made at this interchange boundary.
        """

        if self.magnetic_flux_density_t is not None:
            return self._canonical_phasor(self.magnetic_flux_density_t)
        permeability = (
            self.relative_permeability_map
            if relative_permeability is None
            else relative_permeability
        )
        if permeability is None:
            raise ValueError(
                "H-only solutions require relative_permeability to compute B"
            )
        mu_r = np.asarray(permeability, dtype=np.float64)
        try:
            mu_r = np.broadcast_to(mu_r, self.domain.shape)
        except ValueError as exc:
            raise ValueError(
                "relative_permeability must be scalar or match domain.shape"
            ) from exc
        if not np.all(np.isfinite(mu_r)) or np.any(mu_r <= 0.0):
            raise ValueError("relative_permeability must contain positive finite values")
        assert self.magnetic_field_a_per_m is not None
        magnetic_h = self._canonical_phasor(self.magnetic_field_a_per_m)
        return MU_0 * mu_r[..., np.newaxis] * magnetic_h

    def b1_component(
        self,
        b0_direction: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        *,
        handedness: int = 1,
        relative_permeability: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return the complex circular B1 component in the package convention."""

        from spin_dynamics.motion import circular_b1_component

        b0 = np.asarray(b0_direction, dtype=np.float64)
        if b0.shape == (3,):
            b0 = np.broadcast_to(b0, self.field_shape)
        elif b0.shape != self.field_shape:
            raise ValueError("b0_direction must have shape (3,) or domain.shape + (3,)")
        return circular_b1_component(
            b0,
            self.magnetic_flux_density(
                relative_permeability=relative_permeability
            ),
            handedness=handedness,
        )

    def b1_plus(
        self,
        b0_direction: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        *,
        relative_permeability: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return complex transmit B1+ in the canonical phasor convention."""

        return self.b1_component(
            b0_direction,
            handedness=1,
            relative_permeability=relative_permeability,
        )

    def b1_minus(
        self,
        b0_direction: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        *,
        relative_permeability: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Return complex reciprocal receive B1- in the canonical convention."""

        return self.b1_component(
            b0_direction,
            handedness=-1,
            relative_permeability=relative_permeability,
        )

    def to_spatial_field_maps(
        self,
        *,
        rho: np.ndarray | float,
        t1_map: np.ndarray | float,
        t2_map: np.ndarray | float,
        b0_map: np.ndarray | float = 0.0,
        diffusion_map: np.ndarray | float | None = None,
        gradient_sensitivity: tuple[np.ndarray, ...] | None = None,
        b0_direction: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
        tx_scale: float = 1.0,
        rx_scale: float = 1.0,
        relative_permeability: np.ndarray | float | None = None,
    ) -> SpatialFieldMaps:
        """Adapt B1 magnitudes into the existing scalar spatial-map bundle.

        ``tx_scale`` and ``rx_scale`` convert the stored normalization to the
        drive convention expected by a simulation.  Complex B1 phase remains
        available through :meth:`b1_plus` and :meth:`b1_minus`; this scalar
        adapter intentionally follows :class:`SpatialFieldMaps`' nonnegative
        sensitivity convention.
        """

        _finite_optional(tx_scale, "tx_scale")
        _finite_optional(rx_scale, "rx_scale")
        if tx_scale < 0.0 or rx_scale < 0.0:
            raise ValueError("tx_scale and rx_scale must be non-negative")

        def scalar_map(value: np.ndarray | float, name: str) -> np.ndarray:
            try:
                result = np.array(
                    np.broadcast_to(np.asarray(value, dtype=np.float64), self.domain.shape),
                    copy=True,
                )
            except ValueError as exc:
                raise ValueError(f"{name} must be scalar or match domain.shape") from exc
            if not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must contain finite values")
            return result

        diffusion = None
        if diffusion_map is not None:
            diffusion = scalar_map(diffusion_map, "diffusion_map")
        return SpatialFieldMaps(
            domain=self.domain,
            rho=scalar_map(rho, "rho"),
            t1_map=scalar_map(t1_map, "t1_map"),
            t2_map=scalar_map(t2_map, "t2_map"),
            b0_map=scalar_map(b0_map, "b0_map"),
            b1_tx_map=tx_scale
            * np.abs(
                self.b1_plus(
                    b0_direction,
                    relative_permeability=relative_permeability,
                )
            ),
            b1_rx_map=rx_scale
            * np.abs(
                self.b1_minus(
                    b0_direction,
                    relative_permeability=relative_permeability,
                )
            ),
            diffusion_map=diffusion,
            gradient_sensitivity=gradient_sensitivity,
        )


def _metadata_dict(solution: HarmonicEMSolution) -> dict[str, object]:
    return {
        "normalization": solution.normalization.to_dict(),
        "ports": [port.to_dict() for port in solution.ports],
        "materials": [material.to_dict() for material in solution.materials],
        "provenance": solution.provenance.to_dict(),
        "convergence": (
            None if solution.convergence is None else solution.convergence.to_dict()
        ),
    }


def _metadata_objects(
    text: str,
) -> tuple[
    HarmonicFieldNormalization,
    tuple[HarmonicPort, ...],
    tuple[HarmonicMaterial, ...],
    HarmonicSolverProvenance,
    HarmonicConvergence | None,
]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("harmonic EM metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("harmonic EM metadata must be a JSON object")
    normalization_raw = raw.get("normalization", {})
    provenance_raw = raw.get("provenance", {"backend": "unknown"})
    ports_raw = raw.get("ports", [])
    materials_raw = raw.get("materials", [])
    convergence_raw = raw.get("convergence")
    if not isinstance(normalization_raw, Mapping):
        raise ValueError("normalization metadata must be an object")
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("provenance metadata must be an object")
    if not isinstance(ports_raw, list) or not all(
        isinstance(item, Mapping) for item in ports_raw
    ):
        raise ValueError("ports metadata must be a list of objects")
    if not isinstance(materials_raw, list) or not all(
        isinstance(item, Mapping) for item in materials_raw
    ):
        raise ValueError("materials metadata must be a list of objects")
    if convergence_raw is not None and not isinstance(convergence_raw, Mapping):
        raise ValueError("convergence metadata must be an object or null")
    return (
        HarmonicFieldNormalization.from_dict(normalization_raw),
        tuple(HarmonicPort.from_dict(item) for item in ports_raw),
        tuple(HarmonicMaterial.from_dict(item) for item in materials_raw),
        HarmonicSolverProvenance.from_dict(provenance_raw),
        (
            None
            if convergence_raw is None
            else HarmonicConvergence.from_dict(convergence_raw)
        ),
    )


def _text_scalar(value: object, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    item = array.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def save_harmonic_em_npz(
    path: str | Path,
    solution: HarmonicEMSolution,
) -> None:
    """Write the versioned harmonic-field interchange schema to NPZ."""

    arrays: dict[str, object] = {
        "schema": np.asarray(HARMONIC_EM_SCHEMA),
        "frequency_hz": np.asarray(solution.frequency_hz),
        "phasor_convention": np.asarray(solution.phasor_convention),
        "axis_count": np.asarray(solution.domain.ndim, dtype=np.int64),
        "electric_field_v_per_m": solution.electric_field_v_per_m,
        "metadata_json": np.asarray(
            json.dumps(_metadata_dict(solution), allow_nan=False, sort_keys=True)
        ),
    }
    for index, axis in enumerate(solution.domain.axes):
        arrays[f"axis_{index}_m"] = axis
    if solution.magnetic_flux_density_t is not None:
        arrays["magnetic_flux_density_t"] = solution.magnetic_flux_density_t
    if solution.magnetic_field_a_per_m is not None:
        arrays["magnetic_field_a_per_m"] = solution.magnetic_field_a_per_m
    if solution.relative_permeability_map is not None:
        arrays["relative_permeability_map"] = solution.relative_permeability_map
    np.savez_compressed(Path(path), **arrays)


def load_harmonic_em_npz(path: str | Path) -> HarmonicEMSolution:
    """Load and validate a versioned harmonic-field NPZ archive."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "schema",
            "frequency_hz",
            "phasor_convention",
            "axis_count",
            "electric_field_v_per_m",
            "metadata_json",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"harmonic EM NPZ is missing keys: {', '.join(missing)}")
        schema = _text_scalar(archive["schema"], "schema")
        if schema != HARMONIC_EM_SCHEMA:
            raise ValueError(
                f"unsupported harmonic EM schema {schema!r}; "
                f"expected {HARMONIC_EM_SCHEMA!r}"
            )
        axis_count_array = np.asarray(archive["axis_count"])
        if axis_count_array.shape != ():
            raise ValueError("axis_count must be a scalar")
        axis_count = int(axis_count_array.item())
        if not 1 <= axis_count <= 3:
            raise ValueError("axis_count must be between 1 and 3")
        axis_keys = [f"axis_{index}_m" for index in range(axis_count)]
        missing_axes = [key for key in axis_keys if key not in archive.files]
        if missing_axes:
            raise ValueError(
                f"harmonic EM NPZ is missing axes: {', '.join(missing_axes)}"
            )
        normalization, ports, materials, provenance, convergence = _metadata_objects(
            _text_scalar(archive["metadata_json"], "metadata_json")
        )

        def optional(name: str) -> np.ndarray | None:
            return archive[name] if name in archive.files else None
        return HarmonicEMSolution(
            domain=SpatialDomain(tuple(archive[key] for key in axis_keys)),
            frequency_hz=float(np.asarray(archive["frequency_hz"]).item()),
            phasor_convention=_text_scalar(
                archive["phasor_convention"], "phasor_convention"
            ),  # type: ignore[arg-type]
            electric_field_v_per_m=archive["electric_field_v_per_m"],
            magnetic_flux_density_t=optional("magnetic_flux_density_t"),
            magnetic_field_a_per_m=optional("magnetic_field_a_per_m"),
            relative_permeability_map=optional("relative_permeability_map"),
            normalization=normalization,
            ports=ports,
            materials=materials,
            provenance=provenance,
            convergence=convergence,
        )


def _h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "HDF5 harmonic-field interchange requires h5py; install "
            "python-spin-dynamics[fullwave] or h5py directly"
        ) from exc
    return h5py


def save_harmonic_em_hdf5(
    path: str | Path,
    solution: HarmonicEMSolution,
) -> None:
    """Write the versioned harmonic-field interchange schema to HDF5."""

    h5py = _h5py()
    with h5py.File(Path(path), "w") as handle:
        handle.attrs["schema"] = HARMONIC_EM_SCHEMA
        handle.attrs["frequency_hz"] = solution.frequency_hz
        handle.attrs["phasor_convention"] = solution.phasor_convention
        handle.attrs["metadata_json"] = json.dumps(
            _metadata_dict(solution), allow_nan=False, sort_keys=True
        )
        axes_group = handle.create_group("coordinates")
        for index, axis in enumerate(solution.domain.axes):
            axes_group.create_dataset(f"axis_{index}_m", data=axis)
        fields_group = handle.create_group("fields")
        fields_group.create_dataset(
            "electric_field_v_per_m", data=solution.electric_field_v_per_m
        )
        if solution.magnetic_flux_density_t is not None:
            fields_group.create_dataset(
                "magnetic_flux_density_t", data=solution.magnetic_flux_density_t
            )
        if solution.magnetic_field_a_per_m is not None:
            fields_group.create_dataset(
                "magnetic_field_a_per_m", data=solution.magnetic_field_a_per_m
            )
        if solution.relative_permeability_map is not None:
            fields_group.create_dataset(
                "relative_permeability_map",
                data=solution.relative_permeability_map,
            )


def load_harmonic_em_hdf5(path: str | Path) -> HarmonicEMSolution:
    """Load and validate a versioned harmonic-field HDF5 file."""

    h5py = _h5py()
    with h5py.File(Path(path), "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        if schema != HARMONIC_EM_SCHEMA:
            raise ValueError(
                f"unsupported harmonic EM schema {schema!r}; "
                f"expected {HARMONIC_EM_SCHEMA!r}"
            )
        for key in ("frequency_hz", "phasor_convention", "metadata_json"):
            if key not in handle.attrs:
                raise ValueError(f"harmonic EM HDF5 is missing attribute {key!r}")
        if "coordinates" not in handle or "fields" not in handle:
            raise ValueError("harmonic EM HDF5 requires coordinates and fields groups")
        axes_group = handle["coordinates"]
        axis_keys = sorted(
            (key for key in axes_group.keys() if key.startswith("axis_") and key.endswith("_m")),
            key=lambda key: int(key.split("_")[1]),
        )
        if not 1 <= len(axis_keys) <= 3:
            raise ValueError("harmonic EM HDF5 must contain one to three axes")
        fields_group = handle["fields"]
        if "electric_field_v_per_m" not in fields_group:
            raise ValueError("harmonic EM HDF5 is missing electric_field_v_per_m")
        normalization, ports, materials, provenance, convergence = _metadata_objects(
            str(handle.attrs["metadata_json"])
        )

        def optional(name: str) -> np.ndarray | None:
            return None if name not in fields_group else np.asarray(fields_group[name])

        return HarmonicEMSolution(
            domain=SpatialDomain(tuple(np.asarray(axes_group[key]) for key in axis_keys)),
            frequency_hz=float(handle.attrs["frequency_hz"]),
            phasor_convention=str(handle.attrs["phasor_convention"]),  # type: ignore[arg-type]
            electric_field_v_per_m=np.asarray(
                fields_group["electric_field_v_per_m"]
            ),
            magnetic_flux_density_t=optional("magnetic_flux_density_t"),
            magnetic_field_a_per_m=optional("magnetic_field_a_per_m"),
            relative_permeability_map=optional("relative_permeability_map"),
            normalization=normalization,
            ports=ports,
            materials=materials,
            provenance=provenance,
            convergence=convergence,
        )


def save_harmonic_em(path: str | Path, solution: HarmonicEMSolution) -> None:
    """Save NPZ or HDF5 interchange data based on the file extension."""

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        save_harmonic_em_npz(path, solution)
    elif suffix in {".h5", ".hdf5"}:
        save_harmonic_em_hdf5(path, solution)
    else:
        raise ValueError("harmonic EM files must use .npz, .h5, or .hdf5")


def load_harmonic_em(path: str | Path) -> HarmonicEMSolution:
    """Load NPZ or HDF5 interchange data based on the file extension."""

    suffix = Path(path).suffix.lower()
    if suffix == ".npz":
        return load_harmonic_em_npz(path)
    if suffix in {".h5", ".hdf5"}:
        return load_harmonic_em_hdf5(path)
    raise ValueError("harmonic EM files must use .npz, .h5, or .hdf5")


__all__ = [
    "HARMONIC_EM_SCHEMA",
    "MU_0",
    "HarmonicConvergence",
    "HarmonicEMSolution",
    "HarmonicFieldNormalization",
    "HarmonicMaterial",
    "HarmonicPort",
    "HarmonicSolverProvenance",
    "load_harmonic_em",
    "load_harmonic_em_hdf5",
    "load_harmonic_em_npz",
    "save_harmonic_em",
    "save_harmonic_em_hdf5",
    "save_harmonic_em_npz",
]
