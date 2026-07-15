"""Electropermanent AlNiCo rods, bundles, and static B0 field maps.

The first electropermanent-magnet (EPM) implementation deliberately separates
four concerns that are often conflated: material data, retained magnetic state,
physical geometry, and programming-coil metadata.  This module handles the
static part of that model.  Pulse-power dynamics and history-dependent state
updates build on these records in a later phase.

Fields outside the magnetic material are evaluated by volume cubature of point
dipoles.  Rod axes may have any 3-D orientation.  An exact on-axis expression
is also provided for convergence tests and fast axial profiles.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, MU0

EvidenceClass = Literal["measured", "simulated", "specified", "inferred"]
RemanenceBranch = Literal[
    "unknown",
    "positive_saturation",
    "negative_saturation",
    "partial",
]


@dataclass(frozen=True)
class EvidenceRecord:
    """Traceable provenance for one model value or preset.

    ``classification`` is intentionally restricted to the four evidence tags
    used by the EPM design plan.  ``source`` should identify a publication,
    archived filename, drawing, or dataset without depending on an absolute
    path on one workstation.
    """

    source: str
    classification: EvidenceClass
    detail: str = ""
    revision: str | None = None

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise ValueError("source must not be empty")
        if self.classification not in {
            "measured",
            "simulated",
            "specified",
            "inferred",
        }:
            raise ValueError(
                "classification must be measured, simulated, specified, or inferred"
            )


@dataclass(frozen=True)
class AlNiCoMaterial:
    """AlNiCo material properties used by EPM field and state models.

    ``remanence_t`` and ``coercivity_a_per_m`` describe the nominal material,
    not necessarily the retained operating point of a finite rod.  That point
    belongs to :class:`RemanenceState`.  ``bh_curve`` stores ``(B, H)`` pairs in
    tesla and A/m.  It is a field-solver curve, not a hysteresis history model.
    """

    name: str
    remanence_t: float
    coercivity_a_per_m: float
    recoil_relative_permeability: float
    conductivity_s_per_m: float | None = None
    density_kg_per_m3: float | None = None
    bh_curve: tuple[tuple[float, float], ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        for name in ("remanence_t", "coercivity_a_per_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not np.isfinite(self.recoil_relative_permeability)
            or self.recoil_relative_permeability < 1.0
        ):
            raise ValueError(
                "recoil_relative_permeability must be finite and at least one"
            )
        for name in ("conductivity_s_per_m", "density_kg_per_m3"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when supplied")
        curve = tuple((float(b), float(h)) for b, h in self.bh_curve)
        if any(not np.isfinite(b) or not np.isfinite(h) for b, h in curve):
            raise ValueError("bh_curve must contain finite (B, H) pairs")
        if any(
            curve[index + 1][0] < curve[index][0]
            or curve[index + 1][1] < curve[index][1]
            for index in range(len(curve) - 1)
        ):
            raise ValueError("bh_curve B and H coordinates must be non-decreasing")
        object.__setattr__(self, "bh_curve", curve)
        object.__setattr__(self, "evidence", tuple(self.evidence))


_AC500_EVIDENCE = EvidenceRecord(
    source="Weinberg archive: Magnet Coil Calculator.xlsx",
    classification="specified",
    detail="AC500 design inputs Br=12.7 kG and Hc=0.64 kOe",
)

ALNICO5_AC500 = AlNiCoMaterial(
    name="AlNiCo-5 AC500",
    remanence_t=1.27,
    coercivity_a_per_m=0.64 * 1.0e3 * 1.0e3 / (4.0 * np.pi),
    recoil_relative_permeability=1.5,
    conductivity_s_per_m=2.25e6,
    density_kg_per_m3=7300.0,
    evidence=(_AC500_EVIDENCE,),
)

_FEMM_ALNICO5_CURVE = (
    (0.0, 0.0),
    (0.12561, 253.852),
    (0.30829, 707.444),
    (0.48334, 1452.289),
    (0.61734, 2162.916),
    (0.75499, 3024.740),
    (0.84413, 3845.979),
    (0.92928, 5110.465),
    (0.98077, 6495.909),
    (1.02467, 8023.001),
    (1.04640, 9234.170),
    (1.07150, 11043.762),
    (1.09290, 12850.171),
    (1.10680, 14650.213),
    (1.12420, 16900.663),
    (1.13790, 19147.135),
    (1.15170, 21245.593),
    (1.16520, 23790.481),
    (1.17500, 26182.580),
    (1.19150, 30071.531),
    (1.20460, 33361.263),
    (1.21770, 36650.996),
    (1.22730, 39490.320),
    (1.23970, 43971.487),
    (1.24880, 47704.784),
    (1.25440, 50988.028),
)

ALNICO5_FEMM_2019 = AlNiCoMaterial(
    name="AlNiCo-5 FEMM 2019",
    remanence_t=1.27,
    coercivity_a_per_m=50963.0,
    recoil_relative_permeability=1.5,
    conductivity_s_per_m=2.25e6,
    density_kg_per_m3=7300.0,
    bh_curve=_FEMM_ALNICO5_CURVE,
    evidence=(
        EvidenceRecord(
            source="Weinberg archive: gap_magnet_2_AlNiCo.FEM",
            classification="simulated",
            detail="AlNiCo-5 material record and 26-point FEMM B-H table",
            revision="2019",
        ),
    ),
)


@dataclass(frozen=True)
class RemanenceState:
    """Retained axial remanence and history metadata for one EPM element.

    ``remanence_t`` is signed along the element's geometric axis.  It is kept
    separate from nominal material remanence because demagnetizing fields and
    partial programming can leave a much smaller effective operating value.
    """

    remanence_t: float
    branch: RemanenceBranch = "unknown"
    reversal_fields_a_per_m: tuple[float, ...] = ()
    temperature_k: float = 293.15
    calibration_id: str | None = None
    uncertainty_t: float | None = None
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not np.isfinite(self.remanence_t):
            raise ValueError("remanence_t must be finite")
        if self.branch not in {
            "unknown",
            "positive_saturation",
            "negative_saturation",
            "partial",
        }:
            raise ValueError("invalid remanence branch")
        reversal = tuple(float(value) for value in self.reversal_fields_a_per_m)
        if any(not np.isfinite(value) for value in reversal):
            raise ValueError("reversal_fields_a_per_m must be finite")
        if not np.isfinite(self.temperature_k) or self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be finite and positive")
        if self.uncertainty_t is not None and (
            not np.isfinite(self.uncertainty_t) or self.uncertainty_t < 0.0
        ):
            raise ValueError("uncertainty_t must be finite and non-negative")
        object.__setattr__(self, "reversal_fields_a_per_m", reversal)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def fraction_of(self, material: AlNiCoMaterial) -> float:
        """Return signed retained remanence divided by nominal material Br."""

        return float(self.remanence_t / material.remanence_t)


UNPROGRAMMED_STATE = RemanenceState(
    0.0,
    evidence=(
        EvidenceRecord(
            source="PythonSpinDynamics EPM preset",
            classification="inferred",
            detail="Zero is a safe placeholder when no retained-field calibration exists",
        ),
    ),
)


@dataclass(frozen=True)
class ProgrammingCoil:
    """Geometry-independent metadata for an EPM programming winding."""

    turns: int
    wire_gauge_awg: int | None = None
    inductance_h: float | None = None
    resistance_ohm: float | None = None
    winding_length_m: float | None = None
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if int(self.turns) != self.turns or self.turns <= 0:
            raise ValueError("turns must be a positive integer")
        if self.wire_gauge_awg is not None and (
            int(self.wire_gauge_awg) != self.wire_gauge_awg
            or self.wire_gauge_awg < 0
        ):
            raise ValueError("wire_gauge_awg must be a non-negative integer")
        for name in ("inductance_h", "resistance_ohm", "winding_length_m"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive when supplied")
        object.__setattr__(self, "evidence", tuple(self.evidence))


def _unit_vector(values: Sequence[float], name: str) -> tuple[float, float, float]:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return tuple(float(value) for value in vector / norm)


@dataclass(frozen=True)
class ElectropermanentRod:
    """A finite cylindrical AlNiCo rod with a retained axial state."""

    center_m: tuple[float, float, float]
    axis: tuple[float, float, float]
    length_m: float
    radius_m: float
    material: AlNiCoMaterial
    state: RemanenceState
    coil: ProgrammingCoil | None = None
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_m must be a finite 3-vector")
        object.__setattr__(self, "center_m", tuple(float(value) for value in center))
        object.__setattr__(self, "axis", _unit_vector(self.axis, "axis"))
        for name in ("length_m", "radius_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if abs(self.state.remanence_t) > self.material.remanence_t * (1.0 + 1e-12):
            raise ValueError("state remanence exceeds nominal material remanence")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def volume_m3(self) -> float:
        """Rod volume in cubic meters."""

        return float(np.pi * self.radius_m**2 * self.length_m)

    @property
    def remanence_vector_t(self) -> np.ndarray:
        """Retained remanence vector in tesla."""

        return self.state.remanence_t * np.asarray(self.axis)

    @property
    def magnetic_moment_am2(self) -> np.ndarray:
        """Uniform-magnetization dipole moment in A m^2."""

        return self.remanence_vector_t * self.volume_m3 / MU0

    def with_state(self, state: RemanenceState) -> ElectropermanentRod:
        """Return the same physical rod with a different retained state."""

        return replace(self, state=state)

    def field_vectors(
        self,
        points: np.ndarray,
        *,
        n_cross: int = 7,
        n_length: int = 31,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Return the external B field at arbitrary 3-D points."""

        return electropermanent_field(
            points,
            (self,),
            n_cross=n_cross,
            n_length=n_length,
            chunk_size=chunk_size,
        )


@dataclass(frozen=True)
class ElectropermanentBundle:
    """A collection of individually represented EPM rods and one bundle coil."""

    rods: tuple[ElectropermanentRod, ...]
    coil: ProgrammingCoil | None = None
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        rods = tuple(self.rods)
        if not rods:
            raise ValueError("an electropermanent bundle must contain at least one rod")
        object.__setattr__(self, "rods", rods)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def volume_m3(self) -> float:
        """Total AlNiCo volume in cubic meters."""

        return float(sum(rod.volume_m3 for rod in self.rods))

    @property
    def magnetic_moment_am2(self) -> np.ndarray:
        """Vector sum of all retained rod moments in A m^2."""

        return np.sum([rod.magnetic_moment_am2 for rod in self.rods], axis=0)

    @property
    def center_m(self) -> np.ndarray:
        """Volume-weighted bundle center."""

        volumes = np.asarray([rod.volume_m3 for rod in self.rods])
        centers = np.asarray([rod.center_m for rod in self.rods])
        return np.average(centers, axis=0, weights=volumes)

    @property
    def axis(self) -> np.ndarray:
        """Common directed bundle axis, raising if rods are not aligned."""

        axis = np.asarray(self.rods[0].axis)
        if any(float(np.dot(axis, rod.axis)) < 1.0 - 1e-10 for rod in self.rods):
            raise ValueError("bundle rods do not have a common directed axis")
        return axis

    @property
    def outer_radius_m(self) -> float:
        """Maximum transverse center offset plus rod radius."""

        axis = self.axis
        center = self.center_m
        radii = []
        for rod in self.rods:
            offset = np.asarray(rod.center_m) - center
            transverse = offset - np.dot(offset, axis) * axis
            radii.append(float(np.linalg.norm(transverse) + rod.radius_m))
        return max(radii)

    def with_state(self, state: RemanenceState) -> ElectropermanentBundle:
        """Return a bundle whose rods all share ``state``."""

        return replace(self, rods=tuple(rod.with_state(state) for rod in self.rods))

    def equivalent_cylinder(self, *, label: str | None = None) -> ElectropermanentRod:
        """Return an area-equivalent rod preserving volume and dipole moment.

        This fast approximation requires parallel rods of equal length and the
        same material.  It does not preserve near-field detail at the bundle
        face.
        """

        first = self.rods[0]
        axis = self.axis
        if any(abs(rod.length_m - first.length_m) > 1e-12 for rod in self.rods):
            raise ValueError("equivalent cylinder requires equal rod lengths")
        if any(rod.material != first.material for rod in self.rods):
            raise ValueError("equivalent cylinder requires one material")
        retained = sum(
            rod.state.remanence_t * rod.volume_m3 for rod in self.rods
        ) / self.volume_m3
        state = RemanenceState(
            retained,
            branch="partial",
            temperature_k=float(
                np.average(
                    [rod.state.temperature_k for rod in self.rods],
                    weights=[rod.volume_m3 for rod in self.rods],
                )
            ),
            calibration_id="equivalent-cylinder",
            evidence=(
                EvidenceRecord(
                    source="PythonSpinDynamics equivalent-cylinder reduction",
                    classification="inferred",
                    detail="Preserves total material volume and magnetic moment",
                ),
            ),
        )
        radius = np.sqrt(self.volume_m3 / (np.pi * first.length_m))
        return ElectropermanentRod(
            center_m=tuple(self.center_m),
            axis=tuple(axis),
            length_m=first.length_m,
            radius_m=float(radius),
            material=first.material,
            state=state,
            coil=self.coil,
            label=self.label if label is None else label,
            evidence=self.evidence,
        )

    def field_vectors(
        self,
        points: np.ndarray,
        *,
        n_cross: int = 7,
        n_length: int = 31,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Return the superposed external field of every rod."""

        return electropermanent_field(
            points,
            (self,),
            n_cross=n_cross,
            n_length=n_length,
            chunk_size=chunk_size,
        )


ElectropermanentSource = ElectropermanentRod | ElectropermanentBundle


def _transverse_frame(axis: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(axis, dtype=np.float64)
    reference = (
        np.asarray([0.0, 0.0, 1.0])
        if abs(direction[2]) < 0.9
        else np.asarray([1.0, 0.0, 0.0])
    )
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def _rod_dipoles(
    rod: ElectropermanentRod,
    n_cross: int,
    n_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_cross = int(n_cross)
    n_length = int(n_length)
    if n_cross < 1 or n_length < 1:
        raise ValueError("n_cross and n_length must be at least one")
    coords = (
        (np.arange(n_cross, dtype=np.float64) + 0.5) / n_cross - 0.5
    ) * (2.0 * rod.radius_m)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    mask = xx**2 + yy**2 <= rod.radius_m**2
    transverse_x = xx[mask]
    transverse_y = yy[mask]
    axial = (
        (np.arange(n_length, dtype=np.float64) + 0.5) / n_length - 0.5
    ) * rod.length_m
    tx, zz = np.meshgrid(transverse_x, axial, indexing="ij")
    ty, _ = np.meshgrid(transverse_y, axial, indexing="ij")
    first, second = _transverse_frame(rod.axis)
    offsets = (
        tx.ravel()[:, None] * first
        + ty.ravel()[:, None] * second
        + zz.ravel()[:, None] * np.asarray(rod.axis)
    )
    positions = offsets + np.asarray(rod.center_m)
    moment = rod.magnetic_moment_am2 / positions.shape[0]
    moments = np.broadcast_to(moment, positions.shape).copy()
    return positions, moments


def _flatten_sources(
    sources: Sequence[ElectropermanentSource],
) -> tuple[ElectropermanentRod, ...]:
    rods: list[ElectropermanentRod] = []
    for source in sources:
        if isinstance(source, ElectropermanentRod):
            rods.append(source)
        elif isinstance(source, ElectropermanentBundle):
            rods.extend(source.rods)
        else:
            raise TypeError("sources must contain electropermanent rods or bundles")
    return tuple(rods)


def electropermanent_field(
    points: np.ndarray,
    sources: Sequence[ElectropermanentSource],
    *,
    n_cross: int = 7,
    n_length: int = 31,
    chunk_size: int = 4096,
) -> np.ndarray:
    """Return the external B field of EPM rods or bundles in tesla.

    ``points`` has shape ``(..., 3)`` in meters.  The dipole cubature is meant
    for points outside the magnetic material; use convergence checks near a rod
    face.  Zero-remanence rods are skipped.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim < 1 or pts.shape[-1] != 3 or not np.all(np.isfinite(pts)):
        raise ValueError("points must be finite with shape (..., 3)")
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")
    rods = tuple(rod for rod in _flatten_sources(sources) if rod.state.remanence_t)
    if not rods:
        return np.zeros_like(pts)
    source_positions: list[np.ndarray] = []
    source_moments: list[np.ndarray] = []
    for rod in rods:
        positions, moments = _rod_dipoles(rod, n_cross, n_length)
        source_positions.append(positions)
        source_moments.append(moments)
    positions = np.concatenate(source_positions, axis=0)
    moments = np.concatenate(source_moments, axis=0)
    flat = pts.reshape(-1, 3)
    result = np.zeros_like(flat)
    coefficient = MU0 / (4.0 * np.pi)
    for start in range(0, flat.shape[0], chunk_size):
        stop = min(start + chunk_size, flat.shape[0])
        displacement = flat[start:stop, None, :] - positions[None, :, :]
        radius_squared = np.sum(displacement**2, axis=-1)
        moment_dot_radius = np.sum(moments[None, :, :] * displacement, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_radius_cubed = np.where(
                radius_squared > 0.0,
                radius_squared**-1.5,
                0.0,
            )
            inverse_radius_fifth = np.where(
                radius_squared > 0.0,
                inverse_radius_cubed / radius_squared,
                0.0,
            )
        field = (
            3.0
            * displacement
            * moment_dot_radius[..., None]
            * inverse_radius_fifth[..., None]
            - moments[None, :, :] * inverse_radius_cubed[..., None]
        )
        result[start:stop] = coefficient * np.sum(field, axis=1)
    return result.reshape(pts.shape)


def finite_cylinder_on_axis_field(
    axial_position_m: float | np.ndarray,
    rod: ElectropermanentRod,
) -> float | np.ndarray:
    """Exact axial field component of a uniformly magnetized finite cylinder.

    ``axial_position_m`` is measured from the rod center along its axis.  The
    returned sign follows the rod's signed retained remanence.
    """

    position = np.asarray(axial_position_m, dtype=np.float64)
    if not np.all(np.isfinite(position)):
        raise ValueError("axial_position_m must be finite")
    half_length = 0.5 * rod.length_m
    upper = position + half_length
    lower = position - half_length
    factor = 0.5 * (
        upper / np.sqrt(upper**2 + rod.radius_m**2)
        - lower / np.sqrt(lower**2 + rod.radius_m**2)
    )
    field = rod.state.remanence_t * factor
    return float(field) if field.ndim == 0 else field


def hexagonal_rod_offsets(
    count: int,
    pitch_m: float,
    *,
    axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Return ``count`` deterministic hexagonal-lattice offsets in 3-D."""

    count = int(count)
    if count < 1:
        raise ValueError("count must be at least one")
    if not np.isfinite(pitch_m) or pitch_m <= 0.0:
        raise ValueError("pitch_m must be finite and positive")
    direction = np.asarray(_unit_vector(axis, "axis"))
    first, second = _transverse_frame(direction)
    ring = 0
    candidates: list[tuple[int, float, float]] = []
    while len(candidates) < count:
        candidates = []
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                distance = max(abs(q), abs(r), abs(q + r))
                if distance <= ring:
                    x = pitch_m * (q + 0.5 * r)
                    y = pitch_m * (np.sqrt(3.0) / 2.0) * r
                    candidates.append((distance, x, y))
        ring += 1
    candidates.sort(key=lambda value: (value[0], np.arctan2(value[2], value[1])))
    selected = candidates[:count]
    return np.asarray([x * first + y * second for _, x, y in selected])


def close_packed_rod_bundle(
    count: int,
    *,
    rod_radius_m: float,
    rod_length_m: float,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    gap_m: float = 0.0,
    material: AlNiCoMaterial = ALNICO5_AC500,
    state: RemanenceState = UNPROGRAMMED_STATE,
    coil: ProgrammingCoil | None = None,
    label: str = "",
    evidence: Sequence[EvidenceRecord] = (),
) -> ElectropermanentBundle:
    """Build a close-packed bundle of equal parallel cylindrical rods."""

    if not np.isfinite(gap_m) or gap_m < 0.0:
        raise ValueError("gap_m must be finite and non-negative")
    center = np.asarray(center_m, dtype=np.float64)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("center_m must be a finite 3-vector")
    direction = _unit_vector(axis, "axis")
    offsets = hexagonal_rod_offsets(
        count,
        2.0 * float(rod_radius_m) + float(gap_m),
        axis=direction,
    )
    source_evidence = tuple(evidence)
    rods = tuple(
        ElectropermanentRod(
            center_m=tuple(center + offset),
            axis=direction,
            length_m=rod_length_m,
            radius_m=rod_radius_m,
            material=material,
            state=state,
            label=f"{label} rod {index + 1}".strip(),
            evidence=source_evidence,
        )
        for index, offset in enumerate(offsets)
    )
    return ElectropermanentBundle(
        rods=rods,
        coil=coil,
        label=label,
        evidence=source_evidence,
    )


def variable_field_nmr_rod(
    *,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    effective_remanence_t: float = 0.33,
) -> ElectropermanentRod:
    """Return the published one-inch by six-inch variable-field NMR rod.

    The default effective remanence is inferred from the reported approximately
    150 mT field 1 mm from the rod face.  It is not the nominal 1.27 T material
    remanence and carries explicit inferred provenance.
    """

    source = EvidenceRecord(
        source="Ropp et al., JMR 303 (2019), 82-90",
        classification="specified",
        detail="25.4 mm diameter, 152.4 mm length, 50 turns, 14 AWG, 25 uH",
    )
    state = RemanenceState(
        effective_remanence_t,
        branch="partial",
        calibration_id="Ropp2019-surface-field-inference",
        uncertainty_t=0.03,
        evidence=(
            EvidenceRecord(
                source="Ropp et al., JMR 303 (2019), 82-90",
                classification="inferred",
                detail="Effective Br inferred from about 150 mT at 1 mm",
            ),
        ),
    )
    coil = ProgrammingCoil(
        turns=50,
        wire_gauge_awg=14,
        inductance_h=25e-6,
        evidence=(source,),
    )
    return ElectropermanentRod(
        center_m=tuple(center_m),
        axis=tuple(axis),
        length_m=0.1524,
        radius_m=0.0127,
        material=ALNICO5_AC500,
        state=state,
        coil=coil,
        label="variable-field NMR AlNiCo-5 rod",
        evidence=(source,),
    )


def weinberg_37_rod_bundle(
    *,
    center_m: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] = (0.0, 0.0, 1.0),
    state: RemanenceState = UNPROGRAMMED_STATE,
) -> ElectropermanentBundle:
    """Return the locally documented 37-rod AlNiCo-5 bundle geometry."""

    source = EvidenceRecord(
        source="Weinberg archive: Magnet Cabling and other physical attributes",
        classification="specified",
        detail="37 close-packed 1/8 inch by 4 inch AlNiCo-5 rods",
        revision="2020-01-05",
    )
    coil_source = EvidenceRecord(
        source="Weinberg archive: Magnet Cabling and other physical attributes",
        classification="specified",
        detail="About 60 turns of 16 AWG wire and L about 50 uH",
        revision="2020-01-05",
    )
    return close_packed_rod_bundle(
        37,
        rod_radius_m=0.5 * 0.125 * 0.0254,
        rod_length_m=4.0 * 0.0254,
        center_m=center_m,
        axis=axis,
        material=ALNICO5_AC500,
        state=state,
        coil=ProgrammingCoil(
            turns=60,
            wire_gauge_awg=16,
            inductance_h=50e-6,
            evidence=(coil_source,),
        ),
        label="Weinberg 37-rod bundle",
        evidence=(source,),
    )


@dataclass(frozen=True)
class ElectropermanentFieldMaps:
    """Sampled EPM field with adapters to existing MR field containers."""

    axes: tuple[np.ndarray, ...]
    cartesian_axes: tuple[int, ...]
    sources: tuple[ElectropermanentSource, ...]
    field_direction: tuple[float, float, float]
    b0_vector: np.ndarray
    b0_projected: np.ndarray
    b0_magnitude: np.ndarray
    b0_gradient: np.ndarray
    larmor_hz: np.ndarray

    @property
    def center_field_t(self) -> float:
        """Projected field at the grid sample nearest the origin."""

        index = tuple(int(np.argmin(np.abs(axis))) for axis in self.axes)
        return float(self.b0_projected[index])

    def to_motion_field_maps(
        self,
        *,
        gyromagnetic_ratio: float = GAMMA_PROTON,
        reference_field_t: float | None = None,
        b1_tx_map: np.ndarray | None = None,
        b1_rx_map: np.ndarray | None = None,
    ):
        """Return the existing motion map with B0 in rad/s off-resonance."""

        from spin_dynamics.motion import make_motion_field_maps

        reference = self.center_field_t if reference_field_t is None else float(
            reference_field_t
        )
        return make_motion_field_maps(
            self.axes,
            b0_map=float(gyromagnetic_ratio) * (self.b0_projected - reference),
            b1_tx_map=b1_tx_map,
            b1_rx_map=b1_rx_map,
        )

    def to_spatial_field_maps(
        self,
        *,
        rho: float | np.ndarray = 1.0,
        t1_map: float | np.ndarray = 1.0,
        t2_map: float | np.ndarray = 0.1,
        b1_tx_map: float | np.ndarray = 1.0,
        b1_rx_map: float | np.ndarray | None = None,
        gyromagnetic_ratio: float = GAMMA_PROTON,
        reference_field_t: float | None = None,
    ):
        """Return :class:`SpatialFieldMaps` for existing imaging workflows."""

        from spin_dynamics.fields.domain import SpatialDomain
        from spin_dynamics.fields.maps import SpatialFieldMaps

        shape = self.b0_projected.shape

        def array(value: float | np.ndarray, name: str) -> np.ndarray:
            result = np.asarray(value, dtype=np.float64)
            try:
                result = np.broadcast_to(result, shape).copy()
            except ValueError as error:
                raise ValueError(f"{name} must be scalar or broadcast to {shape}") from error
            if not np.all(np.isfinite(result)):
                raise ValueError(f"{name} must be finite")
            return result

        reference = self.center_field_t if reference_field_t is None else float(
            reference_field_t
        )
        transmit = array(b1_tx_map, "b1_tx_map")
        receive = transmit.copy() if b1_rx_map is None else array(
            b1_rx_map, "b1_rx_map"
        )
        return SpatialFieldMaps(
            domain=SpatialDomain(self.axes),
            rho=array(rho, "rho"),
            t1_map=array(t1_map, "t1_map"),
            t2_map=array(t2_map, "t2_map"),
            b0_map=float(gyromagnetic_ratio) * (self.b0_projected - reference),
            b1_tx_map=transmit,
            b1_rx_map=receive,
        )


def _spatial_axes(
    axes: Sequence[Sequence[float] | np.ndarray],
    cartesian_axes: Sequence[int] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    coordinate_axes = tuple(np.asarray(axis, dtype=np.float64).reshape(-1) for axis in axes)
    if not 1 <= len(coordinate_axes) <= 3:
        raise ValueError("axes must contain one, two, or three coordinate axes")
    if any(
        axis.size < 2
        or not np.all(np.isfinite(axis))
        or not np.all(np.diff(axis) > 0.0)
        for axis in coordinate_axes
    ):
        raise ValueError("each coordinate axis must be finite and strictly increasing")
    if cartesian_axes is None:
        defaults = {1: (2,), 2: (0, 2), 3: (0, 1, 2)}
        cartesian = defaults[len(coordinate_axes)]
    else:
        cartesian = tuple(int(axis) for axis in cartesian_axes)
    if (
        len(cartesian) != len(coordinate_axes)
        or len(set(cartesian)) != len(cartesian)
        or any(axis not in (0, 1, 2) for axis in cartesian)
    ):
        raise ValueError("cartesian_axes must select one distinct xyz index per axis")
    return coordinate_axes, cartesian


def sample_electropermanent_field(
    axes: Sequence[Sequence[float] | np.ndarray],
    sources: Sequence[ElectropermanentSource],
    *,
    cartesian_axes: Sequence[int] | None = None,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    n_cross: int = 7,
    n_length: int = 31,
    chunk_size: int = 4096,
    gyromagnetic_ratio: float = GAMMA_PROTON,
) -> ElectropermanentFieldMaps:
    """Sample an EPM assembly onto a 1-D, 2-D, or 3-D Cartesian grid."""

    coordinate_axes, cartesian = _spatial_axes(axes, cartesian_axes)
    direction = _unit_vector(field_direction, "field_direction")
    grids = np.meshgrid(*coordinate_axes, indexing="ij")
    points = np.zeros(grids[0].shape + (3,), dtype=np.float64)
    for grid, axis in zip(grids, cartesian):
        points[..., axis] = grid
    source_tuple = tuple(sources)
    field = electropermanent_field(
        points,
        source_tuple,
        n_cross=n_cross,
        n_length=n_length,
        chunk_size=chunk_size,
    )
    projected = field @ np.asarray(direction)
    magnitude = np.linalg.norm(field, axis=-1)
    derivatives = np.gradient(magnitude, *coordinate_axes)
    if len(coordinate_axes) == 1:
        derivatives = (derivatives,)
    gradient = np.sqrt(sum(np.asarray(value) ** 2 for value in derivatives))
    larmor = float(gyromagnetic_ratio) * np.abs(projected) / (2.0 * np.pi)
    return ElectropermanentFieldMaps(
        axes=coordinate_axes,
        cartesian_axes=cartesian,
        sources=source_tuple,
        field_direction=direction,
        b0_vector=field,
        b0_projected=projected,
        b0_magnitude=magnitude,
        b0_gradient=gradient,
        larmor_hz=larmor,
    )


__all__ = [
    "EvidenceClass",
    "EvidenceRecord",
    "AlNiCoMaterial",
    "ALNICO5_AC500",
    "ALNICO5_FEMM_2019",
    "RemanenceBranch",
    "RemanenceState",
    "UNPROGRAMMED_STATE",
    "ProgrammingCoil",
    "ElectropermanentRod",
    "ElectropermanentBundle",
    "ElectropermanentSource",
    "ElectropermanentFieldMaps",
    "electropermanent_field",
    "finite_cylinder_on_axis_field",
    "hexagonal_rod_offsets",
    "close_packed_rod_bundle",
    "variable_field_nmr_rod",
    "weinberg_37_rod_bundle",
    "sample_electropermanent_field",
]
