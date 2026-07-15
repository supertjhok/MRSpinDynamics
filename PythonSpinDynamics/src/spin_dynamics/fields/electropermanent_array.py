"""Hybrid electropermanent arrays and constrained field-state synthesis.

The system geometry follows the locally documented hierarchy: two opposing
3-by-3 panels, one fixed NdFeB contribution and four independently programmed
AlNiCo elements per hybrid sub-unit, for 18 sub-units and 72 programmable
elements.  Exact dimensions of the historical array are incomplete, so the
built-in geometry is explicitly illustrative while preserving the published
hierarchy, 150 mm panel gap, and 40 mm control region.

Magnetostatics are separated from control.  A field-basis object caches the
fixed field and the vector field per tesla of retained remanence for every
programmable element.  Uniform imaging, field-off, and affine transport modes
then become bounded regularized least-squares problems on that shared basis.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.electropermanent import (
    ALNICO5_AC500,
    ElectropermanentRod,
    EvidenceRecord,
    RemanenceState,
    electropermanent_field,
)
from spin_dynamics.fields.magnetostatics import FiniteMagnetRod, finite_magnet_array_b0

SynthesisMode = Literal["imaging", "field_off", "transport", "custom"]
SynthesisBackend = Literal["auto", "numpy", "scipy"]


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or np.any(~np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return vector


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = _finite_vector(values, name)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


@dataclass(frozen=True)
class HybridEPMSubunit:
    """One fixed NdFeB source plus independently programmable AlNiCo rods."""

    fixed_magnets: tuple[FiniteMagnetRod, ...]
    programmable_elements: tuple[ElectropermanentRod, ...]
    remanence_limit_t: float
    panel_index: int = 0
    grid_index: tuple[int, int] = (0, 0)
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        fixed = tuple(self.fixed_magnets)
        programmable = tuple(self.programmable_elements)
        if not fixed:
            raise ValueError("fixed_magnets must not be empty")
        if not programmable:
            raise ValueError("programmable_elements must not be empty")
        if not np.isfinite(self.remanence_limit_t) or self.remanence_limit_t <= 0.0:
            raise ValueError("remanence_limit_t must be finite and positive")
        if any(self.remanence_limit_t > rod.material.remanence_t for rod in programmable):
            raise ValueError("remanence_limit_t exceeds a programmable material limit")
        if int(self.panel_index) != self.panel_index or self.panel_index < 0:
            raise ValueError("panel_index must be a non-negative integer")
        if len(self.grid_index) != 2 or any(int(value) != value for value in self.grid_index):
            raise ValueError("grid_index must contain two integers")
        object.__setattr__(self, "fixed_magnets", fixed)
        object.__setattr__(self, "programmable_elements", programmable)
        object.__setattr__(self, "grid_index", tuple(int(value) for value in self.grid_index))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def center_m(self) -> np.ndarray:
        """Mean programmable-element center."""

        return np.mean([rod.center_m for rod in self.programmable_elements], axis=0)


@dataclass(frozen=True)
class ElectropermanentArray:
    """A collection of hybrid sub-units with flattened control channels."""

    subunits: tuple[HybridEPMSubunit, ...]
    label: str = ""
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        subunits = tuple(self.subunits)
        if not subunits:
            raise ValueError("subunits must not be empty")
        object.__setattr__(self, "subunits", subunits)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    @property
    def fixed_magnets(self) -> tuple[FiniteMagnetRod, ...]:
        """Flattened fixed NdFeB sources."""

        return tuple(magnet for subunit in self.subunits for magnet in subunit.fixed_magnets)

    @property
    def programmable_elements(self) -> tuple[ElectropermanentRod, ...]:
        """Flattened AlNiCo control elements."""

        return tuple(
            element
            for subunit in self.subunits
            for element in subunit.programmable_elements
        )

    @property
    def remanence_limits_t(self) -> np.ndarray:
        """Per-element symmetric retained-remanence limits."""

        return np.asarray(
            [
                subunit.remanence_limit_t
                for subunit in self.subunits
                for _ in subunit.programmable_elements
            ],
            dtype=np.float64,
        )

    @property
    def retained_remanence_t(self) -> np.ndarray:
        """Current flattened retained-state vector."""

        return np.asarray(
            [element.state.remanence_t for element in self.programmable_elements],
            dtype=np.float64,
        )

    def with_remanence(self, remanence_t: Sequence[float] | np.ndarray) -> ElectropermanentArray:
        """Return the same array geometry with a new flattened state vector."""

        values = np.asarray(remanence_t, dtype=np.float64)
        elements = self.programmable_elements
        if values.shape != (len(elements),) or np.any(~np.isfinite(values)):
            raise ValueError("remanence_t must match programmable_elements")
        if np.any(np.abs(values) > self.remanence_limits_t * (1.0 + 1e-12)):
            raise ValueError("remanence_t exceeds an array element limit")
        cursor = 0
        updated_subunits = []
        for subunit in self.subunits:
            updated_elements = []
            for element in subunit.programmable_elements:
                value = float(values[cursor])
                limit = subunit.remanence_limit_t
                if value >= limit * (1.0 - 1e-10):
                    branch = "positive_saturation"
                elif value <= -limit * (1.0 - 1e-10):
                    branch = "negative_saturation"
                else:
                    branch = "partial"
                updated_elements.append(
                    element.with_state(
                        RemanenceState(
                            value,
                            branch=branch,
                            temperature_k=element.state.temperature_k,
                            calibration_id=element.state.calibration_id,
                            uncertainty_t=element.state.uncertainty_t,
                            evidence=element.state.evidence,
                        )
                    )
                )
                cursor += 1
            updated_subunits.append(
                HybridEPMSubunit(
                    fixed_magnets=subunit.fixed_magnets,
                    programmable_elements=tuple(updated_elements),
                    remanence_limit_t=subunit.remanence_limit_t,
                    panel_index=subunit.panel_index,
                    grid_index=subunit.grid_index,
                    label=subunit.label,
                    evidence=subunit.evidence,
                )
            )
        return ElectropermanentArray(tuple(updated_subunits), self.label, self.evidence)

    def field_vectors(
        self,
        points_m: np.ndarray,
        *,
        remanence_t: Sequence[float] | np.ndarray | None = None,
        n_cross: int = 3,
        n_length: int = 11,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Evaluate fixed and programmed fields directly at arbitrary points."""

        points = np.asarray(points_m, dtype=np.float64)
        if points.shape[-1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("points_m must have shape (..., 3) and be finite")
        active = self if remanence_t is None else self.with_remanence(remanence_t)
        fixed = finite_magnet_array_b0(
            points,
            active.fixed_magnets,
            n_cross=n_cross,
            n_length=n_length,
            chunk_size=chunk_size,
        )
        programmable = electropermanent_field(
            points,
            active.programmable_elements,
            n_cross=n_cross,
            n_length=n_length,
            chunk_size=chunk_size,
        )
        return fixed + programmable

    def build_field_basis(
        self,
        points_m: np.ndarray,
        *,
        n_cross: int = 3,
        n_length: int = 11,
        chunk_size: int = 4096,
    ) -> ElectropermanentArrayFieldBasis:
        """Cache fixed and per-tesla programmable vector fields."""

        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("points_m must have shape (n_points, 3) and be finite")
        fixed = finite_magnet_array_b0(
            points,
            self.fixed_magnets,
            n_cross=n_cross,
            n_length=n_length,
            chunk_size=chunk_size,
        )
        elements = self.programmable_elements
        fields = np.empty((points.shape[0], 3, len(elements)), dtype=np.float64)
        for index, element in enumerate(elements):
            unit = element.with_state(
                RemanenceState(
                    1.0,
                    branch="partial",
                    calibration_id="unit-remanence-field-basis",
                )
            )
            fields[:, :, index] = electropermanent_field(
                points,
                (unit,),
                n_cross=n_cross,
                n_length=n_length,
                chunk_size=chunk_size,
            )
        return ElectropermanentArrayFieldBasis(
            array=self,
            points_m=points.copy(),
            fixed_field_t=fixed,
            programmable_field_t_per_t=fields,
            n_cross=int(n_cross),
            n_length=int(n_length),
        )


@dataclass(frozen=True)
class ElectropermanentArrayFieldBasis:
    """Cached fixed field and one vector-field column per AlNiCo element."""

    array: ElectropermanentArray
    points_m: np.ndarray
    fixed_field_t: np.ndarray
    programmable_field_t_per_t: np.ndarray
    n_cross: int
    n_length: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_m, dtype=np.float64)
        fixed = np.asarray(self.fixed_field_t, dtype=np.float64)
        programmable = np.asarray(self.programmable_field_t_per_t, dtype=np.float64)
        count = len(self.array.programmable_elements)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("points_m must have shape (n_points, 3)")
        if fixed.shape != points.shape or np.any(~np.isfinite(fixed)):
            raise ValueError("fixed_field_t must match points_m")
        if programmable.shape != (points.shape[0], 3, count) or np.any(~np.isfinite(programmable)):
            raise ValueError("programmable_field_t_per_t has an invalid shape")
        object.__setattr__(self, "points_m", points)
        object.__setattr__(self, "fixed_field_t", fixed)
        object.__setattr__(self, "programmable_field_t_per_t", programmable)

    def field_vectors(self, remanence_t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate a retained-state vector without rerunning magnetostatics."""

        values = np.asarray(remanence_t, dtype=np.float64)
        count = self.programmable_field_t_per_t.shape[2]
        if values.shape != (count,) or np.any(~np.isfinite(values)):
            raise ValueError("remanence_t must match the field-basis columns")
        if np.any(np.abs(values) > self.array.remanence_limits_t * (1.0 + 1e-12)):
            raise ValueError("remanence_t exceeds an array element limit")
        return self.fixed_field_t + np.einsum(
            "pce,e->pc",
            self.programmable_field_t_per_t,
            values,
        )

    def projected_matrix(self, direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Return fixed field and control matrix projected on one direction."""

        unit = _unit_vector(direction, "direction")
        fixed = self.fixed_field_t @ unit
        matrix = np.einsum("pce,c->pe", self.programmable_field_t_per_t, unit)
        return fixed, matrix


@dataclass(frozen=True)
class ArrayStateSynthesisResult:
    """Bounded retained-state solution for one array field objective."""

    mode: SynthesisMode
    basis: ElectropermanentArrayFieldBasis
    field_direction: tuple[float, float, float]
    target_projected_field_t: np.ndarray
    remanence_t: np.ndarray
    predicted_field_t: np.ndarray
    predicted_projected_field_t: np.ndarray
    weights: np.ndarray
    regularization: float
    reference_remanence_t: np.ndarray
    backend: str
    iterations: int
    converged: bool
    condition_number: float

    @property
    def residual_t(self) -> np.ndarray:
        """Projected field error at every synthesis point."""

        return self.predicted_projected_field_t - self.target_projected_field_t

    @property
    def rms_error_t(self) -> float:
        """Root-mean-square projected field error."""

        return float(np.sqrt(np.mean(self.residual_t**2)))

    @property
    def max_abs_error_t(self) -> float:
        """Maximum absolute projected field error."""

        return float(np.max(np.abs(self.residual_t)))

    @property
    def mean_projected_field_t(self) -> float:
        """Mean achieved projected field."""

        return float(np.mean(self.predicted_projected_field_t))

    @property
    def homogeneity_ppm(self) -> float:
        """Standard-deviation homogeneity relative to the achieved mean."""

        mean = abs(self.mean_projected_field_t)
        return float(np.inf if mean == 0.0 else 1e6 * np.std(self.predicted_projected_field_t) / mean)

    def applied_array(self) -> ElectropermanentArray:
        """Return the array geometry carrying the synthesized retained states."""

        return self.basis.array.with_remanence(self.remanence_t)


def _target_array(target_field_t: float | Sequence[float] | np.ndarray, count: int) -> np.ndarray:
    target = np.asarray(target_field_t, dtype=np.float64)
    if target.ndim == 0:
        target = np.full(count, float(target))
    if target.shape != (count,) or np.any(~np.isfinite(target)):
        raise ValueError("target_field_t must be scalar or match basis points")
    return target


def synthesize_epm_array_state(
    basis: ElectropermanentArrayFieldBasis,
    target_field_t: float | Sequence[float] | np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    mode: SynthesisMode = "custom",
    weights: float | Sequence[float] | np.ndarray = 1.0,
    regularization: float = 1e-6,
    reference_remanence_t: Sequence[float] | np.ndarray | None = None,
    backend: SynthesisBackend = "auto",
    max_iterations: int = 20_000,
    tolerance_t: float = 1e-10,
) -> ArrayStateSynthesisResult:
    """Solve a bounded regularized projected-field least-squares problem."""

    if mode not in {"imaging", "field_off", "transport", "custom"}:
        raise ValueError("invalid synthesis mode")
    if backend not in {"auto", "numpy", "scipy"}:
        raise ValueError("backend must be 'auto', 'numpy', or 'scipy'")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance_t) or tolerance_t <= 0.0:
        raise ValueError("tolerance_t must be finite and positive")
    unit = _unit_vector(field_direction, "field_direction")
    fixed, matrix = basis.projected_matrix(unit)
    target = _target_array(target_field_t, basis.points_m.shape[0])
    weight = np.asarray(weights, dtype=np.float64)
    if weight.ndim == 0:
        weight = np.full(target.shape, float(weight))
    if weight.shape != target.shape or np.any(~np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weights must be finite, non-negative, and match basis points")
    if not np.any(weight > 0.0):
        raise ValueError("at least one weight must be positive")
    count = matrix.shape[1]
    if reference_remanence_t is None:
        reference = np.zeros(count)
    else:
        reference = np.asarray(reference_remanence_t, dtype=np.float64)
        if reference.shape != (count,) or np.any(~np.isfinite(reference)):
            raise ValueError("reference_remanence_t must match array elements")
    limits = basis.array.remanence_limits_t
    reference = np.clip(reference, -limits, limits)
    root_weight = np.sqrt(weight)
    weighted_matrix = root_weight[:, None] * matrix
    weighted_rhs = root_weight * (target - fixed)
    singular_values = np.linalg.svd(weighted_matrix, compute_uv=False)
    condition = float(
        np.inf
        if not singular_values.size or singular_values[-1] <= np.finfo(float).eps * singular_values[0]
        else singular_values[0] / singular_values[-1]
    )

    selected_backend = backend
    if backend in {"auto", "scipy"}:
        try:
            from scipy.optimize import lsq_linear

            augmented_matrix = weighted_matrix
            augmented_rhs = weighted_rhs
            if regularization > 0.0:
                scale = np.sqrt(regularization)
                augmented_matrix = np.vstack((weighted_matrix, scale * np.eye(count)))
                augmented_rhs = np.concatenate((weighted_rhs, scale * reference))
            solved = lsq_linear(
                augmented_matrix,
                augmented_rhs,
                bounds=(-limits, limits),
                tol=tolerance_t,
                max_iter=max_iterations,
            )
            remanence = solved.x
            iterations = int(solved.nit if solved.nit is not None else 0)
            converged = bool(solved.success)
            selected_backend = "scipy"
        except ImportError:
            if backend == "scipy":
                raise ImportError("SciPy is required for backend='scipy'") from None
            selected_backend = "numpy"
    if selected_backend == "numpy":
        spectral_norm = float(np.linalg.norm(weighted_matrix, ord=2))
        lipschitz = spectral_norm**2 + regularization
        if lipschitz == 0.0:
            raise ValueError("synthesis system has zero sensitivity and regularization")
        remanence = reference.copy()
        accelerated = remanence.copy()
        momentum = 1.0
        converged = False
        for iterations in range(1, max_iterations + 1):
            gradient = weighted_matrix.T @ (
                weighted_matrix @ accelerated - weighted_rhs
            ) + regularization * (accelerated - reference)
            updated = np.clip(accelerated - gradient / lipschitz, -limits, limits)
            if np.max(np.abs(updated - remanence)) <= tolerance_t:
                remanence = updated
                converged = True
                break
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            accelerated = updated + (momentum - 1.0) / next_momentum * (updated - remanence)
            remanence = updated
            momentum = next_momentum
    predicted = basis.field_vectors(remanence)
    projected = predicted @ unit
    return ArrayStateSynthesisResult(
        mode=mode,
        basis=basis,
        field_direction=tuple(float(value) for value in unit),
        target_projected_field_t=target,
        remanence_t=remanence,
        predicted_field_t=predicted,
        predicted_projected_field_t=projected,
        weights=weight,
        regularization=float(regularization),
        reference_remanence_t=reference,
        backend=selected_backend,
        iterations=iterations,
        converged=converged,
        condition_number=condition,
    )


def synthesize_uniform_imaging_state(
    basis: ElectropermanentArrayFieldBasis,
    field_t: float,
    **kwargs,
) -> ArrayStateSynthesisResult:
    """Synthesize a uniform projected imaging field over the basis points."""

    if not np.isfinite(field_t):
        raise ValueError("field_t must be finite")
    return synthesize_epm_array_state(
        basis,
        float(field_t),
        mode="imaging",
        **kwargs,
    )


def synthesize_field_off_state(
    basis: ElectropermanentArrayFieldBasis,
    **kwargs,
) -> ArrayStateSynthesisResult:
    """Minimize the projected field over the basis points."""

    return synthesize_epm_array_state(basis, 0.0, mode="field_off", **kwargs)


def affine_transport_target(
    points_m: np.ndarray,
    *,
    bias_field_t: float,
    gradient_t_per_m: Sequence[float],
    center_m: Sequence[float] | None = None,
) -> np.ndarray:
    """Return ``bias + gradient dot (r-center)`` for directional transport."""

    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
        raise ValueError("points_m must have shape (n_points, 3)")
    if not np.isfinite(bias_field_t):
        raise ValueError("bias_field_t must be finite")
    gradient = _finite_vector(gradient_t_per_m, "gradient_t_per_m")
    center = np.mean(points, axis=0) if center_m is None else _finite_vector(center_m, "center_m")
    return float(bias_field_t) + (points - center) @ gradient


def synthesize_transport_state(
    basis: ElectropermanentArrayFieldBasis,
    *,
    bias_field_t: float,
    gradient_t_per_m: Sequence[float],
    center_m: Sequence[float] | None = None,
    **kwargs,
) -> ArrayStateSynthesisResult:
    """Synthesize an affine projected field for a directional transport burst."""

    target = affine_transport_target(
        basis.points_m,
        bias_field_t=bias_field_t,
        gradient_t_per_m=gradient_t_per_m,
        center_m=center_m,
    )
    return synthesize_epm_array_state(
        basis,
        target,
        mode="transport",
        **kwargs,
    )


def illustrative_hybrid_epm_array(
    *,
    panel_gap_m: float = 0.150,
    grid_pitch_m: float = 0.035,
    subunit_element_offset_m: float = 0.008,
    magnet_length_m: float = 0.030,
    fixed_ndfeb_remanence_t: float = 1.20,
    alnico_remanence_limit_t: float = 0.33,
) -> ElectropermanentArray:
    """Return the published 18-sub-unit/72-element hierarchy.

    The hierarchy, panel gap, and 40 mm control-region intent are sourced from
    the Weinberg project poster.  Element radii, pitch, length, and exact hybrid
    layout are illustrative engineering assumptions until complete CAD or
    measured field bases are available.
    """

    for name, value in (
        ("panel_gap_m", panel_gap_m),
        ("grid_pitch_m", grid_pitch_m),
        ("subunit_element_offset_m", subunit_element_offset_m),
        ("magnet_length_m", magnet_length_m),
        ("fixed_ndfeb_remanence_t", fixed_ndfeb_remanence_t),
        ("alnico_remanence_limit_t", alnico_remanence_limit_t),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    hierarchy = EvidenceRecord(
        source="Weinberg archive: Tunable Electropermanent System poster",
        classification="specified",
        detail=(
            "Two 3x3 panels, 18 hybrid sub-units, four adjustable AlNiCo "
            "elements per sub-unit, 72 controls, 150 mm gap, 40 mm ROI"
        ),
    )
    inferred = EvidenceRecord(
        source="PythonSpinDynamics illustrative hybrid-array geometry",
        classification="inferred",
        detail=(
            "35 mm grid pitch, 30 mm magnet length, 3 mm radii, and 8 mm "
            "AlNiCo offsets selected for a non-overlapping synthesis model"
        ),
    )
    offsets = (
        (subunit_element_offset_m, 0.0),
        (-subunit_element_offset_m, 0.0),
        (0.0, subunit_element_offset_m),
        (0.0, -subunit_element_offset_m),
    )
    subunits = []
    for panel, z in enumerate((-0.5 * panel_gap_m, 0.5 * panel_gap_m)):
        for row, y_index in enumerate((-1, 0, 1)):
            for column, x_index in enumerate((-1, 0, 1)):
                x = x_index * grid_pitch_m
                y = y_index * grid_pitch_m
                fixed = FiniteMagnetRod(
                    center=(x, y, z),
                    length=magnet_length_m,
                    br=(0.0, 0.0, fixed_ndfeb_remanence_t),
                    shape="cylinder",
                    radius=0.003,
                )
                elements = tuple(
                    ElectropermanentRod(
                        center_m=(x + dx, y + dy, z),
                        axis=(0.0, 0.0, 1.0),
                        length_m=magnet_length_m,
                        radius_m=0.003,
                        material=ALNICO5_AC500,
                        state=RemanenceState(
                            0.0,
                            branch="partial",
                            calibration_id="illustrative-hybrid-array-zero",
                            evidence=(inferred,),
                        ),
                        label=(
                            f"panel {panel} subunit {row},{column} "
                            f"AlNiCo {element_index}"
                        ),
                        evidence=(hierarchy, inferred),
                    )
                    for element_index, (dx, dy) in enumerate(offsets)
                )
                subunits.append(
                    HybridEPMSubunit(
                        fixed_magnets=(fixed,),
                        programmable_elements=elements,
                        remanence_limit_t=alnico_remanence_limit_t,
                        panel_index=panel,
                        grid_index=(row, column),
                        label=f"panel {panel} hybrid subunit {row},{column}",
                        evidence=(hierarchy, inferred),
                    )
                )
    return ElectropermanentArray(
        tuple(subunits),
        label="illustrative 18-sub-unit / 72-element hybrid EPM array",
        evidence=(hierarchy, inferred),
    )


__all__ = [
    "HybridEPMSubunit",
    "ElectropermanentArray",
    "ElectropermanentArrayFieldBasis",
    "ArrayStateSynthesisResult",
    "synthesize_epm_array_state",
    "synthesize_uniform_imaging_state",
    "synthesize_field_off_state",
    "affine_transport_target",
    "synthesize_transport_state",
    "illustrative_hybrid_epm_array",
]
