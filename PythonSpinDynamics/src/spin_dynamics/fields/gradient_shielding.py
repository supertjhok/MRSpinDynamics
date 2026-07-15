"""Actively shielded cylindrical gradient-coil design.

Two concentric stream-function surfaces are optimized together: the primary
surface produces the requested field in the imaging volume while the shield
surface cancels fringe field at exterior observation points.  Each surface has
its own exact axial KCL constraints and optional current-penalty weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.gradient_coils import (
    CylindricalWindingSurface,
    SolverName,
    _l_curve_curvature,
    _solve_multi_reduced_system,
)
from spin_dynamics.fields.magnetostatics import segment_field_sensitivity


@dataclass(frozen=True)
class ActivelyShieldedGradientSystem:
    """Combined primary/shield source matrix and its two field regions."""

    primary_surface: CylindricalWindingSurface
    shield_surface: CylindricalWindingSurface
    target_points: np.ndarray
    shield_points: np.ndarray
    field_direction: np.ndarray
    sensitivity: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target_points, dtype=np.float64)
        shield = np.asarray(self.shield_points, dtype=np.float64)
        direction = np.asarray(self.field_direction, dtype=np.float64)
        sensitivity = np.asarray(self.sensitivity, dtype=np.float64)
        if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] == 0:
            raise ValueError("target_points must have shape (n_target, 3)")
        if shield.ndim != 2 or shield.shape[1] != 3 or shield.shape[0] == 0:
            raise ValueError("shield_points must have shape (n_shield, 3)")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(shield)):
            raise ValueError("field points must be finite")
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("field_direction must be a finite 3-vector")
        if not np.isclose(np.linalg.norm(direction), 1.0):
            raise ValueError("field_direction must be normalized")
        expected = (
            target.shape[0] + shield.shape[0],
            self.primary_surface.segment_count + self.shield_surface.segment_count,
        )
        if sensitivity.shape != expected or not np.all(np.isfinite(sensitivity)):
            raise ValueError(f"sensitivity must be a finite array of shape {expected}")

    @property
    def surfaces(self) -> tuple[CylindricalWindingSurface, ...]:
        return self.primary_surface, self.shield_surface

    @property
    def shapes(self) -> tuple[tuple[int, int], ...]:
        return self.primary_surface.shape, self.shield_surface.shape

    @property
    def target_sensitivity(self) -> np.ndarray:
        return self.sensitivity[: self.target_points.shape[0]]

    @property
    def shield_sensitivity(self) -> np.ndarray:
        return self.sensitivity[self.target_points.shape[0] :]


@dataclass(frozen=True)
class ActivelyShieldedGradientDesignResult:
    """Primary/shield currents, stream functions, fields, and diagnostics."""

    system: ActivelyShieldedGradientSystem
    target_field_t: np.ndarray
    predicted_target_field_t: np.ndarray
    predicted_shield_field_t: np.ndarray
    primary_segment_currents_a: np.ndarray
    shield_segment_currents_a: np.ndarray
    primary_stream_function_a: np.ndarray
    shield_stream_function_a: np.ndarray
    regularization_t2_per_a2: float
    objective: float
    target_weighted_rms_error_t: float
    target_relative_rms_error: float
    shield_weighted_rms_field_t: float
    shield_normalized_rms_field: float
    shield_normalized_max_field: float
    primary_current_norm_a: float
    shield_current_norm_a: float
    closure_error_a: float
    solver: str
    stop_code: int
    iterations: int

    @property
    def target_residual_t(self) -> np.ndarray:
        return self.predicted_target_field_t - self.target_field_t

    @property
    def primary_stream_z(self) -> np.ndarray:
        return _stream_z(self.system.primary_surface)

    @property
    def shield_stream_z(self) -> np.ndarray:
        return _stream_z(self.system.shield_surface)


@dataclass(frozen=True)
class ActivelyShieldedRegularizationPath:
    """L-curve diagnostics for an actively shielded design."""

    regularizations_t2_per_a2: np.ndarray
    weighted_residual_norms_t: np.ndarray
    weighted_current_norms_a: np.ndarray
    l_curve_curvature: np.ndarray
    selected_index: int
    results: tuple[ActivelyShieldedGradientDesignResult, ...]

    @property
    def selected_regularization(self) -> float:
        return float(self.regularizations_t2_per_a2[self.selected_index])

    @property
    def selected_result(self) -> ActivelyShieldedGradientDesignResult:
        return self.results[self.selected_index]


def spherical_shell_points(
    radius: float,
    *,
    n_points: int = 96,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return approximately uniform Fibonacci points on a spherical shell."""

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if int(n_points) != n_points or n_points < 8:
        raise ValueError("n_points must be an integer of at least 8")
    origin = np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite 3-vector")
    index = np.arange(int(n_points), dtype=np.float64)
    z = 1.0 - 2.0 * (index + 0.5) / int(n_points)
    phi = index * (np.pi * (3.0 - np.sqrt(5.0)))
    radial = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    unit = np.column_stack([radial * np.cos(phi), radial * np.sin(phi), z])
    return origin + float(radius) * unit


def cylindrical_shield_points(
    radius: float,
    length: float,
    *,
    n_phi: int = 32,
    n_z: int = 17,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return a symmetric point grid on a cylindrical shield-control surface.

    The grid samples the lateral surface, including both axial end coordinates
    but not the circular end caps.  Symmetric azimuthal sampling is useful when
    the target and winding surfaces share an axis.
    """

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("length must be finite and positive")
    if int(n_phi) != n_phi or n_phi < 4:
        raise ValueError("n_phi must be an integer of at least 4")
    if int(n_z) != n_z or n_z < 2:
        raise ValueError("n_z must be an integer of at least 2")
    origin = np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite 3-vector")
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    z = np.linspace(-0.5 * float(length), 0.5 * float(length), int(n_z))
    phi_grid, z_grid = np.meshgrid(phi, z, indexing="ij")
    return origin + np.column_stack(
        [
            float(radius) * np.cos(phi_grid.ravel()),
            float(radius) * np.sin(phi_grid.ravel()),
            z_grid.ravel(),
        ]
    )


def build_actively_shielded_gradient_system(
    primary_surface: CylindricalWindingSurface,
    shield_surface: CylindricalWindingSurface,
    target_points: np.ndarray,
    shield_points: np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    chunk_size: int = 128,
) -> ActivelyShieldedGradientSystem:
    """Build the combined field matrix for concentric primary/shield cylinders."""

    primary_center = np.asarray(primary_surface.center, dtype=np.float64)
    shield_center = np.asarray(shield_surface.center, dtype=np.float64)
    if not np.allclose(primary_center, shield_center, atol=1.0e-12, rtol=0.0):
        raise ValueError("primary and shield surfaces must be concentric")
    if shield_surface.radius <= primary_surface.radius:
        raise ValueError("shield_surface radius must exceed primary_surface radius")
    target = _points(target_points, "target_points")
    exterior = _points(shield_points, "shield_points")
    direction = np.asarray(field_direction, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("field_direction must be a finite 3-vector")
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("field_direction must be nonzero")
    direction = direction / norm
    source_segments = primary_surface.segments() + shield_surface.segments()
    sensitivity = segment_field_sensitivity(
        np.vstack([target, exterior]),
        source_segments,
        direction=direction,
        chunk_size=chunk_size,
    )
    return ActivelyShieldedGradientSystem(
        primary_surface=primary_surface,
        shield_surface=shield_surface,
        target_points=target,
        shield_points=exterior,
        field_direction=direction,
        sensitivity=sensitivity,
    )


def solve_actively_shielded_gradient_coil(
    system: ActivelyShieldedGradientSystem,
    target_field_t: np.ndarray,
    *,
    regularization: float = 0.0,
    field_weights: np.ndarray | None = None,
    shield_weights: np.ndarray | float | None = None,
    surface_regularization_weights: Sequence[float] = (1.0, 1.0),
    solver: SolverName = "auto",
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    max_iterations: int | None = None,
) -> ActivelyShieldedGradientDesignResult:
    """Fit the target field while driving the exterior shield target to zero."""

    n_target = system.target_points.shape[0]
    n_shield = system.shield_points.shape[0]
    target = np.asarray(target_field_t, dtype=np.float64)
    if target.shape != (n_target,) or not np.all(np.isfinite(target)):
        raise ValueError(f"target_field_t must be a finite array of shape ({n_target},)")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    target_weights = _weights(field_weights, n_target, "field_weights")
    exterior_weights = _weights(shield_weights, n_shield, "shield_weights")
    surface_weights = np.asarray(surface_regularization_weights, dtype=np.float64)
    if surface_weights.shape != (2,) or not np.all(np.isfinite(surface_weights)):
        raise ValueError("surface_regularization_weights must contain two finite values")
    if np.any(surface_weights <= 0.0):
        raise ValueError("surface_regularization_weights must be positive")
    current_weights = np.concatenate(
        [
            np.full(system.primary_surface.segment_count, surface_weights[0]),
            np.full(system.shield_surface.segment_count, surface_weights[1]),
        ]
    )
    combined_target = np.concatenate([target, np.zeros(n_shield)])
    combined_weights = np.concatenate([target_weights, exterior_weights])
    currents, solver_used, stop_code, iterations = _solve_multi_reduced_system(
        system.sensitivity,
        combined_target,
        combined_weights,
        system.shapes,
        float(regularization),
        current_weights,
        solver,
        float(atol),
        float(btol),
        max_iterations,
    )
    predicted = system.sensitivity @ currents
    target_predicted = predicted[:n_target]
    shield_predicted = predicted[n_target:]
    target_residual = target_predicted - target
    target_rms = _weighted_rms(target_residual, target_weights)
    target_reference_rms = _weighted_rms(target, target_weights)
    exterior_rms = _weighted_rms(shield_predicted, exterior_weights)
    target_peak = float(np.max(np.abs(target), initial=0.0))
    primary_count = system.primary_surface.segment_count
    primary_flat = currents[:primary_count]
    shield_flat = currents[primary_count:]
    primary_currents = primary_flat.reshape(system.primary_surface.shape)
    shield_currents = shield_flat.reshape(system.shield_surface.shape)
    closure_error = max(
        float(np.max(np.abs(np.sum(primary_currents, axis=1)), initial=0.0)),
        float(np.max(np.abs(np.sum(shield_currents, axis=1)), initial=0.0)),
    )
    field_objective = float(np.sum(combined_weights * (predicted - combined_target) ** 2))
    current_objective = float(regularization * np.sum(current_weights * currents**2))
    return ActivelyShieldedGradientDesignResult(
        system=system,
        target_field_t=target,
        predicted_target_field_t=target_predicted,
        predicted_shield_field_t=shield_predicted,
        primary_segment_currents_a=primary_currents,
        shield_segment_currents_a=shield_currents,
        primary_stream_function_a=_currents_to_stream(primary_currents),
        shield_stream_function_a=_currents_to_stream(shield_currents),
        regularization_t2_per_a2=float(regularization),
        objective=field_objective + current_objective,
        target_weighted_rms_error_t=target_rms,
        target_relative_rms_error=(
            target_rms / target_reference_rms if target_reference_rms > 0.0 else np.inf
        ),
        shield_weighted_rms_field_t=exterior_rms,
        shield_normalized_rms_field=(
            exterior_rms / target_peak if target_peak > 0.0 else np.inf
        ),
        shield_normalized_max_field=(
            float(np.max(np.abs(shield_predicted), initial=0.0)) / target_peak
            if target_peak > 0.0
            else np.inf
        ),
        primary_current_norm_a=float(np.linalg.norm(primary_flat)),
        shield_current_norm_a=float(np.linalg.norm(shield_flat)),
        closure_error_a=closure_error,
        solver=solver_used,
        stop_code=stop_code,
        iterations=iterations,
    )


def solve_actively_shielded_regularization_path(
    system: ActivelyShieldedGradientSystem,
    target_field_t: np.ndarray,
    regularizations: Sequence[float],
    **solve_options,
) -> ActivelyShieldedRegularizationPath:
    """Solve an alpha grid and select the active design's L-curve corner."""

    values = np.asarray(regularizations, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("regularizations must contain at least three values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("regularizations must be finite and positive")
    values = np.unique(values)
    if values.size < 3:
        raise ValueError("regularizations must contain at least three unique values")
    values.sort()
    results = tuple(
        solve_actively_shielded_gradient_coil(
            system,
            target_field_t,
            regularization=float(value),
            **solve_options,
        )
        for value in values
    )
    target_weights = _weights(
        solve_options.get("field_weights"),
        system.target_points.shape[0],
        "field_weights",
    )
    exterior_weights = _weights(
        solve_options.get("shield_weights"),
        system.shield_points.shape[0],
        "shield_weights",
    )
    surface_weights = np.asarray(
        solve_options.get("surface_regularization_weights", (1.0, 1.0)),
        dtype=np.float64,
    )
    residual_norms = np.asarray(
        [
            np.sqrt(
                np.sum(target_weights * result.target_residual_t**2)
                + np.sum(
                    exterior_weights * result.predicted_shield_field_t**2
                )
            )
            for result in results
        ]
    )
    current_norms = np.asarray(
        [
            np.sqrt(
                surface_weights[0] * result.primary_current_norm_a**2
                + surface_weights[1] * result.shield_current_norm_a**2
            )
            for result in results
        ]
    )
    curvature = _l_curve_curvature(values, residual_norms, current_norms)
    selected = int(np.argmax(curvature))
    return ActivelyShieldedRegularizationPath(
        regularizations_t2_per_a2=values,
        weighted_residual_norms_t=residual_norms,
        weighted_current_norms_a=current_norms,
        l_curve_curvature=curvature,
        selected_index=selected,
        results=results,
    )


def design_actively_shielded_gradient_coil(
    primary_surface: CylindricalWindingSurface,
    shield_surface: CylindricalWindingSurface,
    target_points: np.ndarray,
    shield_points: np.ndarray,
    target_field_t: np.ndarray,
    **options,
) -> ActivelyShieldedGradientDesignResult:
    """Build and solve an actively shielded cylindrical design in one call."""

    build_keys = {"field_direction", "chunk_size"}
    build_options = {key: options.pop(key) for key in tuple(options) if key in build_keys}
    system = build_actively_shielded_gradient_system(
        primary_surface,
        shield_surface,
        target_points,
        shield_points,
        **build_options,
    )
    return solve_actively_shielded_gradient_coil(
        system,
        target_field_t,
        **options,
    )


def _points(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_points, 3), n_points > 0")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _weights(
    values: np.ndarray | float | None,
    count: int,
    name: str,
) -> np.ndarray:
    if values is None:
        result = np.ones(count, dtype=np.float64)
    elif np.asarray(values).ndim == 0:
        result = np.full(count, float(values), dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (count,):
            raise ValueError(f"{name} must be scalar or have shape ({count},)")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.any(result > 0.0):
        raise ValueError(f"at least one {name} value must be positive")
    return result


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def _currents_to_stream(currents: np.ndarray) -> np.ndarray:
    stream = np.zeros((currents.shape[0], currents.shape[1] + 1))
    stream[:, 1:] = -np.cumsum(currents, axis=1)
    return stream


def _stream_z(surface: CylindricalWindingSurface) -> np.ndarray:
    return np.linspace(
        surface.center[2] - 0.5 * surface.length,
        surface.center[2] + 0.5 * surface.length,
        int(surface.n_z) + 1,
    )


__all__ = [
    "ActivelyShieldedGradientSystem",
    "ActivelyShieldedGradientDesignResult",
    "ActivelyShieldedRegularizationPath",
    "spherical_shell_points",
    "cylindrical_shield_points",
    "build_actively_shielded_gradient_system",
    "solve_actively_shielded_gradient_coil",
    "solve_actively_shielded_regularization_path",
    "design_actively_shielded_gradient_coil",
]
