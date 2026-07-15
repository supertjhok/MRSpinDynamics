"""Stream-function design of MRI gradient coils on cylindrical surfaces.

The first implementation follows the regular-surface construction from the
textbook treatment: azimuthal thin-wire source segments are optimized to
produce a target ``B_z`` field, an axial KCL constraint closes each azimuthal
current column, and cumulative current along ``z`` recovers a discrete stream
function. The companion :mod:`gradient_windings` module extracts periodic
winding contours, while :mod:`gradient_shielding` jointly optimizes concentric
primary and shield surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.magnetostatics import segment_field_sensitivity

Segment = tuple[np.ndarray, np.ndarray]
SolverName = Literal["auto", "scipy", "numpy"]


@dataclass(frozen=True)
class CylindricalWindingSurface:
    """Regular z-axis cylindrical winding surface.

    ``n_phi`` short azimuthal segments are placed at each of ``n_z`` axial
    positions, including the two ends of the cylinder. Segment ordering is
    ``(phi_index, z_index)`` so a flat current vector reshapes directly to
    ``(n_phi, n_z)``.
    """

    radius: float
    length: float
    n_phi: int
    n_z: int
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if not np.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length must be finite and positive")
        if (
            not np.isfinite(self.n_phi)
            or int(self.n_phi) != self.n_phi
            or self.n_phi < 4
        ):
            raise ValueError("n_phi must be an integer of at least 4")
        if not np.isfinite(self.n_z) or int(self.n_z) != self.n_z or self.n_z < 2:
            raise ValueError("n_z must be an integer of at least 2")
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite 3-vector")

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the independent segment-current grid."""

        return int(self.n_phi), int(self.n_z)

    @property
    def segment_count(self) -> int:
        """Number of independently weighted azimuthal source segments."""

        return int(self.n_phi) * int(self.n_z)

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal center coordinate of each segment column (rad)."""

        return np.arange(int(self.n_phi), dtype=np.float64) * (
            2.0 * np.pi / int(self.n_phi)
        )

    @property
    def z(self) -> np.ndarray:
        """Axial coordinate of each source row (m)."""

        return self.center[2] + np.linspace(
            -0.5 * self.length,
            0.5 * self.length,
            int(self.n_z),
        )

    def segments(self) -> tuple[Segment, ...]:
        """Return the azimuthal thin-wire source segments in grid order."""

        center = np.asarray(self.center, dtype=np.float64)
        half_step = np.pi / int(self.n_phi)
        segments: list[Segment] = []
        for phi in self.phi:
            phi_start = phi - half_step
            phi_end = phi + half_step
            for z in self.z:
                start = center + np.array(
                    [
                        self.radius * np.cos(phi_start),
                        self.radius * np.sin(phi_start),
                        z - center[2],
                    ]
                )
                end = center + np.array(
                    [
                        self.radius * np.cos(phi_end),
                        self.radius * np.sin(phi_end),
                        z - center[2],
                    ]
                )
                segments.append((start, end))
        return tuple(segments)


@dataclass(frozen=True)
class CylindricalGradientSystem:
    """A cylindrical source mesh and its field sensitivity matrix."""

    surface: CylindricalWindingSurface
    target_points: np.ndarray
    field_direction: np.ndarray
    sensitivity: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.target_points, dtype=np.float64)
        direction = np.asarray(self.field_direction, dtype=np.float64)
        sensitivity = np.asarray(self.sensitivity, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("target_points must have shape (n_points, 3)")
        if direction.shape != (3,) or np.linalg.norm(direction) == 0.0:
            raise ValueError("field_direction must be a nonzero 3-vector")
        expected = (points.shape[0], self.surface.segment_count)
        if sensitivity.shape != expected:
            raise ValueError(f"sensitivity must have shape {expected}")
        if not np.all(np.isfinite(points)):
            raise ValueError("target_points must be finite")
        if not np.all(np.isfinite(direction)):
            raise ValueError("field_direction must be finite")
        if not np.all(np.isfinite(sensitivity)):
            raise ValueError("sensitivity must be finite")


@dataclass(frozen=True)
class GradientCoilDesignResult:
    """Constrained current solution, stream function, and fit diagnostics."""

    system: CylindricalGradientSystem
    target_field_t: np.ndarray
    predicted_field_t: np.ndarray
    segment_currents_a: np.ndarray
    stream_function_a: np.ndarray
    regularization_t2_per_a2: float
    objective: float
    weighted_rms_error_t: float
    relative_rms_error: float
    normalized_max_error: float
    current_norm_a: float
    closure_error_a: float
    solver: str
    stop_code: int
    iterations: int

    @property
    def residual_t(self) -> np.ndarray:
        """Predicted minus target projected field (T)."""

        return self.predicted_field_t - self.target_field_t

    @property
    def stream_z(self) -> np.ndarray:
        """Axial coordinates of the stream-function cell edges (m)."""

        surface = self.system.surface
        return np.linspace(
            surface.center[2] - 0.5 * surface.length,
            surface.center[2] + 0.5 * surface.length,
            int(surface.n_z) + 1,
        )


@dataclass(frozen=True)
class GradientCoilRegularizationPath:
    """Current/error trade-off over a positive regularization grid."""

    regularizations_t2_per_a2: np.ndarray
    weighted_residual_norms_t: np.ndarray
    current_norms_a: np.ndarray
    l_curve_curvature: np.ndarray
    selected_index: int
    results: tuple[GradientCoilDesignResult, ...]

    @property
    def selected_regularization(self) -> float:
        """Regularization value at the discrete L-curve corner."""

        return float(self.regularizations_t2_per_a2[self.selected_index])

    @property
    def selected_result(self) -> GradientCoilDesignResult:
        """Coil design at the discrete L-curve corner."""

        return self.results[self.selected_index]


def spherical_target_points(
    radius: float,
    *,
    points_per_axis: int = 9,
    center: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return a Cartesian point grid clipped to a spherical target volume."""

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if int(points_per_axis) != points_per_axis or points_per_axis < 2:
        raise ValueError("points_per_axis must be an integer of at least 2")
    origin = np.asarray(center, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite 3-vector")
    axis = np.linspace(-radius, radius, int(points_per_axis))
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    inside = np.sum(offsets**2, axis=1) <= radius**2 * (1.0 + 1.0e-14)
    return offsets[inside] + origin


def linear_gradient_target(
    points: np.ndarray,
    gradient: Sequence[float],
    *,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    offset_t: float = 0.0,
) -> np.ndarray:
    """Return ``offset + gradient dot (position - center)`` in tesla.

    ``points`` and ``center`` are in metres and ``gradient`` is in T/m. For a
    conventional MRI y-gradient with the main field along z, use
    ``gradient=(0, Gy, 0)``.
    """

    positions = np.asarray(points, dtype=np.float64)
    gradient_vector = np.asarray(gradient, dtype=np.float64)
    origin = np.asarray(center, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("points must have shape (n_points, 3)")
    if gradient_vector.shape != (3,) or not np.all(np.isfinite(gradient_vector)):
        raise ValueError("gradient must be a finite 3-vector")
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("center must be a finite 3-vector")
    if not np.all(np.isfinite(positions)) or not np.isfinite(offset_t):
        raise ValueError("points and offset_t must be finite")
    return (positions - origin) @ gradient_vector + float(offset_t)


def build_cylindrical_gradient_system(
    surface: CylindricalWindingSurface,
    target_points: np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    chunk_size: int = 128,
) -> CylindricalGradientSystem:
    """Build the projected field-per-ampere matrix for a cylindrical mesh."""

    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("target_points must have shape (n_points, 3), n_points > 0")
    direction = np.asarray(field_direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.isfinite(direction_norm):
        raise ValueError("field_direction must be a finite 3-vector")
    if direction_norm == 0.0:
        raise ValueError("field_direction must be nonzero")
    direction = direction / direction_norm
    sensitivity = segment_field_sensitivity(
        points,
        surface.segments(),
        direction=direction,
        chunk_size=chunk_size,
    )
    return CylindricalGradientSystem(
        surface=surface,
        target_points=points,
        field_direction=direction,
        sensitivity=sensitivity,
    )


def solve_gradient_coil(
    system: CylindricalGradientSystem,
    target_field_t: np.ndarray,
    *,
    regularization: float = 0.0,
    field_weights: np.ndarray | None = None,
    solver: SolverName = "auto",
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    max_iterations: int | None = None,
) -> GradientCoilDesignResult:
    """Solve the KCL-constrained, Tikhonov-regularized coil design.

    The minimized objective is ``sum(w * (S I - b)**2) + alpha * sum(I**2)``.
    Consequently, ``regularization`` has units T^2/A^2. One current per
    azimuthal column is eliminated so ``sum_z I[phi, z] == 0`` by construction.
    """

    target = np.asarray(target_field_t, dtype=np.float64)
    n_points = system.target_points.shape[0]
    if target.shape != (n_points,) or not np.all(np.isfinite(target)):
        raise ValueError(f"target_field_t must be a finite array of shape ({n_points},)")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if solver not in {"auto", "scipy", "numpy"}:
        raise ValueError("solver must be 'auto', 'scipy', or 'numpy'")
    if atol <= 0.0 or btol <= 0.0:
        raise ValueError("atol and btol must be positive")
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    if field_weights is None:
        weights = np.ones(n_points, dtype=np.float64)
    else:
        weights = np.asarray(field_weights, dtype=np.float64)
        if weights.shape != (n_points,):
            raise ValueError(f"field_weights must have shape ({n_points},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("field_weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one field weight must be positive")

    currents_flat, solver_used, stop_code, iterations = _solve_reduced_system(
        system.sensitivity,
        target,
        weights,
        system.surface.shape,
        float(regularization),
        solver,
        float(atol),
        float(btol),
        max_iterations,
    )
    currents = currents_flat.reshape(system.surface.shape)
    predicted = system.sensitivity @ currents_flat
    residual = predicted - target
    weighted_rms = float(
        np.sqrt(np.sum(weights * residual**2) / np.sum(weights))
    )
    target_rms = float(np.sqrt(np.sum(weights * target**2) / np.sum(weights)))
    relative_rms = weighted_rms / target_rms if target_rms > 0.0 else np.inf
    target_peak = float(np.max(np.abs(target), initial=0.0))
    max_error = float(np.max(np.abs(residual), initial=0.0))
    normalized_max = max_error / target_peak if target_peak > 0.0 else np.inf
    current_norm = float(np.linalg.norm(currents_flat))
    closure_error = float(np.max(np.abs(np.sum(currents, axis=1)), initial=0.0))
    objective = float(
        np.sum(weights * residual**2) + regularization * current_norm**2
    )

    stream_function = np.zeros(
        (currents.shape[0], currents.shape[1] + 1), dtype=np.float64
    )
    stream_function[:, 1:] = -np.cumsum(currents, axis=1)
    return GradientCoilDesignResult(
        system=system,
        target_field_t=target,
        predicted_field_t=predicted,
        segment_currents_a=currents,
        stream_function_a=stream_function,
        regularization_t2_per_a2=float(regularization),
        objective=objective,
        weighted_rms_error_t=weighted_rms,
        relative_rms_error=float(relative_rms),
        normalized_max_error=float(normalized_max),
        current_norm_a=current_norm,
        closure_error_a=closure_error,
        solver=solver_used,
        stop_code=stop_code,
        iterations=iterations,
    )


def solve_regularization_path(
    system: CylindricalGradientSystem,
    target_field_t: np.ndarray,
    regularizations: Sequence[float],
    *,
    field_weights: np.ndarray | None = None,
    solver: SolverName = "auto",
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    max_iterations: int | None = None,
) -> GradientCoilRegularizationPath:
    """Solve a positive alpha grid and select its discrete L-curve corner.

    The expensive field-sensitivity system is reused for every candidate. The
    corner is the maximum curvature magnitude of
    ``(log residual norm, log current norm)`` parameterized by ``log(alpha)``;
    the two path endpoints are never selected.
    """

    values = np.asarray(regularizations, dtype=np.float64)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("regularizations must contain at least three values")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("regularizations must be finite and positive")
    values = np.unique(values)
    if values.size < 3:
        raise ValueError("regularizations must contain at least three unique values")
    values.sort()

    n_points = system.target_points.shape[0]
    if field_weights is None:
        weights = np.ones(n_points, dtype=np.float64)
    else:
        weights = np.asarray(field_weights, dtype=np.float64)
        if weights.shape != (n_points,):
            raise ValueError(f"field_weights must have shape ({n_points},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("field_weights must be finite and non-negative")
        if not np.any(weights > 0.0):
            raise ValueError("at least one field weight must be positive")

    results = tuple(
        solve_gradient_coil(
            system,
            target_field_t,
            regularization=float(value),
            field_weights=weights,
            solver=solver,
            atol=atol,
            btol=btol,
            max_iterations=max_iterations,
        )
        for value in values
    )
    weight_sum_sqrt = float(np.sqrt(np.sum(weights)))
    residual_norms = np.asarray(
        [result.weighted_rms_error_t * weight_sum_sqrt for result in results]
    )
    current_norms = np.asarray([result.current_norm_a for result in results])
    curvature = _l_curve_curvature(values, residual_norms, current_norms)
    selected_index = int(np.argmax(curvature))
    return GradientCoilRegularizationPath(
        regularizations_t2_per_a2=values,
        weighted_residual_norms_t=residual_norms,
        current_norms_a=current_norms,
        l_curve_curvature=curvature,
        selected_index=selected_index,
        results=results,
    )


def _l_curve_curvature(
    regularizations: np.ndarray,
    residual_norms: np.ndarray,
    current_norms: np.ndarray,
) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    parameter = np.log(np.maximum(regularizations, tiny))
    x = np.log(np.maximum(residual_norms, tiny))
    y = np.log(np.maximum(current_norms, tiny))
    dx = np.gradient(x, parameter, edge_order=2)
    dy = np.gradient(y, parameter, edge_order=2)
    ddx = np.gradient(dx, parameter, edge_order=2)
    ddy = np.gradient(dy, parameter, edge_order=2)
    denominator = np.maximum((dx**2 + dy**2) ** 1.5, tiny)
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    curvature[~np.isfinite(curvature)] = 0.0
    curvature[[0, -1]] = 0.0
    return curvature


def design_cylindrical_gradient_coil(
    surface: CylindricalWindingSurface,
    target_points: np.ndarray,
    target_field_t: np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    regularization: float = 0.0,
    field_weights: np.ndarray | None = None,
    chunk_size: int = 128,
    solver: SolverName = "auto",
    atol: float = 1.0e-10,
    btol: float = 1.0e-10,
    max_iterations: int | None = None,
) -> GradientCoilDesignResult:
    """Build and solve a cylindrical gradient-coil problem in one call."""

    system = build_cylindrical_gradient_system(
        surface,
        target_points,
        field_direction=field_direction,
        chunk_size=chunk_size,
    )
    return solve_gradient_coil(
        system,
        target_field_t,
        regularization=regularization,
        field_weights=field_weights,
        solver=solver,
        atol=atol,
        btol=btol,
        max_iterations=max_iterations,
    )


def _expand_closed_currents(
    free_currents: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    n_phi, n_z = shape
    free = np.asarray(free_currents, dtype=np.float64).reshape(n_phi, n_z - 1)
    currents = np.empty((n_phi, n_z), dtype=np.float64)
    currents[:, :-1] = free
    currents[:, -1] = -np.sum(free, axis=1)
    return currents.ravel()


def _closed_current_adjoint(
    currents: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    n_phi, n_z = shape
    values = np.asarray(currents, dtype=np.float64).reshape(n_phi, n_z)
    return (values[:, :-1] - values[:, -1, np.newaxis]).ravel()


def _closure_matrix(shape: tuple[int, int]) -> np.ndarray:
    n_phi, n_z = shape
    n_currents = n_phi * n_z
    n_free = n_phi * (n_z - 1)
    transform = np.zeros((n_currents, n_free), dtype=np.float64)
    for column in range(n_free):
        basis = np.zeros(n_free)
        basis[column] = 1.0
        transform[:, column] = _expand_closed_currents(basis, shape)
    return transform


def _solve_reduced_system(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    regularization: float,
    solver: SolverName,
    atol: float,
    btol: float,
    max_iterations: int | None,
) -> tuple[np.ndarray, str, int, int]:
    scipy_available = False
    linear_operator = None
    lsmr = None
    if solver in {"auto", "scipy"}:
        try:
            from scipy.sparse.linalg import LinearOperator, lsmr as scipy_lsmr

            linear_operator = LinearOperator
            lsmr = scipy_lsmr
            scipy_available = True
        except ImportError:
            if solver == "scipy":
                raise ImportError(
                    "SciPy is required for solver='scipy'; install the 'opt' extra"
                ) from None

    if scipy_available:
        assert linear_operator is not None
        assert lsmr is not None
        return _solve_with_scipy_lsmr(
            sensitivity,
            target,
            weights,
            shape,
            regularization,
            atol,
            btol,
            max_iterations,
            linear_operator,
            lsmr,
        )
    return _solve_with_numpy(
        sensitivity,
        target,
        weights,
        shape,
        regularization,
    )


def _solve_with_scipy_lsmr(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    regularization: float,
    atol: float,
    btol: float,
    max_iterations: int | None,
    linear_operator,
    lsmr,
) -> tuple[np.ndarray, str, int, int]:
    n_points, n_currents = sensitivity.shape
    n_free = shape[0] * (shape[1] - 1)
    sqrt_weights = np.sqrt(weights)
    sqrt_regularization = np.sqrt(regularization)
    regularized = regularization > 0.0
    row_count = n_points + (n_currents if regularized else 0)

    def matvec(free: np.ndarray) -> np.ndarray:
        currents = _expand_closed_currents(free, shape)
        field = sqrt_weights * (sensitivity @ currents)
        if not regularized:
            return field
        return np.concatenate([field, sqrt_regularization * currents])

    def rmatvec(values: np.ndarray) -> np.ndarray:
        field_values = values[:n_points]
        current_gradient = sensitivity.T @ (sqrt_weights * field_values)
        if regularized:
            current_gradient = (
                current_gradient + sqrt_regularization * values[n_points:]
            )
        return _closed_current_adjoint(current_gradient, shape)

    operator = linear_operator(
        (row_count, n_free),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    right_hand_side = sqrt_weights * target
    if regularized:
        right_hand_side = np.concatenate(
            [right_hand_side, np.zeros(n_currents, dtype=np.float64)]
        )
    solve = lsmr(
        operator,
        right_hand_side,
        atol=atol,
        btol=btol,
        conlim=0.0,
        maxiter=max_iterations,
    )
    currents = _expand_closed_currents(solve[0], shape)
    return currents, "scipy.sparse.linalg.lsmr", int(solve[1]), int(solve[2])


def _solve_with_numpy(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, int],
    regularization: float,
) -> tuple[np.ndarray, str, int, int]:
    n_free = shape[0] * (shape[1] - 1)
    if n_free > 1024:
        raise ImportError(
            "SciPy is required for gradient-coil systems with more than 1024 "
            "free currents; install the 'opt' extra"
        )
    transform = _closure_matrix(shape)
    sqrt_weights = np.sqrt(weights)
    design = sqrt_weights[:, np.newaxis] * (sensitivity @ transform)
    right_hand_side = sqrt_weights * target
    if regularization > 0.0:
        design = np.vstack([design, np.sqrt(regularization) * transform])
        right_hand_side = np.concatenate(
            [right_hand_side, np.zeros(transform.shape[0], dtype=np.float64)]
        )
    free, _, _, _ = np.linalg.lstsq(design, right_hand_side, rcond=None)
    currents = _expand_closed_currents(free, shape)
    return currents, "numpy.linalg.lstsq", 0, 0


def _expand_multi_closed_currents(
    free_currents: np.ndarray,
    shapes: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Expand KCL-reduced currents for several independent winding surfaces."""

    free = np.asarray(free_currents, dtype=np.float64)
    expanded: list[np.ndarray] = []
    offset = 0
    for shape in shapes:
        count = shape[0] * (shape[1] - 1)
        expanded.append(_expand_closed_currents(free[offset : offset + count], shape))
        offset += count
    if offset != free.size:
        raise ValueError("free current vector has the wrong size for surface shapes")
    return np.concatenate(expanded)


def _multi_closed_current_adjoint(
    currents: np.ndarray,
    shapes: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Adjoint of :func:`_expand_multi_closed_currents`."""

    values = np.asarray(currents, dtype=np.float64)
    reduced: list[np.ndarray] = []
    offset = 0
    for shape in shapes:
        count = shape[0] * shape[1]
        reduced.append(_closed_current_adjoint(values[offset : offset + count], shape))
        offset += count
    if offset != values.size:
        raise ValueError("current vector has the wrong size for surface shapes")
    return np.concatenate(reduced)


def _multi_closure_matrix(
    shapes: Sequence[tuple[int, int]],
) -> np.ndarray:
    n_currents = sum(shape[0] * shape[1] for shape in shapes)
    n_free = sum(shape[0] * (shape[1] - 1) for shape in shapes)
    transform = np.zeros((n_currents, n_free), dtype=np.float64)
    for column in range(n_free):
        basis = np.zeros(n_free)
        basis[column] = 1.0
        transform[:, column] = _expand_multi_closed_currents(basis, shapes)
    return transform


def _solve_multi_reduced_system(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shapes: Sequence[tuple[int, int]],
    regularization: float,
    current_weights: np.ndarray,
    solver: SolverName,
    atol: float,
    btol: float,
    max_iterations: int | None,
) -> tuple[np.ndarray, str, int, int]:
    """Solve a KCL-constrained inverse problem over multiple surfaces."""

    shape_tuple = tuple(shapes)
    if not shape_tuple:
        raise ValueError("at least one surface shape is required")
    n_currents = sum(shape[0] * shape[1] for shape in shape_tuple)
    penalty_weights = np.asarray(current_weights, dtype=np.float64)
    if penalty_weights.shape != (n_currents,):
        raise ValueError(f"current_weights must have shape ({n_currents},)")
    if not np.all(np.isfinite(penalty_weights)) or np.any(penalty_weights <= 0.0):
        raise ValueError("current_weights must be finite and positive")

    if solver in {"auto", "scipy"}:
        try:
            from scipy.sparse.linalg import LinearOperator, lsmr

            return _solve_multi_with_scipy_lsmr(
                sensitivity,
                target,
                weights,
                shape_tuple,
                regularization,
                penalty_weights,
                atol,
                btol,
                max_iterations,
                LinearOperator,
                lsmr,
            )
        except ImportError:
            if solver == "scipy":
                raise ImportError(
                    "SciPy is required for solver='scipy'; install the 'opt' extra"
                ) from None
    return _solve_multi_with_numpy(
        sensitivity,
        target,
        weights,
        shape_tuple,
        regularization,
        penalty_weights,
    )


def _solve_multi_with_scipy_lsmr(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shapes: Sequence[tuple[int, int]],
    regularization: float,
    current_weights: np.ndarray,
    atol: float,
    btol: float,
    max_iterations: int | None,
    linear_operator,
    lsmr,
) -> tuple[np.ndarray, str, int, int]:
    n_points, n_currents = sensitivity.shape
    n_free = sum(shape[0] * (shape[1] - 1) for shape in shapes)
    sqrt_weights = np.sqrt(weights)
    sqrt_penalty = np.sqrt(regularization * current_weights)
    regularized = regularization > 0.0
    row_count = n_points + (n_currents if regularized else 0)

    def matvec(free: np.ndarray) -> np.ndarray:
        currents = _expand_multi_closed_currents(free, shapes)
        field = sqrt_weights * (sensitivity @ currents)
        if not regularized:
            return field
        return np.concatenate([field, sqrt_penalty * currents])

    def rmatvec(values: np.ndarray) -> np.ndarray:
        gradient = sensitivity.T @ (sqrt_weights * values[:n_points])
        if regularized:
            gradient = gradient + sqrt_penalty * values[n_points:]
        return _multi_closed_current_adjoint(gradient, shapes)

    operator = linear_operator(
        (row_count, n_free),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )
    right_hand_side = sqrt_weights * target
    if regularized:
        right_hand_side = np.concatenate(
            [right_hand_side, np.zeros(n_currents, dtype=np.float64)]
        )
    solve = lsmr(
        operator,
        right_hand_side,
        atol=atol,
        btol=btol,
        conlim=0.0,
        maxiter=max_iterations,
    )
    currents = _expand_multi_closed_currents(solve[0], shapes)
    return currents, "scipy.sparse.linalg.lsmr", int(solve[1]), int(solve[2])


def _solve_multi_with_numpy(
    sensitivity: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    shapes: Sequence[tuple[int, int]],
    regularization: float,
    current_weights: np.ndarray,
) -> tuple[np.ndarray, str, int, int]:
    n_free = sum(shape[0] * (shape[1] - 1) for shape in shapes)
    if n_free > 1024:
        raise ImportError(
            "SciPy is required for gradient-coil systems with more than 1024 "
            "free currents; install the 'opt' extra"
        )
    transform = _multi_closure_matrix(shapes)
    sqrt_weights = np.sqrt(weights)
    design = sqrt_weights[:, np.newaxis] * (sensitivity @ transform)
    right_hand_side = sqrt_weights * target
    if regularization > 0.0:
        penalty = np.sqrt(regularization * current_weights)[:, np.newaxis]
        design = np.vstack([design, penalty * transform])
        right_hand_side = np.concatenate(
            [right_hand_side, np.zeros(transform.shape[0], dtype=np.float64)]
        )
    free, _, _, _ = np.linalg.lstsq(design, right_hand_side, rcond=None)
    currents = _expand_multi_closed_currents(free, shapes)
    return currents, "numpy.linalg.lstsq", 0, 0


__all__ = [
    "CylindricalWindingSurface",
    "CylindricalGradientSystem",
    "GradientCoilDesignResult",
    "GradientCoilRegularizationPath",
    "spherical_target_points",
    "linear_gradient_target",
    "build_cylindrical_gradient_system",
    "solve_gradient_coil",
    "solve_regularization_path",
    "design_cylindrical_gradient_coil",
]
