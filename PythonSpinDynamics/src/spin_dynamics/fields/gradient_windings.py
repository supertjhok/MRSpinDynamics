"""Convert cylindrical stream functions into realizable winding contours."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.gradient_coils import (
    CylindricalWindingSurface,
    GradientCoilDesignResult,
)
from spin_dynamics.fields.gradient_shielding import (
    ActivelyShieldedGradientDesignResult,
)

Segment = tuple[np.ndarray, np.ndarray]
NodeKey = tuple[int, int]


@dataclass(frozen=True)
class WindingContour:
    """One oriented stream-function contour on a cylindrical surface."""

    level_a: float
    phi_z: np.ndarray
    points: np.ndarray
    closed: bool

    def __post_init__(self) -> None:
        phi_z = np.asarray(self.phi_z, dtype=np.float64)
        points = np.asarray(self.points, dtype=np.float64)
        if phi_z.ndim != 2 or phi_z.shape[1] != 2:
            raise ValueError("phi_z must have shape (n_points, 2)")
        if points.shape != (phi_z.shape[0], 3):
            raise ValueError("points must have shape (n_points, 3)")
        if phi_z.shape[0] < 2:
            raise ValueError("a winding contour must contain at least two points")
        if not np.all(np.isfinite(phi_z)) or not np.all(np.isfinite(points)):
            raise ValueError("contour coordinates must be finite")

    def segments(self) -> tuple[Segment, ...]:
        """Return the contour in the package's straight-segment format."""

        return tuple(
            (self.points[index], self.points[index + 1])
            for index in range(self.points.shape[0] - 1)
        )


@dataclass(frozen=True)
class GradientWinding:
    """Realized contours on one surface carrying a common turn current."""

    surface: CylindricalWindingSurface
    contours: tuple[WindingContour, ...]
    current_per_turn_a: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_per_turn_a) or self.current_per_turn_a <= 0.0:
            raise ValueError("current_per_turn_a must be finite and positive")

    @property
    def segments(self) -> tuple[Segment, ...]:
        return winding_segments(self.contours)


@dataclass(frozen=True)
class ActivelyShieldedWinding:
    """Primary and shield contour sets from a joint active-shield solve."""

    primary: GradientWinding
    shield: GradientWinding

    @property
    def segments(self) -> tuple[Segment, ...]:
        """All oriented segments for unit common current drive."""

        if not np.isclose(
            self.primary.current_per_turn_a,
            self.shield.current_per_turn_a,
        ):
            raise ValueError("combined segments require equal primary/shield turn currents")
        return self.primary.segments + self.shield.segments

    @property
    def current_per_turn_a(self) -> float:
        if not np.isclose(
            self.primary.current_per_turn_a,
            self.shield.current_per_turn_a,
        ):
            raise ValueError("primary and shield turn currents differ")
        return self.primary.current_per_turn_a


def winding_contour_levels(
    stream_function_a: np.ndarray,
    current_per_turn_a: float,
) -> np.ndarray:
    """Return half-step-centered contour levels separated by one turn current."""

    values = np.asarray(stream_function_a, dtype=np.float64)
    if values.ndim != 2 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("stream_function_a must be a finite 2-D array")
    if not np.isfinite(current_per_turn_a) or current_per_turn_a <= 0.0:
        raise ValueError("current_per_turn_a must be finite and positive")
    half_step = 0.5 * float(current_per_turn_a)
    positive = np.arange(
        half_step,
        float(np.max(values)) + half_step * 0.5,
        float(current_per_turn_a),
    )
    negative = -np.arange(
        half_step,
        -float(np.min(values)) + half_step * 0.5,
        float(current_per_turn_a),
    )
    return np.sort(np.concatenate([negative, positive]))


def extract_winding_contours(
    result: GradientCoilDesignResult,
    *,
    current_per_turn_a: float,
    levels_a: Sequence[float] | None = None,
    require_closed: bool = True,
) -> tuple[WindingContour, ...]:
    """Extract oriented 3-D windings from a solved cylindrical stream function.

    When ``levels_a`` is omitted, levels are separated by
    ``current_per_turn_a`` and offset by half a step from zero. The periodic
    azimuthal seam is stitched before the contours are mapped to 3-D. With the
    exact KCL constraint and nonzero levels, contours should be closed; set
    ``require_closed=False`` to inspect an under-resolved exceptional case.
    """

    if levels_a is None:
        levels = winding_contour_levels(
            result.stream_function_a, current_per_turn_a
        )
    else:
        if not np.isfinite(current_per_turn_a) or current_per_turn_a <= 0.0:
            raise ValueError("current_per_turn_a must be finite and positive")
        levels = np.asarray(levels_a, dtype=np.float64)
    return stream_function_contours(
        result.system.surface,
        result.stream_function_a,
        levels,
        z_coordinates=result.stream_z,
        require_closed=require_closed,
    )


def winding_from_design(
    result: GradientCoilDesignResult,
    *,
    current_per_turn_a: float,
    levels_a: Sequence[float] | None = None,
    require_closed: bool = True,
) -> GradientWinding:
    """Return a one-surface winding object from a gradient design result."""

    return GradientWinding(
        surface=result.system.surface,
        contours=extract_winding_contours(
            result,
            current_per_turn_a=current_per_turn_a,
            levels_a=levels_a,
            require_closed=require_closed,
        ),
        current_per_turn_a=float(current_per_turn_a),
    )


def extract_actively_shielded_winding(
    result: ActivelyShieldedGradientDesignResult,
    *,
    current_per_turn_a: float,
    require_closed: bool = True,
) -> ActivelyShieldedWinding:
    """Extract primary and shield contours at one common current per turn."""

    current = float(current_per_turn_a)
    primary_levels = winding_contour_levels(
        result.primary_stream_function_a,
        current,
    )
    shield_levels = winding_contour_levels(
        result.shield_stream_function_a,
        current,
    )
    primary = GradientWinding(
        surface=result.system.primary_surface,
        contours=stream_function_contours(
            result.system.primary_surface,
            result.primary_stream_function_a,
            primary_levels,
            z_coordinates=result.primary_stream_z,
            require_closed=require_closed,
        ),
        current_per_turn_a=current,
    )
    shield = GradientWinding(
        surface=result.system.shield_surface,
        contours=stream_function_contours(
            result.system.shield_surface,
            result.shield_stream_function_a,
            shield_levels,
            z_coordinates=result.shield_stream_z,
            require_closed=require_closed,
        ),
        current_per_turn_a=current,
    )
    return ActivelyShieldedWinding(primary=primary, shield=shield)


def stream_function_contours(
    surface: CylindricalWindingSurface,
    stream_function_a: np.ndarray,
    levels_a: Sequence[float],
    *,
    z_coordinates: np.ndarray | None = None,
    require_closed: bool = True,
) -> tuple[WindingContour, ...]:
    """Contour a periodic cylindrical stream-function grid without Matplotlib."""

    values = np.asarray(stream_function_a, dtype=np.float64)
    expected_rows = int(surface.n_phi)
    if values.ndim != 2 or values.shape[0] != expected_rows:
        raise ValueError(
            f"stream_function_a must have shape ({expected_rows}, n_z_values)"
        )
    if values.shape[1] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("stream_function_a must be finite with at least two z values")
    levels = np.asarray(levels_a, dtype=np.float64)
    if levels.ndim != 1 or not np.all(np.isfinite(levels)):
        raise ValueError("levels_a must be a finite one-dimensional sequence")
    levels = np.unique(levels)
    if z_coordinates is None:
        z_axis = np.linspace(
            surface.center[2] - 0.5 * surface.length,
            surface.center[2] + 0.5 * surface.length,
            values.shape[1],
        )
    else:
        z_axis = np.asarray(z_coordinates, dtype=np.float64)
        if z_axis.shape != (values.shape[1],):
            raise ValueError(
                f"z_coordinates must have shape ({values.shape[1]},)"
            )
        if not np.all(np.isfinite(z_axis)) or np.any(np.diff(z_axis) <= 0.0):
            raise ValueError("z_coordinates must be finite and strictly increasing")

    phi_axis = surface.phi
    phi_periodic = np.concatenate([phi_axis, [phi_axis[0] + 2.0 * np.pi]])
    values_periodic = np.vstack([values, values[0:1]])
    dphi = 2.0 * np.pi / int(surface.n_phi)
    dz = float(np.min(np.diff(z_axis)))
    all_contours: list[WindingContour] = []
    for requested_level in levels:
        level = float(requested_level)
        if np.any(values_periodic == level):
            level = float(np.nextafter(level, np.inf))
        raw_segments = _marching_segments(
            phi_periodic,
            z_axis,
            values_periodic,
            level,
        )
        paths = _stitch_periodic_segments(raw_segments, dphi=dphi, dz=dz)
        for path, closed in paths:
            oriented = _orient_path(path, phi_axis, z_axis, values, surface.radius)
            points = _cylindrical_points(surface, oriented)
            contour = WindingContour(
                level_a=float(requested_level),
                phi_z=oriented,
                points=points,
                closed=closed,
            )
            if require_closed and not contour.closed:
                raise RuntimeError(
                    "stream-function contour is open; refine the source mesh or "
                    "set require_closed=False"
                )
            all_contours.append(contour)
    return tuple(all_contours)


def winding_segments(contours: Sequence[WindingContour]) -> tuple[Segment, ...]:
    """Flatten several winding contours into straight source segments."""

    return tuple(segment for contour in contours for segment in contour.segments())


def _marching_segments(
    phi_axis: np.ndarray,
    z_axis: np.ndarray,
    values: np.ndarray,
    level: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for phi_index in range(phi_axis.size - 1):
        for z_index in range(z_axis.size - 1):
            corners = np.array(
                [
                    [phi_axis[phi_index], z_axis[z_index]],
                    [phi_axis[phi_index + 1], z_axis[z_index]],
                    [phi_axis[phi_index + 1], z_axis[z_index + 1]],
                    [phi_axis[phi_index], z_axis[z_index + 1]],
                ]
            )
            corner_values = np.array(
                [
                    values[phi_index, z_index],
                    values[phi_index + 1, z_index],
                    values[phi_index + 1, z_index + 1],
                    values[phi_index, z_index + 1],
                ]
            )
            intersections: dict[int, np.ndarray] = {}
            for edge, (first, second) in enumerate(
                ((0, 1), (1, 2), (2, 3), (3, 0))
            ):
                if (corner_values[first] > level) == (
                    corner_values[second] > level
                ):
                    continue
                fraction = (level - corner_values[first]) / (
                    corner_values[second] - corner_values[first]
                )
                intersections[edge] = corners[first] + fraction * (
                    corners[second] - corners[first]
                )
            if len(intersections) == 2:
                first, second = intersections.values()
                segments.append((first, second))
            elif len(intersections) == 4:
                center_above = float(np.mean(corner_values)) > level
                corner_above = corner_values[0] > level
                pairs = (
                    ((0, 1), (2, 3))
                    if center_above == corner_above
                    else ((0, 3), (1, 2))
                )
                segments.extend(
                    (intersections[first], intersections[second])
                    for first, second in pairs
                )
    return segments


def _stitch_periodic_segments(
    segments: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    dphi: float,
    dz: float,
) -> list[tuple[np.ndarray, bool]]:
    if not segments:
        return []
    phi_tolerance = max(abs(dphi) * 1.0e-8, 1.0e-12)
    z_tolerance = max(abs(dz) * 1.0e-8, 1.0e-12)
    coordinates: dict[NodeKey, np.ndarray] = {}
    adjacency: dict[NodeKey, set[NodeKey]] = defaultdict(set)

    def key(point: np.ndarray) -> NodeKey:
        phi = float(np.mod(point[0], 2.0 * np.pi))
        if abs(phi - 2.0 * np.pi) <= phi_tolerance:
            phi = 0.0
        node = (
            int(round(phi / phi_tolerance)),
            int(round(float(point[1]) / z_tolerance)),
        )
        coordinates.setdefault(node, np.array([phi, float(point[1])]))
        return node

    edges: set[tuple[NodeKey, NodeKey]] = set()
    for first_point, second_point in segments:
        first = key(first_point)
        second = key(second_point)
        if first == second:
            continue
        edge = tuple(sorted((first, second)))
        edges.add(edge)
        adjacency[first].add(second)
        adjacency[second].add(first)

    unused = set(edges)
    paths: list[tuple[np.ndarray, bool]] = []

    def trace(start: NodeKey) -> tuple[np.ndarray, bool]:
        nodes = [start]
        current = start
        while True:
            candidates = [
                neighbor
                for neighbor in adjacency[current]
                if tuple(sorted((current, neighbor))) in unused
            ]
            if not candidates:
                break
            neighbor = sorted(candidates)[0]
            unused.remove(tuple(sorted((current, neighbor))))
            nodes.append(neighbor)
            current = neighbor
            if current == start:
                break
        closed = len(nodes) > 2 and nodes[-1] == start
        return np.asarray([coordinates[node] for node in nodes]), closed

    endpoints = sorted(node for node, neighbors in adjacency.items() if len(neighbors) == 1)
    for endpoint in endpoints:
        if any(tuple(sorted((endpoint, other))) in unused for other in adjacency[endpoint]):
            paths.append(trace(endpoint))
    while unused:
        start = min(unused)[0]
        paths.append(trace(start))
    return paths


def _orient_path(
    path: np.ndarray,
    phi_axis: np.ndarray,
    z_axis: np.ndarray,
    values: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Orient a contour along the surface current ``K = (-dpsi/dz, dpsi/(R dphi))``.

    The current direction is evaluated per segment from the gradient of the
    bilinear stream-function interpolant in the grid cell holding the segment's
    midpoint. On a contour that gradient is normal to the segment and nonzero,
    so every segment votes with weight ``|dl| |grad psi|`` and the sign is
    unambiguous. (Sampling the gradient at the nearest grid node instead gave
    small closed contours around one node a constant vote vector, whose sum over
    a closed loop is zero, so their orientation was decided by round-off.)
    """

    if path.shape[0] < 2:
        return path
    n_phi = phi_axis.size
    dphi = 2.0 * np.pi / n_phi
    alignment = 0.0
    for first, second in zip(path[:-1], path[1:], strict=True):
        delta_phi = (second[0] - first[0] + np.pi) % (2.0 * np.pi) - np.pi
        delta_z = second[1] - first[1]
        midpoint_phi = (first[0] + 0.5 * delta_phi - phi_axis[0]) % (2.0 * np.pi)
        midpoint_z = first[1] + 0.5 * delta_z
        phi_cell = min(int(np.floor(midpoint_phi / dphi)), n_phi - 1)
        s = midpoint_phi / dphi - phi_cell
        z_cell = int(np.clip(np.searchsorted(z_axis, midpoint_z) - 1, 0, z_axis.size - 2))
        dz = float(z_axis[z_cell + 1] - z_axis[z_cell])
        t = (midpoint_z - z_axis[z_cell]) / dz
        next_phi = (phi_cell + 1) % n_phi
        v00 = values[phi_cell, z_cell]
        v10 = values[next_phi, z_cell]
        v01 = values[phi_cell, z_cell + 1]
        v11 = values[next_phi, z_cell + 1]
        dpsi_dphi = ((1.0 - t) * (v10 - v00) + t * (v11 - v01)) / dphi
        dpsi_dz = ((1.0 - s) * (v01 - v00) + s * (v11 - v10)) / dz
        desired_phi = -dpsi_dz
        desired_z = dpsi_dphi / radius
        alignment += radius * delta_phi * desired_phi + delta_z * desired_z
    return path if alignment >= 0.0 else path[::-1].copy()


def _cylindrical_points(
    surface: CylindricalWindingSurface,
    phi_z: np.ndarray,
) -> np.ndarray:
    center = np.asarray(surface.center, dtype=np.float64)
    return np.column_stack(
        [
            center[0] + surface.radius * np.cos(phi_z[:, 0]),
            center[1] + surface.radius * np.sin(phi_z[:, 0]),
            phi_z[:, 1],
        ]
    )


__all__ = [
    "WindingContour",
    "GradientWinding",
    "ActivelyShieldedWinding",
    "winding_contour_levels",
    "extract_winding_contours",
    "winding_from_design",
    "extract_actively_shielded_winding",
    "stream_function_contours",
    "winding_segments",
]
