"""Engineering metrics and workflow adapters for realized gradient windings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.coil_properties import (
    ANNEALED_COPPER,
    ConductorMaterial,
)
from spin_dynamics.fields.gradient_windings import (
    ActivelyShieldedWinding,
    GradientWinding,
    WindingContour,
)
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, MU0, biot_savart
from spin_dynamics.fields.quasistatic import (
    mutual_inductance,
    self_inductance_circular,
)

Winding = GradientWinding | ActivelyShieldedWinding
BackgroundField = (
    np.ndarray | Sequence[float] | Callable[[np.ndarray], np.ndarray]
)


@dataclass(frozen=True)
class GradientFieldMetrics:
    """Efficiency, fitted gradient, target fidelity, and exterior leakage."""

    field_direction: np.ndarray
    gradient_t_per_m: np.ndarray
    efficiency_t_per_m_per_a: np.ndarray
    offset_t: float
    target_relative_rms_error: float | None
    target_correlation: float | None
    shield_rms_field_t: float | None
    shield_peak_field_t: float | None


@dataclass(frozen=True)
class GradientElectricalMetrics:
    """Low-frequency series-equivalent winding properties."""

    contour_count: int
    wire_length_m: float
    reference_current_a: float
    dc_resistance_ohm: float
    estimated_inductance_h: float
    ohmic_power_w: float
    stored_energy_j: float
    voltage_per_current_slew_h: float


@dataclass(frozen=True)
class GradientMechanicalMetrics:
    """Lorentz force and torque in a specified background field."""

    net_force_n: np.ndarray
    net_torque_nm: np.ndarray
    contour_forces_n: np.ndarray
    contour_torques_nm: np.ndarray
    peak_segment_force_n: float


@dataclass(frozen=True)
class GradientCoilEngineeringMetrics:
    """Field, electrical, and mechanical metrics for one realized winding."""

    field: GradientFieldMetrics
    electrical: GradientElectricalMetrics
    mechanical: GradientMechanicalMetrics


@dataclass(frozen=True)
class GradientImagingFieldMap:
    """Sampled gradient field in tesla and imaging angular-frequency units."""

    axes: tuple[np.ndarray, ...]
    cartesian_axes: tuple[int, ...]
    projected_field_t: np.ndarray
    angular_offset_rad_s: np.ndarray

    def to_motion_field_maps(
        self,
        *,
        b1_tx_map: np.ndarray | None = None,
        b1_rx_map: np.ndarray | None = None,
    ):
        """Return the existing :class:`MotionFieldMaps` workflow container."""

        from spin_dynamics.motion import make_motion_field_maps

        return make_motion_field_maps(
            self.axes,
            b0_map=self.angular_offset_rad_s,
            b1_tx_map=b1_tx_map,
            b1_rx_map=b1_rx_map,
        )


def winding_field(
    points: np.ndarray,
    winding: Winding,
    *,
    current_scale: float = 1.0,
) -> np.ndarray:
    """Evaluate the vector field of one or two realized contour surfaces."""

    positions = np.asarray(points, dtype=np.float64)
    if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
        raise ValueError("points must be finite with shape (..., 3)")
    if not np.isfinite(current_scale):
        raise ValueError("current_scale must be finite")
    total = np.zeros_like(positions)
    for winding_set in _winding_sets(winding):
        total += biot_savart(
            positions,
            winding_set.segments,
            current=float(current_scale) * winding_set.current_per_turn_a,
        )
    return total


def winding_force_torque(
    winding: Winding,
    background_field: BackgroundField,
    *,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    current_scale: float = 1.0,
) -> GradientMechanicalMetrics:
    """Integrate ``I dl cross B`` force and moment over every contour segment."""

    reference = np.asarray(origin, dtype=np.float64)
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("origin must be a finite 3-vector")
    contour_forces: list[np.ndarray] = []
    contour_torques: list[np.ndarray] = []
    peak_force = 0.0
    for winding_set in _winding_sets(winding):
        current = float(current_scale) * winding_set.current_per_turn_a
        for contour in winding_set.contours:
            starts = contour.points[:-1]
            ends = contour.points[1:]
            midpoints = 0.5 * (starts + ends)
            field = _background_values(background_field, midpoints)
            forces = current * np.cross(ends - starts, field)
            torques = np.cross(midpoints - reference, forces)
            contour_forces.append(np.sum(forces, axis=0))
            contour_torques.append(np.sum(torques, axis=0))
            peak_force = max(
                peak_force,
                float(np.max(np.linalg.norm(forces, axis=1), initial=0.0)),
            )
    forces_array = np.asarray(contour_forces, dtype=np.float64).reshape(-1, 3)
    torques_array = np.asarray(contour_torques, dtype=np.float64).reshape(-1, 3)
    return GradientMechanicalMetrics(
        net_force_n=np.sum(forces_array, axis=0),
        net_torque_nm=np.sum(torques_array, axis=0),
        contour_forces_n=forces_array,
        contour_torques_nm=torques_array,
        peak_segment_force_n=peak_force,
    )


def estimate_gradient_electrical_metrics(
    winding: Winding,
    *,
    wire_radius: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    temperature: float | None = None,
) -> GradientElectricalMetrics:
    """Estimate series-equivalent DC resistance and filamentary inductance.

    Contours are treated as series-connected with their extracted orientation.
    Self inductance uses an equal-perimeter circular-loop approximation; mutual
    inductance uses the existing Neumann/vector-potential calculation on the
    actual 3-D paths.  Use :func:`winding_peec_conductors` for detailed AC/PEEC
    extraction after choosing physical crossover and terminal routing.
    """

    if not np.isfinite(wire_radius) or wire_radius <= 0.0:
        raise ValueError("wire_radius must be finite and positive")
    sets = _winding_sets(winding)
    contours: list[WindingContour] = []
    currents: list[float] = []
    for winding_set in sets:
        contours.extend(winding_set.contours)
        currents.extend(
            [winding_set.current_per_turn_a] * len(winding_set.contours)
        )
    if not contours:
        raise ValueError("winding must contain at least one contour")
    current_array = np.asarray(currents, dtype=np.float64)
    lengths = np.asarray(
        [np.sum(np.linalg.norm(np.diff(c.points, axis=0), axis=1)) for c in contours]
    )
    resistivity = material.resistivity_at(temperature)
    resistances = resistivity * lengths / (np.pi * wire_radius**2)
    reference_current = float(np.max(np.abs(current_array)))
    power = float(np.sum(resistances * current_array**2))
    equivalent_resistance = power / reference_current**2

    inductance_matrix = np.zeros((len(contours), len(contours)))
    for first, contour in enumerate(contours):
        equivalent_radius = float(lengths[first] / (2.0 * np.pi))
        if equivalent_radius > wire_radius:
            self_l = self_inductance_circular(equivalent_radius, wire_radius)
        else:
            ratio = max(2.0 * lengths[first] / wire_radius, 1.0 + 1.0e-12)
            self_l = MU0 * lengths[first] / (2.0 * np.pi) * max(
                np.log(ratio) - 1.0,
                1.0e-6,
            )
        inductance_matrix[first, first] = self_l
        first_segments = contour.segments()
        for second in range(first + 1, len(contours)):
            mutual_forward = mutual_inductance(
                first_segments,
                contours[second].segments(),
            )
            mutual_reverse = mutual_inductance(
                contours[second].segments(),
                first_segments,
            )
            mutual = 0.5 * (mutual_forward + mutual_reverse)
            inductance_matrix[first, second] = mutual
            inductance_matrix[second, first] = mutual
    energy = max(
        0.5 * float(current_array @ inductance_matrix @ current_array),
        0.0,
    )
    equivalent_inductance = 2.0 * energy / reference_current**2
    return GradientElectricalMetrics(
        contour_count=len(contours),
        wire_length_m=float(np.sum(lengths)),
        reference_current_a=reference_current,
        dc_resistance_ohm=float(equivalent_resistance),
        estimated_inductance_h=float(equivalent_inductance),
        ohmic_power_w=power,
        stored_energy_j=energy,
        voltage_per_current_slew_h=float(equivalent_inductance),
    )


def gradient_coil_engineering_metrics(
    winding: Winding,
    target_points: np.ndarray,
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    target_field_t: np.ndarray | None = None,
    shield_points: np.ndarray | None = None,
    wire_radius: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    temperature: float | None = None,
    background_field: BackgroundField = (0.0, 0.0, 0.0),
    force_origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> GradientCoilEngineeringMetrics:
    """Extract field, low-frequency electrical, and mechanical metrics."""

    points = _point_array(target_points, "target_points")
    direction = _unit_vector(field_direction, "field_direction")
    projected = winding_field(points, winding) @ direction
    design = np.column_stack([points, np.ones(points.shape[0])])
    fit, _, _, _ = np.linalg.lstsq(design, projected, rcond=None)
    reference_current = max(
        winding_set.current_per_turn_a for winding_set in _winding_sets(winding)
    )
    relative_error: float | None = None
    correlation: float | None = None
    if target_field_t is not None:
        target = np.asarray(target_field_t, dtype=np.float64)
        if target.shape != (points.shape[0],) or not np.all(np.isfinite(target)):
            raise ValueError("target_field_t must match target_points")
        residual_rms = float(np.sqrt(np.mean((projected - target) ** 2)))
        target_rms = float(np.sqrt(np.mean(target**2)))
        relative_error = residual_rms / target_rms if target_rms > 0.0 else np.inf
        if np.std(target) > 0.0 and np.std(projected) > 0.0:
            correlation = float(np.corrcoef(target, projected)[0, 1])
    shield_rms: float | None = None
    shield_peak: float | None = None
    if shield_points is not None:
        exterior = _point_array(shield_points, "shield_points")
        exterior_field = winding_field(exterior, winding) @ direction
        shield_rms = float(np.sqrt(np.mean(exterior_field**2)))
        shield_peak = float(np.max(np.abs(exterior_field), initial=0.0))
    field_metrics = GradientFieldMetrics(
        field_direction=direction,
        gradient_t_per_m=fit[:3],
        efficiency_t_per_m_per_a=fit[:3] / reference_current,
        offset_t=float(fit[3]),
        target_relative_rms_error=relative_error,
        target_correlation=correlation,
        shield_rms_field_t=shield_rms,
        shield_peak_field_t=shield_peak,
    )
    return GradientCoilEngineeringMetrics(
        field=field_metrics,
        electrical=estimate_gradient_electrical_metrics(
            winding,
            wire_radius=wire_radius,
            material=material,
            temperature=temperature,
        ),
        mechanical=winding_force_torque(
            winding,
            background_field,
            origin=force_origin,
        ),
    )


def winding_peec_conductors(
    winding: Winding,
    *,
    wire_radius: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    temperature: float | None = None,
    n_radial: int = 6,
    n_angular: int = 8,
):
    """Return one existing PEEC :class:`Conductor` per closed contour.

    Each contour is opened at its first sample to define PEEC terminals.  The
    tuple preserves contour orientation.  Series crossovers are intentionally
    not invented; add them to a routed path before treating all loops as one
    manufacturable conductor.
    """

    from spin_dynamics.fields.coil_peec import Conductor

    return tuple(
        Conductor(
            path_points=contour.points,
            wire_radius=float(wire_radius),
            material=material,
            temperature=temperature,
            n_radial=int(n_radial),
            n_angular=int(n_angular),
        )
        for winding_set in _winding_sets(winding)
        for contour in winding_set.contours
    )


def winding_to_gradient_driver(
    winding: Winding,
    eddy_modes,
    *,
    tau_rl: float,
    **driver_options,
):
    """Build the existing eddy/pre-emphasis driver from realized windings."""

    current_values = [
        winding_set.current_per_turn_a for winding_set in _winding_sets(winding)
    ]
    if not np.allclose(current_values, current_values[0]):
        raise ValueError("eddy adapter requires a common current per turn")
    segments = tuple(
        segment
        for winding_set in _winding_sets(winding)
        for segment in winding_set.segments
    )
    return eddy_modes.to_gradient_driver(
        segments,
        tau_rl=tau_rl,
        **driver_options,
    )


def winding_imaging_field_map(
    winding: Winding,
    axes: Sequence[Sequence[float] | np.ndarray],
    *,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    cartesian_axes: Sequence[int] | None = None,
    gamma_rad_s_t: float = GAMMA_PROTON,
    current_scale: float = 1.0,
) -> GradientImagingFieldMap:
    """Sample a winding into the angular off-resonance maps imaging consumes.

    By default a one-dimensional map samples physical ``z``, a two-dimensional
    map samples ``(x, z)`` to match the existing imaging and motion convention,
    and a three-dimensional map samples ``(x, y, z)``.  ``cartesian_axes`` can
    override that mapping with unique component indices from 0 through 2.
    """

    axis_tuple = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    if not 1 <= len(axis_tuple) <= 3:
        raise ValueError("axes must contain one, two, or three coordinate arrays")
    for axis in axis_tuple:
        if axis.ndim != 1 or axis.size < 2 or np.any(np.diff(axis) <= 0.0):
            raise ValueError("each coordinate axis must be strictly increasing")
    if not np.isfinite(gamma_rad_s_t):
        raise ValueError("gamma_rad_s_t must be finite")
    default_cartesian_axes = {1: (2,), 2: (0, 2), 3: (0, 1, 2)}
    coordinate_indices = (
        default_cartesian_axes[len(axis_tuple)]
        if cartesian_axes is None
        else tuple(int(value) for value in cartesian_axes)
    )
    if (
        len(coordinate_indices) != len(axis_tuple)
        or len(set(coordinate_indices)) != len(coordinate_indices)
        or any(value not in (0, 1, 2) for value in coordinate_indices)
    ):
        raise ValueError(
            "cartesian_axes must contain one unique Cartesian index per axis"
        )
    mesh = np.meshgrid(*axis_tuple, indexing="ij")
    coordinates = [np.zeros(mesh[0].size) for _ in range(3)]
    for component, physical_axis in zip(mesh, coordinate_indices, strict=True):
        coordinates[physical_axis] = component.ravel()
    points = np.column_stack(coordinates)
    direction = _unit_vector(field_direction, "field_direction")
    field = winding_field(points, winding, current_scale=current_scale) @ direction
    shape = tuple(axis.size for axis in axis_tuple)
    projected = field.reshape(shape)
    return GradientImagingFieldMap(
        axes=axis_tuple,
        cartesian_axes=coordinate_indices,
        projected_field_t=projected,
        angular_offset_rad_s=float(gamma_rad_s_t) * projected,
    )


def _winding_sets(winding: Winding) -> tuple[GradientWinding, ...]:
    if isinstance(winding, GradientWinding):
        return (winding,)
    if isinstance(winding, ActivelyShieldedWinding):
        return winding.primary, winding.shield
    raise TypeError("winding must be GradientWinding or ActivelyShieldedWinding")


def _background_values(field: BackgroundField, points: np.ndarray) -> np.ndarray:
    values = field(points) if callable(field) else np.asarray(field, dtype=np.float64)
    result = np.asarray(values, dtype=np.float64)
    if result.shape == (3,):
        result = np.broadcast_to(result, points.shape)
    if result.shape != points.shape or not np.all(np.isfinite(result)):
        raise ValueError("background_field must return shape (n_segments, 3)")
    return result


def _point_array(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError(f"{name} must have shape (n_points, 3), n_points >= 4")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be finite")
    return points


def _unit_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


__all__ = [
    "GradientFieldMetrics",
    "GradientElectricalMetrics",
    "GradientMechanicalMetrics",
    "GradientCoilEngineeringMetrics",
    "GradientImagingFieldMap",
    "winding_field",
    "winding_force_torque",
    "estimate_gradient_electrical_metrics",
    "gradient_coil_engineering_metrics",
    "winding_peec_conductors",
    "winding_to_gradient_driver",
    "winding_imaging_field_map",
]
