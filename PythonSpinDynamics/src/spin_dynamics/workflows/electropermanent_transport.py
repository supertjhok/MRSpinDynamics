"""Magnetophoretic nanoparticle transport in electropermanent-magnet fields.

The workflow converts a sampled vector field into ``grad(|B|^2)``, evaluates
either the linear-superparamagnetic or a smoothly saturating Langevin force,
and advances overdamped particles with Stokes drag, background flow, Brownian
diffusion, rectangular boundary handling, and irreversible target capture.

The particle object can represent one nanoparticle or an effective aggregate.
It is deliberately independent of the EPM geometry: any sampled vector field
with SI axes can drive the same transport calculation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.motion import advect_diffuse_positions

__all__ = [
    "MagneticForceMap2D",
    "MagnetophoreticTransportResult",
    "SuperparamagneticParticle",
    "magnetic_force_from_gradient",
    "magnetic_force_map_2d",
    "simulate_magnetophoretic_transport",
]

MU0 = 4.0e-7 * np.pi
BOLTZMANN = 1.380649e-23

MagnetizationModel = Literal["linear", "langevin"]
TransportBoundary = Literal["reflect", "periodic", "clip"]
TransportVelocity = (
    Sequence[float]
    | np.ndarray
    | Callable[[np.ndarray, float], np.ndarray]
    | None
)


def _strict_axis(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64)
    if (
        axis.ndim != 1
        or axis.size < 2
        or np.any(~np.isfinite(axis))
        or np.any(np.diff(axis) <= 0.0)
    ):
        raise ValueError(f"{name} must be finite, one-dimensional, and increasing")
    return axis


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class SuperparamagneticParticle:
    """Magnetic and hydrodynamic properties of a particle or aggregate.

    ``magnetic_core_radius_m`` is the radius of the equivalent sphere occupied
    by magnetic material before applying ``magnetic_volume_fraction``.
    ``hydrodynamic_radius_m`` controls Stokes drag and diffusion.  The
    susceptibility is the low-field SI volume susceptibility of the magnetic
    material and ``saturation_magnetization_a_m`` supplies the Langevin cap.
    """

    magnetic_core_radius_m: float
    hydrodynamic_radius_m: float
    volume_susceptibility: float
    saturation_magnetization_a_m: float
    fluid_viscosity_pa_s: float = 1.0e-3
    temperature_k: float = 310.0
    magnetic_volume_fraction: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        core = _positive(self.magnetic_core_radius_m, "magnetic_core_radius_m")
        hydro = _positive(self.hydrodynamic_radius_m, "hydrodynamic_radius_m")
        susceptibility = float(self.volume_susceptibility)
        if not np.isfinite(susceptibility) or susceptibility < 0.0:
            raise ValueError("volume_susceptibility must be finite and non-negative")
        saturation = _positive(
            self.saturation_magnetization_a_m,
            "saturation_magnetization_a_m",
        )
        viscosity = _positive(self.fluid_viscosity_pa_s, "fluid_viscosity_pa_s")
        temperature = _positive(self.temperature_k, "temperature_k")
        fraction = float(self.magnetic_volume_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("magnetic_volume_fraction must be in (0, 1]")
        object.__setattr__(self, "magnetic_core_radius_m", core)
        object.__setattr__(self, "hydrodynamic_radius_m", hydro)
        object.__setattr__(self, "volume_susceptibility", susceptibility)
        object.__setattr__(self, "saturation_magnetization_a_m", saturation)
        object.__setattr__(self, "fluid_viscosity_pa_s", viscosity)
        object.__setattr__(self, "temperature_k", temperature)
        object.__setattr__(self, "magnetic_volume_fraction", fraction)

    @property
    def magnetic_volume_m3(self) -> float:
        """Magnetic material volume in cubic metres."""

        return float(
            self.magnetic_volume_fraction
            * (4.0 / 3.0)
            * np.pi
            * self.magnetic_core_radius_m**3
        )

    @property
    def drag_coefficient_n_s_m(self) -> float:
        """Stokes drag coefficient ``6*pi*eta*r_h``."""

        return float(
            6.0
            * np.pi
            * self.fluid_viscosity_pa_s
            * self.hydrodynamic_radius_m
        )

    @property
    def diffusion_coefficient_m2_s(self) -> float:
        """Stokes--Einstein translational diffusion coefficient."""

        return float(
            BOLTZMANN * self.temperature_k / self.drag_coefficient_n_s_m
        )

    def magnetization_a_m(
        self,
        field_magnitude_t: float | np.ndarray,
        *,
        model: MagnetizationModel = "langevin",
    ) -> np.ndarray:
        """Return induced magnetization magnitude for ``|B|``.

        The Langevin argument is chosen as
        ``3*chi*|B|/(mu0*Ms)``.  It therefore has the exact low-field limit
        ``M = chi*|B|/mu0`` while approaching ``Ms`` smoothly.
        """

        field = np.asarray(field_magnitude_t, dtype=np.float64)
        if np.any(~np.isfinite(field)) or np.any(field < 0.0):
            raise ValueError("field_magnitude_t must be finite and non-negative")
        if model == "linear":
            return self.volume_susceptibility * field / MU0
        if model != "langevin":
            raise ValueError("model must be 'linear' or 'langevin'")
        alpha = (
            3.0
            * self.volume_susceptibility
            * field
            / (MU0 * self.saturation_magnetization_a_m)
        )
        small = np.abs(alpha) < 1.0e-4
        safe = np.where(small, 1.0, alpha)
        langevin = 1.0 / np.tanh(safe) - 1.0 / safe
        series = alpha / 3.0 - alpha**3 / 45.0 + 2.0 * alpha**5 / 945.0
        return self.saturation_magnetization_a_m * np.where(small, series, langevin)


def magnetic_force_from_gradient(
    particle: SuperparamagneticParticle,
    field_magnitude_t: float | np.ndarray,
    grad_b_squared_t2_m: np.ndarray,
    *,
    model: MagnetizationModel = "langevin",
) -> np.ndarray:
    """Return magnetic force from ``|B|`` and ``grad(|B|^2)``.

    In the linear limit this is exactly
    ``V*chi*grad(|B|^2)/(2*mu0)``.  The Langevin path replaces ``chi*B/mu0``
    by the saturating magnetization and evaluates ``V*M*grad(B)``.
    """

    field = np.asarray(field_magnitude_t, dtype=np.float64)
    gradient = np.asarray(grad_b_squared_t2_m, dtype=np.float64)
    if gradient.shape != field.shape + (2,) or np.any(~np.isfinite(gradient)):
        raise ValueError("grad_b_squared_t2_m must have shape field.shape + (2,)")
    if model == "linear":
        coefficient = (
            particle.magnetic_volume_m3
            * particle.volume_susceptibility
            / (2.0 * MU0)
        )
        return coefficient * gradient
    magnetization = particle.magnetization_a_m(field, model=model)
    coefficient = np.empty_like(field)
    nonzero = field > np.finfo(np.float64).tiny
    coefficient[nonzero] = (
        particle.magnetic_volume_m3
        * magnetization[nonzero]
        / (2.0 * field[nonzero])
    )
    coefficient[~nonzero] = (
        particle.magnetic_volume_m3
        * particle.volume_susceptibility
        / (2.0 * MU0)
    )
    return coefficient[..., np.newaxis] * gradient


@dataclass(frozen=True)
class MagneticForceMap2D:
    """Sampled vector field and ``grad(|B|^2)`` on an ``(x, y)`` plane."""

    x_m: np.ndarray
    y_m: np.ndarray
    field_t: np.ndarray
    grad_b_squared_t2_m: np.ndarray

    def __post_init__(self) -> None:
        x_axis = _strict_axis(self.x_m, "x_m")
        y_axis = _strict_axis(self.y_m, "y_m")
        field = np.asarray(self.field_t, dtype=np.float64)
        gradient = np.asarray(self.grad_b_squared_t2_m, dtype=np.float64)
        shape = (y_axis.size, x_axis.size)
        if field.shape != shape + (3,) or np.any(~np.isfinite(field)):
            raise ValueError("field_t must be finite with shape (len(y_m), len(x_m), 3)")
        if gradient.shape != shape + (2,) or np.any(~np.isfinite(gradient)):
            raise ValueError(
                "grad_b_squared_t2_m must be finite with shape "
                "(len(y_m), len(x_m), 2)"
            )
        object.__setattr__(self, "x_m", x_axis)
        object.__setattr__(self, "y_m", y_axis)
        object.__setattr__(self, "field_t", field)
        object.__setattr__(self, "grad_b_squared_t2_m", gradient)

    @property
    def field_magnitude_t(self) -> np.ndarray:
        """Field magnitude at map nodes."""

        return np.linalg.norm(self.field_t, axis=-1)

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Rectangular ``(x, y)`` limits used by the motion engine."""

        return (
            (float(self.x_m[0]), float(self.x_m[-1])),
            (float(self.y_m[0]), float(self.y_m[-1])),
        )

    def sample(self, positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Bilinearly sample ``(|B|, grad(|B|^2))`` at particle positions."""

        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != 2 or np.any(~np.isfinite(positions)):
            raise ValueError("positions_m must have shape (n_particles, 2) and be finite")
        magnitude = _bilinear_sample(
            self.field_magnitude_t,
            self.x_m,
            self.y_m,
            positions,
        )
        gradient = np.column_stack(
            [
                _bilinear_sample(
                    self.grad_b_squared_t2_m[..., component],
                    self.x_m,
                    self.y_m,
                    positions,
                )
                for component in range(2)
            ]
        )
        return magnitude, gradient

    def magnetic_force(
        self,
        particle: SuperparamagneticParticle,
        positions_m: np.ndarray,
        *,
        model: MagnetizationModel = "langevin",
    ) -> np.ndarray:
        """Return particle forces at arbitrary in-plane positions."""

        magnitude, gradient = self.sample(positions_m)
        return magnetic_force_from_gradient(
            particle,
            magnitude,
            gradient,
            model=model,
        )


def magnetic_force_map_2d(
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    field_t: np.ndarray,
) -> MagneticForceMap2D:
    """Construct a force map by differentiating a sampled vector field."""

    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    field = np.asarray(field_t, dtype=np.float64)
    if field.shape != (y_axis.size, x_axis.size, 3) or np.any(~np.isfinite(field)):
        raise ValueError("field_t must be finite with shape (len(y_m), len(x_m), 3)")
    b_squared = np.sum(field**2, axis=-1)
    derivative_y, derivative_x = np.gradient(
        b_squared,
        y_axis,
        x_axis,
        edge_order=2 if min(x_axis.size, y_axis.size) >= 3 else 1,
    )
    gradient = np.stack((derivative_x, derivative_y), axis=-1)
    return MagneticForceMap2D(x_axis, y_axis, field, gradient)


@dataclass(frozen=True)
class MagnetophoreticTransportResult:
    """Particle histories and capture diagnostics for one transport burst."""

    time_s: np.ndarray
    positions_m: np.ndarray
    magnetic_force_n: np.ndarray
    magnetic_velocity_m_s: np.ndarray
    captured: np.ndarray
    capture_time_s: np.ndarray
    target_center_m: np.ndarray
    target_radius_m: float
    particle: SuperparamagneticParticle
    force_map: MagneticForceMap2D
    magnetization_model: MagnetizationModel

    @property
    def capture_fraction(self) -> float:
        """Fraction of particles captured by the end of the burst."""

        return float(np.mean(self.captured))

    @property
    def peak_force_n(self) -> float:
        """Peak magnetic-force magnitude across time and particles."""

        return float(np.max(np.linalg.norm(self.magnetic_force_n, axis=-1)))

    @property
    def median_force_n(self) -> float:
        """Median magnetic-force magnitude across time and particles."""

        return float(np.median(np.linalg.norm(self.magnetic_force_n, axis=-1)))

    @property
    def peak_magnetic_speed_m_s(self) -> float:
        """Peak magnetophoretic drift speed before background flow."""

        return float(np.max(np.linalg.norm(self.magnetic_velocity_m_s, axis=-1)))

    @property
    def cumulative_capture_fraction(self) -> np.ndarray:
        """Captured fraction at every saved time."""

        return np.asarray(
            [np.mean(self.capture_time_s <= time) for time in self.time_s],
            dtype=np.float64,
        )


def simulate_magnetophoretic_transport(
    force_map: MagneticForceMap2D,
    particle: SuperparamagneticParticle,
    initial_positions_m: np.ndarray,
    *,
    duration_s: float,
    time_step_s: float,
    target_center_m: Sequence[float],
    target_radius_m: float,
    background_velocity_m_s: TransportVelocity = None,
    magnetization_model: MagnetizationModel = "langevin",
    boundary: TransportBoundary = "reflect",
    initial_captured: Sequence[bool] | np.ndarray | None = None,
    seed: int | None = None,
) -> MagnetophoreticTransportResult:
    """Advance overdamped particles and irreversibly capture target entries."""

    positions = np.asarray(initial_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or np.any(~np.isfinite(positions)):
        raise ValueError("initial_positions_m must have shape (n_particles, 2)")
    if positions.shape[0] < 1:
        raise ValueError("initial_positions_m must contain at least one particle")
    duration = _positive(duration_s, "duration_s")
    requested_step = _positive(time_step_s, "time_step_s")
    center = np.asarray(target_center_m, dtype=np.float64)
    if center.shape != (2,) or np.any(~np.isfinite(center)):
        raise ValueError("target_center_m must be a finite 2-vector")
    radius = _positive(target_radius_m, "target_radius_m")
    if boundary not in {"reflect", "periodic", "clip"}:
        raise ValueError("boundary must be 'reflect', 'periodic', or 'clip'")
    if magnetization_model not in {"linear", "langevin"}:
        raise ValueError("magnetization_model must be 'linear' or 'langevin'")

    step_count = int(np.ceil(duration / requested_step))
    time = np.linspace(0.0, duration, step_count + 1)
    dt = float(time[1] - time[0])
    count = positions.shape[0]
    history = np.empty((step_count + 1, count, 2), dtype=np.float64)
    forces = np.empty_like(history)
    velocities = np.empty_like(history)
    capture_time = np.full(count, np.inf, dtype=np.float64)
    if initial_captured is None:
        captured = np.zeros(count, dtype=bool)
    else:
        captured = np.asarray(initial_captured, dtype=bool)
        if captured.shape != (count,):
            raise ValueError("initial_captured must have one value per particle")
        captured = captured.copy()
    captured |= np.linalg.norm(positions - center, axis=1) <= radius
    capture_time[captured] = 0.0
    history[0] = positions
    rng = np.random.default_rng(seed)

    for step in range(step_count + 1):
        force = force_map.magnetic_force(
            particle,
            positions,
            model=magnetization_model,
        )
        magnetic_velocity = force / particle.drag_coefficient_n_s_m
        force[captured] = 0.0
        magnetic_velocity[captured] = 0.0
        forces[step] = force
        velocities[step] = magnetic_velocity
        if step == step_count:
            break

        background = _velocity_array(background_velocity_m_s, positions, time[step])
        total_velocity = magnetic_velocity + background
        total_velocity[captured] = 0.0
        updated = positions.copy()
        active = ~captured
        if np.any(active):
            updated[active] = advect_diffuse_positions(
                positions[active],
                dt,
                velocity=total_velocity[active],
                diffusion_coefficient=particle.diffusion_coefficient_m2_s,
                rng=rng,
                time=float(time[step]),
                bounds=force_map.bounds,
                boundary=boundary,
            )
        entry_fraction = _circle_entry_fraction(positions, updated, center, radius)
        newly_captured = active & np.isfinite(entry_fraction)
        capture_time[newly_captured] = (
            time[step] + entry_fraction[newly_captured] * dt
        )
        updated[newly_captured] = positions[newly_captured] + entry_fraction[
            newly_captured, np.newaxis
        ] * (updated[newly_captured] - positions[newly_captured])
        captured |= newly_captured
        positions = updated
        history[step + 1] = positions

    return MagnetophoreticTransportResult(
        time_s=time,
        positions_m=history,
        magnetic_force_n=forces,
        magnetic_velocity_m_s=velocities,
        captured=captured,
        capture_time_s=capture_time,
        target_center_m=center,
        target_radius_m=radius,
        particle=particle,
        force_map=force_map,
        magnetization_model=magnetization_model,
    )


def _velocity_array(
    velocity: TransportVelocity,
    positions: np.ndarray,
    time_s: float,
) -> np.ndarray:
    if velocity is None:
        return np.zeros_like(positions)
    values = velocity(positions, time_s) if callable(velocity) else velocity
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (2,):
        array = np.broadcast_to(array, positions.shape)
    if array.shape != positions.shape or np.any(~np.isfinite(array)):
        raise ValueError(
            "background_velocity_m_s must be a 2-vector, match positions, "
            "or be a callable returning either"
        )
    return np.asarray(array)


def _circle_entry_fraction(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Return first segment/circle entry in ``[0, 1]``, or infinity."""

    offset = start - center
    step = end - start
    a = np.sum(step * step, axis=1)
    b = 2.0 * np.sum(offset * step, axis=1)
    c = np.sum(offset * offset, axis=1) - radius**2
    result = np.full(start.shape[0], np.inf)
    result[c <= 0.0] = 0.0
    moving = a > 0.0
    discriminant = b * b - 4.0 * a * c
    intersects = moving & (discriminant >= 0.0) & (c > 0.0)
    root = np.sqrt(np.maximum(discriminant, 0.0))
    entry = np.divide(
        -b - root,
        2.0 * a,
        out=np.full_like(a, np.inf),
        where=moving,
    )
    valid = intersects & (entry >= 0.0) & (entry <= 1.0)
    result[valid] = entry[valid]
    return result


def _bilinear_sample(
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    x = np.clip(positions[:, 0], x_axis[0], x_axis[-1])
    y = np.clip(positions[:, 1], y_axis[0], y_axis[-1])
    ix = np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, x_axis.size - 2)
    iy = np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, y_axis.size - 2)
    tx = (x - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
    ty = (y - y_axis[iy]) / (y_axis[iy + 1] - y_axis[iy])
    return (
        (1.0 - tx) * (1.0 - ty) * values[iy, ix]
        + tx * (1.0 - ty) * values[iy, ix + 1]
        + (1.0 - tx) * ty * values[iy + 1, ix]
        + tx * ty * values[iy + 1, ix + 1]
    )
