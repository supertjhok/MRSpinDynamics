"""Dynamic-inversion trapping of ferromagnetic particles.

The workflow implements the polarize-delay-antialigned-gradient sequence of
Nacev et al., Nano Letters 15 (2015), 359-364. Unlike the induced-moment
transport workflow, the magnetic moment has its own orientation and can remain
opposed to an applied gradient for a finite time. Mechanical rotation, thermal
rotation, optional internal magnetic relaxation, anisotropic rod drag, flow,
and translational diffusion are integrated explicitly.

The external gradient source is represented by a regularized inverse-power
field centered outside the region of interest. It is an analytical trap model,
not a calibrated reproduction of the paper's coils. The paper reports coil
geometry, current, and timing but not the field maps required for that claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "DynamicInversionHardwareAssessment",
    "DynamicInversionHardwareConfig",
    "DynamicInversionResult",
    "DynamicInversionSequence",
    "DynamicInversionStability",
    "FerromagneticParticle",
    "assess_dynamic_inversion_hardware",
    "nacev_2015_sequence",
    "simulate_dynamic_inversion",
]

MU0 = 4.0e-7 * np.pi
BOLTZMANN = 1.380649e-23

ParticleShape = Literal["sphere", "rod"]
HardwareArchitecture = Literal["coils", "epm", "hybrid"]


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _unit_directions(values: Sequence[Sequence[float]]) -> np.ndarray:
    directions = np.asarray(values, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 2:
        raise ValueError("directions must have shape (n, 2)")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(~np.isfinite(directions)) or np.any(norms <= 0.0):
        raise ValueError("directions must be finite and nonzero")
    return directions / norms[:, np.newaxis]


@dataclass(frozen=True)
class FerromagneticParticle:
    """Rigid magnetic sphere or blunt-ended cylindrical rod in a fluid.

    Rod diffusion follows the Tirado-Martinez-de la Torre finite-cylinder
    expressions. Their fitted end corrections were validated for aspect ratios
    from roughly 2 through 30; higher aspect ratios use their slender-cylinder
    asymptote and should be treated as an extrapolation.

    ``internal_relaxation_time_s=None`` locks the magnetic moment to the body.
    A finite value allows the moment to rotate toward an applied field without
    waiting for the particle body. This is useful for demonstrating why rapidly
    relaxing superparamagnetic moments do not support the Nacev mechanism, but
    it is not a microscopic Neel-relaxation model.
    """

    shape: ParticleShape
    length_m: float
    diameter_m: float
    volume_susceptibility: float
    saturation_magnetization_a_m: float
    remanent_magnetization_a_m: float | None = None
    fluid_viscosity_pa_s: float = 1.0e-3
    temperature_k: float = 310.0
    magnetic_volume_fraction: float = 1.0
    internal_relaxation_time_s: float | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.shape not in {"sphere", "rod"}:
            raise ValueError("shape must be 'sphere' or 'rod'")
        length = _positive(self.length_m, "length_m")
        diameter = _positive(self.diameter_m, "diameter_m")
        if self.shape == "sphere" and not np.isclose(
            length, diameter, rtol=1e-12, atol=0.0
        ):
            raise ValueError("sphere length_m and diameter_m must be equal")
        if self.shape == "rod" and length / diameter < 2.0:
            raise ValueError("rod aspect ratio must be at least 2")
        susceptibility = float(self.volume_susceptibility)
        if not np.isfinite(susceptibility) or susceptibility <= 0.0:
            raise ValueError("volume_susceptibility must be finite and positive")
        _positive(
            self.saturation_magnetization_a_m,
            "saturation_magnetization_a_m",
        )
        if self.remanent_magnetization_a_m is not None:
            remanence = _positive(
                self.remanent_magnetization_a_m,
                "remanent_magnetization_a_m",
            )
            if remanence > self.saturation_magnetization_a_m:
                raise ValueError(
                    "remanent_magnetization_a_m cannot exceed saturation"
                )
        _positive(self.fluid_viscosity_pa_s, "fluid_viscosity_pa_s")
        _positive(self.temperature_k, "temperature_k")
        fraction = float(self.magnetic_volume_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("magnetic_volume_fraction must be in (0, 1]")
        if self.internal_relaxation_time_s is not None:
            _positive(
                self.internal_relaxation_time_s,
                "internal_relaxation_time_s",
            )

    @property
    def aspect_ratio(self) -> float:
        """Length divided by diameter."""

        return float(self.length_m / self.diameter_m)

    @property
    def volume_m3(self) -> float:
        """Magnetic material volume."""

        if self.shape == "sphere":
            geometric = np.pi * self.diameter_m**3 / 6.0
        else:
            geometric = np.pi * self.diameter_m**2 * self.length_m / 4.0
        return float(self.magnetic_volume_fraction * geometric)

    @property
    def translational_drag_parallel_n_s_m(self) -> float:
        """Drag parallel to the body axis."""

        eta = self.fluid_viscosity_pa_s
        if self.shape == "sphere":
            return float(3.0 * np.pi * eta * self.diameter_m)
        p = self.aspect_ratio
        correction = -0.207 + 0.980 / p - 0.133 / p**2
        return float(2.0 * np.pi * eta * self.length_m / (np.log(p) + correction))

    @property
    def translational_drag_perpendicular_n_s_m(self) -> float:
        """Drag perpendicular to the body axis."""

        eta = self.fluid_viscosity_pa_s
        if self.shape == "sphere":
            return float(3.0 * np.pi * eta * self.diameter_m)
        p = self.aspect_ratio
        correction = 0.839 + 0.185 / p + 0.233 / p**2
        return float(4.0 * np.pi * eta * self.length_m / (np.log(p) + correction))

    @property
    def rotational_drag_n_m_s(self) -> float:
        """Tumbling drag about an axis perpendicular to the body axis."""

        eta = self.fluid_viscosity_pa_s
        if self.shape == "sphere":
            return float(np.pi * eta * self.diameter_m**3)
        p = self.aspect_ratio
        correction = -0.662 + 0.917 / p - 0.050 / p**2
        return float(
            np.pi * eta * self.length_m**3
            / (3.0 * (np.log(p) + correction))
        )

    @property
    def rotational_diffusion_rad2_s(self) -> float:
        """Rotational Einstein diffusion coefficient."""

        return float(BOLTZMANN * self.temperature_k / self.rotational_drag_n_m_s)

    @property
    def brownian_orientation_correlation_time_s(self) -> float:
        """Free 2-D director correlation time ``1/(2 D_r)``."""

        return float(1.0 / (2.0 * self.rotational_diffusion_rad2_s))

    @property
    def translational_diffusion_parallel_m2_s(self) -> float:
        return float(
            BOLTZMANN
            * self.temperature_k
            / self.translational_drag_parallel_n_s_m
        )

    @property
    def translational_diffusion_perpendicular_m2_s(self) -> float:
        return float(
            BOLTZMANN
            * self.temperature_k
            / self.translational_drag_perpendicular_n_s_m
        )

    def polarized_magnetization_a_m(self, field_t: float) -> float:
        """Retained magnetization initialized by a polarizing field.

        The low-field relation is capped at the supplied saturation value. The
        resulting moment is retained between sequence segments; this is the
        ferromagnetic assumption that differs from induced SPION transport.
        """

        field = _positive(field_t, "field_t")
        if self.remanent_magnetization_a_m is not None:
            return float(self.remanent_magnetization_a_m)
        linear = self.volume_susceptibility * field / MU0
        return float(min(linear, self.saturation_magnetization_a_m))

    def magnetic_moment_a_m2(self, polarizing_field_t: float) -> float:
        return float(self.volume_m3 * self.polarized_magnetization_a_m(polarizing_field_t))

    def mechanical_alignment_time_s(
        self,
        field_t: float,
        *,
        polarizing_field_t: float | None = None,
    ) -> float:
        """Small-angle viscous alignment time ``zeta_r/(m B)``."""

        field = _positive(field_t, "field_t")
        polarizer = field if polarizing_field_t is None else polarizing_field_t
        moment = self.magnetic_moment_a_m2(polarizer)
        return float(self.rotational_drag_n_m_s / (moment * field))

    def orientation_memory_time_s(
        self,
        field_t: float,
        *,
        polarizing_field_t: float | None = None,
    ) -> float:
        """Combined mechanical/internal characteristic alignment time."""

        mechanical = self.mechanical_alignment_time_s(
            field_t,
            polarizing_field_t=polarizing_field_t,
        )
        if self.internal_relaxation_time_s is None:
            return mechanical
        return float(
            1.0 / (1.0 / mechanical + 1.0 / self.internal_relaxation_time_s)
        )


@dataclass(frozen=True)
class DynamicInversionSequence:
    """Polarize-delay-gradient sequence and analytical source geometry."""

    polarizing_field_t: float
    gradient_field_at_center_t: float
    actuator_radius_m: float
    polarizing_duration_s: float = 600e-6
    delay_s: float = 5e-6
    gradient_duration_s: float = 50e-6
    element_period_s: float = 60.6e-3
    directions: tuple[tuple[float, float], ...] = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    field_decay_exponent: float = 3.0
    polarizing_substeps: int = 12
    gradient_substeps: int = 10
    free_substeps: int = 2
    provenance: str = ""

    def __post_init__(self) -> None:
        for name in (
            "polarizing_field_t",
            "gradient_field_at_center_t",
            "actuator_radius_m",
            "polarizing_duration_s",
            "gradient_duration_s",
            "element_period_s",
            "field_decay_exponent",
        ):
            _positive(getattr(self, name), name)
        _positive(self.delay_s, "delay_s", allow_zero=True)
        occupied = (
            self.polarizing_duration_s + self.delay_s + self.gradient_duration_s
        )
        if self.element_period_s < occupied:
            raise ValueError("element_period_s must contain all pulse segments")
        _unit_directions(self.directions)
        for name in ("polarizing_substeps", "gradient_substeps", "free_substeps"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def center_gradient_t_m(self) -> float:
        """Gradient magnitude at the trap center for the inverse-power source."""

        return float(
            self.field_decay_exponent
            * self.gradient_field_at_center_t
            / self.actuator_radius_m
        )

    @property
    def full_cycle_period_s(self) -> float:
        return float(len(self.directions) * self.element_period_s)

    @property
    def active_duty_cycle(self) -> float:
        return float(
            (self.polarizing_duration_s + self.gradient_duration_s)
            / self.element_period_s
        )


def nacev_2015_sequence(
    *,
    polarizing_field_t: float = 50e-3,
    gradient_field_at_center_t: float = 10e-3,
    actuator_radius_m: float = 25e-3,
) -> DynamicInversionSequence:
    """Return the reported 2015 timing with explicit inferred field inputs.

    The 600 us / 5 us / 50 us timing and 60.6 ms element period are reported.
    The field magnitudes and effective source radius are caller-visible
    assumptions because the article does not publish calibrated field maps.
    """

    return DynamicInversionSequence(
        polarizing_field_t=polarizing_field_t,
        gradient_field_at_center_t=gradient_field_at_center_t,
        actuator_radius_m=actuator_radius_m,
        provenance=(
            "Nacev et al. Nano Letters 15 (2015) 359-364 timing; "
            "field magnitude and inverse-power source radius inferred"
        ),
    )


@dataclass(frozen=True)
class DynamicInversionStability:
    """Concentration and retention metrics without irreversible capture."""

    target_radius_m: float
    initial_rms_radius_m: float
    final_rms_radius_m: float
    concentration_gain: float
    initial_target_fraction: float
    final_target_fraction: float
    retained_target_fraction: float
    escaped_target_fraction: float
    repulsive_gradient_fraction: float
    log_contraction_rate_s: float

    @property
    def contracts(self) -> bool:
        return self.final_rms_radius_m < self.initial_rms_radius_m


@dataclass(frozen=True)
class DynamicInversionResult:
    """Saved trajectories, orientations, and trap-stability diagnostics."""

    sequence: DynamicInversionSequence
    particle: FerromagneticParticle
    time_s: np.ndarray
    positions_m: np.ndarray
    body_angles_rad: np.ndarray
    moment_angles_rad: np.ndarray
    element_count: int
    max_radius_m: np.ndarray
    stability: DynamicInversionStability

    @property
    def final_positions_m(self) -> np.ndarray:
        return self.positions_m[-1]

    @property
    def final_body_angles_rad(self) -> np.ndarray:
        return self.body_angles_rad[-1]


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _apply_internal_relaxation(
    moment_angle: np.ndarray,
    field_angle: np.ndarray,
    duration_s: float,
    relaxation_time_s: float | None,
) -> np.ndarray:
    if relaxation_time_s is None or duration_s == 0.0:
        return moment_angle
    delta = _wrap_angle(moment_angle - field_angle)
    factor = np.exp(-duration_s / relaxation_time_s)
    relaxed = 2.0 * np.arctan(np.tan(0.5 * delta) * factor)
    return _wrap_angle(field_angle + relaxed)


def _mobilized_velocity(
    force_n: np.ndarray,
    body_angle: np.ndarray,
    particle: FerromagneticParticle,
) -> np.ndarray:
    axis = np.column_stack((np.cos(body_angle), np.sin(body_angle)))
    parallel_force = np.sum(force_n * axis, axis=1)
    parallel = parallel_force[:, np.newaxis] * axis
    perpendicular = force_n - parallel
    return (
        parallel / particle.translational_drag_parallel_n_s_m
        + perpendicular / particle.translational_drag_perpendicular_n_s_m
    )


def _thermal_displacement(
    rng: np.random.Generator,
    body_angle: np.ndarray,
    duration_s: float,
    particle: FerromagneticParticle,
) -> np.ndarray:
    if duration_s <= 0.0:
        return np.zeros((body_angle.size, 2))
    axis = np.column_stack((np.cos(body_angle), np.sin(body_angle)))
    normal = np.column_stack((-axis[:, 1], axis[:, 0]))
    parallel = rng.normal(size=body_angle.size) * np.sqrt(
        2.0 * particle.translational_diffusion_parallel_m2_s * duration_s
    )
    perpendicular = rng.normal(size=body_angle.size) * np.sqrt(
        2.0 * particle.translational_diffusion_perpendicular_m2_s * duration_s
    )
    return parallel[:, np.newaxis] * axis + perpendicular[:, np.newaxis] * normal


def _reflect_positions(positions: np.ndarray, bounds_m: np.ndarray | None) -> None:
    if bounds_m is None:
        return
    for axis in range(2):
        lower, upper = bounds_m[axis]
        width = upper - lower
        shifted = (positions[:, axis] - lower) % (2.0 * width)
        positions[:, axis] = lower + np.where(
            shifted <= width,
            shifted,
            2.0 * width - shifted,
        )


def _advance_uniform_field(
    positions: np.ndarray,
    body_angle: np.ndarray,
    moment_angle: np.ndarray,
    field_vector_t: np.ndarray,
    duration_s: float,
    substeps: int,
    particle: FerromagneticParticle,
    moment_a_m2: float,
    background_velocity_m_s: np.ndarray,
    rng: np.random.Generator,
    bounds_m: np.ndarray | None,
    brownian: bool,
) -> tuple[np.ndarray, np.ndarray]:
    dt = duration_s / substeps
    field_angle = np.full(body_angle.shape, np.arctan2(field_vector_t[1], field_vector_t[0]))
    field_magnitude = float(np.linalg.norm(field_vector_t))
    for _ in range(substeps):
        torque = moment_a_m2 * field_magnitude * np.sin(field_angle - moment_angle)
        mechanical = torque * dt / particle.rotational_drag_n_m_s
        if brownian:
            mechanical += rng.normal(size=body_angle.size) * np.sqrt(
                2.0 * particle.rotational_diffusion_rad2_s * dt
            )
        body_angle = _wrap_angle(body_angle + mechanical)
        moment_angle = _wrap_angle(moment_angle + mechanical)
        moment_angle = _apply_internal_relaxation(
            moment_angle,
            field_angle,
            dt,
            particle.internal_relaxation_time_s,
        )
        positions += background_velocity_m_s * dt
        if brownian:
            positions += _thermal_displacement(rng, body_angle, dt, particle)
        _reflect_positions(positions, bounds_m)
    return body_angle, moment_angle


def _advance_free(
    positions: np.ndarray,
    body_angle: np.ndarray,
    moment_angle: np.ndarray,
    duration_s: float,
    substeps: int,
    particle: FerromagneticParticle,
    background_velocity_m_s: np.ndarray,
    rng: np.random.Generator,
    bounds_m: np.ndarray | None,
    brownian: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0.0:
        return body_angle, moment_angle
    dt = duration_s / substeps
    for _ in range(substeps):
        rotation = np.zeros(body_angle.size)
        if brownian:
            rotation = rng.normal(size=body_angle.size) * np.sqrt(
                2.0 * particle.rotational_diffusion_rad2_s * dt
            )
        body_angle = _wrap_angle(body_angle + rotation)
        moment_angle = _wrap_angle(moment_angle + rotation)
        positions += background_velocity_m_s * dt
        if brownian:
            positions += _thermal_displacement(rng, body_angle, dt, particle)
        _reflect_positions(positions, bounds_m)
    return body_angle, moment_angle


def _advance_gradient(
    positions: np.ndarray,
    body_angle: np.ndarray,
    moment_angle: np.ndarray,
    direction: np.ndarray,
    sequence: DynamicInversionSequence,
    particle: FerromagneticParticle,
    moment_a_m2: float,
    background_velocity_m_s: np.ndarray,
    rng: np.random.Generator,
    bounds_m: np.ndarray | None,
    brownian: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    dt = sequence.gradient_duration_s / sequence.gradient_substeps
    source = -sequence.actuator_radius_m * direction
    exponent = sequence.field_decay_exponent
    coefficient = (
        sequence.gradient_field_at_center_t
        * sequence.actuator_radius_m**exponent
    )
    repulsive = 0
    samples = 0
    for _ in range(sequence.gradient_substeps):
        offset = positions - source
        radius = np.linalg.norm(offset, axis=1)
        floor = 0.05 * sequence.actuator_radius_m
        radius = np.maximum(radius, floor)
        scalar_field = -coefficient / radius**exponent
        field = scalar_field[:, np.newaxis] * direction
        field_magnitude = np.abs(scalar_field)
        field_angle = np.arctan2(field[:, 1], field[:, 0])
        moment_axis = np.column_stack((np.cos(moment_angle), np.sin(moment_angle)))
        alignment = moment_axis @ direction
        gradient_scalar = (
            exponent
            * coefficient
            * offset
            / radius[:, np.newaxis] ** (exponent + 2.0)
        )
        force = moment_a_m2 * alignment[:, np.newaxis] * gradient_scalar
        velocity = _mobilized_velocity(force, body_angle, particle)
        torque = moment_a_m2 * field_magnitude * np.sin(field_angle - moment_angle)
        mechanical = torque * dt / particle.rotational_drag_n_m_s
        if brownian:
            mechanical += rng.normal(size=body_angle.size) * np.sqrt(
                2.0 * particle.rotational_diffusion_rad2_s * dt
            )
        body_angle = _wrap_angle(body_angle + mechanical)
        moment_angle = _wrap_angle(moment_angle + mechanical)
        moment_angle = _apply_internal_relaxation(
            moment_angle,
            field_angle,
            dt,
            particle.internal_relaxation_time_s,
        )
        positions += (velocity + background_velocity_m_s) * dt
        if brownian:
            positions += _thermal_displacement(rng, body_angle, dt, particle)
        _reflect_positions(positions, bounds_m)
        repulsive += int(np.count_nonzero(alignment > 0.0))
        samples += alignment.size
    return body_angle, moment_angle, repulsive, samples


def _stability_metrics(
    time_s: np.ndarray,
    positions_m: np.ndarray,
    max_radius_m: np.ndarray,
    target_radius_m: float,
    repulsive_fraction: float,
) -> DynamicInversionStability:
    radii = np.linalg.norm(positions_m, axis=-1)
    rms = np.sqrt(np.mean(radii**2, axis=1))
    initial_inside = radii[0] <= target_radius_m
    final_inside = radii[-1] <= target_radius_m
    if np.any(initial_inside):
        retained = float(np.mean(final_inside[initial_inside]))
        escaped = float(np.mean(max_radius_m[initial_inside] > target_radius_m))
    else:
        retained = float("nan")
        escaped = float("nan")
    usable = min(time_s.size, max(3, time_s.size // 2 + 1))
    positive = np.maximum(rms[:usable], np.finfo(float).tiny)
    if usable >= 2 and time_s[usable - 1] > time_s[0]:
        contraction = float(np.polyfit(time_s[:usable], np.log(positive), 1)[0])
    else:
        contraction = 0.0
    return DynamicInversionStability(
        target_radius_m=target_radius_m,
        initial_rms_radius_m=float(rms[0]),
        final_rms_radius_m=float(rms[-1]),
        concentration_gain=float(rms[0] / max(rms[-1], np.finfo(float).tiny)),
        initial_target_fraction=float(np.mean(initial_inside)),
        final_target_fraction=float(np.mean(final_inside)),
        retained_target_fraction=retained,
        escaped_target_fraction=escaped,
        repulsive_gradient_fraction=repulsive_fraction,
        log_contraction_rate_s=contraction,
    )


def simulate_dynamic_inversion(
    sequence: DynamicInversionSequence,
    particle: FerromagneticParticle,
    initial_positions_m: np.ndarray,
    *,
    duration_s: float,
    target_radius_m: float,
    initial_body_angles_rad: Sequence[float] | np.ndarray | None = None,
    initial_moment_angles_rad: Sequence[float] | np.ndarray | None = None,
    background_velocity_m_s: Sequence[float] = (0.0, 0.0),
    bounds_m: Sequence[Sequence[float]] | np.ndarray | None = None,
    brownian: bool = True,
    seed: int | None = None,
    save_every_full_cycles: int = 1,
) -> DynamicInversionResult:
    """Integrate repeated dynamic-inversion cycles without sticky capture."""

    duration = _positive(duration_s, "duration_s")
    target_radius = _positive(target_radius_m, "target_radius_m")
    positions = np.asarray(initial_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or positions.shape[0] < 1:
        raise ValueError("initial_positions_m must have shape (n_particles, 2)")
    if np.any(~np.isfinite(positions)):
        raise ValueError("initial_positions_m must be finite")
    positions = positions.copy()
    count = positions.shape[0]
    if initial_body_angles_rad is None:
        body_angle = np.zeros(count)
    else:
        body_angle = np.asarray(initial_body_angles_rad, dtype=np.float64)
    if body_angle.shape != (count,) or np.any(~np.isfinite(body_angle)):
        raise ValueError("initial_body_angles_rad must have one finite value per particle")
    body_angle = _wrap_angle(body_angle.copy())
    if initial_moment_angles_rad is None:
        moment_angle = body_angle.copy()
    else:
        moment_angle = np.asarray(initial_moment_angles_rad, dtype=np.float64)
    if moment_angle.shape != (count,) or np.any(~np.isfinite(moment_angle)):
        raise ValueError("initial_moment_angles_rad must have one finite value per particle")
    moment_angle = _wrap_angle(moment_angle.copy())
    velocity = np.asarray(background_velocity_m_s, dtype=np.float64)
    if velocity.shape != (2,) or np.any(~np.isfinite(velocity)):
        raise ValueError("background_velocity_m_s must be a finite 2-vector")
    if bounds_m is None:
        bounds = None
    else:
        bounds = np.asarray(bounds_m, dtype=np.float64)
        if (
            bounds.shape != (2, 2)
            or np.any(~np.isfinite(bounds))
            or np.any(bounds[:, 1] <= bounds[:, 0])
        ):
            raise ValueError("bounds_m must contain increasing x and y pairs")
    if int(save_every_full_cycles) != save_every_full_cycles or save_every_full_cycles < 1:
        raise ValueError("save_every_full_cycles must be a positive integer")

    directions = _unit_directions(sequence.directions)
    element_count = int(np.floor(duration / sequence.element_period_s))
    if element_count < len(directions):
        raise ValueError("duration_s must contain at least one full direction cycle")
    save_stride = int(save_every_full_cycles) * len(directions)
    save_elements = list(range(save_stride, element_count + 1, save_stride))
    if not save_elements or save_elements[-1] != element_count:
        save_elements.append(element_count)
    save_set = set(save_elements)
    saved_time = [0.0]
    saved_positions = [positions.copy()]
    saved_body = [body_angle.copy()]
    saved_moment = [moment_angle.copy()]
    max_radius = np.linalg.norm(positions, axis=1)
    repulsive = 0
    gradient_samples = 0
    rng = np.random.default_rng(seed)
    moment = particle.magnetic_moment_a_m2(sequence.polarizing_field_t)
    occupied = (
        sequence.polarizing_duration_s + sequence.delay_s + sequence.gradient_duration_s
    )
    idle = sequence.element_period_s - occupied

    for element in range(element_count):
        direction = directions[element % directions.shape[0]]
        body_angle, moment_angle = _advance_uniform_field(
            positions,
            body_angle,
            moment_angle,
            sequence.polarizing_field_t * direction,
            sequence.polarizing_duration_s,
            sequence.polarizing_substeps,
            particle,
            moment,
            velocity,
            rng,
            bounds,
            brownian,
        )
        body_angle, moment_angle = _advance_free(
            positions,
            body_angle,
            moment_angle,
            sequence.delay_s,
            1,
            particle,
            velocity,
            rng,
            bounds,
            brownian,
        )
        body_angle, moment_angle, repelled, samples = _advance_gradient(
            positions,
            body_angle,
            moment_angle,
            direction,
            sequence,
            particle,
            moment,
            velocity,
            rng,
            bounds,
            brownian,
        )
        repulsive += repelled
        gradient_samples += samples
        body_angle, moment_angle = _advance_free(
            positions,
            body_angle,
            moment_angle,
            idle,
            sequence.free_substeps,
            particle,
            velocity,
            rng,
            bounds,
            brownian,
        )
        max_radius = np.maximum(max_radius, np.linalg.norm(positions, axis=1))
        completed = element + 1
        if completed in save_set:
            saved_time.append(completed * sequence.element_period_s)
            saved_positions.append(positions.copy())
            saved_body.append(body_angle.copy())
            saved_moment.append(moment_angle.copy())

    time = np.asarray(saved_time)
    position_history = np.asarray(saved_positions)
    repulsive_fraction = repulsive / max(gradient_samples, 1)
    stability = _stability_metrics(
        time,
        position_history,
        max_radius,
        target_radius,
        repulsive_fraction,
    )
    return DynamicInversionResult(
        sequence=sequence,
        particle=particle,
        time_s=time,
        positions_m=position_history,
        body_angles_rad=np.asarray(saved_body),
        moment_angles_rad=np.asarray(saved_moment),
        element_count=element_count,
        max_radius_m=max_radius,
        stability=stability,
    )


@dataclass(frozen=True)
class DynamicInversionHardwareConfig:
    """Timing, parallelism, and optional energy assumptions for one architecture."""

    architecture: HardwareArchitecture
    epm_channel_count: int = 72
    epm_parallel_channels: int = 18
    epm_changed_channel_fraction: float = 1.0
    epm_programming_pulse_s: float = 50e-6
    epm_settle_s: float = 20e-6
    coil_rise_time_s: float = 2e-6
    coil_energy_per_pulse_j: float | None = None
    epm_energy_per_channel_pulse_j: float | None = None
    hybrid_coil_energy_fraction: float = 0.65
    epm_transition_field_controlled: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in {"coils", "epm", "hybrid"}:
            raise ValueError("architecture must be 'coils', 'epm', or 'hybrid'")
        for name in ("epm_channel_count", "epm_parallel_channels"):
            value = getattr(self, name)
            if int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        fraction = float(self.epm_changed_channel_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("epm_changed_channel_fraction must be in (0, 1]")
        for name in ("epm_programming_pulse_s", "epm_settle_s", "coil_rise_time_s"):
            _positive(getattr(self, name), name, allow_zero=name != "epm_programming_pulse_s")
        if self.coil_energy_per_pulse_j is not None:
            _positive(self.coil_energy_per_pulse_j, "coil_energy_per_pulse_j")
        if self.epm_energy_per_channel_pulse_j is not None:
            _positive(
                self.epm_energy_per_channel_pulse_j,
                "epm_energy_per_channel_pulse_j",
            )
        fraction = float(self.hybrid_coil_energy_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("hybrid_coil_energy_fraction must be in (0, 1]")


@dataclass(frozen=True)
class DynamicInversionHardwareAssessment:
    """Pulse-count, timing, energy, and orientation-memory consequences."""

    config: DynamicInversionHardwareConfig
    duration_s: float
    element_count: int
    full_cycle_count: int
    coil_pulse_count: int
    epm_retained_state_changes: int
    epm_channel_pulse_count: int
    epm_batches_per_state: int
    field_transition_s: float
    effective_antialigned_delay_s: float
    minimum_element_period_s: float
    timing_margin_s: float
    orientation_memory_s: float
    active_coil_duty_cycle: float
    programming_duty_cycle: float
    estimated_energy_j: float | None
    estimated_average_power_w: float | None
    timing_feasible: bool
    orientation_memory_feasible: bool
    waveform_fidelity_feasible: bool
    consequence: str

    @property
    def viable(self) -> bool:
        return (
            self.timing_feasible
            and self.orientation_memory_feasible
            and self.waveform_fidelity_feasible
        )


def assess_dynamic_inversion_hardware(
    sequence: DynamicInversionSequence,
    particle: FerromagneticParticle,
    *,
    duration_s: float,
    config: DynamicInversionHardwareConfig,
) -> DynamicInversionHardwareAssessment:
    """Compare fast coils, EPM-only switching, or an EPM/coil hybrid."""

    duration = _positive(duration_s, "duration_s")
    elements = int(np.floor(duration / sequence.element_period_s))
    cycles = elements // len(sequence.directions)
    changed_channels = int(
        np.ceil(config.epm_channel_count * config.epm_changed_channel_fraction)
    )
    batches = int(np.ceil(changed_channels / config.epm_parallel_channels))
    epm_state_time = batches * (
        config.epm_programming_pulse_s + config.epm_settle_s
    )
    memory = particle.orientation_memory_time_s(
        sequence.gradient_field_at_center_t,
        polarizing_field_t=sequence.polarizing_field_t,
    )

    if config.architecture == "coils":
        coil_pulses = 2 * elements
        state_changes = 0
        channel_pulses = 0
        transition = config.coil_rise_time_s
        effective_delay = sequence.delay_s + transition
        minimum_period = (
            sequence.polarizing_duration_s
            + sequence.delay_s
            + sequence.gradient_duration_s
            + 4.0 * transition
        )
        coil_duty = sequence.active_duty_cycle
        programming_duty = 0.0
        fidelity = True
        consequence = (
            "Fast coils reproduce rectangular polarize/gradient pulses, but "
            "repeat copper loss and driver voltage dominate long runs."
        )
        energy = (
            None
            if config.coil_energy_per_pulse_j is None
            else coil_pulses * config.coil_energy_per_pulse_j
        )
    elif config.architecture == "epm":
        coil_pulses = 0
        state_changes = 3 * elements
        channel_pulses = changed_channels * state_changes
        transition = epm_state_time
        effective_delay = sequence.delay_s + transition
        minimum_period = (
            sequence.polarizing_duration_s
            + sequence.delay_s
            + sequence.gradient_duration_s
            + 3.0 * transition
        )
        coil_duty = 0.0
        programming_duty = min(
            1.0,
            state_changes * batches * config.epm_programming_pulse_s / duration,
        )
        fidelity = config.epm_transition_field_controlled
        consequence = (
            "EPM-only operation needs polarize, gradient, and field-off retained "
            "states every element; the programming transient is part of the "
            "particle waveform and is not calibrated as a rectangular field."
        )
        energy = (
            None
            if config.epm_energy_per_channel_pulse_j is None
            else channel_pulses * config.epm_energy_per_channel_pulse_j
        )
    else:
        coil_pulses = 2 * elements
        state_changes = 2
        channel_pulses = 2 * changed_channels
        transition = config.coil_rise_time_s
        effective_delay = sequence.delay_s + transition
        minimum_period = (
            sequence.polarizing_duration_s
            + sequence.delay_s
            + sequence.gradient_duration_s
            + 4.0 * transition
        )
        coil_duty = sequence.active_duty_cycle
        programming_duty = min(
            1.0,
            2.0 * batches * config.epm_programming_pulse_s / duration,
        )
        fidelity = True
        consequence = (
            "The EPM array supplies a slowly programmed bias/shaping state while "
            "fast coils preserve pulse timing; coil loss remains but EPM cycling "
            "and retained-state wear are nearly eliminated."
        )
        coil_energy = (
            None
            if config.coil_energy_per_pulse_j is None
            else (
                coil_pulses
                * config.coil_energy_per_pulse_j
                * config.hybrid_coil_energy_fraction
            )
        )
        epm_energy = (
            None
            if config.epm_energy_per_channel_pulse_j is None
            else channel_pulses * config.epm_energy_per_channel_pulse_j
        )
        energy = (
            None
            if coil_energy is None or epm_energy is None
            else coil_energy + epm_energy
        )

    timing_margin = sequence.element_period_s - minimum_period
    memory_feasible = (
        effective_delay + sequence.gradient_duration_s < memory
    )
    power = None if energy is None else energy / duration
    return DynamicInversionHardwareAssessment(
        config=config,
        duration_s=duration,
        element_count=elements,
        full_cycle_count=cycles,
        coil_pulse_count=coil_pulses,
        epm_retained_state_changes=state_changes,
        epm_channel_pulse_count=channel_pulses,
        epm_batches_per_state=batches,
        field_transition_s=transition,
        effective_antialigned_delay_s=effective_delay,
        minimum_element_period_s=minimum_period,
        timing_margin_s=timing_margin,
        orientation_memory_s=memory,
        active_coil_duty_cycle=coil_duty,
        programming_duty_cycle=programming_duty,
        estimated_energy_j=energy,
        estimated_average_power_w=power,
        timing_feasible=timing_margin >= 0.0,
        orientation_memory_feasible=memory_feasible,
        waveform_fidelity_feasible=fidelity,
        consequence=consequence,
    )
