"""Closed-loop imaging and magnetophoretic transport with EPM arrays.

The controller alternates target and particle imaging, retained-state
synthesis/programming, a magnetophoretic transport burst, and verification
imaging.  Every cycle re-aims the affine field gradient from the image-estimated
centroid of the uncaptured population toward the newly localized target.  True
particle coordinates remain inside the transport simulator and are exposed on
results only as validation diagnostics.

The controller records retained-remanence total variation as a programming
effort metric.  It intentionally does not turn that variation into electrical
energy: doing so requires a calibrated multi-channel pulse schedule and driver
model for the selected array.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.electropermanent_array import (
    ArrayStateSynthesisResult,
    synthesize_transport_state,
)
from spin_dynamics.workflows.electropermanent_imaging import (
    EPMNonlinearImagingResult,
    NonlinearEPMEncoding,
    run_epm_nonlinear_imaging,
)
from spin_dynamics.workflows.electropermanent_particle_imaging import (
    EPMParticleImagingResult,
    run_epm_particle_imaging,
)
from spin_dynamics.workflows.electropermanent_transport import (
    MagneticForceMap2D,
    MagnetizationModel,
    MagnetophoreticTransportResult,
    SuperparamagneticParticle,
    TransportBoundary,
    TransportVelocity,
    magnetic_force_map_2d,
    simulate_magnetophoretic_transport,
)

__all__ = [
    "ControllerModeInterval",
    "EPMTherapyControllerConfig",
    "EPMTherapyControllerResult",
    "EPMTherapyCycleResult",
    "localize_epm_target",
    "run_epm_image_guided_controller",
]

ControllerMode = Literal["imaging", "programming", "transport"]
StopReason = Literal["capture_goal", "max_cycles"]


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


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


@dataclass(frozen=True)
class EPMTherapyControllerConfig:
    """Timing, field objective, reconstruction, and stopping parameters."""

    max_cycles: int = 4
    capture_goal: float = 0.70
    imaging_window_s: float = 90.0
    programming_window_s: float = 0.25
    transport_window_s: float = 1500.0
    transport_time_step_s: float = 5.0
    target_radius_m: float = 4.2e-3
    transport_bias_field_t: float = 2.0e-3
    transport_gradient_t_m: float = 0.150
    localization_threshold_fraction: float = 0.90
    imaging_regularization: float = 1.0e-4
    imaging_snr_db: float | None = 35.0
    synthesis_regularization: float = 1.0e-10
    synthesis_tolerance_t: float = 1.0e-10
    magnetization_model: MagnetizationModel = "langevin"
    boundary: TransportBoundary = "reflect"
    seed: int = 0
    verification_imaging_window_s: float = 90.0
    particle_imaging_regularization: float = 1.0e-4
    particle_imaging_snr_db: float | None = 35.0
    particle_signal_per_particle: float = 1.0
    particle_support_threshold_fraction: float = 0.01
    particle_boundary_capture_correction: bool = True

    def __post_init__(self) -> None:
        if int(self.max_cycles) != self.max_cycles or self.max_cycles < 1:
            raise ValueError("max_cycles must be a positive integer")
        goal = float(self.capture_goal)
        if not np.isfinite(goal) or not 0.0 < goal <= 1.0:
            raise ValueError("capture_goal must be in (0, 1]")
        for name in (
            "imaging_window_s",
            "programming_window_s",
            "transport_window_s",
            "transport_time_step_s",
            "verification_imaging_window_s",
            "target_radius_m",
            "transport_gradient_t_m",
            "particle_signal_per_particle",
            "synthesis_tolerance_t",
        ):
            _positive(getattr(self, name), name)
        _positive(
            self.transport_bias_field_t,
            "transport_bias_field_t",
            allow_zero=True,
        )
        fraction = float(self.localization_threshold_fraction)
        if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("localization_threshold_fraction must be in (0, 1]")
        _positive(
            self.imaging_regularization,
            "imaging_regularization",
            allow_zero=True,
        )
        if self.imaging_snr_db is not None:
            _positive(self.imaging_snr_db, "imaging_snr_db")
        _positive(
            self.particle_imaging_regularization,
            "particle_imaging_regularization",
            allow_zero=True,
        )
        if self.particle_imaging_snr_db is not None:
            _positive(self.particle_imaging_snr_db, "particle_imaging_snr_db")
        support = float(self.particle_support_threshold_fraction)
        if not np.isfinite(support) or not 0.0 <= support < 1.0:
            raise ValueError("particle_support_threshold_fraction must be in [0, 1)")
        _positive(
            self.synthesis_regularization,
            "synthesis_regularization",
            allow_zero=True,
        )
        if self.magnetization_model not in {"linear", "langevin"}:
            raise ValueError("magnetization_model must be 'linear' or 'langevin'")
        if self.boundary not in {"reflect", "periodic", "clip"}:
            raise ValueError("boundary must be 'reflect', 'periodic', or 'clip'")
        object.__setattr__(self, "max_cycles", int(self.max_cycles))
        object.__setattr__(self, "seed", int(self.seed))


@dataclass(frozen=True)
class ControllerModeInterval:
    """One controller mode occupying a half-open wall-clock interval."""

    cycle_index: int
    mode: ControllerMode
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)


@dataclass(frozen=True)
class EPMTherapyCycleResult:
    """Image, decision, programmed state, and transport output for one cycle."""

    cycle_index: int
    intervals: tuple[ControllerModeInterval, ...]
    imaging: EPMNonlinearImagingResult
    particle_imaging_before: EPMParticleImagingResult
    particle_imaging_after: EPMParticleImagingResult
    target_mask: np.ndarray
    target_center_m: np.ndarray
    localization_threshold: float
    source_centroid_m: np.ndarray
    ground_truth_source_centroid_m: np.ndarray
    requested_direction: np.ndarray
    transport_state: ArrayStateSynthesisResult
    force_map: MagneticForceMap2D
    transport: MagnetophoreticTransportResult
    capture_fraction_before: float
    estimated_capture_fraction_before: float
    imaging_remanence_variation_t: float
    transport_remanence_variation_t: float
    peak_transport_remanence_change_t: float

    @property
    def capture_fraction_after(self) -> float:
        """Ground-truth capture fraction retained for simulator evaluation."""

        return self.transport.capture_fraction

    @property
    def estimated_capture_fraction_after(self) -> float:
        """Post-transport target occupancy estimated from verification imaging."""

        return self.particle_imaging_after.estimate.capture_fraction

    @property
    def newly_captured_fraction(self) -> float:
        return self.capture_fraction_after - self.capture_fraction_before

    @property
    def estimated_newly_captured_fraction(self) -> float:
        return self.estimated_capture_fraction_after - self.estimated_capture_fraction_before

    @property
    def source_centroid_error_m(self) -> float:
        return float(np.linalg.norm(self.source_centroid_m - self.ground_truth_source_centroid_m))

    @property
    def start_s(self) -> float:
        return self.intervals[0].start_s

    @property
    def end_s(self) -> float:
        return self.intervals[-1].end_s


@dataclass(frozen=True)
class EPMTherapyControllerResult:
    """Complete image-estimate-program-transport-verify simulation."""

    config: EPMTherapyControllerConfig
    cycles: tuple[EPMTherapyCycleResult, ...]
    initial_positions_m: np.ndarray
    final_positions_m: np.ndarray
    final_captured: np.ndarray
    stop_reason: StopReason

    @property
    def capture_fraction(self) -> float:
        """Ground-truth final capture, retained for simulation evaluation."""

        return float(np.mean(self.final_captured))

    @property
    def estimated_capture_fraction(self) -> float:
        """Final capture fraction reported by particle verification imaging."""

        if not self.cycles:
            return 0.0
        return self.cycles[-1].estimated_capture_fraction_after

    @property
    def estimated_final_positions_m(self) -> np.ndarray:
        """Image-resolved representative particle locations after the final cycle."""

        if not self.cycles:
            return np.empty((0, 2))
        return self.cycles[-1].particle_imaging_after.estimate.positions_m

    @property
    def total_time_s(self) -> float:
        return 0.0 if not self.cycles else self.cycles[-1].end_s

    @property
    def intervals(self) -> tuple[ControllerModeInterval, ...]:
        return tuple(interval for cycle in self.cycles for interval in cycle.intervals)

    @property
    def capture_fraction_by_cycle(self) -> np.ndarray:
        """Ground-truth capture history used to score the closed-loop estimate."""

        return np.asarray(
            [cycle.capture_fraction_after for cycle in self.cycles],
            dtype=np.float64,
        )

    @property
    def estimated_capture_fraction_by_cycle(self) -> np.ndarray:
        """Capture history available to the controller through imaging."""

        return np.asarray(
            [cycle.estimated_capture_fraction_after for cycle in self.cycles],
            dtype=np.float64,
        )

    @property
    def particle_centroid_error_by_cycle_m(self) -> np.ndarray:
        """Post-transport particle-centroid estimation error by cycle."""

        return np.asarray(
            [cycle.particle_imaging_after.centroid_error_m for cycle in self.cycles],
            dtype=np.float64,
        )

    @property
    def localized_targets_m(self) -> np.ndarray:
        return np.asarray([cycle.target_center_m for cycle in self.cycles])

    @property
    def total_remanence_variation_t(self) -> float:
        return float(
            sum(
                cycle.imaging_remanence_variation_t
                + cycle.transport_remanence_variation_t
                for cycle in self.cycles
            )
        )

    @property
    def trajectory_time_s(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for index, cycle in enumerate(self.cycles):
            transport_interval = next(
                interval for interval in cycle.intervals if interval.mode == "transport"
            )
            time = transport_interval.start_s + cycle.transport.time_s
            values.append(time if index == 0 else time[1:])
        return np.concatenate(values) if values else np.zeros(0)

    @property
    def trajectory_positions_m(self) -> np.ndarray:
        values: list[np.ndarray] = []
        for index, cycle in enumerate(self.cycles):
            positions = cycle.transport.positions_m
            values.append(positions if index == 0 else positions[1:])
        if not values:
            return np.empty((0, self.initial_positions_m.shape[0], 2))
        return np.concatenate(values, axis=0)


def localize_epm_target(
    reconstructed_image: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    *,
    threshold_fraction: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return peak-relative target mask, signal-weighted centroid, and threshold."""

    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    image = np.asarray(reconstructed_image, dtype=np.float64)
    if image.shape != (y_axis.size, x_axis.size) or np.any(~np.isfinite(image)):
        raise ValueError("reconstructed_image must be finite and match the axes")
    if np.any(image < 0.0):
        raise ValueError("reconstructed_image must be non-negative")
    fraction = float(threshold_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("threshold_fraction must be in (0, 1]")
    peak = float(np.max(image))
    if peak <= 0.0:
        raise ValueError("reconstructed_image must contain positive signal")
    threshold = fraction * peak
    mask = image >= threshold
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    weights = np.where(mask, image, 0.0)
    total = float(np.sum(weights))
    center = np.asarray(
        (
            np.sum(weights * x_grid) / total,
            np.sum(weights * y_grid) / total,
        )
    )
    return mask, center, threshold


def run_epm_image_guided_controller(
    encoding: NonlinearEPMEncoding,
    expected_image: np.ndarray,
    x_m: Sequence[float] | np.ndarray,
    y_m: Sequence[float] | np.ndarray,
    particle: SuperparamagneticParticle,
    initial_positions_m: np.ndarray,
    *,
    config: EPMTherapyControllerConfig | None = None,
    background_velocity_m_s: TransportVelocity = None,
) -> EPMTherapyControllerResult:
    """Run image-estimate-program-transport-verify controller cycles.

    Particle coordinates enter the imaging forward model and the physical
    transport integrator, but control decisions use only reconstructed particle
    state.  Ground-truth fields on the result are diagnostic scores.
    """

    settings = EPMTherapyControllerConfig() if config is None else config
    x_axis = _strict_axis(x_m, "x_m")
    y_axis = _strict_axis(y_m, "y_m")
    image = np.asarray(expected_image, dtype=np.float64)
    if image.shape != encoding.image_shape or image.shape != (y_axis.size, x_axis.size):
        raise ValueError("expected_image, encoding image_shape, and axes must match")
    positions = np.asarray(initial_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2 or np.any(~np.isfinite(positions)):
        raise ValueError("initial_positions_m must have shape (n_particles, 2)")
    if positions.shape[0] < 1:
        raise ValueError("initial_positions_m must contain at least one particle")
    points = encoding.basis.points_m
    expected_points = np.column_stack(
        (
            np.tile(x_axis, y_axis.size),
            np.repeat(y_axis, x_axis.size),
            np.zeros(x_axis.size * y_axis.size),
        )
    )
    if not np.allclose(points, expected_points, rtol=0.0, atol=1e-12):
        raise ValueError("encoding basis points must follow the supplied Cartesian axes")

    cycles: list[EPMTherapyCycleResult] = []
    captured = np.zeros(positions.shape[0], dtype=bool)
    previous_state = np.zeros(len(encoding.basis.array.programmable_elements))
    elapsed = 0.0

    for cycle_index in range(settings.max_cycles):
        imaging_start = elapsed
        imaging_end = imaging_start + settings.imaging_window_s
        programming_end = imaging_end + settings.programming_window_s
        transport_end = programming_end + settings.transport_window_s
        verification_end = transport_end + settings.verification_imaging_window_s
        intervals = (
            ControllerModeInterval(cycle_index, "imaging", imaging_start, imaging_end),
            ControllerModeInterval(cycle_index, "programming", imaging_end, programming_end),
            ControllerModeInterval(cycle_index, "transport", programming_end, transport_end),
            ControllerModeInterval(cycle_index, "imaging", transport_end, verification_end),
        )

        seed_base = settings.seed + 5 * cycle_index
        imaging = run_epm_nonlinear_imaging(
            encoding,
            image,
            regularization=settings.imaging_regularization,
            snr_db=settings.imaging_snr_db,
            seed=seed_base,
        )
        target_mask, target_center, threshold = localize_epm_target(
            imaging.reconstructed_image,
            x_axis,
            y_axis,
            threshold_fraction=settings.localization_threshold_fraction,
        )
        captured |= np.linalg.norm(positions - target_center, axis=1) <= settings.target_radius_m
        uncaptured = ~captured
        ground_truth_source_centroid = (
            np.mean(positions[uncaptured], axis=0)
            if np.any(uncaptured)
            else target_center.copy()
        )
        particle_imaging_before = run_epm_particle_imaging(
            encoding,
            positions,
            x_axis,
            y_axis,
            target_center_m=target_center,
            target_radius_m=settings.target_radius_m,
            signal_per_particle=settings.particle_signal_per_particle,
            support_threshold_fraction=settings.particle_support_threshold_fraction,
            boundary_capture_correction=settings.particle_boundary_capture_correction,
            regularization=settings.particle_imaging_regularization,
            snr_db=settings.particle_imaging_snr_db,
            seed=seed_base + 1,
            ground_truth_captured=captured,
        )
        source_centroid = particle_imaging_before.estimate.uncaptured_centroid_m
        direction = target_center - source_centroid
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(np.float64).eps:
            direction = np.asarray((1.0, 0.0))
        else:
            direction /= norm

        transport_state = synthesize_transport_state(
            encoding.basis,
            bias_field_t=settings.transport_bias_field_t,
            gradient_t_per_m=(
                settings.transport_gradient_t_m * direction[0],
                settings.transport_gradient_t_m * direction[1],
                0.0,
            ),
            center_m=(0.0, 0.0, 0.0),
            field_direction=encoding.field_direction,
            regularization=settings.synthesis_regularization,
            tolerance_t=settings.synthesis_tolerance_t,
        )
        transport_field = encoding.basis.field_vectors(
            transport_state.remanence_t
        ).reshape(encoding.image_shape + (3,))
        force_map = magnetic_force_map_2d(x_axis, y_axis, transport_field)

        imaging_path = np.vstack((previous_state, encoding.remanence_states_t))
        initial_imaging_variation = float(np.sum(np.abs(np.diff(imaging_path, axis=0))))
        final_imaging_state = encoding.remanence_states_t[-1]
        transport_delta = transport_state.remanence_t - final_imaging_state
        transport_variation = float(np.sum(np.abs(transport_delta)))
        peak_transport_change = float(np.max(np.abs(transport_delta)))
        capture_before = float(np.mean(captured))

        transport = simulate_magnetophoretic_transport(
            force_map,
            particle,
            positions,
            duration_s=settings.transport_window_s,
            time_step_s=settings.transport_time_step_s,
            target_center_m=target_center,
            target_radius_m=settings.target_radius_m,
            background_velocity_m_s=background_velocity_m_s,
            magnetization_model=settings.magnetization_model,
            boundary=settings.boundary,
            initial_captured=captured,
            seed=seed_base + 2,
        )
        positions = transport.positions_m[-1].copy()
        captured = transport.captured.copy()
        particle_imaging_after = run_epm_particle_imaging(
            encoding,
            positions,
            x_axis,
            y_axis,
            target_center_m=target_center,
            target_radius_m=settings.target_radius_m,
            signal_per_particle=settings.particle_signal_per_particle,
            support_threshold_fraction=settings.particle_support_threshold_fraction,
            boundary_capture_correction=settings.particle_boundary_capture_correction,
            regularization=settings.particle_imaging_regularization,
            snr_db=settings.particle_imaging_snr_db,
            seed=seed_base + 3,
            ground_truth_captured=captured,
        )
        verification_path = np.vstack(
            (transport_state.remanence_t, encoding.remanence_states_t)
        )
        verification_variation = float(
            np.sum(np.abs(np.diff(verification_path, axis=0)))
        )
        imaging_variation = initial_imaging_variation + verification_variation
        cycles.append(
            EPMTherapyCycleResult(
                cycle_index=cycle_index,
                intervals=intervals,
                imaging=imaging,
                particle_imaging_before=particle_imaging_before,
                particle_imaging_after=particle_imaging_after,
                target_mask=target_mask,
                target_center_m=target_center,
                localization_threshold=threshold,
                source_centroid_m=source_centroid,
                ground_truth_source_centroid_m=ground_truth_source_centroid,
                requested_direction=direction,
                transport_state=transport_state,
                force_map=force_map,
                transport=transport,
                capture_fraction_before=capture_before,
                estimated_capture_fraction_before=(
                    particle_imaging_before.estimate.capture_fraction
                ),
                imaging_remanence_variation_t=imaging_variation,
                transport_remanence_variation_t=transport_variation,
                peak_transport_remanence_change_t=peak_transport_change,
            )
        )
        elapsed = verification_end
        previous_state = final_imaging_state
        if particle_imaging_after.estimate.capture_fraction >= settings.capture_goal:
            return EPMTherapyControllerResult(
                config=settings,
                cycles=tuple(cycles),
                initial_positions_m=np.asarray(initial_positions_m, dtype=np.float64).copy(),
                final_positions_m=positions,
                final_captured=captured,
                stop_reason="capture_goal",
            )

    return EPMTherapyControllerResult(
        config=settings,
        cycles=tuple(cycles),
        initial_positions_m=np.asarray(initial_positions_m, dtype=np.float64).copy(),
        final_positions_m=positions,
        final_captured=captured,
        stop_reason="max_cycles",
    )
