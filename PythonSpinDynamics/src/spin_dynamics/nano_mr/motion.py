"""Diffusing nuclear-spin trajectories and dipolar field records.

This module connects the package-wide Brownian/advection engine in
``spin_dynamics.motion`` to nano-MR sensing. Positions exposed by this module
use nanometres; diffusion coefficients and drift velocities retain SI units.
The simulated nuclear moments precess classically while reproducing the
quantum transverse second moment of the selected bath species.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.motion import Boundary, advect_diffuse_positions
from spin_dynamics.nano_mr.baths import (
    PLANCK_CONSTANT_J_S,
    NuclearBathSpecies,
)
from spin_dynamics.nano_mr.frames import rotation_from_z, unit_vector
from spin_dynamics.nano_mr.geometry import MU0_OVER_4PI


FieldComponent = Literal["sensor_axis", "x", "y", "z"]


@dataclass(frozen=True, eq=False)
class DipolarFieldTrajectory:
    """Seeded positions and magnetic field sampled at one defect sensor.

    ``positions_lab_nm`` has shape ``(T, N, 3)`` and ``field_lab_tesla`` has
    shape ``(T, 3)``. ``sensor_axis_field_tesla`` is the field projected onto
    the supplied sensor axis. Each walker represents one nuclear spin unless
    ``spin_weights`` was supplied to the simulator.
    """

    times_seconds: np.ndarray
    positions_lab_nm: np.ndarray
    field_lab_tesla: np.ndarray
    sensor_axis_field_tesla: np.ndarray
    sensor_position_lab_nm: np.ndarray
    sensor_axis_lab: np.ndarray
    diffusion_coefficient_m2_s: np.ndarray
    drift_velocity_m_s: np.ndarray
    larmor_frequency_hz: float
    seed: int | None

    @property
    def sample_interval_seconds(self) -> float:
        """Return the uniform record interval."""

        if self.times_seconds.size < 2:
            raise ValueError("at least two samples are required to infer an interval")
        return float(self.times_seconds[1] - self.times_seconds[0])


@dataclass(frozen=True)
class FieldCorrelationResult:
    """One-sided autocorrelation of a uniformly sampled field record."""

    lag_seconds: np.ndarray
    autocorrelation_tesla2: np.ndarray
    normalized: bool


@dataclass(frozen=True)
class FieldSpectrumResult:
    """One-sided power spectral density of a field record."""

    frequency_hz: np.ndarray
    power_spectral_density_tesla2_per_hz: np.ndarray
    window: str
    segment_count: int


@dataclass(frozen=True)
class TrajectoryDisplacementResult:
    """Mean displacement and mean-squared displacement of the walkers."""

    times_seconds: np.ndarray
    mean_displacement_nm: np.ndarray
    mean_squared_displacement_nm2: np.ndarray


def simulate_diffusing_dipolar_field(
    initial_positions_lab_nm,
    species: NuclearBathSpecies,
    *,
    sensor_position_lab_nm=(0.0, 0.0, 0.0),
    sensor_axis_lab=(0.0, 0.0, 1.0),
    static_field_lab_tesla=(0.0, 0.0, 0.0),
    sample_interval_seconds: float,
    sample_count: int,
    motion_substeps_per_sample: int = 1,
    diffusion_coefficient_m2_s=0.0,
    drift_velocity_m_s=(0.0, 0.0, 0.0),
    bounds_lab_nm=None,
    boundary: Boundary = "reflect",
    seed: int | None = None,
    initial_phases_rad=None,
    spin_weights=None,
    minimum_distance_nm: float = 0.1,
    include_longitudinal_mean: bool = True,
) -> DipolarFieldTrajectory:
    """Simulate a seeded nuclear trajectory and its dipolar sensor field.

    Brownian increments use variance ``2 D dt`` independently on each axis.
    ``motion_substeps_per_sample`` decouples the Brownian integration step from
    the field-record interval; increase it until a typical motion step is small
    relative to the sensor depth and confinement dimensions.
    ``drift_velocity_m_s`` is added before applying the optional rectangular
    ``bounds_lab_nm`` with an existing ``spin_dynamics.motion`` boundary mode.

    The transverse classical moment amplitude is chosen so that random phases
    reproduce ``<mu_x^2> = <mu_y^2>`` for the species' quantum spin state.
    A thermally or explicitly polarized longitudinal mean can also be included.
    ``minimum_distance_nm`` is a hard near-field cutoff that prevents the point
    dipole kernel from diverging if a model permits a walker to cross the
    sensor.
    """

    positions_nm = _positions(initial_positions_lab_nm)
    if not isinstance(species, NuclearBathSpecies):
        raise TypeError("species must be a NuclearBathSpecies")
    dt = float(sample_interval_seconds)
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("sample_interval_seconds must be positive and finite")
    count = int(sample_count)
    if count < 2 or count != sample_count:
        raise ValueError("sample_count must be an integer of at least two")
    substeps = int(motion_substeps_per_sample)
    if substeps <= 0 or substeps != motion_substeps_per_sample:
        raise ValueError("motion_substeps_per_sample must be a positive integer")
    motion_dt = dt / substeps
    sensor_position_nm = _point(sensor_position_lab_nm, "sensor_position_lab_nm")
    sensor_axis = unit_vector(sensor_axis_lab, name="sensor_axis_lab")
    static_field = _point(static_field_lab_tesla, "static_field_lab_tesla")
    drift = _point(drift_velocity_m_s, "drift_velocity_m_s")
    cutoff_nm = float(minimum_distance_nm)
    if cutoff_nm <= 0.0 or not np.isfinite(cutoff_nm):
        raise ValueError("minimum_distance_nm must be positive and finite")

    particle_count = positions_nm.shape[0]
    diffusion = _particle_values(
        diffusion_coefficient_m2_s,
        particle_count,
        "diffusion_coefficient_m2_s",
    )
    if np.any(diffusion < 0.0):
        raise ValueError("diffusion_coefficient_m2_s must be non-negative")
    weights = _particle_values(
        1.0 if spin_weights is None else spin_weights,
        particle_count,
        "spin_weights",
    )
    if np.any(weights < 0.0):
        raise ValueError("spin_weights must be non-negative")
    generator = np.random.default_rng(seed)
    if initial_phases_rad is None:
        phases = generator.uniform(0.0, 2.0 * np.pi, particle_count)
    else:
        phases = _particle_values(
            initial_phases_rad,
            particle_count,
            "initial_phases_rad",
        )

    bounds_m = None
    if bounds_lab_nm is not None:
        bounds_m = _bounds(bounds_lab_nm)
        initial_m = positions_nm * 1.0e-9
        for axis, (lower, upper) in enumerate(bounds_m):
            if np.any(initial_m[:, axis] < lower) or np.any(
                initial_m[:, axis] > upper
            ):
                raise ValueError("initial positions must lie inside bounds_lab_nm")

    field_magnitude = float(np.linalg.norm(static_field))
    quantization_axis = (
        sensor_axis if field_magnitude == 0.0 else static_field / field_magnitude
    )
    rotation = rotation_from_z(quantization_axis)
    transverse_x = rotation[:, 0]
    transverse_y = rotation[:, 1]
    larmor_hz = species.gamma_hz_per_t * field_magnitude
    second_moment = species.transverse_spin_second_moment(field_magnitude)
    transverse_moment = (
        PLANCK_CONSTANT_J_S
        * species.gamma_hz_per_t
        * np.sqrt(2.0 * second_moment)
    )
    longitudinal_moment = 0.0
    if include_longitudinal_mean:
        longitudinal_moment = (
            PLANCK_CONSTANT_J_S
            * species.gamma_hz_per_t
            * species.mean_spin_projection(field_magnitude)
        )

    times = np.arange(count, dtype=np.float64) * dt
    trajectory_m = np.empty((count, particle_count, 3), dtype=np.float64)
    field_record = np.empty((count, 3), dtype=np.float64)
    current_m = positions_nm * 1.0e-9
    sensor_position_m = sensor_position_nm * 1.0e-9
    cutoff_m = cutoff_nm * 1.0e-9

    for index, time_seconds in enumerate(times):
        trajectory_m[index] = current_m
        angles = phases + 2.0 * np.pi * larmor_hz * time_seconds
        moments = transverse_moment * (
            np.cos(angles)[:, np.newaxis] * transverse_x
            + np.sin(angles)[:, np.newaxis] * transverse_y
        )
        if longitudinal_moment != 0.0:
            moments = moments + longitudinal_moment * quantization_axis
        field_record[index] = dipolar_field_from_moments(
            current_m,
            moments,
            sensor_position_m=sensor_position_m,
            weights=weights,
            minimum_distance_m=cutoff_m,
        )
        if index + 1 < count:
            for substep in range(substeps):
                current_m = advect_diffuse_positions(
                    current_m,
                    motion_dt,
                    velocity=drift,
                    diffusion_coefficient=diffusion,
                    rng=generator,
                    time=time_seconds + substep * motion_dt,
                    bounds=bounds_m,
                    boundary=boundary,
                )

    projected = field_record @ sensor_axis
    return DipolarFieldTrajectory(
        times_seconds=times,
        positions_lab_nm=trajectory_m * 1.0e9,
        field_lab_tesla=field_record,
        sensor_axis_field_tesla=projected,
        sensor_position_lab_nm=sensor_position_nm,
        sensor_axis_lab=sensor_axis,
        diffusion_coefficient_m2_s=diffusion,
        drift_velocity_m_s=drift,
        larmor_frequency_hz=float(larmor_hz),
        seed=seed,
    )


def dipolar_field_from_moments(
    positions_m,
    moments_j_per_t,
    *,
    sensor_position_m=(0.0, 0.0, 0.0),
    weights=1.0,
    minimum_distance_m: float = 0.0,
) -> np.ndarray:
    """Return the summed point-dipole magnetic field in tesla.

    Positions and sensor coordinates are in metres and magnetic moments are in
    joules per tesla. A positive ``minimum_distance_m`` radially clips the
    singular point-dipole distance.
    """

    positions = _positions(positions_m)
    moments = np.asarray(moments_j_per_t, dtype=np.float64)
    if moments.shape != positions.shape or not np.all(np.isfinite(moments)):
        raise ValueError("moments_j_per_t must be a finite array matching positions")
    sensor = _point(sensor_position_m, "sensor_position_m")
    scale = _particle_values(weights, positions.shape[0], "weights")
    cutoff = float(minimum_distance_m)
    if cutoff < 0.0 or not np.isfinite(cutoff):
        raise ValueError("minimum_distance_m must be finite and non-negative")

    displacement = positions - sensor
    distances = np.linalg.norm(displacement, axis=1)
    if np.any(distances == 0.0):
        raise ValueError("a dipole position coincides with the sensor")
    effective_distance = np.maximum(distances, cutoff)
    direction = displacement / distances[:, np.newaxis]
    projection = np.sum(moments * direction, axis=1)
    individual = MU0_OVER_4PI * (
        3.0 * direction * projection[:, np.newaxis] - moments
    ) / effective_distance[:, np.newaxis] ** 3
    return np.sum(scale[:, np.newaxis] * individual, axis=0)


def field_autocorrelation(
    record_or_values: DipolarFieldTrajectory | np.ndarray,
    *,
    sample_interval_seconds: float | None = None,
    component: FieldComponent = "sensor_axis",
    max_lag: int | None = None,
    demean: bool = True,
    unbiased: bool = True,
    normalize: bool = False,
) -> FieldCorrelationResult:
    """Return the non-negative-lag field autocorrelation.

    FFT convolution makes the estimator practical for long trajectory records.
    ``normalize=True`` divides by the zero-lag value after optional demeaning.
    """

    values, dt = _field_values_and_interval(
        record_or_values,
        sample_interval_seconds,
        component,
    )
    size = values.size
    lag_count = size if max_lag is None else int(max_lag) + 1
    if lag_count <= 0 or lag_count > size:
        raise ValueError("max_lag must lie between zero and record length minus one")
    centered = values - np.mean(values) if demean else values.copy()
    fft_size = 1 << (2 * size - 1).bit_length()
    transform = np.fft.rfft(centered, n=fft_size)
    correlation = np.fft.irfft(transform * np.conjugate(transform), n=fft_size)[:size]
    denominator = np.arange(size, 0, -1) if unbiased else float(size)
    correlation = correlation / denominator
    correlation = correlation[:lag_count]
    if normalize:
        zero_lag = float(correlation[0])
        if zero_lag <= 0.0:
            raise ValueError("cannot normalize a record with zero variance")
        correlation = correlation / zero_lag
    return FieldCorrelationResult(
        lag_seconds=np.arange(lag_count, dtype=np.float64) * dt,
        autocorrelation_tesla2=correlation,
        normalized=bool(normalize),
    )


def field_power_spectral_density(
    record_or_values: DipolarFieldTrajectory | np.ndarray,
    *,
    sample_interval_seconds: float | None = None,
    component: FieldComponent = "sensor_axis",
    demean: bool = True,
    window: Literal["hann", "rectangular"] = "hann",
    segment_length: int | None = None,
    overlap_fraction: float = 0.5,
) -> FieldSpectrumResult:
    """Return a one-sided PSD in tesla squared per hertz.

    The default is a full-record periodogram. Set ``segment_length`` to average
    overlapping modified periodograms (Welch estimation); ``overlap_fraction``
    must then lie in ``[0, 1)``.
    """

    values, dt = _field_values_and_interval(
        record_or_values,
        sample_interval_seconds,
        component,
    )
    if segment_length is None:
        length = values.size
        starts = [0]
    else:
        length = int(segment_length)
        if length < 4 or length > values.size or length != segment_length:
            raise ValueError("segment_length must be an integer from 4 to record length")
        overlap = float(overlap_fraction)
        if overlap < 0.0 or overlap >= 1.0 or not np.isfinite(overlap):
            raise ValueError("overlap_fraction must lie in [0, 1)")
        step = max(1, int(round(length * (1.0 - overlap))))
        starts = list(range(0, values.size - length + 1, step))
    if window == "hann":
        weights = np.hanning(length)
    elif window == "rectangular":
        weights = np.ones(length)
    else:
        raise ValueError("window must be 'hann' or 'rectangular'")
    window_power = float(np.sum(weights**2))
    if window_power == 0.0:
        raise ValueError("record is too short for the selected window")
    sample_rate = 1.0 / dt
    psd = np.zeros(length // 2 + 1, dtype=np.float64)
    for start in starts:
        segment = values[start : start + length].copy()
        if demean:
            segment -= np.mean(segment)
        transform = np.fft.rfft(segment * weights)
        psd += np.abs(transform) ** 2 / (sample_rate * window_power)
    psd /= len(starts)
    if length % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0
    return FieldSpectrumResult(
        frequency_hz=np.fft.rfftfreq(length, d=dt),
        power_spectral_density_tesla2_per_hz=psd,
        window=window,
        segment_count=len(starts),
    )


def trajectory_displacement_statistics(
    trajectory: DipolarFieldTrajectory,
) -> TrajectoryDisplacementResult:
    """Return ensemble displacement statistics relative to the initial sample.

    Do not interpret wrapped periodic coordinates with this helper: their
    apparent displacements jump at the periodic seam. Free and reflecting
    trajectories have the usual direct-coordinate interpretation.
    """

    displacement = (
        trajectory.positions_lab_nm - trajectory.positions_lab_nm[0][np.newaxis]
    )
    return TrajectoryDisplacementResult(
        times_seconds=trajectory.times_seconds.copy(),
        mean_displacement_nm=np.mean(displacement, axis=1),
        mean_squared_displacement_nm2=np.mean(
            np.sum(displacement**2, axis=2),
            axis=1,
        ),
    )


def free_diffusion_return_density(
    time_seconds,
    diffusion_coefficient_m2_s: float,
    *,
    spatial_dimension: int = 3,
) -> np.ndarray:
    """Return the Gaussian free-diffusion propagator at zero displacement.

    ``G(0,t) = (4*pi*D*t)^(-d/2)`` provides the stationary free-diffusion and
    long-time ``t^(-d/2)`` reference used when validating trajectory-derived
    correlations. Times must be strictly positive.
    """

    times = np.asarray(time_seconds, dtype=np.float64)
    diffusion = float(diffusion_coefficient_m2_s)
    dimension = int(spatial_dimension)
    if times.size == 0 or np.any(times <= 0.0) or not np.all(np.isfinite(times)):
        raise ValueError("time_seconds must contain finite positive values")
    if diffusion <= 0.0 or not np.isfinite(diffusion):
        raise ValueError("diffusion_coefficient_m2_s must be positive and finite")
    if dimension <= 0 or dimension != spatial_dimension:
        raise ValueError("spatial_dimension must be a positive integer")
    return (4.0 * np.pi * diffusion * times) ** (-0.5 * dimension)


def _field_values_and_interval(
    record_or_values: DipolarFieldTrajectory | np.ndarray,
    sample_interval_seconds: float | None,
    component: FieldComponent,
) -> tuple[np.ndarray, float]:
    if isinstance(record_or_values, DipolarFieldTrajectory):
        if sample_interval_seconds is not None:
            raise ValueError(
                "sample_interval_seconds is inferred when a trajectory is supplied"
            )
        if component == "sensor_axis":
            values = record_or_values.sensor_axis_field_tesla
        else:
            index = {"x": 0, "y": 1, "z": 2}.get(component)
            if index is None:
                raise ValueError("component must be 'sensor_axis', 'x', 'y', or 'z'")
            values = record_or_values.field_lab_tesla[:, index]
        dt = record_or_values.sample_interval_seconds
    else:
        values = np.asarray(record_or_values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("raw field values must be a one-dimensional array")
        if sample_interval_seconds is None:
            raise ValueError("sample_interval_seconds is required for raw values")
        dt = float(sample_interval_seconds)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("field record must contain at least two finite samples")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("sample_interval_seconds must be positive and finite")
    return values.astype(np.float64, copy=True), dt


def _positions(values) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or positions.shape[0] == 0
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("positions must be a finite non-empty Nx3 array")
    return positions.copy()


def _point(values, name: str) -> np.ndarray:
    point = np.asarray(values, dtype=np.float64)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite three-component vector")
    return point.copy()


def _particle_values(values, size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    else:
        array = array.reshape(-1)
        if array.size != size:
            raise ValueError(f"{name} must be scalar or have one value per spin")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _bounds(values) -> tuple[tuple[float, float], ...]:
    if len(values) != 3:
        raise ValueError("bounds_lab_nm must contain three (lower, upper) pairs")
    bounds = []
    for pair in values:
        if len(pair) != 2:
            raise ValueError("each bounds_lab_nm entry must contain two values")
        lower = float(pair[0]) * 1.0e-9
        upper = float(pair[1]) * 1.0e-9
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError("each bounds_lab_nm entry must have finite lower < upper")
        bounds.append((lower, upper))
    return tuple(bounds)


__all__ = [
    "DipolarFieldTrajectory",
    "FieldComponent",
    "FieldCorrelationResult",
    "FieldSpectrumResult",
    "TrajectoryDisplacementResult",
    "dipolar_field_from_moments",
    "field_autocorrelation",
    "field_power_spectral_density",
    "free_diffusion_return_density",
    "simulate_diffusing_dipolar_field",
    "trajectory_displacement_statistics",
]
