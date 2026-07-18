"""Nano-MRI scan geometries, dipolar forward models, and inverse solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.nano_mr.baths import (
    PLANCK_CONSTANT_J_S,
    NuclearBathSpecies,
)
from spin_dynamics.nano_mr.frames import unit_vector
from spin_dynamics.nano_mr.geometry import MU0_OVER_4PI
from spin_dynamics.nano_mr.statistical import (
    planar_transverse_variance_geometry,
)


ScanMode = Literal["raster", "arbitrary", "sensor_array"]
ImagingResponse = Literal["field", "transverse_variance"]


@dataclass(frozen=True, eq=False)
class NanoMRScan:
    """Ordered scanning-sensor positions or a simultaneous sensor array."""

    positions_lab_nm: np.ndarray
    sensor_axes_lab: np.ndarray
    mode: ScanMode
    raster_shape: tuple[int, int] | None = None
    raster_indices: np.ndarray | None = None
    label: str = ""

    @property
    def measurement_count(self) -> int:
        """Return the number of sensor samples."""

        return int(self.positions_lab_nm.shape[0])

    def reshape_raster(self, values) -> np.ndarray:
        """Map acquisition-ordered values back to ``(y, x)`` raster order."""

        if self.raster_shape is None or self.raster_indices is None:
            raise ValueError("reshape_raster requires a raster scan")
        array = np.asarray(values)
        if array.shape != (self.measurement_count,):
            raise ValueError("values must contain one value per measurement")
        image = np.empty(self.raster_shape, dtype=array.dtype)
        image[self.raster_indices[:, 0], self.raster_indices[:, 1]] = array
        return image


@dataclass(frozen=True, eq=False)
class PlanarVoxelGrid:
    """Rectilinear planar voxel centers and volumes."""

    x_axis_nm: np.ndarray
    y_axis_nm: np.ndarray
    z_nm: float
    thickness_nm: float
    positions_lab_nm: np.ndarray
    voxel_volumes_nm3: np.ndarray
    shape: tuple[int, int]

    def reshape(self, values) -> np.ndarray:
        """Reshape a source vector into ``(y, x)`` image order."""

        array = np.asarray(values)
        if array.size != int(np.prod(self.shape)):
            raise ValueError("values must contain one value per voxel")
        return array.reshape(self.shape)


@dataclass(frozen=True, eq=False)
class DipolarForwardOperator:
    """Linear projected-field or transverse-variance nano-MRI operator."""

    matrix: np.ndarray
    scan: NanoMRScan
    source_positions_lab_nm: np.ndarray
    source_shape: tuple[int, ...]
    response_kind: ImagingResponse
    field_axis_lab: np.ndarray | None
    moment_direction_lab: np.ndarray | None
    minimum_distance_nm: float
    source_quantity: str
    source_units: str
    measurement_units: str

    @property
    def source_count(self) -> int:
        """Return the number of candidate source positions."""

        return int(self.source_positions_lab_nm.shape[0])

    def predict(self, source_amplitudes) -> np.ndarray:
        """Apply the forward operator to field moments or moment variances."""

        amplitudes = np.asarray(source_amplitudes, dtype=np.float64)
        if amplitudes.shape == self.source_shape:
            amplitudes = amplitudes.reshape(-1)
        if amplitudes.shape != (self.source_count,) or not np.all(
            np.isfinite(amplitudes)
        ):
            raise ValueError("source_amplitudes must be finite and match source_shape")
        return self.matrix @ amplitudes


@dataclass(frozen=True, eq=False)
class DepthProfileOperator:
    """Analytic planar-slab field-variance operator."""

    matrix_t2_per_m3: np.ndarray
    sensor_depths_nm: np.ndarray
    layer_edges_nm: np.ndarray
    species: NuclearBathSpecies
    field_tesla: float

    @property
    def source_shape(self) -> tuple[int]:
        """Return the number of depth bins as a one-dimensional shape."""

        return (self.layer_edges_nm.size - 1,)

    def predict(self, number_density_m3) -> np.ndarray:
        """Predict field variance for a piecewise-constant depth profile."""

        density = np.asarray(number_density_m3, dtype=np.float64)
        if density.shape != self.source_shape or not np.all(np.isfinite(density)):
            raise ValueError("number_density_m3 must be finite and match depth bins")
        if np.any(density < 0.0):
            raise ValueError("number_density_m3 must be non-negative")
        return self.matrix_t2_per_m3 @ density


@dataclass(frozen=True, eq=False)
class DensityReconstructionResult:
    """Nonnegative regularized density estimate and linearized uncertainty."""

    density: np.ndarray
    predicted_measurements: np.ndarray
    residual: np.ndarray
    standard_deviation: np.ndarray
    covariance: np.ndarray
    regularization: float
    regularization_order: int
    iterations: int
    converged: bool
    noise_std: float
    effective_degrees_of_freedom: float
    measurement_covariance: np.ndarray | None = None


@dataclass(frozen=True, eq=False)
class PointLocalizationResult:
    """Nonlinear sparse point-source fit and local covariance estimate."""

    positions_lab_nm: np.ndarray
    amplitudes: np.ndarray
    predicted_measurements: np.ndarray
    residual: np.ndarray
    position_standard_deviation_nm: np.ndarray
    amplitude_standard_deviation: np.ndarray
    covariance: np.ndarray
    success: bool
    evaluations: int
    cost: float
    noise_std: float
    measurement_covariance: np.ndarray | None = None


def raster_scan(
    x_axis_nm,
    y_axis_nm,
    *,
    z_nm: float,
    sensor_axis_lab=(0.0, 0.0, 1.0),
    serpentine: bool = True,
    label: str = "raster scan",
) -> NanoMRScan:
    """Return a planar raster with optional serpentine acquisition ordering."""

    x_axis = _strict_axis(x_axis_nm, "x_axis_nm")
    y_axis = _strict_axis(y_axis_nm, "y_axis_nm")
    z_value = float(z_nm)
    if not np.isfinite(z_value):
        raise ValueError("z_nm must be finite")
    positions = []
    indices = []
    for row, y_value in enumerate(y_axis):
        columns = range(x_axis.size)
        if serpentine and row % 2:
            columns = reversed(range(x_axis.size))
        for column in columns:
            positions.append((x_axis[column], y_value, z_value))
            indices.append((row, column))
    return _make_scan(
        positions,
        sensor_axis_lab,
        mode="raster",
        raster_shape=(y_axis.size, x_axis.size),
        raster_indices=np.asarray(indices, dtype=np.int64),
        label=label,
    )


def arbitrary_scan(
    positions_lab_nm,
    *,
    sensor_axes_lab=(0.0, 0.0, 1.0),
    label: str = "arbitrary scan",
) -> NanoMRScan:
    """Return an explicitly ordered scanning-sensor trajectory."""

    return _make_scan(
        positions_lab_nm,
        sensor_axes_lab,
        mode="arbitrary",
        label=label,
    )


def sensor_array(
    positions_lab_nm,
    *,
    sensor_axes_lab=(0.0, 0.0, 1.0),
    label: str = "sensor array",
) -> NanoMRScan:
    """Return simultaneous sensor positions and independently oriented axes."""

    return _make_scan(
        positions_lab_nm,
        sensor_axes_lab,
        mode="sensor_array",
        label=label,
    )


def planar_voxel_grid(
    x_axis_nm,
    y_axis_nm,
    *,
    z_nm: float,
    thickness_nm: float,
) -> PlanarVoxelGrid:
    """Return planar voxel centers with midpoint-rule lateral volumes."""

    x_axis = _strict_axis(x_axis_nm, "x_axis_nm")
    y_axis = _strict_axis(y_axis_nm, "y_axis_nm")
    z_value = float(z_nm)
    thickness = float(thickness_nm)
    if not np.isfinite(z_value):
        raise ValueError("z_nm must be finite")
    if thickness <= 0.0 or not np.isfinite(thickness):
        raise ValueError("thickness_nm must be positive and finite")
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")
    positions = np.column_stack(
        (
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, z_value),
        )
    )
    volumes = (
        _cell_widths(y_axis)[:, np.newaxis]
        * _cell_widths(x_axis)[np.newaxis, :]
        * thickness
    )
    return PlanarVoxelGrid(
        x_axis_nm=x_axis,
        y_axis_nm=y_axis,
        z_nm=z_value,
        thickness_nm=thickness,
        positions_lab_nm=positions,
        voxel_volumes_nm3=volumes.ravel(),
        shape=(y_axis.size, x_axis.size),
    )


def build_dipolar_forward_operator(
    scan: NanoMRScan,
    source_positions_lab_nm,
    *,
    source_shape: tuple[int, ...] | None = None,
    response_kind: ImagingResponse = "transverse_variance",
    field_axis_lab=(0.0, 0.0, 1.0),
    moment_direction_lab=(0.0, 0.0, 1.0),
    minimum_distance_nm: float = 0.1,
) -> DipolarForwardOperator:
    """Build a scanning-sensor or sensor-array dipolar forward matrix.

    ``field`` maps signed magnetic moments in joules per tesla to projected
    field in tesla. ``transverse_variance`` maps per-source transverse magnetic
    moment variance in ``(J/T)^2`` to sensor-axis field variance in ``T^2``.
    """

    if not isinstance(scan, NanoMRScan):
        raise TypeError("scan must be a NanoMRScan")
    sources = _positions(source_positions_lab_nm, "source_positions_lab_nm")
    shape = (sources.shape[0],) if source_shape is None else tuple(source_shape)
    if not shape or int(np.prod(shape)) != sources.shape[0]:
        raise ValueError("source_shape must contain one entry per source position")
    cutoff = float(minimum_distance_nm)
    if cutoff <= 0.0 or not np.isfinite(cutoff):
        raise ValueError("minimum_distance_nm must be positive and finite")
    if response_kind not in {"field", "transverse_variance"}:
        raise ValueError("response_kind must be 'field' or 'transverse_variance'")

    displacement_m = (
        scan.positions_lab_nm[:, np.newaxis, :]
        - sources[np.newaxis, :, :]
    ) * 1.0e-9
    distances_m = np.linalg.norm(displacement_m, axis=2)
    if np.any(distances_m == 0.0):
        raise ValueError("source positions must not coincide with sensor positions")
    directions = displacement_m / distances_m[:, :, np.newaxis]
    effective_distance = np.maximum(distances_m, cutoff * 1.0e-9)
    tensors = (
        3.0
        * directions[:, :, :, np.newaxis]
        * directions[:, :, np.newaxis, :]
        - np.eye(3)
    )
    projected = np.einsum("mi,mnij->mnj", scan.sensor_axes_lab, tensors)

    field_axis = None
    moment_direction = None
    if response_kind == "field":
        moment_direction = unit_vector(
            moment_direction_lab,
            name="moment_direction_lab",
        )
        matrix = (
            MU0_OVER_4PI
            * np.einsum("mnj,j->mn", projected, moment_direction)
            / effective_distance**3
        )
    else:
        field_axis = unit_vector(field_axis_lab, name="field_axis_lab")
        transverse_projector = np.eye(3) - np.outer(field_axis, field_axis)
        transverse = np.einsum("mni,ij->mnj", projected, transverse_projector)
        matrix = (
            MU0_OVER_4PI**2
            * np.sum(transverse**2, axis=2)
            / effective_distance**6
        )
    return DipolarForwardOperator(
        matrix=matrix,
        scan=scan,
        source_positions_lab_nm=sources,
        source_shape=shape,
        response_kind=response_kind,
        field_axis_lab=field_axis,
        moment_direction_lab=moment_direction,
        minimum_distance_nm=cutoff,
        source_quantity=(
            "magnetic moment"
            if response_kind == "field"
            else "transverse magnetic-moment variance"
        ),
        source_units="J/T" if response_kind == "field" else "(J/T)^2",
        measurement_units="T" if response_kind == "field" else "T^2",
    )


def build_voxel_density_forward_operator(
    scan: NanoMRScan,
    grid: PlanarVoxelGrid,
    species: NuclearBathSpecies,
    *,
    field_tesla: float,
    field_axis_lab=(0.0, 0.0, 1.0),
    minimum_distance_nm: float = 0.1,
) -> DipolarForwardOperator:
    """Build a planar number-density to statistical-field-variance operator."""

    if not isinstance(grid, PlanarVoxelGrid):
        raise TypeError("grid must be a PlanarVoxelGrid")
    base = build_dipolar_forward_operator(
        scan,
        grid.positions_lab_nm,
        source_shape=grid.shape,
        response_kind="transverse_variance",
        field_axis_lab=field_axis_lab,
        minimum_distance_nm=minimum_distance_nm,
    )
    per_density_amplitude = nuclear_voxel_variance_amplitudes(
        species,
        field_tesla,
        np.ones(grid.voxel_volumes_nm3.size),
        grid.voxel_volumes_nm3,
    )
    return DipolarForwardOperator(
        matrix=base.matrix * per_density_amplitude[np.newaxis, :],
        scan=base.scan,
        source_positions_lab_nm=base.source_positions_lab_nm,
        source_shape=base.source_shape,
        response_kind=base.response_kind,
        field_axis_lab=base.field_axis_lab,
        moment_direction_lab=base.moment_direction_lab,
        minimum_distance_nm=base.minimum_distance_nm,
        source_quantity="number density",
        source_units="m^-3",
        measurement_units="T^2",
    )


def nuclear_voxel_variance_amplitudes(
    species: NuclearBathSpecies,
    field_tesla: float,
    number_density_m3,
    voxel_volumes_nm3,
) -> np.ndarray:
    """Return per-voxel transverse moment variance in ``(J/T)^2``."""

    if not isinstance(species, NuclearBathSpecies):
        raise TypeError("species must be a NuclearBathSpecies")
    field = float(field_tesla)
    if field < 0.0 or not np.isfinite(field):
        raise ValueError("field_tesla must be finite and non-negative")
    volumes = np.asarray(voxel_volumes_nm3, dtype=np.float64)
    if volumes.ndim != 1 or np.any(volumes <= 0.0) or not np.all(
        np.isfinite(volumes)
    ):
        raise ValueError("voxel_volumes_nm3 must be a finite positive vector")
    density = np.asarray(number_density_m3, dtype=np.float64)
    if density.ndim == 0:
        density = np.full(volumes.size, float(density))
    if density.shape != volumes.shape or np.any(density < 0.0) or not np.all(
        np.isfinite(density)
    ):
        raise ValueError("number_density_m3 must be non-negative and match volumes")
    nuclei = density * volumes * 1.0e-27
    moment_variance = (
        PLANCK_CONSTANT_J_S * species.gamma_hz_per_t
    ) ** 2 * species.transverse_spin_second_moment(field)
    return nuclei * moment_variance


def build_depth_profile_operator(
    sensor_depths_nm,
    layer_edges_nm,
    species: NuclearBathSpecies,
    *,
    field_tesla: float,
    sensor_axis_lab=(0.0, 0.0, 1.0),
    field_axis_lab=(0.0, 0.0, 1.0),
    surface_normal_lab=(0.0, 0.0, 1.0),
) -> DepthProfileOperator:
    """Build an analytic planar-slab density-to-field-variance operator."""

    depths = np.asarray(sensor_depths_nm, dtype=np.float64).reshape(-1)
    edges = np.asarray(layer_edges_nm, dtype=np.float64).reshape(-1)
    if (
        depths.size == 0
        or np.any(depths <= 0.0)
        or not np.all(np.isfinite(depths))
    ):
        raise ValueError("sensor_depths_nm must contain finite positive values")
    if (
        edges.size < 2
        or edges[0] < 0.0
        or np.any(np.diff(edges) <= 0.0)
        or not np.all(np.isfinite(edges))
    ):
        raise ValueError("layer_edges_nm must be finite, non-negative, and increasing")
    if not isinstance(species, NuclearBathSpecies):
        raise TypeError("species must be a NuclearBathSpecies")
    field = float(field_tesla)
    if field < 0.0 or not np.isfinite(field):
        raise ValueError("field_tesla must be finite and non-negative")
    sensor_axis = unit_vector(sensor_axis_lab, name="sensor_axis_lab")
    field_axis = unit_vector(field_axis_lab, name="field_axis_lab")
    surface_normal = unit_vector(surface_normal_lab, name="surface_normal_lab")
    matrix = np.empty((depths.size, edges.size - 1), dtype=np.float64)
    prefactor = (
        MU0_OVER_4PI * PLANCK_CONSTANT_J_S * species.gamma_hz_per_t
    ) ** 2 * species.transverse_spin_second_moment(field)
    for row, depth_nm in enumerate(depths):
        lower = (depth_nm + edges[:-1]) * 1.0e-9
        upper = (depth_nm + edges[1:]) * 1.0e-9
        inverse_cube_range = lower**-3 - upper**-3
        for column, inverse_cube in enumerate(inverse_cube_range):
            matrix[row, column] = prefactor * planar_transverse_variance_geometry(
                sensor_axis,
                field_axis,
                surface_normal,
                float(inverse_cube),
            )
    return DepthProfileOperator(
        matrix_t2_per_m3=matrix,
        sensor_depths_nm=depths,
        layer_edges_nm=edges,
        species=species,
        field_tesla=field,
    )


def reconstruct_nonnegative_density(
    operator: DipolarForwardOperator | DepthProfileOperator | np.ndarray,
    measurements,
    *,
    source_shape: tuple[int, ...] | None = None,
    regularization: float = 1.0e-3,
    regularization_order: int = 1,
    noise_std: float | None = None,
    noise_covariance=None,
    max_iterations: int = 10000,
    tolerance: float = 1.0e-8,
) -> DensityReconstructionResult:
    """Solve nonnegative dimensionless-Tikhonov density inversion.

    The regularization strength is relative to the squared spectral norm of
    the forward matrix. Uncertainty is a local Gaussian approximation to the
    regularized linear model and does not include positivity-boundary bias.
    ``noise_covariance`` enables generalized least squares for correlated,
    heteroscedastic measurements and is mutually exclusive with ``noise_std``.
    """

    matrix, inferred_shape = _matrix_and_shape(operator)
    shape = inferred_shape if source_shape is None else tuple(source_shape)
    if not shape or int(np.prod(shape)) != matrix.shape[1]:
        raise ValueError("source_shape must match the forward-operator columns")
    data = np.asarray(measurements, dtype=np.float64)
    if data.shape != (matrix.shape[0],) or not np.all(np.isfinite(data)):
        raise ValueError("measurements must be finite and match operator rows")
    if noise_std is not None and noise_covariance is not None:
        raise ValueError("provide noise_std or noise_covariance, not both")
    covariance_input = _measurement_covariance(
        noise_covariance,
        matrix.shape[0],
    )
    if covariance_input is None:
        fit_matrix = matrix
        fit_data = data
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_input)
        whitener = (
            eigenvectors / np.sqrt(eigenvalues)[np.newaxis, :]
        ).T
        fit_matrix = whitener @ matrix
        fit_data = whitener @ data
    strength = float(regularization)
    if strength < 0.0 or not np.isfinite(strength):
        raise ValueError("regularization must be finite and non-negative")
    iterations_limit = int(max_iterations)
    if iterations_limit <= 0 or iterations_limit != max_iterations:
        raise ValueError("max_iterations must be a positive integer")
    convergence_tolerance = float(tolerance)
    if convergence_tolerance <= 0.0 or not np.isfinite(convergence_tolerance):
        raise ValueError("tolerance must be positive and finite")

    penalty = _difference_penalty(shape, regularization_order)
    matrix_norm = float(np.linalg.norm(fit_matrix, ord=2))
    if matrix_norm == 0.0:
        raise ValueError("forward operator has zero sensitivity")
    data_scale = max(
        float(np.sqrt(np.mean(fit_data**2))),
        np.finfo(float).tiny,
    )
    least_squares_scale, *_ = np.linalg.lstsq(
        fit_matrix / matrix_norm,
        fit_data / matrix_norm,
        rcond=1.0e-10,
    )
    amplitude_scale = max(
        float(np.max(np.abs(least_squares_scale), initial=0.0)),
        data_scale / matrix_norm,
        np.finfo(float).tiny,
    )
    scaled_matrix = fit_matrix * amplitude_scale / data_scale
    scaled_data = fit_data / data_scale
    spectral_norm = float(np.linalg.norm(scaled_matrix, ord=2))
    normal = scaled_matrix.T @ scaled_matrix
    if penalty.size:
        normal = normal + strength * spectral_norm**2 * (penalty.T @ penalty)
    lipschitz = float(np.linalg.eigvalsh(normal)[-1])
    if lipschitz <= 0.0:
        raise ValueError("regularized system has zero curvature")

    estimate = np.zeros(matrix.shape[1], dtype=np.float64)
    accelerated = estimate.copy()
    momentum = 1.0
    converged = False
    for iteration in range(1, iterations_limit + 1):
        gradient = scaled_matrix.T @ (
            scaled_matrix @ accelerated - scaled_data
        )
        if penalty.size:
            gradient += (
                strength
                * spectral_norm**2
                * (penalty.T @ (penalty @ accelerated))
            )
        updated = np.maximum(accelerated - gradient / lipschitz, 0.0)
        threshold = convergence_tolerance * (
            1.0 + float(np.max(np.abs(updated)))
        )
        if float(np.max(np.abs(updated - estimate))) <= threshold:
            estimate = updated
            converged = True
            break
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        accelerated = updated + (momentum - 1.0) / next_momentum * (
            updated - estimate
        )
        estimate = updated
        momentum = next_momentum

    estimate_physical = estimate * amplitude_scale
    predicted = matrix @ estimate_physical
    residual = data - predicted
    inverse_normal = np.linalg.pinv(normal, rcond=1.0e-12)
    effective_dof = float(
        np.trace(inverse_normal @ (scaled_matrix.T @ scaled_matrix))
    )
    if covariance_input is not None:
        resolved_noise = float(
            np.sqrt(np.mean(np.diag(covariance_input)))
        )
        covariance_scale = amplitude_scale / data_scale
    elif noise_std is None:
        denominator = max(1.0, matrix.shape[0] - effective_dof)
        resolved_noise = float(np.linalg.norm(residual) / np.sqrt(denominator))
        covariance_scale = amplitude_scale * resolved_noise / data_scale
    else:
        resolved_noise = float(noise_std)
        if resolved_noise <= 0.0 or not np.isfinite(resolved_noise):
            raise ValueError("noise_std must be positive and finite")
        covariance_scale = amplitude_scale * resolved_noise / data_scale
    covariance = covariance_scale**2 * inverse_normal
    standard_deviation = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return DensityReconstructionResult(
        density=estimate_physical.reshape(shape),
        predicted_measurements=predicted,
        residual=residual,
        standard_deviation=standard_deviation.reshape(shape),
        covariance=covariance,
        regularization=strength,
        regularization_order=int(regularization_order),
        iterations=iteration,
        converged=converged,
        noise_std=resolved_noise,
        effective_degrees_of_freedom=effective_dof,
        measurement_covariance=(
            None if covariance_input is None else covariance_input.copy()
        ),
    )


def localize_point_sources(
    scan: NanoMRScan,
    measurements,
    initial_positions_lab_nm,
    *,
    response_kind: ImagingResponse = "transverse_variance",
    field_axis_lab=(0.0, 0.0, 1.0),
    moment_direction_lab=(0.0, 0.0, 1.0),
    initial_amplitudes=None,
    bounds_lab_nm=((-np.inf, np.inf),) * 3,
    minimum_distance_nm: float = 0.1,
    noise_std: float | None = None,
    noise_covariance=None,
    max_evaluations: int = 2000,
) -> PointLocalizationResult:
    """Fit sparse point positions and amplitudes by bounded least squares.

    This optional nonlinear workflow requires SciPy. Returned covariance is a
    local Jacobian approximation; multimodal position ambiguity is not folded
    into these standard deviations.
    """

    try:
        from scipy.optimize import least_squares, nnls
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "point-source localization requires SciPy; install "
            "python-spin-dynamics[opt]"
        ) from exc

    positions0 = _positions(initial_positions_lab_nm, "initial_positions_lab_nm")
    data = np.asarray(measurements, dtype=np.float64)
    if data.shape != (scan.measurement_count,) or not np.all(np.isfinite(data)):
        raise ValueError("measurements must be finite and match the scan")
    if noise_std is not None and noise_covariance is not None:
        raise ValueError("provide noise_std or noise_covariance, not both")
    covariance_input = _measurement_covariance(
        noise_covariance,
        scan.measurement_count,
    )
    if covariance_input is None:
        whitener = None
        fit_data = data
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_input)
        whitener = (
            eigenvectors / np.sqrt(eigenvalues)[np.newaxis, :]
        ).T
        fit_data = whitener @ data
    evaluations = int(max_evaluations)
    if evaluations != max_evaluations or evaluations <= 0:
        raise ValueError("max_evaluations must be a positive integer")
    bounds = _position_bounds(bounds_lab_nm)
    point_count = positions0.shape[0]
    lower_positions = np.tile(bounds[:, 0], point_count)
    upper_positions = np.tile(bounds[:, 1], point_count)
    if np.any(positions0.reshape(-1) < lower_positions) or np.any(
        positions0.reshape(-1) > upper_positions
    ):
        raise ValueError("initial positions must lie inside bounds_lab_nm")

    initial_operator = build_dipolar_forward_operator(
        scan,
        positions0,
        response_kind=response_kind,
        field_axis_lab=field_axis_lab,
        moment_direction_lab=moment_direction_lab,
        minimum_distance_nm=minimum_distance_nm,
    )
    fit_initial_matrix = initial_operator.matrix
    if whitener is not None:
        fit_initial_matrix = whitener @ fit_initial_matrix
    if initial_amplitudes is None:
        if response_kind == "transverse_variance":
            amplitudes0, _ = nnls(fit_initial_matrix, fit_data)
        else:
            amplitudes0, *_ = np.linalg.lstsq(
                fit_initial_matrix,
                fit_data,
                rcond=None,
            )
    else:
        amplitudes0 = np.asarray(initial_amplitudes, dtype=np.float64)
        if amplitudes0.shape != (point_count,) or not np.all(
            np.isfinite(amplitudes0)
        ):
            raise ValueError("initial_amplitudes must be finite and match points")
    if response_kind == "transverse_variance" and np.any(amplitudes0 < 0.0):
        raise ValueError("variance amplitudes must be non-negative")
    fallback_amplitude = float(
        np.max(np.abs(data))
        / max(np.max(np.abs(initial_operator.matrix)), np.finfo(float).tiny)
    )
    amplitude_scale = max(
        float(np.max(np.abs(amplitudes0), initial=0.0)),
        fallback_amplitude,
        np.finfo(float).tiny,
    )
    scaled_amplitudes0 = amplitudes0 / amplitude_scale
    residual_scale = 1.0 if whitener is not None else (
        float(noise_std)
        if noise_std is not None
        else max(float(np.sqrt(np.mean(data**2))), np.finfo(float).tiny)
    )
    if residual_scale <= 0.0 or not np.isfinite(residual_scale):
        raise ValueError("noise_std must be positive and finite")

    parameters0 = np.concatenate((positions0.reshape(-1), scaled_amplitudes0))
    amplitude_lower = 0.0 if response_kind == "transverse_variance" else -np.inf
    lower = np.concatenate(
        (lower_positions, np.full(point_count, amplitude_lower))
    )
    upper = np.concatenate(
        (upper_positions, np.full(point_count, np.inf))
    )

    def residual_function(parameters: np.ndarray) -> np.ndarray:
        positions = parameters[: 3 * point_count].reshape(point_count, 3)
        amplitudes = parameters[3 * point_count :] * amplitude_scale
        operator = build_dipolar_forward_operator(
            scan,
            positions,
            response_kind=response_kind,
            field_axis_lab=field_axis_lab,
            moment_direction_lab=moment_direction_lab,
            minimum_distance_nm=minimum_distance_nm,
        )
        raw = operator.matrix @ amplitudes - data
        if whitener is not None:
            return whitener @ raw
        return raw / residual_scale

    fit = least_squares(
        residual_function,
        parameters0,
        bounds=(lower, upper),
        max_nfev=evaluations,
        x_scale="jac",
    )
    fitted_positions = fit.x[: 3 * point_count].reshape(point_count, 3)
    fitted_amplitudes = fit.x[3 * point_count :] * amplitude_scale
    fitted_operator = build_dipolar_forward_operator(
        scan,
        fitted_positions,
        response_kind=response_kind,
        field_axis_lab=field_axis_lab,
        moment_direction_lab=moment_direction_lab,
        minimum_distance_nm=minimum_distance_nm,
    )
    predicted = fitted_operator.matrix @ fitted_amplitudes
    raw_residual = data - predicted
    degrees_of_freedom = max(1, data.size - fit.x.size)
    variance_factor = (
        1.0
        if noise_std is not None or covariance_input is not None
        else float(2.0 * fit.cost / degrees_of_freedom)
    )
    if covariance_input is not None:
        reported_noise = float(
            np.sqrt(np.mean(np.diag(covariance_input)))
        )
    elif noise_std is not None:
        reported_noise = residual_scale
    else:
        reported_noise = float(
            np.linalg.norm(raw_residual) / np.sqrt(degrees_of_freedom)
        )
    covariance_scaled = variance_factor * np.linalg.pinv(
        fit.jac.T @ fit.jac,
        rcond=1.0e-12,
    )
    conversion = np.ones(fit.x.size)
    conversion[3 * point_count :] = amplitude_scale
    covariance = (
        conversion[:, np.newaxis]
        * covariance_scaled
        * conversion[np.newaxis, :]
    )
    standard = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return PointLocalizationResult(
        positions_lab_nm=fitted_positions,
        amplitudes=fitted_amplitudes,
        predicted_measurements=predicted,
        residual=raw_residual,
        position_standard_deviation_nm=standard[: 3 * point_count].reshape(
            point_count,
            3,
        ),
        amplitude_standard_deviation=standard[3 * point_count :],
        covariance=covariance,
        success=bool(fit.success),
        evaluations=int(fit.nfev),
        cost=float(fit.cost),
        noise_std=reported_noise,
        measurement_covariance=(
            None if covariance_input is None else covariance_input.copy()
        ),
    )


def _make_scan(
    positions_lab_nm,
    sensor_axes_lab,
    *,
    mode: ScanMode,
    raster_shape: tuple[int, int] | None = None,
    raster_indices: np.ndarray | None = None,
    label: str,
) -> NanoMRScan:
    positions = _positions(positions_lab_nm, "positions_lab_nm")
    axes = np.asarray(sensor_axes_lab, dtype=np.float64)
    if axes.shape == (3,):
        axes = np.repeat(
            unit_vector(axes, name="sensor_axes_lab")[np.newaxis],
            positions.shape[0],
            axis=0,
        )
    elif axes.shape == positions.shape:
        axes = np.vstack(
            [unit_vector(axis, name="sensor_axes_lab") for axis in axes]
        )
    else:
        raise ValueError("sensor_axes_lab must be length 3 or match positions")
    return NanoMRScan(
        positions_lab_nm=positions,
        sensor_axes_lab=axes,
        mode=mode,
        raster_shape=raster_shape,
        raster_indices=raster_indices,
        label=str(label),
    )


def _matrix_and_shape(
    operator: DipolarForwardOperator | DepthProfileOperator | np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    if isinstance(operator, DipolarForwardOperator):
        return operator.matrix, operator.source_shape
    if isinstance(operator, DepthProfileOperator):
        return operator.matrix_t2_per_m3, operator.source_shape
    matrix = np.asarray(operator, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError("operator must be a finite non-empty matrix")
    return matrix, (matrix.shape[1],)


def _difference_penalty(shape: tuple[int, ...], order: int) -> np.ndarray:
    if order not in {0, 1, 2}:
        raise ValueError("regularization_order must be 0, 1, or 2")
    size = int(np.prod(shape))
    if order == 0:
        return np.eye(size)
    rows = []
    for axis, axis_size in enumerate(shape):
        stencil_size = order + 1
        if axis_size < stencil_size:
            continue
        stencil = np.array([-1.0, 1.0]) if order == 1 else np.array([1.0, -2.0, 1.0])
        other_shape = tuple(
            dimension if index != axis else axis_size - order
            for index, dimension in enumerate(shape)
        )
        for base in np.ndindex(other_shape):
            row = np.zeros(size)
            for offset, coefficient in enumerate(stencil):
                index = list(base)
                index[axis] = base[axis] + offset
                row[np.ravel_multi_index(tuple(index), shape)] = coefficient
            rows.append(row)
    if not rows:
        return np.eye(size)
    return np.vstack(rows)


def _position_bounds(values) -> np.ndarray:
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (3, 2) or np.any(np.isnan(bounds)):
        raise ValueError("bounds_lab_nm must contain three (lower, upper) pairs")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("each position bound must satisfy lower < upper")
    return bounds


def _positions(values, name: str) -> np.ndarray:
    positions = np.asarray(values, dtype=np.float64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or positions.shape[0] == 0
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(f"{name} must be a finite non-empty Nx3 array")
    return positions.copy()


def _strict_axis(values, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain at least two finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _cell_widths(axis: np.ndarray) -> np.ndarray:
    edges = np.empty(axis.size + 1)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return np.diff(edges)


def _measurement_covariance(values, count: int) -> np.ndarray | None:
    if values is None:
        return None
    covariance = np.asarray(values, dtype=np.float64)
    if (
        covariance.shape != (count, count)
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError(
            "noise_covariance must be finite and match operator rows"
        )
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("noise_covariance must be positive definite")
    return covariance


__all__ = [
    "DensityReconstructionResult",
    "DepthProfileOperator",
    "DipolarForwardOperator",
    "ImagingResponse",
    "NanoMRScan",
    "PlanarVoxelGrid",
    "PointLocalizationResult",
    "ScanMode",
    "arbitrary_scan",
    "build_depth_profile_operator",
    "build_dipolar_forward_operator",
    "build_voxel_density_forward_operator",
    "localize_point_sources",
    "nuclear_voxel_variance_amplitudes",
    "planar_voxel_grid",
    "raster_scan",
    "reconstruct_nonnegative_density",
    "sensor_array",
]
