"""Nonlinear imaging with programmable electropermanent-magnet arrays.

Each retained-remanence state produces a known spatial field profile.  After
receiver demodulation removes the state-wide reference frequency, the remaining
phase is a generally nonlinear encoding pattern.  The resulting dense encoding
matrix is reconstructed directly rather than passed through an FFT.

The module also provides a compact two-tissue phantom for examples and
regression tests.  Its relaxation values are illustrative, not a clinical
tissue-property database.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.electropermanent_array import (
    ElectropermanentArrayFieldBasis,
)
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON


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
class TissuePhantom2D:
    """Simple proton-density/T1/T2 phantom on a Cartesian imaging plane."""

    x_m: np.ndarray
    y_m: np.ndarray
    proton_density: np.ndarray
    t1_s: np.ndarray
    t2_s: np.ndarray
    tissue_labels: np.ndarray
    label_names: tuple[str, ...] = ("background", "tissue", "target")

    def __post_init__(self) -> None:
        x_axis = _strict_axis(self.x_m, "x_m")
        y_axis = _strict_axis(self.y_m, "y_m")
        shape = (y_axis.size, x_axis.size)
        density = np.asarray(self.proton_density, dtype=np.float64)
        t1_map = np.asarray(self.t1_s, dtype=np.float64)
        t2_map = np.asarray(self.t2_s, dtype=np.float64)
        labels = np.asarray(self.tissue_labels)
        for name, values in (
            ("proton_density", density),
            ("t1_s", t1_map),
            ("t2_s", t2_map),
        ):
            if values.shape != shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must be finite and match the axis shape")
        if labels.shape != shape or np.any(labels < 0):
            raise ValueError("tissue_labels must be non-negative and match the axes")
        if np.any(density < 0.0):
            raise ValueError("proton_density must be non-negative")
        if np.any(t1_map <= 0.0) or np.any(t2_map <= 0.0):
            raise ValueError("t1_s and t2_s must be positive")
        if len(self.label_names) <= int(np.max(labels)):
            raise ValueError("label_names must cover every tissue label")
        object.__setattr__(self, "x_m", x_axis)
        object.__setattr__(self, "y_m", y_axis)
        object.__setattr__(self, "proton_density", density)
        object.__setattr__(self, "t1_s", t1_map)
        object.__setattr__(self, "t2_s", t2_map)
        object.__setattr__(self, "tissue_labels", labels.astype(np.int64))
        object.__setattr__(self, "label_names", tuple(self.label_names))

    @property
    def shape(self) -> tuple[int, int]:
        """Image shape in ``(y, x)`` order."""

        return self.proton_density.shape

    @property
    def points_m(self) -> np.ndarray:
        """Flattened ``(x, y, 0)`` voxel centers matching NumPy row order."""

        x_grid, y_grid = np.meshgrid(self.x_m, self.y_m, indexing="xy")
        return np.column_stack(
            (x_grid.ravel(), y_grid.ravel(), np.zeros(x_grid.size))
        )

    @property
    def target_mask(self) -> np.ndarray:
        """Boolean mask for the image-guided target inclusion."""

        return self.tissue_labels == 2

    def spin_echo_image(
        self,
        *,
        repetition_time_s: float,
        echo_time_s: float,
    ) -> np.ndarray:
        """Return ideal spin-echo contrast ``rho*(1-exp(-TR/T1))*exp(-TE/T2)``."""

        if not np.isfinite(repetition_time_s) or repetition_time_s <= 0.0:
            raise ValueError("repetition_time_s must be finite and positive")
        if not np.isfinite(echo_time_s) or echo_time_s < 0.0:
            raise ValueError("echo_time_s must be finite and non-negative")
        return (
            self.proton_density
            * (1.0 - np.exp(-float(repetition_time_s) / self.t1_s))
            * np.exp(-float(echo_time_s) / self.t2_s)
        )


def simple_tissue_phantom(
    matrix_size: int = 16,
    *,
    field_of_view_m: float = 0.040,
) -> TissuePhantom2D:
    """Return an illustrative soft-tissue ellipse with one off-center target."""

    if int(matrix_size) != matrix_size or matrix_size < 8:
        raise ValueError("matrix_size must be an integer of at least 8")
    if not np.isfinite(field_of_view_m) or field_of_view_m <= 0.0:
        raise ValueError("field_of_view_m must be finite and positive")
    axis = np.linspace(-field_of_view_m / 2.0, field_of_view_m / 2.0, int(matrix_size))
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")

    density = np.zeros_like(x_grid)
    t1_map = np.full_like(x_grid, 1.0)
    t2_map = np.full_like(x_grid, 0.080)
    labels = np.zeros_like(x_grid, dtype=np.int64)

    body = (x_grid / (0.43 * field_of_view_m)) ** 2 + (
        y_grid / (0.38 * field_of_view_m)
    ) ** 2 <= 1.0
    density[body] = 0.78
    t1_map[body] = 0.72
    t2_map[body] = 0.075
    labels[body] = 1

    target = (
        (x_grid - 0.15 * field_of_view_m) ** 2
        + (y_grid + 0.10 * field_of_view_m) ** 2
        <= (0.105 * field_of_view_m) ** 2
    ) & body
    density[target] = 1.00
    t1_map[target] = 1.10
    t2_map[target] = 0.120
    labels[target] = 2

    return TissuePhantom2D(axis, axis.copy(), density, t1_map, t2_map, labels)


@dataclass(frozen=True)
class NonlinearEPMEncoding:
    """Known EPM field states and their dense nonlinear encoding matrix."""

    basis: ElectropermanentArrayFieldBasis
    image_shape: tuple[int, int]
    remanence_states_t: np.ndarray
    projected_fields_t: np.ndarray
    phase_rad: np.ndarray
    matrix: np.ndarray
    field_direction: tuple[float, float, float]
    phase_encoding_s: float
    gamma_rad_s_t: float
    reference_point_index: int

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape)
        if len(shape) != 2 or any(value < 1 for value in shape):
            raise ValueError("image_shape must contain two positive dimensions")
        point_count = self.basis.points_m.shape[0]
        if np.prod(shape) != point_count:
            raise ValueError("image_shape must match the number of basis points")
        states = np.asarray(self.remanence_states_t, dtype=np.float64)
        fields = np.asarray(self.projected_fields_t, dtype=np.float64)
        phase = np.asarray(self.phase_rad, dtype=np.float64)
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        control_count = len(self.basis.array.programmable_elements)
        if states.ndim != 2 or states.shape[1] != control_count:
            raise ValueError("remanence_states_t has an invalid shape")
        expected = (states.shape[0], point_count)
        for name, values in (
            ("projected_fields_t", fields),
            ("phase_rad", phase),
            ("matrix", matrix),
        ):
            if values.shape != expected or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} has an invalid shape or non-finite values")
        if not np.isfinite(self.phase_encoding_s) or self.phase_encoding_s <= 0.0:
            raise ValueError("phase_encoding_s must be finite and positive")
        if not np.isfinite(self.gamma_rad_s_t) or self.gamma_rad_s_t <= 0.0:
            raise ValueError("gamma_rad_s_t must be finite and positive")
        if not 0 <= int(self.reference_point_index) < point_count:
            raise ValueError("reference_point_index is out of range")
        unit = _unit_vector(self.field_direction, "field_direction")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "remanence_states_t", states)
        object.__setattr__(self, "projected_fields_t", fields)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "field_direction", tuple(float(v) for v in unit))
        object.__setattr__(self, "reference_point_index", int(self.reference_point_index))

    @property
    def encoding_count(self) -> int:
        """Number of retained-state acquisitions."""

        return self.matrix.shape[0]

    @property
    def condition_number(self) -> float:
        """Condition number of the equivalent real-valued inverse system."""

        real_system = np.vstack((self.matrix.real, self.matrix.imag))
        if real_system.shape[0] < real_system.shape[1]:
            return float(np.inf)
        return float(np.linalg.cond(real_system))

    @property
    def field_span_t(self) -> np.ndarray:
        """Peak-to-peak projected field across the image for every state."""

        return np.ptp(self.projected_fields_t, axis=1)

    def encode(self, image: np.ndarray) -> np.ndarray:
        """Encode one real image into a complex acquisition vector."""

        values = np.asarray(image, dtype=np.float64)
        if values.shape != self.image_shape or np.any(~np.isfinite(values)):
            raise ValueError("image must be finite and match image_shape")
        return self.matrix @ values.ravel()


def random_epm_encoding_states(
    basis: ElectropermanentArrayFieldBasis,
    encoding_count: int,
    *,
    amplitude_fraction: float = 0.60,
    seed: int = 0,
    include_reference: bool = True,
) -> np.ndarray:
    """Generate deterministic bounded Rademacher retained-remanence states."""

    if int(encoding_count) != encoding_count or encoding_count < 2:
        raise ValueError("encoding_count must be an integer of at least 2")
    if (
        not np.isfinite(amplitude_fraction)
        or amplitude_fraction <= 0.0
        or amplitude_fraction > 1.0
    ):
        raise ValueError("amplitude_fraction must be in (0, 1]")
    rng = np.random.default_rng(seed)
    limits = basis.array.remanence_limits_t
    states = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(int(encoding_count), limits.size),
    )
    states *= float(amplitude_fraction) * limits[None, :]
    if include_reference:
        states[0] = 0.0
    return states


def build_epm_nonlinear_encoding(
    basis: ElectropermanentArrayFieldBasis,
    remanence_states_t: np.ndarray,
    *,
    image_shape: tuple[int, int],
    phase_encoding_s: float = 300e-6,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    gamma_rad_s_t: float = GAMMA_PROTON,
    reference_point_index: int | None = None,
) -> NonlinearEPMEncoding:
    """Build nonlinear phase encoding from retained states and a cached basis."""

    states = np.asarray(remanence_states_t, dtype=np.float64)
    control_count = len(basis.array.programmable_elements)
    if (
        states.ndim != 2
        or states.shape[0] < 1
        or states.shape[1] != control_count
        or np.any(~np.isfinite(states))
    ):
        raise ValueError("remanence_states_t must have shape (encoding, controls)")
    limits = basis.array.remanence_limits_t
    if np.any(np.abs(states) > limits[None, :] * (1.0 + 1e-12)):
        raise ValueError("remanence_states_t exceeds an array element limit")
    if not np.isfinite(phase_encoding_s) or phase_encoding_s <= 0.0:
        raise ValueError("phase_encoding_s must be finite and positive")
    if not np.isfinite(gamma_rad_s_t) or gamma_rad_s_t <= 0.0:
        raise ValueError("gamma_rad_s_t must be finite and positive")
    unit = _unit_vector(field_direction, "field_direction")
    fixed, controls = basis.projected_matrix(unit)
    projected_fields = fixed[None, :] + states @ controls.T

    if reference_point_index is None:
        center = np.mean(basis.points_m, axis=0)
        reference_index = int(np.argmin(np.linalg.norm(basis.points_m - center, axis=1)))
    else:
        reference_index = int(reference_point_index)
        if not 0 <= reference_index < basis.points_m.shape[0]:
            raise ValueError("reference_point_index is out of range")
    demodulated = projected_fields - projected_fields[:, [reference_index]]
    phase = float(gamma_rad_s_t) * float(phase_encoding_s) * demodulated
    matrix = np.exp(-1j * phase) / np.sqrt(states.shape[0])
    return NonlinearEPMEncoding(
        basis=basis,
        image_shape=image_shape,
        remanence_states_t=states,
        projected_fields_t=projected_fields,
        phase_rad=phase,
        matrix=matrix,
        field_direction=tuple(float(value) for value in unit),
        phase_encoding_s=float(phase_encoding_s),
        gamma_rad_s_t=float(gamma_rad_s_t),
        reference_point_index=reference_index,
    )


@dataclass(frozen=True)
class EPMNonlinearImagingResult:
    """Signal, reconstruction, and diagnostics for nonlinear EPM imaging."""

    encoding: NonlinearEPMEncoding
    expected_image: np.ndarray
    signal_clean: np.ndarray
    signal: np.ndarray
    reconstructed_image: np.ndarray
    regularization: float
    nonnegative: bool
    iterations: int
    converged: bool
    noise_std: float
    snr_db: float | None

    @property
    def nrmse(self) -> float:
        """Normalized root-mean-square image error."""

        denominator = float(np.linalg.norm(self.expected_image))
        if denominator == 0.0:
            return float(np.linalg.norm(self.reconstructed_image))
        return float(np.linalg.norm(self.reconstructed_image - self.expected_image) / denominator)

    @property
    def signal_residual(self) -> np.ndarray:
        """Measured signal minus the signal predicted by the reconstruction."""

        return self.signal - self.encoding.encode(self.reconstructed_image)


def reconstruct_epm_nonlinear_image(
    encoding: NonlinearEPMEncoding,
    signal: Sequence[complex] | np.ndarray,
    *,
    regularization: float = 1e-4,
    nonnegative: bool = True,
    max_iterations: int = 4000,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, int, bool]:
    """Reconstruct a real image with dimensionless Tikhonov regularization."""

    measured = np.asarray(signal, dtype=np.complex128)
    if measured.shape != (encoding.encoding_count,) or np.any(~np.isfinite(measured)):
        raise ValueError("signal must be finite and match the encoding count")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if int(max_iterations) != max_iterations or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")

    system = np.vstack((encoding.matrix.real, encoding.matrix.imag))
    data = np.concatenate((measured.real, measured.imag))
    singular_values = np.linalg.svd(system, compute_uv=False)
    spectral_sq = float(singular_values[0] ** 2)
    penalty = float(regularization) * spectral_sq

    if not nonnegative:
        normal = system.T @ system + penalty * np.eye(system.shape[1])
        reconstruction = np.linalg.solve(normal, system.T @ data)
        return reconstruction.reshape(encoding.image_shape), 1, True

    lipschitz = spectral_sq + penalty
    if lipschitz == 0.0:
        raise ValueError("encoding system has zero sensitivity")
    estimate = np.zeros(system.shape[1], dtype=np.float64)
    accelerated = estimate.copy()
    momentum = 1.0
    converged = False
    for iterations in range(1, int(max_iterations) + 1):
        gradient = system.T @ (system @ accelerated - data) + penalty * accelerated
        updated = np.maximum(accelerated - gradient / lipschitz, 0.0)
        threshold = float(tolerance) * (1.0 + float(np.max(np.abs(updated))))
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
    return estimate.reshape(encoding.image_shape), iterations, converged


def run_epm_nonlinear_imaging(
    encoding: NonlinearEPMEncoding,
    expected_image: np.ndarray,
    *,
    regularization: float = 1e-4,
    nonnegative: bool = True,
    snr_db: float | None = None,
    seed: int = 0,
    max_iterations: int = 4000,
    tolerance: float = 1e-8,
) -> EPMNonlinearImagingResult:
    """Encode, optionally add complex noise, and reconstruct one EPM image."""

    image = np.asarray(expected_image, dtype=np.float64)
    if image.shape != encoding.image_shape or np.any(~np.isfinite(image)):
        raise ValueError("expected_image must be finite and match the encoding")
    signal_clean = encoding.encode(image)
    noise_std = 0.0
    signal = signal_clean.copy()
    if snr_db is not None:
        if not np.isfinite(snr_db) or snr_db <= 0.0:
            raise ValueError("snr_db must be finite and positive")
        rms = float(np.sqrt(np.mean(np.abs(signal_clean) ** 2)))
        noise_std = rms * 10.0 ** (-float(snr_db) / 20.0) / np.sqrt(2.0)
        rng = np.random.default_rng(seed)
        signal += noise_std * (
            rng.normal(size=signal.size) + 1j * rng.normal(size=signal.size)
        )
    reconstructed, iterations, converged = reconstruct_epm_nonlinear_image(
        encoding,
        signal,
        regularization=regularization,
        nonnegative=nonnegative,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return EPMNonlinearImagingResult(
        encoding=encoding,
        expected_image=image,
        signal_clean=signal_clean,
        signal=signal,
        reconstructed_image=reconstructed,
        regularization=float(regularization),
        nonnegative=bool(nonnegative),
        iterations=iterations,
        converged=converged,
        noise_std=noise_std,
        snr_db=None if snr_db is None else float(snr_db),
    )
