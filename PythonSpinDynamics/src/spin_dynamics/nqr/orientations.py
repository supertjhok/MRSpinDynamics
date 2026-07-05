"""Single-crystal and powder orientation helpers for NQR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_vector(vector: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError("orientation vectors must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0:
        raise ValueError("orientation vectors must be non-zero")
    return out / norm


def spherical_direction(alpha: float, beta: float) -> np.ndarray:
    """Return a unit vector from azimuth `alpha` and polar angle `beta`."""

    alpha = float(alpha)
    beta = float(beta)
    return np.array(
        [
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha) * np.sin(beta),
            np.cos(beta),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class OrientationSample:
    """One local EFG orientation relative to lab RF and static fields."""

    b1_direction_pas: np.ndarray
    weight: float = 1.0
    b0_direction_pas: np.ndarray | None = None

    def __post_init__(self) -> None:
        b1_direction_pas = _unit_vector(self.b1_direction_pas)
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be non-negative and finite")
        b0_direction_pas = None
        if self.b0_direction_pas is not None:
            b0_direction_pas = _unit_vector(self.b0_direction_pas)
        object.__setattr__(self, "b1_direction_pas", b1_direction_pas)
        object.__setattr__(self, "b0_direction_pas", b0_direction_pas)
        object.__setattr__(self, "weight", weight)


def single_crystal_orientation(
    alpha: float,
    beta: float,
    *,
    b0_alpha: float | None = None,
    b0_beta: float | None = None,
) -> tuple[OrientationSample, ...]:
    """Return a one-sample orientation ensemble."""

    b0_direction = None
    if b0_alpha is not None or b0_beta is not None:
        if b0_alpha is None or b0_beta is None:
            raise ValueError("b0_alpha and b0_beta must be supplied together")
        b0_direction = spherical_direction(b0_alpha, b0_beta)
    return (
        OrientationSample(
            b1_direction_pas=spherical_direction(alpha, beta),
            b0_direction_pas=b0_direction,
        ),
    )


def powder_average_grid(n_theta: int = 16, n_phi: int = 32) -> tuple[OrientationSample, ...]:
    """Return a normalized spherical powder-average grid."""

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta <= 0 or n_phi <= 0:
        raise ValueError("n_theta and n_phi must be positive")
    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    samples: list[OrientationSample] = []
    for mu, mu_weight in zip(mu_values, mu_weights):
        beta = float(np.arccos(mu))
        for phi_idx in range(n_phi):
            alpha = 2.0 * np.pi * phi_idx / n_phi
            samples.append(
                OrientationSample(
                    b1_direction_pas=spherical_direction(alpha, beta),
                    weight=float(mu_weight) / (2.0 * n_phi),
                )
            )
    return tuple(samples)


@dataclass(frozen=True)
class OrientationFrame:
    """A full crystallite orientation as an orthonormal lab-to-PAS frame.

    ``axes`` is a ``(3, 3)`` orthonormal matrix whose columns are the lab
    x/y/z coil axes imaged into the EFG principal-axis system. A single linear
    coil only needs one direction (:class:`OrientationSample`), but multi-coil
    excitation (circular / bi- / tri-axial) couples to two or three orthogonal
    lab axes at once, so the powder average must run over the full SO(3)
    orientation -- hence a frame rather than a single vector.
    """

    axes: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        axes = np.asarray(self.axes, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(axes)):
            raise ValueError("frame axes must be finite")
        if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-9):
            raise ValueError("frame axes must be orthonormal")
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be non-negative and finite")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "weight", weight)

    @property
    def x(self) -> np.ndarray:
        """Lab x coil axis in the PAS (first column)."""
        return self.axes[:, 0]

    @property
    def y(self) -> np.ndarray:
        """Lab y coil axis in the PAS (second column)."""
        return self.axes[:, 1]

    @property
    def z(self) -> np.ndarray:
        """Lab z coil axis in the PAS (third column)."""
        return self.axes[:, 2]


def _perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(direction[2])) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    first = np.cross(direction, reference)
    first = first / np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def b0_b1_powder_average_grid(
    n_theta: int = 12,
    n_phi: int = 24,
    n_chi: int = 8,
    *,
    b1_b0_angle: float = np.pi / 2.0,
) -> tuple[OrientationSample, ...]:
    """Return a powder grid with correlated lab B0 and RF B1 directions."""

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    n_chi = int(n_chi)
    if n_theta <= 0 or n_phi <= 0 or n_chi <= 0:
        raise ValueError("n_theta, n_phi, and n_chi must be positive")
    b1_b0_angle = float(b1_b0_angle)
    if not np.isfinite(b1_b0_angle):
        raise ValueError("b1_b0_angle must be finite")

    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    samples: list[OrientationSample] = []
    for mu, mu_weight in zip(mu_values, mu_weights):
        beta = float(np.arccos(mu))
        for phi_idx in range(n_phi):
            alpha = 2.0 * np.pi * phi_idx / n_phi
            b0_direction = spherical_direction(alpha, beta)
            e1, e2 = _perpendicular_basis(b0_direction)
            for chi_idx in range(n_chi):
                chi = 2.0 * np.pi * chi_idx / n_chi
                perpendicular = np.cos(chi) * e1 + np.sin(chi) * e2
                b1_direction = (
                    np.cos(b1_b0_angle) * b0_direction
                    + np.sin(b1_b0_angle) * perpendicular
                )
                samples.append(
                    OrientationSample(
                        b1_direction_pas=b1_direction,
                        b0_direction_pas=b0_direction,
                        weight=float(mu_weight) / (2.0 * n_phi * n_chi),
                    )
                )
    return tuple(samples)


def powder_frame_grid(
    n_theta: int = 12, n_phi: int = 24, n_chi: int = 8
) -> tuple[OrientationFrame, ...]:
    """Return a normalized SO(3) powder grid of full lab-to-PAS frames.

    Multi-coil excitation (circular / bi- / tri-axial) couples to two or three
    orthogonal lab axes, so the powder average runs over the full crystallite
    orientation rather than a single axis. Each frame is built from a primary
    axis on a Gauss-Legendre ``cos(beta)`` x uniform ``alpha`` sphere (the same
    quadrature as :func:`powder_average_grid`) and a uniform in-plane rotation
    ``chi`` of the perpendicular pair (as in :func:`b0_b1_powder_average_grid`),
    giving the third Euler angle. Frame columns are the lab x/y/z axes in the
    PAS; weights sum to unity.
    """

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    n_chi = int(n_chi)
    if n_theta <= 0 or n_phi <= 0 or n_chi <= 0:
        raise ValueError("n_theta, n_phi, and n_chi must be positive")

    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    frames: list[OrientationFrame] = []
    for mu, mu_weight in zip(mu_values, mu_weights):
        beta = float(np.arccos(mu))
        for phi_idx in range(n_phi):
            alpha = 2.0 * np.pi * phi_idx / n_phi
            z_axis = spherical_direction(alpha, beta)
            e1, e2 = _perpendicular_basis(z_axis)
            for chi_idx in range(n_chi):
                chi = 2.0 * np.pi * chi_idx / n_chi
                x_axis = np.cos(chi) * e1 + np.sin(chi) * e2
                y_axis = -np.sin(chi) * e1 + np.cos(chi) * e2
                axes = np.column_stack([x_axis, y_axis, z_axis])
                frames.append(
                    OrientationFrame(
                        axes=axes,
                        weight=float(mu_weight) / (2.0 * n_phi * n_chi),
                    )
                )
    return tuple(frames)


def b0_powder_average_grid(
    n_theta: int = 16,
    n_phi: int = 32,
    *,
    b1_direction_pas: np.ndarray | list[float] | tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[OrientationSample, ...]:
    """Return a powder grid over static-field directions in the PAS."""

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    if n_theta <= 0 or n_phi <= 0:
        raise ValueError("n_theta and n_phi must be positive")
    b1_direction = _unit_vector(b1_direction_pas)
    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    samples: list[OrientationSample] = []
    for mu, mu_weight in zip(mu_values, mu_weights):
        beta = float(np.arccos(mu))
        for phi_idx in range(n_phi):
            alpha = 2.0 * np.pi * phi_idx / n_phi
            samples.append(
                OrientationSample(
                    b1_direction_pas=b1_direction,
                    b0_direction_pas=spherical_direction(alpha, beta),
                    weight=float(mu_weight) / (2.0 * n_phi),
                )
            )
    return tuple(samples)


def normalize_orientations(
    orientations: tuple[OrientationSample, ...] | list[OrientationSample],
) -> tuple[OrientationSample, ...]:
    """Return orientation samples with weights normalized to unity."""

    samples = tuple(orientations)
    if not samples:
        raise ValueError("at least one orientation sample is required")
    total = sum(sample.weight for sample in samples)
    if total <= 0:
        raise ValueError("orientation weights must have positive sum")
    return tuple(
        OrientationSample(
            b1_direction_pas=sample.b1_direction_pas,
            b0_direction_pas=sample.b0_direction_pas,
            weight=sample.weight / total,
        )
        for sample in samples
    )
