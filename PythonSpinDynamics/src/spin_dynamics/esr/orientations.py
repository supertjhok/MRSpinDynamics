"""Single-crystal and powder orientation helpers for ESR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit_vector(
    vector: np.ndarray | list[float] | tuple[float, float, float],
) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError("orientation vectors must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0:
        raise ValueError("orientation vectors must be non-zero")
    return out / norm


def spherical_direction(alpha: float, beta: float) -> np.ndarray:
    """Return a unit vector from azimuth ``alpha`` and polar angle ``beta``."""

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
class ESROrientationSample:
    """One local ``g``-tensor orientation relative to lab static and RF fields."""

    b0_direction_g: np.ndarray
    weight: float = 1.0
    b1_direction_g: np.ndarray | None = None

    def __post_init__(self) -> None:
        b0_direction_g = _unit_vector(self.b0_direction_g)
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be non-negative and finite")
        b1_direction_g = None
        if self.b1_direction_g is not None:
            b1_direction_g = _unit_vector(self.b1_direction_g)
        object.__setattr__(self, "b0_direction_g", b0_direction_g)
        object.__setattr__(self, "b1_direction_g", b1_direction_g)
        object.__setattr__(self, "weight", weight)


def single_crystal_orientation(
    alpha: float = 0.0,
    beta: float = 0.0,
    *,
    b1_alpha: float | None = None,
    b1_beta: float | None = None,
) -> tuple[ESROrientationSample, ...]:
    """Return a one-sample ESR orientation ensemble."""

    b1_direction = None
    if b1_alpha is not None or b1_beta is not None:
        if b1_alpha is None or b1_beta is None:
            raise ValueError("b1_alpha and b1_beta must be supplied together")
        b1_direction = spherical_direction(b1_alpha, b1_beta)
    return (
        ESROrientationSample(
            b0_direction_g=spherical_direction(alpha, beta),
            b1_direction_g=b1_direction,
        ),
    )


def _trusted_sample(
    b0_direction_g: np.ndarray, weight: float, b1_direction_g: np.ndarray | None
) -> ESROrientationSample:
    """Build a sample from already validated unit vectors and weight.

    Powder grids hold 10^5 samples, for which ``__post_init__`` validation
    dominates the cost of a spectrum.
    """

    sample = object.__new__(ESROrientationSample)
    object.__setattr__(sample, "b0_direction_g", b0_direction_g)
    object.__setattr__(sample, "weight", weight)
    object.__setattr__(sample, "b1_direction_g", b1_direction_g)
    return sample


def _perpendicular_basis(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors perpendicular to each row of ``directions``."""

    reference = np.where(
        (np.abs(directions[:, 2]) < 0.9)[:, None],
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    first = np.cross(directions, reference)
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(directions, first)
    return first, second


def powder_average_grid(
    n_theta: int = 16,
    n_phi: int = 32,
    n_chi: int = 8,
    *,
    b1_b0_angle: float = np.pi / 2.0,
) -> tuple[ESROrientationSample, ...]:
    """Return a normalized ESR powder grid with correlated lab B0 and B1 axes."""

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    n_chi = int(n_chi)
    if n_theta <= 0 or n_phi <= 0 or n_chi <= 0:
        raise ValueError("n_theta, n_phi, and n_chi must be positive")
    b1_b0_angle = float(b1_b0_angle)
    if not np.isfinite(b1_b0_angle):
        raise ValueError("b1_b0_angle must be finite")

    # Samples run over theta, then phi, then chi (the last fastest).
    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    beta = np.arccos(mu_values)[:, None]
    alpha = (2.0 * np.pi * np.arange(n_phi) / n_phi)[None, :]
    b0 = np.stack(
        np.broadcast_arrays(
            np.cos(alpha) * np.sin(beta), np.sin(alpha) * np.sin(beta), np.cos(beta)
        ),
        axis=-1,
    ).reshape(-1, 3)
    e1, e2 = _perpendicular_basis(b0)
    chi = (2.0 * np.pi * np.arange(n_chi) / n_chi)[None, :, None]
    perpendicular = np.cos(chi) * e1[:, None, :] + np.sin(chi) * e2[:, None, :]
    b1 = (
        np.cos(b1_b0_angle) * b0[:, None, :] + np.sin(b1_b0_angle) * perpendicular
    ).reshape(-1, 3)
    b0 = np.repeat(b0, n_chi, axis=0)
    b0 = b0 / np.linalg.norm(b0, axis=1, keepdims=True)
    b1 = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    weights = np.repeat(mu_weights / (2.0 * n_phi * n_chi), n_phi * n_chi)
    return tuple(
        _trusted_sample(b0[index], float(weights[index]), b1[index])
        for index in range(b0.shape[0])
    )


def normalize_orientations(
    orientations: tuple[ESROrientationSample, ...] | list[ESROrientationSample],
) -> tuple[ESROrientationSample, ...]:
    """Return ESR orientation samples with weights normalized to unity."""

    samples = tuple(orientations)
    if not samples:
        raise ValueError("at least one orientation sample is required")
    total = sum(sample.weight for sample in samples)
    if total <= 0:
        raise ValueError("orientation weights must have positive sum")
    if not np.isfinite(total):
        raise ValueError("orientation weights must have a finite sum")
    # ESROrientationSample instances are already validated; rescaling keeps
    # them valid.
    return tuple(
        _trusted_sample(sample.b0_direction_g, sample.weight / total, sample.b1_direction_g)
        if isinstance(sample, ESROrientationSample)
        else ESROrientationSample(
            b0_direction_g=sample.b0_direction_g,
            b1_direction_g=sample.b1_direction_g,
            weight=sample.weight / total,
        )
        for sample in samples
    )
