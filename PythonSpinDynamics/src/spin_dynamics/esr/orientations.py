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

    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_theta)
    samples: list[ESROrientationSample] = []
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
                    ESROrientationSample(
                        b0_direction_g=b0_direction,
                        b1_direction_g=b1_direction,
                        weight=float(mu_weight) / (2.0 * n_phi * n_chi),
                    )
                )
    return tuple(samples)


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
    return tuple(
        ESROrientationSample(
            b0_direction_g=sample.b0_direction_g,
            b1_direction_g=sample.b1_direction_g,
            weight=sample.weight / total,
        )
        for sample in samples
    )
