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
