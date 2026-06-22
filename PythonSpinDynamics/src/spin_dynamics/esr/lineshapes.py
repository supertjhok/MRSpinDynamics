"""ESR absorption and derivative lineshape helpers."""

from __future__ import annotations

import numpy as np


def gaussian_lineshape(
    axis: np.ndarray,
    center: float,
    width: float,
    *,
    derivative: bool = False,
) -> np.ndarray:
    """Return a unit-height Gaussian absorption line or its first derivative."""

    axis = np.asarray(axis, dtype=np.float64)
    width = _validate_width(width)
    offset = axis - float(center)
    absorption = np.exp(-0.5 * (offset / width) ** 2)
    if derivative:
        return -(offset / width**2) * absorption
    return absorption


def lorentzian_lineshape(
    axis: np.ndarray,
    center: float,
    width: float,
    *,
    derivative: bool = False,
) -> np.ndarray:
    """Return a unit-height Lorentzian absorption line or its first derivative."""

    axis = np.asarray(axis, dtype=np.float64)
    width = _validate_width(width)
    offset = axis - float(center)
    reduced = offset / width
    absorption = 1.0 / (1.0 + reduced**2)
    if derivative:
        return -2.0 * offset / width**2 / (1.0 + reduced**2) ** 2
    return absorption


def spectrum_from_lines(
    axis: np.ndarray,
    centers: np.ndarray | list[float] | tuple[float, ...],
    intensities: np.ndarray | list[float] | tuple[float, ...],
    *,
    width: float,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
) -> np.ndarray:
    """Return a broadened ESR spectrum from weighted line centers."""

    axis = np.asarray(axis, dtype=np.float64).reshape(-1)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float64).reshape(-1)
    if centers.size != intensities.size:
        raise ValueError("centers and intensities must have the same length")
    if not np.all(np.isfinite(axis)):
        raise ValueError("axis must be finite")
    if not np.all(np.isfinite(centers)):
        raise ValueError("centers must be finite")
    if not np.all(np.isfinite(intensities)):
        raise ValueError("intensities must be finite")

    derivative = _is_derivative_mode(detection_mode)
    profile = _profile_function(lineshape)
    out = np.zeros(axis.size, dtype=np.float64)
    for center, intensity in zip(centers, intensities):
        out += float(intensity) * profile(
            axis,
            float(center),
            width,
            derivative=derivative,
        )
    return out


def _validate_width(width: float) -> float:
    width = float(width)
    if width <= 0 or not np.isfinite(width):
        raise ValueError("width must be positive and finite")
    return width


def _is_derivative_mode(detection_mode: str) -> bool:
    if detection_mode == "absorption":
        return False
    if detection_mode == "derivative":
        return True
    raise ValueError("detection_mode must be 'absorption' or 'derivative'")


def _profile_function(lineshape: str):
    if lineshape == "gaussian":
        return gaussian_lineshape
    if lineshape == "lorentzian":
        return lorentzian_lineshape
    raise ValueError("lineshape must be 'gaussian' or 'lorentzian'")
