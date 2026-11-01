"""Private validation and sampling helpers shared by the ``electropermanent_*`` workflows."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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


def _positive(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = float(value)
    valid = result >= 0.0 if allow_zero else result > 0.0
    if not np.isfinite(result) or not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def _bilinear_sample(
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample a ``(y, x)`` map at ``(n, 2)`` positions, clamped to the grid."""

    x = np.clip(positions[:, 0], x_axis[0], x_axis[-1])
    y = np.clip(positions[:, 1], y_axis[0], y_axis[-1])
    ix = np.clip(np.searchsorted(x_axis, x, side="right") - 1, 0, x_axis.size - 2)
    iy = np.clip(np.searchsorted(y_axis, y, side="right") - 1, 0, y_axis.size - 2)
    tx = (x - x_axis[ix]) / (x_axis[ix + 1] - x_axis[ix])
    ty = (y - y_axis[iy]) / (y_axis[iy + 1] - y_axis[iy])
    return (
        (1.0 - tx) * (1.0 - ty) * values[iy, ix]
        + tx * (1.0 - ty) * values[iy, ix + 1]
        + (1.0 - tx) * ty * values[iy + 1, ix]
        + tx * ty * values[iy + 1, ix + 1]
    )
