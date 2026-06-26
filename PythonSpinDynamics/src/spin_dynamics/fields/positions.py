"""Dimension-agnostic position and velocity helpers for moving isochromats."""

from __future__ import annotations

import numpy as np


def positions_nd(values: np.ndarray, ndim: int | None = None) -> np.ndarray:
    """Validate and return an ``(num_particles, d)`` float64 position array.

    When ``ndim`` is given the second axis must match it; otherwise any number
    of spatial dimensions is accepted. The error message keeps the
    ``(num_particles, <d>)`` wording the 2-D callers historically raised.
    """

    arr = np.asarray(values, dtype=np.float64)
    expected = "ndim" if ndim is None else str(ndim)
    if arr.ndim != 2 or (ndim is not None and arr.shape[1] != ndim):
        raise ValueError(f"positions must have shape (num_particles, {expected})")
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions must contain finite values")
    return arr


def velocity_array(
    velocity,
    positions: np.ndarray,
    time: float,
) -> np.ndarray:
    """Return a per-particle velocity array matching ``positions``.

    ``velocity`` may be ``None`` (zero), a length-``d`` vector broadcast to every
    particle, an array shaped like ``positions``, or a callable
    ``velocity(positions, time)`` returning either form.
    """

    if velocity is None:
        return np.zeros_like(positions)
    ndim = positions.shape[1]
    values = velocity(positions, time) if callable(velocity) else velocity
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != ndim:
            raise ValueError(f"velocity vector must have {ndim} components")
        arr = np.tile(arr, (positions.shape[0], 1))
    if arr.shape != positions.shape:
        raise ValueError(f"velocity must have shape ({ndim},) or positions.shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError("velocity must contain finite values")
    return arr


def gradient_offset(positions: np.ndarray, gradient) -> np.ndarray:
    """Return the Lagrangian gradient-induced offset ``positions @ gradient``."""

    pos = np.asarray(positions, dtype=np.float64)
    grad = np.asarray(gradient, dtype=np.float64).reshape(pos.shape[1])
    return pos @ grad
