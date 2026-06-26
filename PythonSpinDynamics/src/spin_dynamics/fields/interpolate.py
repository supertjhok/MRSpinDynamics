"""Dimension-agnostic multilinear sampling of gridded field maps."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def dlinear_sample(
    values: np.ndarray,
    axes: Sequence[np.ndarray],
    positions: np.ndarray,
) -> np.ndarray:
    """Multilinearly sample a ``d``-dimensional ``values`` map at ``positions``.

    ``axes`` is a sequence of ``d`` strictly increasing 1-D coordinate axes and
    ``positions`` has shape ``(num_particles, d)``. The result is the linear
    interpolation of ``values`` at each position, with out-of-range coordinates
    clamped to the axis bounds (the same edge handling as the legacy bilinear
    sampler). For ``d == 2`` this reduces, term for term, to the four-corner
    bilinear blend, so it is numerically identical to the previous
    ``_bilinear_sample`` implementation.
    """

    values = np.asarray(values)
    positions = np.asarray(positions, dtype=np.float64)
    ndim = len(axes)
    if positions.ndim != 2 or positions.shape[1] != ndim:
        raise ValueError(
            f"positions must have shape (num_particles, {ndim})"
        )
    if values.ndim != ndim:
        raise ValueError("values must have one axis per coordinate axis")

    n = positions.shape[0]
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    frac: list[np.ndarray] = []
    for k, axis in enumerate(axes):
        axis = np.asarray(axis, dtype=np.float64)
        coord = np.clip(positions[:, k], axis[0], axis[-1])
        hi = np.clip(np.searchsorted(axis, coord, side="right"), 1, axis.size - 1)
        lo = hi - 1
        a0 = axis[lo]
        a1 = axis[hi]
        t = np.divide(coord - a0, a1 - a0, out=np.zeros_like(coord), where=(a1 != a0))
        lower.append(lo)
        upper.append(hi)
        frac.append(t)

    result = np.zeros(n, dtype=np.result_type(values.dtype, np.float64))
    for corner in range(2**ndim):
        weight = np.ones(n, dtype=np.float64)
        index: list[np.ndarray] = []
        for k in range(ndim):
            if (corner >> k) & 1:
                weight = weight * frac[k]
                index.append(upper[k])
            else:
                weight = weight * (1.0 - frac[k])
                index.append(lower[k])
        result = result + weight * values[tuple(index)]
    return result
