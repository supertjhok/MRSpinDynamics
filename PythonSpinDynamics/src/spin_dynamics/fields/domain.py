"""Dimension-agnostic spatial domain shared by imaging and diffusion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np


def _validate_axis(values: Iterable[float] | np.ndarray, index: int) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 1:
        raise ValueError(f"axis {index} must contain at least one value")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"axis {index} must contain finite values")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"axis {index} must be strictly increasing")
    return axis


def centered_unit_axis(size: int) -> np.ndarray:
    """Return ``(i - size // 2) / ((size - 1) / 2)``: the DFT-centred unit grid.

    Index ``size // 2`` sits exactly at zero, which is the centre that
    ``fftshift``/``ifftshift`` assume. For odd ``size`` this is identical to
    ``linspace(-1, 1, size)``; for even ``size`` it keeps the same step but
    drops the half-integer centre, which otherwise imprints a half-pixel shift
    (and a missing k=0 sample) on an imaging reconstruction.
    """

    n = int(size)
    if n < 1:
        raise ValueError("size must be positive")
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    return (np.arange(n, dtype=np.float64) - n // 2) / ((n - 1) / 2.0)


@dataclass(frozen=True)
class SpatialDomain:
    """A rectilinear voxel grid of one to three spatial axes.

    ``axes`` holds ``d`` strictly increasing 1-D coordinate arrays. The domain
    carries no physics; it only describes where samples live and how an applied
    gradient maps to a per-voxel frequency offset. ``ImagingFieldMaps`` (Eulerian
    voxel grids) and ``MotionFieldMaps2D`` (Lagrangian walkers) both build on it.
    """

    axes: tuple[np.ndarray, ...] = field()

    def __post_init__(self) -> None:
        axes = tuple(_validate_axis(axis, k) for k, axis in enumerate(self.axes))
        if not 1 <= len(axes) <= 3:
            raise ValueError("SpatialDomain supports 1, 2, or 3 axes")
        object.__setattr__(self, "axes", axes)

    @property
    def ndim(self) -> int:
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(axis.size) for axis in self.axes)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((float(axis[0]), float(axis[-1])) for axis in self.axes)

    def normalized_coordinate_grids(self) -> tuple[np.ndarray, ...]:
        """Return one DFT-centred unit coordinate grid per axis, full domain shape.

        Each axis is :func:`centered_unit_axis` (``linspace(-1, 1, n)`` for odd
        ``n``). For ``d == 2`` these equal the default ``del_wx``/``del_wz``
        gradient sensitivity maps produced by ``imaging._default_gradient_maps``.
        """

        shape = self.shape
        grids: list[np.ndarray] = []
        for k, n in enumerate(shape):
            base = centered_unit_axis(n)
            reshaped = base.reshape(tuple(n if j == k else 1 for j in range(self.ndim)))
            grids.append(np.ascontiguousarray(np.broadcast_to(reshaped, shape)))
        return tuple(grids)

    def meshgrid(self, indexing: str = "ij") -> tuple[np.ndarray, ...]:
        """Return the physical coordinate grids (generalizes ``np.meshgrid``)."""

        return tuple(np.meshgrid(*self.axes, indexing=indexing))

    @classmethod
    def normalized(cls, shape: tuple[int, ...]) -> "SpatialDomain":
        """Build a domain whose axes are :func:`centered_unit_axis` for each ``n``.

        This matches the implicit coordinate frame the imaging workflows use,
        where gradient-sensitivity maps default to normalized coordinates with
        index ``n // 2`` at zero (``linspace(-1, 1, n)`` for odd ``n``).
        """

        return cls(tuple(centered_unit_axis(int(n)) for n in shape))
