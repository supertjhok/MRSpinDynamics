"""Per-voxel physics bundle over a :class:`SpatialDomain`.

A :class:`SpatialFieldMaps` holds the only quantities the spin-dynamics kernels
care about -- off-resonance, transmit/receive B1, relaxation, density, and an
optional diffusion coefficient -- as arrays sharing a spatial domain. It exposes
two complementary views of the same physics:

* :meth:`flatten` -- the Eulerian view used by the arbitrary-pulse imaging
  kernels: ravel every voxel to a 1-D isochromat list and broadcast an auxiliary
  off-resonance axis over it.
* :meth:`sample` -- the Lagrangian view used by moving walkers: multilinearly
  interpolate the maps at continuous particle positions.

Both reduce the gradient coupling to ``del_w_local = del_w_static + sum_d g_d *
r_d(sample)`` via :meth:`gradient_coupling`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.interpolate import dlinear_sample


@dataclass(frozen=True)
class SpatialFieldMaps:
    """Spatial sample and field maps shared by imaging and diffusion workflows."""

    domain: SpatialDomain
    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    diffusion_map: np.ndarray | None = None
    gradient_sensitivity: tuple[np.ndarray, ...] | None = None

    def _sensitivity(self) -> tuple[np.ndarray, ...]:
        if self.gradient_sensitivity is not None:
            return self.gradient_sensitivity
        return self.domain.normalized_coordinate_grids()

    def flatten(
        self,
        ny: int,
        maxoffs: float,
        density_normalization: Literal["legacy", "preserve"] = "legacy",
        *,
        axis_names: Sequence[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return flattened 1-D arrays consumed by the arbitrary-pulse kernels.

        The auxiliary off-resonance axis spans ``ny`` samples over
        ``[-maxoffs, maxoffs]`` and is broadcast over every voxel.
        ``density_normalization="legacy"`` assigns each auxiliary sample the full
        voxel density (MATLAB parity); ``"preserve"`` divides density by ``ny``.
        Per-axis gradient-sensitivity columns are returned under ``axis_names``
        (defaulting to ``del_w0 .. del_w{d-1}``); callers requiring the legacy
        ``del_wx``/``del_wz`` keys pass them explicitly.
        """

        if ny <= 0:
            raise ValueError("ny must be positive")
        if density_normalization not in {"legacy", "preserve"}:
            raise ValueError("density_normalization must be 'legacy' or 'preserve'")
        reps = int(ny)
        del_w0y = np.linspace(-float(maxoffs), float(maxoffs), reps)
        b0 = self.b0_map.reshape(-1)
        density_scale = 1.0 if density_normalization == "legacy" else 1.0 / reps
        density = density_scale * self.rho.reshape(-1)

        sensitivity = self._sensitivity()
        if axis_names is None:
            names: tuple[str, ...] = tuple(f"del_w{k}" for k in range(self.domain.ndim))
        else:
            names = tuple(axis_names)
        if len(names) != len(sensitivity):
            raise ValueError("axis_names must have one name per spatial axis")

        out = {
            "del_w": np.concatenate([offset + b0 for offset in del_w0y]),
            "w_1": np.tile(self.b1_tx_map.reshape(-1), reps),
            "w_1r": np.tile(self.b1_rx_map.reshape(-1), reps),
            "m0": np.tile(density, reps),
            "mth": np.tile(density, reps),
            "T1": np.tile(self.t1_map.reshape(-1), reps),
            "T2": np.tile(self.t2_map.reshape(-1), reps),
        }
        for name, grid in zip(names, sensitivity):
            out[name] = np.tile(np.asarray(grid).reshape(-1), reps)
        return out

    def sample(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """Multilinearly sample the maps at ``(num_particles, d)`` positions."""

        axes = self.domain.axes
        out = {
            "b0": dlinear_sample(self.b0_map, axes, positions),
            "b1_tx": dlinear_sample(self.b1_tx_map, axes, positions),
            "b1_rx": dlinear_sample(self.b1_rx_map, axes, positions),
        }
        if self.diffusion_map is not None:
            out["D"] = dlinear_sample(self.diffusion_map, axes, positions)
        return out

    def gradient_coupling(
        self,
        gradient: Sequence[float] | np.ndarray,
        *,
        grids: Sequence[np.ndarray] | None = None,
        positions: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the gradient-induced frequency offset ``sum_d g_d * r_d``.

        With ``positions`` this is the Lagrangian ``positions @ gradient``; with
        ``grids`` (or the stored gradient sensitivity) it is the Eulerian
        ``del_wg`` map summed over axes. Exactly one spatial representation is
        used: ``positions`` takes precedence when supplied.
        """

        ndim = self.domain.ndim
        g = np.asarray(gradient, dtype=np.float64).reshape(ndim)
        if positions is not None:
            return np.asarray(positions, dtype=np.float64) @ g
        sensitivity = self._sensitivity() if grids is None else tuple(grids)
        total = g[0] * sensitivity[0]
        for k in range(1, ndim):
            total = total + g[k] * sensitivity[k]
        return total
