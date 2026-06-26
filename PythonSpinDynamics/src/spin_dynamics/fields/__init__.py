"""Dimension-agnostic spatial field maps shared by imaging and diffusion.

The spin-dynamics kernels only ever see a flat list of per-isochromat scalars
``(del_w, w_1, w_1r, T1, T2, m0, mth)``; spatial structure is incidental. This
package factors that structure into a reusable :class:`SpatialDomain` (1-D, 2-D,
or 3-D voxel grid) plus a :class:`SpatialFieldMaps` physics bundle, so imaging
(Eulerian voxel grids) and diffusion (Lagrangian walkers) share one
representation and one gradient-coupling rule.
"""

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.interpolate import dlinear_sample
from spin_dynamics.fields.maps import SpatialFieldMaps
from spin_dynamics.fields.positions import (
    gradient_offset,
    positions_nd,
    velocity_array,
)

__all__ = [
    "SpatialDomain",
    "SpatialFieldMaps",
    "dlinear_sample",
    "gradient_offset",
    "positions_nd",
    "velocity_array",
]
