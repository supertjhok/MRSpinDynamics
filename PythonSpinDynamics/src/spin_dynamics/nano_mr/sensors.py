"""Defect-spin sensor data models for nanoscale magnetic resonance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nano_mr.frames import CoordinateFrame
from spin_dynamics.nqr.operators import spin_dimension, validate_spin


def as_symmetric_tensor(
    values: float | np.ndarray | list[float] | tuple[float, ...],
    *,
    name: str,
) -> np.ndarray:
    """Return a finite symmetric 3x3 tensor from scalar, principal, or full input."""

    array = np.asarray(values, dtype=np.float64)
    if array.shape == ():
        tensor = np.eye(3) * float(array)
    elif array.shape == (3,):
        tensor = np.diag(array)
    elif array.shape == (3, 3):
        tensor = array.copy()
    else:
        raise ValueError(f"{name} must be a scalar, length-3 vector, or 3x3 matrix")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(tensor, tensor.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return tensor


def zfs_tensor_from_d_e(d_hz: float, e_hz: float = 0.0) -> np.ndarray:
    """Return the traceless principal-axis ZFS tensor for conventional ``D, E``.

    The resulting frequency Hamiltonian is

    ``H/h = D * (Sz^2 - S(S+1)/3) + E * (Sx^2 - Sy^2)``.

    For spin 1 and ``E=0``, ``d_hz`` is the separation of ``m_s=0`` from the
    degenerate ``m_s=+/-1`` manifold.
    """

    d_hz = float(d_hz)
    e_hz = float(e_hz)
    if not np.isfinite(d_hz) or not np.isfinite(e_hz):
        raise ValueError("d_hz and e_hz must be finite")
    return np.diag(
        [
            -d_hz / 3.0 + e_hz,
            -d_hz / 3.0 - e_hz,
            2.0 * d_hz / 3.0,
        ]
    )


@dataclass(frozen=True, eq=False)
class DefectSpinSensor:
    """Ground-state spin model of an optically addressable point defect.

    ``g_tensor`` and ``zfs_tensor_hz`` are expressed in ``frame``.  The defect
    depth is a positive distance below a separately specified sample surface.
    Optical preparation and readout parameters are intentionally deferred to
    the optical readout model.
    """

    spin: float
    g_tensor: float | np.ndarray | list[float] | tuple[float, ...]
    zfs_tensor_hz: np.ndarray | list[float] | tuple[float, ...]
    frame: CoordinateFrame
    depth_nm: float
    material: str
    defect: str
    label: str = "sensor"

    def __post_init__(self) -> None:
        spin = validate_spin(self.spin)
        depth_nm = float(self.depth_nm)
        if depth_nm <= 0.0 or not np.isfinite(depth_nm):
            raise ValueError("depth_nm must be positive and finite")
        if not isinstance(self.frame, CoordinateFrame):
            raise TypeError("frame must be a CoordinateFrame")
        g_tensor = as_symmetric_tensor(self.g_tensor, name="g_tensor")
        if np.linalg.norm(g_tensor) <= 0.0:
            raise ValueError("g_tensor must be non-zero")
        zfs = as_symmetric_tensor(self.zfs_tensor_hz, name="zfs_tensor_hz")
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "g_tensor", g_tensor)
        object.__setattr__(self, "zfs_tensor_hz", zfs)
        object.__setattr__(self, "depth_nm", depth_nm)
        object.__setattr__(self, "material", str(self.material))
        object.__setattr__(self, "defect", str(self.defect))
        object.__setattr__(self, "label", str(self.label))

    @property
    def dimension(self) -> int:
        """Hilbert-space dimension of the ground-state electron spin."""

        return spin_dimension(self.spin)

    @property
    def axis_lab(self) -> np.ndarray:
        """Defect symmetry axis expressed in laboratory components."""

        return self.frame.z_axis_lab

    @property
    def zfs_principal_values_hz(self) -> np.ndarray:
        """Sorted eigenvalues of the ZFS tensor in hertz."""

        return np.linalg.eigvalsh(self.zfs_tensor_hz)


__all__ = [
    "DefectSpinSensor",
    "as_symmetric_tensor",
    "zfs_tensor_from_d_e",
]
