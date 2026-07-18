"""Coordinate-frame helpers for defect-spin sensing.

Nano-MR calculations regularly mix a laboratory frame, a crystal frame, a
defect principal-axis frame, and a surface frame.  This module makes those
rotations explicit.  Rotation matrices map local vector components into the
laboratory frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def unit_vector(values, *, name: str = "vector") -> np.ndarray:
    """Return a validated three-component unit vector."""

    vector = np.asarray(values, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def as_rotation_matrix(values, *, name: str = "rotation") -> np.ndarray:
    """Return a validated proper 3-D rotation matrix."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be a proper rotation with determinant +1")
    return matrix.copy()


def rotation_from_z(axis_lab) -> np.ndarray:
    """Return the minimum proper rotation that maps local ``z`` to ``axis_lab``.

    The rotation fixes the defect axis but intentionally chooses a deterministic
    zero-roll convention.  Supply a complete :class:`CoordinateFrame` when the
    transverse crystal axes also carry physical meaning.
    """

    target = unit_vector(axis_lab, name="axis_lab")
    source = np.array([0.0, 0.0, 1.0])
    cosine = float(np.dot(source, target))
    if np.isclose(cosine, 1.0):
        return np.eye(3)
    if np.isclose(cosine, -1.0):
        return np.diag([1.0, -1.0, -1.0])

    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


@dataclass(frozen=True, eq=False)
class CoordinateFrame:
    """A named local coordinate frame embedded in the laboratory frame."""

    rotation_lab_from_local: np.ndarray
    name: str = "local"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rotation_lab_from_local",
            as_rotation_matrix(self.rotation_lab_from_local),
        )
        object.__setattr__(self, "name", str(self.name))

    @classmethod
    def from_z_axis(cls, axis_lab, *, name: str = "local") -> "CoordinateFrame":
        """Construct a zero-roll frame whose local ``z`` axis is ``axis_lab``."""

        return cls(rotation_from_z(axis_lab), name=name)

    @property
    def x_axis_lab(self) -> np.ndarray:
        """Return the local ``x`` axis expressed in laboratory components."""

        return self.rotation_lab_from_local[:, 0].copy()

    @property
    def y_axis_lab(self) -> np.ndarray:
        """Return the local ``y`` axis expressed in laboratory components."""

        return self.rotation_lab_from_local[:, 1].copy()

    @property
    def z_axis_lab(self) -> np.ndarray:
        """Return the local ``z`` axis expressed in laboratory components."""

        return self.rotation_lab_from_local[:, 2].copy()

    def vector_to_lab(self, vector_local) -> np.ndarray:
        """Transform local vector components into laboratory components."""

        return self.rotation_lab_from_local @ np.asarray(
            vector_local, dtype=np.float64
        ).reshape(3)

    def vector_to_local(self, vector_lab) -> np.ndarray:
        """Transform laboratory vector components into local components."""

        return self.rotation_lab_from_local.T @ np.asarray(
            vector_lab, dtype=np.float64
        ).reshape(3)

    def tensor_to_lab(self, tensor_local) -> np.ndarray:
        """Transform a rank-2 local tensor into the laboratory frame."""

        tensor = np.asarray(tensor_local, dtype=np.float64).reshape(3, 3)
        rotation = self.rotation_lab_from_local
        return rotation @ tensor @ rotation.T

    def tensor_to_local(self, tensor_lab) -> np.ndarray:
        """Transform a rank-2 laboratory tensor into the local frame."""

        tensor = np.asarray(tensor_lab, dtype=np.float64).reshape(3, 3)
        rotation = self.rotation_lab_from_local
        return rotation.T @ tensor @ rotation


__all__ = [
    "CoordinateFrame",
    "as_rotation_matrix",
    "rotation_from_z",
    "unit_vector",
]
