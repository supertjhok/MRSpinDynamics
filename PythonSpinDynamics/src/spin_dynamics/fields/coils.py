"""Coil-geometry generators for the field solvers.

Every generator returns the same straight-segment format as
``magnetostatics.circular_loop`` -- a list of ``(start, end)`` 3-D endpoint
pairs (metres) -- so the geometry feeds ``magnetostatics.biot_savart`` (B) and
``quasistatic.vector_potential`` (A) unchanged. Loops are approximated by
``n_segments`` chords; a "coil" is the flat concatenation of its loops.

Axis convention matches ``circular_loop``: a loop's plane is normal to ``axis``
(``"x"``, ``"y"`` or ``"z"``), and multi-loop coils (solenoid, Maxwell pair,
shield) stack their loops along that same axis.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from spin_dynamics.fields.magnetostatics import circular_loop

Segment = tuple[np.ndarray, np.ndarray]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis: str) -> int:
    if axis not in _AXIS_INDEX:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return _AXIS_INDEX[axis]


def _reversed_loop(loop: Sequence[Segment]) -> list[Segment]:
    """Reverse a loop's current direction by swapping each segment's endpoints."""

    return [(np.asarray(end), np.asarray(start)) for start, end in loop]


def solenoid(
    *,
    radius: float,
    length: float,
    turns: int,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
    n_segments: int = 48,
) -> list[Segment]:
    """A solenoid modelled as ``turns`` coaxial loops evenly spaced over ``length``.

    Loops are normal to ``axis`` and centred on the line through ``center``.
    (The small helical/axial current component is neglected -- the standard
    stacked-loop model, accurate for the interior and near field.)
    """

    if radius <= 0 or turns < 1:
        raise ValueError("radius must be > 0 and turns >= 1")
    c = np.asarray(center, dtype=np.float64)
    ai = _axis_index(axis)
    positions = (
        np.array([0.0]) if turns == 1
        else np.linspace(-length / 2.0, length / 2.0, int(turns))
    )
    segments: list[Segment] = []
    for offset in positions:
        loop_center = c.copy()
        loop_center[ai] += offset
        segments.extend(circular_loop(loop_center, radius, axis=axis, n_segments=n_segments))
    return segments


def planar_spiral(
    *,
    r_inner: float,
    r_outer: float,
    turns: int,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
    n_points: int = 400,
) -> list[Segment]:
    """A planar Archimedean spiral from ``r_inner`` to ``r_outer`` over ``turns``.

    Lies in the plane normal to ``axis`` (a flat "pancake" RF coil).
    """

    if r_inner < 0 or r_outer <= r_inner or turns < 1:
        raise ValueError("require 0 <= r_inner < r_outer and turns >= 1")
    c = np.asarray(center, dtype=np.float64)
    ai = _axis_index(axis)
    theta = np.linspace(0.0, 2.0 * np.pi * turns, int(n_points))
    r = r_inner + (r_outer - r_inner) * theta / (2.0 * np.pi * turns)
    u = r * np.cos(theta)
    v = r * np.sin(theta)
    # In-plane axes are the two axes that are not `axis`.
    plane = [i for i in range(3) if i != ai]
    pts = np.zeros((theta.size, 3))
    pts[:, plane[0]] = u
    pts[:, plane[1]] = v
    pts = pts + c
    return [(pts[i], pts[i + 1]) for i in range(theta.size - 1)]


def maxwell_pair(
    *,
    radius: float,
    separation: float | None = None,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
    n_segments: int = 72,
) -> list[Segment]:
    """A Maxwell pair (anti-Helmholtz): two coaxial loops with opposed currents.

    Produces a linear ``dB_axis/d(axis)`` gradient at the centre. The second loop
    is wound in reverse so a single positive ``current`` flows oppositely in the
    two loops. ``separation`` defaults to the ideal ``radius*sqrt(3)`` that makes
    the gradient maximally linear about the centre.
    """

    if radius <= 0:
        raise ValueError("radius must be > 0")
    sep = radius * np.sqrt(3.0) if separation is None else float(separation)
    c = np.asarray(center, dtype=np.float64)
    ai = _axis_index(axis)
    top = c.copy()
    top[ai] += sep / 2.0
    bot = c.copy()
    bot[ai] -= sep / 2.0
    loop_top = circular_loop(top, radius, axis=axis, n_segments=n_segments)
    loop_bot = circular_loop(bot, radius, axis=axis, n_segments=n_segments)
    return list(loop_top) + _reversed_loop(loop_bot)


def conducting_ring(
    *,
    radius: float,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
    n_segments: int = 144,
) -> list[Segment]:
    """A single circular loop -- e.g. a conducting ring for the eddy model."""

    if radius <= 0:
        raise ValueError("radius must be > 0")
    return circular_loop(np.asarray(center, dtype=np.float64), radius, axis=axis, n_segments=n_segments)


def cylindrical_shield(
    *,
    radius: float,
    length: float,
    n_rings: int,
    center: Sequence[float] = (0.0, 0.0, 0.0),
    axis: str = "z",
) -> tuple[np.ndarray, np.ndarray]:
    """Coaxial-ring model of a thin conducting cylinder (e.g. a cryostat bore).

    Returns ``(radii, centers)``: ``radii`` all equal to ``radius`` and
    ``centers`` the ``(n_rings, 3)`` ring-centre positions spread over ``length``
    along ``axis``. Consumed by :class:`fields.eddy_modes.EddyModes` (which needs
    each ring's radius and position for its resistance/inductance), not by
    Biot-Savart directly.
    """

    if radius <= 0 or n_rings < 1:
        raise ValueError("radius must be > 0 and n_rings >= 1")
    c = np.asarray(center, dtype=np.float64)
    ai = _axis_index(axis)
    offsets = (
        np.array([0.0]) if n_rings == 1
        else np.linspace(-length / 2.0, length / 2.0, int(n_rings))
    )
    centers = np.tile(c, (offsets.size, 1))
    centers[:, ai] = c[ai] + offsets
    radii = np.full(offsets.size, float(radius))
    return radii, centers


__all__ = [
    "solenoid",
    "planar_spiral",
    "maxwell_pair",
    "conducting_ring",
    "cylindrical_shield",
]
