"""Gradiometer pickup geometry and its reciprocal spatial sensitivity.

The frequency-domain detectors (:mod:`~spin_dynamics.detection.squid`,
:mod:`~spin_dynamics.detection.opm`) set *how quiet* a sensor is; a **pickup
geometry** sets *where* it is sensitive. SQUID (and OPM) systems commonly wind
the flux transformer as an axial **gradiometer** -- a stack of coaxial loops with
alternating turns -- so that a spatially uniform ambient field (and, at second
order, a uniform gradient) cancels while a nearby sample still couples strongly.

By the **reciprocity theorem**, a pickup's receive sensitivity to a magnetic
source at position ``r`` is proportional to the field the winding would produce
at ``r`` per unit current. So the gradiometer's sensitive region is exactly the
region it would "excite" as a transmitter, and the net sensitivity of a wound
pickup is the signed sum of its loop fields,

    ``S(r) = sum_i w_i * B_loop,i(r)``   (T per unit current),

which this module evaluates by reusing :func:`spin_dynamics.fields.magnetostatics.circular_loop`
and :func:`~spin_dynamics.fields.magnetostatics.biot_savart`.

Physics reproduced (Clarke, Hatridge & Moessle 2007, Fig. 7): the field of a
dipole and its first and second axial derivatives fall off as ``1/r^3``,
``1/r^4``, ``1/r^5``, so an order-``n`` axial gradiometer's distant-source
sensitivity falls as ``1/r^{3+n}`` and it rejects uniform (``n>=1``) and
first-gradient (``n>=2``) ambient noise, while a sample close to the bottom loop
couples like a plain magnetometer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import biot_savart, circular_loop

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_center(axis: str, coord: float) -> tuple[float, float, float]:
    center = [0.0, 0.0, 0.0]
    center[_AXIS_INDEX[axis]] = float(coord)
    return (center[0], center[1], center[2])


def _axis_unit(axis: str) -> np.ndarray:
    unit = np.zeros(3)
    unit[_AXIS_INDEX[axis]] = 1.0
    return unit


@dataclass(frozen=True)
class Gradiometer:
    """A coaxial pickup: loops at ``positions_m`` with turns/sign ``weights``.

    ``radii_m``, ``positions_m`` (axial coordinate along ``axis``) and ``weights``
    (signed turns) are equal-length sequences, one entry per loop. A single loop
    with weight ``+1`` is a magnetometer; ``(+1, -1)`` an (unbalanced-area-nulled)
    first-order gradiometer; ``(+1, -2, +1)`` a second-order gradiometer.
    """

    radii_m: tuple[float, ...]
    positions_m: tuple[float, ...]
    weights: tuple[float, ...]
    axis: str = "z"
    n_segments: int = 72
    name: str = "gradiometer"

    def __post_init__(self) -> None:
        radii = tuple(float(r) for r in self.radii_m)
        positions = tuple(float(z) for z in self.positions_m)
        weights = tuple(float(w) for w in self.weights)
        n = len(radii)
        if n == 0:
            raise ValueError("a gradiometer needs at least one loop")
        if not (len(positions) == n and len(weights) == n):
            raise ValueError("radii_m, positions_m and weights must have equal length")
        if any(r <= 0 or not np.isfinite(r) for r in radii):
            raise ValueError("radii_m must be positive and finite")
        if any(not np.isfinite(z) for z in positions):
            raise ValueError("positions_m must be finite")
        if self.axis not in _AXIS_INDEX:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if int(self.n_segments) < 3:
            raise ValueError("n_segments must be >= 3")
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "n_segments", int(self.n_segments))
        object.__setattr__(self, "name", str(self.name))

    @property
    def n_loops(self) -> int:
        return len(self.radii_m)

    @property
    def order(self) -> int:
        """Gradiometer order: the number of leading spatial moments that vanish.

        ``0`` for a magnetometer (responds to a uniform field), ``1`` when the
        uniform-field moment is nulled (responds to the first gradient), ``2``
        when the first-gradient moment is also nulled, and so on. Computed from
        the loop moments ``sum_i w_i z_i^k`` evaluated for increasing ``k``.
        """

        z = np.asarray(self.positions_m)
        w = np.asarray(self.weights)
        wscale = np.sum(np.abs(w)) or 1.0
        zscale = np.max(np.abs(z)) if np.any(z) else 1.0
        for k in range(self.n_loops):
            moment = float(np.sum(w * z**k))
            tol = 1e-9 * wscale * (zscale**k if zscale > 0 else 1.0)
            if abs(moment) > tol:
                return k
        return self.n_loops

    def _loop_field(self, points: np.ndarray, radius: float, coord: float) -> np.ndarray:
        segments = circular_loop(
            _axis_center(self.axis, coord), radius, axis=self.axis, n_segments=self.n_segments
        )
        return biot_savart(points, segments, 1.0)

    def sensitivity(self, points) -> np.ndarray:
        """Net reciprocal sensitivity field ``S(r)`` (T per unit current).

        ``points`` is ``(..., 3)`` in metres; returns the same shape. This is the
        signed sum of the loop fields -- equivalently, the B1 field the pickup
        would transmit per unit current.
        """

        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3")
        total = np.zeros(pts.shape, dtype=np.float64)
        for radius, coord, weight in zip(self.radii_m, self.positions_m, self.weights):
            total = total + weight * self._loop_field(pts, radius, coord)
        return total

    def axial_sensitivity(self, points) -> np.ndarray:
        """Component of :meth:`sensitivity` along the pickup axis (``T/A``)."""

        return self.sensitivity(points) @ _axis_unit(self.axis)

    @property
    def reference_point(self) -> tuple[float, float, float]:
        """On-axis centre of the first (bottom/sample) loop -- the coupling reference."""

        return _axis_center(self.axis, self.positions_m[0])

    def reference_sensitivity(self, *, reference=None) -> float:
        """Axial sensitivity at ``reference`` (default :attr:`reference_point`)."""

        ref = self.reference_point if reference is None else reference
        value = float(self.axial_sensitivity(np.asarray([ref], dtype=np.float64))[0])
        if value == 0.0 or not np.isfinite(value):
            raise ValueError("reference sensitivity must be finite and non-zero")
        return value

    def normalized_sensitivity(self, points, *, reference=None) -> np.ndarray:
        """Dimensionless axial sensitivity, ``1`` at the reference point.

        Normalizing to each pickup's own bottom-loop coupling makes pickup
        geometries directly comparable: a sample at the reference couples with
        weight ``1`` regardless of winding, so the differences that remain are
        exactly the distant-source rejection (weight ``<< 1`` far away for a
        gradiometer). Used by :mod:`spin_dynamics.detection.spatial`.
        """

        return self.axial_sensitivity(points) / self.reference_sensitivity(reference=reference)

    def uniform_field_response(self) -> float:
        """Reciprocal response to a spatially uniform axial unit field.

        Equal to the net signed pickup area ``sum_i w_i * pi * r_i^2`` (the flux a
        uniform field threads); zero for a balanced gradiometer, so it quantifies
        common-mode (distant/ambient) rejection.
        """

        return float(sum(w * np.pi * r * r for r, w in zip(self.radii_m, self.weights)))

    def uniform_coupling(self) -> float:
        """Dimensionless response to a uniform axial field vs the bottom loop.

        ``uniform_field_response / (pi * r_bottom^2)`` -- ``1`` for a single-loop
        magnetometer and ``~0`` for a balanced gradiometer. This is the common-mode
        (uniform ambient field) coupling that a gradiometer rejects, on the same
        normalized scale as :meth:`normalized_sensitivity`.
        """

        bottom_area = np.pi * self.radii_m[0] ** 2
        return self.uniform_field_response() / bottom_area

    @classmethod
    def magnetometer(
        cls, *, radius_m: float, position_m: float = 0.0, axis: str = "z", n_segments: int = 72
    ) -> "Gradiometer":
        """Single pickup loop (order 0)."""

        return cls(
            radii_m=(radius_m,),
            positions_m=(position_m,),
            weights=(1.0,),
            axis=axis,
            n_segments=n_segments,
            name="magnetometer",
        )

    @classmethod
    def first_order_axial(
        cls,
        *,
        radius_m: float,
        baseline_m: float,
        bottom_m: float = 0.0,
        axis: str = "z",
        n_segments: int = 72,
    ) -> "Gradiometer":
        """Two coaxial loops ``(+1, -1)`` a baseline apart (order 1).

        The ``+1`` loop sits at ``bottom_m`` (nearest the sample); the ``-1``
        compensation loop a ``baseline_m`` away. Nulls a uniform field.
        """

        return cls(
            radii_m=(radius_m, radius_m),
            positions_m=(bottom_m, bottom_m + baseline_m),
            weights=(1.0, -1.0),
            axis=axis,
            n_segments=n_segments,
            name="first-order gradiometer",
        )

    @classmethod
    def second_order_axial(
        cls,
        *,
        radius_m: float,
        baseline_m: float,
        bottom_m: float = 0.0,
        axis: str = "z",
        n_segments: int = 72,
    ) -> "Gradiometer":
        """Three coaxial loops ``(+1, -2, +1)`` (order 2), the Clarke pickup.

        Nulls both a uniform field and a uniform first gradient; the ``+1`` bottom
        loop at ``bottom_m`` couples to a nearby sample.
        """

        return cls(
            radii_m=(radius_m, radius_m, radius_m),
            positions_m=(bottom_m, bottom_m + baseline_m, bottom_m + 2.0 * baseline_m),
            weights=(1.0, -2.0, 1.0),
            axis=axis,
            n_segments=n_segments,
            name="second-order gradiometer",
        )
