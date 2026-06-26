"""Analytic magnetostatic field sources for MR magnet geometries.

This produces the ``(B0, B1)`` maps that the imaging and moving-isochromat
workflows consume, from first-principles magnet/coil models rather than synthetic
fields. It is pure NumPy and mesh-free:

* **B0** -- the static field of permanent-magnet blocks. A uniformly magnetized
  rectangular bar (2-D cross-section, infinite along z) has a closed-form field
  given by the magnetic "charge sheets" on its faces (``arctan``/``log`` terms),
  exact for the nearly linear rare-earth magnets (NdFeB ``mu_r ~ 1.05``) used in
  single-sided NMR. A soft-iron yoke (the flux return path) is added by the
  method of images across a ``mu -> infinity`` plane (mirror charges of opposite
  sign), which enforces ``B_tangential = 0`` at the iron surface.
* **B1** -- the RF field of a coil by Biot-Savart over straight current segments
  (closed-form per segment). The transverse (imaging-relevant) component is the
  part perpendicular to the local B0, via :func:`transverse_b1_magnitude`.

The canonical example is the NMR-MOUSE: two antiparallel bar magnets on an iron
yoke produce a stray field above the surface with a strong static gradient, the
basis of its depth resolution and built-in diffusion weighting.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

MU0 = 4.0e-7 * np.pi
GAMMA_PROTON = 2.675222e8  # rad/s/T


@dataclass(frozen=True)
class BarMagnet:
    """A 2-D uniformly magnetized rectangular bar (infinite along z).

    ``x0 < x1`` and ``y0 < y1`` are the cross-section extents (m); ``br_x``/
    ``br_y`` are the remanence components (T). The field outside the magnet is
    the field of magnetic surface charges ``Br . n`` on the four faces.
    """

    x0: float
    x1: float
    y0: float
    y1: float
    br_x: float = 0.0
    br_y: float = 0.0


# Each face is encoded as (kind, a, b, fixed, sigma):
#   kind 'h': horizontal strip, charge over x in [a, b] at y = fixed
#   kind 'v': vertical strip,   charge over y in [a, b] at x = fixed
# sigma is the surface charge density in tesla (Br . outward normal).
def _faces(bar: BarMagnet) -> list[tuple[str, float, float, float, float]]:
    return [
        ("h", bar.x0, bar.x1, bar.y1, bar.br_y),    # top,    n = +y
        ("h", bar.x0, bar.x1, bar.y0, -bar.br_y),   # bottom, n = -y
        ("v", bar.y0, bar.y1, bar.x1, bar.br_x),    # right,  n = +x
        ("v", bar.y0, bar.y1, bar.x0, -bar.br_x),   # left,   n = -x
    ]


def _mirror_face(face, yoke_y: float):
    kind, a, b, fixed, sigma = face
    if kind == "h":
        return ("h", a, b, 2.0 * yoke_y - fixed, -sigma)
    return ("v", 2.0 * yoke_y - a, 2.0 * yoke_y - b, fixed, -sigma)


def _strip_field(x, y, face):
    """Field (Bx, By) per unit charge of one charged strip, in tesla."""

    kind, a, b, fixed, sigma = face
    if sigma == 0.0:
        return 0.0, 0.0
    if kind == "h":
        dperp = y - fixed
        u_a, u_b = x - a, x - b
        bx = (sigma / (4.0 * np.pi)) * np.log(
            (u_a**2 + dperp**2) / (u_b**2 + dperp**2)
        )
        by = (sigma / (2.0 * np.pi)) * (
            np.arctan2(u_a, dperp) - np.arctan2(u_b, dperp)
        )
        return bx, by
    # vertical strip: swap roles of x and y
    dperp = x - fixed
    u_a, u_b = y - a, y - b
    by = (sigma / (4.0 * np.pi)) * np.log(
        (u_a**2 + dperp**2) / (u_b**2 + dperp**2)
    )
    bx = (sigma / (2.0 * np.pi)) * (
        np.arctan2(u_a, dperp) - np.arctan2(u_b, dperp)
    )
    return bx, by


def bar_array_b0(
    x: np.ndarray,
    y: np.ndarray,
    bars: Sequence[BarMagnet],
    *,
    yoke_y: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Bx, By)`` (T) of permanent-magnet bars at points ``(x, y)``.

    If ``yoke_y`` is given, a planar soft-iron (``mu -> inf``) return yoke at
    ``y = yoke_y`` is included by the method of images. Points should lie outside
    the magnets (the field on a face is singular).
    """

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    bx = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
    by = np.zeros_like(bx)
    faces: list = []
    for bar in bars:
        bf = _faces(bar)
        faces.extend(bf)
        if yoke_y is not None:
            faces.extend(_mirror_face(f, yoke_y) for f in bf)
    for face in faces:
        fx, fy = _strip_field(x, y, face)
        bx = bx + fx
        by = by + fy
    return bx, by


def biot_savart(
    points: np.ndarray,
    segments: Sequence[tuple[Sequence[float], Sequence[float]]],
    current: float,
) -> np.ndarray:
    """Biot-Savart B field (T) of straight current segments at ``points``.

    ``points`` has shape ``(..., 3)`` (m); ``segments`` is a sequence of
    ``(start, end)`` 3-D endpoints (m); ``current`` is in amperes. Each finite
    straight segment contributes the closed-form
    ``mu0 I / (4 pi d) (cos a1 - cos a2)`` in the azimuthal direction.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    total = np.zeros_like(pts)
    for start, end in segments:
        x1 = np.asarray(start, dtype=np.float64)
        x2 = np.asarray(end, dtype=np.float64)
        length = np.linalg.norm(x2 - x1)
        if length == 0.0:
            continue
        e = (x2 - x1) / length
        r1 = pts - x1
        r2 = pts - x2
        n1 = np.linalg.norm(r1, axis=-1)
        n2 = np.linalg.norm(r2, axis=-1)
        cross = np.cross(e, r1)  # |cross| = perpendicular distance, dir = phi-hat
        d2 = np.sum(cross**2, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = (
                np.sum(e * r1, axis=-1) / n1 - np.sum(e * r2, axis=-1) / n2
            )
            factor = np.where(d2 > 0.0, term / np.where(d2 > 0.0, d2, 1.0), 0.0)
        total = total + (MU0 * current / (4.0 * np.pi)) * factor[..., np.newaxis] * cross
    return total


def circular_loop(
    center: Sequence[float],
    radius: float,
    *,
    axis: str = "y",
    n_segments: int = 72,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return straight-segment endpoints approximating a circular current loop.

    The loop lies in the plane normal to ``axis`` (``"x"``, ``"y"``, or ``"z"``)
    and is centered at ``center`` (3-D, m).
    """

    c = np.asarray(center, dtype=np.float64)
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_segments) + 1)
    u = radius * np.cos(theta)
    v = radius * np.sin(theta)
    zeros = np.zeros_like(theta)
    if axis == "y":
        ring = np.column_stack([u, zeros, v])
    elif axis == "x":
        ring = np.column_stack([zeros, u, v])
    elif axis == "z":
        ring = np.column_stack([u, v, zeros])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    ring = ring + c
    return [(ring[i], ring[i + 1]) for i in range(len(theta) - 1)]


@dataclass(frozen=True)
class MagnetFieldMaps:
    """Sampled B0/B1 of a magnet+coil assembly on a 2-D ``(x, y)`` grid."""

    x_axis: np.ndarray
    y_axis: np.ndarray
    b0_vector: np.ndarray  # (nx, ny, 3) static field (T)
    b0_magnitude: np.ndarray  # (nx, ny) |B0| (T)
    b0_gradient: np.ndarray  # (nx, ny) |grad |B0|| (T/m), the static gradient
    larmor_hz: np.ndarray  # (nx, ny) proton Larmor frequency (Hz)
    b1_vector: np.ndarray | None  # (nx, ny, 3) RF field per amp (T/A)
    b1_transverse: np.ndarray | None  # (nx, ny) |B1 perpendicular to B0| (T/A)


def sample_magnet_field(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    bars: Sequence[BarMagnet],
    *,
    yoke_y: float | None = None,
    coil_segments: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    coil_current: float = 1.0,
    gamma: float = GAMMA_PROTON,
) -> MagnetFieldMaps:
    """Sample a permanent-magnet + RF-coil assembly onto a 2-D grid.

    Returns the B0 vector/magnitude, the static gradient ``|grad |B0||`` (the
    quantity that sets depth resolution and constant-gradient diffusion
    weighting), the proton Larmor map, and -- if a coil is supplied -- the B1
    vector and its transverse (perpendicular-to-B0) component, all ready to feed
    the imaging and motion workflows. The grid is ``(x, y)`` with ``indexing="ij"``;
    z = 0 (the magnet mid-plane).
    """

    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="ij")
    bx, by = bar_array_b0(xx, yy, bars, yoke_y=yoke_y)
    b0_vec = np.stack([bx, by, np.zeros_like(bx)], axis=-1)
    b0_mag = np.linalg.norm(b0_vec, axis=-1)

    gx, gy = np.gradient(b0_mag, x_axis, y_axis)
    b0_grad = np.sqrt(gx**2 + gy**2)
    larmor = gamma * b0_mag / (2.0 * np.pi)

    b1_vec = None
    b1_trans = None
    if coil_segments is not None:
        # Imported lazily: motion.py imports the fields package, so importing it
        # at module scope would create a circular import.
        from spin_dynamics.motion import transverse_b1_magnitude

        grid3d = np.stack([xx, yy, np.zeros_like(xx)], axis=-1)
        b1_vec = biot_savart(grid3d, coil_segments, coil_current)
        b1_trans = transverse_b1_magnitude(b0_vec, b1_vec)

    return MagnetFieldMaps(
        x_axis=x_axis,
        y_axis=y_axis,
        b0_vector=b0_vec,
        b0_magnitude=b0_mag,
        b0_gradient=b0_grad,
        larmor_hz=larmor,
        b1_vector=b1_vec,
        b1_transverse=b1_trans,
    )


def nmr_mouse_magnets(
    *,
    magnet_width: float = 20.0e-3,
    magnet_height: float = 20.0e-3,
    gap: float = 12.0e-3,
    remanence: float = 1.30,
    antiparallel: bool = True,
) -> tuple[list[BarMagnet], float]:
    """Return the bar magnets and yoke plane of a u-shaped NMR-MOUSE.

    Two NdFeB bars sit symmetrically about ``x = 0`` separated by ``gap``, their
    bottoms on an iron yoke at ``y = 0`` (so the sample volume is the stray field
    at ``y > magnet_height``). With ``antiparallel`` the two bars are polarized
    in opposite y directions, which puts a strong, smoothly decaying static
    gradient over the gap -- the MOUSE "sweet spot".
    """

    half_gap = 0.5 * gap
    left = BarMagnet(
        x0=-half_gap - magnet_width, x1=-half_gap,
        y0=0.0, y1=magnet_height, br_y=remanence,
    )
    right = BarMagnet(
        x0=half_gap, x1=half_gap + magnet_width,
        y0=0.0, y1=magnet_height, br_y=-remanence if antiparallel else remanence,
    )
    return [left, right], 0.0
