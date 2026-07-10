"""Partial-element (PEEC) RF property solver for arbitrary wire coils.

Where :mod:`spin_dynamics.fields.coil_properties` extracts inductance, resistance,
capacitance, Q and self-resonance for the *single-layer solenoid* (the QOIL
semi-analytical model), this module does it for an **arbitrary wire path** -- multi-layer
and saddle RF coils, surface loops, gradient sets -- from first principles, with the skin
and proximity effects resolved rather than tabulated.

The method is PEEC (partial-element equivalent circuit), the mesh-light approach behind
FastHenry/FastCap. A conductor is the wire centreline (a polyline ``path``) plus a round
cross-section; to resolve the current distribution the cross-section is tiled into ``K``
parallel sub-filaments, each following the whole path. Because every sub-filament is a
simple series chain from one terminal to the other, the branch system reduces **exactly**
to a ``K x K`` chain-impedance system: sub-filament ``k`` is the path swept to cross-cell
``k``, and the chain inductance ``L[k, k']`` sums the exact closed-form mutual inductance
between every pair of their straight segments (:func:`_mutualfil_matrix`, a vectorized port
of FastHenry's Grover-formula ``mutualfil``). The closed form is essential: coil path
segments share vertices, and a numerical quadrature there over-couples them and inflates the
inductance without bound as the path is refined. Solving ``Z_chain = R + j w L`` for the
terminal impedance ``Z(w) = 1 / (1^T Z_chain^{-1} 1)`` lets the current redistribute across
the cross-section, which is exactly the skin effect (own cross-section) and proximity effect
(neighbouring turns/legs). A dual thin-wire electrostatic solve on the same geometry gives
the self-capacitance and hence the self-resonant frequency.

Inductance matches FastHenry to <1% for both straight bars and helical solenoids (see
``docs/coil_peec.md``), the self-capacitance matches FasterCap to ~1%, and the self-resonant
frequency matches the QOIL model to a few percent.

**Resolution ceiling (state it honestly).** The cross-section must resolve the skin depth
``delta`` for the *resistance*: at ``a/delta`` up to ~5 a few hundred cells give sub-percent
AC-resistance accuracy (validated against the exact Kelvin-function ratio and FastHenry);
deeper skin (high-frequency RF, ``a/delta`` >~ 15) needs many cells or the
:func:`extract_impedance_surface` surface-impedance backend. Inductance and capacitance are
geometry-dominated and accurate at coarse resolution.

**Two formulations: ``chain`` (fast) and ``full`` (proximity-resolving).** The reduced
``K x K`` chain system above gives each sub-filament a *path-constant* current, so the
cross-section distribution cannot vary along the wire to crowd toward each neighbouring
turn -- it captures the skin effect but only a fraction of the proximity effect. FastHenry
does not have this limitation because it never reduces: every segment keeps its own
``nwinc x nhinc`` filaments as independent branches, and its mesh analysis adds, for every
segment, ``K - 1`` "mini-meshes" between adjacent filaments (``fillM.c``), so the
cross-section current split is a free unknown *per segment* and proximity emerges from the
partial-inductance coupling alone -- no empirical factor anywhere in its source. Passing
``formulation="full"`` to :func:`extract_impedance` / :func:`extract_impedance_surface` /
:func:`coil_properties_peec` reproduces exactly that: per-(segment, sub-filament) branch
currents, constrained only by charge conservation at the path nodes (the mesh-analysis
equivalent), at cost ``O((M K)^3)`` per frequency instead of ``O(K^3)``. Use ``full``
whenever turns/legs are within a few wire diameters of each other (any tight coil);
``chain`` remains the fast default for inductance, capacitance and isolated-wire skin
resistance, which it gets right. The Medhurst proximity factor is no longer needed --
it survives only as an independent *validation* reference for single-layer solenoids.

The exact filament mutual is ported from FastHenry (``mutualfil``, Grover's method;
originally M.I.T., maintained by FastFieldSolvers) -- the algorithm, re-implemented in NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from spin_dynamics.fields.coil_properties import ANNEALED_COPPER, ConductorMaterial
from spin_dynamics.fields.magnetostatics import MU0

EPS0 = 8.8541878128e-12  # vacuum permittivity (F/m)
MUOVER4PI = 1.0e-7  # mu0 / 4pi
C0 = 299792458.0  # speed of light (m/s)
ETA0 = 376.730313668  # impedance of free space (ohm)
_FIL_EPS = 1e-13  # geometric tolerance for the filament-mutual branch selection

Segment = tuple[np.ndarray, np.ndarray]

__all__ = [
    "Conductor",
    "conductor_from_segments",
    "helical_solenoid",
    "self_partial_inductance",
    "filament_self_inductance",
    "extract_impedance",
    "extract_impedance_surface",
    "current_distribution",
    "self_capacitance",
    "capacitance_to_ground",
    "GroundedBox",
    "GroundPlane",
    "self_resonant_frequency",
    "radiation_resistance",
    "coil_properties_peec",
    "PEECImpedance",
    "PEECCoilProperties",
]


def self_partial_inductance(length: float, gmd_radius: float) -> float:
    """Self partial inductance (H) of a straight round filament.

    ``(mu0 L / 2 pi) [ln((L + sqrt(L^2 + r^2)) / r) - sqrt(1 + (r/L)^2) + r/L]`` with ``r``
    the filament's geometric mean radius (external / high-frequency form; the internal
    inductance is recovered by the cross-section subdivision). Reduces to the familiar
    ``(mu0 L / 2 pi)(ln(2L/r) - 1)`` for ``L >> r``.
    """

    length = float(length)
    r = float(gmd_radius)
    if length <= 0.0 or r <= 0.0:
        raise ValueError("length and gmd_radius must be positive")
    return float(
        (MU0 * length / (2.0 * np.pi))
        * (np.log((length + np.sqrt(length**2 + r**2)) / r) - np.sqrt(1.0 + (r / length) ** 2) + r / length)
    )


def _rotation_minimizing_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-transported cross-section frames (e1, e2) at each path vertex.

    Returns two ``(M+1, 3)`` arrays of unit vectors perpendicular to the local tangent,
    minimizing twist along the path (the double-reflection method) so sub-filaments stay
    continuous across segment joints.
    """

    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    # Per-vertex tangents (average of adjacent segment directions).
    tang = np.zeros_like(pts)
    tang[:-1] += pts[1:] - pts[:-1]
    tang[1:] += pts[1:] - pts[:-1]
    norms = np.linalg.norm(tang, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    tang = tang / norms

    e1 = np.zeros_like(pts)
    e2 = np.zeros_like(pts)
    # Seed the first frame with any vector not parallel to the tangent.
    t0 = tang[0]
    seed = np.array([1.0, 0.0, 0.0]) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r = seed - np.dot(seed, t0) * t0
    r /= np.linalg.norm(r)
    e1[0] = r
    e2[0] = np.cross(t0, r)
    for i in range(1, n):
        # Parallel-transport e1 from vertex i-1 to i (rotation-minimizing frame).
        v = tang[i] - tang[i - 1]
        prev = e1[i - 1]
        denom = 1.0 + np.dot(tang[i - 1], tang[i])
        if denom > 1e-12:
            prev = prev - (np.dot(v, prev) / denom) * (tang[i - 1] + tang[i])
        prev = prev - np.dot(prev, tang[i]) * tang[i]
        nprev = np.linalg.norm(prev)
        e1[i] = prev / nprev if nprev > 0 else e1[i - 1]
        e2[i] = np.cross(tang[i], e1[i])
    return e1, e2


def _round_cross_section(
    radius: float, n_radial: int, n_angular: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian tiling of a round wire cross-section, masked to the disc.

    Returns ``(offsets, areas, gmd)``: ``offsets`` (K, 2) cell-centroid positions in the
    local frame, ``areas`` (K,) cell areas, and ``gmd`` (K,) the self geometric-mean radius
    of each cell (area-equivalent circle, ``sqrt(A/pi) * e^{-1/4}``). A square grid over the
    bounding box keeps every cell (near-)square -- unlike an equal-area *polar* tiling, whose
    elongated wedges break the square-cell GMD approximation and make the AC resistance
    diverge rather than converge as the count rises. Cell areas are rescaled so they sum to
    ``pi radius^2`` (an exact DC resistance). The requested ``n_radial * n_angular`` sets the
    target cell count; the grid resolution is derived from it.
    """

    n_radial = int(n_radial)
    n_angular = int(n_angular)
    if n_radial < 1 or n_angular < 1:
        raise ValueError("n_radial and n_angular must be >= 1")
    target = n_radial * n_angular
    if target == 1:
        return np.zeros((1, 2)), np.array([np.pi * radius**2]), np.array([radius * np.exp(-0.25)])
    # A grid of n x n over the bounding square keeps ~ (pi/4) n^2 cells inside the disc.
    n_grid = max(2, int(round(np.sqrt(target * 4.0 / np.pi))))
    edges = np.linspace(-radius, radius, n_grid + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    cell = (2.0 * radius / n_grid) ** 2
    gx, gy = np.meshgrid(centres, centres, indexing="ij")
    inside = gx**2 + gy**2 <= radius**2
    if not np.any(inside):  # radius smaller than one cell -> single centred filament
        return np.zeros((1, 2)), np.array([np.pi * radius**2]), np.array([radius * np.exp(-0.25)])
    offsets = np.column_stack([gx[inside], gy[inside]])
    areas = np.full(offsets.shape[0], cell)
    areas *= (np.pi * radius**2) / areas.sum()  # rescale to the exact disc area
    gmd = np.sqrt(areas / np.pi) * np.exp(-0.25)
    return offsets, areas, gmd


def _graded_edges(extent: float, n: int, grading: float) -> np.ndarray:
    """``n+1`` cell edges on ``[-extent/2, extent/2]``, concentrated toward both ends.

    ``grading == 1`` is uniform; ``grading > 1`` clusters cells near the two surfaces
    (a geometric-like power mapping), so the skin layer is resolved with fewer cells.
    """

    t = np.linspace(0.0, 1.0, n + 1)
    if grading == 1.0:
        s = t
    else:
        g = float(grading)
        s = np.where(t < 0.5, 0.5 * (2.0 * t) ** g, 1.0 - 0.5 * (2.0 * (1.0 - t)) ** g)
    return (s - 0.5) * extent


def _rect_cross_section(
    width: float, height: float, n_width: int, n_height: int, grading: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian tiling of a rectangular (or square) wire cross-section.

    Returns ``(offsets, areas, gmd)`` as :func:`_round_cross_section`. ``grading > 1``
    concentrates cells toward the four edges (the skin layer) so the AC resistance converges
    with fewer cells; ``grading == 1`` is the uniform tiling used for FastHenry comparison.
    """

    n_width = int(n_width)
    n_height = int(n_height)
    if n_width < 1 or n_height < 1:
        raise ValueError("n_width and n_height must be >= 1")
    ex = _graded_edges(width, n_width, grading)
    ey = _graded_edges(height, n_height, grading)
    cx = 0.5 * (ex[:-1] + ex[1:])
    cy = 0.5 * (ey[:-1] + ey[1:])
    dwx = np.diff(ex)
    dhy = np.diff(ey)
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    ax, ay = np.meshgrid(dwx, dhy, indexing="ij")
    offsets = np.column_stack([gx.ravel(), gy.ravel()])
    areas = (ax * ay).ravel()
    gmd = np.sqrt(areas / np.pi) * np.exp(-0.25)
    return offsets, areas, gmd


def _sweep_offsets(path_points: np.ndarray, offsets: np.ndarray) -> list[list[Segment]]:
    """Sweep ``(K, 2)`` cross-section offsets along the path with rotation-minimizing frames.

    Returns ``filaments[k]``: the list of ``(start, end)`` segments of sub-filament ``k`` (the
    path displaced to cross-section offset ``k``).
    """

    e1, e2 = _rotation_minimizing_frames(path_points)
    base = path_points[np.newaxis, :, :]
    disp = (
        offsets[:, 0, np.newaxis, np.newaxis] * e1[np.newaxis, :, :]
        + offsets[:, 1, np.newaxis, np.newaxis] * e2[np.newaxis, :, :]
    )
    verts = base + disp
    return [
        [(verts[k, m], verts[k, m + 1]) for m in range(verts.shape[1] - 1)]
        for k in range(verts.shape[0])
    ]


@dataclass(frozen=True)
class Conductor:
    """A single current-carrying wire: a polyline centreline plus a cross-section.

    ``path_points`` is an ``(M+1, 3)`` array of vertices (m); ``material`` a
    :class:`ConductorMaterial`; ``temperature`` (K, optional) the operating temperature.
    ``cross_section`` is ``"round"`` (default; ``wire_radius`` m, tiled into
    ``n_radial * n_angular`` sub-filaments) or ``"rect"`` (``width`` x ``height`` m, tiled
    into ``n_width * n_height`` sub-filaments -- for strip/tape conductors and
    apples-to-apples FastHenry comparison). The two ends of the path are the terminals.
    """

    path_points: np.ndarray
    wire_radius: float = 0.0
    material: ConductorMaterial = ANNEALED_COPPER
    n_radial: int = 6
    n_angular: int = 8
    temperature: float | None = None
    cross_section: str = "round"
    width: float | None = None
    height: float | None = None
    n_width: int = 6
    n_height: int = 6
    grading: float = 1.0
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.path_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError("path_points must be an (M+1, 3) array with M >= 1 segments")
        object.__setattr__(self, "path_points", pts)
        if self.cross_section == "round":
            if self.wire_radius <= 0.0:
                raise ValueError("round cross-section requires wire_radius > 0")
        elif self.cross_section == "rect":
            if not self.width or not self.height or self.width <= 0.0 or self.height <= 0.0:
                raise ValueError("rect cross-section requires width > 0 and height > 0")
            # Area-equivalent radius for the electrostatic self-term.
            object.__setattr__(self, "wire_radius", float(np.sqrt(self.width * self.height / np.pi)))
        else:
            raise ValueError("cross_section must be 'round' or 'rect'")

    @property
    def n_cells(self) -> int:
        """Actual number of cross-section sub-filaments (round tiling masks to the disc)."""

        if self.cross_section == "rect":
            return int(self.n_width) * int(self.n_height)
        return int(self.subfilaments()[1].size)

    @property
    def segment_lengths(self) -> np.ndarray:
        d = np.diff(self.path_points, axis=0)
        return np.linalg.norm(d, axis=1)

    @property
    def total_length(self) -> float:
        return float(np.sum(self.segment_lengths))

    def subfilaments(self) -> tuple[list[list[Segment]], np.ndarray, np.ndarray]:
        """Return ``(filaments, areas, gmd)`` for the ``K`` cross-section sub-filaments.

        ``filaments[k]`` is the list of ``(start, end)`` segments of sub-filament ``k``
        (the path offset to cross-cell ``k`` using parallel-transported frames);
        ``areas`` (K,) and ``gmd`` (K,) are the cell areas and self radii.
        """

        if "subfil" in self._cache:
            return self._cache["subfil"]
        if self.cross_section == "rect":
            offsets, areas, gmd = _rect_cross_section(
                self.width, self.height, self.n_width, self.n_height, self.grading
            )
        else:
            offsets, areas, gmd = _round_cross_section(self.wire_radius, self.n_radial, self.n_angular)
        filaments = _sweep_offsets(self.path_points, offsets)
        result = (filaments, areas, gmd)
        self._cache["subfil"] = result
        return result


def conductor_from_segments(
    segments: Sequence[Segment],
    *,
    wire_radius: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    n_radial: int = 6,
    n_angular: int = 8,
    temperature: float | None = None,
) -> Conductor:
    """Build a :class:`Conductor` from a connected ``(start, end)`` segment list.

    The segments must form a single connected polyline (each segment's start equals the
    previous end, up to a small tolerance) -- e.g. the output of
    :func:`spin_dynamics.fields.coils.solenoid`.
    """

    segs = [(np.asarray(s, dtype=np.float64), np.asarray(e, dtype=np.float64)) for s, e in segments]
    if not segs:
        raise ValueError("segments must be non-empty")
    pts = [segs[0][0]]
    for s, e in segs:
        if np.linalg.norm(s - pts[-1]) > 1e-9:
            raise ValueError("segments must form a single connected polyline")
        pts.append(e)
    return Conductor(
        path_points=np.array(pts),
        wire_radius=float(wire_radius),
        material=material,
        n_radial=int(n_radial),
        n_angular=int(n_angular),
        temperature=temperature,
    )


def filament_self_inductance(filament: list[Segment], gmd_radius: float) -> float:
    """Self partial inductance (H) of one open filamentary wire following the path.

    Sum of each segment's :func:`self_partial_inductance` plus the exact
    (:func:`_mutualfil_matrix`) mutual inductance between distinct segments of the same
    filament (coincident pairs excluded).
    """

    starts = np.array([np.asarray(s) for s, _ in filament])
    ends = np.array([np.asarray(e) for _, e in filament])
    lengths = np.linalg.norm(ends - starts, axis=1)
    self_terms = sum(self_partial_inductance(ln, gmd_radius) for ln in lengths if ln > 0.0)
    cross = _mutualfil_matrix(starts, ends, starts, ends)
    np.fill_diagonal(cross, 0.0)
    return float(self_terms + float(cross.sum()))


def _atanh_clip(x: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(x, -1.0 + 1e-15, 1.0 - 1e-15))


def _mutualfil_matrix(
    s1: np.ndarray, e1: np.ndarray, s2: np.ndarray, e2: np.ndarray
) -> np.ndarray:
    """Exact mutual inductance (H) between every pair of straight filaments.

    ``s1``/``e1`` are the (M1, 3) segment endpoints of the first filament set, ``s2``/``e2``
    the (M2, 3) endpoints of the second; returns the (M1, M2) matrix of exact filament-pair
    mutual inductances. This is a vectorized port of FastHenry's ``mutualfil`` (Grover's
    closed form) with its four branches -- endpoint-touching, perpendicular (zero), parallel
    (``mut_rect``, or a collinear log limit), and the general skew formula. The closed form
    is essential for coils: adjacent path segments share a vertex, and a thin-filament
    numerical quadrature there over-couples them and inflates the inductance without bound as
    the path is refined.
    """

    a1 = s1[:, None, :]
    b1 = e1[:, None, :]
    a2 = s2[None, :, :]
    b2 = e2[None, :, :]
    r1 = np.linalg.norm(b1 - b2, axis=-1)
    r2 = np.linalg.norm(b1 - a2, axis=-1)
    r3 = np.linalg.norm(a1 - a2, axis=-1)
    r4 = np.linalg.norm(a1 - b2, axis=-1)
    r1s, r2s, r3s, r4s = r1**2, r2**2, r3**2, r4**2
    v1 = b1 - a1
    v2 = b2 - a2
    length1 = np.linalg.norm(v1, axis=-1)  # (M1, 1)
    length2 = np.linalg.norm(v2, axis=-1)  # (1, M2)
    lm = length1 * length2
    dotp = np.sum(v1 * v2, axis=-1)
    alpha = r4s - r3s + r2s - r1s

    with np.errstate(all="ignore"):
        realcos = dotp / lm
        cose = np.clip(realcos, -1.0, 1.0)

        # Touching branch (a shared endpoint): Grover's meeting-at-a-point formula.
        r_t = np.where(r1 < _FIL_EPS, r3, np.where(r2 < _FIL_EPS, r4, np.where(r3 < _FIL_EPS, r1, r2)))
        m_touch = MUOVER4PI * 2 * (dotp / lm) * (
            length1 * _atanh_clip(length2 / (length1 + r_t))
            + length2 * _atanh_clip(length1 / (length2 + r_t))
        )

        # Parallel branch: project filament 2 onto filament 1's axis.
        u_hat = v1 / length1[..., None]
        rr = a2 - a1
        proj = np.sum(u_hat * rr, axis=-1, keepdims=True)
        dvec = rr - proj * u_hat
        d = np.linalg.norm(dvec, axis=-1)
        x2_0 = np.sum((a2 - (a1 + dvec)) * u_hat, axis=-1)
        x2_1 = np.sum((b2 - (a1 + dvec)) * u_hat, axis=-1)
        x1_0 = np.zeros_like(d)
        x1_1 = length1 + 0.0 * length2

        def _mut_rect(length, dd):
            dd = np.where(dd < 1e-300, 1e-300, dd)
            return np.sqrt(length * length + dd * dd) - length * np.arcsinh(length / dd)

        m_par = MUOVER4PI * (
            _mut_rect(x2_1 - x1_1, d) - _mut_rect(x2_1 - x1_0, d)
            - _mut_rect(x2_0 - x1_1, d) + _mut_rect(x2_0 - x1_0, d)
        )

        def _flog(x):
            ax = np.abs(x)
            return np.where(ax > 0, ax * np.log(np.where(ax > 0, ax, 1.0)), 0.0)

        m_col = MUOVER4PI * (
            _flog(x2_1 - x1_0) - _flog(x2_1 - x1_1) - _flog(x2_0 - x1_0) + _flog(x2_0 - x1_1)
        )

        # General skew branch.
        l2 = length1**2
        m2 = length2**2
        denom = 4 * l2 * m2 - alpha**2
        denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
        ug = length1 * (2 * m2 * (r2s - r3s - l2) + alpha * (r4s - r3s - m2)) / denom
        vg = length2 * (2 * l2 * (r4s - r3s - m2) + alpha * (r2s - r3s - l2)) / denom
        d2 = r3s - ug**2 - vg**2 + 2 * ug * vg * cose
        dg = np.sqrt(np.clip(d2, 0.0, None))
        sine = np.sqrt(np.clip(1.0 - cose**2, 0.0, None))
        t1 = dg**2 * cose
        t2 = dg * sine
        t3 = sine**2
        s2safe = np.where(np.abs(t2) < 1e-300, 1e-300, t2)
        omega = (
            np.arctan2(t1 + (ug + length1) * (vg + length2) * t3, s2safe * r1)
            - np.arctan2(t1 + (ug + length1) * vg * t3, s2safe * r2)
            + np.arctan2(t1 + ug * vg * t3, s2safe * r3)
            - np.arctan2(t1 + ug * (vg + length2) * t3, s2safe * r4)
        )
        omega = np.where(dg < _FIL_EPS, 0.0, omega)
        t4 = (
            (ug + length1) * _atanh_clip(length2 / (r1 + r2))
            + (vg + length2) * _atanh_clip(length1 / (r1 + r4))
            - ug * _atanh_clip(length2 / (r3 + r4))
            - vg * _atanh_clip(length1 / (r2 + r3))
        )
        t5 = np.where(sine < 1e-150, 0.0, omega * dg / np.where(sine < 1e-150, 1.0, sine))
        m_gen = MUOVER4PI * cose * (2 * t4 - t5)

        # Branch selection (order matters: touching overrides all).
        parallel = np.abs(np.abs(cose) - 1.0) < _FIL_EPS
        collinear = parallel & (d < _FIL_EPS)
        touching = (r1 < _FIL_EPS) | (r2 < _FIL_EPS) | (r3 < _FIL_EPS) | (r4 < _FIL_EPS)
        out = np.where(parallel, np.where(collinear, m_col, m_par), m_gen)
        out = np.where(np.abs(realcos) < _FIL_EPS, 0.0, out)
        out = np.where(touching, m_touch, out)
    return out


def _filament_endpoints(
    filaments: list[list[Segment]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Per-filament ``(starts, ends, lengths)`` arrays for the mutual-inductance kernels."""

    starts = [np.array([s for s, _ in f]) for f in filaments]
    ends = [np.array([e for _, e in f]) for f in filaments]
    lens = [np.linalg.norm(e - s, axis=1) for s, e in zip(starts, ends)]
    return starts, ends, lens


def _chain_lmat_from_filaments(
    starts: list[np.ndarray], ends: list[np.ndarray], lens: list[np.ndarray], gmd: np.ndarray,
    shield: GroundPlane | GroundedBox | None = None,
) -> np.ndarray:
    """K x K chain (path-summed) partial-inductance matrix of the sub-filaments.

    A ``shield`` (:class:`GroundPlane` or :class:`GroundedBox`) adds its flux-exclusion
    image filaments (good-conductor ``B_normal = 0`` mirrors, current reversed), whose
    anti-parallel mutual *lowers* L -- the ``L = L_free - M(2h)`` ground-plane reduction. The
    image is set by the winding current, so it is identical for any terminal drive.
    """

    k = len(starts)
    lmat = np.zeros((k, k))
    for i in range(k):
        # Diagonal (self of sub-filament i): each segment's own self-partial-inductance
        # plus the exact mutual between its distinct segments (coincident pairs excluded).
        self_terms = float(np.sum([self_partial_inductance(ln, gmd[i]) for ln in lens[i] if ln > 0.0]))
        m_self = _mutualfil_matrix(starts[i], ends[i], starts[i], ends[i])
        np.fill_diagonal(m_self, 0.0)
        lmat[i, i] = self_terms + float(m_self.sum())
        for j in range(i + 1, k):
            m = float(_mutualfil_matrix(starts[i], ends[i], starts[j], ends[j]).sum())
            lmat[i, j] = lmat[j, i] = m
    if shield is not None:
        for img_s, img_e in shield.magnetic_image_filaments(starts, ends):
            for i in range(k):
                for j in range(i, k):
                    m = float(_mutualfil_matrix(starts[i], ends[i], img_s[j], img_e[j]).sum())
                    lmat[i, j] += m
                    if j > i:
                        lmat[j, i] += m
    return lmat


def _chain_inductance_matrix(
    conductor: Conductor, shield: GroundPlane | GroundedBox | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L_chain, R_series)``: the K x K partial-inductance matrix of the
    sub-filaments and the per-sub-filament DC series resistance."""

    filaments, areas, gmd = conductor.subfilaments()
    total_len = conductor.total_length
    rho = conductor.material.resistivity_at(conductor.temperature)
    r_series = rho * total_len / areas  # DC resistance of each sub-filament (whole path)
    starts, ends, lens = _filament_endpoints(filaments)
    return _chain_lmat_from_filaments(starts, ends, lens, gmd, shield), r_series


def _parallel_terminal_impedance(z: np.ndarray) -> complex:
    """Reduce parallel branches, retrying a numerically invalid direct solve.

    Some LAPACK/OpenBLAS combinations have very occasionally returned non-finite values
    from ``solve`` for otherwise modest, well-conditioned complex PEEC matrices.  Check the
    solution and its residual rather than allowing that backend failure to become a silent
    ``nan`` terminal impedance.  SVD least squares is an independent, stable fallback.
    """

    ones = np.ones(z.shape[0])

    def valid(currents: np.ndarray) -> bool:
        if not np.isfinite(currents).all():
            return False
        residual = np.linalg.norm(z @ currents - ones)
        scale = np.linalg.norm(z) * np.linalg.norm(currents) + np.linalg.norm(ones)
        return bool(np.isfinite(residual) and residual <= 1e-10 * scale)

    try:
        currents = np.linalg.solve(z, ones)
    except np.linalg.LinAlgError:
        currents = np.full(z.shape[0], np.nan, dtype=complex)
    if not valid(currents):
        currents = np.linalg.lstsq(z, ones, rcond=None)[0]
    if not valid(currents):
        raise np.linalg.LinAlgError("PEEC terminal-current solve did not converge")

    admittance = ones @ currents
    if not np.isfinite(admittance) or abs(admittance) <= np.finfo(float).tiny:
        raise np.linalg.LinAlgError("PEEC terminal admittance is zero or non-finite")
    return complex(1.0 / admittance)


_FULL_UNKNOWN_LIMIT = 6000  # dense-solve guardrail for the full (per-segment) formulation


def _full_lmat_from_filaments(
    starts: list[np.ndarray], ends: list[np.ndarray], lens: list[np.ndarray], gmd: np.ndarray,
    shield: GroundPlane | GroundedBox | None = None,
) -> np.ndarray:
    """Full ``(K M) x (K M)`` branch partial-inductance matrix, branch ``b = k*M + m``.

    This is FastHenry's actual system: one branch per (path segment ``m``, cross-section
    sub-filament ``k``), with the exact closed-form mutual between every pair of straight
    branches and the analytic self partial inductance on the diagonal. The chain matrix is
    the row/column block-sum of this one. A ``shield`` adds its flux-exclusion image
    filaments (:meth:`GroundPlane.magnetic_image_filaments`) to every block.
    """

    k = len(starts)
    m = starts[0].shape[0]
    n = k * m
    if n > _FULL_UNKNOWN_LIMIT:
        raise ValueError(
            f"full formulation would need {n} branch unknowns (> {_FULL_UNKNOWN_LIMIT}); "
            "reduce the path resolution (n_per_turn) and/or cross-section cells "
            "(n_radial*n_angular, n_width*n_height, n_perimeter), or use formulation='chain'"
        )
    images = shield.magnetic_image_filaments(starts, ends) if shield is not None else []
    lb = np.empty((n, n))
    for i in range(k):
        for j in range(i, k):
            blk = _mutualfil_matrix(starts[i], ends[i], starts[j], ends[j])
            if i == j:
                # coincident pairs are the analytic self partial inductance, not mutualfil
                np.fill_diagonal(
                    blk, [self_partial_inductance(ln, gmd[i]) if ln > 0.0 else 0.0 for ln in lens[i]]
                )
            for img_s, img_e in images:  # flux-exclusion images (never coincident with source)
                blk = blk + _mutualfil_matrix(starts[i], ends[i], img_s[j], img_e[j])
            lb[i * m:(i + 1) * m, j * m:(j + 1) * m] = blk
            if j > i:
                lb[j * m:(j + 1) * m, i * m:(i + 1) * m] = blk.T
    return lb


def _solve_full_terminal(
    lb: np.ndarray, zb_diag: np.ndarray, n_fil: int, n_seg: int, omega: float
) -> tuple[complex, np.ndarray]:
    """Terminal impedance of the full per-segment branch system at ``omega``.

    Branch ``b = k*M + m`` carries current ``I[k, m]``; all ``K`` branches of segment ``m``
    connect the same two path nodes (as in FastHenry, where a segment's filaments share its
    end nodes), so KVL gives one segment voltage ``V_m`` common to its branches and KCL gives
    ``sum_k I[k, m] = I`` for every ``m`` -- the per-segment redistribution freedom that the
    chain reduction removes and that carries the proximity effect. Eliminating the branch
    currents ``I = Y B V`` (``Y = Z_b^{-1}``, ``B`` the segment-summing selector) leaves the
    M x M system ``A V = I 1`` with ``A = B^T Y B``, and the terminal impedance
    ``Z = sum_m V_m / I = 1^T A^{-1} 1`` (for ``M = 1`` this is exactly the chain formula
    ``1 / (1^T Z_b^{-1} 1)``). Returns ``(Z, branch currents at I = 1)``.
    """

    n = n_fil * n_seg
    zb = (1j * omega) * lb
    zb[np.diag_indices(n)] += zb_diag
    b = np.zeros((n, n_seg))
    b[np.arange(n), np.tile(np.arange(n_seg), n_fil)] = 1.0
    x = np.linalg.solve(zb, b)
    a = b.T @ x
    v = np.linalg.solve(a, np.ones(n_seg))
    return complex(np.sum(v)), x @ v


@dataclass(frozen=True)
class PEECImpedance:
    """Frequency-swept terminal impedance of a coil from the PEEC solve."""

    frequency: np.ndarray      # (F,) Hz
    inductance: np.ndarray     # (F,) L(w) = Im Z / w (H)
    resistance: np.ndarray     # (F,) R(w) = Re Z (ohm)
    dc_inductance: float       # L(0): all sub-filaments in parallel, no current crowding (H)
    dc_resistance: float       # R(0) = rho * length / total_area (ohm)


def extract_impedance(
    conductor: Conductor, frequencies: Sequence[float], *, formulation: str = "chain",
    ground_plane: GroundPlane | GroundedBox | None = None,
) -> PEECImpedance:
    """Solve the PEEC system for ``L(w)`` and ``R(w)``.

    ``formulation="chain"`` (default) builds the reduced ``K x K`` chain impedance
    ``Z = R + j w L`` once and, at each frequency, reduces the parallel sub-filaments to a
    terminal impedance ``Z(w) = 1 / (1^T Z^{-1} 1)``: the current crowds to the surface
    (skin effect) but each sub-filament's current is constant along the path, so the
    turn-to-turn **proximity** loss is only partially captured. Fast; right for inductance,
    isolated wires, and well-spaced windings.

    ``formulation="full"`` keeps one branch per (segment, sub-filament) -- FastHenry's
    actual system -- so the cross-section distribution redistributes at every segment and
    the proximity effect is resolved from first principles (see
    :func:`_solve_full_terminal`). Cost ``O((M K)^3)`` per frequency; use it for any
    closely-wound coil. Returns ``L(w) = Im Z / w`` and ``R(w) = Re Z``.

    ``ground_plane`` (a :class:`GroundPlane`, or a :class:`GroundedBox` for a full enclosure)
    adds the good-conductor shield's flux-exclusion image (a lossless PEC mirror), which lowers
    ``L`` -- the standard ``L = L_free - M(2h)`` reduction. It is set by the winding current, so
    it is **identical for any terminal drive** (single-ended or differential): the shield's
    effect on inductance is mode-independent (its resistive eddy loss is a separate
    surface-resistance integral). The magnetic dual of the electrostatic image in
    :func:`self_capacitance`.
    """

    if formulation not in ("chain", "full"):
        raise ValueError("formulation must be 'chain' or 'full'")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    filaments, areas, gmd = conductor.subfilaments()
    starts, ends, lens = _filament_endpoints(filaments)
    rho = conductor.material.resistivity_at(conductor.temperature)
    r_series = rho * conductor.total_length / areas  # per-sub-filament whole-path DC resistance

    l_out = np.empty(freqs.size)
    r_out = np.empty(freqs.size)
    if formulation == "full":
        k = len(filaments)
        m = starts[0].shape[0]
        lb = _full_lmat_from_filaments(starts, ends, lens, gmd, ground_plane)
        # Per-branch DC resistance: rho * segment_length / cell_area.
        rb = np.concatenate([rho * lens[i] / areas[i] for i in range(k)])
        for idx, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            z_term, _ = _solve_full_terminal(lb, rb.astype(complex), k, m, omega)
            r_out[idx] = z_term.real
            l_out[idx] = z_term.imag / omega if omega > 0 else np.nan
        # Chain matrix for the DC limits below: exactly the block-sum of the full matrix.
        lmat = lb.reshape(k, m, k, m).sum(axis=(1, 3))
    else:
        lmat = _chain_lmat_from_filaments(starts, ends, lens, gmd, ground_plane)
        k = lmat.shape[0]
        for idx, f in enumerate(freqs):
            omega = 2.0 * np.pi * f
            z = np.diag(r_series) + 1j * omega * lmat
            z_term = _parallel_terminal_impedance(z)
            r_out[idx] = z_term.real
            l_out[idx] = z_term.imag / omega if omega > 0 else np.nan

    # DC limits. At w -> 0 resistance dominates, so the current divides by conductance,
    # i0 = R^{-1}1 / (1^T R^{-1}1); the DC inductance is the inductive energy at that
    # distribution, L_dc = i0^T L i0 (not the minimum-energy 1/(1^T L^{-1} 1), which is the
    # high-frequency limit where current redistributes to minimize flux).
    g = 1.0 / r_series
    dc_r = 1.0 / np.sum(g)
    i0 = g / np.sum(g)
    dc_l = float(i0 @ lmat @ i0)
    return PEECImpedance(
        frequency=freqs,
        inductance=l_out,
        resistance=r_out,
        dc_inductance=float(dc_l),
        dc_resistance=float(dc_r),
    )


def _skin_depth(conductor: Conductor, frequency: float) -> float:
    rho = conductor.material.resistivity_at(conductor.temperature)
    return float(np.sqrt(rho / (np.pi * frequency * MU0 * conductor.material.mu_r)))


def _perimeter_offsets(conductor: Conductor, delta: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-section perimeter cells for the surface-impedance model.

    Returns ``(offsets, widths)``: ``offsets`` (n, 2) filament positions one half skin depth
    inside the boundary, ``widths`` (n,) the along-perimeter patch widths. Round -> a ring;
    rect -> the four inset edges.
    """

    if conductor.cross_section == "rect":
        w, h = float(conductor.width), float(conductor.height)
        wi, hi = max(w - delta, w * 1e-3), max(h - delta, h * 1e-3)
        perim = 2.0 * (wi + hi)
        s = (np.arange(n) + 0.5) / n * perim  # arc length around the inset rectangle
        xs, ys = np.empty(n), np.empty(n)
        for i, si in enumerate(s):
            t = si
            if t < wi:  # bottom edge
                xs[i], ys[i] = -wi / 2 + t, -hi / 2
            elif t < wi + hi:  # right edge
                xs[i], ys[i] = wi / 2, -hi / 2 + (t - wi)
            elif t < 2 * wi + hi:  # top edge
                xs[i], ys[i] = wi / 2 - (t - wi - hi), hi / 2
            else:  # left edge
                xs[i], ys[i] = -wi / 2, hi / 2 - (t - 2 * wi - hi)
        widths = np.full(n, 2.0 * (w + h) / n)
        return np.column_stack([xs, ys]), widths
    a = conductor.wire_radius
    r_fil = max(a - delta / 2.0, a * 0.5)
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    widths = np.full(n, 2.0 * np.pi * a / n)
    return np.column_stack([r_fil * np.cos(th), r_fil * np.sin(th)]), widths


def extract_impedance_surface(
    conductor: Conductor, frequencies: Sequence[float], *, n_perimeter: int = 48,
    formulation: str = "chain", shield: GroundPlane | GroundedBox | None = None,
) -> PEECImpedance:
    """Surface-impedance (SIBC) solve for the deep-skin (high ``a/delta``) regime.

    Instead of tiling the whole cross-section (which needs cells thinner than the skin depth
    -- prohibitively many when ``a/delta`` is large), this places ``n_perimeter`` filaments
    around the cross-section boundary, one half skin depth inside, and gives each the analytic
    surface impedance ``Z_s = (1+j) rho/delta * (length / patch_width)`` (its real part is the
    skin resistance, its imaginary part the internal inductance). The external inductance is
    the exact mutual between the ring filaments. Cost is ``O(n_perimeter^2)`` per frequency,
    independent of ``a/delta`` -- and the accuracy *improves* with frequency, the opposite of
    the volume solver. Complementary to :func:`extract_impedance` (best at low/moderate skin).

    ``formulation="chain"`` (default) gives each perimeter filament a path-constant current:
    skin resistance only, essentially **no proximity** (a tight solenoid comes out at the
    isolated-wire R). ``formulation="full"`` keeps one branch per (segment, perimeter cell)
    so the current redistributes *around the perimeter at every segment* -- skin (analytic,
    via ``Z_s``) **plus proximity** resolved, at deep skin where the volume solver cannot go.
    This is the right mode for a closely-wound RF coil.

    ``shield`` (a :class:`GroundedBox` enclosure or a :class:`GroundPlane`) adds the
    good-conductor flux-exclusion image, lowering ``L`` (and raising the self-resonance) just
    as in :func:`extract_impedance` -- the magnetic counterpart of the same shield's
    electrostatic image in :func:`self_capacitance`, and mode-independent. Its resistive wall
    loss is a separate surface-resistance integral (see ``examples/plot_shielded_nqr_coil.py``).
    """

    if formulation not in ("chain", "full"):
        raise ValueError("formulation must be 'chain' or 'full'")
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    rho = conductor.material.resistivity_at(conductor.temperature)
    total_len = conductor.total_length
    # The ring sits one half skin depth inside the boundary; delta/2 << a over a band, so the
    # geometry (and its external-inductance matrix) is built ONCE at a reference skin
    # depth, and only the surface impedance varies per frequency.
    delta_ref = _skin_depth(conductor, float(np.sqrt(freqs[0] * freqs[-1])))
    offsets, widths = _perimeter_offsets(conductor, delta_ref, int(n_perimeter))
    filaments = _sweep_offsets(conductor.path_points, offsets)
    starts, ends, lens = _filament_endpoints(filaments)
    n = len(filaments)
    gmd = np.sqrt(widths * delta_ref / np.pi) * np.exp(-0.25)

    l_out = np.empty(freqs.size)
    r_out = np.empty(freqs.size)
    if formulation == "full":
        m = starts[0].shape[0]
        lb = _full_lmat_from_filaments(starts, ends, lens, gmd, shield)
        for idx, f in enumerate(freqs):
            delta = _skin_depth(conductor, f)
            # Per-branch surface impedance: (1+j) rho/delta * segment_length / patch_width.
            zb = np.concatenate(
                [(1.0 + 1j) * (rho / delta) * lens[i] / widths[i] for i in range(n)]
            )
            omega = 2.0 * np.pi * f
            z_term, _ = _solve_full_terminal(lb, zb, n, m, omega)
            r_out[idx] = z_term.real
            l_out[idx] = z_term.imag / omega
    else:
        lmat = _chain_lmat_from_filaments(starts, ends, lens, gmd, shield)
        for idx, f in enumerate(freqs):
            delta = _skin_depth(conductor, f)
            z_surface = (1.0 + 1j) * (rho / delta) * (total_len / widths)
            omega = 2.0 * np.pi * f
            z = np.diag(z_surface) + 1j * omega * lmat
            z_term = _parallel_terminal_impedance(z)
            r_out[idx] = z_term.real
            l_out[idx] = z_term.imag / omega
    dc_r = rho * total_len / (np.pi * conductor.wire_radius**2) if conductor.cross_section == "round" \
        else rho * total_len / (conductor.width * conductor.height)
    return PEECImpedance(frequency=freqs, inductance=l_out, resistance=r_out,
                         dc_inductance=float(l_out[0]) if l_out.size else float("nan"), dc_resistance=float(dc_r))


def current_distribution(
    conductor: Conductor, frequency: float, *, formulation: str = "chain",
    segment: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sub-filament current magnitude across the cross-section at ``frequency``.

    Returns ``(offsets, current)``: ``offsets`` (K, 2) the cross-section cell positions in
    the local frame and ``current`` (K,) the normalized current magnitude ``|I_k|``
    (summing to 1). As frequency rises the current crowds to the surface (skin effect) and,
    for a multi-turn/leg path, toward or away from neighbouring conductors (proximity) --
    the physical picture behind the AC resistance. Useful for visualization.

    With ``formulation="full"`` the distribution is taken at one path segment (``segment``,
    default the middle one), where the per-segment redistribution shows the *actual*
    proximity crowding toward the neighbouring turns -- the chain default can only show the
    path-averaged (skin-dominated) pattern.
    """

    if formulation not in ("chain", "full"):
        raise ValueError("formulation must be 'chain' or 'full'")
    filaments, areas, gmd = conductor.subfilaments()
    if conductor.cross_section == "rect":
        offsets, _, _ = _rect_cross_section(conductor.width, conductor.height, conductor.n_width, conductor.n_height)
    else:
        offsets, _, _ = _round_cross_section(conductor.wire_radius, conductor.n_radial, conductor.n_angular)
    omega = 2.0 * np.pi * float(frequency)
    rho = conductor.material.resistivity_at(conductor.temperature)
    if formulation == "full":
        starts, ends, lens = _filament_endpoints(filaments)
        k = len(filaments)
        m = starts[0].shape[0]
        seg = m // 2 if segment is None else int(segment)
        if not 0 <= seg < m:
            raise ValueError(f"segment must be in [0, {m})")
        lb = _full_lmat_from_filaments(starts, ends, lens, gmd)
        rb = np.concatenate([rho * lens[i] / areas[i] for i in range(k)])
        _, i_branch = _solve_full_terminal(lb, rb.astype(complex), k, m, omega)
        i_vec = i_branch[seg::m]  # branch b = k*M + m -> filament currents at segment `seg`
    else:
        lmat, r_series = _chain_inductance_matrix(conductor)
        z = np.diag(r_series) + 1j * omega * lmat
        i_vec = np.linalg.solve(z, np.ones(lmat.shape[0]))
    mag = np.abs(i_vec)
    return offsets, mag / np.sum(mag)


@dataclass(frozen=True)
class GroundedBox:
    """A grounded rectangular shield enclosing the coil (walls at potential zero).

    ``center`` and ``half_extents`` (m) define the box; the six walls at ``center +/- half``
    are grounded and enter the electrostatic solve as a 3-D lattice of image charges (method
    of images for a rectangular cavity), which raise the coil's capacitance and lower its
    self-resonant frequency. ``n_orders`` is the number of reflections kept per axis
    (``1`` -> the leading lattice, usually converged for a box a few coil-sizes across).
    """

    center: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    n_orders: int = 1

    def _axis_images(self, x: np.ndarray, lo: float, hi: float) -> list[tuple[np.ndarray, float]]:
        """Per-axis image coordinates/signs of ``x`` between grounded planes ``lo``, ``hi``."""

        span = hi - lo
        out: list[tuple[np.ndarray, float]] = []
        for k in range(-int(self.n_orders), int(self.n_orders) + 1):
            out.append((x + 2.0 * k * span, +1.0))          # even reflections
            out.append((2.0 * lo - x + 2.0 * k * span, -1.0))  # odd reflections
        return out

    def image_terms(self, points: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return ``(mirror_points, sign)`` for the full 3-D image lattice (real charge
        excluded)."""

        c = np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.half_extents, dtype=np.float64)
        lo, hi = c - h, c + h
        ax = [self._axis_images(points[:, a], lo[a], hi[a]) for a in range(3)]
        terms: list[tuple[np.ndarray, float]] = []
        for ix, (xa, sx) in enumerate(ax[0]):
            for iy, (ya, sy) in enumerate(ax[1]):
                for iz, (za, sz) in enumerate(ax[2]):
                    sign = sx * sy * sz
                    # the real charge is the k=0 even-reflection term on every axis
                    if sign == 1.0 and ix == self.n_orders * 2 and iy == self.n_orders * 2 and iz == self.n_orders * 2:
                        continue
                    terms.append((np.column_stack([xa, ya, za]), sign))
        return terms

    def fundamental_resonance(self) -> float:
        """Lowest resonant frequency (Hz) of the box treated as a closed PEC cavity.

        The rectangular-cavity mode spectrum is ``f = (c/2) sqrt((l/Lx)^2 + (m/Ly)^2 +
        (n/Lz)^2)`` over interior side lengths ``L = 2 * half_extents``; the lowest mode with
        two non-zero indices (the smallest that actually resonates -- a mode needs at least
        two indices) sets the cutoff. Below it a closed conducting box supports no propagating
        field, so external radiation is evanescent (suppressed); it is the frequency above
        which the closed-cavity radiation approximation in :func:`radiation_resistance` breaks
        down. Used only for that guard, not the electrostatic capacitance.
        """

        sides = 2.0 * np.asarray(self.half_extents, dtype=np.float64)
        inv2 = 1.0 / sides**2
        # smallest sum of two distinct 1/L^2 terms -> drop the largest side's contribution
        two = float(np.sum(inv2) - np.max(inv2))
        return float(0.5 * C0 * np.sqrt(two))

    def magnetic_image_filaments(self, starts, ends):
        """Flux-exclusion image filaments for the box's six primary wall reflections.

        Each wall is a good conductor holding ``B_normal = 0``; it reflects the filaments
        across itself and reverses their current (a single reflection). Returns one
        ``(img_starts, img_ends)`` per wall. These six are the leading term of the
        rectangular-cavity image lattice -- the dominant flux-exclusion inductance reduction
        (higher-order edge/corner reflections, a further ~1% here, are omitted so the SIBC
        solve stays fast). The magnetic dual of the electrostatic :meth:`image_terms`.
        """

        c = np.asarray(self.center, dtype=np.float64)
        h = np.asarray(self.half_extents, dtype=np.float64)
        images = []
        for a in range(3):
            for sgn in (-1.0, 1.0):
                wall = c[a] + sgn * h[a]

                def refl(p, a=a, wall=wall):
                    q = p.copy()
                    q[:, a] = 2.0 * wall - q[:, a]
                    return q

                # reflect ends -> new starts (start/end swap reverses the current)
                images.append(([refl(e) for e in ends], [refl(s) for s in starts]))
        return images


@dataclass(frozen=True)
class GroundPlane:
    """A single infinite grounded conducting plane at zero potential (a half-space wall).

    The special case of a PCB or chassis ground plane below a planar coil: unlike
    :class:`GroundedBox` (a closed cavity) it does not enclose the coil, and enters the
    electrostatic solve as the single image charge of the method of images (a mirror charge
    of opposite sign reflected across the plane), which holds the plane at zero potential.
    ``point`` is any point on the plane and ``normal`` its (not necessarily unit) normal.

    Feed it to :func:`self_capacitance` / :func:`capacitance_to_ground` exactly like a
    ``GroundedBox``. Driving the winding single-ended (the default linear potential, one
    terminal grounded) versus differentially (an antisymmetric ``potential`` profile, the
    winding centre at virtual ground) then gives the two very different coil-to-ground
    couplings behind the single-ended/differential Q split (see
    ``examples/plot_pcb_coil_ground_mode_q.py``).
    """

    point: tuple[float, float, float]
    normal: tuple[float, float, float]

    def reflect(self, points: np.ndarray) -> np.ndarray:
        """Mirror ``points`` (…, 3) geometrically across the plane."""

        p0 = np.asarray(self.point, dtype=np.float64)
        n = np.asarray(self.normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        signed = (points - p0) @ n
        return points - 2.0 * signed[..., None] * n

    def image_terms(self, points: np.ndarray) -> list[tuple[np.ndarray, float]]:
        """Return ``[(mirror_points, -1.0)]``: the electrostatic image (opposite charge)."""

        return [(self.reflect(points), -1.0)]

    def magnetic_image_filaments(self, starts, ends):
        """Flux-exclusion image filaments for a good-conductor plane (one image).

        Returns ``[(img_starts, img_ends)]`` -- each per-filament segment array reflected
        across the plane with its current reversed (start/end swapped), the ``B_normal = 0``
        image whose (anti-parallel) mutual with the coil *lowers* L. See
        :func:`extract_impedance`.
        """

        return [([self.reflect(e) for e in ends], [self.reflect(s) for s in starts])]


def _potential_coefficient_matrix(
    conductor: Conductor, shield: GroundedBox | GroundPlane | None = None,
    relative_permittivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell potential-coefficient matrix ``P`` (1/F) over the path segments, and lengths.

    Thin-wire electrostatics: each path segment carries a uniform total charge ``Q_j``;
    ``P[i, j]`` is the potential at segment ``i``'s midpoint per unit ``Q_j``. Self term:
    the analytic uniformly-charged straight-segment midpoint potential; off-diagonal: the
    midpoint approximation ``1/(4 pi eps0 d_ij)``. A grounded ``shield`` (:class:`GroundedBox`
    or :class:`GroundPlane`) adds image charges (the conductor at zero potential);
    ``relative_permittivity`` scales for a uniform dielectric.
    """

    pts = conductor.path_points
    mids = 0.5 * (pts[:-1] + pts[1:])
    lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    a = conductor.wire_radius
    k_e = 1.0 / (4.0 * np.pi * EPS0 * relative_permittivity)

    d = np.linalg.norm(mids[:, None, :] - mids[None, :, :], axis=-1)
    np.fill_diagonal(d, a)
    p = k_e / np.maximum(d, a)
    np.fill_diagonal(p, k_e / lengths * 2.0 * np.arcsinh(lengths / (2.0 * a)))

    if shield is not None:
        for mirror, sign in shield.image_terms(mids):
            di = np.linalg.norm(mids[:, None, :] - mirror[None, :, :], axis=-1)
            p = p + sign * k_e / np.maximum(di, a)
    return p, lengths


def self_capacitance(
    conductor: Conductor, *, shield: GroundedBox | GroundPlane | None = None,
    relative_permittivity: float = 1.0, potential=None,
) -> float:
    """Lumped self-capacitance (F) of the coil from an electrostatic energy method.

    Solves the thin-wire potential-coefficient system for the charge distribution under a
    prescribed winding potential and returns ``C_eff = 2 W_elec / V0^2 = v^T P^{-1} v``.
    The default ``potential`` is the physically-realistic **single-ended** linear ramp
    (0 at the grounded terminal rising to ``V0 = 1`` at the driven one, as in a resonating
    coil with one end grounded). Pass ``potential`` as either an ``(N,)`` array of
    per-segment potentials or a callable of the arc-length fraction ``s in [0, 1]`` to model
    other drives -- e.g. ``potential=lambda s: s - 0.5`` for **differential** (balanced)
    drive, antisymmetric about a virtual ground at the winding centre, which nulls most of
    the common-mode coupling to a nearby ground plane.

    A grounded ``shield`` (:class:`GroundedBox` walls or a :class:`GroundPlane`, at zero
    potential) and a dielectric former (``relative_permittivity``) both raise ``C_eff`` and
    so lower the self-resonant frequency; the *difference* ``C_eff(with shield) -
    C_eff(without)`` is the coil-to-ground capacitance driving the ground-coupled loss.
    """

    p, lengths = _potential_coefficient_matrix(conductor, shield, relative_permittivity)
    cumlen = np.cumsum(lengths) - 0.5 * lengths
    s = cumlen / conductor.total_length  # arc-length fraction of each segment midpoint
    if potential is None:
        v = s  # single-ended linear winding potential, V0 = 1
    elif callable(potential):
        v = np.asarray(potential(s), dtype=np.float64)
    else:
        v = np.asarray(potential, dtype=np.float64)
        if v.shape != s.shape:
            raise ValueError(f"potential array must have one entry per segment ({s.size})")
    q = np.linalg.solve(p, v)
    return float(float(q @ v))


def capacitance_to_ground(
    conductor: Conductor, *, shield: GroundedBox | GroundPlane | None = None,
    relative_permittivity: float = 1.0,
) -> float:
    """Isolated self-capacitance to ground (F): the charge held at unit potential,
    ``C = 1^T P^{-1} 1``. With a grounded ``shield`` this is the coil-to-shield capacitance.
    """

    p, _ = _potential_coefficient_matrix(conductor, shield, relative_permittivity)
    n = p.shape[0]
    q = np.linalg.solve(p, np.ones(n))
    return float(np.sum(q))


def helical_solenoid(
    *,
    diameter: float,
    length: float,
    turns: int,
    wire_radius: float,
    material: ConductorMaterial = ANNEALED_COPPER,
    n_per_turn: int = 16,
    n_radial: int = 6,
    n_angular: int = 8,
    temperature: float | None = None,
    axis: str = "z",
) -> Conductor:
    """Build a :class:`Conductor` for a helical single-layer solenoid.

    Unlike :func:`spin_dynamics.fields.coils.solenoid` (a stack of disconnected loops), this
    is a genuine connected helix -- the single continuous wire the PEEC solver needs.
    ``diameter``/``length``/``wire_radius`` are in metres; ``n_per_turn`` sets the path
    resolution.
    """

    turns = int(turns)
    if turns < 1:
        raise ValueError("turns must be >= 1")
    theta = np.linspace(0.0, 2.0 * np.pi * turns, turns * int(n_per_turn) + 1)
    axial = np.linspace(-length / 2.0, length / 2.0, theta.size)
    u = (diameter / 2.0) * np.cos(theta)
    v = (diameter / 2.0) * np.sin(theta)
    idx = {"x": 0, "y": 1, "z": 2}[axis]
    plane = [i for i in range(3) if i != idx]
    pts = np.zeros((theta.size, 3))
    pts[:, plane[0]] = u
    pts[:, plane[1]] = v
    pts[:, idx] = axial
    return Conductor(
        path_points=pts,
        wire_radius=float(wire_radius),
        material=material,
        n_radial=int(n_radial),
        n_angular=int(n_angular),
        temperature=temperature,
    )


def self_resonant_frequency(conductor: Conductor) -> float:
    """First self-resonant frequency (Hz) ``1 / (2 pi sqrt(L C))``.

    Uses the DC inductance and the electrostatic :func:`self_capacitance`; this is the
    physical (Medhurst-style) self-resonance, distinct from the QOIL lumped-equivalent
    ``C_p`` fitting capacitance.
    """

    lmat, r_series = _chain_inductance_matrix(conductor)
    g = 1.0 / r_series
    i0 = g / np.sum(g)
    l_dc = float(i0 @ lmat @ i0)
    c = self_capacitance(conductor)
    return float(1.0 / (2.0 * np.pi * np.sqrt(l_dc * c)))


def radiation_resistance(
    conductor: Conductor, frequency: float, *, shield: GroundedBox | GroundPlane | None = None
) -> float:
    """First-order (magnetic-dipole) radiation resistance (ohm) of the coil.

    An electrically small current loop radiates predominantly as a magnetic dipole with
    moment ``m = I * A_net``, giving ``R_rad = eta0 k^4 |A_net|^2 / (6 pi)`` with
    ``k = 2 pi f / c`` and ``A_net`` the **net vector area** of the wire path,
    ``A_net = (1/2) sum_i r_i x r_(i+1)`` around the polyline closed end-to-end (the closure
    stands in for the feed/tuning connection; the vector area of a closed polygon is
    translation-invariant). For a single circular loop this reduces to the textbook
    ``20 pi^2 (C/lambda)^4`` and for an N-turn solenoid it scales as ``N^2``; opposed-loop
    geometries (Maxwell pairs, gradient coils) have ``A_net ~ 0`` and radiate only through
    the neglected higher multipoles.

    **Shielding.** A grounded enclosure is a cavity for radiation, not the image-charge
    lattice used for capacitance: below its lowest resonant mode
    (:meth:`GroundedBox.fundamental_resonance`) a closed conductor supports no propagating
    external field, so the far-field radiation is suppressed to ~zero -- the near-field
    energy that would have radiated is instead stored reactively and dissipated in the walls
    (the wall eddy loss, modelled separately). Passing ``shield=GroundedBox(...)`` therefore
    returns 0 below the box's cavity cutoff (consistent with feeding the same box to
    :func:`self_capacitance`), and a ``UserWarning`` fires as the frequency approaches that
    cutoff, where the closed-cavity approximation no longer holds.

    First-order means: magnetic-dipole term only, valid for coils small compared with the
    wavelength (a ``UserWarning`` is emitted when the coil extent exceeds ~lambda/12, where
    higher multipoles and the electric-dipole/antenna term start to matter). Common-mode
    ("antenna-effect") radiation via the feed leads is a property of the installation, not
    the coil, and is not modelled.
    """

    pts = conductor.path_points
    f = float(frequency)
    if f <= 0.0:
        return 0.0
    if isinstance(shield, GroundedBox):
        # Only a CLOSED conductor suppresses radiation (a cavity below cutoff). An open
        # GroundPlane does not enclose the coil and is ignored here (it modifies the pattern
        # but does not null the far field -- beyond this first-order model).
        import warnings

        f_cav = shield.fundamental_resonance()
        if f >= 0.5 * f_cav:
            warnings.warn(
                f"frequency {f:.3g} Hz approaches the shield's cavity resonance "
                f"({f_cav:.3g} Hz): the closed-cavity radiation-suppression model no longer "
                "holds (cavity modes and aperture leakage are not modelled)",
                UserWarning,
                stacklevel=2,
            )
        if f < f_cav:
            # closed grounded box below cutoff: external radiation is evanescent (~0); the
            # would-be radiated power appears in the wall eddy loss, modelled separately.
            return 0.0
    a_vec = 0.5 * (np.cross(pts[:-1], pts[1:]).sum(axis=0) + np.cross(pts[-1], pts[0]))
    k = 2.0 * np.pi * f / C0
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if k * extent > 0.5:
        import warnings

        warnings.warn(
            f"coil extent {extent:.3g} m is not small vs the wavelength "
            f"({2 * np.pi / k:.3g} m): the magnetic-dipole radiation model is first-order "
            "and under-predicts radiation here",
            UserWarning,
            stacklevel=2,
        )
    return float(ETA0 * k**4 * float(a_vec @ a_vec) / (6.0 * np.pi))


@dataclass(frozen=True)
class PEECCoilProperties:
    """Lumped RF properties of an arbitrary coil from the PEEC solve.

    The universal subset of
    :class:`spin_dynamics.fields.coil_properties.CoilProperties` (which is solenoid-specific),
    with the same field names so it drops into the same probe-noise / matching pipeline.
    """

    frequency: float
    inductance: float          # L(w) at frequency (H)
    ac_resistance: float       # ohmic R(w) at frequency: skin + proximity (ohm)
    q_factor: float            # w L / (R_ohmic + R_rad)
    self_capacitance: float    # electrostatic self-capacitance (F)
    self_resonant_frequency: float  # 1/(2 pi sqrt(L_dc C)) (Hz)
    dc_inductance: float       # L(0) (H)
    dc_resistance: float       # R(0) (ohm)
    radiation_resistance: float = 0.0  # first-order magnetic-dipole R_rad (ohm)

    @property
    def total_resistance(self) -> float:
        """Total series loss resistance: ohmic (skin + proximity) + radiation (ohm)."""

        return self.ac_resistance + self.radiation_resistance

    def tuning_capacitance(self, frequency: float | None = None) -> float:
        """Capacitance that resonates the coil at ``frequency`` (default: solve frequency)."""

        f = self.frequency if frequency is None else float(frequency)
        return float(1.0 / ((2.0 * np.pi * f) ** 2 * self.inductance))

    def to_probe_params(self) -> dict[str, float]:
        """Return ``{"L", "R", "C"}`` for the probe noise model (``R`` = total series loss)."""

        return {"L": self.inductance, "R": self.total_resistance, "C": self.tuning_capacitance()}


def coil_properties_peec(
    conductor: Conductor, frequency: float, *, formulation: str = "full",
    include_radiation: bool = True, shield: GroundedBox | GroundPlane | None = None,
    relative_permittivity: float = 1.0,
) -> PEECCoilProperties:
    """Extract lumped RF properties of an arbitrary coil at ``frequency`` via PEEC.

    Solves the PEEC impedance for ``L(w)``/``R(w)``, the electrostatic self-capacitance and
    the self-resonance, and packages them like
    :class:`spin_dynamics.fields.coil_properties.CoilProperties` so an arbitrary-geometry
    coil feeds the same SNR / matching workflow. The default ``formulation="full"`` resolves
    the turn-to-turn proximity loss (FastHenry's per-segment system) so the Q of a tight coil
    is right; pass ``formulation="chain"`` for the fast reduced solve when only L / C /
    self-resonance matter. See the module docstring for the resistance resolution ceiling at
    high ``a/delta``.

    ``include_radiation`` (default True) adds the first-order magnetic-dipole
    :func:`radiation_resistance` to the loss: ``q_factor`` and ``to_probe_params()`` use the
    total ``R_ohmic + R_rad`` while ``ac_resistance`` stays purely ohmic. With its ``k^4``
    scaling this is negligible at NMR/NQR frequencies for ordinary coil sizes and becomes
    the Q ceiling at VHF or for large loops.

    ``shield`` (a :class:`GroundedBox` enclosure or a :class:`GroundPlane`) and
    ``relative_permittivity`` (a dielectric former) enter the self-capacitance and
    self-resonance as in :func:`self_capacitance`. The same shield is a good conductor
    magnetically, so it **also lowers L** via the flux-exclusion image (:func:`extract_impedance`)
    -- both effects on the self-resonance, partly offsetting -- and a `GroundedBox` additionally
    **suppresses the radiation** below its cavity cutoff (:func:`radiation_resistance`). Its own
    resistive wall/eddy loss is a separate surface integral (see the shielded-coil examples) and
    is not added here.
    """

    f = float(frequency)
    imp = extract_impedance(conductor, [f], formulation=formulation, ground_plane=shield)
    c = self_capacitance(conductor, shield=shield, relative_permittivity=relative_permittivity)
    f_res = float(1.0 / (2.0 * np.pi * np.sqrt(imp.dc_inductance * c)))
    ind = float(imp.inductance[0])
    res = float(imp.resistance[0])
    r_rad = radiation_resistance(conductor, f, shield=shield) if include_radiation else 0.0
    r_total = res + r_rad
    q = float(2.0 * np.pi * f * ind / r_total) if r_total > 0 else float("inf")
    return PEECCoilProperties(
        frequency=f,
        inductance=ind,
        ac_resistance=res,
        q_factor=q,
        self_capacitance=float(c),
        self_resonant_frequency=f_res,
        dc_inductance=float(imp.dc_inductance),
        dc_resistance=float(imp.dc_resistance),
        radiation_resistance=float(r_rad),
    )
