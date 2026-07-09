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
``k``, and the chain inductance ``L[k, k'] = mutual_inductance(filament_k, filament_k')``
reuses the Neumann kernel in :mod:`spin_dynamics.fields.quasistatic`. Solving
``Z_chain = R + j w L`` for the terminal impedance
``Z(w) = 1 / (1^T Z_chain^{-1} 1)`` lets the current redistribute across the cross-section,
which is exactly the skin effect (own cross-section) and proximity effect (neighbouring
turns/legs of the same path). A dual thin-wire electrostatic solve on the same geometry
gives the self-capacitance and hence the self-resonant frequency.

**Resolution ceiling (state it honestly).** The cross-section must resolve the skin depth
``delta``: at ``a/delta`` up to ~5 a few hundred cells give sub-percent AC-resistance
accuracy (validated against the exact Kelvin-function ratio); deeper skin (high-frequency
RF, ``a/delta`` >~ 15) needs many cells and the resistance is under-resolved unless the
cell count is raised. Inductance and capacitance are geometry-dominated and remain accurate
at coarse cross-section resolution. A surface-impedance backend for the deep-skin regime is
a planned follow-on; for a single-layer solenoid in that regime prefer
:func:`spin_dynamics.fields.coil_properties.solenoid_properties`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from spin_dynamics.fields.coil_properties import ANNEALED_COPPER, ConductorMaterial
from spin_dynamics.fields.magnetostatics import MU0
from spin_dynamics.fields.quasistatic import mutual_inductance

EPS0 = 8.8541878128e-12  # vacuum permittivity (F/m)

Segment = tuple[np.ndarray, np.ndarray]

__all__ = [
    "Conductor",
    "conductor_from_segments",
    "helical_solenoid",
    "self_partial_inductance",
    "filament_self_inductance",
    "extract_impedance",
    "current_distribution",
    "self_capacitance",
    "self_resonant_frequency",
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
    """Equal-area polar tiling of a round wire cross-section.

    Returns ``(offsets, areas, gmd)``: ``offsets`` (K, 2) cell-centroid positions in the
    local frame, ``areas`` (K,) cell areas, and ``gmd`` (K,) the self geometric-mean radius
    of each cell (area-equivalent circle, ``sqrt(A/pi) * e^{-1/4}``).
    """

    n_radial = int(n_radial)
    n_angular = int(n_angular)
    if n_radial < 1 or n_angular < 1:
        raise ValueError("n_radial and n_angular must be >= 1")
    r_edges = radius * np.sqrt(np.linspace(0.0, 1.0, n_radial + 1))
    xs, ys, areas = [], [], []
    for i in range(n_radial):
        r0, r1 = r_edges[i], r_edges[i + 1]
        rc = (2.0 / 3.0) * (r1**3 - r0**3) / (r1**2 - r0**2) if r1 > r0 else 0.5 * (r0 + r1)
        for j in range(n_angular):
            area = 0.5 * (r1**2 - r0**2) * (2.0 * np.pi / n_angular)
            if n_angular == 1:
                # A full-circle sector's centroid is on the axis, not at radius rc.
                xs.append(0.0)
                ys.append(0.0)
            else:
                thc = 2.0 * np.pi * (j + 0.5) / n_angular
                xs.append(rc * np.cos(thc))
                ys.append(rc * np.sin(thc))
            areas.append(area)
    offsets = np.column_stack([xs, ys])
    areas = np.asarray(areas, dtype=np.float64)
    gmd = np.sqrt(areas / np.pi) * np.exp(-0.25)
    return offsets, areas, gmd


@dataclass(frozen=True)
class Conductor:
    """A single current-carrying wire: a polyline centreline plus a round cross-section.

    ``path_points`` is an ``(M+1, 3)`` array of vertices (m); ``wire_radius`` the conductor
    radius (m); ``material`` a :class:`ConductorMaterial`; ``temperature`` (K, optional) the
    operating temperature for the resistivity. The cross-section is tiled into
    ``n_radial * n_angular`` parallel sub-filaments to resolve the current distribution.
    The two ends of the path are the terminals.
    """

    path_points: np.ndarray
    wire_radius: float
    material: ConductorMaterial = ANNEALED_COPPER
    n_radial: int = 6
    n_angular: int = 8
    temperature: float | None = None
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.path_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
            raise ValueError("path_points must be an (M+1, 3) array with M >= 1 segments")
        if self.wire_radius <= 0.0:
            raise ValueError("wire_radius must be positive")
        object.__setattr__(self, "path_points", pts)

    @property
    def n_cells(self) -> int:
        return int(self.n_radial) * int(self.n_angular)

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
        e1, e2 = _rotation_minimizing_frames(self.path_points)
        offsets, areas, gmd = _round_cross_section(self.wire_radius, self.n_radial, self.n_angular)
        # Vertex positions for each cell: (K, M+1, 3)
        base = self.path_points[np.newaxis, :, :]
        disp = offsets[:, 0, np.newaxis, np.newaxis] * e1[np.newaxis, :, :] \
            + offsets[:, 1, np.newaxis, np.newaxis] * e2[np.newaxis, :, :]
        verts = base + disp
        filaments = [
            [(verts[k, m], verts[k, m + 1]) for m in range(verts.shape[1] - 1)]
            for k in range(verts.shape[0])
        ]
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

    Sum of each segment's :func:`self_partial_inductance` plus the mutual partial
    inductance between distinct segments of the same filament. The cross-segment part is
    ``mutual_inductance(filament, filament)`` -- its coincident-segment terms vanish (the
    vector potential is guarded on the segment axis), leaving exactly the ``i != j`` sum.
    """

    lengths = [float(np.linalg.norm(np.asarray(e) - np.asarray(s))) for s, e in filament]
    self_terms = sum(self_partial_inductance(ln, gmd_radius) for ln in lengths if ln > 0.0)
    cross = mutual_inductance(filament, filament)
    return float(self_terms + cross)


def _pair_mutual(mid_i: np.ndarray, dl_i: np.ndarray, s_j: np.ndarray, e_j: np.ndarray) -> float:
    """Neumann mutual inductance (H) between two filaments, vectorized over segments.

    ``mid_i``/``dl_i`` are the (M_i, 3) segment midpoints and vectors of filament i;
    ``s_j``/``e_j`` the (M_j, 3) segment endpoints of filament j. Sums the exact
    straight-segment vector potential of every j-segment evaluated at every i-midpoint,
    dotted with ``dl_i`` -- the same closed form as
    :func:`quasistatic.vector_potential` but with the inner segment loop broadcast away.
    Coincident-segment terms vanish (guarded), so it also gives the cross-segment self sum.
    """

    length_j = np.linalg.norm(e_j - s_j, axis=1)  # (Mj,)
    good = length_j > 0.0
    s_j, e_j, length_j = s_j[good], e_j[good], length_j[good]
    e_hat = (e_j - s_j) / length_j[:, None]  # (Mj, 3)
    # r1, r2 from every midpoint (Mi) to every j-segment endpoint (Mj): (Mi, Mj)
    r1 = np.linalg.norm(mid_i[:, None, :] - s_j[None, :, :], axis=-1)
    r2 = np.linalg.norm(mid_i[:, None, :] - e_j[None, :, :], axis=-1)
    num = r1 + r2 + length_j[None, :]
    denom = r1 + r2 - length_j[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logt = np.where(denom > 0.0, np.log(num / np.where(denom > 0.0, denom, 1.0)), 0.0)
    # A(mid_i) = sum_j (mu0/4pi) log_ij e_hat_j ; then M = sum_i dl_i . A(mid_i)
    a_vec = (MU0 / (4.0 * np.pi)) * np.einsum("ij,jk->ik", logt, e_hat)  # (Mi, 3)
    return float(np.sum(dl_i * a_vec))


def _chain_inductance_matrix(conductor: Conductor) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L_chain, R_series)``: the K x K partial-inductance matrix of the
    sub-filaments and the per-sub-filament DC series resistance."""

    filaments, areas, gmd = conductor.subfilaments()
    k = len(filaments)
    total_len = conductor.total_length
    rho = conductor.material.resistivity_at(conductor.temperature)
    r_series = rho * total_len / areas  # DC resistance of each sub-filament (whole path)

    # Precompute per-filament segment endpoints, midpoints, vectors and lengths.
    starts = [np.array([s for s, _ in f]) for f in filaments]
    ends = [np.array([e for _, e in f]) for f in filaments]
    mids = [0.5 * (s + e) for s, e in zip(starts, ends)]
    dls = [e - s for s, e in zip(starts, ends)]
    lens = [np.linalg.norm(d, axis=1) for d in dls]

    lmat = np.zeros((k, k))
    for i in range(k):
        self_terms = float(np.sum([self_partial_inductance(ln, gmd[i]) for ln in lens[i] if ln > 0.0]))
        lmat[i, i] = self_terms + _pair_mutual(mids[i], dls[i], starts[i], ends[i])
        for j in range(i + 1, k):
            m = _pair_mutual(mids[i], dls[i], starts[j], ends[j])
            lmat[i, j] = lmat[j, i] = m
    return lmat, r_series


@dataclass(frozen=True)
class PEECImpedance:
    """Frequency-swept terminal impedance of a coil from the PEEC solve."""

    frequency: np.ndarray      # (F,) Hz
    inductance: np.ndarray     # (F,) L(w) = Im Z / w (H)
    resistance: np.ndarray     # (F,) R(w) = Re Z (ohm)
    dc_inductance: float       # L(0): all sub-filaments in parallel, no current crowding (H)
    dc_resistance: float       # R(0) = rho * length / total_area (ohm)


def extract_impedance(
    conductor: Conductor, frequencies: Sequence[float]
) -> PEECImpedance:
    """Solve the PEEC chain system for ``L(w)`` and ``R(w)`` including skin + proximity.

    Builds the ``K x K`` chain-impedance ``Z = R + j w L`` once and, at each frequency,
    reduces the parallel sub-filaments to a terminal impedance
    ``Z(w) = 1 / (1^T Z^{-1} 1)``; the current that flows is ``I_k proportional to
    (Z^{-1} 1)_k``, which crowds to the surface (skin) and toward/away from neighbouring
    conductors (proximity) as ``w`` rises. Returns ``L(w) = Im Z / w`` and ``R(w) = Re Z``.
    """

    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    lmat, r_series = _chain_inductance_matrix(conductor)
    k = lmat.shape[0]
    ones = np.ones(k)

    l_out = np.empty(freqs.size)
    r_out = np.empty(freqs.size)
    for idx, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        z = np.diag(r_series) + 1j * omega * lmat
        y = ones @ np.linalg.solve(z, ones)
        z_term = 1.0 / y
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


def current_distribution(
    conductor: Conductor, frequency: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sub-filament current magnitude across the cross-section at ``frequency``.

    Returns ``(offsets, current)``: ``offsets`` (K, 2) the cross-section cell positions in
    the local frame and ``current`` (K,) the normalized current magnitude ``|I_k|``
    (summing to 1). As frequency rises the current crowds to the surface (skin effect) and,
    for a multi-turn/leg path, toward or away from neighbouring conductors (proximity) --
    the physical picture behind the AC resistance. Useful for visualization.
    """

    _, areas, _ = conductor.subfilaments()
    offsets, _, _ = _round_cross_section(conductor.wire_radius, conductor.n_radial, conductor.n_angular)
    lmat, r_series = _chain_inductance_matrix(conductor)
    omega = 2.0 * np.pi * float(frequency)
    z = np.diag(r_series) + 1j * omega * lmat
    k = lmat.shape[0]
    i_vec = np.linalg.solve(z, np.ones(k))
    mag = np.abs(i_vec)
    return offsets, mag / np.sum(mag)


def _potential_coefficient_matrix(conductor: Conductor) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell potential-coefficient matrix ``P`` (1/F) over the path segments, and the
    segment lengths.

    Thin-wire electrostatics: each path segment carries a uniform total charge ``Q_j``;
    ``P[i, j]`` is the potential at segment ``i``'s midpoint per unit ``Q_j``. The self
    term is the analytic potential at the midpoint of a uniformly charged straight segment,
    ``(1/(4 pi eps0 L)) * 2 asinh(L/2a)``; the off-diagonal is the midpoint approximation
    ``1/(4 pi eps0 d_ij)`` (per unit total charge, so no length factor).
    """

    pts = conductor.path_points
    starts = pts[:-1]
    ends = pts[1:]
    mids = 0.5 * (starts + ends)
    lengths = np.linalg.norm(ends - starts, axis=1)
    m = lengths.size
    a = conductor.wire_radius
    k_e = 1.0 / (4.0 * np.pi * EPS0)

    p = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                ln = lengths[i]
                p[i, j] = k_e / ln * 2.0 * np.arcsinh(ln / (2.0 * a))
            else:
                d = np.linalg.norm(mids[i] - mids[j])
                p[i, j] = k_e / max(d, a)
    return p, lengths


def self_capacitance(conductor: Conductor) -> float:
    """Lumped self-capacitance (F) of the coil from an electrostatic energy method.

    Solves the thin-wire potential-coefficient system for the charge distribution under
    the physically-realistic linear winding potential (0 at one terminal rising to ``V0``
    at the other, as in a resonating coil) and returns ``C_eff = 2 W_elec / V0^2 = q^T V /
    V0^2``. This is the Medhurst-style lumped self-capacitance that, with the inductance,
    sets the self-resonant frequency.
    """

    p, lengths = _potential_coefficient_matrix(conductor)
    # Linear potential along the winding, sampled at segment midpoints (V0 = 1).
    cumlen = np.cumsum(lengths) - 0.5 * lengths
    v = cumlen / conductor.total_length
    # P q = V  ->  q = P^{-1} V ; energy W = 1/2 q^T V ; C_eff = 2W / V0^2, V0 = 1.
    q = np.linalg.solve(p, v)
    energy = 0.5 * float(q @ v)
    return float(2.0 * energy)


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


@dataclass(frozen=True)
class PEECCoilProperties:
    """Lumped RF properties of an arbitrary coil from the PEEC solve.

    The universal subset of
    :class:`spin_dynamics.fields.coil_properties.CoilProperties` (which is solenoid-specific),
    with the same field names so it drops into the same probe-noise / matching pipeline.
    """

    frequency: float
    inductance: float          # L(w) at frequency (H)
    ac_resistance: float       # R(w) at frequency (ohm)
    q_factor: float            # w L / R
    self_capacitance: float    # electrostatic self-capacitance (F)
    self_resonant_frequency: float  # 1/(2 pi sqrt(L_dc C)) (Hz)
    dc_inductance: float       # L(0) (H)
    dc_resistance: float       # R(0) (ohm)

    def tuning_capacitance(self, frequency: float | None = None) -> float:
        """Capacitance that resonates the coil at ``frequency`` (default: solve frequency)."""

        f = self.frequency if frequency is None else float(frequency)
        return float(1.0 / ((2.0 * np.pi * f) ** 2 * self.inductance))

    def to_probe_params(self) -> dict[str, float]:
        """Return ``{"L", "R", "C"}`` for the probe noise model."""

        return {"L": self.inductance, "R": self.ac_resistance, "C": self.tuning_capacitance()}


def coil_properties_peec(conductor: Conductor, frequency: float) -> PEECCoilProperties:
    """Extract lumped RF properties of an arbitrary coil at ``frequency`` via PEEC.

    Solves the chain impedance for ``L(w)``/``R(w)`` (skin + proximity), the electrostatic
    self-capacitance and the self-resonance, and packages them like
    :class:`spin_dynamics.fields.coil_properties.CoilProperties` so an arbitrary-geometry
    coil feeds the same SNR / matching workflow. See the module docstring for the
    resistance resolution ceiling at high ``a/delta``.
    """

    f = float(frequency)
    imp = extract_impedance(conductor, [f])
    c = self_capacitance(conductor)
    f_res = float(1.0 / (2.0 * np.pi * np.sqrt(imp.dc_inductance * c)))
    ind = float(imp.inductance[0])
    res = float(imp.resistance[0])
    q = float(2.0 * np.pi * f * ind / res) if res > 0 else float("inf")
    return PEECCoilProperties(
        frequency=f,
        inductance=ind,
        ac_resistance=res,
        q_factor=q,
        self_capacitance=float(c),
        self_resonant_frequency=f_res,
        dc_inductance=float(imp.dc_inductance),
        dc_resistance=float(imp.dc_resistance),
    )
