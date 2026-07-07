"""Nonlinear magnetostatic field solvers with soft-magnetic and permanent-magnet
materials.

The analytic solvers in :mod:`spin_dynamics.fields.magnetostatics` superpose
permanent-magnet dipoles and Biot-Savart coil filaments in free space -- they
cannot represent flux-shaping magnetic materials: high-permeability RF ferrites
that focus a ``B1`` field, or saturable iron that shapes and returns a ``B0``
field. This module adds a structured-grid nonlinear magnetostatic solver that
does.

The linear system at each nonlinear step is solved by a SciPy sparse LU
factorization when SciPy is available (fast, FEMM-style), falling back to a
NumPy-only matrix-free conjugate gradient otherwise; the nonlinear ``B``-``H``
loop is Anderson-accelerated Picard with residual-stall detection. No meshing
dependency (structured grid). Two coordinate systems share the material model,
the nonlinear driver, and the linear solve:

* :class:`PlanarMagnetostatics` -- 2D translationally-invariant ``(x, y)``
  problems via the out-of-plane vector potential ``A_z`` (``B`` in-plane). Used
  for transverse-field magnet arrays such as a diametric ring shimmed by soft
  iron.
* :class:`AxisymmetricMagnetostatics` -- rotationally-symmetric ``(r, z)``
  problems via the azimuthal vector potential ``A_phi``. Used for solenoid /
  loop coils with ferrite cores.

The saturable-material model (a smooth Brauer reluctivity ``nu(B^2)`` and its
Newton tangent ``dnu/dB^2``) and the nonlinear iteration scheme (relaxed Picard
and a magnetostatic Newton tangent) are adapted from the pure-Python FE backend
of the author's Motor_Design solver, re-expressed here on a structured grid with
a physical ``mu_r >= 1`` saturation clamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # SciPy gives a fast sparse direct solve; fall back to matrix-free CG.
    from scipy.sparse import coo_matrix as _coo_matrix
    from scipy.sparse.linalg import splu as _splu

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised only without SciPy installed
    _HAVE_SCIPY = False

MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# Material model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrauerBH:
    """Smooth Brauer soft-magnetic reluctivity ``nu(B^2) = bk1 + bk2 exp(bk3 B^2)``.

    ``nu`` is the reluctivity ``1/mu`` (SI). ``b2cap`` clamps the exponent so deep
    saturation stays bounded, and the evaluation additionally clamps ``nu`` to
    ``1/mu0`` so the relative permeability never drops below one (vacuum). At
    ``B = 0`` the small-signal permeability is ``mu_r0 = 1 / (mu0 (bk1 + bk2))``.
    """

    bk1: float
    bk2: float
    bk3: float
    b2cap: float = 8.0

    def nu_of_b2(self, b2: np.ndarray) -> np.ndarray:
        """Return reluctivity at ``B^2`` (clamped to a physical ``mu_r >= 1``)."""

        b2 = np.asarray(b2, dtype=np.float64)
        nu = self.bk1 + self.bk2 * np.exp(self.bk3 * np.minimum(b2, self.b2cap))
        return np.minimum(nu, 1.0 / MU0)

    def dnu_db2(self, b2: np.ndarray) -> np.ndarray:
        """Return ``d(nu)/d(B^2)`` (zero where ``nu`` is clamped)."""

        b2 = np.asarray(b2, dtype=np.float64)
        raw = self.bk1 + self.bk2 * np.exp(self.bk3 * np.minimum(b2, self.b2cap))
        d = self.bk2 * self.bk3 * np.exp(self.bk3 * np.minimum(b2, self.b2cap))
        return np.where((b2 >= self.b2cap) | (raw >= 1.0 / MU0), 0.0, d)

    @property
    def mu_r0(self) -> float:
        """Small-signal relative permeability at ``B = 0``."""

        return 1.0 / (MU0 * (self.bk1 + self.bk2))


# Brauer curves (bk1 + bk2 ~ 1/(mu0 mu_r0); bk3 sets the knee). The mu_r >= 1
# clamp in BrauerBH.nu_of_b2 provides the vacuum asymptote at deep saturation.
BRAUER_CURVES: dict[str, BrauerBH] = {
    # mu_r0 ~ 5000, knee ~ 1.5 T -- generic soft iron / low-carbon steel. b2cap
    # is set so the mu_r >= 1 clamp (not the exponent cap) governs deep
    # saturation: nu -> 1/mu0 above ~2.9 T.
    "soft_iron": BrauerBH(bk1=10.0, bk2=149.0, bk3=1.0, b2cap=12.0),
    # mu_r0 ~ 4000, knee ~ 1.0 T -- NiFe / permalloy (Motor_Design 'nife').
    "nife": BrauerBH(bk1=100.0, bk2=99.0, bk3=2.95, b2cap=4.0),
    # mu_r0 ~ 4000, knee ~ 2.3 T -- CoFe / Hiperco (Motor_Design 'cofe').
    "cofe": BrauerBH(bk1=100.0, bk2=99.0, bk3=0.55, b2cap=6.5),
}


@dataclass(frozen=True)
class MagneticMaterial:
    """A magnetic material region: linear or saturable, optionally a magnet.

    ``mu_r`` is the (linear) relative permeability, used unless ``bh`` is given.
    ``bh`` supplies a nonlinear ``B``-``H`` curve (its ``mu_r0`` replaces ``mu_r``
    as the small-signal value). ``remanence_t`` is the permanent-magnet remanent
    flux density ``|B_r|`` (0 for a non-magnet); the direction is set per region
    when the material is painted onto the grid.
    """

    name: str
    mu_r: float = 1.0
    bh: BrauerBH | None = None
    remanence_t: float = 0.0

    def __post_init__(self) -> None:
        if self.mu_r <= 0.0 or not np.isfinite(self.mu_r):
            raise ValueError("mu_r must be positive and finite")
        if self.remanence_t < 0.0 or not np.isfinite(self.remanence_t):
            raise ValueError("remanence_t must be non-negative and finite")


def air() -> MagneticMaterial:
    """Return free space (``mu_r = 1``)."""

    return MagneticMaterial("air", mu_r=1.0)


def linear_material(mu_r: float, *, name: str = "linear") -> MagneticMaterial:
    """Return a linear material of relative permeability ``mu_r``."""

    return MagneticMaterial(name, mu_r=float(mu_r))


def rf_ferrite(mu_r: float = 1000.0, *, bh: BrauerBH | None = None) -> MagneticMaterial:
    """Return a high-permeability RF ferrite (linear by default, below saturation)."""

    return MagneticMaterial("ferrite", mu_r=float(mu_r), bh=bh)


def soft_iron(curve: str = "soft_iron") -> MagneticMaterial:
    """Return a saturable soft-iron material from a named Brauer curve."""

    if curve not in BRAUER_CURVES:
        raise KeyError(f"unknown B-H curve {curve!r}; choose from {list(BRAUER_CURVES)}")
    bh = BRAUER_CURVES[curve]
    return MagneticMaterial(f"soft_iron:{curve}", mu_r=bh.mu_r0, bh=bh)


def ndfeb(remanence_t: float = 1.3, *, mu_rec: float = 1.05) -> MagneticMaterial:
    """Return an NdFeB permanent magnet (recoil permeability ``mu_rec``)."""

    return MagneticMaterial("ndfeb", mu_r=float(mu_rec), remanence_t=float(remanence_t))


# ---------------------------------------------------------------------------
# Matrix-free preconditioned conjugate gradient (SPD systems)
# ---------------------------------------------------------------------------


def _conjugate_gradient(apply, rhs, diagonal, free, *, x0=None, tol=1e-8, max_iter=5000):
    """Solve ``apply(x) = rhs`` on ``free`` nodes with Jacobi-preconditioned CG.

    ``apply`` returns the operator action on the full grid (zero on non-free
    nodes); ``diagonal`` is the operator diagonal for Jacobi preconditioning.
    Both the operator and right-hand side are symmetric positive definite on the
    free (interior) set, so CG converges. Passing ``x0`` (the previous nonlinear
    iterate) warm-starts the solve, so successive Picard solves converge in a
    handful of iterations instead of from scratch.
    """

    x = np.zeros_like(rhs) if x0 is None else x0.copy()
    x[~free] = 0.0
    minv = np.zeros_like(diagonal)
    minv[free] = 1.0 / diagonal[free]
    r = rhs - apply(x)
    r[~free] = 0.0
    rhs_norm = float(np.sqrt(np.sum(rhs[free] ** 2))) or 1.0
    if np.sqrt(np.sum(r[free] ** 2)) <= tol * rhs_norm:
        return x
    z = minv * r
    p = z.copy()
    rz = float(np.sum(r * z))
    for _ in range(int(max_iter)):
        ap = apply(p)
        ap[~free] = 0.0
        pap = float(np.sum(p * ap))
        if pap <= 0.0:
            break
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        if np.sqrt(np.sum(r[free] ** 2)) <= tol * rhs_norm:
            break
        z = minv * r
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _sparse_factorize(diag, c_ip, c_im, c_jp, c_jm):
    """Factorize the 5-point stencil sparse matrix (boundary rows are identity).

    ``diag`` and the four directional coupling arrays are shaped like the grid;
    boundary rows must already be set to identity (``diag = 1`` and couplings
    zero). Returns ``linsolve(rhs, x0=None)`` using a sparse LU factorization.
    """

    n0, n1 = diag.shape
    total = n0 * n1
    idx = np.arange(total).reshape(n0, n1)
    rows = [idx.ravel(), idx[:-1, :].ravel(), idx[1:, :].ravel(),
            idx[:, :-1].ravel(), idx[:, 1:].ravel()]
    cols = [idx.ravel(), idx[1:, :].ravel(), idx[:-1, :].ravel(),
            idx[:, 1:].ravel(), idx[:, :-1].ravel()]
    data = [diag.ravel(), -c_ip[:-1, :].ravel(), -c_im[1:, :].ravel(),
            -c_jp[:, :-1].ravel(), -c_jm[:, 1:].ravel()]
    matrix = _coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total, total),
    ).tocsc()
    lu = _splu(matrix)

    def linsolve(rhs, x0=None):
        return lu.solve(rhs.ravel()).reshape(rhs.shape)

    return linsolve


def _sparse_factorize_3d(diag, c_ip, c_im, c_jp, c_jm, c_kp, c_km):
    """Factorize the 7-point stencil sparse matrix (boundary rows are identity).

    The 3D analog of :func:`_sparse_factorize`: ``diag`` and the six directional
    coupling arrays are shaped like the ``(nx, ny, nz)`` grid, with boundary rows
    already set to identity. Returns ``linsolve(rhs, x0=None)`` using a sparse LU
    factorization. Fill-in is far heavier than in 2D (~O(N^{4/3}) memory); the
    practical grid ceiling is ~50-64^3 -- see docs/reduced_scalar_potential_3d.md.
    """

    n0, n1, n2 = diag.shape
    total = n0 * n1 * n2
    idx = np.arange(total).reshape(n0, n1, n2)
    rows = [
        idx.ravel(),
        idx[:-1, :, :].ravel(), idx[1:, :, :].ravel(),
        idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel(),
        idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel(),
    ]
    cols = [
        idx.ravel(),
        idx[1:, :, :].ravel(), idx[:-1, :, :].ravel(),
        idx[:, 1:, :].ravel(), idx[:, :-1, :].ravel(),
        idx[:, :, 1:].ravel(), idx[:, :, :-1].ravel(),
    ]
    data = [
        diag.ravel(),
        -c_ip[:-1, :, :].ravel(), -c_im[1:, :, :].ravel(),
        -c_jp[:, :-1, :].ravel(), -c_jm[:, 1:, :].ravel(),
        -c_kp[:, :, :-1].ravel(), -c_km[:, :, 1:].ravel(),
    ]
    matrix = _coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total, total),
    ).tocsc()
    lu = _splu(matrix)

    def linsolve(rhs, x0=None):
        return lu.solve(rhs.ravel()).reshape(rhs.shape)

    return linsolve


def _anderson_picard(
    coeff_fn,
    make_linsolve,
    rhs_full,
    free,
    *,
    linear,
    relax,
    tol,
    max_iter,
    depth,
    n_load_steps,
    cg_tol,
    cg_max_iter,
    rhs_fn=None,
):
    """Solve a nonlinear magnetostatic problem by Anderson-accelerated Picard.

    ``coeff_fn(a)`` returns the per-node operator coefficient for potential ``a``
    (reluctivity ``nu`` for the vector-potential solvers, permeability ``mu`` for
    the reduced-scalar-potential solver) and ``make_linsolve(coeff, ...)`` returns
    a linear solver for the current operator. Anderson mixing of depth ``depth``
    (the technique the Motor_Design solver uses at the re-entrant iron corner)
    collapses the iteration count that plain relaxed Picard needs;
    ``n_load_steps > 1`` ramps the sources for hard-saturation robustness.

    ``rhs_fn`` is an optional ``rhs_fn(coeff) -> rhs`` used when the source term
    itself depends on the (evolving) coefficient -- the reduced scalar potential
    needs this because its right-hand side ``div(mu H_s + B_r)`` carries ``mu``.
    When ``rhs_fn is None`` the fixed ``rhs_full`` is used and the iteration is
    identical to the vector-potential path.
    """

    a = np.zeros_like(rhs_full)
    if linear:
        nu = coeff_fn(a)
        rhs0 = rhs_full if rhs_fn is None else rhs_fn(nu)
        a = make_linsolve(nu, free, cg_tol, cg_max_iter)(rhs0, a)
        a[~free] = 0.0
        return a, 1, 0.0

    # Staircased curved boundaries create re-entrant corners whose algebraic
    # residual plateaus above tol while the field itself is fully converged. So
    # stop on tol OR a residual stall, and return the best iterate seen.
    patience = depth + 1
    total = 0
    best_resid, best_a = np.inf, a.copy()
    loads = list(np.linspace(1.0 / n_load_steps, 1.0, int(n_load_steps)))
    for step, load in enumerate(loads):
        rhs = None if rhs_fn is not None else load * rhs_full
        is_final = step == len(loads) - 1
        g_hist: list[np.ndarray] = []
        f_hist: list[np.ndarray] = []
        local_best, stall = np.inf, 0
        for _ in range(int(max_iter)):
            total += 1
            nu = coeff_fn(a)
            if rhs_fn is not None:  # source depends on the evolving coefficient
                rhs = load * rhs_fn(nu)
            solved = make_linsolve(nu, free, cg_tol, cg_max_iter)(rhs, a)
            g = (1.0 - relax) * a + relax * solved
            fk = g - a
            g_hist.append(g)
            f_hist.append(fk)
            if len(f_hist) > depth + 1:
                g_hist.pop(0)
                f_hist.pop(0)
            if len(f_hist) >= 2:
                df = np.column_stack(
                    [(f_hist[j + 1] - f_hist[j]).ravel() for j in range(len(f_hist) - 1)]
                )
                dg = np.column_stack(
                    [(g_hist[j + 1] - g_hist[j]).ravel() for j in range(len(g_hist) - 1)]
                )
                gamma, *_ = np.linalg.lstsq(df, fk.ravel(), rcond=1e-12)
                a_new = g - (dg @ gamma).reshape(g.shape)
            else:
                a_new = g
            last = float(np.linalg.norm(fk)) / (float(np.linalg.norm(a_new)) or 1.0)
            a = a_new
            # Reset the stall counter only on a meaningful (>=2%) improvement, so a
            # slow residual crawl toward a corner plateau stops promptly; best_a
            # still tracks the true best iterate.
            stall = 0 if last < 0.98 * local_best else stall + 1
            if last < local_best - 1e-12:
                local_best = last
                if is_final:
                    best_resid, best_a = last, a.copy()
            if last < tol:
                if is_final:
                    best_resid, best_a = last, a.copy()
                break
            if stall >= patience:
                break
    best_a[~free] = 0.0
    return best_a, total, best_resid


# ---------------------------------------------------------------------------
# Planar (A_z) solver
# ---------------------------------------------------------------------------


@dataclass
class PlanarSolution:
    """Result of a planar magnetostatic solve."""

    x_m: np.ndarray
    y_m: np.ndarray
    a_z: np.ndarray
    b_x: np.ndarray
    b_y: np.ndarray
    iterations: int
    residual: float

    @property
    def b_magnitude(self) -> np.ndarray:
        """Return ``|B|`` at every grid node."""

        return np.hypot(self.b_x, self.b_y)

    def sample(self, x_m, y_m) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(Bx, By)`` bilinearly interpolated at query points."""

        xi = np.interp(np.asarray(x_m, dtype=np.float64), self.x_m, np.arange(self.x_m.size))
        yi = np.interp(np.asarray(y_m, dtype=np.float64), self.y_m, np.arange(self.y_m.size))
        return _bilinear(self.b_x, xi, yi), _bilinear(self.b_y, xi, yi)

    def homogeneity_ppm(self, *, center=(0.0, 0.0), radius_m: float) -> float:
        """Return peak-to-peak ``|B|`` inhomogeneity (ppm) over a disk of interest."""

        gx, gy = np.meshgrid(self.x_m, self.y_m, indexing="ij")
        mask = (gx - center[0]) ** 2 + (gy - center[1]) ** 2 <= radius_m**2
        if not np.any(mask):
            raise ValueError("no grid nodes inside the region of interest")
        mag = self.b_magnitude[mask]
        mean = float(np.mean(mag))
        if mean == 0.0:
            return np.inf
        return 1.0e6 * float(np.ptp(mag)) / mean


class PlanarMagnetostatics:
    """2D translationally-invariant nonlinear magnetostatics via ``A_z``.

    Build a problem on a uniform ``(x, y)`` node grid, paint material regions
    (linear, saturable, or permanent magnet) and coil currents, then
    :meth:`solve`. Boundaries are Dirichlet ``A_z = 0`` (place them far enough
    that the flux has decayed).
    """

    def __init__(self, x_m: Sequence[float] | np.ndarray, y_m: Sequence[float] | np.ndarray):
        self.x = np.asarray(x_m, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y_m, dtype=np.float64).reshape(-1)
        if self.x.size < 3 or self.y.size < 3:
            raise ValueError("grid must have at least 3 nodes per axis")
        self.hx = float(self.x[1] - self.x[0])
        self.hy = float(self.y[1] - self.y[0])
        if not (np.allclose(np.diff(self.x), self.hx) and np.allclose(np.diff(self.y), self.hy)):
            raise ValueError("grid must be uniformly spaced on each axis")
        shape = (self.x.size, self.y.size)
        self.mu_r = np.ones(shape, dtype=np.float64)
        self.nl = np.zeros(shape, dtype=bool)
        self._bk1 = np.zeros(shape)
        self._bk2 = np.zeros(shape)
        self._bk3 = np.zeros(shape)
        self._b2cap = np.ones(shape)
        self.jz = np.zeros(shape, dtype=np.float64)
        self.hcx = np.zeros(shape, dtype=np.float64)
        self.hcy = np.zeros(shape, dtype=np.float64)
        self._gx, self._gy = np.meshgrid(self.x, self.y, indexing="ij")

    # -- region masks -------------------------------------------------------
    def rectangle(self, x_limits, y_limits) -> np.ndarray:
        """Return a node mask for an axis-aligned rectangle."""

        (x0, x1), (y0, y1) = x_limits, y_limits
        return (
            (self._gx >= min(x0, x1))
            & (self._gx <= max(x0, x1))
            & (self._gy >= min(y0, y1))
            & (self._gy <= max(y0, y1))
        )

    def disk(self, center, radius_m) -> np.ndarray:
        """Return a node mask for a filled disk."""

        return (self._gx - center[0]) ** 2 + (self._gy - center[1]) ** 2 <= radius_m**2

    def annulus(self, center, r_inner_m, r_outer_m) -> np.ndarray:
        """Return a node mask for an annulus."""

        rr = (self._gx - center[0]) ** 2 + (self._gy - center[1]) ** 2
        return (rr >= r_inner_m**2) & (rr <= r_outer_m**2)

    # -- painting -----------------------------------------------------------
    def add_material(
        self,
        mask: np.ndarray,
        material: MagneticMaterial,
        *,
        remanence_angle_deg: float = 0.0,
        current_total_a: float | None = None,
    ) -> None:
        """Paint ``material`` (and optional coil current) onto masked nodes.

        ``remanence_angle_deg`` orients a permanent magnet's remanence in the
        ``x``-``y`` plane (0 = ``+x``). ``current_total_a`` distributes a total
        z-directed current uniformly over the region as a current density.
        """

        mask = np.asarray(mask, dtype=bool)
        if material.bh is not None:
            self.nl[mask] = True
            self._bk1[mask] = material.bh.bk1
            self._bk2[mask] = material.bh.bk2
            self._bk3[mask] = material.bh.bk3
            self._b2cap[mask] = material.bh.b2cap
            self.mu_r[mask] = material.bh.mu_r0
        else:
            self.nl[mask] = False
            self.mu_r[mask] = material.mu_r
        if material.remanence_t > 0.0:
            angle = np.deg2rad(remanence_angle_deg)
            hc = material.remanence_t / MU0
            self.hcx[mask] = hc * np.cos(angle)
            self.hcy[mask] = hc * np.sin(angle)
        if current_total_a is not None:
            area = float(np.count_nonzero(mask)) * self.hx * self.hy
            if area <= 0.0:
                raise ValueError("current region is empty")
            self.jz[mask] = float(current_total_a) / area

    # -- physics ------------------------------------------------------------
    def _b_field(self, a_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # B = curl(A_z z) = (dA/dy, -dA/dx)
        b_x = np.gradient(a_z, self.y, axis=1)
        b_y = -np.gradient(a_z, self.x, axis=0)
        return b_x, b_y

    def _reluctivity(self, a_z: np.ndarray) -> np.ndarray:
        nu_lin = 1.0 / (MU0 * self.mu_r)
        if not self.nl.any():
            return nu_lin
        b_x, b_y = self._b_field(a_z)
        b2 = b_x**2 + b_y**2
        nu_nl = self._bk1 + self._bk2 * np.exp(
            self._bk3 * np.minimum(b2, self._b2cap)
        )
        nu_nl = np.minimum(nu_nl, 1.0 / MU0)
        return np.where(self.nl, nu_nl, nu_lin)

    def _operator(self, nu: np.ndarray):
        nux = 0.5 * (nu[1:, :] + nu[:-1, :])  # x-faces
        nuy = 0.5 * (nu[:, 1:] + nu[:, :-1])  # y-faces
        hx2, hy2 = self.hx**2, self.hy**2

        def apply(a_z: np.ndarray) -> np.ndarray:
            out = np.zeros_like(a_z)
            fx = nux * (a_z[1:, :] - a_z[:-1, :])
            fy = nuy * (a_z[:, 1:] - a_z[:, :-1])
            out[1:-1, 1:-1] = -(
                (fx[1:, 1:-1] - fx[:-1, 1:-1]) / hx2
                + (fy[1:-1, 1:] - fy[1:-1, :-1]) / hy2
            )
            return out

        diag = np.ones_like(nu)
        diag[1:-1, 1:-1] = (nux[1:, 1:-1] + nux[:-1, 1:-1]) / hx2 + (
            nuy[1:-1, 1:] + nuy[1:-1, :-1]
        ) / hy2
        return apply, diag

    def _sparse_coeffs(self, nu: np.ndarray):
        hx2, hy2 = self.hx**2, self.hy**2
        nux = 0.5 * (nu[1:, :] + nu[:-1, :])
        nuy = 0.5 * (nu[:, 1:] + nu[:, :-1])
        c_ip = np.zeros_like(nu)
        c_ip[:-1, :] = nux / hx2
        c_im = np.zeros_like(nu)
        c_im[1:, :] = nux / hx2
        c_jp = np.zeros_like(nu)
        c_jp[:, :-1] = nuy / hy2
        c_jm = np.zeros_like(nu)
        c_jm[:, 1:] = nuy / hy2
        diag = c_ip + c_im + c_jp + c_jm
        b = np.zeros(nu.shape, dtype=bool)
        b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
        diag[b] = 1.0
        for arr in (c_ip, c_im, c_jp, c_jm):
            arr[b] = 0.0
        return diag, c_ip, c_im, c_jp, c_jm

    def _make_linsolve(self, nu, free, cg_tol, cg_max_iter):
        if _HAVE_SCIPY:
            return _sparse_factorize(*self._sparse_coeffs(nu))
        apply, diag = self._operator(nu)

        def linsolve(rhs, x0=None):
            return _conjugate_gradient(
                apply, rhs, diag, free, x0=x0, tol=cg_tol, max_iter=cg_max_iter
            )

        return linsolve

    def _rhs(self) -> np.ndarray:
        # -div(nu grad A) = J_z + curl(H_c) ; H_c = B_r / mu0 (bound-current model)
        pm = np.gradient(self.hcy, self.x, axis=0) - np.gradient(self.hcx, self.y, axis=1)
        rhs = self.jz + pm
        rhs[0, :] = rhs[-1, :] = rhs[:, 0] = rhs[:, -1] = 0.0
        return rhs

    def solve(
        self,
        *,
        max_iter: int = 60,
        tol: float = 1e-6,
        relax: float = 0.7,
        depth: int = 5,
        n_load_steps: int = 1,
        cg_tol: float = 1e-9,
        cg_max_iter: int = 20000,
    ) -> PlanarSolution:
        """Solve the (possibly nonlinear) problem and return the fields.

        Nonlinear problems use Anderson-accelerated Picard with warm-started
        conjugate-gradient linear solves; ``depth`` sets the Anderson history and
        ``n_load_steps > 1`` ramps the sources for hard-saturation robustness.
        """

        free = np.ones((self.x.size, self.y.size), dtype=bool)
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        a_z, total, last = _anderson_picard(
            self._reluctivity,
            self._make_linsolve,
            self._rhs(),
            free,
            linear=not self.nl.any(),
            relax=relax,
            tol=tol,
            max_iter=max_iter,
            depth=depth,
            n_load_steps=n_load_steps,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
        )
        b_x, b_y = self._b_field(a_z)
        return PlanarSolution(
            x_m=self.x,
            y_m=self.y,
            a_z=a_z,
            b_x=b_x,
            b_y=b_y,
            iterations=total,
            residual=last,
        )

    def stored_energy_per_length(self, solution: PlanarSolution) -> float:
        """Return magnetic energy per unit z-length ``integral (1/2) nu B^2 dA``."""

        nu = self._reluctivity(solution.a_z)
        b2 = solution.b_x**2 + solution.b_y**2
        return float(np.sum(0.5 * nu * b2) * self.hx * self.hy)


@dataclass
class AxisymmetricSolution:
    """Result of an axisymmetric magnetostatic solve."""

    r_m: np.ndarray
    z_m: np.ndarray
    a_phi: np.ndarray
    b_r: np.ndarray
    b_z: np.ndarray
    iterations: int
    residual: float

    @property
    def b_magnitude(self) -> np.ndarray:
        """Return ``|B|`` at every grid node."""

        return np.hypot(self.b_r, self.b_z)

    def on_axis_bz(self) -> np.ndarray:
        """Return the axial field ``B_z(0, z)`` on the symmetry axis."""

        return self.b_z[0, :]

    def sample_bz(self, r_m, z_m) -> np.ndarray:
        """Return ``B_z`` bilinearly interpolated at query points."""

        ri = np.interp(np.asarray(r_m, dtype=np.float64), self.r_m, np.arange(self.r_m.size))
        zi = np.interp(np.asarray(z_m, dtype=np.float64), self.z_m, np.arange(self.z_m.size))
        return _bilinear(self.b_z, ri, zi)


class AxisymmetricMagnetostatics:
    """Rotationally-symmetric nonlinear magnetostatics via ``A_phi``.

    Build a problem on a uniform ``(r, z)`` node grid (``r`` starting at 0), paint
    material regions and azimuthal coil currents, then :meth:`solve`. The
    symmetry axis carries ``A_phi = 0``; the outer ``r`` and both ``z`` boundaries
    are Dirichlet ``A_phi = 0`` (place them far enough that the flux has decayed).
    The equation is solved in its ``r``-weighted symmetric form so the same
    conjugate-gradient driver applies.
    """

    def __init__(self, r_m: Sequence[float] | np.ndarray, z_m: Sequence[float] | np.ndarray):
        self.r = np.asarray(r_m, dtype=np.float64).reshape(-1)
        self.z = np.asarray(z_m, dtype=np.float64).reshape(-1)
        if self.r.size < 3 or self.z.size < 3:
            raise ValueError("grid must have at least 3 nodes per axis")
        if self.r[0] < 0.0:
            raise ValueError("r grid must start at r >= 0")
        self.hr = float(self.r[1] - self.r[0])
        self.hz = float(self.z[1] - self.z[0])
        if not (np.allclose(np.diff(self.r), self.hr) and np.allclose(np.diff(self.z), self.hz)):
            raise ValueError("grid must be uniformly spaced on each axis")
        shape = (self.r.size, self.z.size)
        self.mu_r = np.ones(shape, dtype=np.float64)
        self.nl = np.zeros(shape, dtype=bool)
        self._bk1 = np.zeros(shape)
        self._bk2 = np.zeros(shape)
        self._bk3 = np.zeros(shape)
        self._b2cap = np.ones(shape)
        self.jphi = np.zeros(shape, dtype=np.float64)
        self.hcr = np.zeros(shape, dtype=np.float64)
        self.hcz = np.zeros(shape, dtype=np.float64)
        self._gr, self._gz = np.meshgrid(self.r, self.z, indexing="ij")

    # -- region masks -------------------------------------------------------
    def rectangle(self, r_limits, z_limits) -> np.ndarray:
        """Return a node mask for an axis-aligned ``(r, z)`` rectangle."""

        (r0, r1), (z0, z1) = r_limits, z_limits
        return (
            (self._gr >= min(r0, r1))
            & (self._gr <= max(r0, r1))
            & (self._gz >= min(z0, z1))
            & (self._gz <= max(z0, z1))
        )

    # -- painting -----------------------------------------------------------
    def add_material(
        self,
        mask: np.ndarray,
        material: MagneticMaterial,
        *,
        remanence_axis: str = "z",
        current_total_a: float | None = None,
    ) -> None:
        """Paint ``material`` (and optional azimuthal coil current) onto nodes.

        ``current_total_a`` is the total azimuthal current threading the region's
        ``(r, z)`` cross-section (turns * amps), spread uniformly as ``J_phi``.
        ``remanence_axis`` (``"z"`` or ``"r"``) orients a permanent magnet.
        """

        mask = np.asarray(mask, dtype=bool)
        if material.bh is not None:
            self.nl[mask] = True
            self._bk1[mask] = material.bh.bk1
            self._bk2[mask] = material.bh.bk2
            self._bk3[mask] = material.bh.bk3
            self._b2cap[mask] = material.bh.b2cap
            self.mu_r[mask] = material.bh.mu_r0
        else:
            self.nl[mask] = False
            self.mu_r[mask] = material.mu_r
        if material.remanence_t > 0.0:
            hc = material.remanence_t / MU0
            if remanence_axis == "z":
                self.hcz[mask] = hc
            elif remanence_axis == "r":
                self.hcr[mask] = hc
            else:
                raise ValueError("remanence_axis must be 'r' or 'z'")
        if current_total_a is not None:
            area = float(np.count_nonzero(mask)) * self.hr * self.hz
            if area <= 0.0:
                raise ValueError("current region is empty")
            self.jphi[mask] = float(current_total_a) / area

    # -- physics ------------------------------------------------------------
    def _b_field(self, a_phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # B_r = -dA/dz ; B_z = (1/r) d(r A)/dr  (with the on-axis limit 2 dA/dr).
        b_r = -np.gradient(a_phi, self.z, axis=1)
        r_col = self.r[:, None]
        r_a = r_col * a_phi
        b_z = np.empty_like(a_phi)
        b_z[1:, :] = np.gradient(r_a, self.r, axis=0)[1:, :] / r_col[1:, :]
        b_z[0, :] = 2.0 * (a_phi[1, :] - a_phi[0, :]) / self.hr
        return b_r, b_z

    def _reluctivity(self, a_phi: np.ndarray) -> np.ndarray:
        nu_lin = 1.0 / (MU0 * self.mu_r)
        if not self.nl.any():
            return nu_lin
        b_r, b_z = self._b_field(a_phi)
        b2 = b_r**2 + b_z**2
        nu_nl = self._bk1 + self._bk2 * np.exp(
            self._bk3 * np.minimum(b2, self._b2cap)
        )
        nu_nl = np.minimum(nu_nl, 1.0 / MU0)
        return np.where(self.nl, nu_nl, nu_lin)

    def _operator(self, nu: np.ndarray):
        r_col = self.r[:, None]
        rface = (0.5 * (self.r[1:] + self.r[:-1]))[:, None]
        nur = 0.5 * (nu[1:, :] + nu[:-1, :])
        nuz = 0.5 * (nu[:, 1:] + nu[:, :-1])
        cr = rface * nur  # r-weighted radial face conductance
        hr2, hz2 = self.hr**2, self.hz**2

        def apply(a_phi: np.ndarray) -> np.ndarray:
            out = np.zeros_like(a_phi)
            fr = cr * (a_phi[1:, :] - a_phi[:-1, :])
            fz = nuz * (a_phi[:, 1:] - a_phi[:, :-1])
            radial = (fr[1:, 1:-1] - fr[:-1, 1:-1]) / hr2
            axial = r_col[1:-1, :] * (fz[1:-1, 1:] - fz[1:-1, :-1]) / hz2
            out[1:-1, 1:-1] = (
                -(radial + axial)
                + nu[1:-1, 1:-1] * a_phi[1:-1, 1:-1] / r_col[1:-1, :]
            )
            return out

        diag = np.ones_like(nu)
        diag[1:-1, 1:-1] = (
            (cr[1:, 1:-1] + cr[:-1, 1:-1]) / hr2
            + r_col[1:-1, :] * (nuz[1:-1, 1:] + nuz[1:-1, :-1]) / hz2
            + nu[1:-1, 1:-1] / r_col[1:-1, :]
        )
        return apply, diag

    def _sparse_coeffs(self, nu: np.ndarray):
        hr2, hz2 = self.hr**2, self.hz**2
        r_col = self.r[:, None]
        rface = (0.5 * (self.r[1:] + self.r[:-1]))[:, None]
        nur = 0.5 * (nu[1:, :] + nu[:-1, :])
        nuz = 0.5 * (nu[:, 1:] + nu[:, :-1])
        cr = rface * nur
        c_ip = np.zeros_like(nu)
        c_ip[:-1, :] = cr / hr2
        c_im = np.zeros_like(nu)
        c_im[1:, :] = cr / hr2
        c_jp = np.zeros_like(nu)
        c_jp[:, :-1] = r_col * nuz / hz2
        c_jm = np.zeros_like(nu)
        c_jm[:, 1:] = r_col * nuz / hz2
        reaction = np.zeros_like(nu)
        reaction[1:, :] = nu[1:, :] / r_col[1:, :]
        diag = c_ip + c_im + c_jp + c_jm + reaction
        b = np.zeros(nu.shape, dtype=bool)
        b[0, :] = b[-1, :] = b[:, 0] = b[:, -1] = True
        diag[b] = 1.0
        for arr in (c_ip, c_im, c_jp, c_jm):
            arr[b] = 0.0
        return diag, c_ip, c_im, c_jp, c_jm

    def _make_linsolve(self, nu, free, cg_tol, cg_max_iter):
        if _HAVE_SCIPY:
            return _sparse_factorize(*self._sparse_coeffs(nu))
        apply, diag = self._operator(nu)

        def linsolve(rhs, x0=None):
            return _conjugate_gradient(
                apply, rhs, diag, free, x0=x0, tol=cg_tol, max_iter=cg_max_iter
            )

        return linsolve

    def _rhs(self) -> np.ndarray:
        # curl(H_c)_phi = dHc_r/dz - dHc_z/dr ; weighted by r for the symmetric form
        pm = np.gradient(self.hcr, self.z, axis=1) - np.gradient(self.hcz, self.r, axis=0)
        rhs = self.r[:, None] * (self.jphi + pm)
        rhs[0, :] = rhs[-1, :] = rhs[:, 0] = rhs[:, -1] = 0.0
        return rhs

    def solve(
        self,
        *,
        max_iter: int = 60,
        tol: float = 1e-6,
        relax: float = 0.7,
        depth: int = 5,
        n_load_steps: int = 1,
        cg_tol: float = 1e-9,
        cg_max_iter: int = 20000,
    ) -> AxisymmetricSolution:
        """Solve the (possibly nonlinear) problem and return the fields.

        Nonlinear problems use Anderson-accelerated Picard with warm-started
        conjugate-gradient linear solves.
        """

        free = np.ones((self.r.size, self.z.size), dtype=bool)
        free[0, :] = free[-1, :] = free[:, 0] = free[:, -1] = False
        a_phi, total, last = _anderson_picard(
            self._reluctivity,
            self._make_linsolve,
            self._rhs(),
            free,
            linear=not self.nl.any(),
            relax=relax,
            tol=tol,
            max_iter=max_iter,
            depth=depth,
            n_load_steps=n_load_steps,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
        )
        b_r, b_z = self._b_field(a_phi)
        return AxisymmetricSolution(
            r_m=self.r,
            z_m=self.z,
            a_phi=a_phi,
            b_r=b_r,
            b_z=b_z,
            iterations=total,
            residual=last,
        )

    def stored_energy(self, solution: AxisymmetricSolution) -> float:
        """Return total magnetic energy ``integral (1/2) nu B^2 dV`` (joules)."""

        nu = self._reluctivity(solution.a_phi)
        b2 = solution.b_r**2 + solution.b_z**2
        return float(
            np.sum(2.0 * np.pi * self.r[:, None] * 0.5 * nu * b2) * self.hr * self.hz
        )


def _bilinear(grid: np.ndarray, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    xi = np.clip(xi, 0, grid.shape[0] - 1.000001)
    yi = np.clip(yi, 0, grid.shape[1] - 1.000001)
    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    fx = xi - x0
    fy = yi - y0
    return (
        grid[x0, y0] * (1 - fx) * (1 - fy)
        + grid[x0 + 1, y0] * fx * (1 - fy)
        + grid[x0, y0 + 1] * (1 - fx) * fy
        + grid[x0 + 1, y0 + 1] * fx * fy
    )
