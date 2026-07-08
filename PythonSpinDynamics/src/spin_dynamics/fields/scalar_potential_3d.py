"""3D nonlinear magnetostatics via the reduced scalar potential (RSP).

The vector-potential solvers in :mod:`spin_dynamics.fields.nonlinear_magnetostatics`
reduce ``A`` to a single scalar only because of 2D/axisymmetric symmetry. Full 3D
has no such symmetry and the curl-curl operator ``curl(nu curl A) = J`` is
singular on a nodal grid (large gradient null space), which forces edge elements
or a gauged block system -- a different solver.

This module takes the pragmatic route flagged in
``docs/reduced_scalar_potential_3d.md``: the **reduced scalar potential**. Split
the field as ``H = H_s - grad(psi)`` where ``H_s`` is the Biot-Savart source field
of the coil currents in free space (so Ampere's law holds for any ``psi``). Gauss'
law ``div(B) = 0`` with ``B = mu(|B|) (H_s - grad psi) + B_r`` then gives a single
scalar, variable-coefficient, SPD Poisson problem::

    div(mu grad psi) = div(mu H_s + B_r) ,

discretized on the same structured grid with the same 7-point (3D analog of the
5-point) stencil, the same Anderson-accelerated Picard driver, and the same
sparse-LU / matrix-free-CG linear solve as the 2D solvers. Permanent magnets and
soft-magnetic flux shaping (shim iron, magnet arrays, ferrite cores) are the
intended use.

RSP has known limits -- most importantly a cancellation error inside
high-permeability iron that grows as ``mu_r (h/L)^2`` (trustworthy for
``mu_r <~ 100-300``), quantified in ``docs/reduced_scalar_potential_3d.md``. The
classical Simkin-Trowbridge total/reduced split does *not* help on this centered
nodal-FV grid -- the two formulations are related by an exact linear variable
shift and give the identical discrete solution (verified to machine precision), so
the error is discretization, not float64 cancellation. The effective lever is grid
refinement, made practical by the optional AMG solver (``solve(linear_solver=...)``,
:func:`_amg_linsolve_3d`), which has grid-independent iteration counts and scales
past the ~50-64^3 sparse-LU fill-in ceiling to >10^6 unknowns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import biot_savart
from spin_dynamics.fields.nonlinear_magnetostatics import (
    MU0,
    MagneticMaterial,
    _amg_linsolve_3d,
    _anderson_picard,
    _conjugate_gradient,
    _HAVE_PYAMG,
    _HAVE_SCIPY,
    _sparse_factorize_3d,
)

__all__ = ["ReducedScalarPotential3D", "ScalarPotentialSolution"]

# Crossover above which AMG's O(N) solve beats sparse LU in wall-clock (measured:
# AMG is already ~8x faster at 21^3 and ~50x at 41^3, since 3D LU fill-in scales
# ~O(N^2)). Below it, LU is sub-second and exact, so ``solve(linear_solver="auto")``
# keeps LU for trivially small grids and prefers AMG above -- when pyamg is
# available (see docs/reduced_scalar_potential_3d.md).
_SPLU_MAX_UNKNOWNS = 20**3


@dataclass
class ScalarPotentialSolution:
    """Result of a 3D reduced-scalar-potential magnetostatic solve."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    psi: np.ndarray
    b_x: np.ndarray
    b_y: np.ndarray
    b_z: np.ndarray
    iterations: int
    residual: float

    @property
    def b_magnitude(self) -> np.ndarray:
        """Return ``|B|`` at every grid node."""

        return np.sqrt(self.b_x**2 + self.b_y**2 + self.b_z**2)

    def sample(self, x_m, y_m, z_m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(Bx, By, Bz)`` trilinearly interpolated at query points."""

        xi = np.interp(np.asarray(x_m, dtype=np.float64), self.x_m, np.arange(self.x_m.size))
        yi = np.interp(np.asarray(y_m, dtype=np.float64), self.y_m, np.arange(self.y_m.size))
        zi = np.interp(np.asarray(z_m, dtype=np.float64), self.z_m, np.arange(self.z_m.size))
        return (
            _trilinear(self.b_x, xi, yi, zi),
            _trilinear(self.b_y, xi, yi, zi),
            _trilinear(self.b_z, xi, yi, zi),
        )

    def mean_b_in(self, mask: np.ndarray) -> np.ndarray:
        """Return the mean ``(Bx, By, Bz)`` over a node ``mask``.

        Prefer this over the pointwise ``|B|`` peak as a field/saturation metric:
        staircased curved surfaces spike ``|B|`` by ``O(h)`` at corners (see the
        RSP limits doc).
        """

        mask = np.asarray(mask, dtype=bool)
        if not np.any(mask):
            raise ValueError("mask selects no grid nodes")
        return np.array(
            [float(np.mean(self.b_x[mask])),
             float(np.mean(self.b_y[mask])),
             float(np.mean(self.b_z[mask]))]
        )

    def homogeneity_ppm(self, *, center=(0.0, 0.0, 0.0), radius_m: float) -> float:
        """Return peak-to-peak ``|B|`` inhomogeneity (ppm) over a ball of interest."""

        gx, gy, gz = np.meshgrid(self.x_m, self.y_m, self.z_m, indexing="ij")
        mask = (
            (gx - center[0]) ** 2 + (gy - center[1]) ** 2 + (gz - center[2]) ** 2
            <= radius_m**2
        )
        if not np.any(mask):
            raise ValueError("no grid nodes inside the region of interest")
        mag = self.b_magnitude[mask]
        mean = float(np.mean(mag))
        if mean == 0.0:
            return np.inf
        return 1.0e6 * float(np.ptp(mag)) / mean


class ReducedScalarPotential3D:
    """3D nonlinear magnetostatics via the reduced scalar potential ``psi``.

    Build a problem on a uniform ``(x, y, z)`` node grid, paint material regions
    (linear, saturable, or permanent magnet), add coil currents and/or a uniform
    applied source field, then :meth:`solve`. Boundaries are Dirichlet ``psi = 0``
    (place them ``>~ 5x`` the feature size so the ``1/r^2`` reduced-potential tail
    has decayed -- see the RSP limits doc).

    Accuracy caveat: the field inside high-permeability iron is amplified by a
    cancellation error ``~ mu_r (h/L)^2``; the solver is trustworthy for
    ``mu_r <~ 100-300`` (excellent for permanent magnets, ``mu_r ~ 1.05``).
    """

    def __init__(
        self,
        x_m: Sequence[float] | np.ndarray,
        y_m: Sequence[float] | np.ndarray,
        z_m: Sequence[float] | np.ndarray,
    ):
        self.x = np.asarray(x_m, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y_m, dtype=np.float64).reshape(-1)
        self.z = np.asarray(z_m, dtype=np.float64).reshape(-1)
        if self.x.size < 3 or self.y.size < 3 or self.z.size < 3:
            raise ValueError("grid must have at least 3 nodes per axis")
        self.hx = float(self.x[1] - self.x[0])
        self.hy = float(self.y[1] - self.y[0])
        self.hz = float(self.z[1] - self.z[0])
        if not (
            np.allclose(np.diff(self.x), self.hx)
            and np.allclose(np.diff(self.y), self.hy)
            and np.allclose(np.diff(self.z), self.hz)
        ):
            raise ValueError("grid must be uniformly spaced on each axis")
        shape = (self.x.size, self.y.size, self.z.size)
        # Permeability (linear/small-signal) and nonlinear B-H parameters.
        self.mu = np.full(shape, MU0, dtype=np.float64)
        self.nl = np.zeros(shape, dtype=bool)
        self._bk1 = np.zeros(shape)
        self._bk2 = np.zeros(shape)
        self._bk3 = np.zeros(shape)
        self._b2cap = np.ones(shape)
        # Source field H_s (A/m) and remanence B_r (T), per component.
        self.hsx = np.zeros(shape, dtype=np.float64)
        self.hsy = np.zeros(shape, dtype=np.float64)
        self.hsz = np.zeros(shape, dtype=np.float64)
        self.brx = np.zeros(shape, dtype=np.float64)
        self.bry = np.zeros(shape, dtype=np.float64)
        self.brz = np.zeros(shape, dtype=np.float64)
        self._gx, self._gy, self._gz = np.meshgrid(
            self.x, self.y, self.z, indexing="ij"
        )
        self._solver_mode = "auto"

    # -- region masks -------------------------------------------------------
    def box(self, x_limits, y_limits, z_limits) -> np.ndarray:
        """Return a node mask for an axis-aligned box."""

        (x0, x1), (y0, y1), (z0, z1) = x_limits, y_limits, z_limits
        return (
            (self._gx >= min(x0, x1)) & (self._gx <= max(x0, x1))
            & (self._gy >= min(y0, y1)) & (self._gy <= max(y0, y1))
            & (self._gz >= min(z0, z1)) & (self._gz <= max(z0, z1))
        )

    def sphere(self, center, radius_m) -> np.ndarray:
        """Return a node mask for a filled ball."""

        return (
            (self._gx - center[0]) ** 2
            + (self._gy - center[1]) ** 2
            + (self._gz - center[2]) ** 2
            <= radius_m**2
        )

    def cylinder(self, center, radius_m, half_length_m, *, axis: str = "z") -> np.ndarray:
        """Return a node mask for a finite circular cylinder along ``axis``."""

        cx, cy, cz = center
        if axis == "z":
            radial = (self._gx - cx) ** 2 + (self._gy - cy) ** 2
            along = np.abs(self._gz - cz)
        elif axis == "y":
            radial = (self._gx - cx) ** 2 + (self._gz - cz) ** 2
            along = np.abs(self._gy - cy)
        elif axis == "x":
            radial = (self._gy - cy) ** 2 + (self._gz - cz) ** 2
            along = np.abs(self._gx - cx)
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        return (radial <= radius_m**2) & (along <= half_length_m)

    # -- painting -----------------------------------------------------------
    def add_material(
        self,
        mask: np.ndarray,
        material: MagneticMaterial,
        *,
        remanence_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> None:
        """Paint ``material`` onto masked nodes.

        A saturable ``material`` (``material.bh is not None``) marks the nodes
        nonlinear and seeds the small-signal permeability; a linear material sets
        ``mu = mu0 mu_r``. A permanent magnet (``remanence_t > 0``) writes a
        remanence vector ``B_r = remanence_t * unit(remanence_direction)``.
        """

        mask = np.asarray(mask, dtype=bool)
        if material.bh is not None:
            self.nl[mask] = True
            self._bk1[mask] = material.bh.bk1
            self._bk2[mask] = material.bh.bk2
            self._bk3[mask] = material.bh.bk3
            self._b2cap[mask] = material.bh.b2cap
            self.mu[mask] = MU0 * material.bh.mu_r0
        else:
            self.nl[mask] = False
            self.mu[mask] = MU0 * material.mu_r
        if material.remanence_t > 0.0:
            d = np.asarray(remanence_direction, dtype=np.float64)
            norm = float(np.linalg.norm(d))
            if norm == 0.0:
                raise ValueError("remanence_direction must be nonzero")
            d = d / norm
            self.brx[mask] = material.remanence_t * d[0]
            self.bry[mask] = material.remanence_t * d[1]
            self.brz[mask] = material.remanence_t * d[2]

    def add_coil(self, segments, current_a: float) -> None:
        """Add the Biot-Savart source field of a filament coil.

        ``segments`` is a sequence of ``(start, end)`` 3-D endpoints (m), as
        produced by :func:`spin_dynamics.fields.circular_loop`; ``current_a`` is
        the loop current (turns * amps). The coil contributes ``H_s = B_bs / mu0``
        with ``B_bs`` the free-space Biot-Savart field.
        """

        pts = np.stack([self._gx, self._gy, self._gz], axis=-1)
        b_bs = biot_savart(pts, segments, float(current_a))
        self.hsx += b_bs[..., 0] / MU0
        self.hsy += b_bs[..., 1] / MU0
        self.hsz += b_bs[..., 2] / MU0

    def add_uniform_source_field(self, h_vector: Sequence[float]) -> None:
        """Add a spatially uniform applied source field ``H_s`` (A/m).

        A uniform field is curl-free with no enclosed current, so it is a valid
        ``H_s``; use it to place a sample in a uniform applied field.
        """

        hx, hy, hz = (float(v) for v in h_vector)
        self.hsx += hx
        self.hsy += hy
        self.hsz += hz

    # -- physics ------------------------------------------------------------
    def _grad(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gx = np.gradient(psi, self.x, axis=0)
        gy = np.gradient(psi, self.y, axis=1)
        gz = np.gradient(psi, self.z, axis=2)
        return gx, gy, gz

    def _b_field(self, psi: np.ndarray, mu: np.ndarray):
        gx, gy, gz = self._grad(psi)
        b_x = mu * (self.hsx - gx) + self.brx
        b_y = mu * (self.hsy - gy) + self.bry
        b_z = mu * (self.hsz - gz) + self.brz
        return b_x, b_y, b_z

    def _coeff(self, psi: np.ndarray) -> np.ndarray:
        """Return the permeability field for the current potential (Picard step).

        ``mu`` is a *stateless* function of ``psi``: ``H = H_s - grad(psi)`` is
        available from the unknown alone, and for an isotropic material the B-H
        curve gives a one-to-one ``|H| -> mu``. Making ``mu`` depend only on
        ``psi`` (not on a carried-over permeability) gives the same clean
        ``psi -> mu(psi) -> solve`` fixed point the vector-potential solvers enjoy
        via ``B = curl A`` -- a hidden ``mu`` state instead puts the mu<->psi
        feedback outside the Anderson mixing and it fails to converge under
        partial saturation. Linear problems short-circuit to the fixed ``mu``.
        """

        if not self.nl.any():
            return self.mu
        gx, gy, gz = self._grad(psi)
        h2 = (self.hsx - gx) ** 2 + (self.hsy - gy) ** 2 + (self.hsz - gz) ** 2
        return np.where(self.nl, self._mu_from_h2(h2), self.mu)

    def _mu_from_h2(self, h2: np.ndarray) -> np.ndarray:
        """Invert the Brauer B-H curve ``nu(B^2) B = |H|`` for ``mu = |B|/|H|``.

        Newton on ``B`` (monotone, so it converges globally from the small-signal
        upper bound ``B0 = |H|/(bk1+bk2)``). Evaluated on every node with the
        per-node ``bk`` arrays; non-nl nodes carry ``bk = 0`` and are discarded by
        the caller's ``np.where``, so a safe denominator is used for them.
        """

        h = np.sqrt(h2)
        nu0 = np.where(self.nl, self._bk1 + self._bk2, 1.0)  # small-signal reluctivity
        b = h / nu0  # small-signal (upper-bound) start
        for _ in range(40):
            b2 = np.minimum(b * b, self._b2cap)
            capped = b * b >= self._b2cap
            e = np.exp(self._bk3 * b2)
            nu = self._bk1 + self._bk2 * e
            clamped = nu >= 1.0 / MU0
            nu = np.minimum(nu, 1.0 / MU0)
            f = nu * b - h
            dnu_db = np.where(capped | clamped, 0.0, 2.0 * b * self._bk2 * self._bk3 * e)
            # df = nu + b dnu/db > 0 on nl nodes (bk1 > 0); air nodes (bk = 0) are
            # discarded by the caller, so use a safe denominator there.
            df = np.where(self.nl, nu + b * dnu_db, 1.0)
            b = np.maximum(b - f / df, 0.0)
        # mu = B/H, with the small-signal limit 1/(bk1+bk2) as H -> 0.
        tiny = h <= 0.0
        return np.where(tiny, 1.0 / nu0, b / np.where(tiny, 1.0, h))

    def _faces(self, arr: np.ndarray):
        """Return arithmetic face averages of a node array along each axis."""

        fx = 0.5 * (arr[1:, :, :] + arr[:-1, :, :])
        fy = 0.5 * (arr[:, 1:, :] + arr[:, :-1, :])
        fz = 0.5 * (arr[:, :, 1:] + arr[:, :, :-1])
        return fx, fy, fz

    def _rhs(self, mu: np.ndarray) -> np.ndarray:
        """Return ``-div((mu - mu0) H_s + B_r)`` with operator-consistent averaging.

        The Biot-Savart source ``H_s`` is analytically divergence-free, so
        ``div(mu H_s) = div((mu - mu0) H_s)`` (the vacuum part ``div(mu0 H_s)``
        vanishes). Subtracting it is essential, not cosmetic: the raw
        ``div(mu0 H_s)`` of the near-singular sampled field close to a coil wire is
        large and grid-erratic and would swamp the solve, whereas ``(mu - mu0)``
        is nonzero only inside materials (away from the wire) where ``H_s`` is
        smooth. This also makes free space exact -- ``psi == 0``, ``B = mu0 H_s``.
        The source flux uses the same face-averaged ``mu`` as the operator.
        """

        mux, muy, muz = self._faces(mu)
        hsx_f, _, _ = self._faces(self.hsx)
        _, hsy_f, _ = self._faces(self.hsy)
        _, _, hsz_f = self._faces(self.hsz)
        brx_f, _, _ = self._faces(self.brx)
        _, bry_f, _ = self._faces(self.bry)
        _, _, brz_f = self._faces(self.brz)
        sx = (mux - MU0) * hsx_f + brx_f  # normal source flux on x-faces (T)
        sy = (muy - MU0) * hsy_f + bry_f
        sz = (muz - MU0) * hsz_f + brz_f
        div = np.zeros(mu.shape, dtype=np.float64)
        div[1:-1, 1:-1, 1:-1] = (
            (sx[1:, 1:-1, 1:-1] - sx[:-1, 1:-1, 1:-1]) / self.hx
            + (sy[1:-1, 1:, 1:-1] - sy[1:-1, :-1, 1:-1]) / self.hy
            + (sz[1:-1, 1:-1, 1:] - sz[1:-1, 1:-1, :-1]) / self.hz
        )
        rhs = -div
        rhs[0, :, :] = rhs[-1, :, :] = 0.0
        rhs[:, 0, :] = rhs[:, -1, :] = 0.0
        rhs[:, :, 0] = rhs[:, :, -1] = 0.0
        return rhs

    def _operator(self, mu: np.ndarray):
        mux, muy, muz = self._faces(mu)
        hx2, hy2, hz2 = self.hx**2, self.hy**2, self.hz**2

        def apply(psi: np.ndarray) -> np.ndarray:
            out = np.zeros_like(psi)
            fx = mux * (psi[1:, :, :] - psi[:-1, :, :])
            fy = muy * (psi[:, 1:, :] - psi[:, :-1, :])
            fz = muz * (psi[:, :, 1:] - psi[:, :, :-1])
            out[1:-1, 1:-1, 1:-1] = -(
                (fx[1:, 1:-1, 1:-1] - fx[:-1, 1:-1, 1:-1]) / hx2
                + (fy[1:-1, 1:, 1:-1] - fy[1:-1, :-1, 1:-1]) / hy2
                + (fz[1:-1, 1:-1, 1:] - fz[1:-1, 1:-1, :-1]) / hz2
            )
            return out

        diag = np.ones_like(mu)
        diag[1:-1, 1:-1, 1:-1] = (
            (mux[1:, 1:-1, 1:-1] + mux[:-1, 1:-1, 1:-1]) / hx2
            + (muy[1:-1, 1:, 1:-1] + muy[1:-1, :-1, 1:-1]) / hy2
            + (muz[1:-1, 1:-1, 1:] + muz[1:-1, 1:-1, :-1]) / hz2
        )
        return apply, diag

    def _sparse_coeffs(self, mu: np.ndarray):
        hx2, hy2, hz2 = self.hx**2, self.hy**2, self.hz**2
        mux, muy, muz = self._faces(mu)
        c_ip = np.zeros_like(mu)
        c_ip[:-1, :, :] = mux / hx2
        c_im = np.zeros_like(mu)
        c_im[1:, :, :] = mux / hx2
        c_jp = np.zeros_like(mu)
        c_jp[:, :-1, :] = muy / hy2
        c_jm = np.zeros_like(mu)
        c_jm[:, 1:, :] = muy / hy2
        c_kp = np.zeros_like(mu)
        c_kp[:, :, :-1] = muz / hz2
        c_km = np.zeros_like(mu)
        c_km[:, :, 1:] = muz / hz2
        diag = c_ip + c_im + c_jp + c_jm + c_kp + c_km
        b = np.zeros(mu.shape, dtype=bool)
        b[0, :, :] = b[-1, :, :] = True
        b[:, 0, :] = b[:, -1, :] = True
        b[:, :, 0] = b[:, :, -1] = True
        diag[b] = 1.0
        for arr in (c_ip, c_im, c_jp, c_jm, c_kp, c_km):
            arr[b] = 0.0
        return diag, c_ip, c_im, c_jp, c_jm, c_kp, c_km

    def _resolve_solver(self, n_unknowns: int) -> str:
        """Pick the concrete linear solver for ``linear_solver='auto'``.

        Sparse LU is exact and best for a single small solve; AMG wins for large
        grids (splu fill-in) and for nonlinear problems (many solves, where splu
        re-factorizes every Picard step while AMG re-uses a cheap O(N) setup).
        """

        mode = self._solver_mode
        if mode != "auto":
            return mode
        nonlinear = bool(self.nl.any())
        if _HAVE_PYAMG and (nonlinear or n_unknowns > _SPLU_MAX_UNKNOWNS):
            return "amg"
        if _HAVE_SCIPY and n_unknowns <= _SPLU_MAX_UNKNOWNS:
            return "splu"  # small linear: exact direct solve, no tolerance to tune
        if _HAVE_PYAMG:
            return "amg"
        if _HAVE_SCIPY:
            return "splu"
        return "cg"

    def _make_linsolve(self, mu, free, cg_tol, cg_max_iter):
        mode = self._resolve_solver(mu.size)
        if mode == "splu":
            if not _HAVE_SCIPY:
                raise RuntimeError("linear_solver='splu' requires SciPy")
            return _sparse_factorize_3d(*self._sparse_coeffs(mu))
        if mode == "amg":
            if not _HAVE_PYAMG:
                raise RuntimeError("linear_solver='amg' requires pyamg")
            return _amg_linsolve_3d(self._sparse_coeffs(mu), cg_tol, cg_max_iter)
        apply, diag = self._operator(mu)

        def linsolve(rhs, x0=None):
            return _conjugate_gradient(
                apply, rhs, diag, free, x0=x0, tol=cg_tol, max_iter=cg_max_iter
            )

        return linsolve

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
        linear_solver: str = "auto",
    ) -> ScalarPotentialSolution:
        """Solve the (possibly nonlinear) problem and return the fields.

        Nonlinear problems reuse the vector-potential solvers' Anderson-accelerated
        Picard driver; the reduced-scalar-potential right-hand side
        ``div(mu H_s + B_r)`` is recomputed each iteration (it depends on the
        evolving ``mu``).

        ``linear_solver`` selects the inner linear solve: ``"auto"`` (default)
        prefers AMG-preconditioned CG whenever pyamg is available and the grid is
        beyond trivially small (~20^3), keeping the exact sparse LU only for tiny
        grids and as the fallback when pyamg is absent; ``"splu"``, ``"amg"``, and
        ``"cg"`` force a specific solver. AMG is the path to grids beyond the
        ~50-64^3 sparse-LU ceiling -- and grid refinement is the effective cure for
        the high-mu_r cancellation error (the total/reduced-potential split does
        not help on this centered scheme; see docs/reduced_scalar_potential_3d.md).
        """

        valid = {"auto", "splu", "amg", "cg"}
        if linear_solver not in valid:
            raise ValueError(f"linear_solver must be one of {sorted(valid)}")
        self._solver_mode = linear_solver
        shape = (self.x.size, self.y.size, self.z.size)
        free = np.ones(shape, dtype=bool)
        free[0, :, :] = free[-1, :, :] = False
        free[:, 0, :] = free[:, -1, :] = False
        free[:, :, 0] = free[:, :, -1] = False
        psi, total, last = _anderson_picard(
            self._coeff,
            self._make_linsolve,
            np.zeros(shape),
            free,
            linear=not self.nl.any(),
            relax=relax,
            tol=tol,
            max_iter=max_iter,
            depth=depth,
            n_load_steps=n_load_steps,
            cg_tol=cg_tol,
            cg_max_iter=cg_max_iter,
            rhs_fn=self._rhs,
        )
        mu_final = self._coeff(psi)
        b_x, b_y, b_z = self._b_field(psi, mu_final)
        return ScalarPotentialSolution(
            x_m=self.x,
            y_m=self.y,
            z_m=self.z,
            psi=psi,
            b_x=b_x,
            b_y=b_y,
            b_z=b_z,
            iterations=total,
            residual=last,
        )

    def stored_energy(self, solution: ScalarPotentialSolution) -> float:
        """Return total magnetic energy ``integral (1/2) B.H dV`` (joules)."""

        gx, gy, gz = self._grad(solution.psi)
        h_x, h_y, h_z = self.hsx - gx, self.hsy - gy, self.hsz - gz
        density = 0.5 * (solution.b_x * h_x + solution.b_y * h_y + solution.b_z * h_z)
        return float(np.sum(density) * self.hx * self.hy * self.hz)


def _trilinear(grid: np.ndarray, xi: np.ndarray, yi: np.ndarray, zi: np.ndarray) -> np.ndarray:
    xi = np.clip(xi, 0, grid.shape[0] - 1.000001)
    yi = np.clip(yi, 0, grid.shape[1] - 1.000001)
    zi = np.clip(zi, 0, grid.shape[2] - 1.000001)
    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)
    z0 = np.floor(zi).astype(int)
    fx, fy, fz = xi - x0, yi - y0, zi - z0
    out = np.zeros(np.broadcast_shapes(fx.shape, fy.shape, fz.shape))
    for dx in (0, 1):
        wx = fx if dx else (1 - fx)
        for dy in (0, 1):
            wy = fy if dy else (1 - fy)
            for dz in (0, 1):
                wz = fz if dz else (1 - fz)
                out = out + grid[x0 + dx, y0 + dy, z0 + dz] * wx * wy * wz
    return out
