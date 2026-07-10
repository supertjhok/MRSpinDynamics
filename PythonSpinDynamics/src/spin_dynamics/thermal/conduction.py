"""1D / axisymmetric conduction solver (Phase 3 of thermal_modeling.md).

Solves the heat equation on a slab, an infinite cylinder (radial), or a
sphere (radial) with a spatially varying volumetric source, optional Pennes
perfusion for biological tissue, and mixed boundary conditions -- the two
places internal temperature *profiles* matter that the lumped network cannot
resolve (SAR hot-spot inside a lossy/tissue sample, coil-former stack).

Geometry enters only through the divergence operator, written in the unified
form ``(1/r^m) d/dr (r^m k dT/dr)`` with ``m = 0`` (slab), ``1`` (cylinder),
``2`` (sphere). A conservative finite-volume discretization on face-centered
conductances keeps the steady state exact for piecewise-constant ``k`` and
conserves energy; time stepping is Crank-Nicolson (unconditionally stable).

Pennes bioheat: ``rho c_p dT/dt = div(k grad T) + q_v + w_b c_b (T_a - T)``,
with blood perfusion ``w_b`` (kg/m^3/s), blood specific heat ``c_b``, and
arterial temperature ``T_a`` -- a linear sink toward ``T_a`` with rate
``w_b c_b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "PerfusionModel",
    "ConductionResult",
    "Conduction1D",
]

_GEOMETRY_M = {"slab": 0, "cylinder": 1, "sphere": 2}


@dataclass(frozen=True)
class PerfusionModel:
    """Pennes blood-perfusion sink ``w_b c_b (T_a - T)``.

    ``blood_perfusion`` ``w_b`` in kg/(m^3 s), ``blood_specific_heat`` ``c_b``
    in J/(kg K), ``arterial_temperature`` ``T_a`` in K. The volumetric sink
    rate is ``w_b c_b`` in W/(m^3 K).
    """

    blood_perfusion: float
    blood_specific_heat: float = 3617.0
    arterial_temperature: float = 310.15  # 37 C

    def __post_init__(self) -> None:
        if self.blood_perfusion < 0 or not np.isfinite(self.blood_perfusion):
            raise ValueError("blood_perfusion must be finite and non-negative")
        if self.blood_specific_heat <= 0 or not np.isfinite(self.blood_specific_heat):
            raise ValueError("blood_specific_heat must be finite and positive")
        if self.arterial_temperature <= 0 or not np.isfinite(self.arterial_temperature):
            raise ValueError("arterial_temperature must be finite and positive")

    @property
    def sink_rate(self) -> float:
        """Volumetric conductance to arterial blood ``w_b c_b`` (W/m^3/K)."""

        return self.blood_perfusion * self.blood_specific_heat


@dataclass(frozen=True)
class ConductionResult:
    """Radial/linear temperature profile(s)."""

    r: np.ndarray
    temperature: np.ndarray            # steady: (n,); transient: (steps, n)
    times: np.ndarray | None = None

    @property
    def peak(self) -> float:
        return float(np.max(self.temperature))

    def final(self) -> np.ndarray:
        return self.temperature if self.temperature.ndim == 1 else self.temperature[-1]


class Conduction1D:
    """Finite-volume conduction on a slab / cylinder / sphere radial grid.

    ``r`` is the cell-center coordinate (m), uniformly spaced from a center/
    inner face to the outer face. ``conductivity``, ``rho_cp`` (J/m^3/K), and
    the volumetric ``source`` (W/m^3) are per-cell arrays or scalars.

    Boundary conditions per end are ``("insulated",)``,
    ``("temperature", T)``, or ``("convection", h, T_inf)``. The inner end of
    a cylinder/sphere defaults to insulated (symmetry) and should be left so
    when the grid starts at ``r = 0``.
    """

    def __init__(
        self,
        r: np.ndarray,
        *,
        geometry: str = "slab",
        conductivity,
        rho_cp,
        source=0.0,
        perfusion: PerfusionModel | None = None,
        inner_bc: tuple = ("insulated",),
        outer_bc: tuple = ("insulated",),
    ) -> None:
        if geometry not in _GEOMETRY_M:
            raise ValueError("geometry must be 'slab', 'cylinder', or 'sphere'")
        self.m = _GEOMETRY_M[geometry]
        self.geometry = geometry
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        n = r.size
        if n < 3 or not np.all(np.diff(r) > 0):
            raise ValueError("r must be strictly increasing with >= 3 cells")
        dr = r[1] - r[0]
        if not np.allclose(np.diff(r), dr):
            raise ValueError("r must be uniformly spaced")
        self.r = r
        self.n = n
        self.dr = float(dr)
        self.k = np.broadcast_to(np.asarray(conductivity, dtype=np.float64), (n,)).copy()
        self.rho_cp = np.broadcast_to(np.asarray(rho_cp, dtype=np.float64), (n,)).copy()
        if np.any(self.k <= 0) or np.any(self.rho_cp <= 0):
            raise ValueError("conductivity and rho_cp must be positive")
        self.source = np.broadcast_to(np.asarray(source, dtype=np.float64), (n,)).copy()
        self.perfusion = perfusion
        self.inner_bc = inner_bc
        self.outer_bc = outer_bc
        # Face radii (n+1 faces) and their geometric weights r_face^m.
        self.r_faces = np.concatenate(([r[0] - dr / 2], r + dr / 2))
        if self.m > 0 and self.r_faces[0] < 0:
            self.r_faces[0] = 0.0
        self._build_operator()

    # ------------------------------------------------------------------

    def _face_weight(self, r_face: np.ndarray) -> np.ndarray:
        return r_face**self.m if self.m > 0 else np.ones_like(r_face)

    def _cell_weight(self) -> np.ndarray:
        # Finite-volume cell measure ~ r^m dr (m=0 slab -> dr).
        if self.m == 0:
            return np.full(self.n, self.dr)
        return self.r**self.m * self.dr

    def _build_operator(self) -> None:
        """Assemble the conduction matrix ``A`` (W/K per cell measure) and BC vectors."""

        n, dr = self.n, self.dr
        wf = self._face_weight(self.r_faces)  # (n+1,)
        # Harmonic-mean face conductivity for interior faces.
        k = self.k
        k_face = np.zeros(n + 1)
        k_face[1:-1] = 2.0 * k[:-1] * k[1:] / (k[:-1] + k[1:])
        # Interior face conductances g_face = w_face * k_face / dr.
        g = wf * k_face / dr  # (n+1,), ends filled by BCs below

        a = np.zeros((n, n))
        self._bc_source = np.zeros(n)      # fixed inflow per cell measure (W)
        self._bc_diag = np.zeros(n)        # extra diagonal from Dirichlet/convection

        for i in range(n):
            if i > 0:
                a[i, i - 1] += g[i]
                a[i, i] -= g[i]
            if i < n - 1:
                a[i, i + 1] += g[i + 1]
                a[i, i] -= g[i + 1]

        self._apply_boundary(a, side="inner", face=0, cell=0, weight=wf[0])
        self._apply_boundary(a, side="outer", face=n, cell=n - 1, weight=wf[-1])
        self.A = a
        self.cell_measure = self._cell_weight()

    def _apply_boundary(self, a, *, side, face, cell, weight):
        bc = self.inner_bc if side == "inner" else self.outer_bc
        kind = bc[0]
        if kind == "insulated":
            return
        k_b = self.k[cell]
        half = self.dr / 2.0
        # Conductance from cell center to the boundary face.
        g_b = weight * k_b / half
        if kind == "temperature":
            (t_bc,) = bc[1:]
            a[cell, cell] -= g_b
            self._bc_source[cell] += g_b * t_bc
        elif kind == "convection":
            h, t_inf = bc[1:]
            # Series of wall conduction (g_b) and film (weight * h).
            g_film = weight * h
            g_eff = 1.0 / (1.0 / g_b + 1.0 / g_film)
            a[cell, cell] -= g_eff
            self._bc_source[cell] += g_eff * t_inf
        else:
            raise ValueError(f"unknown boundary condition '{kind}'")

    def _perfusion_terms(self):
        if self.perfusion is None:
            return np.zeros(self.n), np.zeros(self.n)
        rate = self.perfusion.sink_rate
        measure = self.cell_measure
        diag = rate * measure                    # W/K, sink
        src = rate * self.perfusion.arterial_temperature * measure  # W
        return diag, src

    # ------------------------------------------------------------------

    def steady_state(self) -> ConductionResult:
        """Solve ``A T + sources = 0`` for the steady profile."""

        perf_diag, perf_src = self._perfusion_terms()
        a = self.A.copy()
        a[np.diag_indices(self.n)] -= perf_diag
        rhs = -(self.source * self.cell_measure + self._bc_source + perf_src)
        temp = np.linalg.solve(a, rhs)
        return ConductionResult(r=self.r, temperature=temp)

    def transient(
        self,
        times: np.ndarray,
        *,
        initial_temperature,
    ) -> ConductionResult:
        """Crank-Nicolson march. ``initial_temperature`` scalar or per-cell."""

        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if times.size < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing with >= 2 samples")
        t0 = np.broadcast_to(
            np.asarray(initial_temperature, dtype=np.float64), (self.n,)
        ).astype(np.float64)
        perf_diag, perf_src = self._perfusion_terms()
        measure = self.cell_measure
        cap = self.rho_cp * measure                 # J/K per cell
        # dT/dt = M^{-1} (A T + b), with A including perfusion sink.
        a = self.A.copy()
        a[np.diag_indices(self.n)] -= perf_diag
        b = self.source * measure + self._bc_source + perf_src
        minv = np.diag(1.0 / cap)
        op = minv @ a
        rhs_const = minv @ b

        out = np.zeros((times.size, self.n))
        out[0] = t0
        temp = t0.copy()
        identity = np.eye(self.n)
        for idx in range(times.size - 1):
            dt = float(times[idx + 1] - times[idx])
            lhs = identity - 0.5 * dt * op
            rhs = temp + 0.5 * dt * (op @ temp + 2.0 * rhs_const)
            temp = np.linalg.solve(lhs, rhs)
            out[idx + 1] = temp
        return ConductionResult(r=self.r, temperature=out, times=times)

    def lumped_capacity(self) -> float:
        """Total heat capacity ``sum(rho_cp * cell_measure)`` for cross-checks.

        For a slab this is per unit area; for cylinder/sphere it carries the
        ``r^m`` weight (per unit length / per ``4 pi`` solid angle).
        """

        return float(np.sum(self.rho_cp * self.cell_measure))
