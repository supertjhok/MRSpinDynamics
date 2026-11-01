from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.nonlinear_magnetostatics import (
    MU0,
    AxisymmetricMagnetostatics,
    MagneticMaterial,
    PlanarMagnetostatics,
    air,
    linear_material,
    ndfeb,
    rf_ferrite,
    soft_iron,
)


class MaterialModelTests(unittest.TestCase):
    def test_brauer_small_signal_and_saturation_clamp(self) -> None:
        bh = soft_iron().bh
        # Small-signal permeability at B = 0.
        self.assertAlmostEqual(bh.nu_of_b2(0.0), bh.bk1 + bh.bk2)
        self.assertGreater(bh.mu_r0, 1000.0)
        # Deep saturation clamps to vacuum reluctivity (mu_r >= 1).
        self.assertAlmostEqual(float(bh.nu_of_b2(1e6)), 1.0 / MU0)
        # Reluctivity rises monotonically with B^2 (permeability falls).
        b2 = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
        nu = bh.nu_of_b2(b2)
        self.assertTrue(np.all(np.diff(nu) >= 0.0))
        # Tangent is non-negative and vanishes in the clamped region.
        self.assertTrue(np.all(bh.dnu_db2(b2) >= 0.0))
        self.assertEqual(float(bh.dnu_db2(1e6)), 0.0)

    def test_material_presets(self) -> None:
        self.assertEqual(air().mu_r, 1.0)
        self.assertEqual(linear_material(500.0).mu_r, 500.0)
        self.assertGreater(rf_ferrite(1200.0).mu_r, 1000.0)
        magnet = ndfeb(1.3)
        self.assertAlmostEqual(magnet.remanence_t, 1.3)
        self.assertGreater(soft_iron().bh.mu_r0, 1.0)
        with self.assertRaises(ValueError):
            MagneticMaterial("bad", mu_r=-1.0)


class PlanarSolverTests(unittest.TestCase):
    def test_uniformly_magnetized_cylinder_interior_is_half_remanence(self) -> None:
        # A uniformly magnetized cylinder (mu_r = 1) has interior B = B_r / 2,
        # uniform and parallel to the remanence.
        radius = 0.03
        half = 12.0 * radius
        n = 161
        prob = PlanarMagnetostatics(
            np.linspace(-half, half, n), np.linspace(-half, half, n)
        )
        prob.add_material(
            prob.disk((0.0, 0.0), radius),
            MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
            remanence_angle_deg=0.0,
        )
        sol = prob.solve()
        bx, by = sol.sample([0.0, 0.005, -0.005], [0.0, 0.005, -0.005])
        self.assertTrue(np.allclose(bx, 0.5, atol=0.03))
        self.assertTrue(np.allclose(by, 0.0, atol=0.02))
        # Interior is homogeneous to a few percent over a sub-region.
        self.assertLess(sol.homogeneity_ppm(radius_m=0.5 * radius), 40000.0)

    def test_rotating_remanence_rotates_interior_field(self) -> None:
        radius = 0.03
        half = 10.0 * radius
        prob = PlanarMagnetostatics(
            np.linspace(-half, half, 121), np.linspace(-half, half, 121)
        )
        prob.add_material(
            prob.disk((0.0, 0.0), radius),
            MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
            remanence_angle_deg=90.0,
        )
        sol = prob.solve()
        bx, by = sol.sample([0.0], [0.0])
        # Remanence along +y -> interior field along +y.
        self.assertLess(abs(float(bx[0])), 0.05)
        self.assertGreater(float(by[0]), 0.4)

    @staticmethod
    def _cylinder_interior_bx(mu_r: float) -> float:
        radius = 0.03
        half = 12.0 * radius
        axis = np.linspace(-half, half, 161)
        prob = PlanarMagnetostatics(axis, axis)
        prob.add_material(
            prob.disk((0.0, 0.0), radius),
            MagneticMaterial("mag", mu_r=mu_r, remanence_t=1.0),
        )
        bx, _ = prob.solve().sample([0.0, 0.005, -0.005], [0.0, 0.005, -0.005])
        return float(np.mean(bx))

    def test_recoil_magnet_cylinder_matches_closed_form(self) -> None:
        # B = mu0 mu_rec H + B_r in a transversely magnetized cylinder gives the
        # interior field B_r / (1 + mu_rec). Pre-fix the source was B_r/mu0 rather
        # than B_r/(mu0 mu_rec), so mu_rec = 1.5 gave 0.606 instead of 0.4 (and the
        # ratio to the mu_rec = 1 cylinder was 1.22 instead of 0.8).
        b_recoil = self._cylinder_interior_bx(1.5)
        b_unit = self._cylinder_interior_bx(1.0)
        self.assertAlmostEqual(b_recoil, 1.0 / 2.5, delta=0.012)
        self.assertAlmostEqual(b_recoil / b_unit, 2.0 / 2.5, delta=0.02)

    def test_linear_solve_matches_interior_only_reference(self) -> None:
        # Independent assembly of -div(nu grad A) = J + curl(H_c) on the interior
        # nodes only (no Dirichlet rows), solved directly. Pre-fix the unit
        # Dirichlet rows next to ~1e11-scale interior rows left ~1e-4 relative
        # LU round-off on the boundary, so the returned A differed by ~1e-4.
        from scipy.sparse import diags, kron
        from scipy.sparse import identity as sparse_identity
        from scipy.sparse.linalg import spsolve

        n = 161
        axis = np.linspace(-0.36, 0.36, n)
        prob = PlanarMagnetostatics(axis, axis)
        prob.add_material(
            prob.disk((0.0, 0.0), 0.03),
            MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
        )
        sol = prob.solve()

        # Uniform nu = 1/mu0 everywhere here, so the operator is a scaled Laplacian.
        h = float(axis[1] - axis[0])
        m = n - 2
        second = diags([-np.ones(m - 1), 2.0 * np.ones(m), -np.ones(m - 1)], [-1, 0, 1])
        laplacian = (kron(second, sparse_identity(m)) + kron(sparse_identity(m), second)) / h**2
        rhs = prob._rhs()[1:-1, 1:-1].ravel()
        reference = spsolve((laplacian / MU0).tocsc(), rhs).reshape(m, m)

        scale = float(np.max(np.abs(reference)))
        np.testing.assert_allclose(sol.a_z[1:-1, 1:-1], reference, rtol=0.0, atol=1e-9 * scale)


class AxisymmetricSolverTests(unittest.TestCase):
    def _solenoid(self, core=None, core_r=0.04):
        a, length, ampere_turns = 0.05, 0.2, 1000.0
        nr, nz = 141, 281
        prob = AxisymmetricMagnetostatics(
            np.linspace(0.0, 0.6, nr), np.linspace(-0.6, 0.6, nz)
        )
        dr = 0.6 / (nr - 1)
        prob.add_material(
            prob.rectangle((a, a + 2 * dr), (-length / 2, length / 2)),
            air(),
            current_total_a=ampere_turns,
        )
        if core is not None:
            prob.add_material(
                prob.rectangle((0.0, core_r), (-length / 2, length / 2)), core
            )
        return prob, prob.solve(max_iter=40), a, length, ampere_turns

    def test_air_solenoid_matches_analytic_centre_field(self) -> None:
        _, sol, a, length, ni = self._solenoid()
        zc = int(np.argmin(np.abs(sol.z_m)))
        centre = float(sol.on_axis_bz()[zc])
        analytic = MU0 * ni / (2.0 * np.sqrt((length / 2) ** 2 + a**2))
        self.assertAlmostEqual(centre, analytic, delta=0.05 * analytic)
        # Radial field vanishes on the symmetry axis.
        self.assertLess(float(np.max(np.abs(sol.b_r[0, :]))), 1e-9)

    def test_high_permeability_core_boosts_axial_field(self) -> None:
        _, sol_air, _, _, _ = self._solenoid()
        _, sol_fer, _, _, _ = self._solenoid(core=rf_ferrite(1000.0))
        zc = int(np.argmin(np.abs(sol_air.z_m)))
        b_air = float(sol_air.on_axis_bz()[zc])
        b_fer = float(sol_fer.on_axis_bz()[zc])
        # A mu_r = 1000 core of length/diameter 2.5 is demagnetization-limited: the
        # central gain is ~1/N_f with the fluxmetric demagnetizing factor
        # N_f ~ 0.07-0.12 at this aspect ratio, i.e. a gain of roughly 8-14. (The
        # pre-fix operator, which did not keep H_z continuous at the core surface,
        # gave 1.97.)
        self.assertGreater(b_fer / b_air, 8.0)
        self.assertLess(b_fer / b_air, 14.0)

    def test_recoil_magnet_sphere_matches_closed_form(self) -> None:
        # Axially magnetized sphere with B = mu0 mu_rec H + B_r: interior field
        # 2 B_r / (2 + mu_rec). Pre-fix (source B_r/mu0) mu_rec = 1.5 gave 0.865
        # instead of 0.571, and the ratio to mu_rec = 1 was 1.295 instead of 0.857.
        def interior_bz(mu_rec: float) -> float:
            h = 1.0e-3
            r = np.arange(0.0, 0.2 + 0.5 * h, h)
            z = np.arange(-0.2, 0.2 + 0.5 * h, h)
            prob = AxisymmetricMagnetostatics(r, z)
            prob.add_material(
                prob._gr**2 + prob._gz**2 <= 0.02**2,
                MagneticMaterial("mag", mu_r=mu_rec, remanence_t=1.0),
                remanence_axis="z",
            )
            sol = prob.solve()
            return float(np.mean(sol.b_z[:6, int(np.argmin(np.abs(z)))]))

        b_recoil = interior_bz(1.5)
        b_unit = interior_bz(1.0)
        self.assertAlmostEqual(b_recoil, 2.0 / 3.5, delta=0.012)
        self.assertAlmostEqual(b_unit, 2.0 / 3.0, delta=0.012)
        self.assertAlmostEqual(b_recoil / b_unit, 3.0 / 3.5, delta=0.015)

    @staticmethod
    def _long_cored_solenoid(mu_r: float, *, h_r: float = 1.0e-3, r_max: float = 0.16):
        """Coil R = 10 mm, L = 0.6 m, full-length coaxial core rc = 6 mm."""

        r = np.arange(0.0, r_max + 0.5 * h_r, h_r)
        z = np.arange(-0.4, 0.4 + 1.0e-3, 2.0e-3)
        prob = AxisymmetricMagnetostatics(r, z)
        coil_r, length, core_r, ampere_turns = 0.010, 0.6, 0.006, 1000.0
        prob.add_material(
            prob.rectangle((coil_r, coil_r + h_r), (-length / 2, length / 2)),
            air(),
            current_total_a=ampere_turns,
        )
        prob.add_material(
            prob.rectangle((0.0, core_r), (-length / 2, length / 2)),
            linear_material(mu_r),
        )
        return prob, prob.solve(), ampere_turns / length

    def test_long_cored_solenoid_matches_closed_form(self) -> None:
        # Midplane of a long solenoid with a coaxial linear core: H_z is uniform
        # inside the winding (continuous across the core surface), jumps by the
        # sheet current n I across the winding, and the core carries
        # B = mu_r mu0 n I. Pre-fix values: core B / (mu_r mu0 n I) = 0.34 and
        # H_z(core) / H_z(gap) = 0.35.
        mu_r = 5.0
        prob, sol, sheet = self._long_cored_solenoid(mu_r)
        iz = int(np.argmin(np.abs(sol.z_m)))
        b_z = sol.b_z[:, iz]

        def at(radius):
            return float(b_z[int(np.argmin(np.abs(prob.r - radius)))])

        b_core, b_gap, b_out = at(0.003), at(0.008), at(0.030)
        self.assertAlmostEqual((b_core / mu_r) / b_gap, 1.0, delta=2e-3)
        self.assertAlmostEqual((b_gap - b_out) / (MU0 * sheet), 1.0, delta=2e-3)
        # The finite Dirichlet box returns ~1% of the flux outside the winding, which
        # lowers the interior H by the same ~1% (Ampere).
        self.assertAlmostEqual(b_core / (mu_r * MU0 * sheet), 1.0, delta=0.02)

    def test_ferrite_core_matches_conservative_flux_function_reference(self) -> None:
        # Independent assembly of curl(nu curl A) in the flux function psi = r A:
        # -d/dr(nu/r dpsi/dr) - d/dz(nu/r dpsi/dz) = J with face-averaged nu and
        # psi = 0 on the boundary. The solver must reproduce it (pre-fix the core
        # B_z at the midplane was 1.58 mu0 n I instead of 4.25 mu0 n I).
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve

        h = 1.0e-3
        r = np.arange(0.0, 0.04 + 0.5 * h, h)
        z = np.arange(-0.15, 0.15 + 0.5 * h, h)
        prob = AxisymmetricMagnetostatics(r, z)
        coil_r, length = 0.010, 0.24
        prob.add_material(
            prob.rectangle((coil_r, coil_r + h), (-length / 2, length / 2)),
            air(),
            current_total_a=1000.0,
        )
        prob.add_material(
            prob.rectangle((0.0, 0.006), (-length / 2 + 0.02, length / 2 - 0.02)),
            linear_material(5.0),
        )
        sol = prob.solve()

        nu = 1.0 / (MU0 * prob.mu_r)
        nr, nz = nu.shape
        index = np.arange(nr * nz).reshape(nr, nz)
        interior = np.zeros((nr, nz), dtype=bool)
        interior[1:-1, 1:-1] = True
        rows, cols, vals = [index[~interior]], [index[~interior]], [np.ones((~interior).sum())]
        ii, jj = np.nonzero(interior)
        r_face_hi = 0.5 * (r[ii] + r[ii + 1])
        r_face_lo = 0.5 * (r[ii] + r[ii - 1])
        c_e = 0.5 * (nu[ii, jj] + nu[ii + 1, jj]) / r_face_hi / h**2
        c_w = 0.5 * (nu[ii, jj] + nu[ii - 1, jj]) / r_face_lo / h**2
        c_n = 0.5 * (nu[ii, jj] + nu[ii, jj + 1]) / r[ii] / h**2
        c_s = 0.5 * (nu[ii, jj] + nu[ii, jj - 1]) / r[ii] / h**2
        here = index[ii, jj]
        for neighbour, coeff in (
            (index[ii + 1, jj], c_e),
            (index[ii - 1, jj], c_w),
            (index[ii, jj + 1], c_n),
            (index[ii, jj - 1], c_s),
        ):
            rows += [here, here]
            cols += [neighbour, here]
            vals += [-coeff, coeff]
        matrix = coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nr * nz, nr * nz),
        ).tocsr()
        rhs = np.where(interior, prob.jphi, 0.0).ravel()
        psi = spsolve(matrix, rhs).reshape(nr, nz)
        b_z_reference = np.gradient(psi, r, axis=0)[1:, :] / r[1:, None]

        scale = float(np.max(np.abs(b_z_reference)))
        np.testing.assert_allclose(sol.b_z[1:, :], b_z_reference, rtol=0.0, atol=1e-9 * scale)
        iz = int(np.argmin(np.abs(z)))
        i_core = int(np.argmin(np.abs(r - 0.004)))
        self.assertAlmostEqual(
            sol.b_z[i_core, iz] / (MU0 * 1000.0 / length), 4.25, delta=0.02
        )


class NonlinearBehaviourTests(unittest.TestCase):
    def test_saturable_core_matches_linear_below_the_knee(self) -> None:
        # At a weak drive the saturable iron behaves like its small-signal mu_r.
        a, length = 0.05, 0.2
        nr, nz = 121, 241
        weak, strong = [], []
        for material, out in ((linear_material(soft_iron().bh.mu_r0), weak),
                              (soft_iron(), strong)):
            prob = AxisymmetricMagnetostatics(
                np.linspace(0.0, 0.6, nr), np.linspace(-0.6, 0.6, nz)
            )
            dr = 0.6 / (nr - 1)
            prob.add_material(
                prob.rectangle((a, a + 2 * dr), (-length / 2, length / 2)),
                air(),
                current_total_a=50.0,
            )
            prob.add_material(prob.rectangle((0.0, 0.04), (-length / 2, length / 2)), material)
            sol = prob.solve(max_iter=60, n_load_steps=2)
            zc = int(np.argmin(np.abs(sol.z_m)))
            out.append(float(sol.on_axis_bz()[zc]))
        # Well below saturation the nonlinear and linear cores agree closely.
        self.assertAlmostEqual(strong[0], weak[0], delta=0.1 * abs(weak[0]))

    def test_iron_ring_creates_and_bounds_bore_field(self) -> None:
        # Diametric ring alone gives ~0 bore field; the iron ring creates it, and
        # the iron saturates (bounded |B|) under a strong NdFeB drive.
        n = 121
        half = 0.35
        bare = PlanarMagnetostatics(np.linspace(-half, half, n), np.linspace(-half, half, n))
        bare.add_material(bare.annulus((0, 0), 0.02, 0.045), ndfeb(1.3), remanence_angle_deg=0.0)
        sol_bare = bare.solve()
        bx_bare = abs(float(sol_bare.sample([0.0], [0.0])[0][0]))

        withiron = PlanarMagnetostatics(np.linspace(-half, half, n), np.linspace(-half, half, n))
        withiron.add_material(withiron.annulus((0, 0), 0.02, 0.045), ndfeb(1.3), remanence_angle_deg=0.0)
        withiron.add_material(withiron.annulus((0, 0), 0.05, 0.09), soft_iron())
        sol_iron = withiron.solve(max_iter=120, relax=0.4, n_load_steps=3)
        bx_iron = abs(float(sol_iron.sample([0.0], [0.0])[0][0]))

        self.assertLess(bx_bare, 0.05)          # tube cancellation
        self.assertGreater(bx_iron, 0.2)        # iron establishes the bore field
        self.assertLess(sol_iron.residual, 1e-4)  # converged
        iron_mask = withiron.annulus((0, 0), 0.05, 0.09)
        self.assertLess(float(np.max(sol_iron.b_magnitude[iron_mask])), 2.6)


class ValidationTests(unittest.TestCase):
    def test_grid_and_material_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3 nodes"):
            PlanarMagnetostatics([0.0, 1.0], [0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            PlanarMagnetostatics([0.0, 1.0, 3.0], [0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "r >= 0"):
            AxisymmetricMagnetostatics([-1.0, 0.0, 1.0], [0.0, 1.0, 2.0])
        with self.assertRaises(KeyError):
            soft_iron("unobtanium")


if __name__ == "__main__":
    unittest.main()
