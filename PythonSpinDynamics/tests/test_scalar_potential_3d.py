from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.magnetostatics import circular_loop
from spin_dynamics.fields.nonlinear_magnetostatics import (
    MU0,
    BrauerBH,
    MagneticMaterial,
    _HAVE_PYAMG,
    linear_material,
)
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D


def _grid(half: float, n: int) -> np.ndarray:
    return np.linspace(-half, half, n)


class FreeSpaceTests(unittest.TestCase):
    def test_uniform_source_field_is_exact_and_psi_is_zero(self) -> None:
        # No materials: B = mu0 H_s exactly and the reduced potential vanishes
        # (the source term subtracts the vacuum part).
        g = _grid(0.1, 15)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_uniform_source_field((1000.0, 0.0, 0.0))
        sol = prob.solve()
        self.assertEqual(float(np.max(np.abs(sol.psi))), 0.0)
        self.assertLess(float(np.max(np.abs(sol.b_x - MU0 * 1000.0))), 1e-12)
        self.assertLess(float(np.max(np.abs(sol.b_y))), 1e-12)

    def test_coil_in_free_space_matches_biot_savart(self) -> None:
        # A loop in free space must reproduce its Biot-Savart field (psi == 0);
        # the vacuum-subtracted source is essential -- the raw div(H_s) of the
        # near-singular sampled field near the wire would otherwise swamp it.
        radius = 0.05
        g = _grid(0.15, 25)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_coil(circular_loop((0.0, 0.0, 0.0), radius, axis="z", n_segments=160), 100.0)
        sol = prob.solve()
        self.assertEqual(float(np.max(np.abs(sol.psi))), 0.0)
        c = 12
        analytic = MU0 * 100.0 / (2.0 * radius)  # on-axis loop centre
        self.assertAlmostEqual(float(sol.b_z[c, c, c]), analytic, delta=0.01 * analytic)


class PermanentMagnetTests(unittest.TestCase):
    def test_uniformly_magnetized_sphere_interior_is_two_thirds_remanence(self) -> None:
        # Uniformly magnetized sphere (mu_r = 1): interior B = (2/3) B_r, uniform,
        # parallel to the remanence (3D demagnetizing factor 1/3).
        g = _grid(0.15, 31)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_material(
            prob.sphere((0.0, 0.0, 0.0), 0.03),
            MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
            remanence_direction=(0.0, 0.0, 1.0),
        )
        sol = prob.solve()
        mean = sol.mean_b_in(prob.sphere((0.0, 0.0, 0.0), 0.015))
        self.assertAlmostEqual(float(mean[2]), 2.0 / 3.0, delta=0.03)
        self.assertLess(abs(float(mean[0])), 0.02)
        self.assertLess(abs(float(mean[1])), 0.02)

    def test_remanence_direction_sets_field_direction(self) -> None:
        g = _grid(0.15, 31)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_material(
            prob.sphere((0.0, 0.0, 0.0), 0.03),
            MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
            remanence_direction=(1.0, 0.0, 0.0),
        )
        sol = prob.solve()
        mean = sol.mean_b_in(prob.sphere((0.0, 0.0, 0.0), 0.015))
        self.assertGreater(float(mean[0]), 0.4)
        self.assertLess(abs(float(mean[2])), 0.02)


class LinearMaterialTests(unittest.TestCase):
    def test_low_permeability_sphere_matches_analytic(self) -> None:
        # A mu_r = 3 sphere in a uniform field: interior B = mu0 * 3 mu_r/(mu_r+2) H0.
        # RSP is accurate at low mu_r (the cancellation error grows as mu_r).
        mu_r, h0 = 3.0, 1000.0
        g = _grid(0.15, 31)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_material(prob.sphere((0.0, 0.0, 0.0), 0.045), linear_material(mu_r))
        prob.add_uniform_source_field((0.0, 0.0, h0))
        sol = prob.solve()
        b_in = float(sol.mean_b_in(prob.sphere((0.0, 0.0, 0.0), 0.02))[2])
        analytic = MU0 * mu_r * 3.0 * h0 / (mu_r + 2.0)
        self.assertAlmostEqual(b_in / analytic, 1.0, delta=0.08)


class NonlinearTests(unittest.TestCase):
    def _core_mu_eff(self, material, drive, **kw):
        g = _grid(0.15, 31)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_material(prob.sphere((0.0, 0.0, 0.0), 0.03), material)
        prob.add_uniform_source_field((0.0, 0.0, drive))
        sol = prob.solve(**kw)
        inner = prob.sphere((0.0, 0.0, 0.0), 0.015)
        gx, gy, gz = prob._grad(sol.psi)
        h_mag = np.mean(
            np.sqrt((prob.hsx - gx) ** 2 + (prob.hsy - gy) ** 2 + (prob.hsz - gz) ** 2)[inner]
        )
        return float(np.mean(sol.b_magnitude[inner]) / h_mag / MU0), sol.residual

    def test_saturable_core_matches_linear_below_the_knee(self) -> None:
        # A moderate saturable material (mu_r0 ~ 50, in the RSP-trustworthy band).
        bh = BrauerBH(bk1=5915.0, bk2=10000.0, bk3=3.0)
        self.assertAlmostEqual(bh.mu_r0, 50.0, delta=1.0)
        mu_sat, res = self._core_mu_eff(MagneticMaterial("sat", bh=bh), 50.0)
        mu_lin, _ = self._core_mu_eff(linear_material(bh.mu_r0), 50.0)
        self.assertLess(res, 1e-4)  # clean convergence at weak drive
        self.assertAlmostEqual(mu_sat / mu_lin, 1.0, delta=0.02)

    def test_strong_drive_saturates_toward_vacuum(self) -> None:
        # Under an overwhelming drive the effective permeability collapses to ~1
        # (the mu_r >= 1 vacuum clamp). This is the stateless-mu(psi) fixed point;
        # a carried-over mu state fails to converge under partial saturation.
        bh = BrauerBH(bk1=5915.0, bk2=10000.0, bk3=3.0)
        mu_eff, res = self._core_mu_eff(MagneticMaterial("sat", bh=bh), 1.0e6)
        self.assertLess(res, 1e-4)
        self.assertLess(mu_eff, 1.2)


class LinearSolverTests(unittest.TestCase):
    def _sphere_b_in(self, solver, n, mu_r):
        g = _grid(0.15, n)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_material(prob.sphere((0.0, 0.0, 0.0), 0.045), linear_material(mu_r))
        prob.add_uniform_source_field((0.0, 0.0, 1000.0))
        sol = prob.solve(linear_solver=solver, cg_tol=1e-10)
        return float(sol.mean_b_in(prob.sphere((0.0, 0.0, 0.0), 0.02))[2])

    def test_amg_and_cg_agree_with_splu(self) -> None:
        # All three linear solvers must reach the same field on a linear problem.
        ref = self._sphere_b_in("splu", 27, 5.0)
        if _HAVE_PYAMG:
            self.assertAlmostEqual(self._sphere_b_in("amg", 27, 5.0) / ref, 1.0, delta=1e-4)
        self.assertAlmostEqual(self._sphere_b_in("cg", 27, 5.0) / ref, 1.0, delta=1e-4)

    def test_refinement_reduces_high_mu_error(self) -> None:
        # Grid refinement is the effective cure for the high-mu_r cancellation
        # error (the total/reduced-potential split does not help; see the doc).
        if not _HAVE_PYAMG:
            self.skipTest("pyamg not installed")
        mu_r = 500.0
        analytic = MU0 * mu_r * 3.0 * 1000.0 / (mu_r + 2.0)
        err_coarse = abs(self._sphere_b_in("amg", 31, mu_r) / analytic - 1.0)
        err_fine = abs(self._sphere_b_in("amg", 51, mu_r) / analytic - 1.0)
        self.assertLess(err_fine, err_coarse)

    def test_invalid_solver_rejected(self) -> None:
        g = _grid(0.1, 11)
        prob = ReducedScalarPotential3D(g, g, g)
        with self.assertRaisesRegex(ValueError, "linear_solver"):
            prob.solve(linear_solver="bogus")

    def test_auto_dispatch(self) -> None:
        g = _grid(0.1, 11)
        prob = ReducedScalarPotential3D(g, g, g)
        prob._solver_mode = "auto"
        # Small linear problem -> exact sparse LU.
        self.assertEqual(prob._resolve_solver(27_000), "splu")
        if _HAVE_PYAMG:
            # Large problem -> AMG-preconditioned CG.
            self.assertEqual(prob._resolve_solver(500_000), "amg")


class ValidationTests(unittest.TestCase):
    def test_grid_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3 nodes"):
            ReducedScalarPotential3D([0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "uniformly spaced"):
            ReducedScalarPotential3D([0.0, 1.0, 3.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])

    def test_zero_remanence_direction_rejected(self) -> None:
        g = _grid(0.1, 11)
        prob = ReducedScalarPotential3D(g, g, g)
        with self.assertRaisesRegex(ValueError, "nonzero"):
            prob.add_material(
                prob.sphere((0.0, 0.0, 0.0), 0.03),
                MagneticMaterial("mag", mu_r=1.0, remanence_t=1.0),
                remanence_direction=(0.0, 0.0, 0.0),
            )

    def test_mean_b_in_empty_mask_raises(self) -> None:
        g = _grid(0.1, 11)
        prob = ReducedScalarPotential3D(g, g, g)
        prob.add_uniform_source_field((0.0, 0.0, 100.0))
        sol = prob.solve()
        with self.assertRaisesRegex(ValueError, "no grid nodes|selects no"):
            sol.mean_b_in(np.zeros((g.size, g.size, g.size), dtype=bool))


if __name__ == "__main__":
    unittest.main()
