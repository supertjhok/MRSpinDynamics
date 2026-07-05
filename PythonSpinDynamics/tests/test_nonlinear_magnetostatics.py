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
        self.assertGreater(b_fer, 1.5 * b_air)


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
