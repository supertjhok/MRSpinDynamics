"""Tests for the quasistatic E-field / eddy-current field solvers.

Validated against analytics: curl(A) = Biot-Savart B, the long-solenoid Faraday
field, Maxwell's mutual-inductance formula, the conducting-sphere eddy power, the
charge-correction screening/tangency, and the single-ring L/R time constant.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import coils  # noqa: E402
from spin_dynamics.fields.eddy_modes import EddyModes  # noqa: E402
from spin_dynamics.fields.magnetostatics import MU0, biot_savart, circular_loop  # noqa: E402
from spin_dynamics.fields.quasistatic import (  # noqa: E402
    _charge_correction,
    coil_inductance,
    coil_loading,
    eddy_currents,
    eddy_power,
    induced_efield,
    mutual_inductance,
    reflected_resistance,
    self_inductance_circular,
    vector_potential,
)


class VectorPotentialTests(unittest.TestCase):
    def test_curl_a_matches_biot_savart(self) -> None:
        loop = circular_loop([0, 0, 0], 0.05, axis="z", n_segments=200)
        p = np.array([0.01, 0.008, 0.012])
        h = 1e-6

        def a(pt):
            return vector_potential(pt[None, :], loop, 1.0)[0]

        curl = np.zeros(3)
        for c, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
            di = np.zeros(3)
            di[i] = h
            dj = np.zeros(3)
            dj[j] = h
            curl[c] = (a(p + di)[j] - a(p - di)[j]) / (2 * h) - (a(p + dj)[i] - a(p - dj)[i]) / (2 * h)
        b = biot_savart(p[None, :], loop, 1.0)[0]
        self.assertLess(np.linalg.norm(curl - b) / np.linalg.norm(b), 1e-6)


class InducedFieldTests(unittest.TestCase):
    def test_solenoid_faraday_field(self) -> None:
        # Inside a long solenoid B_z = mu0 n I is uniform, so E_phi = -(r/2) dB/dt.
        radius, length, turns = 0.03, 0.4, 400
        n = turns / length
        sol = coils.solenoid(radius=radius, length=length, turns=turns, axis="z", n_segments=60)
        dIdt = 1000.0
        r = 0.01
        e = induced_efield(np.array([[r, 0.0, 0.0]]), sol, dIdt)[0]
        expected = -0.5 * r * MU0 * n * dIdt  # E_phi at (r,0,0) is E_y
        self.assertLess(abs(e[1] - expected) / abs(expected), 0.03)
        self.assertLess(abs(e[0]), abs(expected) * 0.05)  # radial component small


class InductanceTests(unittest.TestCase):
    def test_mutual_matches_maxwell_formula(self) -> None:
        from scipy.special import ellipe, ellipk

        a, b, d = 0.05, 0.04, 0.03
        la = circular_loop([0, 0, 0], a, axis="z", n_segments=400)
        lb = circular_loop([0, 0, d], b, axis="z", n_segments=400)
        m = mutual_inductance(la, lb)
        k2 = 4 * a * b / ((a + b) ** 2 + d**2)
        k = np.sqrt(k2)
        m_ref = MU0 * np.sqrt(a * b) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))
        self.assertLess(abs(m - m_ref) / abs(m_ref), 1e-3)

    def test_self_inductance_positive_and_grows_with_radius(self) -> None:
        l1 = self_inductance_circular(0.05, 1e-3)
        l2 = self_inductance_circular(0.10, 1e-3)
        self.assertGreater(l1, 0)
        self.assertGreater(l2, l1)


class EddyPowerTests(unittest.TestCase):
    def test_conducting_sphere_power(self) -> None:
        # Uniform B0 cos(wt) along z; azimuthal E amplitude 1/2 w B0 rho.
        a_s, sigma, w, b0 = 0.03, 1e6, 2 * np.pi * 1e3, 1e-3
        nx = 41
        ax = np.linspace(-a_s, a_s, nx)
        dx = ax[1] - ax[0]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        mask = (X**2 + Y**2 + Z**2) <= a_s**2
        e = np.zeros((nx, nx, nx, 3))
        e[..., 0] = 0.5 * w * b0 * Y
        e[..., 1] = -0.5 * w * b0 * X
        e[~mask] = 0.0
        p = eddy_power(e, sigma, dx**3, time_average=True)
        p_ref = np.pi * sigma * w**2 * b0**2 * a_s**5 / 15
        self.assertLess(abs(p - p_ref) / p_ref, 0.02)

    def test_charge_correction_screens_uniform_field_in_box(self) -> None:
        nx = 21
        ax = np.linspace(-0.02, 0.02, nx)
        dx = ax[1] - ax[0]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        box = (np.abs(X) <= 0.014) & (np.abs(Y) <= 0.014) & (np.abs(Z) <= 0.014)
        e = np.zeros((nx, nx, nx, 3))
        e[..., 0] = 1.0
        e[~box] = 0.0
        corrected = e - _charge_correction(e, box, (dx, dx, dx))
        corrected[~box] = 0.0
        # A uniform curl-free field is fully screened by surface charge.
        self.assertLess(eddy_power(corrected, 1.0, dx**3), 0.02 * eddy_power(e, 1.0, dx**3))

    def test_charge_correction_preserves_tangential_field(self) -> None:
        nx = 25
        ax = np.linspace(-0.02, 0.02, nx)
        dx = ax[1] - ax[0]
        X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
        cyl = (X**2 + Y**2 <= 0.015**2) & (np.abs(Z) <= 0.015)
        e = np.zeros((nx, nx, nx, 3))
        e[..., 0] = Y
        e[..., 1] = -X
        e[~cyl] = 0.0
        corrected = e - _charge_correction(e, cyl, (dx, dx, dx))
        corrected[~cyl] = 0.0
        ratio = eddy_power(corrected, 1.0, dx**3) / eddy_power(e, 1.0, dx**3)
        self.assertGreater(ratio, 0.9)  # azimuthal current already tangential

    def test_eddy_currents_end_to_end(self) -> None:
        sol = coils.solenoid(radius=0.05, length=0.3, turns=200, axis="z", n_segments=48)
        nx = 15
        ax = np.linspace(-0.02, 0.02, nx)
        dx = ax[1] - ax[0]
        grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
        mask = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2) <= 0.015
        res = eddy_currents(
            grid, sol, dcurrent_dt=1e4, conductivity=1e6, mask=mask,
            spacing=(dx, dx, dx), charge_correction=True,
        )
        self.assertGreater(res.power, 0.0)
        self.assertEqual(res.e_field.shape, grid.shape)
        self.assertTrue(np.all(res.e_field[~mask] == 0.0))


class CoilLoadingTests(unittest.TestCase):
    def _brine_setup(self):
        coil = coils.solenoid(radius=0.045, length=0.1, turns=8, axis="z", n_segments=36)
        nxy, nz = 21, 23
        axx = np.linspace(-0.12, 0.12, nxy)
        axz = np.linspace(-0.16, 0.16, nz)
        dxy = axx[1] - axx[0]
        dz = axz[1] - axz[0]
        X, Y, Z = np.meshgrid(axx, axx, axz, indexing="ij")
        rho = np.sqrt(X**2 + Y**2)
        brine = (rho >= 0.05) & (rho <= 0.11) & (np.abs(Z) <= 0.14)
        grid = np.stack([X, Y, Z], axis=-1)
        return coil, grid, brine, (dxy, dxy, dz)

    def test_reflected_resistance_scales_as_omega_squared(self) -> None:
        coil, grid, brine, spacing = self._brine_setup()
        kw = dict(conductivity=10.0, mask=brine, spacing=spacing)
        r1 = reflected_resistance(grid, coil, frequency=1e6, **kw)
        r2 = reflected_resistance(grid, coil, frequency=2e6, **kw)
        self.assertGreater(r1, 0.0)
        self.assertAlmostEqual(r2 / r1, 4.0, places=3)  # omega^2

    def test_reflected_resistance_linear_in_conductivity(self) -> None:
        coil, grid, brine, spacing = self._brine_setup()
        r_lo = reflected_resistance(grid, coil, frequency=1e6, conductivity=5.0, mask=brine, spacing=spacing)
        r_hi = reflected_resistance(grid, coil, frequency=1e6, conductivity=10.0, mask=brine, spacing=spacing)
        self.assertAlmostEqual(r_hi / r_lo, 2.0, places=6)

    def test_coil_loading_degrades_q_and_raises_noise(self) -> None:
        coil, grid, brine, spacing = self._brine_setup()
        radii = np.full(8, 0.045)
        centers = np.zeros((8, 3))
        centers[:, 2] = np.linspace(-0.05, 0.05, 8)
        inductance = coil_inductance(radii, centers, wire_radius=1e-3)
        cl = coil_loading(
            grid, coil, conductivity=10.0, mask=brine, spacing=spacing,
            frequencies=[0.5e6, 1e6, 2e6], inductance=inductance, coil_resistance=0.5,
        )
        self.assertTrue(np.all(cl.q_loaded < cl.q_unloaded))
        self.assertTrue(np.all(cl.noise_penalty > 1.0))
        # loading (hence noise penalty) worsens with frequency
        self.assertTrue(np.all(np.diff(cl.noise_penalty) > 0))


class CoilGeometryTests(unittest.TestCase):
    def test_solenoid_interior_field_uniform(self) -> None:
        sol = coils.solenoid(radius=0.03, length=0.5, turns=500, axis="z", n_segments=48)
        n = 500 / 0.5
        b_center = biot_savart(np.array([[0.0, 0.0, 0.0]]), sol, 1.0)[0]
        self.assertAlmostEqual(b_center[2], MU0 * n * 1.0, delta=0.05 * MU0 * n)

    def test_maxwell_pair_makes_odd_gradient(self) -> None:
        mp = coils.maxwell_pair(radius=0.04, axis="z", n_segments=72)
        bz_pos = biot_savart(np.array([[0.0, 0.0, 0.01]]), mp, 1.0)[0, 2]
        bz_neg = biot_savart(np.array([[0.0, 0.0, -0.01]]), mp, 1.0)[0, 2]
        bz_center = biot_savart(np.array([[0.0, 0.0, 0.0]]), mp, 1.0)[0, 2]
        self.assertAlmostEqual(bz_center, 0.0, places=9)  # null at centre
        self.assertAlmostEqual(bz_pos, -bz_neg, places=9)  # odd symmetry -> linear gradient
        self.assertGreater(abs(bz_pos), 1e-9)


class EddyModeTests(unittest.TestCase):
    def test_single_ring_time_constant(self) -> None:
        rho, r_w, a = 1.7e-8, 1e-3, 0.05
        em = EddyModes([a], np.array([[0, 0, 0.02]]), wire_radius=r_w, resistivity=rho, axis="z")
        drive = coils.maxwell_pair(radius=0.04, axis="z", n_segments=72)
        spec = em.spectrum(drive, sample_point=(0, 0, 0))
        tau_ref = self_inductance_circular(a, r_w) / (rho * 2 * a / r_w**2)
        self.assertAlmostEqual(spec.tau[0], tau_ref, delta=1e-4 * tau_ref)
        self.assertGreater(spec.alpha[0], 0.0)  # Lenz droop

    def test_shield_terms_feed_gradient_driver(self) -> None:
        radii, centers = coils.cylindrical_shield(radius=0.06, length=0.3, n_rings=10, axis="z")
        em = EddyModes(radii, centers, wire_radius=1e-3, resistivity=1.7e-8, axis="z")
        drive = coils.maxwell_pair(radius=0.04, axis="z", n_segments=72)
        terms = em.eddy_terms(drive, max_terms=4)
        self.assertTrue(all(a > 0 and t > 0 for a, t in terms))
        self.assertLess(sum(a for a, _ in terms), 1.0)
        gd = em.to_gradient_driver(drive, tau_rl=1e-4, max_terms=4)
        out = np.asarray(gd.apply(np.ones(4000), 1e-5, xp=np))
        self.assertAlmostEqual(out[-1], 1.0, places=3)  # settles to commanded
        self.assertLess(out[30], 1.0)  # droops during the eddy transient


if __name__ == "__main__":
    unittest.main()
