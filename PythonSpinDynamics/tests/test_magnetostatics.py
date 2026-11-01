from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.magnetostatics import (
    MU0,
    BarMagnet,
    bar_array_b0,
    biot_savart,
    circular_loop,
    finite_magnet_array_b0,
    halbach_dipole_magnets,
    nmr_mouse_magnets,
    sample_halbach_dipole_field,
    sample_magnet_field,
)


class BiotSavartTests(unittest.TestCase):
    def test_loop_center_matches_analytic(self):
        radius, current = 0.01, 1.0
        loop = circular_loop((0, 0, 0), radius, axis="z", n_segments=200)
        B = biot_savart(np.array([[0.0, 0.0, 0.0]]), loop, current)[0]
        expected = MU0 * current / (2 * radius)  # along +z (right-hand rule)
        self.assertAlmostEqual(B[2], expected, delta=1e-3 * expected)
        self.assertLess(abs(B[0]), 1e-12)
        self.assertLess(abs(B[1]), 1e-12)

    def test_field_scales_with_current(self):
        loop = circular_loop((0, 0, 0), 0.01, axis="z", n_segments=64)
        p = np.array([[0.0, 0.0, 0.005]])
        b1 = biot_savart(p, loop, 1.0)
        b3 = biot_savart(p, loop, 3.0)
        np.testing.assert_allclose(b3, 3.0 * b1, rtol=1e-9, atol=1e-15)

    def test_points_shape_validated(self):
        with self.assertRaises(ValueError):
            biot_savart(np.zeros((4, 2)), [], 1.0)


class CoilHandednessTests(unittest.TestCase):
    # Every generator winds right-handed about +axis. Pre-fix, axis="y" was
    # left-handed: the loop-centre field was -mu0 I/(2R) along y and the y Maxwell
    # pair gradient was -3.2e-4 T/m/A while x and z gave +3.2e-4.

    def test_loop_centre_field_points_along_plus_axis(self):
        from spin_dynamics.fields import coils

        radius = 0.01
        expected = MU0 / (2.0 * radius)
        for index, axis in enumerate("xyz"):
            for segments in (
                circular_loop((0, 0, 0), radius, axis=axis, n_segments=400),
                coils.conducting_ring(radius=radius, axis=axis, n_segments=400),
            ):
                b = biot_savart(np.zeros((1, 3)), segments, 1.0)[0]
                self.assertAlmostEqual(b[index] / expected, 1.0, delta=1e-4, msg=axis)

    def test_solenoid_centre_field_matches_finite_solenoid_formula(self):
        from spin_dynamics.fields import coils

        radius, length, turns = 0.01, 0.1, 50
        pitch = length / (turns - 1)  # loops span centre-to-centre `length`
        sheet = turns / (length + pitch)
        span = length + pitch
        expected = MU0 * sheet * span / np.sqrt(span**2 + 4.0 * radius**2)
        for index, axis in enumerate("xyz"):
            segments = coils.solenoid(
                radius=radius, length=length, turns=turns, axis=axis, n_segments=96
            )
            b = biot_savart(np.zeros((1, 3)), segments, 1.0)[0]
            self.assertAlmostEqual(b[index] / expected, 1.0, delta=0.01, msg=axis)

    def test_maxwell_pair_gradient_matches_closed_form_for_every_axis(self):
        from spin_dynamics.fields import coils

        radius = 0.05
        half = 0.5 * np.sqrt(3.0) * radius
        expected = 3.0 * MU0 * radius**2 * half / (radius**2 + half**2) ** 2.5
        for index, axis in enumerate("xyz"):
            segments = coils.maxwell_pair(radius=radius, axis=axis, n_segments=144)
            points = np.zeros((2, 3))
            points[0, index], points[1, index] = 1e-4, -1e-4
            b = biot_savart(points, segments, 1.0)
            gradient = (b[0, index] - b[1, index]) / 2e-4
            self.assertAlmostEqual(gradient / expected, 1.0, delta=0.005, msg=axis)

    def test_planar_spiral_moment_points_along_plus_axis(self):
        from spin_dynamics.fields import coils

        for index, axis in enumerate("xyz"):
            segments = coils.planar_spiral(
                r_inner=0.002, r_outer=0.01, turns=4, axis=axis, n_points=2000
            )
            # Area vector 1/2 sum(r x dl) of the open spiral is along the winding axis.
            area = 0.5 * sum(np.cross(start, end - start) for start, end in segments)
            self.assertGreater(area[index], 0.0, msg=axis)
            b = biot_savart(np.zeros((1, 3)), segments, 1.0)[0]
            self.assertGreater(b[index], 0.0, msg=axis)


class YokeImageTests(unittest.TestCase):
    def setUp(self):
        # A bar floating above the yoke plane at y=0.
        self.bar = BarMagnet(x0=-0.01, x1=0.01, y0=0.002, y1=0.022, br_y=1.3)
        self.xs = np.linspace(-0.04, 0.04, 41)

    def test_tangential_field_vanishes_on_iron_plane(self):
        bx, _ = bar_array_b0(self.xs, np.zeros_like(self.xs), [self.bar], yoke_y=0.0)
        self.assertLess(np.max(np.abs(bx)), 1e-9)

    def test_no_yoke_has_tangential_field(self):
        bx, _ = bar_array_b0(self.xs, np.zeros_like(self.xs), [self.bar], yoke_y=None)
        self.assertGreater(np.max(np.abs(bx)), 1e-3)

    def test_oblique_bar_on_yoke_matches_explicit_image_bar(self):
        # A mu -> inf plane at y = 0 is equivalent to the mirror bar at -y with the
        # tangential (x) magnetization reversed and the normal (y) one kept. Pre-fix
        # the x-magnetized image had the wrong sign: |Bx| on the iron reached
        # 0.43 T and Bx at (0, 30 mm) was -0.074 T instead of -0.030 T.
        bar = BarMagnet(x0=-0.01, x1=0.01, y0=0.002, y1=0.012, br_x=1.0, br_y=0.6)
        image = BarMagnet(x0=-0.01, x1=0.01, y0=-0.012, y1=-0.002, br_x=-1.0, br_y=0.6)
        xs = np.linspace(-0.05, 0.05, 21)
        for y in (0.015, 0.03, 0.06):
            ys = np.full_like(xs, y)
            got = np.array(bar_array_b0(xs, ys, [bar], yoke_y=0.0))
            expected = np.array(bar_array_b0(xs, ys, [bar, image]))
            np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-12)
        bx_surface, _ = bar_array_b0(xs, np.zeros_like(xs), [bar], yoke_y=0.0)
        self.assertLess(np.max(np.abs(bx_surface)), 1e-9)

    def test_yoke_image_for_nonzero_plane_height(self):
        bar = BarMagnet(x0=0.0, x1=0.01, y0=0.012, y1=0.02, br_x=0.8)
        image = BarMagnet(x0=0.0, x1=0.01, y0=0.0, y1=0.008, br_x=-0.8)
        xs = np.linspace(-0.03, 0.04, 15)
        bx, _ = bar_array_b0(xs, np.full_like(xs, 0.01), [bar], yoke_y=0.01)
        self.assertLess(np.max(np.abs(bx)), 1e-9)
        ys = np.full_like(xs, 0.035)
        np.testing.assert_allclose(
            np.array(bar_array_b0(xs, ys, [bar], yoke_y=0.01)),
            np.array(bar_array_b0(xs, ys, [bar, image])),
            rtol=0.0,
            atol=1e-12,
        )

    def test_mouse_preset_matches_explicit_image_bars(self):
        # The NMR-MOUSE bars are y-magnetized, whose images were already correct.
        bars, yoke = nmr_mouse_magnets()
        images = [
            BarMagnet(x0=b.x0, x1=b.x1, y0=2 * yoke - b.y1, y1=2 * yoke - b.y0, br_y=b.br_y)
            for b in bars
        ]
        xs = np.linspace(-0.02, 0.02, 21)
        ys = np.full_like(xs, 0.03)
        np.testing.assert_allclose(
            np.array(bar_array_b0(xs, ys, bars, yoke_y=yoke)),
            np.array(bar_array_b0(xs, ys, list(bars) + images)),
            rtol=0.0,
            atol=1e-12,
        )

    def test_yoke_enhances_field_above_magnet(self):
        pt_x, pt_y = [0.0], [0.030]
        b_yoke = np.hypot(*bar_array_b0(pt_x, pt_y, [self.bar], yoke_y=0.0))[0]
        b_free = np.hypot(*bar_array_b0(pt_x, pt_y, [self.bar], yoke_y=None))[0]
        self.assertGreater(b_yoke, b_free)


class MouseFieldTests(unittest.TestCase):
    def test_field_magnitude_and_gradient_in_mouse_regime(self):
        bars, yoke = nmr_mouse_magnets(
            magnet_width=0.02, magnet_height=0.02, gap=0.012, remanence=1.30
        )
        x_axis = np.linspace(-0.02, 0.02, 61)
        y_axis = np.linspace(0.021, 0.045, 61)  # sample region above the bars
        fm = sample_magnet_field(x_axis, y_axis, bars, yoke_y=yoke)
        # Single-sided fields: a few hundred mT, gradient tens of T/m.
        self.assertTrue(0.05 < fm.b0_magnitude.max() < 1.0)
        self.assertTrue(5.0 < np.median(fm.b0_gradient) < 60.0)
        # B0 decays with depth on the gap axis.
        ix0 = int(np.argmin(np.abs(x_axis)))
        col = fm.b0_magnitude[ix0, :]
        self.assertLess(col[-1], col[0])
        self.assertEqual(fm.b0_vector.shape, (61, 61, 3))
        self.assertIsNone(fm.b1_transverse)

    def test_coil_adds_transverse_b1(self):
        bars, yoke = nmr_mouse_magnets()
        x_axis = np.linspace(-0.02, 0.02, 31)
        y_axis = np.linspace(0.022, 0.045, 31)
        coil = circular_loop((0.0, 0.033, 0.0), 0.008, axis="y", n_segments=48)
        fm = sample_magnet_field(x_axis, y_axis, bars, yoke_y=yoke,
                                 coil_segments=coil, coil_current=1.0)
        self.assertIsNotNone(fm.b1_transverse)
        self.assertEqual(fm.b1_transverse.shape, (31, 31))
        self.assertGreater(fm.b1_transverse.max(), 0.0)
        # transverse component cannot exceed the full |B1|
        self.assertLessEqual(
            fm.b1_transverse.max(),
            np.linalg.norm(fm.b1_vector, axis=-1).max() + 1e-12,
        )


class HalbachDipoleTests(unittest.TestCase):
    def test_four_rod_halbach_phasing_points_bore_field_along_x(self):
        rods = halbach_dipole_magnets(
            center_radius=0.025,
            rod_radius=0.006,
            length=0.06,
            remanence=1.2,
        )
        self.assertEqual(len(rods), 4)
        np.testing.assert_allclose(rods[0].center, (0.025, 0.0, 0.0), atol=1e-15)
        np.testing.assert_allclose(rods[1].center, (0.0, 0.025, 0.0), atol=1e-15)
        # M(phi) = Br [cos(2 phi), sin(2 phi), 0] for a +x bore field.
        self.assertGreater(rods[0].br[0], 0.0)
        self.assertLess(rods[1].br[0], 0.0)
        self.assertGreater(rods[2].br[0], 0.0)
        self.assertLess(rods[3].br[0], 0.0)

        b0 = finite_magnet_array_b0(
            np.array([[0.0, 0.0, 0.0]]),
            rods,
            n_cross=5,
            n_length=9,
        )[0]
        self.assertGreater(b0[0], 0.0)
        self.assertLess(abs(b0[1]), 1e-12)
        self.assertLess(abs(b0[2]), 1e-12)

    def test_field_angle_rotates_bore_field(self):
        rods = halbach_dipole_magnets(
            center_radius=0.025,
            rod_radius=0.006,
            length=0.06,
            remanence=1.2,
            field_angle=0.5 * np.pi,
        )
        b0 = finite_magnet_array_b0(
            np.array([[0.0, 0.0, 0.0]]),
            rods,
            n_cross=5,
            n_length=9,
        )[0]
        self.assertGreater(b0[1], 0.0)
        self.assertLess(abs(b0[0]), 1e-12)
        self.assertLess(abs(b0[2]), 1e-12)

    def test_sample_halbach_field_has_finite_length_end_falloff(self):
        axis = np.linspace(-0.005, 0.005, 5)
        z_axis = np.linspace(-0.045, 0.045, 7)
        fm = sample_halbach_dipole_field(
            axis,
            axis,
            z_axis,
            center_radius=0.025,
            rod_radius=0.006,
            length=0.06,
            remanence=1.2,
            n_cross=5,
            n_length=11,
        )
        self.assertEqual(fm.b0_vector.shape, (5, 5, 7, 3))
        self.assertEqual(fm.b0_gradient.shape, (5, 5, 7))
        ix0 = int(np.argmin(np.abs(axis)))
        iz0 = int(np.argmin(np.abs(z_axis)))
        center = fm.b0_magnitude[ix0, ix0, iz0]
        end = fm.b0_magnitude[ix0, ix0, -1]
        self.assertGreater(center, end)
        self.assertTrue(0.01 < center < 1.0)

    def test_square_rods_are_supported(self):
        rods = halbach_dipole_magnets(
            center_radius=0.025,
            rod_shape="square",
            rod_width=0.010,
            length=0.06,
            remanence=1.2,
        )
        self.assertTrue(all(rod.shape == "square" for rod in rods))
        b0 = finite_magnet_array_b0(
            np.array([[0.0, 0.0, 0.0], [0.002, -0.001, 0.003]]),
            rods,
            n_cross=4,
            n_length=7,
        )
        self.assertEqual(b0.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(b0)))
        self.assertGreater(b0[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
