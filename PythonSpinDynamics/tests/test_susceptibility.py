from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.motion import MotionFieldMaps2D
from spin_dynamics.susceptibility import (
    PROTON_GAMMA,
    CylindricalInclusion,
    internal_gradient_distribution,
    internal_gradient_maps,
    make_susceptibility_field_maps,
    susceptibility_offresonance_map,
)


def _grid(n: int = 201, half: float = 100e-6):
    axis = np.linspace(-half, half, n)
    return axis, axis.copy()


class InclusionValidationTests(unittest.TestCase):
    def test_radius_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            CylindricalInclusion(0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            CylindricalInclusion(0.0, 0.0, -1e-6)

    def test_non_finite_center_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CylindricalInclusion(np.inf, 0.0, 1e-6)


class DipoleFieldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.a = 10e-6
        self.chi = 1e-6
        self.b0 = 1.0
        self.x, self.z = _grid(401)
        self.field = susceptibility_offresonance_map(
            self.x,
            self.z,
            [CylindricalInclusion(0.0, 0.0, self.a)],
            b0_tesla=self.b0,
            susceptibility_difference=self.chi,
            gamma=PROTON_GAMMA,
        )

    def test_external_field_matches_2d_dipole_on_axis(self) -> None:
        delta_b = self.field.offresonance_rad / PROTON_GAMMA
        iz = int(np.argmin(np.abs(self.z)))
        for radius in (20e-6, 40e-6, 80e-6):
            ix = int(np.argmin(np.abs(self.x - radius)))
            expected = self.b0 * 0.5 * self.chi * (self.a / radius) ** 2
            self.assertAlmostEqual(delta_b[ix, iz], expected, places=12)

    def test_quadrupolar_angular_dependence_flips_sign(self) -> None:
        delta_b = self.field.offresonance_rad / PROTON_GAMMA
        radius = 40e-6
        on_x = delta_b[int(np.argmin(np.abs(self.x - radius))), int(np.argmin(np.abs(self.z)))]
        on_z = delta_b[int(np.argmin(np.abs(self.x))), int(np.argmin(np.abs(self.z - radius)))]
        # cos(2 phi): +1 along B0 (x), -1 perpendicular (z)
        self.assertAlmostEqual(on_x, -on_z, places=12)
        self.assertGreater(on_x, 0.0)

    def test_field_decays_as_inverse_square(self) -> None:
        delta_b = self.field.offresonance_rad / PROTON_GAMMA
        iz = int(np.argmin(np.abs(self.z)))
        near = delta_b[int(np.argmin(np.abs(self.x - 20e-6))), iz]
        far = delta_b[int(np.argmin(np.abs(self.x - 40e-6))), iz]
        self.assertAlmostEqual(near / far, 4.0, places=6)

    def test_interior_uniform_fill_and_mask(self) -> None:
        delta_b = self.field.offresonance_rad / PROTON_GAMMA
        center = (int(np.argmin(np.abs(self.x))), int(np.argmin(np.abs(self.z))))
        self.assertTrue(self.field.inclusion_mask[center])
        self.assertAlmostEqual(delta_b[center], -self.b0 * 0.5 * self.chi, places=12)

    def test_in_plane_angle_rotates_pattern(self) -> None:
        rotated = susceptibility_offresonance_map(
            self.x,
            self.z,
            [CylindricalInclusion(0.0, 0.0, self.a)],
            b0_tesla=self.b0,
            susceptibility_difference=self.chi,
            gamma=PROTON_GAMMA,
            b0_in_plane_angle=np.pi / 2,
        )
        delta_b = rotated.offresonance_rad / PROTON_GAMMA
        radius = 40e-6
        # with B0 along z, the positive lobe now lies on the z axis
        on_z = delta_b[int(np.argmin(np.abs(self.x))), int(np.argmin(np.abs(self.z - radius)))]
        expected = self.b0 * 0.5 * self.chi * (self.a / radius) ** 2
        self.assertAlmostEqual(on_z, expected, places=12)


class SuperpositionTests(unittest.TestCase):
    def test_two_cylinders_superpose_linearly(self) -> None:
        x, z = _grid(301)
        a = 8e-6
        chi = 5e-7
        left = CylindricalInclusion(-40e-6, 0.0, a)
        right = CylindricalInclusion(40e-6, 0.0, a)
        single_left = susceptibility_offresonance_map(
            x, z, [left], b0_tesla=1.0, susceptibility_difference=chi
        ).offresonance_rad
        single_right = susceptibility_offresonance_map(
            x, z, [right], b0_tesla=1.0, susceptibility_difference=chi
        ).offresonance_rad
        both = susceptibility_offresonance_map(
            x, z, [left, right], b0_tesla=1.0, susceptibility_difference=chi
        )
        # evaluate at a point outside both inclusions
        ix = int(np.argmin(np.abs(x - 0.0)))
        iz = int(np.argmin(np.abs(z - 60e-6)))
        self.assertAlmostEqual(
            both.offresonance_rad[ix, iz],
            single_left[ix, iz] + single_right[ix, iz],
            places=10,
        )

    def test_per_inclusion_susceptibility_override(self) -> None:
        x, z = _grid(101)
        inc = CylindricalInclusion(0.0, 0.0, 10e-6, susceptibility_difference=2e-6)
        field = susceptibility_offresonance_map(
            x, z, [inc], b0_tesla=1.0, susceptibility_difference=0.0
        )
        # the override drives a non-zero field even though the default is zero
        self.assertGreater(float(np.max(np.abs(field.offresonance_rad))), 0.0)


class InternalGradientTests(unittest.TestCase):
    def test_gradient_magnitude_matches_radial_derivative(self) -> None:
        a, chi, b0 = 10e-6, 1e-6, 1.0
        x, z = _grid(401)
        field = susceptibility_offresonance_map(
            x, z, [CylindricalInclusion(0.0, 0.0, a)],
            b0_tesla=b0, susceptibility_difference=chi,
        )
        _, _, g_mag = internal_gradient_maps(field)
        iz = int(np.argmin(np.abs(z)))
        radius = 40e-6
        ix = int(np.argmin(np.abs(x - radius)))
        # |d/dr (chi/2)(a/r)^2 B0| = chi a^2 B0 / r^3 along the radial axis
        expected = chi * a**2 * b0 / radius**3
        self.assertAlmostEqual(g_mag[ix, iz] / expected, 1.0, places=2)

    def test_distribution_statistics_are_ordered(self) -> None:
        x, z = _grid(301)
        field = susceptibility_offresonance_map(
            x, z, [CylindricalInclusion(0.0, 0.0, 12e-6)],
            b0_tesla=1.0, susceptibility_difference=1e-6,
        )
        dist = internal_gradient_distribution(field, bins=32)
        self.assertGreater(dist.rms, 0.0)
        self.assertLessEqual(dist.mean, dist.rms + 1e-18)
        self.assertLessEqual(dist.rms, dist.maximum + 1e-18)
        self.assertEqual(dist.histogram.shape[0], 32)
        self.assertEqual(dist.bin_edges.shape[0], 33)

    def test_stronger_contrast_scales_gradient_linearly(self) -> None:
        x, z = _grid(301)
        weak = internal_gradient_distribution(
            susceptibility_offresonance_map(
                x, z, [CylindricalInclusion(0.0, 0.0, 12e-6)],
                b0_tesla=1.0, susceptibility_difference=1e-6,
            )
        )
        strong = internal_gradient_distribution(
            susceptibility_offresonance_map(
                x, z, [CylindricalInclusion(0.0, 0.0, 12e-6)],
                b0_tesla=1.0, susceptibility_difference=3e-6,
            )
        )
        self.assertAlmostEqual(strong.rms / weak.rms, 3.0, places=4)


class MotionMapsTests(unittest.TestCase):
    def test_make_motion_maps_returns_finite_b0(self) -> None:
        x, z = _grid(101)
        field = susceptibility_offresonance_map(
            x, z, [CylindricalInclusion(0.0, 0.0, 10e-6)],
            b0_tesla=1.0, susceptibility_difference=1e-6,
        )
        maps = make_susceptibility_field_maps(field)
        self.assertIsInstance(maps, MotionFieldMaps2D)
        self.assertTrue(np.all(np.isfinite(maps.b0_map)))
        np.testing.assert_allclose(maps.b0_map, field.offresonance_rad)

    def test_nan_interior_is_rejected_by_motion_maps(self) -> None:
        x, z = _grid(51)
        field = susceptibility_offresonance_map(
            x, z, [CylindricalInclusion(0.0, 0.0, 10e-6)],
            b0_tesla=1.0, susceptibility_difference=1e-6,
            interior_fill="nan",
        )
        with self.assertRaises(ValueError):
            make_susceptibility_field_maps(field)


if __name__ == "__main__":
    unittest.main()
