"""Tests for periodic stream-function winding extraction."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.gradient_coils import (  # noqa: E402
    CylindricalWindingSurface,
    build_cylindrical_gradient_system,
    linear_gradient_target,
    solve_gradient_coil,
    spherical_target_points,
)
from spin_dynamics.fields.gradient_windings import (  # noqa: E402
    extract_winding_contours,
    stream_function_contours,
    winding_contour_levels,
    winding_segments,
)
from spin_dynamics.fields.magnetostatics import biot_savart  # noqa: E402


class SyntheticContourTests(unittest.TestCase):
    def test_periodic_seam_is_stitched_into_closed_cylindrical_loops(self) -> None:
        surface = CylindricalWindingSurface(0.04, 0.10, 48, 20)
        z = np.linspace(-0.05, 0.05, 41)
        psi = np.cos(surface.phi)[:, np.newaxis] * (
            1.0 - (z[np.newaxis, :] / 0.05) ** 2
        )
        contours = stream_function_contours(
            surface,
            psi,
            levels_a=(-0.35, 0.35),
            z_coordinates=z,
        )
        self.assertEqual(len(contours), 2)
        self.assertTrue(all(contour.closed for contour in contours))
        self.assertTrue(any(np.ptp(contour.phi_z[:, 0]) > np.pi for contour in contours))
        for contour in contours:
            radial = np.linalg.norm(contour.points[:, :2], axis=1)
            np.testing.assert_allclose(radial, surface.radius, atol=1.0e-12)
            np.testing.assert_allclose(
                contour.points[0], contour.points[-1], atol=1.0e-12
            )

    def test_automatic_levels_are_half_step_centered(self) -> None:
        psi = np.array([[-2.1, 0.0, 2.1]])
        levels = winding_contour_levels(psi, current_per_turn_a=1.0)
        np.testing.assert_allclose(levels, [-1.5, -0.5, 0.5, 1.5])


class SolvedWindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.surface = CylindricalWindingSurface(0.05, 0.12, 24, 11)
        cls.points = spherical_target_points(0.018, points_per_axis=5)
        target = linear_gradient_target(cls.points, (0.0, 1.0e-3, 0.0))
        system = build_cylindrical_gradient_system(cls.surface, cls.points)
        cls.result = solve_gradient_coil(
            system,
            target,
            regularization=1.0e-16,
            solver="numpy",
        )

    def test_solved_stream_function_has_equal_boundary_values(self) -> None:
        np.testing.assert_allclose(
            self.result.stream_function_a[:, 0],
            self.result.stream_function_a[:, -1],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            -np.diff(self.result.stream_function_a, axis=1),
            self.result.segment_currents_a,
            atol=1.0e-12,
        )

    def test_solved_stream_function_produces_closed_windings(self) -> None:
        peak = float(np.max(np.abs(self.result.stream_function_a)))
        current_per_turn = peak / 4.0
        contours = extract_winding_contours(
            self.result,
            current_per_turn_a=current_per_turn,
        )
        self.assertGreaterEqual(len(contours), 4)
        self.assertTrue(all(contour.closed for contour in contours))
        segments = winding_segments(contours)
        self.assertEqual(
            len(segments),
            sum(contour.points.shape[0] - 1 for contour in contours),
        )
        winding_field = biot_savart(
            self.points,
            segments,
            current=current_per_turn,
        )[:, 2]
        correlation = float(
            np.corrcoef(winding_field, self.result.target_field_t)[0, 1]
        )
        self.assertGreater(correlation, 0.98)
        self.assertGreater(
            float(np.dot(winding_field, self.result.target_field_t)), 0.0
        )


if __name__ == "__main__":
    unittest.main()
