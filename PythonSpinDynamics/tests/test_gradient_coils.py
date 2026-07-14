"""Tests for cylindrical stream-function gradient-coil design."""

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
    solve_regularization_path,
    spherical_target_points,
)
from spin_dynamics.fields.magnetostatics import (  # noqa: E402
    biot_savart,
    segment_field_sensitivity,
)


class SegmentSensitivityTests(unittest.TestCase):
    def test_columns_match_individual_biot_savart_fields(self) -> None:
        surface = CylindricalWindingSurface(
            radius=0.04,
            length=0.08,
            n_phi=8,
            n_z=3,
        )
        segments = surface.segments()[2:7]
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.005, -0.004, 0.01],
                [-0.003, 0.006, -0.015],
            ]
        )
        sensitivity = segment_field_sensitivity(
            points,
            segments,
            direction=(0.0, 0.0, 1.0),
            chunk_size=2,
        )
        for index, segment in enumerate(segments):
            direct = biot_savart(points, [segment], current=1.0)[:, 2]
            np.testing.assert_allclose(
                sensitivity[:, index], direct, rtol=1.0e-13, atol=1.0e-18
            )

    def test_weighted_columns_reproduce_independent_segment_sum(self) -> None:
        surface = CylindricalWindingSurface(0.04, 0.08, 8, 3)
        segments = surface.segments()
        points = np.array([[0.002, -0.003, 0.0], [0.0, 0.004, 0.012]])
        currents = np.linspace(-0.8, 1.1, len(segments))
        sensitivity = segment_field_sensitivity(points, segments)
        expected = np.zeros(points.shape[0])
        for current, segment in zip(currents, segments, strict=True):
            expected += biot_savart(points, [segment], current=current)[:, 2]
        np.testing.assert_allclose(
            sensitivity @ currents, expected, rtol=2.0e-13, atol=1.0e-18
        )


class CylindricalGradientDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.surface = CylindricalWindingSurface(
            radius=0.05,
            length=0.12,
            n_phi=12,
            n_z=7,
        )
        cls.points = spherical_target_points(0.018, points_per_axis=5)
        cls.target = linear_gradient_target(
            cls.points,
            gradient=(0.0, 1.0e-3, 0.0),
        )
        cls.system = build_cylindrical_gradient_system(
            cls.surface,
            cls.points,
            chunk_size=7,
        )

    def test_mesh_matches_book_grid_convention(self) -> None:
        surface = CylindricalWindingSurface(0.4, 0.9, 56, 61)
        self.assertEqual(surface.shape, (56, 61))
        self.assertEqual(surface.segment_count, 3416)
        self.assertAlmostEqual(surface.z[1] - surface.z[0], 0.015)

    def test_numpy_solve_enforces_kcl_and_fits_linear_target(self) -> None:
        result = solve_gradient_coil(
            self.system,
            self.target,
            regularization=1.0e-18,
            solver="numpy",
        )
        self.assertLess(result.closure_error_a, 1.0e-12)
        self.assertLess(result.relative_rms_error, 0.08)
        np.testing.assert_allclose(
            -np.diff(result.stream_function_a, axis=1),
            result.segment_currents_a,
            rtol=1.0e-13,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            result.predicted_field_t,
            self.system.sensitivity @ result.segment_currents_a.ravel(),
            rtol=1.0e-13,
            atol=1.0e-18,
        )

    def test_solution_scales_with_target_amplitude(self) -> None:
        first = solve_gradient_coil(
            self.system,
            self.target,
            regularization=0.0,
            solver="numpy",
        )
        second = solve_gradient_coil(
            self.system,
            2.5 * self.target,
            regularization=0.0,
            solver="numpy",
        )
        np.testing.assert_allclose(
            second.segment_currents_a,
            2.5 * first.segment_currents_a,
            rtol=1.0e-9,
            atol=1.0e-10,
        )

    def test_regularization_reduces_current_norm(self) -> None:
        weak = solve_gradient_coil(
            self.system,
            self.target,
            regularization=1.0e-18,
            solver="numpy",
        )
        strong = solve_gradient_coil(
            self.system,
            self.target,
            regularization=1.0e-12,
            solver="numpy",
        )
        self.assertLess(strong.current_norm_a, weak.current_norm_a)
        self.assertGreater(strong.weighted_rms_error_t, weak.weighted_rms_error_t)

    def test_regularization_path_selects_an_interior_candidate(self) -> None:
        path = solve_regularization_path(
            self.system,
            self.target,
            np.logspace(-20, -10, 11),
            solver="numpy",
        )
        self.assertGreater(path.selected_index, 0)
        self.assertLess(path.selected_index, len(path.results) - 1)
        self.assertIs(path.selected_result, path.results[path.selected_index])
        self.assertTrue(np.all(np.diff(path.current_norms_a) <= 1.0e-8))

    def test_auto_solver_returns_finite_result(self) -> None:
        result = solve_gradient_coil(
            self.system,
            self.target,
            regularization=1.0e-18,
            solver="auto",
        )
        reference = solve_gradient_coil(
            self.system,
            self.target,
            regularization=1.0e-18,
            solver="numpy",
        )
        self.assertTrue(np.all(np.isfinite(result.segment_currents_a)))
        self.assertLess(result.closure_error_a, 1.0e-12)
        np.testing.assert_allclose(
            result.segment_currents_a,
            reference.segment_currents_a,
            rtol=5.0e-7,
            atol=1.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
