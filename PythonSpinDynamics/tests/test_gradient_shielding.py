"""Tests for active shielding, engineering metrics, and workflow adapters."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.eddy_modes import EddyModes  # noqa: E402
from spin_dynamics.fields.gradient_coils import (  # noqa: E402
    CylindricalWindingSurface,
    build_cylindrical_gradient_system,
    linear_gradient_target,
    solve_gradient_coil,
    spherical_target_points,
)
from spin_dynamics.fields.gradient_engineering import (  # noqa: E402
    gradient_coil_engineering_metrics,
    winding_imaging_field_map,
    winding_peec_conductors,
    winding_to_gradient_driver,
)
from spin_dynamics.fields.gradient_shielding import (  # noqa: E402
    build_actively_shielded_gradient_system,
    cylindrical_shield_points,
    solve_actively_shielded_gradient_coil,
    solve_actively_shielded_regularization_path,
    spherical_shell_points,
)
from spin_dynamics.fields.gradient_windings import (  # noqa: E402
    extract_actively_shielded_winding,
)
from spin_dynamics.fields.magnetostatics import GAMMA_PROTON  # noqa: E402


class ActivelyShieldedGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.primary = CylindricalWindingSurface(0.05, 0.12, 16, 9)
        cls.shield = CylindricalWindingSurface(0.065, 0.15, 16, 9)
        cls.target_points = spherical_target_points(0.015, points_per_axis=5)
        cls.shield_points = spherical_shell_points(0.085, n_points=64)
        cls.target = linear_gradient_target(
            cls.target_points,
            gradient=(0.0, 0.0, 1.0e-3),
        )
        cls.system = build_actively_shielded_gradient_system(
            cls.primary,
            cls.shield,
            cls.target_points,
            cls.shield_points,
        )
        cls.result = solve_actively_shielded_gradient_coil(
            cls.system,
            cls.target,
            regularization=1.0e-16,
            shield_weights=5.0,
            solver="numpy",
        )

    def test_joint_solve_closes_both_surfaces_and_suppresses_fringe_field(
        self,
    ) -> None:
        result = self.result
        self.assertLess(result.closure_error_a, 1.0e-11)
        self.assertLess(result.target_relative_rms_error, 0.01)
        np.testing.assert_allclose(
            -np.diff(result.primary_stream_function_a, axis=1),
            result.primary_segment_currents_a,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            -np.diff(result.shield_stream_function_a, axis=1),
            result.shield_segment_currents_a,
            atol=1.0e-12,
        )

        unshielded_system = build_cylindrical_gradient_system(
            self.primary,
            self.target_points,
        )
        unshielded = solve_gradient_coil(
            unshielded_system,
            self.target,
            regularization=1.0e-16,
            solver="numpy",
        )
        exterior_unshielded = (
            self.system.shield_sensitivity[:, : self.primary.segment_count]
            @ unshielded.segment_currents_a.ravel()
        )
        unshielded_rms = float(np.sqrt(np.mean(exterior_unshielded**2)))
        self.assertLess(result.shield_weighted_rms_field_t, unshielded_rms / 100.0)

    def test_shell_points_lie_on_requested_radius(self) -> None:
        points = spherical_shell_points(
            0.1,
            n_points=31,
            center=(0.01, -0.02, 0.03),
        )
        radii = np.linalg.norm(points - np.array([0.01, -0.02, 0.03]), axis=1)
        np.testing.assert_allclose(radii, 0.1, atol=1.0e-15)
        cylinder = cylindrical_shield_points(
            0.12,
            0.3,
            n_phi=12,
            n_z=7,
            center=(0.01, -0.02, 0.03),
        )
        centered = cylinder - np.array([0.01, -0.02, 0.03])
        np.testing.assert_allclose(
            np.linalg.norm(centered[:, :2], axis=1),
            0.12,
            atol=1.0e-15,
        )
        self.assertAlmostEqual(float(np.min(centered[:, 2])), -0.15)
        self.assertAlmostEqual(float(np.max(centered[:, 2])), 0.15)

    def test_regularization_path_reports_weighted_tradeoff(self) -> None:
        path = solve_actively_shielded_regularization_path(
            self.system,
            self.target,
            np.logspace(-19, -15, 5),
            shield_weights=5.0,
            surface_regularization_weights=(1.0, 2.0),
            solver="numpy",
        )
        self.assertEqual(len(path.results), 5)
        self.assertTrue(np.all(np.diff(path.weighted_current_norms_a) <= 1.0e-8))
        expected = np.sqrt(
            np.sum(path.selected_result.target_residual_t**2)
            + 5.0 * np.sum(path.selected_result.predicted_shield_field_t**2)
        )
        self.assertAlmostEqual(
            path.weighted_residual_norms_t[path.selected_index],
            expected,
        )


class GradientEngineeringWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not hasattr(ActivelyShieldedGradientTests, "result"):
            ActivelyShieldedGradientTests.setUpClass()
        design = ActivelyShieldedGradientTests.result
        turn_current = min(
            float(np.max(np.abs(design.primary_stream_function_a))),
            float(np.max(np.abs(design.shield_stream_function_a))),
        ) / 3.0
        cls.winding = extract_actively_shielded_winding(
            design,
            current_per_turn_a=turn_current,
        )

    def test_metrics_and_peec_adapter_preserve_realized_geometry(self) -> None:
        design = ActivelyShieldedGradientTests
        metrics = gradient_coil_engineering_metrics(
            self.winding,
            design.target_points,
            target_field_t=design.target,
            shield_points=design.shield_points,
            wire_radius=0.5e-3,
            background_field=(0.0, 0.0, 3.0),
        )
        contour_count = len(self.winding.primary.contours) + len(
            self.winding.shield.contours
        )
        self.assertEqual(metrics.electrical.contour_count, contour_count)
        self.assertGreater(metrics.electrical.wire_length_m, 0.0)
        self.assertGreater(metrics.electrical.dc_resistance_ohm, 0.0)
        self.assertGreater(metrics.electrical.estimated_inductance_h, 0.0)
        self.assertGreater(metrics.field.target_correlation, 0.98)
        self.assertTrue(np.all(np.isfinite(metrics.mechanical.net_force_n)))
        conductors = winding_peec_conductors(
            self.winding,
            wire_radius=0.5e-3,
            n_radial=1,
            n_angular=4,
        )
        self.assertEqual(len(conductors), contour_count)
        self.assertTrue(all(conductor.total_length > 0.0 for conductor in conductors))

    def test_imaging_map_uses_x_z_convention_and_motion_container(self) -> None:
        axes = (np.linspace(-0.01, 0.01, 4), np.linspace(-0.015, 0.015, 5))
        field_map = winding_imaging_field_map(self.winding, axes)
        self.assertEqual(field_map.cartesian_axes, (0, 2))
        self.assertEqual(field_map.projected_field_t.shape, (4, 5))
        np.testing.assert_allclose(
            field_map.angular_offset_rad_s,
            GAMMA_PROTON * field_map.projected_field_t,
        )
        motion_maps = field_map.to_motion_field_maps()
        self.assertEqual(motion_maps.domain.shape, (4, 5))
        np.testing.assert_allclose(
            motion_maps.b0_map,
            field_map.angular_offset_rad_s,
        )

    def test_eddy_mode_adapter_builds_existing_gradient_driver(self) -> None:
        modes = EddyModes(
            [0.075, 0.075],
            np.array([[0.0, 0.0, -0.03], [0.0, 0.0, 0.03]]),
            wire_radius=1.0e-3,
            resistivity=1.7e-8,
            axis="z",
            n_segments=24,
        )
        driver = winding_to_gradient_driver(
            self.winding,
            modes,
            tau_rl=1.0e-3,
            min_alpha=0.0,
            max_terms=2,
        )
        self.assertAlmostEqual(driver.tau_rl, 1.0e-3)
        self.assertEqual(len(driver.eddy_terms), 2)
        self.assertTrue(all(np.isfinite(term) for pair in driver.eddy_terms for term in pair))


if __name__ == "__main__":
    unittest.main()
