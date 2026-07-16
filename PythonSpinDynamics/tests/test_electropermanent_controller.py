from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.fields import illustrative_hybrid_epm_array
from spin_dynamics.workflows import (
    EPMTherapyControllerConfig,
    SuperparamagneticParticle,
    build_epm_nonlinear_encoding,
    estimate_particle_state_from_image,
    localize_epm_target,
    particle_distribution_image,
    random_epm_encoding_states,
    run_epm_image_guided_controller,
    run_epm_particle_imaging,
    simple_tissue_phantom,
)


class EPMTargetLocalizationTests(unittest.TestCase):
    def test_peak_relative_localization_returns_weighted_centroid(self) -> None:
        x = np.asarray((-2.0, 0.0, 2.0)) * 1e-3
        y = np.asarray((-1.0, 1.0)) * 1e-3
        image = np.asarray(((0.0, 2.0, 4.0), (0.0, 0.0, 3.0)))

        mask, center, threshold = localize_epm_target(
            image,
            x,
            y,
            threshold_fraction=0.70,
        )

        np.testing.assert_array_equal(
            mask,
            np.asarray(((False, False, True), (False, False, True))),
        )
        np.testing.assert_allclose(center, (2e-3, -1e-3 / 7.0))
        self.assertAlmostEqual(threshold, 2.8)


class EPMParticleStateEstimationTests(unittest.TestCase):
    def test_cloud_in_cell_image_preserves_count_and_centroid(self) -> None:
        axis = np.linspace(-2.0e-3, 2.0e-3, 5)
        positions = np.asarray(
            (
                (-1.5e-3, -0.5e-3),
                (-0.25e-3, 0.75e-3),
                (1.25e-3, 1.5e-3),
            )
        )

        image = particle_distribution_image(positions, axis, axis)
        x_grid, y_grid = np.meshgrid(axis, axis, indexing="xy")

        self.assertAlmostEqual(float(np.sum(image)), 3.0)
        self.assertAlmostEqual(float(np.sum(image * x_grid) / np.sum(image)), np.mean(positions[:, 0]))
        self.assertAlmostEqual(float(np.sum(image * y_grid) / np.sum(image)), np.mean(positions[:, 1]))

    def test_image_estimator_recovers_count_positions_centroid_and_occupancy(self) -> None:
        axis = np.linspace(-2.0e-3, 2.0e-3, 5)
        positions = np.asarray(((-2.0e-3, 0.0), (0.0, 0.0), (1.0e-3, 0.0)))
        image = particle_distribution_image(positions, axis, axis)

        estimate = estimate_particle_state_from_image(
            image,
            axis,
            axis,
            target_center_m=(0.0, 0.0),
            target_radius_m=0.25e-3,
            support_threshold_fraction=0.0,
        )

        self.assertEqual(estimate.particle_count, 3)
        self.assertEqual(estimate.positions_m.shape, (3, 2))
        np.testing.assert_allclose(estimate.centroid_m, np.mean(positions, axis=0))
        self.assertAlmostEqual(estimate.capture_fraction, 1.0 / 3.0)
        np.testing.assert_allclose(estimate.uncaptured_centroid_m, (-0.5e-3, 0.0))

    def test_boundary_capture_correction_uses_grid_geometry_not_particle_truth(self) -> None:
        axis = np.linspace(-2.0e-3, 2.0e-3, 5)
        angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        positions = 0.8e-3 * np.column_stack((np.cos(angles), np.sin(angles)))
        image = particle_distribution_image(positions, axis, axis)

        estimate = estimate_particle_state_from_image(
            image,
            axis,
            axis,
            target_center_m=(0.0, 0.0),
            target_radius_m=0.8e-3,
            support_threshold_fraction=0.0,
            boundary_capture_correction=True,
        )

        self.assertGreater(estimate.target_partial_volume_sensitivity, 0.0)
        self.assertLess(estimate.target_partial_volume_sensitivity, 1.0)
        self.assertAlmostEqual(estimate.capture_fraction, 1.0, places=8)


class EPMTherapyControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.phantom = simple_tissue_phantom(10, field_of_view_m=0.040)
        cls.basis = illustrative_hybrid_epm_array().build_field_basis(
            cls.phantom.points_m,
            n_cross=1,
            n_length=5,
        )
        states = random_epm_encoding_states(cls.basis, 160, seed=4)
        cls.encoding = build_epm_nonlinear_encoding(
            cls.basis,
            states,
            image_shape=cls.phantom.shape,
            phase_encoding_s=300e-6,
        )
        cls.image = cls.phantom.spin_echo_image(
            repetition_time_s=1.2,
            echo_time_s=0.040,
        )
        cls.particle = SuperparamagneticParticle(
            magnetic_core_radius_m=12e-6,
            hydrodynamic_radius_m=15e-6,
            volume_susceptibility=1.4,
            saturation_magnetization_a_m=4.5e5,
            fluid_viscosity_pa_s=1.5e-3,
            magnetic_volume_fraction=0.60,
        )
        cls.initial = np.column_stack(
            (
                np.full(16, -0.014),
                np.linspace(-2.0e-3, 2.0e-3, 16),
            )
        )

    def _run(self):
        return run_epm_image_guided_controller(
            self.encoding,
            self.image,
            self.phantom.x_m,
            self.phantom.y_m,
            self.particle,
            self.initial,
            config=EPMTherapyControllerConfig(
                max_cycles=2,
                capture_goal=0.99,
                imaging_window_s=10.0,
                programming_window_s=0.5,
                transport_window_s=2400.0,
                transport_time_step_s=10.0,
                target_radius_m=4.2e-3,
                transport_gradient_t_m=0.150,
                seed=12,
            ),
            background_velocity_m_s=(2.5e-6, 0.0),
        )

    def test_noisy_particle_imaging_is_reproducible_and_image_resolved(self) -> None:
        target_indices = np.argwhere(self.phantom.target_mask)
        target_center = np.asarray(
            (
                np.mean(self.phantom.x_m[target_indices[:, 1]]),
                np.mean(self.phantom.y_m[target_indices[:, 0]]),
            )
        )
        first = run_epm_particle_imaging(
            self.encoding,
            self.initial,
            self.phantom.x_m,
            self.phantom.y_m,
            target_center_m=target_center,
            target_radius_m=4.2e-3,
            snr_db=30.0,
            seed=21,
        )
        second = run_epm_particle_imaging(
            self.encoding,
            self.initial,
            self.phantom.x_m,
            self.phantom.y_m,
            target_center_m=target_center,
            target_radius_m=4.2e-3,
            snr_db=30.0,
            seed=21,
        )

        np.testing.assert_array_equal(
            first.imaging.reconstructed_image,
            second.imaging.reconstructed_image,
        )
        np.testing.assert_array_equal(first.estimate.positions_m, second.estimate.positions_m)
        self.assertLessEqual(abs(first.estimate.particle_count - self.initial.shape[0]), 2)
        self.assertLess(first.centroid_error_m, max(first.estimate.spatial_resolution_m))
        self.assertEqual(first.ground_truth_capture_fraction, 0.0)

    def test_controller_alternates_modes_reaims_and_accumulates_capture(self) -> None:
        result = self._run()

        self.assertEqual(len(result.cycles), 2)
        self.assertEqual(
            [interval.mode for interval in result.intervals],
            ["imaging", "programming", "transport", "imaging"] * 2,
        )
        for previous, current in zip(result.intervals, result.intervals[1:]):
            self.assertAlmostEqual(previous.end_s, current.start_s)
        self.assertTrue(np.all(np.diff(result.capture_fraction_by_cycle) >= 0.0))
        self.assertGreater(result.capture_fraction, 0.0)
        self.assertGreater(result.estimated_capture_fraction, 0.0)
        self.assertGreater(result.estimated_final_positions_m.shape[0], 0)
        self.assertGreater(result.total_remanence_variation_t, 0.0)
        for cycle in result.cycles:
            np.testing.assert_allclose(
                cycle.source_centroid_m,
                cycle.particle_imaging_before.estimate.uncaptured_centroid_m,
            )
            expected = cycle.target_center_m - cycle.source_centroid_m
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(cycle.requested_direction, expected)
            self.assertLess(
                cycle.source_centroid_error_m,
                2.0 * np.max(cycle.particle_imaging_before.estimate.spatial_resolution_m),
            )
            self.assertGreater(cycle.particle_imaging_before.estimate.particle_count, 0)
            self.assertGreater(cycle.particle_imaging_after.estimate.particle_count, 0)
            self.assertGreater(cycle.transport.peak_force_n, 0.0)
        expected_stop = (
            "capture_goal"
            if result.estimated_capture_fraction >= result.config.capture_goal
            else "max_cycles"
        )
        self.assertEqual(result.stop_reason, expected_stop)

    def test_seeded_controller_is_reproducible(self) -> None:
        first = self._run()
        second = self._run()

        np.testing.assert_array_equal(first.final_positions_m, second.final_positions_m)
        np.testing.assert_array_equal(first.final_captured, second.final_captured)
        np.testing.assert_array_equal(first.localized_targets_m, second.localized_targets_m)
        np.testing.assert_array_equal(
            first.estimated_final_positions_m,
            second.estimated_final_positions_m,
        )
        np.testing.assert_array_equal(
            first.estimated_capture_fraction_by_cycle,
            second.estimated_capture_fraction_by_cycle,
        )


if __name__ == "__main__":
    unittest.main()
