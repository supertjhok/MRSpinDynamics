from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.workflows import (
    SuperparamagneticParticle,
    magnetic_force_from_gradient,
    magnetic_force_map_2d,
    simulate_magnetophoretic_transport,
)
from spin_dynamics.workflows.electropermanent_transport import BOLTZMANN, MU0


class ParticlePhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.particle = SuperparamagneticParticle(
            magnetic_core_radius_m=100e-9,
            hydrodynamic_radius_m=140e-9,
            volume_susceptibility=1.2,
            saturation_magnetization_a_m=4.5e5,
            fluid_viscosity_pa_s=1.5e-3,
            temperature_k=310.0,
            magnetic_volume_fraction=0.65,
        )

    def test_stokes_einstein_diffusion_and_drag_are_consistent(self) -> None:
        expected_drag = 6.0 * np.pi * 1.5e-3 * 140e-9
        self.assertAlmostEqual(self.particle.drag_coefficient_n_s_m, expected_drag)
        self.assertAlmostEqual(
            self.particle.diffusion_coefficient_m2_s,
            BOLTZMANN * 310.0 / expected_drag,
        )

    def test_langevin_force_recovers_linear_low_field_and_saturates(self) -> None:
        field = np.asarray([1e-8, 2.0])
        gradient = np.asarray([[2e-6, -3e-6], [2e-6, -3e-6]])
        linear = magnetic_force_from_gradient(
            self.particle,
            field,
            gradient,
            model="linear",
        )
        langevin = magnetic_force_from_gradient(
            self.particle,
            field,
            gradient,
            model="langevin",
        )

        np.testing.assert_allclose(langevin[0], linear[0], rtol=1e-10)
        magnetization = self.particle.magnetization_a_m(field)
        self.assertLessEqual(float(magnetization[-1]), self.particle.saturation_magnetization_a_m)
        self.assertLess(float(np.linalg.norm(langevin[-1])), float(np.linalg.norm(linear[-1])))

    def test_linear_force_matches_closed_form_identity(self) -> None:
        gradient = np.asarray([[0.5, -0.25]])
        force = magnetic_force_from_gradient(
            self.particle,
            np.asarray([0.01]),
            gradient,
            model="linear",
        )
        expected = (
            self.particle.magnetic_volume_m3
            * self.particle.volume_susceptibility
            * gradient
            / (2.0 * MU0)
        )
        np.testing.assert_allclose(force, expected, rtol=1e-14)


class MagnetophoreticTransportTests(unittest.TestCase):
    @staticmethod
    def _force_map():
        x = np.linspace(-1e-3, 1e-3, 21)
        y = np.linspace(-1e-3, 1e-3, 19)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        field = np.zeros(xx.shape + (3,))
        field[..., 2] = 0.020 + 2.0 * xx - 0.5 * yy
        return magnetic_force_map_2d(x, y, field)

    def test_force_map_recovers_affine_field_gradient(self) -> None:
        force_map = self._force_map()
        xx, yy = np.meshgrid(force_map.x_m, force_map.y_m, indexing="xy")
        bz = 0.020 + 2.0 * xx - 0.5 * yy
        expected_x = 4.0 * bz
        expected_y = -1.0 * bz
        np.testing.assert_allclose(
            force_map.grad_b_squared_t2_m[..., 0],
            expected_x,
            rtol=2e-13,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            force_map.grad_b_squared_t2_m[..., 1],
            expected_y,
            rtol=2e-13,
            atol=2e-15,
        )

    def test_seeded_transport_is_reproducible_and_captures_target(self) -> None:
        particle = SuperparamagneticParticle(
            magnetic_core_radius_m=4e-6,
            hydrodynamic_radius_m=5e-6,
            volume_susceptibility=1.0,
            saturation_magnetization_a_m=4e5,
            fluid_viscosity_pa_s=1e-3,
        )
        initial = np.column_stack(
            (np.full(12, -0.8e-3), np.linspace(0.05e-3, 0.35e-3, 12))
        )
        kwargs = dict(
            duration_s=30.0,
            time_step_s=0.25,
            target_center_m=(0.35e-3, 0.0),
            target_radius_m=0.22e-3,
            background_velocity_m_s=(20e-6, 0.0),
            seed=9,
        )
        first = simulate_magnetophoretic_transport(
            self._force_map(), particle, initial, **kwargs
        )
        second = simulate_magnetophoretic_transport(
            self._force_map(), particle, initial, **kwargs
        )

        np.testing.assert_array_equal(first.positions_m, second.positions_m)
        self.assertGreater(first.capture_fraction, 0.0)
        self.assertGreater(first.peak_force_n, 0.0)
        self.assertTrue(np.all(np.diff(first.cumulative_capture_fraction) >= 0.0))

    def test_target_capture_detects_a_crossing_between_time_samples(self) -> None:
        particle = SuperparamagneticParticle(
            magnetic_core_radius_m=100e-9,
            hydrodynamic_radius_m=150e-9,
            volume_susceptibility=0.0,
            saturation_magnetization_a_m=4e5,
            fluid_viscosity_pa_s=1e-3,
            temperature_k=1e-9,
        )
        result = simulate_magnetophoretic_transport(
            self._force_map(),
            particle,
            np.asarray([[-0.8e-3, 0.0]]),
            duration_s=1.0,
            time_step_s=1.0,
            target_center_m=(0.0, 0.0),
            target_radius_m=0.1e-3,
            background_velocity_m_s=(1.6e-3, 0.0),
            seed=1,
        )

        self.assertTrue(result.captured[0])
        self.assertAlmostEqual(result.capture_time_s[0], 0.4375, places=6)
        self.assertAlmostEqual(result.positions_m[-1, 0, 0], -0.1e-3, places=12)

    def test_initial_capture_mask_keeps_particles_immobilized(self) -> None:
        particle = SuperparamagneticParticle(
            magnetic_core_radius_m=2e-6,
            hydrodynamic_radius_m=3e-6,
            volume_susceptibility=1.0,
            saturation_magnetization_a_m=4e5,
        )
        initial = np.asarray(((-0.7e-3, 0.4e-3), (-0.7e-3, -0.4e-3)))
        result = simulate_magnetophoretic_transport(
            self._force_map(),
            particle,
            initial,
            duration_s=2.0,
            time_step_s=0.5,
            target_center_m=(0.8e-3, 0.0),
            target_radius_m=0.05e-3,
            background_velocity_m_s=(100e-6, 0.0),
            initial_captured=(True, False),
            seed=3,
        )

        np.testing.assert_array_equal(
            result.positions_m[:, 0],
            np.repeat(initial[[0]], result.time_s.size, axis=0),
        )
        self.assertTrue(result.captured[0])
        self.assertGreater(result.positions_m[-1, 1, 0], initial[1, 0])


if __name__ == "__main__":
    unittest.main()
