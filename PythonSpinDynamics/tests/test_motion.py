from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.motion import (
    apply_boundary,
    apply_free_precession,
    apply_rf_rotation,
    advect_diffuse_positions,
    free_precession_with_motion_step,
    initialize_ensemble_from_density,
    make_circular_reflector,
    make_motion_field_maps_2d,
    receive_signal,
    transverse_b1_magnitude,
)


class MotionTests(unittest.TestCase):
    def test_field_maps_bilinearly_sample_b0_and_b1(self) -> None:
        x_axis = np.array([0.0, 1.0], dtype=np.float64)
        z_axis = np.array([0.0, 2.0], dtype=np.float64)
        b0 = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float64)
        b1_tx = 2.0 * np.ones_like(b0)
        fields = make_motion_field_maps_2d(
            x_axis,
            z_axis,
            b0_map=b0,
            b1_tx_map=b1_tx,
        )

        sampled = fields.sample(np.array([[0.5, 1.0], [1.0, 2.0]], dtype=np.float64))

        np.testing.assert_allclose(sampled["b0"], [1.5, 3.0])
        np.testing.assert_allclose(sampled["b1_tx"], [2.0, 2.0])
        np.testing.assert_allclose(sampled["b1_rx"], [2.0, 2.0])

    def test_vector_b1_maps_are_projected_perpendicular_to_local_b0(self) -> None:
        b0_vector = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )
        b1_vector = np.array(
            [
                [[3.0, 4.0, 5.0], [2.0, 6.0, 0.0]],
                [[1.0, 7.0, 2.0], [3.0, 1.0, 4.0]],
            ],
            dtype=np.float64,
        )

        transverse = transverse_b1_magnitude(b0_vector, b1_vector)
        fields = make_motion_field_maps_2d(
            [0.0, 1.0],
            [0.0, 1.0],
            b0_vector_map=b0_vector,
            b1_tx_vector_map=b1_vector,
        )

        np.testing.assert_allclose(
            transverse,
            [
                [5.0, 6.0],
                [np.sqrt(5.0), np.sqrt(18.0)],
            ],
        )
        np.testing.assert_allclose(fields.b1_tx_map, transverse)
        np.testing.assert_allclose(fields.b1_rx_map, transverse)

    def test_density_initialization_preserves_total_weight(self) -> None:
        rho = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float64)
        ensemble = initialize_ensemble_from_density(
            rho,
            [0.0, 1.0],
            [0.0, 1.0],
            walkers_per_cell=3,
            diffusion_coefficient=2e-9,
        )

        self.assertEqual(ensemble.num_particles, 12)
        self.assertAlmostEqual(float(np.sum(ensemble.weights)), float(np.sum(rho)))
        np.testing.assert_allclose(ensemble.magnetization[0], 1.0)
        np.testing.assert_allclose(ensemble.magnetization[1:], 0.0)
        np.testing.assert_allclose(ensemble.diffusion_coefficient, 2e-9)

    def test_advection_diffusion_and_boundaries_are_seeded(self) -> None:
        positions = np.array([[0.2, 0.2], [0.9, 0.9]], dtype=np.float64)
        bounds = ((0.0, 1.0), (0.0, 1.0))

        advected = advect_diffuse_positions(
            positions,
            0.5,
            velocity=np.array([0.4, -0.2]),
            bounds=bounds,
            boundary="clip",
        )
        np.testing.assert_allclose(advected, [[0.4, 0.1], [1.0, 0.8]])

        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        noisy1 = advect_diffuse_positions(
            positions,
            0.25,
            diffusion_coefficient=0.01,
            rng=rng1,
            bounds=bounds,
            boundary="reflect",
        )
        noisy2 = advect_diffuse_positions(
            positions,
            0.25,
            diffusion_coefficient=0.01,
            rng=rng2,
            bounds=bounds,
            boundary="reflect",
        )
        np.testing.assert_allclose(noisy1, noisy2)
        self.assertTrue(np.all((noisy1 >= 0.0) & (noisy1 <= 1.0)))

        reflected = apply_boundary(
            np.array([[-0.1, 1.2], [2.3, -1.4]], dtype=np.float64),
            bounds,
            "reflect",
        )
        np.testing.assert_allclose(reflected, [[0.1, 0.8], [0.3, 0.6]])

    def test_circular_reflector_confines_walkers_to_the_disc(self) -> None:
        reflector = make_circular_reflector((0.0, 0.0), 2.0)

        # Interior points are untouched; exterior points fold back radially.
        inside = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)
        np.testing.assert_allclose(reflector(inside), inside)

        # A point at radius 2.5 along +x reflects to radius 1.5 (2*r - d).
        reflected = reflector(np.array([[2.5, 0.0], [0.0, -3.0]], dtype=np.float64))
        np.testing.assert_allclose(reflected, [[1.5, 0.0], [0.0, -1.0]])

        # A diffusing cloud driven through the reflector stays within the disc.
        rng = np.random.default_rng(7)
        positions = rng.uniform(-1.0, 1.0, size=(500, 2))
        for _ in range(40):
            positions = advect_diffuse_positions(
                positions,
                1.0,
                diffusion_coefficient=0.05,
                rng=rng,
                bounds=((-2.0, 2.0), (-2.0, 2.0)),
                boundary=reflector,
            )
        radii = np.hypot(positions[:, 0], positions[:, 1])
        self.assertTrue(np.all(radii <= 2.0 + 1e-9))

    def test_make_circular_reflector_rejects_nonpositive_radius(self) -> None:
        with self.assertRaises(ValueError):
            make_circular_reflector((0.0, 0.0), 0.0)

    def test_free_precession_matches_analytical_phase_and_relaxation(self) -> None:
        ensemble = initialize_ensemble_from_density(
            np.ones((1, 2), dtype=np.float64),
            [0.0],
            [0.0, 1.0],
        )
        magnetization = ensemble.magnetization.copy()
        magnetization[0, :] = 0.2
        magnetization[1, :] = 1.0 + 0j
        magnetization[2, :] = 1.0 + 0j
        ensemble = ensemble.with_updates(magnetization=magnetization)

        evolved = apply_free_precession(
            ensemble,
            0.25,
            off_resonance=np.array([1.0, -2.0]),
            t1=2.0,
            t2=1.5,
            mth=1.0,
        )

        e1 = np.exp(-0.25 / 2.0)
        e2 = np.exp(-0.25 / 1.5)
        np.testing.assert_allclose(evolved.magnetization[0], e1 * 0.2 + (1.0 - e1))
        np.testing.assert_allclose(
            evolved.magnetization[1],
            e2 * np.exp(-1j * np.array([1.0, -2.0]) * 0.25),
        )
        np.testing.assert_allclose(
            evolved.magnetization[2],
            e2 * np.exp(1j * np.array([1.0, -2.0]) * 0.25),
        )

    def test_rf_rotation_and_receive_signal_use_local_fields(self) -> None:
        ensemble = initialize_ensemble_from_density(
            np.array([[1.0, 2.0]], dtype=np.float64),
            [0.0],
            [0.0, 1.0],
        )
        rotated = apply_rf_rotation(
            ensemble,
            np.pi / 2,
            np.pi / 2,
            1.0,
            off_resonance=0.0,
        )
        fields = make_motion_field_maps_2d(
            [0.0, 1.0],
            [0.0, 1.0],
            b1_rx_map=np.array([[1.0, 0.5], [1.0, 0.5]], dtype=np.float64),
        )

        signal = receive_signal(rotated, fields)

        self.assertGreater(abs(signal), 0.0)
        self.assertLess(abs(signal), float(np.sum(ensemble.weights)) + 1e-12)

    def test_free_precession_with_motion_samples_new_position(self) -> None:
        fields = make_motion_field_maps_2d(
            [0.0, 1.0],
            [0.0, 1.0],
            b0_map=np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64),
        )
        ensemble = initialize_ensemble_from_density(
            np.ones((1, 1), dtype=np.float64),
            [0.0],
            [0.0],
        )
        magnetization = ensemble.magnetization.copy()
        magnetization[1, :] = 1.0
        magnetization[2, :] = 1.0
        ensemble = ensemble.with_updates(
            positions=np.array([[0.0, 0.5]], dtype=np.float64),
            magnetization=magnetization,
        )

        moved = free_precession_with_motion_step(
            ensemble,
            fields,
            0.5,
            velocity=np.array([1.0, 0.0]),
            t1=np.inf,
            t2=np.inf,
            boundary="clip",
        )

        np.testing.assert_allclose(moved.positions, [[0.5, 0.5]])
        np.testing.assert_allclose(moved.magnetization[1, 0], np.exp(-1j * 1.0 * 0.5))


if __name__ == "__main__":
    unittest.main()
