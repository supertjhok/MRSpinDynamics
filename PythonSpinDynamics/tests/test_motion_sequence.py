from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.motion import (
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import (
    MotionSequenceStep,
    run_motion_cpmg_sequence,
    run_motion_sequence,
)


class MotionSequenceTests(unittest.TestCase):
    def test_acquisition_samples_free_precession_interval(self) -> None:
        fields = make_motion_field_maps_2d(
            [0.0, 1.0],
            [0.0, 1.0],
            b0_map=2.0 * np.ones((2, 2), dtype=np.float64),
        )
        ensemble = initialize_ensemble_from_density(
            np.ones((1, 1), dtype=np.float64),
            [0.0],
            [0.0],
        )
        magnetization = ensemble.magnetization.copy()
        magnetization[1, :] = 1.0
        magnetization[2, :] = 1.0
        ensemble = ensemble.with_updates(magnetization=magnetization)

        result = run_motion_sequence(
            ensemble,
            fields,
            [
                MotionSequenceStep(
                    duration=1.0,
                    acquire=True,
                    num_samples=2,
                    substeps=4,
                    label="readout",
                )
            ],
        )

        np.testing.assert_allclose(result.sample_times, [0.5, 1.0])
        np.testing.assert_allclose(
            result.signal,
            np.exp(-1j * 2.0 * result.sample_times),
        )
        self.assertEqual(result.sample_labels, ("readout", "readout"))

    def test_sequence_substeps_motion_through_gradient(self) -> None:
        fields = make_motion_field_maps_2d([0.0, 1.0], [0.0, 1.0])
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

        result = run_motion_sequence(
            ensemble,
            fields,
            [
                MotionSequenceStep(
                    duration=1.0,
                    gradient=(1.0, 0.0),
                    acquire=True,
                    num_samples=1,
                    substeps=2,
                )
            ],
            velocity=np.array([1.0, 0.0], dtype=np.float64),
            boundary="clip",
        )

        np.testing.assert_allclose(result.final_ensemble.positions, [[1.0, 0.5]])
        np.testing.assert_allclose(result.signal[0], np.exp(-0.75j))

    def test_cpmg_sequence_refocuses_static_gradient_without_diffusion(
        self,
    ) -> None:
        x_axis = np.linspace(-0.5, 0.5, 21)
        z_axis = np.array([0.0], dtype=np.float64)
        rho = np.ones((x_axis.size, 1), dtype=np.float64)
        fields = make_motion_field_maps_2d([-0.5, 0.5], [0.0, 1.0])
        ensemble = initialize_ensemble_from_density(rho, x_axis, z_axis)
        ensemble = ensemble.with_updates(
            positions=np.column_stack((ensemble.positions[:, 0], np.full(21, 0.5)))
        )

        no_gradient = run_motion_cpmg_sequence(
            ensemble,
            fields,
            num_echoes=2,
            echo_spacing=0.08,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(0.0, 0.0),
            substeps_per_interval=8,
        )
        with_gradient = run_motion_cpmg_sequence(
            ensemble,
            fields,
            num_echoes=2,
            echo_spacing=0.08,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(30.0, 0.0),
            substeps_per_interval=8,
        )

        np.testing.assert_allclose(
            np.abs(with_gradient.signal),
            np.abs(no_gradient.signal),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_cpmg_sequence_diffusion_attenuates_static_gradient_echoes(
        self,
    ) -> None:
        x_axis = np.linspace(-0.5, 0.5, 31)
        z_axis = np.array([0.0], dtype=np.float64)
        rho = np.ones((x_axis.size, 1), dtype=np.float64)
        fields = make_motion_field_maps_2d([-0.8, 0.8], [0.0, 1.0])
        ensemble = initialize_ensemble_from_density(
            rho,
            x_axis,
            z_axis,
            walkers_per_cell=4,
            diffusion_coefficient=0.003,
            seed=123,
            jitter=True,
        )
        ensemble = ensemble.with_updates(
            positions=np.column_stack(
                (
                    ensemble.positions[:, 0],
                    np.full(ensemble.num_particles, 0.5),
                )
            )
        )

        stationary = ensemble.with_updates(
            positions=ensemble.positions.copy(),
        )
        stationary = stationary.__class__(
            positions=stationary.positions,
            magnetization=stationary.magnetization,
            weights=stationary.weights,
            diffusion_coefficient=np.zeros_like(stationary.diffusion_coefficient),
        )
        no_diffusion = run_motion_cpmg_sequence(
            stationary,
            fields,
            num_echoes=4,
            echo_spacing=0.08,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(35.0, 0.0),
            substeps_per_interval=6,
            rng=np.random.default_rng(99),
        )
        diffusing = run_motion_cpmg_sequence(
            ensemble,
            fields,
            num_echoes=4,
            echo_spacing=0.08,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(35.0, 0.0),
            substeps_per_interval=6,
            rng=np.random.default_rng(99),
        )

        self.assertLess(np.abs(diffusing.signal[-1]), np.abs(no_diffusion.signal[-1]))


if __name__ == "__main__":
    unittest.main()
