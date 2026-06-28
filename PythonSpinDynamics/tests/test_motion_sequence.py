from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.motion import (
    ParticleEnsemble,
    initialize_ensemble_from_density,
    make_motion_field_maps,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import (
    MotionSequenceStep,
    make_motion_udd_sequence,
    run_motion_cpmg_sequence,
    run_motion_sequence,
    run_motion_udd_sequence,
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

    def test_sequence_accepts_time_dependent_detuning_waveform(self) -> None:
        fields = make_motion_field_maps_2d([0.0, 1.0], [0.0, 1.0])
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
                    num_samples=1,
                    substeps=8,
                )
            ],
            detuning_waveform=lambda time: 2.0 * time,
        )

        # The midpoint-rule substeps integrate int_0^1 2t dt exactly.
        np.testing.assert_allclose(result.signal[0], np.exp(-1j), atol=1e-12)

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

    def test_cpmg_sequence_accepts_three_dimensional_fields(self) -> None:
        axes = (
            np.linspace(-0.5, 0.5, 3),
            np.linspace(-0.5, 0.5, 3),
            np.linspace(-0.5, 0.5, 3),
        )
        fields = make_motion_field_maps(axes)
        positions = np.array(
            [[-0.2, 0.0, 0.1], [0.2, 0.0, -0.1]],
            dtype=np.float64,
        )
        magnetization = np.zeros((3, positions.shape[0]), dtype=np.complex128)
        magnetization[0, :] = 1.0
        ensemble = ParticleEnsemble(
            positions=positions,
            magnetization=magnetization,
            weights=np.full(positions.shape[0], 0.5, dtype=np.float64),
            diffusion_coefficient=np.zeros(positions.shape[0], dtype=np.float64),
        )

        result = run_motion_cpmg_sequence(
            ensemble,
            fields,
            num_echoes=2,
            echo_spacing=0.08,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(0.0, 0.0, 0.0),
            substeps_per_interval=2,
        )

        self.assertEqual(result.signal.shape, (2,))
        self.assertEqual(result.final_ensemble.positions.shape, (2, 3))

    def test_udd_sequence_uses_uhrig_pulse_centers(self) -> None:
        steps = make_motion_udd_sequence(
            2,
            1.0,
            excitation_duration=0.01,
            refocusing_duration=0.1,
            substeps_per_interval=2,
        )

        self.assertEqual(
            tuple(step.label for step in steps),
            (
                "excitation_90",
                "udd_1_pre",
                "udd_1_180",
                "udd_2_pre",
                "udd_2_180",
                "udd_echo",
            ),
        )
        np.testing.assert_allclose(
            [step.duration for step in steps],
            [0.01, 0.2, 0.1, 0.4, 0.1, 0.2],
        )
        self.assertTrue(steps[-1].acquire)

    def test_udd_sequence_refocuses_static_gradient_without_diffusion(self) -> None:
        x_axis = np.linspace(-0.5, 0.5, 21)
        z_axis = np.array([0.0], dtype=np.float64)
        rho = np.ones((x_axis.size, 1), dtype=np.float64)
        fields = make_motion_field_maps_2d([-0.5, 0.5], [0.0, 1.0])
        ensemble = initialize_ensemble_from_density(rho, x_axis, z_axis)
        ensemble = ensemble.with_updates(
            positions=np.column_stack((ensemble.positions[:, 0], np.full(21, 0.5)))
        )

        no_gradient = run_motion_udd_sequence(
            ensemble,
            fields,
            num_pulses=3,
            total_duration=0.24,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(0.0, 0.0),
            substeps_per_interval=8,
        )
        with_gradient = run_motion_udd_sequence(
            ensemble,
            fields,
            num_pulses=3,
            total_duration=0.24,
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
        np.testing.assert_allclose(with_gradient.sample_times, [0.242])

    def test_udd_sequence_diffusion_attenuates_static_gradient_signal(self) -> None:
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

        stationary = ensemble.__class__(
            positions=ensemble.positions.copy(),
            magnetization=ensemble.magnetization.copy(),
            weights=ensemble.weights.copy(),
            diffusion_coefficient=np.zeros_like(ensemble.diffusion_coefficient),
        )
        no_diffusion = run_motion_udd_sequence(
            stationary,
            fields,
            num_pulses=4,
            total_duration=0.32,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(35.0, 0.0),
            substeps_per_interval=6,
            rng=np.random.default_rng(99),
        )
        diffusing = run_motion_udd_sequence(
            ensemble,
            fields,
            num_pulses=4,
            total_duration=0.32,
            excitation_duration=0.002,
            refocusing_duration=0.004,
            gradient=(35.0, 0.0),
            substeps_per_interval=6,
            rng=np.random.default_rng(99),
        )

        self.assertLess(np.abs(diffusing.signal[-1]), np.abs(no_diffusion.signal[-1]))

    def test_udd_suppresses_low_frequency_detuning_better_than_cpmg(self) -> None:
        fields = make_motion_field_maps_2d([-1.0, 1.0], [-1.0, 1.0])
        x_axis = np.linspace(-0.5, 0.5, 41)
        ensemble = initialize_ensemble_from_density(
            np.ones((x_axis.size, 1), dtype=np.float64),
            x_axis,
            [0.0],
            walkers_per_cell=2,
            diffusion_coefficient=0.0,
        )
        duration = 0.6

        def detuning(time, positions):
            return 1500.0 * positions[:, 0] * np.cos(2.0 * np.pi * 0.35 * time)

        common = dict(
            ensemble=ensemble,
            fields=fields,
            excitation_duration=0.00025,
            refocusing_duration=0.0005,
            t1=5.0,
            t2=2.0,
            substeps_per_interval=24,
            detuning_waveform=detuning,
        )

        cpmg = run_motion_cpmg_sequence(
            num_echoes=4,
            echo_spacing=duration / 4,
            **common,
        )
        udd = run_motion_udd_sequence(
            num_pulses=4,
            total_duration=duration,
            **common,
        )

        self.assertGreater(np.abs(udd.signal[-1]), 1.5 * np.abs(cpmg.signal[-1]))


if __name__ == "__main__":
    unittest.main()
