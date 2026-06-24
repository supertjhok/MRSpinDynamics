from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows import (
    gradient_moment_b_value,
    pgse_b_value,
    run_pgse,
    run_pgse_moment,
    run_pgse_walkers,
)


class PGSETests(unittest.TestCase):
    def test_moment_b_value_matches_stejskal_tanner_formula(self) -> None:
        gamma = 2.675e8
        gradient = 0.08
        delta = 3.0e-3
        delta_big = 24.0e-3

        direct = pgse_b_value(gradient, delta, delta_big, gamma=gamma)
        moment = gradient_moment_b_value(
            [
                (delta, gradient),
                (delta_big - delta, 0.0),
                (delta, -gradient),
            ],
            gamma=gamma,
        )

        self.assertAlmostEqual(direct, moment, places=6)

    def test_moment_signal_has_expected_diffusion_slope(self) -> None:
        diffusion = 2.2e-9
        result = run_pgse_moment(
            num_echoes=3,
            gradient_amplitude=0.07,
            gradient_duration=2.5e-3,
            diffusion_time=18.0e-3,
            diffusion_coefficient=diffusion,
            t2_seconds=0.150,
            first_echo_time_seconds=40.0e-3,
            echo_spacing_seconds=20.0e-3,
        )

        expected_diffusion = np.exp(-result.b_value * diffusion)
        expected_t2 = np.exp(-result.echo_times / 0.150)
        np.testing.assert_allclose(result.diffusion_attenuation, expected_diffusion)
        np.testing.assert_allclose(np.abs(result.signal), expected_diffusion * expected_t2)

    def test_zero_gradient_or_diffusion_has_no_pgse_loss(self) -> None:
        no_gradient = run_pgse_moment(
            gradient_amplitude=0.0,
            gradient_duration=2.0e-3,
            diffusion_time=20.0e-3,
            diffusion_coefficient=2.3e-9,
        )
        no_diffusion = run_pgse_moment(
            gradient_amplitude=0.08,
            gradient_duration=2.0e-3,
            diffusion_time=20.0e-3,
            diffusion_coefficient=0.0,
        )

        self.assertAlmostEqual(no_gradient.diffusion_attenuation, 1.0)
        self.assertAlmostEqual(no_diffusion.diffusion_attenuation, 1.0)

    def test_dispatches_to_requested_backend(self) -> None:
        result = run_pgse(
            backend="moment",
            gradient_amplitude=0.03,
            gradient_duration=1.0e-3,
            diffusion_time=10.0e-3,
        )

        self.assertEqual(result.backend, "moment")

    def test_walker_pgse_converges_toward_moment_attenuation(self) -> None:
        common = {
            "gradient_amplitude": 0.05,
            "gradient_duration": 2.0e-3,
            "diffusion_time": 16.0e-3,
            "walkers_per_cell": 12000,
            "seed": 123,
            "substeps_per_interval": 6,
            "excitation_duration": 60.0e-6,
            "refocusing_duration": 120.0e-6,
            "boundary": "reflect",
        }
        diffusion = 2.5e-9
        reference = run_pgse_walkers(
            **common,
            diffusion_coefficient=0.0,
        )
        diffusing = run_pgse_walkers(
            **common,
            diffusion_coefficient=diffusion,
        )
        expected = np.exp(-diffusing.b_value * diffusion)
        measured = abs(diffusing.signal[0]) / abs(reference.signal[0])

        self.assertLess(measured, 1.0)
        self.assertAlmostEqual(measured, expected, delta=0.08)

    def test_walker_spin_echo_is_num_echoes_one_special_case(self) -> None:
        result = run_pgse_walkers(
            num_echoes=1,
            gradient_amplitude=0.02,
            gradient_duration=1.0e-3,
            diffusion_time=8.0e-3,
            diffusion_coefficient=0.0,
            walkers_per_cell=128,
            seed=9,
        )

        self.assertEqual(result.signal.shape, (1,))
        self.assertEqual(result.sequence.sample_labels, ("echo_1",))


if __name__ == "__main__":
    unittest.main()
