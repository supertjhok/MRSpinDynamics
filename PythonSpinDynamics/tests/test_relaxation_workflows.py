from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import quadrupolar_site  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    run_arrhenius_quadrupolar_relaxation,
    run_field_cycling_nmrd,
    run_nmrd,
    run_quadrupolar_relaxation,
)


class QuadrupolarRelaxationWorkflowTests(unittest.TestCase):
    def test_transition_resolved_rates_have_documented_shape(self) -> None:
        site = quadrupolar_site("14N", cq_hz=3.0e6, eta=0.2)

        result = run_quadrupolar_relaxation(
            site,
            [1.0e-9, 2.0e-9],
            fluctuation_amplitude_hz=20.0e3,
        )

        self.assertEqual(result.transition_labels, ("z", "y", "x"))
        self.assertEqual(result.r1_per_second.shape, (2, 3))
        self.assertEqual(result.r2_per_second.shape, (2, 3))
        self.assertTrue(np.all(result.r1_per_second > 0.0))
        self.assertTrue(np.all(result.r2_per_second > 0.0))
        np.testing.assert_allclose(result.t1_seconds, 1.0 / result.r1_per_second)
        np.testing.assert_allclose(result.t2_seconds, 1.0 / result.r2_per_second)

    def test_extreme_narrowing_rates_scale_with_tau(self) -> None:
        site = quadrupolar_site("14N", cq_hz=3.0e6, eta=0.2)

        result = run_quadrupolar_relaxation(
            site,
            [0.5e-10, 1.0e-10],
            fluctuation_amplitude_hz=10.0e3,
        )

        np.testing.assert_allclose(
            result.r1_per_second[1] / result.r1_per_second[0],
            2.0,
            rtol=2.0e-4,
        )
        np.testing.assert_allclose(
            result.r2_per_second[1] / result.r2_per_second[0],
            2.0,
            rtol=2.0e-4,
        )

    def test_arrhenius_wrapper_preserves_temperature_and_trend(self) -> None:
        site = quadrupolar_site("14N", cq_hz=3.0e6, eta=0.2)

        result = run_arrhenius_quadrupolar_relaxation(
            site,
            [280.0, 300.0, 330.0],
            tau_ref_seconds=1.0e-9,
            reference_temperature_kelvin=300.0,
            activation_energy_j_per_mol=18.0e3,
            fluctuation_amplitude_hz=15.0e3,
        )

        np.testing.assert_array_equal(result.temperature_kelvin, [280.0, 300.0, 330.0])
        self.assertGreater(result.correlation_time_seconds[0], result.correlation_time_seconds[1])
        self.assertGreater(result.correlation_time_seconds[1], result.correlation_time_seconds[2])


class NMRDWorkflowTests(unittest.TestCase):
    def test_nmrd_returns_condition_by_frequency_grid(self) -> None:
        frequency = np.array([0.0, 1.0e6, 10.0e6])
        tau = np.array([1.0e-9, 10.0e-9])

        result = run_nmrd(
            frequency,
            tau,
            coupling_scale_per_second2=4.0e8,
        )

        self.assertEqual(result.relaxivity_shape, (2, 3))
        self.assertIsNone(result.field_tesla)
        self.assertIsNone(result.temperature_kelvin)
        self.assertTrue(np.all(np.diff(result.r1_per_second, axis=1) <= 0.0))
        np.testing.assert_allclose(result.j0_seconds[:, 0], 2.0 * tau)

    def test_field_cycling_maps_field_to_frequency_and_arrhenius_tau(self) -> None:
        fields = np.array([0.0, 1.0e-3, 0.1, 1.0])
        temperatures = np.array([280.0, 320.0])

        result = run_field_cycling_nmrd(
            fields,
            temperatures,
            gamma_hz_per_t=42.57747892e6,
            tau_ref_seconds=8.0e-9,
            reference_temperature_kelvin=300.0,
            activation_energy_j_per_mol=16.0e3,
            coupling_scale_per_second2=5.0e8,
        )

        np.testing.assert_allclose(result.larmor_frequency_hz, 42.57747892e6 * fields)
        np.testing.assert_array_equal(result.field_tesla, fields)
        np.testing.assert_array_equal(result.temperature_kelvin, temperatures)
        self.assertGreater(result.correlation_time_seconds[0], result.correlation_time_seconds[1])
        self.assertGreater(result.r1_per_second[0, 0], result.r1_per_second[1, 0])

    def test_invalid_axis_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            run_nmrd(
                [1.0e6, 2.0e6],
                [1.0e-9],
                coupling_scale_per_second2=1.0,
                field_tesla=[0.1],
            )


if __name__ == "__main__":
    unittest.main()
