"""Validation tests for monochromatic Floquet crossover RF dynamics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    linear_rf_floquet_hamiltonian,
    sample_linear_rf_pulse,
    simulate_floquet_rf,
    simulate_lab_frame_rf,
)


class FloquetRFTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.17e6,
        )
        self.b0_direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        self.b0 = 1.0e6 / self.site.gamma_hz_per_t * self.b0_direction
        self.b1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
        self.carrier = 1.26191814e6
        self.duration = 4.0 / self.carrier
        self.initial = np.zeros((4, 4), dtype=np.complex128)
        self.initial[0, 0] = 1.0

    def test_sambe_hamiltonian_is_hermitian_with_expected_size(self) -> None:
        hamiltonian = linear_rf_floquet_hamiltonian(
            self.site,
            self.b0,
            nutation_hz=50.0e3,
            rf_frequency_hz=self.carrier,
            phase_radians=0.4,
            b1_direction_pas=self.b1,
            sidebands=3,
        )
        self.assertEqual(hamiltonian.shape, (28, 28))
        np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1.0e-9)

    def test_sideband_convergence_matches_lab_frame_reference(self) -> None:
        durations, fields = sample_linear_rf_pulse(
            self.duration,
            1.0 / (300.0 * self.carrier),
            2.0 * 50.0e3 / self.site.gamma_hz_per_t,
            self.carrier,
            phase_radians=0.4,
            direction_pas=self.b1,
        )
        lab = simulate_lab_frame_rf(
            self.site,
            self.b0,
            durations,
            fields,
            initial_density=self.initial,
        )

        def result(sidebands: int):
            return simulate_floquet_rf(
                self.site,
                self.b0,
                nutation_hz=50.0e3,
                rf_frequency_hz=self.carrier,
                pulse_duration_seconds=self.duration,
                phase_radians=0.4,
                b1_direction_pas=self.b1,
                sidebands=sidebands,
                initial_density=self.initial,
            )

        one = result(1)
        four = result(4)
        one_error = np.linalg.norm(one.density_matrix - lab.density_matrices[-1])
        four_error = np.linalg.norm(four.density_matrix - lab.density_matrices[-1])
        self.assertGreater(one_error, 1.0e-2)
        self.assertLess(four_error, 5.0e-5)
        self.assertLess(four.unitarity_error, 1.0e-5)


if __name__ == "__main__":
    unittest.main()
