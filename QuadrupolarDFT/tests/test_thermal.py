import unittest

import numpy as np

from quadrupolar_dft import (
    bose_occupation,
    mean_square_normal_coordinate,
    thermal_quantum_factor,
    wavenumber_to_angular_frequency,
)
from quadrupolar_dft.constants import (
    BOLTZMANN_CONSTANT_J_PER_K,
    REDUCED_PLANCK_CONSTANT_J_S,
)


class ThermalTests(unittest.TestCase):
    def test_wavenumber_conversion(self):
        # 1 cm^-1 -> 2 pi c (cm/s) = 2 pi * 2.998e10 ~ 1.883e11 rad/s.
        omega = wavenumber_to_angular_frequency(1.0)
        self.assertAlmostEqual(omega, 1.883651e11, delta=1e8)

    def test_quantum_factor_zero_point_limit(self):
        omega = wavenumber_to_angular_frequency(200.0)
        # coth -> 1 as T -> 0 (pure zero-point motion).
        self.assertAlmostEqual(thermal_quantum_factor(omega, 1e-6), 1.0, places=6)
        self.assertAlmostEqual(thermal_quantum_factor(omega, 0.0), 1.0, places=12)

    def test_quantum_factor_classical_limit(self):
        omega = wavenumber_to_angular_frequency(50.0)
        temperature = 2000.0
        expected = (
            2.0 * BOLTZMANN_CONSTANT_J_PER_K * temperature
            / (REDUCED_PLANCK_CONSTANT_J_S * omega)
        )
        self.assertAlmostEqual(
            thermal_quantum_factor(omega, temperature), expected, delta=0.05
        )

    def test_mean_square_zero_point_and_growth(self):
        omega = wavenumber_to_angular_frequency(150.0)
        zero_point = REDUCED_PLANCK_CONSTANT_J_S / (2.0 * omega)
        self.assertAlmostEqual(
            mean_square_normal_coordinate(omega, 1e-6), zero_point, delta=zero_point * 1e-5
        )
        # Monotonically increasing with temperature.
        values = [mean_square_normal_coordinate(omega, t) for t in (10, 100, 300, 600)]
        self.assertTrue(np.all(np.diff(values) > 0.0))

    def test_high_temperature_mean_square(self):
        omega = wavenumber_to_angular_frequency(100.0)
        temperature = 1500.0
        classical = BOLTZMANN_CONSTANT_J_PER_K * temperature / omega**2
        self.assertAlmostEqual(
            mean_square_normal_coordinate(omega, temperature),
            classical,
            delta=classical * 0.02,
        )

    def test_bose_occupation_limits(self):
        omega = wavenumber_to_angular_frequency(200.0)
        self.assertAlmostEqual(bose_occupation(omega, 1e-6), 0.0, places=10)
        self.assertGreater(bose_occupation(omega, 600.0), bose_occupation(omega, 100.0))


if __name__ == "__main__":
    unittest.main()
