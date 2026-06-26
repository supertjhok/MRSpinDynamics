import unittest

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    average_tensors,
    coupling_constant_hz,
    nqr_frequencies_hz,
)


class TensorTests(unittest.TestCase):
    def test_principal_components_follow_efg_convention(self):
        tensor = EFGTensor.from_components(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.8],
            ]
        )

        principal = tensor.principal_components_si
        self.assertLess(abs(principal[0]), abs(principal[1]))
        self.assertLess(abs(principal[1]), abs(principal[2]))
        self.assertAlmostEqual(tensor.eta, 0.6)

    def test_average_tensors_before_diagonalization(self):
        first = EFGTensor.from_components(np.diag([-1.0, 0.2, 0.8]))
        second = EFGTensor.from_components(np.diag([1.0, -0.2, -0.8]))

        average = average_tensors([first, second])

        self.assertTrue(np.allclose(average.matrix_si, 0.0))
        self.assertAlmostEqual(average.eta, 0.0)


class QuadrupolarTests(unittest.TestCase):
    def test_coupling_constant_uses_barns(self):
        cq_hz = coupling_constant_hz(1e21, 1.0)

        self.assertAlmostEqual(cq_hz / 1e6, 24.179891, places=5)

    def test_spin_three_halves_matches_closed_form(self):
        cq_hz = 2.0e6
        eta = 0.4

        transitions = nqr_frequencies_hz(spin=1.5, cq_hz=cq_hz, eta=eta)

        expected = abs(cq_hz) / 2.0 * np.sqrt(1.0 + eta**2 / 3.0)
        self.assertEqual(transitions.shape, (1,))
        self.assertAlmostEqual(transitions[0], expected)

    def test_spin_one_returns_three_pairwise_transitions(self):
        cq_hz = 4.0e6
        eta = 0.3

        transitions = nqr_frequencies_hz(spin=1.0, cq_hz=cq_hz, eta=eta)

        expected = np.array(
            [
                eta * abs(cq_hz) / 2.0,
                3.0 * abs(cq_hz) / 4.0 * (1.0 - eta / 3.0),
                3.0 * abs(cq_hz) / 4.0 * (1.0 + eta / 3.0),
            ]
        )
        self.assertTrue(np.allclose(transitions, expected))


if __name__ == "__main__":
    unittest.main()
