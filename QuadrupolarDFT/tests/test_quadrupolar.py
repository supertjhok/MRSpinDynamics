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

    def test_axial_higher_spin_returns_only_allowed_ladder(self):
        # Regression: every level difference used to be returned, so spin-5/2
        # at eta=0 gave [1.5, 3.0, 4.5] MHz (4.5 MHz is the forbidden
        # +-1/2 <-> +-5/2 line). The axial allowed lines are n * nu_Q with
        # nu_Q = 3 C_Q / (2 I (2I - 1)) and n = 1 .. I - 1/2.
        cq_hz = 10.0e6
        for spin in (2.5, 3.5, 4.5):
            with self.subTest(spin=spin):
                nu_q = 3.0 * cq_hz / (2.0 * spin * (2.0 * spin - 1.0))
                expected = nu_q * np.arange(1, int(spin - 0.5) + 1)
                transitions = nqr_frequencies_hz(spin=spin, cq_hz=cq_hz, eta=0.0)
                np.testing.assert_allclose(transitions, expected, rtol=1e-12)
        everything = nqr_frequencies_hz(
            spin=2.5, cq_hz=cq_hz, eta=0.0, include_forbidden=True
        )
        np.testing.assert_allclose(everything, [1.5e6, 3.0e6, 4.5e6], rtol=1e-12)

    def test_spin_five_halves_matches_creel_series(self):
        # Creel et al. perturbation series for I = 5/2 (accurate to O(eta^6)):
        # nu1 = 3 C_Q/20 (1 + 1.0926 eta^2 - 0.6340 eta^4),
        # nu2 = 3 C_Q/10 (1 - 0.2037 eta^2 + 0.1622 eta^4).
        cq_hz = 10.0e6
        eta = 0.1
        transitions = nqr_frequencies_hz(spin=2.5, cq_hz=cq_hz, eta=eta)
        nu1 = 3.0 * cq_hz / 20.0 * (1 + 1.0926 * eta**2 - 0.6340 * eta**4)
        nu2 = 3.0 * cq_hz / 10.0 * (1 - 0.2037 * eta**2 + 0.1622 * eta**4)
        # With eta > 0 the +-1/2 <-> +-5/2 line is weakly allowed (nu1 + nu2).
        self.assertEqual(transitions.shape, (3,))
        np.testing.assert_allclose(transitions[:2], [nu1, nu2], rtol=1e-5)
        self.assertAlmostEqual(transitions[2], transitions[0] + transitions[1], delta=1e-3)


if __name__ == "__main__":
    unittest.main()
