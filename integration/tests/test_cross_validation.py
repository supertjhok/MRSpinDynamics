"""The headline test: two independent Hamiltonian implementations must agree."""

from __future__ import annotations

import unittest

import numpy as np

from mr_integration import predicted_lines, match_lines


class SelfConsistencyTests(unittest.TestCase):
    def test_spin1_simulator_matches_dft_module(self) -> None:
        # NaNO2 14N (DFT ICSD 82857 run).
        pl = predicted_lines(cq_hz=5.034045e6, eta=0.111906, spin=1.0, isotope="14N")
        self.assertEqual(pl.simulator_hz.size, pl.dft_hz.size)
        self.assertTrue(pl.self_consistent(atol_hz=1.0))
        self.assertLess(pl.max_abs_discrepancy_hz, 1.0)

    def test_spin1_zero_eta_line_equals_nu_q(self) -> None:
        pl = predicted_lines(cq_hz=4.0e6, eta=0.0, spin=1.0)
        # At eta=0 the spin-1 doublet collapses to a single line at nu_Q.
        self.assertTrue(np.allclose(pl.simulator_hz, pl.nu_q_hz, atol=1.0))

    def test_spin_three_half_self_consistent(self) -> None:
        # 35Cl-like parameters.
        pl = predicted_lines(cq_hz=54.0e6, eta=0.05, spin=1.5, isotope="35Cl")
        self.assertTrue(pl.self_consistent(atol_hz=1.0))

    def test_spin_three_half_zero_eta_line(self) -> None:
        pl = predicted_lines(cq_hz=20.0e6, eta=0.0, spin=1.5)
        # The single spin-3/2 line at eta=0 is nu_Q = C_Q / 2.
        self.assertEqual(pl.simulator_hz.size, 1)
        self.assertAlmostEqual(pl.simulator_hz[0], 10.0e6, delta=1.0)


class MatchLinesTests(unittest.TestCase):
    def test_pairs_nearest(self) -> None:
        pairs = match_lines([1.0e6, 3.0e6], [3.1e6, 0.9e6])
        # measured sorted ascending: 0.9, 3.1
        self.assertAlmostEqual(pairs[0][0], 0.9e6)
        self.assertAlmostEqual(pairs[0][1], 1.0e6)
        self.assertAlmostEqual(pairs[1][0], 3.1e6)
        self.assertAlmostEqual(pairs[1][1], 3.0e6)

    def test_empty_predictions_give_nan(self) -> None:
        pairs = match_lines([], [1.0e6])
        self.assertTrue(np.isnan(pairs[0][1]))


if __name__ == "__main__":
    unittest.main()
