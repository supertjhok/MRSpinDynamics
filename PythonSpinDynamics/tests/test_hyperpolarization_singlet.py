from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.hyperpolarization import (
    parahydrogen_state,
    singlet_order_amplitude,
    singlet_order_operator,
    singlet_population,
    singlet_projector,
    spin_pair_swap_operator,
    triplet_population,
    triplet_projector,
)


class SingletStateTests(unittest.TestCase):
    def test_singlet_and_triplet_projectors_are_complete_and_orthogonal(self):
        singlet = singlet_projector()
        triplet = triplet_projector()
        identity = np.eye(4)

        np.testing.assert_allclose(singlet, singlet.conj().T)
        np.testing.assert_allclose(triplet, triplet.conj().T)
        np.testing.assert_allclose(singlet @ singlet, singlet, atol=1e-15)
        np.testing.assert_allclose(triplet @ triplet, triplet, atol=1e-15)
        np.testing.assert_allclose(singlet @ triplet, 0.0, atol=1e-15)
        np.testing.assert_allclose(singlet + triplet, identity, atol=1e-15)
        self.assertAlmostEqual(float(np.trace(singlet).real), 1.0)
        self.assertAlmostEqual(float(np.trace(triplet).real), 3.0)

    def test_singlet_order_is_trace_zero_and_swap_antisymmetric(self):
        singlet = singlet_projector()
        triplet = triplet_projector()
        order = singlet_order_operator()
        swap = spin_pair_swap_operator()

        self.assertAlmostEqual(float(np.trace(order).real), 0.0)
        np.testing.assert_allclose(swap @ singlet, -singlet, atol=1e-15)
        np.testing.assert_allclose(swap @ triplet, triplet, atol=1e-15)
        np.testing.assert_allclose(swap @ swap, np.eye(4), atol=1e-15)
        np.testing.assert_allclose(
            np.linalg.eigvalsh(order),
            [-1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 1.0],
            atol=1e-15,
        )

    def test_statistical_hydrogen_has_no_deviation_order(self):
        state = parahydrogen_state(0.25)

        np.testing.assert_allclose(state.density_matrix, np.eye(4) / 4.0)
        np.testing.assert_allclose(state.deviation_density, 0.0)
        self.assertAlmostEqual(state.singlet_population, 0.25)
        self.assertAlmostEqual(singlet_order_amplitude(state.density_matrix), 0.0)
        self.assertAlmostEqual(state.para_excess, 0.0)

    def test_pure_parahydrogen_is_the_singlet_projector(self):
        state = parahydrogen_state(1.0)

        np.testing.assert_allclose(state.density_matrix, singlet_projector())
        np.testing.assert_allclose(
            state.deviation_density,
            state.density_matrix - np.eye(4) / 4.0,
        )
        self.assertAlmostEqual(state.singlet_population, 1.0)
        self.assertAlmostEqual(triplet_population(state.density_matrix), 0.0)
        self.assertAlmostEqual(
            singlet_order_amplitude(state.deviation_density), 0.75
        )

    def test_para_excess_sets_singlet_order_amplitude(self):
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            with self.subTest(para_fraction=fraction):
                state = parahydrogen_state(fraction)
                self.assertAlmostEqual(
                    singlet_order_amplitude(state.deviation_density),
                    fraction - 0.25,
                )
                self.assertAlmostEqual(
                    singlet_population(state.density_matrix), fraction
                )
                self.assertAlmostEqual(
                    triplet_population(state.density_matrix), 1.0 - fraction
                )

    def test_pair_projector_embeds_in_a_larger_spin_system(self):
        embedded = singlet_projector(3, (0, 2))
        density = embedded / np.trace(embedded)

        self.assertEqual(embedded.shape, (8, 8))
        self.assertAlmostEqual(float(np.trace(embedded).real), 2.0)
        self.assertAlmostEqual(singlet_population(density, pair=(0, 2)), 1.0)
        self.assertAlmostEqual(
            singlet_order_amplitude(density, pair=(0, 2)), 0.75
        )

    def test_invalid_inputs_are_rejected(self):
        for fraction in (-0.01, 1.01, np.nan):
            with self.subTest(para_fraction=fraction):
                with self.assertRaises(ValueError):
                    parahydrogen_state(fraction)
        with self.assertRaises(ValueError):
            singlet_projector(2, (0, 0))
        with self.assertRaises(ValueError):
            singlet_population(np.eye(4))
        nonpositive = np.diag([1.1, -0.1, 0.0, 0.0])
        with self.assertRaises(ValueError):
            singlet_population(nonpositive)
        with self.assertRaises(ValueError):
            singlet_order_amplitude(np.eye(3))


if __name__ == "__main__":
    unittest.main()
