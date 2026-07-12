from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.coupling import coupled_spin_system
from spin_dynamics.hyperpolarization import (
    PHIPFieldSegment,
    hydrogenative_phip_state,
    parahydrogen_state,
    simulate_hydrogenative_phip,
    simulate_slic_lls,
    singlet_order_amplitude,
    singlet_population,
    store_singlet_order,
)


class LongLivedSingletWorkflowTests(unittest.TestCase):
    def test_physical_singlet_relaxes_to_statistical_population(self) -> None:
        initial = parahydrogen_state(1.0).density_matrix

        stored = store_singlet_order(initial, np.log(3.0), 1.0)

        self.assertAlmostEqual(float(np.trace(stored).real), 1.0)
        self.assertAlmostEqual(singlet_population(stored), 0.25 + 0.75 / 3.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(stored).min()), -1e-12)

    def test_storage_purge_retains_only_exponentially_decaying_singlet_mode(self) -> None:
        order = parahydrogen_state(0.8).deviation_density
        contaminant = np.diag([0.1, -0.1, 0.1, -0.1])

        stored = store_singlet_order(order + contaminant, 2.0, 2.0)

        expected = np.exp(-1.0) * singlet_order_amplitude(order)
        self.assertAlmostEqual(singlet_order_amplitude(stored), expected)
        np.testing.assert_allclose(stored, expected * (order / singlet_order_amplitude(order)))

    def test_slic_prepare_store_readout_follows_ts(self) -> None:
        system = coupled_spin_system(
            [-0.35, 0.35],
            [[0.0, 7.0], [7.0, 0.0]],
        )
        times = np.array([0.0, 0.5, 1.0])

        result = simulate_slic_lls(
            system,
            times,
            singlet_lifetime_seconds=1.0,
        )

        self.assertAlmostEqual(result.matching_nutation_hz, 7.0)
        self.assertAlmostEqual(result.prepared_singlet_amplitude, -1.0, delta=0.002)
        np.testing.assert_allclose(
            result.singlet_amplitudes / result.singlet_amplitudes[0],
            np.exp(-times),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.normalized_signal / result.normalized_signal[0],
            np.exp(-times),
            atol=1e-12,
        )
        self.assertFalse(result.normalized_signal.flags.writeable)


class HydrogenativePHIPWorkflowTests(unittest.TestCase):
    def test_reaction_map_scales_with_para_excess_and_pairwise_yield(self) -> None:
        full = hydrogenative_phip_state(
            2,
            para_fraction=0.75,
            pairwise_addition_fraction=1.0,
        )
        half = hydrogenative_phip_state(
            2,
            para_fraction=0.75,
            pairwise_addition_fraction=0.5,
        )
        statistical = hydrogenative_phip_state(2, para_fraction=0.25)

        np.testing.assert_allclose(half.deviation_density, 0.5 * full.deviation_density)
        np.testing.assert_allclose(statistical.deviation_density, 0.0)
        self.assertAlmostEqual(float(np.trace(half.density_matrix).real), 1.0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(half.density_matrix).min()), -1e-12)

    def test_product_pair_can_be_mapped_into_a_larger_spin_system(self) -> None:
        state = hydrogenative_phip_state(
            3,
            para_fraction=1.0,
            product_pair=(0, 2),
        )

        self.assertEqual(state.density_matrix.shape, (8, 8))
        self.assertAlmostEqual(singlet_population(state.density_matrix, pair=(0, 2)), 1.0)

    def test_pasadena_generates_antiphase_fid_and_scales_linearly(self) -> None:
        system = coupled_spin_system(
            [-50.0, 50.0],
            [[0.0, 7.0], [7.0, 0.0]],
        )
        times = np.linspace(0.0, 0.04, 64)
        full = simulate_hydrogenative_phip(
            system,
            times,
            protocol="pasadena",
            para_fraction=0.75,
        )
        half = simulate_hydrogenative_phip(
            system,
            times,
            protocol="pasadena",
            para_fraction=0.75,
            pairwise_addition_fraction=0.5,
        )

        self.assertGreater(float(np.max(np.abs(full.signal))), 0.1)
        np.testing.assert_allclose(half.signal, 0.5 * full.signal, atol=1e-12)

    def test_altadena_requires_explicit_field_trajectory(self) -> None:
        system = coupled_spin_system(
            [-50.0, 50.0],
            [[0.0, 7.0], [7.0, 0.0]],
        )
        with self.assertRaisesRegex(ValueError, "field_trajectory"):
            simulate_hydrogenative_phip(
                system,
                [0.0, 0.001],
                protocol="altadena",
                para_fraction=0.9,
            )

        result = simulate_hydrogenative_phip(
            system,
            [0.0, 0.001],
            protocol="altadena",
            para_fraction=0.9,
            field_trajectory=[
                PHIPFieldSegment(scale, 0.002)
                for scale in np.linspace(0.05, 1.0, 20)
            ],
        )
        self.assertTrue(np.all(np.isfinite(result.signal)))


if __name__ == "__main__":
    unittest.main()
