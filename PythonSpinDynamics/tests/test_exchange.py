from __future__ import annotations

import importlib.util
import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.analysis import invert_t2_t2
from spin_dynamics.exchange import (
    ExchangeSite,
    ExchangeSystem,
    exchange_generator,
    exchange_spectrum,
    mixing_propagator,
    simulate_exchange_fid,
    simulate_relaxation_exchange_2d,
    transverse_generator,
    two_site_exchange,
)


HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class ExchangeSystemValidationTests(unittest.TestCase):
    def test_rate_matrix_shape_is_checked(self) -> None:
        sites = (ExchangeSite("A", 0.5), ExchangeSite("B", 0.5))
        with self.assertRaises(ValueError):
            ExchangeSystem(sites, np.zeros((3, 3)))

    def test_negative_off_diagonal_rate_is_rejected(self) -> None:
        sites = (ExchangeSite("A", 0.5), ExchangeSite("B", 0.5))
        with self.assertRaises(ValueError):
            ExchangeSystem(sites, np.array([[0.0, -1.0], [1.0, 0.0]]))

    def test_populations_are_normalized(self) -> None:
        sites = (ExchangeSite("A", 3.0), ExchangeSite("B", 1.0))
        system = ExchangeSystem(sites, np.array([[0.0, 1.0], [3.0, 0.0]]))
        np.testing.assert_allclose(system.populations, [0.75, 0.25])

    def test_detailed_balance_violation_warns(self) -> None:
        sites = (ExchangeSite("A", 0.5), ExchangeSite("B", 0.5))
        rates = np.array([[0.0, 10.0], [1.0, 0.0]])
        with self.assertWarns(RuntimeWarning):
            ExchangeSystem(sites, rates, balance="warn")
        with self.assertRaises(ValueError):
            ExchangeSystem(sites, rates, balance="raise")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ExchangeSystem(sites, rates, balance="off")

    def test_bad_site_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExchangeSite("A", population=-1.0)
        with self.assertRaises(ValueError):
            ExchangeSite("A", t2_seconds=0.0)


class ExchangeGeneratorTests(unittest.TestCase):
    def test_generator_columns_conserve_magnetization(self) -> None:
        system = two_site_exchange(
            offset_a_hz=-50.0, offset_b_hz=50.0, k_ab_hz=30.0, k_ba_hz=10.0
        )
        generator = exchange_generator(system)
        np.testing.assert_allclose(generator.sum(axis=0), [0.0, 0.0], atol=1e-12)
        # off-diagonal X[i, j] is the rate from j into i
        self.assertAlmostEqual(generator[0, 1], 10.0)  # k_{B->A}
        self.assertAlmostEqual(generator[1, 0], 30.0)  # k_{A->B}

    def test_transverse_generator_carries_offsets_and_relaxation(self) -> None:
        system = two_site_exchange(
            offset_a_hz=-50.0,
            offset_b_hz=50.0,
            k_ab_hz=0.0,
            k_ba_hz=0.0,
            t2_a_seconds=0.1,
            t2_b_seconds=0.2,
        )
        generator = transverse_generator(system)
        self.assertAlmostEqual(generator[0, 0].imag, 2.0 * np.pi * -50.0)
        self.assertAlmostEqual(generator[0, 0].real, -1.0 / 0.1)
        self.assertAlmostEqual(generator[1, 1].real, -1.0 / 0.2)


class MixingPropagatorTests(unittest.TestCase):
    def test_zero_time_is_identity(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0, offset_b_hz=0.0, k_ab_hz=20.0, k_ba_hz=20.0
        )
        np.testing.assert_allclose(
            mixing_propagator(system, 0.0, include_t1=False), np.eye(2)
        )

    def test_propagator_is_column_stochastic_without_t1(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0, offset_b_hz=0.0, k_ab_hz=15.0, k_ba_hz=45.0
        )
        propagator = mixing_propagator(system, 0.02, include_t1=False)
        np.testing.assert_allclose(propagator.sum(axis=0), [1.0, 1.0], atol=1e-12)

    def test_long_mixing_reaches_population_equilibrium(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0, offset_b_hz=0.0, k_ab_hz=15.0, k_ba_hz=45.0
        )
        propagator = mixing_propagator(system, 1.0e6, include_t1=False)
        populations = system.populations
        for column in range(2):
            np.testing.assert_allclose(
                propagator[:, column], populations, atol=1e-6
            )

    def test_symmetric_two_site_matches_analytic_form(self) -> None:
        k = 20.0
        t = 0.01
        system = two_site_exchange(
            offset_a_hz=0.0, offset_b_hz=0.0, k_ab_hz=k, k_ba_hz=k
        )
        propagator = mixing_propagator(system, t, include_t1=False)
        decay = np.exp(-2.0 * k * t)
        self.assertAlmostEqual(propagator[0, 0], 0.5 * (1.0 + decay), places=12)
        self.assertAlmostEqual(propagator[1, 0], 0.5 * (1.0 - decay), places=12)

    def test_t1_decay_reduces_stored_magnetization(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0,
            offset_b_hz=0.0,
            k_ab_hz=10.0,
            k_ba_hz=10.0,
            t1_a_seconds=0.5,
            t1_b_seconds=0.5,
        )
        with_t1 = mixing_propagator(system, 0.1, include_t1=True)
        without_t1 = mixing_propagator(system, 0.1, include_t1=False)
        self.assertLess(with_t1.sum(axis=0)[0], 1.0)
        np.testing.assert_allclose(without_t1.sum(axis=0), [1.0, 1.0], atol=1e-12)


class ExchangeFIDTests(unittest.TestCase):
    def test_initial_signal_is_total_population(self) -> None:
        system = two_site_exchange(
            offset_a_hz=-100.0, offset_b_hz=100.0, k_ab_hz=10.0, k_ba_hz=10.0
        )
        signal = simulate_exchange_fid(system, np.array([0.0]))
        self.assertAlmostEqual(signal[0].real, 1.0)
        self.assertAlmostEqual(signal[0].imag, 0.0)

    def test_slow_exchange_keeps_two_resolved_lines(self) -> None:
        system = two_site_exchange(
            offset_a_hz=-120.0,
            offset_b_hz=120.0,
            k_ab_hz=2.0,
            k_ba_hz=2.0,
            t2_a_seconds=0.5,
            t2_b_seconds=0.5,
        )
        frequencies, spectrum = exchange_spectrum(system, num_points=8192)
        peak = frequencies[np.argmax(np.abs(spectrum))]
        # resolved lines sit near the site offsets, far from the average (0 Hz)
        self.assertGreater(abs(peak), 80.0)

    def test_fast_exchange_coalesces_to_average_offset(self) -> None:
        system = two_site_exchange(
            offset_a_hz=-120.0,
            offset_b_hz=120.0,
            k_ab_hz=4000.0,
            k_ba_hz=4000.0,
            t2_a_seconds=0.5,
            t2_b_seconds=0.5,
        )
        frequencies, spectrum = exchange_spectrum(system, num_points=8192)
        peak = frequencies[np.argmax(np.abs(spectrum))]
        # equal populations average to 0 Hz
        self.assertLess(abs(peak), 15.0)

    def test_unequal_fast_exchange_averages_by_population(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0,
            offset_b_hz=200.0,
            k_ab_hz=5000.0,
            population_a=0.75,
            t2_a_seconds=0.5,
            t2_b_seconds=0.5,
        )
        frequencies, spectrum = exchange_spectrum(system, num_points=8192)
        peak = frequencies[np.argmax(np.abs(spectrum))]
        expected = 0.75 * 0.0 + 0.25 * 200.0
        self.assertLess(abs(peak - expected), 15.0)


class TwoSiteBuilderTests(unittest.TestCase):
    def test_population_input_derives_balanced_backward_rate(self) -> None:
        system = two_site_exchange(
            offset_a_hz=0.0, offset_b_hz=10.0, k_ab_hz=30.0, population_a=0.25
        )
        np.testing.assert_allclose(system.populations, [0.25, 0.75])
        # detailed balance: p_a k_ab == p_b k_ba -> k_ba = k_ab p_a / p_b = 10
        self.assertAlmostEqual(system.exchange_rates_hz[1, 0], 10.0)

    def test_requires_exactly_one_of_rate_or_population(self) -> None:
        with self.assertRaises(ValueError):
            two_site_exchange(offset_a_hz=0.0, offset_b_hz=1.0, k_ab_hz=1.0)
        with self.assertRaises(ValueError):
            two_site_exchange(
                offset_a_hz=0.0,
                offset_b_hz=1.0,
                k_ab_hz=1.0,
                k_ba_hz=1.0,
                population_a=0.5,
            )


class RelaxationExchange2DTests(unittest.TestCase):
    def _system(self, k: float) -> ExchangeSystem:
        return two_site_exchange(
            offset_a_hz=0.0,
            offset_b_hz=0.0,
            k_ab_hz=k,
            k_ba_hz=k,
            t2_a_seconds=0.01,
            t2_b_seconds=0.2,
        )

    def test_no_exchange_is_separable_diagonal(self) -> None:
        system = self._system(0.0)
        encode = np.linspace(0.0, 0.05, 12)
        detect = np.linspace(0.0, 0.6, 12)
        result = simulate_relaxation_exchange_2d(
            system, encode, detect, mixing_time=0.05
        )
        np.testing.assert_allclose(result.mixing_propagator, np.eye(2), atol=1e-12)
        # with G = I the data is a rank-2 separable sum of pure decays
        r2 = system.r2_rates
        pops = system.populations
        expected = np.zeros((encode.size, detect.size))
        for site in range(2):
            enc = pops[site] * np.exp(-encode * r2[site])
            det = np.exp(-detect * r2[site])
            expected += np.outer(enc, det)
        np.testing.assert_allclose(result.data, expected, atol=1e-12)

    def test_exchange_transfers_amplitude_off_diagonal(self) -> None:
        no_exchange = simulate_relaxation_exchange_2d(
            self._system(0.0),
            np.linspace(0.0, 0.05, 10),
            np.linspace(0.0, 0.6, 10),
            mixing_time=0.05,
        )
        with_exchange = simulate_relaxation_exchange_2d(
            self._system(8.0),
            np.linspace(0.0, 0.05, 10),
            np.linspace(0.0, 0.6, 10),
            mixing_time=0.05,
        )
        self.assertAlmostEqual(no_exchange.mixing_propagator[0, 1], 0.0)
        self.assertGreater(with_exchange.mixing_propagator[0, 1], 0.05)

    @unittest.skipUnless(HAS_SCIPY, "non-negative ILT requires SciPy")
    def test_inverse_t2_t2_recovers_exchange_cross_peaks(self) -> None:
        system = self._system(6.0)
        encode = np.linspace(0.0, 0.12, 32)
        detect = np.linspace(0.0, 0.8, 32)
        result = simulate_relaxation_exchange_2d(
            system, encode, detect, mixing_time=0.06
        )
        t2_axis = np.logspace(-3, 0, 48)
        ilt = invert_t2_t2(
            result.data,
            encode,
            detect,
            t2_axis,
            regularization=1e-3,
            regularization_order=2,
        )
        self.assertLess(
            ilt.residual_norm / np.linalg.norm(result.data), 0.05
        )
        distribution = ilt.distribution
        fast = int(np.argmin(np.abs(t2_axis - 0.01)))
        slow = int(np.argmin(np.abs(t2_axis - 0.2)))
        window = 3
        def block(i: int, j: int) -> float:
            return float(
                distribution[
                    max(0, i - window) : i + window + 1,
                    max(0, j - window) : j + window + 1,
                ].sum()
            )

        diagonal_mass = block(fast, fast) + block(slow, slow)
        cross_mass = block(fast, slow) + block(slow, fast)
        # exchange must place real intensity on the off-diagonal cross peaks
        self.assertGreater(cross_mass, 0.05 * diagonal_mass)


if __name__ == "__main__":
    unittest.main()
