"""Tests for the C_Q <-> nu_Q conversion and site construction."""

from __future__ import annotations

import unittest

import numpy as np

from mr_integration import cq_hz_from_nu_q, nu_q_from_cq_hz, quadrupolar_site_from_cq
from mr_integration.conversions import quadrupolar_site_from_efg_record
from mr_integration.cross_validation import predicted_lines


class ConversionTests(unittest.TestCase):
    def test_spin1_factor_is_three_quarters(self) -> None:
        cq = 5.0e6
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 1.0), 0.75 * cq, places=3)

    def test_spin_three_half_factor_is_one_half(self) -> None:
        cq = 10.0e6
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 1.5), 0.5 * cq, places=3)

    def test_sign_of_cq_does_not_change_nu_q(self) -> None:
        self.assertEqual(nu_q_from_cq_hz(-5.0e6, 1.0), nu_q_from_cq_hz(5.0e6, 1.0))

    def test_roundtrip(self) -> None:
        for spin in (1.0, 1.5):
            nu = nu_q_from_cq_hz(7.3e6, spin)
            self.assertAlmostEqual(cq_hz_from_nu_q(nu, spin), 7.3e6, places=3)

    def test_non_quadrupolar_spin_raises(self) -> None:
        for spin in (0.5, 0.0, float("nan")):
            with self.subTest(spin=spin), self.assertRaises(ValueError):
                nu_q_from_cq_hz(5.0e6, spin)

    def test_higher_spins_use_simulator_mapping(self) -> None:
        # nu_Q = C_Q * d / (4 I (2I - 1)) with d = 3 (integer) or 6 (half-integer).
        expected = {2.0: 3 / 24, 2.5: 6 / 40, 3.5: 6 / 84, 4.5: 6 / 144}
        for spin, ratio in expected.items():
            with self.subTest(spin=spin):
                self.assertAlmostEqual(nu_q_from_cq_hz(1.0e6, spin), ratio * 1.0e6)
                self.assertAlmostEqual(
                    cq_hz_from_nu_q(nu_q_from_cq_hz(7.3e6, spin), spin), 7.3e6, places=3
                )

    def test_higher_spin_lines_agree_with_independent_dft_hamiltonian(self) -> None:
        for spin in (2.0, 2.5, 3.5, 4.5):
            for eta in (0.0, 0.3, 0.9):
                with self.subTest(spin=spin, eta=eta):
                    lines = predicted_lines(cq_hz=10.3e6, eta=eta, spin=spin, isotope="X")
                    self.assertTrue(lines.self_consistent(atol_hz=1.0))

    def test_efg_record_spin_falls_back_to_simulator_registry(self) -> None:
        class FakeRecord:
            cq_mhz = 10.34
            eta = 0.55
            atom_index = 5

        site = quadrupolar_site_from_efg_record(FakeRecord(), isotope="17O")
        self.assertEqual(site.spin, 2.5)
        self.assertAlmostEqual(site.quadrupole_frequency_hz, 0.15 * 10.34e6, places=1)
        with self.assertRaises(ValueError):
            quadrupolar_site_from_efg_record(FakeRecord(), isotope="999Xx")

    def test_site_from_cq_has_expected_frequency(self) -> None:
        site = quadrupolar_site_from_cq(cq_hz=5.0e6, eta=0.1, spin=1.0)
        self.assertAlmostEqual(site.quadrupole_frequency_hz, 3.75e6, places=3)
        self.assertEqual(site.spin, 1.0)
        self.assertAlmostEqual(site.eta, 0.1)

    def test_site_from_efg_record_duck_typed(self) -> None:
        class FakeRecord:
            cq_mhz = -5.034045
            eta = 0.111906
            atom_index = 3

        site = quadrupolar_site_from_efg_record(FakeRecord(), isotope="14N")
        self.assertEqual(site.spin, 1.0)
        self.assertAlmostEqual(
            site.quadrupole_frequency_hz, 0.75 * 5.034045e6, places=1
        )
        self.assertTrue(np.isclose(site.eta, 0.111906))


if __name__ == "__main__":
    unittest.main()
