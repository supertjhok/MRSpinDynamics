from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    eseem_spectrum,
    modulation_depth,
    nuclear_frequencies,
    three_pulse_eseem,
    three_pulse_eseem_quantum,
    two_pulse_eseem,
    two_pulse_eseem_quantum,
)


class HyperfineCouplingTests(unittest.TestCase):
    def test_nuclear_frequencies_secular_only(self) -> None:
        # B = 0: nu = |omega_I +/- A/2|.
        coupling = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=0.0)
        nu_alpha, nu_beta = nuclear_frequencies(coupling)
        self.assertAlmostEqual(nu_alpha, 14.5e6 + 1.5e6, places=3)
        self.assertAlmostEqual(nu_beta, 14.5e6 - 1.5e6, places=3)

    def test_modulation_depth_zero_without_anisotropy(self) -> None:
        coupling = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=0.0)
        self.assertEqual(modulation_depth(coupling), 0.0)

    def test_modulation_depth_in_unit_interval(self) -> None:
        coupling = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=4.0e6)
        k = modulation_depth(coupling)
        self.assertGreater(k, 0.0)
        self.assertLessEqual(k, 1.0)

    def test_invalid_larmor_raises(self) -> None:
        with self.assertRaises(ValueError):
            HyperfineCoupling(larmor_hz=0.0)
        with self.assertRaises(ValueError):
            HyperfineCoupling(larmor_hz=-1.0)


class TwoPulseEseemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=2.0e6
        )
        self.tau = np.linspace(0.0, 4.0e-6, 400)

    def test_starts_at_unity(self) -> None:
        trace = two_pulse_eseem(self.tau, self.coupling)
        self.assertAlmostEqual(trace[0], 1.0, places=12)

    def test_quantum_matches_analytic(self) -> None:
        analytic = two_pulse_eseem(self.tau, self.coupling)
        quantum = two_pulse_eseem_quantum(self.tau, self.coupling, electron_offset_hz=8.0e6)
        np.testing.assert_allclose(quantum, analytic, atol=1e-9)

    def test_no_modulation_without_anisotropy(self) -> None:
        isotropic = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=0.0)
        trace = two_pulse_eseem(self.tau, isotropic)
        np.testing.assert_allclose(trace, 1.0, atol=1e-12)

    def test_spectrum_peaks_at_nuclear_frequencies(self) -> None:
        trace = two_pulse_eseem(self.tau, self.coupling)
        freqs, spectrum = eseem_spectrum(self.tau, trace, zero_fill=8)
        nu_alpha, nu_beta = nuclear_frequencies(self.coupling)
        # The two strongest peaks are the basic nuclear frequencies.
        top = freqs[np.argsort(spectrum)[-2:]]
        recovered = np.sort(top)
        np.testing.assert_allclose(
            recovered, np.sort([nu_beta, nu_alpha]), rtol=0.02
        )


class ThreePulseEseemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=2.0e6
        )
        self.T = np.linspace(0.0, 8.0e-6, 400)
        self.tau = 0.2e-6

    def test_quantum_matches_analytic(self) -> None:
        analytic = three_pulse_eseem(self.T, self.coupling, tau_seconds=self.tau)
        quantum = three_pulse_eseem_quantum(self.T, self.coupling, tau_seconds=self.tau)
        np.testing.assert_allclose(quantum, analytic, atol=1e-9)

    def test_only_basic_frequencies_appear(self) -> None:
        # 3p-ESEEM shows nu_alpha and nu_beta but not their sum/difference.
        trace = three_pulse_eseem(self.T, self.coupling, tau_seconds=self.tau)
        freqs, spectrum = eseem_spectrum(self.T, trace, zero_fill=8)
        nu_alpha, nu_beta = nuclear_frequencies(self.coupling)
        sum_freq = nu_alpha + nu_beta
        # Intensity near the sum frequency should be negligible.
        near_sum = np.argmin(np.abs(freqs - sum_freq))
        self.assertLess(spectrum[near_sum], 0.05 * np.max(spectrum))

    def test_no_modulation_without_anisotropy(self) -> None:
        isotropic = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=0.0)
        trace = three_pulse_eseem(self.T, isotropic, tau_seconds=self.tau)
        np.testing.assert_allclose(trace, 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
