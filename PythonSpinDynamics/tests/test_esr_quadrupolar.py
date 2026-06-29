"""ESEEM/HYSCORE/ENDOR for quadrupolar nuclei (I=1, 3/2)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    cross_peak_positions,
    electron_nuclear_hamiltonian,
    eseem_spectrum,
    hyscore_signal,
    hyscore_spectrum,
    manifold_frequencies,
    nuclear_frequencies,
    two_pulse_eseem_quantum,
)


class CouplingValidationTests(unittest.TestCase):
    def test_spin_half_rejects_quadrupole(self) -> None:
        with self.assertRaises(ValueError):
            HyperfineCoupling(larmor_hz=1e6, nuclear_spin=0.5, quadrupole_hz=1e6)

    def test_eta_out_of_range_rejects(self) -> None:
        with self.assertRaises(ValueError):
            HyperfineCoupling(larmor_hz=1e6, nuclear_spin=1.0, quadrupole_hz=1e6, eta=1.5)

    def test_invalid_spin_rejects(self) -> None:
        with self.assertRaises(ValueError):
            HyperfineCoupling(larmor_hz=1e6, nuclear_spin=0.7)

    def test_hamiltonian_dimension_and_hermiticity(self) -> None:
        for spin, dim in ((1.0, 6), (1.5, 8)):
            coupling = HyperfineCoupling(
                larmor_hz=1.0e6, secular_hz=2.0e6, pseudosecular_hz=0.5e6,
                nuclear_spin=spin, quadrupole_hz=2.0e6, eta=0.3,
            )
            h = electron_nuclear_hamiltonian(coupling)
            self.assertEqual(h.shape, (dim, dim))
            np.testing.assert_allclose(h, h.conj().T, atol=1e-6)


class SpinHalfReductionTests(unittest.TestCase):
    def test_manifold_frequencies_match_closed_form(self) -> None:
        coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=2.0e6
        )
        alpha, beta = manifold_frequencies(coupling)
        # Each manifold has a single transition for I=1/2.
        self.assertEqual(alpha.size, 1)
        self.assertEqual(beta.size, 1)
        np.testing.assert_allclose(
            np.sort(np.concatenate([alpha, beta])),
            np.sort(nuclear_frequencies(coupling)),
            rtol=1e-9,
        )


class Spin1ExactCancellationTests(unittest.TestCase):
    """14N exact cancellation: one manifold becomes a pure quadrupole interaction."""

    def setUp(self) -> None:
        self.larmor = 1.05e6
        self.e2qQ_h = 3.5e6
        self.nu_q = 0.75 * self.e2qQ_h  # package convention nu_Q = (3/4) e2qQ/h
        self.eta = 0.5
        # A = 2*larmor cancels the nuclear Zeeman in the m_S = +1/2 manifold.
        self.coupling = HyperfineCoupling(
            larmor_hz=self.larmor,
            secular_hz=2.0 * self.larmor,
            pseudosecular_hz=0.0,
            nuclear_spin=1.0,
            quadrupole_hz=self.nu_q,
            eta=self.eta,
        )

    def test_cancellation_manifold_gives_pure_nqr_frequencies(self) -> None:
        nu_plus = self.nu_q * (1.0 + self.eta / 3.0)
        nu_minus = self.nu_q * (1.0 - self.eta / 3.0)
        nu_zero = (2.0 / 3.0) * self.nu_q * self.eta
        alpha, _beta = manifold_frequencies(self.coupling)
        np.testing.assert_allclose(
            alpha, np.sort([nu_zero, nu_minus, nu_plus]), rtol=1e-6
        )

    def test_eseem_spectrum_shows_cancellation_line(self) -> None:
        # Small pseudosecular term to give a clear modulation depth.
        coupling = HyperfineCoupling(
            larmor_hz=self.larmor, secular_hz=2.0 * self.larmor,
            pseudosecular_hz=0.15e6, nuclear_spin=1.0,
            quadrupole_hz=self.nu_q, eta=self.eta,
        )
        # 50 ns dwell -> 10 MHz Nyquist, above all lines.
        tau = np.arange(1200) * 50e-9
        trace = two_pulse_eseem_quantum(tau, coupling)
        freqs, spectrum = eseem_spectrum(tau, trace, zero_fill=4)
        nu_zero = (2.0 / 3.0) * self.nu_q * self.eta
        band = (freqs > 0.3e6) & (freqs < 1.5e6)
        peak = freqs[band][int(np.argmax(spectrum[band]))]
        self.assertAlmostEqual(peak, nu_zero, delta=0.1e6)


class Spin3HalfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=3.0e6, secular_hz=2.0e6, pseudosecular_hz=1.0e6,
            nuclear_spin=1.5, quadrupole_hz=1.5e6, eta=0.2,
        )

    def test_manifold_frequencies_are_finite_and_positive(self) -> None:
        alpha, beta = manifold_frequencies(self.coupling)
        # I=3/2 has up to 3 distinct transition frequencies per manifold.
        self.assertGreaterEqual(alpha.size, 1)
        self.assertTrue(np.all(alpha > 0) and np.all(np.isfinite(alpha)))
        self.assertTrue(np.all(beta > 0) and np.all(np.isfinite(beta)))

    def test_eseem_spectrum_peak_matches_a_manifold_frequency(self) -> None:
        tau = np.arange(1000) * 30e-9  # 16.7 MHz Nyquist
        trace = two_pulse_eseem_quantum(tau, self.coupling)
        freqs, spectrum = eseem_spectrum(tau, trace, zero_fill=4)
        alpha, beta = manifold_frequencies(self.coupling)
        all_lines = np.concatenate([alpha, beta])
        band = freqs > 0.3e6
        peak = freqs[band][int(np.argmax(spectrum[band]))]
        self.assertTrue(
            np.min(np.abs(all_lines - peak)) < 0.2e6,
            f"peak {peak/1e6:.3f} MHz not near a manifold line "
            f"{np.round(all_lines/1e6, 3)}",
        )


class Spin1HyscoreTests(unittest.TestCase):
    def test_cross_peaks_correlate_manifold_frequencies(self) -> None:
        coupling = HyperfineCoupling(
            larmor_hz=2.0e6, secular_hz=1.5e6, pseudosecular_hz=0.8e6,
            nuclear_spin=1.0, quadrupole_hz=1.0e6, eta=0.4,
        )
        grid = np.arange(64) * 40e-9  # 12.5 MHz Nyquist
        signal = hyscore_signal(grid, grid, coupling, tau_seconds=120e-9)
        spec = hyscore_spectrum(grid, grid, signal, zero_fill=4)
        peaks = cross_peak_positions(coupling)
        self.assertGreater(len(peaks), 0)

        f1, f2 = spec.frequencies1_hz, spec.frequencies2_hz
        global_max = float(spec.spectrum.max())

        def intensity_at(nu1: float, nu2: float) -> float:
            i = int(np.argmin(np.abs(f1 - nu1)))
            j = int(np.argmin(np.abs(f2 - nu2)))
            return float(spec.spectrum[i, j])

        # The off-diagonal alpha<->beta cross-peaks should carry real intensity
        # (diagonal/auto peaks are typically stronger, so this is a fractional
        # threshold rather than a global maximum).
        best_cross = max(intensity_at(a, b) for a, b in peaks)
        self.assertGreater(best_cross, 0.1 * global_max)


if __name__ == "__main__":
    unittest.main()
