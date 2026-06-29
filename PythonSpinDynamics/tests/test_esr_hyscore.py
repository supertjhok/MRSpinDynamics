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
    hyscore_signal,
    hyscore_spectrum,
    nuclear_frequencies,
)


class HyscoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=2.5e6
        )
        # 15 ns step -> 33 MHz Nyquist, above both nuclear frequencies.
        self.t1 = np.arange(72) * 15e-9
        self.t2 = np.arange(72) * 15e-9
        self.tau = 136e-9

    def test_cross_peak_positions_are_swapped_nuclear_frequencies(self) -> None:
        nu_alpha, nu_beta = nuclear_frequencies(self.coupling)
        positions = cross_peak_positions(self.coupling)
        self.assertEqual(positions, ((nu_alpha, nu_beta), (nu_beta, nu_alpha)))

    def test_signal_is_real_with_expected_shape(self) -> None:
        signal = hyscore_signal(self.t1, self.t2, self.coupling, tau_seconds=self.tau)
        self.assertEqual(signal.shape, (self.t1.size, self.t2.size))
        self.assertTrue(np.all(np.isfinite(signal)))

    def test_cross_peak_in_spectrum_matches_nuclear_frequencies(self) -> None:
        signal = hyscore_signal(self.t1, self.t2, self.coupling, tau_seconds=self.tau)
        spec = hyscore_spectrum(self.t1, self.t2, signal, zero_fill=4)
        nu_alpha, nu_beta = nuclear_frequencies(self.coupling)

        f1 = spec.frequencies1_hz
        f2 = spec.frequencies2_hz
        band1 = f1 > 2.0e6
        band2 = f2 > 2.0e6
        sub = spec.spectrum[np.ix_(band1, band2)]
        ii, jj = np.unravel_index(int(np.argmax(sub)), sub.shape)
        peak = (f1[band1][ii], f2[band2][jj])

        candidates = [(nu_alpha, nu_beta), (nu_beta, nu_alpha)]
        match = any(
            abs(peak[0] - a) < 0.3e6 and abs(peak[1] - b) < 0.3e6
            for a, b in candidates
        )
        self.assertTrue(match, f"peak {peak} not at a cross-peak {candidates}")

    def test_isotropic_coupling_has_no_cross_peaks(self) -> None:
        isotropic = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=0.0
        )
        signal = hyscore_signal(self.t1, self.t2, isotropic, tau_seconds=self.tau)
        # No pseudosecular coupling -> no modulation -> flat (constant) signal.
        self.assertLess(np.ptp(signal), 1e-9)

    def test_validation_errors(self) -> None:
        with self.assertRaises(ValueError):
            hyscore_signal([], self.t2, self.coupling, tau_seconds=self.tau)
        with self.assertRaises(ValueError):
            hyscore_signal(self.t1, self.t2, self.coupling, tau_seconds=-1.0)
        signal = hyscore_signal(self.t1, self.t2, self.coupling, tau_seconds=self.tau)
        with self.assertRaises(ValueError):
            hyscore_spectrum(self.t1, self.t2, signal[:, :3], zero_fill=4)


if __name__ == "__main__":
    unittest.main()
