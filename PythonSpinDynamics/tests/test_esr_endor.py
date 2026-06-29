from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.esr import (  # noqa: E402
    HyperfineCoupling,
    davies_endor_spectrum,
    endor_frequencies,
    mims_blind_spot_factor,
    mims_endor_spectrum,
    nuclear_frequencies,
)


class EndorFrequencyTests(unittest.TestCase):
    def test_endor_frequencies_match_nuclear_frequencies(self) -> None:
        coupling = HyperfineCoupling(larmor_hz=14.5e6, secular_hz=3.0e6, pseudosecular_hz=2.0e6)
        self.assertEqual(endor_frequencies(coupling), nuclear_frequencies(coupling))


class DaviesEndorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=4.0e6, pseudosecular_hz=2.0e6
        )
        self.axis = np.linspace(8.0e6, 22.0e6, 4000)

    def test_peaks_sit_at_nuclear_frequencies(self) -> None:
        result = davies_endor_spectrum(self.axis, self.coupling, linewidth_hz=5.0e4)
        nu_alpha, nu_beta = nuclear_frequencies(self.coupling)
        # Local maxima near each predicted line.
        for nu in (nu_alpha, nu_beta):
            window = np.abs(self.axis - nu) < 3.0e5
            peak = self.axis[window][int(np.argmax(result.spectrum[window]))]
            self.assertAlmostEqual(peak, nu, delta=self.axis[1] - self.axis[0] + 1.0)

    def test_no_blind_spots_equal_intensities(self) -> None:
        result = davies_endor_spectrum(self.axis, self.coupling, linewidth_hz=5.0e4)
        self.assertEqual(result.line_intensities, (1.0, 1.0))

    def test_linewidth_validation(self) -> None:
        with self.assertRaises(ValueError):
            davies_endor_spectrum(self.axis, self.coupling, linewidth_hz=0.0)


class MimsEndorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coupling = HyperfineCoupling(
            larmor_hz=14.5e6, secular_hz=4.0e6, pseudosecular_hz=2.0e6
        )
        self.axis = np.linspace(8.0e6, 22.0e6, 4000)

    def test_blind_spot_factor_zero_at_integer_and_one_at_half_integer(self) -> None:
        nu = 16.0e6
        self.assertAlmostEqual(mims_blind_spot_factor(nu, 1.0 / nu), 0.0, places=12)
        self.assertAlmostEqual(mims_blind_spot_factor(nu, 2.0 / nu), 0.0, places=12)
        self.assertAlmostEqual(mims_blind_spot_factor(nu, 0.5 / nu), 1.0, places=12)

    def test_alpha_line_suppressed_at_its_blind_spot(self) -> None:
        nu_alpha, _ = nuclear_frequencies(self.coupling)
        blind_tau = 1.0 / nu_alpha
        result = mims_endor_spectrum(
            self.axis, self.coupling, tau_seconds=blind_tau, linewidth_hz=5.0e4
        )
        # The alpha line intensity (first) is the blind-spot-suppressed one.
        self.assertAlmostEqual(result.line_intensities[0], 0.0, places=9)

    def test_intensities_recover_away_from_blind_spots(self) -> None:
        nu_alpha, _ = nuclear_frequencies(self.coupling)
        good_tau = 0.5 / nu_alpha
        result = mims_endor_spectrum(
            self.axis, self.coupling, tau_seconds=good_tau, linewidth_hz=5.0e4
        )
        self.assertAlmostEqual(result.line_intensities[0], 1.0, places=9)

    def test_tau_validation(self) -> None:
        with self.assertRaises(ValueError):
            mims_endor_spectrum(self.axis, self.coupling, tau_seconds=0.0)


if __name__ == "__main__":
    unittest.main()
