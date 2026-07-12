"""Focused validation of exact lab-frame versus RWA crossover pulses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    compare_rwa_to_lab_frame,
    scan_rwa_validity,
)


class RWAComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.17e6,
        )

    def _zero_field_error(self, nutation_hz: float) -> float:
        result = compare_rwa_to_lab_frame(
            self.site,
            (0.0, 0.0, 0.0),
            nutation_hz=nutation_hz,
            rf_frequency_hz=1.0e6,
            pulse_duration_seconds=20.0e-6,
            phase_radians=0.7,
            samples_per_carrier_cycle=160,
        )
        return result.maximum_element_error

    def test_weak_drive_error_has_bloch_siegert_scaling(self) -> None:
        ratio = self._zero_field_error(2.0e3) / self._zero_field_error(1.0e3)
        self.assertTrue(3.0 < ratio < 5.0, f"unexpected scaling ratio {ratio:.2f}")

    def test_lab_frame_sampling_is_converged(self) -> None:
        common = dict(
            nutation_hz=20.0e3,
            rf_frequency_hz=1.0e6,
            pulse_duration_seconds=12.0e-6,
            phase_radians=0.3,
        )
        coarse = compare_rwa_to_lab_frame(
            self.site, (0.0, 0.0, 0.0), samples_per_carrier_cycle=80, **common
        )
        fine = compare_rwa_to_lab_frame(
            self.site, (0.0, 0.0, 0.0), samples_per_carrier_cycle=160, **common
        )
        self.assertLess(
            abs(coarse.relative_response_error - fine.relative_response_error),
            2.0e-4,
        )

    def test_negative_gamma_preserves_the_rwa_phase_convention(self) -> None:
        negative_gamma = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=-4.17e6,
        )
        result = compare_rwa_to_lab_frame(
            negative_gamma,
            (0.0, 0.0, 0.0),
            nutation_hz=1.0e3,
            rf_frequency_hz=1.0e6,
            pulse_duration_seconds=20.0e-6,
            phase_radians=0.7,
            samples_per_carrier_cycle=160,
        )
        self.assertLess(result.relative_response_error, 2.0e-3)


class RWAValidityMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.17e6,
        )

    def test_scan_follows_the_nqr_nmr_band(self) -> None:
        interaction = np.array([0.01, 0.1, 1.0, 10.0])
        drive = np.array([0.001, 0.01])
        result = scan_rwa_validity(
            self.site,
            interaction,
            drive,
            b0_direction_pas=(1.0, 1.0, 1.0),
            b1_direction_pas=(1.0, -1.0, 0.0),
            duration_in_carrier_cycles=3.0,
            samples_per_carrier_cycle=80,
        )
        reference = np.maximum(1.0, interaction) * self.site.quadrupole_frequency_hz
        self.assertEqual(result.relative_response_error.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(result.relative_response_error)))
        self.assertTrue(np.all(result.carrier_frequencies_hz >= 0.5 * reference))
        self.assertTrue(np.all(result.minimum_target_isolation_hz > 0.0))

    def test_crossover_exposes_larger_single_band_rwa_error(self) -> None:
        result = scan_rwa_validity(
            self.site,
            [1.0, 10.0],
            [0.001],
            b0_direction_pas=(1.0, 1.0, 1.0),
            b1_direction_pas=(1.0, -1.0, 0.0),
            duration_in_carrier_cycles=4.0,
            samples_per_carrier_cycle=100,
        )
        crossover, high_field = result.relative_response_error[:, 0]
        self.assertGreater(crossover, 10.0 * high_field)


if __name__ == "__main__":
    unittest.main()
