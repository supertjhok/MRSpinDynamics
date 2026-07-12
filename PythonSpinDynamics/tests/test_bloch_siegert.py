from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.workflows.bloch_siegert import (
    counter_rotating_common_phase,
    estimate_larmor_hz_from_counter_rotating_phase,
    estimate_nutation_hz,
    mandal_multislice_correction,
    simulate_bloch_siegert_phase_sweep,
)


class BlochSiegertWorkflowTests(unittest.TestCase):
    def test_mandal_2014_phase_slope_and_t90_are_reproduced(self):
        durations = np.arange(40.0e-6, 400.0e-6 + 1e-12, 40.0e-6)
        t90_s = 102.0e-6
        result = simulate_bloch_siegert_phase_sweep(
            durations,
            larmor_hz=1.48e6,
            offset_hz=25.0e3,
            nutation_hz=1.0 / (4.0 * t90_s),
        )

        self.assertAlmostEqual(
            float(np.degrees(result.differential_phase_rad[-1])),
            35.0,
            delta=1.0,
        )
        np.testing.assert_allclose(
            result.differential_phase_rad,
            result.rwa_differential_phase_rad,
            rtol=0.025,
            atol=np.deg2rad(0.15),
        )
        fitted_nutation = estimate_nutation_hz(
            result.differential_phase_rad[-1],
            durations[-1],
            offset_hz=25.0e3,
        )
        fitted_t90 = 1.0 / (4.0 * float(fitted_nutation))
        self.assertAlmostEqual(fitted_t90, t90_s, delta=2.0e-6)

    def test_common_phase_is_counter_rotating_and_inverts_larmor(self):
        duration = 400.0e-6
        larmor = np.array([60.0e3, 100.0e3, 180.0e3, 1.48e6])
        common = counter_rotating_common_phase(
            larmor,
            duration,
            offset_hz=25.0e3,
            nutation_hz=1.0 / (4.0 * 102.0e-6),
        )
        recovered = estimate_larmor_hz_from_counter_rotating_phase(
            common,
            duration,
            offset_hz=25.0e3,
            nutation_hz=1.0 / (4.0 * 102.0e-6),
        )

        np.testing.assert_allclose(recovered, larmor, rtol=1e-12)
        self.assertGreater(common[0], 10.0 * common[-1])

    def test_exact_low_frequency_common_phase_exposes_rwa_limit(self):
        result = simulate_bloch_siegert_phase_sweep(
            np.array([400.0e-6]),
            larmor_hz=100.0e3,
            offset_hz=25.0e3,
            nutation_hz=1.0 / (4.0 * 102.0e-6),
            steps_per_cycle=64,
        )

        self.assertGreater(abs(float(result.common_mode_phase_rad[0])), 0.05)
        self.assertAlmostEqual(
            float(result.common_mode_phase_rad[0]),
            float(result.second_order_common_mode_phase_rad[0]),
            delta=0.01,
        )

    def test_mandal_multislice_phase_and_timing_corrections(self):
        t90_s = 140.0e-6
        correction = mandal_multislice_correction(
            4,
            nutation_hz=1.0 / (4.0 * t90_s),
            slice_spacing_hz=20.0e3,
            t90_s=t90_s,
        )

        self.assertAlmostEqual(
            float(np.degrees(correction.excitation_phase_error_rad[1])),
            -4.0179,
            places=3,
        )
        np.testing.assert_allclose(
            correction.excitation_phase_error_rad
            + correction.excitation_phase_correction_rad,
            0.0,
        )
        self.assertAlmostEqual(correction.excitation_timing_shift_s[1], t90_s)
        self.assertAlmostEqual(
            correction.excitation_timing_shift_s[2], 1.25 * t90_s
        )


if __name__ == "__main__":
    unittest.main()
