"""Paper-grounded tests for coherent-basis-mapping ENDOR-QDyne."""

from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.nano_mr import (
    EndorQdyneProtocol,
    OpticalReadoutModel,
    initialization_infidelity_decay_rate,
    meinel_2023_endor_qdyne_protocol,
    simulate_endor_qdyne,
)


class EndorQdyneTests(unittest.TestCase):
    """Validate the model against Meinel et al., Commun. Phys. 6, 302."""

    def test_meinel_protocol_reproduces_reported_timing(self) -> None:
        protocol = meinel_2023_endor_qdyne_protocol()

        self.assertAlmostEqual(protocol.free_precession_seconds, 10.0e-6)
        self.assertAlmostEqual(protocol.sensing_seconds, 15.0e-6)
        self.assertAlmostEqual(protocol.rf_pi_over_two_seconds, 16.5e-6)
        self.assertAlmostEqual(protocol.rf_three_pi_over_two_seconds, 49.5e-6)
        self.assertAlmostEqual(protocol.rf_control_seconds, 66.0e-6)
        self.assertAlmostEqual(protocol.minimum_cycle_seconds, 105.4e-6)
        self.assertAlmostEqual(protocol.repetition_interval_seconds, 105.5e-6)
        self.assertAlmostEqual(protocol.longitudinal_hyperfine_hz, 6.0e3)
        self.assertAlmostEqual(protocol.rf_phase_increment_rad, np.pi / 2.0)

    def test_ideal_limit_matches_main_text_equations_1_and_2(self) -> None:
        reference = 2.0e6
        detuning = 1200.0
        protocol = EndorQdyneProtocol(
            free_precession_seconds=10.0e-6,
            sensing_seconds=15.0e-6,
            repetition_interval_seconds=100.0e-6,
            rf_reference_frequency_hz=reference,
            longitudinal_hyperfine_hz=6.0e3,
        )
        result = simulate_endor_qdyne(
            protocol,
            target_frequency_hz=reference + detuning,
            shot_count=32,
            include_measurement_backaction=False,
        )

        index = np.arange(32)
        phase = 2.0 * np.pi * detuning * index * 10.0e-6
        expected_iz = 0.5 * np.cos(phase)
        alpha = 2.0 * np.pi * 6.0e3 * 15.0e-6
        expected_sz = -0.5 * np.sin(alpha) * np.cos(phase)
        np.testing.assert_allclose(result.nuclear_z_expectation, expected_iz)
        np.testing.assert_allclose(result.sensor_z_expectation, expected_sz)

    def test_phase_cycle_reproduces_reported_2368_hz_carrier(self) -> None:
        reference = 2.5e6
        protocol = meinel_2023_endor_qdyne_protocol(
            rf_reference_frequency_hz=reference,
            sensor_initialization_fidelity=1.0,
        )
        result = simulate_endor_qdyne(
            protocol,
            target_frequency_hz=reference,
            shot_count=512,
            include_measurement_backaction=False,
        )

        expected = 1.0 / (4.0 * 105.5e-6)
        resolution = 1.0 / (512 * 105.5e-6)
        self.assertAlmostEqual(result.raw_cycle_frequency_hz, expected)
        self.assertAlmostEqual(result.expected_beat_frequency_hz, expected)
        self.assertLess(abs(result.estimated_beat_frequency_hz - expected), resolution)
        # Figure 3 reports the intended phase-cycled carrier as 2.368 kHz.
        self.assertLess(abs(expected - 2368.0), 2.0)

    def test_initialization_linewidth_uses_reported_hamiltonian_hz_convention(
        self,
    ) -> None:
        exact = initialization_infidelity_decay_rate(
            6.0e3,
            105.0e-6,
            0.9,
        )
        leading = initialization_infidelity_decay_rate(
            6.0e3,
            105.0e-6,
            0.9,
            leading_order=True,
        )

        # The paper's 2*pi*Azz*Sz*Iz Hamiltonian and reported near-1-kHz
        # estimate use spectroscopic Hz; nearby prose calling Azz*tau a phase
        # omits the 2*pi conversion and is conventionally inconsistent.
        self.assertAlmostEqual(exact, 1011.0956901733897)
        self.assertAlmostEqual(leading, 1012.1814471707744)
        self.assertLess(abs(exact - leading) / exact, 0.002)

    def test_backaction_uses_supplementary_equation_14(self) -> None:
        phase_step = 0.37
        protocol = EndorQdyneProtocol(
            free_precession_seconds=1.0e-6,
            sensing_seconds=5.0e-6,
            repetition_interval_seconds=20.0e-6,
            rf_reference_frequency_hz=0.0,
            longitudinal_hyperfine_hz=8.0e3,
            rf_phase_increment_rad=phase_step,
        )
        result = simulate_endor_qdyne(
            protocol,
            target_frequency_hz=0.0,
            shot_count=3,
            include_measurement_backaction=True,
        )

        cosine_alpha = np.cos(protocol.measurement_strength_rad)
        expected_third = 0.5 * (
            np.cos(phase_step) ** 2 - cosine_alpha * np.sin(phase_step) ** 2
        )
        self.assertAlmostEqual(result.nuclear_z_expectation[0], 0.5)
        self.assertAlmostEqual(
            result.nuclear_z_expectation[1],
            0.5 * np.cos(phase_step),
        )
        self.assertAlmostEqual(result.nuclear_z_expectation[2], expected_third)

    def test_zero_measurement_strength_has_no_estimated_peak(self) -> None:
        protocol = EndorQdyneProtocol(
            free_precession_seconds=10.0e-6,
            sensing_seconds=15.0e-6,
            repetition_interval_seconds=105.5e-6,
            rf_reference_frequency_hz=0.0,
            longitudinal_hyperfine_hz=0.0,
        )
        result = simulate_endor_qdyne(
            protocol,
            target_frequency_hz=0.0,
            shot_count=16,
            include_measurement_backaction=False,
        )

        np.testing.assert_array_equal(
            result.spectrum,
            np.zeros_like(result.spectrum),
        )
        self.assertTrue(np.isnan(result.estimated_beat_frequency_hz))

    def test_optical_record_and_input_validation(self) -> None:
        protocol = meinel_2023_endor_qdyne_protocol(sensor_initialization_fidelity=1.0)
        result = simulate_endor_qdyne(
            protocol,
            target_frequency_hz=0.0,
            shot_count=16,
            optical_model=OpticalReadoutModel(),
            seed=7,
        )
        self.assertIsNotNone(result.optical_readout)
        self.assertEqual(result.optical_readout.sampled_counts.shape, (16,))

        with self.assertRaisesRegex(ValueError, "shorter"):
            EndorQdyneProtocol(
                free_precession_seconds=10.0e-6,
                sensing_seconds=15.0e-6,
                repetition_interval_seconds=20.0e-6,
                rf_reference_frequency_hz=0.0,
                longitudinal_hyperfine_hz=6.0e3,
            )
        with self.assertRaisesRegex(ValueError, "at least 2"):
            simulate_endor_qdyne(
                protocol,
                target_frequency_hz=0.0,
                shot_count=1,
            )


if __name__ == "__main__":
    unittest.main()
