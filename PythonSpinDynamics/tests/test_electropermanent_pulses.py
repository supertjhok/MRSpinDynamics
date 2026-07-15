"""Electrical, thermal, and empirical-state tests for EPM programming pulses."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.electropermanent import (  # noqa: E402
    ALNICO5_AC500,
    ProgrammingCoil,
    RemanenceState,
)
from spin_dynamics.fields.electropermanent_pulses import (  # noqa: E402
    EmpiricalProgrammingCalibration,
    ProgrammingPulse,
    PulsePowerDriver,
    PulseThermalState,
    archived_igbt_pulse_cases,
    published_demagnetization_calibration,
)


def _driver(**overrides) -> PulsePowerDriver:
    parameters = dict(
        capacitance_f=1.0e-3,
        inductance_h=100.0e-6,
        coil_resistance_ohm=0.3,
        series_resistance_ohm=0.2,
        recovery_resistance_ohm=2.0,
        turns=50,
        magnetic_path_length_m=0.1,
        coil_heat_capacity_j_per_k=100.0,
        coil_thermal_conductance_w_per_k=1.0,
        driver_heat_capacity_j_per_k=50.0,
        driver_thermal_conductance_w_per_k=1.0,
    )
    parameters.update(overrides)
    return PulsePowerDriver(**parameters)


class PulseCircuitTests(unittest.TestCase):
    def test_drive_matches_exact_underdamped_series_rlc_current(self) -> None:
        driver = _driver(
            coil_heat_capacity_j_per_k=1.0e12,
            coil_thermal_conductance_w_per_k=0.0,
            driver_heat_capacity_j_per_k=1.0e12,
            driver_thermal_conductance_w_per_k=0.0,
        )
        pulse = ProgrammingPulse(100.0, 150.0e-6)
        times = np.linspace(0.0, 100.0e-6, 101)
        waveform = driver.simulate(times, pulse, max_step_s=0.05e-6)

        resistance = driver.coil_resistance_ohm + driver.series_resistance_ohm
        alpha = resistance / (2.0 * driver.inductance_h)
        omega_d = np.sqrt(
            1.0 / (driver.inductance_h * driver.capacitance_f) - alpha**2
        )
        exact = (
            pulse.capacitor_voltage_v
            / (driver.inductance_h * omega_d)
            * np.exp(-alpha * times)
            * np.sin(omega_d * times)
        )
        np.testing.assert_allclose(waveform.current_a, exact, rtol=2e-8, atol=1e-10)

    def test_negative_bridge_polarity_reverses_current_and_field(self) -> None:
        driver = _driver()
        times = np.linspace(0.0, 300.0e-6, 501)
        positive = driver.simulate(times, ProgrammingPulse(80.0, 50.0e-6, 1))
        negative = driver.simulate(times, ProgrammingPulse(80.0, 50.0e-6, -1))
        np.testing.assert_allclose(negative.current_a, -positive.current_a)
        np.testing.assert_allclose(
            negative.internal_h_a_per_m,
            -positive.internal_h_a_per_m,
        )

    def test_current_limit_clamps_and_voltage_limit_rejects(self) -> None:
        driver = _driver(current_limit_a=20.0, voltage_limit_v=100.0)
        times = np.linspace(0.0, 200.0e-6, 401)
        waveform = driver.simulate(times, ProgrammingPulse(100.0, 100.0e-6))
        self.assertLessEqual(waveform.peak_current_a, 20.0 + 1e-12)
        with self.assertRaises(ValueError):
            driver.simulate(times, ProgrammingPulse(101.0, 100.0e-6))

    def test_recovery_path_decays_current_and_closes_energy_balance(self) -> None:
        driver = _driver()
        pulse = ProgrammingPulse(100.0, 50.0e-6)
        recovery_tau = driver.inductance_h / (
            driver.coil_resistance_ohm + driver.recovery_resistance_ohm
        )
        times = np.linspace(0.0, pulse.duration_s + 12.0 * recovery_tau, 3001)
        waveform = driver.simulate(times, pulse)
        peak_index = int(np.argmax(np.abs(waveform.current_a)))
        recovery = np.abs(waveform.current_a[peak_index:])
        self.assertTrue(np.all(np.diff(recovery) <= 1e-10))
        self.assertLess(abs(waveform.current_a[-1]), 1e-3 * waveform.peak_current_a)
        dissipated = waveform.coil_energy_j + waveform.driver_conduction_energy_j
        self.assertLess(
            abs(waveform.electrical_energy_balance_error_j),
            0.015 * dissipated,
        )

    def test_state_dependent_inductance_and_bias_field_are_explicit(self) -> None:
        driver = _driver(state_inductance_fraction_at_saturation=0.2)
        state = RemanenceState(0.5 * ALNICO5_AC500.remanence_t)
        self.assertAlmostEqual(
            driver.inductance_for_state(state, ALNICO5_AC500),
            1.1 * driver.inductance_h,
        )
        times = np.linspace(0.0, 100.0e-6, 101)
        waveform = driver.simulate(
            times,
            ProgrammingPulse(50.0, 20.0e-6),
            state=state,
            material=ALNICO5_AC500,
            bias_field_a_per_m=-1200.0,
        )
        self.assertAlmostEqual(waveform.inductance_h, 1.1 * driver.inductance_h)
        self.assertAlmostEqual(waveform.internal_h_a_per_m[0], -1200.0)

    def test_programming_coil_adapter_requires_complete_electrical_data(self) -> None:
        complete = ProgrammingCoil(
            turns=60,
            inductance_h=50.0e-6,
            resistance_ohm=0.24,
            winding_length_m=0.1016,
        )
        driver = PulsePowerDriver.from_programming_coil(
            complete,
            capacitance_f=10.0e-3,
        )
        self.assertEqual(driver.turns, 60)
        self.assertAlmostEqual(driver.inductance_h, 50.0e-6)
        with self.assertRaises(ValueError):
            PulsePowerDriver.from_programming_coil(
                ProgrammingCoil(turns=10),
                capacitance_f=1.0e-3,
                magnetic_path_length_m=0.1,
            )


class PulseArchiveRegressionTests(unittest.TestCase):
    def test_archived_peak_current_cases_match_configuration_specific_fits(self) -> None:
        cases = archived_igbt_pulse_cases()
        self.assertEqual(len(cases), 3)
        for case in cases:
            waveform = case.run()
            relative_error = abs(
                waveform.peak_current_a - case.measured_peak_current_a
            ) / case.measured_peak_current_a
            self.assertLess(relative_error, case.relative_tolerance)
            self.assertEqual(case.evidence[0].classification, "measured")
            self.assertEqual(case.evidence[1].classification, "inferred")


class PulseThermalTests(unittest.TestCase):
    def test_pulse_heating_accumulates_and_zero_current_cooling_is_exact(self) -> None:
        driver = _driver(
            switching_energy_per_transition_j=0.01,
            coil_heat_capacity_j_per_k=2.0,
            driver_heat_capacity_j_per_k=1.0,
        )
        times = np.linspace(0.0, 500.0e-6, 1001)
        initial = PulseThermalState(
            cumulative_coil_energy_j=0.5,
            cumulative_driver_energy_j=0.25,
        )
        waveform = driver.simulate(
            times,
            ProgrammingPulse(100.0, 80.0e-6),
            initial_thermal_state=initial,
        )
        final = waveform.final_thermal_state
        self.assertGreater(final.coil_temperature_k, driver.ambient_temperature_k)
        self.assertGreater(final.driver_temperature_k, driver.ambient_temperature_k)
        self.assertAlmostEqual(
            final.cumulative_coil_energy_j,
            0.5 + waveform.coil_energy_j,
        )
        self.assertAlmostEqual(
            final.cumulative_driver_energy_j,
            0.25 + waveform.driver_conduction_energy_j + 0.02,
        )
        cooled = driver.cool_thermal_state(final, 2.0)
        expected = driver.ambient_temperature_k + (
            final.coil_temperature_k - driver.ambient_temperature_k
        ) * np.exp(-2.0 * driver.coil_thermal_conductance_w_per_k / 2.0)
        self.assertAlmostEqual(cooled.coil_temperature_k, expected)
        self.assertEqual(
            cooled.cumulative_coil_energy_j,
            final.cumulative_coil_energy_j,
        )


class EmpiricalProgrammingTests(unittest.TestCase):
    def _negative_waveform(self, command_fraction: float = 0.17):
        driver = _driver()
        pulse = ProgrammingPulse(
            100.0,
            50.0e-6,
            polarity=-1,
            command_fraction=command_fraction,
        )
        times = np.linspace(0.0, 300.0e-6, 601)
        return pulse, driver.simulate(times, pulse)

    def test_published_protocol_zero_crossing_updates_state_and_history(self) -> None:
        pulse, waveform = self._negative_waveform(0.17)
        calibration = published_demagnetization_calibration()
        initial = RemanenceState(0.33, branch="positive_saturation")
        result = calibration.apply(initial, waveform, pulse)
        self.assertAlmostEqual(result.final_state.remanence_t, 0.0)
        self.assertEqual(result.final_state.branch, "partial")
        self.assertEqual(
            result.final_state.calibration_id,
            "published-demag-envelope-v1",
        )
        self.assertAlmostEqual(result.final_state.uncertainty_t, 0.04)
        self.assertEqual(len(result.final_state.reversal_fields_a_per_m), 1)
        self.assertLess(result.final_state.reversal_fields_a_per_m[0], 0.0)
        self.assertEqual(result.final_state.evidence[0].classification, "inferred")

    def test_published_protocol_rejects_wrong_history_polarity_and_range(self) -> None:
        calibration = published_demagnetization_calibration()
        pulse, waveform = self._negative_waveform(0.2)
        with self.assertRaises(ValueError):
            calibration.apply(RemanenceState(0.1, branch="partial"), waveform, pulse)

        positive = ProgrammingPulse(100.0, 50.0e-6, polarity=1, command_fraction=0.2)
        positive_waveform = _driver().simulate(
            np.linspace(0.0, 300.0e-6, 601), positive
        )
        with self.assertRaises(ValueError):
            calibration.apply(
                RemanenceState(0.33, branch="positive_saturation"),
                positive_waveform,
                positive,
            )

        out_of_range = ProgrammingPulse(
            100.0,
            50.0e-6,
            polarity=-1,
            command_fraction=None,
        )
        out_waveform = _driver().simulate(
            np.linspace(0.0, 300.0e-6, 601), out_of_range
        )
        with self.assertRaises(ValueError):
            calibration.apply(
                RemanenceState(0.33, branch="positive_saturation"),
                out_waveform,
                out_of_range,
            )

    def test_waveform_metric_calibration_uses_realized_peak_field(self) -> None:
        driver = _driver()
        pulse = ProgrammingPulse(100.0, 50.0e-6)
        waveform = driver.simulate(np.linspace(0.0, 300.0e-6, 601), pulse)
        peak = waveform.peak_internal_h_a_per_m
        calibration = EmpiricalProgrammingCalibration(
            command_values=(0.0, peak, 2.0 * peak),
            remanence_t=(0.0, 0.2, 0.3),
            metric="peak_internal_h",
            calibration_id="synthetic-peak-h",
            uncertainty_t=0.01,
        )
        result = calibration.apply(RemanenceState(0.0), waveform)
        self.assertAlmostEqual(result.command_value, peak)
        self.assertAlmostEqual(result.final_state.remanence_t, 0.2)

    def test_calibration_validation_rejects_unsorted_or_mismatched_tables(self) -> None:
        with self.assertRaises(ValueError):
            EmpiricalProgrammingCalibration(
                command_values=(0.0, 0.0),
                remanence_t=(0.0, 0.1),
                metric="duty_fraction",
                calibration_id="bad",
                uncertainty_t=0.1,
            )
        with self.assertRaises(ValueError):
            EmpiricalProgrammingCalibration(
                command_values=(0.0, 1.0),
                remanence_t=(0.0,),
                metric="duty_fraction",
                calibration_id="bad",
                uncertainty_t=0.1,
            )


if __name__ == "__main__":
    unittest.main()
