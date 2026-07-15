"""Transient multi-winding electropermanent programming tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.electropermanent import (  # noqa: E402
    RemanenceState,
    variable_field_nmr_rod,
)
from spin_dynamics.fields.electropermanent_hysteresis import (  # noqa: E402
    PlayHysteresisModel,
)
from spin_dynamics.fields.electropermanent_pulses import (  # noqa: E402
    ProgrammingPulse,
    PulsePowerDriver,
)
from spin_dynamics.fields.electropermanent_transient import (  # noqa: E402
    MutualProgrammingCircuit,
    TransientCoupledEPMProgrammer,
)


def _driver(*, state_inductance_fraction: float = 0.0) -> PulsePowerDriver:
    return PulsePowerDriver(
        capacitance_f=2e-3,
        inductance_h=40e-6,
        coil_resistance_ohm=0.12,
        series_resistance_ohm=0.10,
        recovery_resistance_ohm=1.5,
        turns=60,
        magnetic_path_length_m=0.1016,
        current_limit_a=400.0,
        voltage_limit_v=650.0,
        state_inductance_fraction_at_saturation=state_inductance_fraction,
        coil_heat_capacity_j_per_k=50.0,
        driver_heat_capacity_j_per_k=25.0,
    )


def _model() -> PlayHysteresisModel:
    return PlayHysteresisModel(
        thresholds_a_per_m=(5e3, 10e3, 20e3, 40e3),
        weights=(0.2, 0.3, 0.3, 0.2),
        saturation_remanence_t=0.33,
        calibration_id="synthetic-transient-play",
    )


def _sources():
    return (
        variable_field_nmr_rod(
            center_m=(-0.04, 0.0, 0.0),
            effective_remanence_t=0.33,
        ),
        variable_field_nmr_rod(
            center_m=(0.04, 0.0, 0.0),
            effective_remanence_t=0.33,
        ),
    )


class MutualProgrammingCircuitTests(unittest.TestCase):
    def test_zero_coupling_reproduces_isolated_target_waveform(self) -> None:
        driver = _driver()
        circuit = MutualProgrammingCircuit.from_coupling_coefficients(
            (driver, driver),
            np.zeros((2, 2)),
        )
        pulse = ProgrammingPulse(100.0, 30e-6, polarity=-1)
        times = np.linspace(0.0, 180e-6, 901)
        states = (RemanenceState(0.0), RemanenceState(0.0))
        materials = tuple(source.material for source in _sources())
        array = circuit.simulate(
            0,
            times,
            pulse,
            states=states,
            materials=materials,
            max_step_s=0.3e-6,
        )
        isolated = driver.simulate(
            times,
            pulse,
            max_step_s=0.3e-6,
        )
        np.testing.assert_allclose(array.currents_a[:, 0], isolated.current_a, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(array.currents_a[:, 1], 0.0, atol=0.0)
        np.testing.assert_allclose(array.internal_h_a_per_m[:, 0], isolated.internal_h_a_per_m)

    def test_positive_mutual_inductance_opposes_neighbor_current_ramp(self) -> None:
        driver = _driver()
        circuit = MutualProgrammingCircuit.from_coupling_coefficients(
            (driver, driver),
            np.asarray([[0.0, 0.15], [0.15, 0.0]]),
        )
        pulse = ProgrammingPulse(50.0, 20e-6)
        waveform = circuit.simulate(
            0,
            np.linspace(0.0, 40e-6, 201),
            pulse,
            states=(RemanenceState(0.0), RemanenceState(0.0)),
            materials=tuple(source.material for source in _sources()),
        )
        self.assertGreater(waveform.currents_a[1, 0], 0.0)
        self.assertLess(waveform.currents_a[1, 1], 0.0)
        self.assertLess(waveform.mutual_induced_voltage_v[0, 1], 0.0)
        self.assertGreater(waveform.peak_mutual_induced_voltage_v[1], 0.0)

    def test_coupled_circuit_closes_electrical_energy_balance(self) -> None:
        driver = _driver()
        circuit = MutualProgrammingCircuit.from_coupling_coefficients(
            (driver, driver),
            np.asarray([[0.0, 0.12], [0.12, 0.0]]),
        )
        waveform = circuit.simulate(
            0,
            np.linspace(0.0, 350e-6, 3501),
            ProgrammingPulse(80.0, 25e-6),
            states=(RemanenceState(0.0), RemanenceState(0.0)),
            materials=tuple(source.material for source in _sources()),
            max_step_s=0.1e-6,
        )
        relative = abs(waveform.electrical_energy_balance_error_j) / waveform.initial_electrical_energy_j
        self.assertLess(relative, 2e-4)
        self.assertGreater(waveform.coil_energy_j[1], 0.0)
        self.assertGreater(waveform.driver_conduction_energy_j[1], 0.0)

    def test_circuit_rejects_nonphysical_coupling(self) -> None:
        driver = _driver()
        with self.assertRaises(ValueError):
            MutualProgrammingCircuit.from_coupling_coefficients(
                (driver, driver),
                np.asarray([[0.0, 1.1], [1.1, 0.0]]),
            )
        with self.assertRaises(ValueError):
            MutualProgrammingCircuit.from_coupling_coefficients(
                (driver, driver),
                np.asarray([[0.0, 0.1], [0.0, 0.0]]),
            )


class TransientProgrammingTests(unittest.TestCase):
    def _programmer(
        self,
        *,
        mutual_coefficient: float,
        leakage_fraction: float,
        state_inductance_fraction: float = 0.0,
    ) -> TransientCoupledEPMProgrammer:
        driver = _driver(state_inductance_fraction=state_inductance_fraction)
        circuit = MutualProgrammingCircuit.from_coupling_coefficients(
            (driver, driver),
            np.asarray(
                [
                    [0.0, mutual_coefficient],
                    [mutual_coefficient, 0.0],
                ]
            ),
            field_coupling_fractions=np.asarray(
                [
                    [1.0, leakage_fraction],
                    [leakage_fraction, 1.0],
                ]
            ),
        )
        return TransientCoupledEPMProgrammer(
            sources=_sources(),
            hysteresis_models=(_model(), _model()),
            remanence_coupling_a_per_m_per_t=np.zeros((2, 2)),
            circuit=circuit,
        )

    def test_uncoupled_neighbor_retains_its_state(self) -> None:
        programmer = self._programmer(mutual_coefficient=0.0, leakage_fraction=0.0)
        result = programmer.program(
            0,
            np.linspace(0.0, 180e-6, 901),
            ProgrammingPulse(100.0, 30e-6, polarity=-1),
            relaxation=1.0,
            max_step_s=0.3e-6,
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.remanence_change_t[1], 0.0)
        self.assertEqual(result.disturbed_indices, ())

    def test_transient_crosstalk_disturbs_neighbor_return_point_state(self) -> None:
        programmer = self._programmer(mutual_coefficient=0.08, leakage_fraction=0.20)
        result = programmer.program(
            0,
            np.linspace(0.0, 180e-6, 901),
            ProgrammingPulse(100.0, 30e-6, polarity=-1),
            relaxation=1.0,
            max_step_s=0.3e-6,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.disturbed_indices, (1,))
        self.assertLess(result.remanence_change_t[1], 0.0)
        self.assertGreater(result.waveform.peak_current_a[1], 0.0)
        self.assertGreater(result.waveform.peak_internal_h_a_per_m[1], 5e3)

    def test_state_dependent_inductance_outer_iteration_converges(self) -> None:
        programmer = self._programmer(
            mutual_coefficient=0.05,
            leakage_fraction=0.10,
            state_inductance_fraction=0.25,
        )
        result = programmer.program(
            0,
            np.linspace(0.0, 180e-6, 601),
            ProgrammingPulse(90.0, 30e-6, polarity=-1),
            tolerance_t=2e-6,
            relaxation=0.8,
            max_step_s=0.4e-6,
        )
        self.assertTrue(result.converged)
        self.assertGreater(result.iterations, 1)
        self.assertLessEqual(result.residual_t, 2e-6)
        self.assertFalse(
            np.allclose(
                np.diag(result.waveform.inductance_matrix_h),
                [_driver().inductance_h, _driver().inductance_h],
            )
        )

    def test_programmer_validates_dimensions(self) -> None:
        driver = _driver()
        circuit = MutualProgrammingCircuit.from_coupling_coefficients(
            (driver, driver),
            np.zeros((2, 2)),
        )
        with self.assertRaises(ValueError):
            TransientCoupledEPMProgrammer(
                sources=_sources(),
                hysteresis_models=(_model(),),
                remanence_coupling_a_per_m_per_t=np.zeros((2, 2)),
                circuit=circuit,
            )


if __name__ == "__main__":
    unittest.main()
