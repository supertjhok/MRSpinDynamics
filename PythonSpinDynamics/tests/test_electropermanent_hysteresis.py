"""Return-point memory and neighbor-coupled EPM programming tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.electropermanent import (  # noqa: E402
    MU0,
    RemanenceState,
    variable_field_nmr_rod,
)
from spin_dynamics.fields.electropermanent_hysteresis import (  # noqa: E402
    CoupledEPMProgrammer,
    PlayHysteresisModel,
    ReturnPointState,
    illustrative_alnico_return_point_model,
    neighbor_coupling_matrix,
)
from spin_dynamics.fields.electropermanent_pulses import (  # noqa: E402
    ProgrammingPulse,
    archived_igbt_pulse_cases,
)


def _simple_model() -> PlayHysteresisModel:
    return PlayHysteresisModel(
        thresholds_a_per_m=(1.0, 2.0, 4.0),
        weights=(1.0, 1.0, 1.0),
        saturation_remanence_t=1.0,
        calibration_id="synthetic-play",
        uncertainty_t=0.01,
    )


class PlayHysteresisTests(unittest.TestCase):
    def test_return_to_nested_reversal_restores_operator_state_exactly(self) -> None:
        model = _simple_model()
        initial = model.initialize(
            RemanenceState(1.0, branch="positive_saturation")
        )
        first_return = model.propagate(initial, (0.0, -3.0)).final_state
        closed_loop = model.propagate(first_return, (1.0, -3.0)).final_state
        np.testing.assert_allclose(
            closed_loop.play_outputs_a_per_m,
            first_return.play_outputs_a_per_m,
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(
            closed_loop.remanence.remanence_t,
            first_return.remanence.remanence_t,
        )
        self.assertGreaterEqual(len(closed_loop.reversal_stack_a_per_m), 1)

    def test_exceeding_outer_return_point_wipes_inner_memory(self) -> None:
        model = _simple_model()
        initial = model.initialize(
            RemanenceState(1.0, branch="positive_saturation")
        )
        inner = model.propagate(initial, (0.0, -3.0, 1.0)).final_state
        exceeded = model.propagate(inner, (-5.0, 1.0)).final_state
        self.assertFalse(
            np.allclose(
                exceeded.play_outputs_a_per_m,
                inner.play_outputs_a_per_m,
            )
        )
        self.assertNotEqual(
            exceeded.remanence.remanence_t,
            inner.remanence.remanence_t,
        )

    def test_major_loop_reaches_symmetric_retained_saturation(self) -> None:
        model = _simple_model()
        initial = RemanenceState(-1.0, branch="negative_saturation")
        trace = model.propagate(initial, (0.0, 20.0, 0.0, -20.0, 0.0))
        self.assertAlmostEqual(trace.remanence_t[2], 1.0)
        self.assertAlmostEqual(trace.remanence_t[4], -1.0)
        self.assertEqual(trace.final_state.remanence.branch, "negative_saturation")

    def test_temperature_coefficient_scales_saturation_endpoint(self) -> None:
        model = PlayHysteresisModel(
            thresholds_a_per_m=(1.0, 2.0),
            weights=(1.0, 1.0),
            saturation_remanence_t=0.33,
            calibration_id="temperature-test",
            reference_temperature_k=293.15,
            remanence_temperature_coefficient_per_k=-2e-4,
        )
        initial = model.initialize(
            RemanenceState(0.33, branch="positive_saturation")
        )
        trace = model.propagate(initial, (0.0,), temperatures_k=303.15)
        self.assertAlmostEqual(trace.final_state.remanence.remanence_t, 0.33 * 0.998)
        self.assertEqual(trace.final_state.remanence.temperature_k, 303.15)

    def test_illustrative_preset_is_explicitly_inferred(self) -> None:
        model = illustrative_alnico_return_point_model()
        self.assertEqual(model.evidence[0].classification, "inferred")
        self.assertIn("illustrative", model.calibration_id)
        self.assertAlmostEqual(sum(model.weights), 1.0)

    def test_model_and_state_validation_reject_malformed_inputs(self) -> None:
        with self.assertRaises(ValueError):
            PlayHysteresisModel((2.0, 1.0), (1.0, 1.0), 1.0, "bad")
        with self.assertRaises(ValueError):
            PlayHysteresisModel((1.0, 2.0), (1.0,), 1.0, "bad")
        with self.assertRaises(ValueError):
            ReturnPointState(RemanenceState(0.0), ())


class NeighborCouplingTests(unittest.TestCase):
    def _side_by_side_sources(self, neighbor_remanence_t: float = 0.33):
        target = variable_field_nmr_rod(
            center_m=(-0.04, 0.0, 0.0),
            effective_remanence_t=0.33,
        )
        neighbor = variable_field_nmr_rod(
            center_m=(0.04, 0.0, 0.0),
            effective_remanence_t=abs(neighbor_remanence_t),
        ).with_state(
            RemanenceState(
                neighbor_remanence_t,
                branch=(
                    "positive_saturation"
                    if neighbor_remanence_t > 0
                    else "negative_saturation"
                ),
            )
        )
        return target, neighbor

    def test_geometry_coupling_is_symmetric_and_equatorial_field_is_opposed(self) -> None:
        sources = self._side_by_side_sources()
        coupling = neighbor_coupling_matrix(
            sources,
            self_demagnetizing_factors=(0.1, 0.2),
            n_cross=5,
            n_length=31,
        )
        self.assertAlmostEqual(coupling[0, 0], -0.1 / MU0)
        self.assertAlmostEqual(coupling[1, 1], -0.2 / MU0)
        self.assertLess(coupling[0, 1], 0.0)
        self.assertAlmostEqual(coupling[0, 1], coupling[1, 0], delta=1e-10)

    def test_saturated_neighbor_changes_partial_programming_result(self) -> None:
        positive_sources = self._side_by_side_sources(+0.33)
        negative_sources = self._side_by_side_sources(-0.33)
        positive_coupling = neighbor_coupling_matrix(
            positive_sources,
            n_cross=5,
            n_length=31,
        )
        negative_coupling = neighbor_coupling_matrix(
            negative_sources,
            n_cross=5,
            n_length=31,
        )
        case = archived_igbt_pulse_cases()[0]
        driver = case.driver
        model = illustrative_alnico_return_point_model()
        pulse = ProgrammingPulse(100.0, 30e-6, polarity=-1)
        times = np.linspace(0.0, 180e-6, 1001)

        positive_system = CoupledEPMProgrammer(
            sources=positive_sources,
            drivers=(driver, driver),
            hysteresis_models=(model, model),
            coupling_a_per_m_per_t=positive_coupling,
        )
        negative_system = CoupledEPMProgrammer(
            sources=negative_sources,
            drivers=(driver, driver),
            hysteresis_models=(model, model),
            coupling_a_per_m_per_t=negative_coupling,
        )
        positive = positive_system.program(
            0,
            times,
            pulse,
            relaxation=1.0,
        )
        negative = negative_system.program(
            0,
            times,
            pulse,
            relaxation=1.0,
        )
        self.assertTrue(positive.converged)
        self.assertTrue(negative.converged)
        self.assertLess(positive.neighbor_bias_a_per_m, 0.0)
        self.assertGreater(negative.neighbor_bias_a_per_m, 0.0)
        self.assertLess(
            positive.final_states[0].remanence.remanence_t,
            negative.final_states[0].remanence.remanence_t,
        )
        self.assertEqual(
            positive.final_sources[1].state,
            positive_sources[1].state,
        )

    def test_self_demagnetizing_fixed_point_converges(self) -> None:
        sources = self._side_by_side_sources(+0.33)
        coupling = neighbor_coupling_matrix(
            sources,
            self_demagnetizing_factors=(0.02, 0.0),
            n_cross=3,
            n_length=15,
        )
        case = archived_igbt_pulse_cases()[0]
        model = illustrative_alnico_return_point_model()
        system = CoupledEPMProgrammer(
            sources=sources,
            drivers=(case.driver, case.driver),
            hysteresis_models=(model, model),
            coupling_a_per_m_per_t=coupling,
        )
        result = system.program(
            0,
            np.linspace(0.0, 180e-6, 801),
            ProgrammingPulse(100.0, 30e-6, polarity=-1),
            relaxation=0.5,
            tolerance_t=1e-6,
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.residual_t, 1e-6)
        self.assertGreater(result.iterations, 1)

    def test_coupled_programmer_validates_dimensions_and_index(self) -> None:
        sources = self._side_by_side_sources()
        case = archived_igbt_pulse_cases()[0]
        model = illustrative_alnico_return_point_model()
        with self.assertRaises(ValueError):
            CoupledEPMProgrammer(
                sources=sources,
                drivers=(case.driver,),
                hysteresis_models=(model, model),
                coupling_a_per_m_per_t=np.zeros((2, 2)),
            )
        system = CoupledEPMProgrammer(
            sources=sources,
            drivers=(case.driver, case.driver),
            hysteresis_models=(model, model),
            coupling_a_per_m_per_t=np.zeros((2, 2)),
        )
        with self.assertRaises(ValueError):
            system.program(
                2,
                np.linspace(0.0, 100e-6, 101),
                ProgrammingPulse(50.0, 20e-6),
            )


if __name__ == "__main__":
    unittest.main()
