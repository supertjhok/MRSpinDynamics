from __future__ import annotations

import unittest

import numpy as np

from spin_dynamics.absolute_phase import (
    AbsolutePhaseSpec,
    InterpolatedPulseShapeModel,
    LongitudinalPhaseKick,
    PulseShape,
    PulseShapeLibrary,
    SinusoidalTransientPerturbation,
    build_finite_cpmg_phase_schedule,
    build_finite_cpmg_pulse_plan,
    build_nonresonant_circuit_pulse_library,
    build_tuned_resonator_pulse_library,
    cpmg_refocus_start_times,
    nonresonant_dc_phase_perturbation,
)
from spin_dynamics.parameters import set_params_tuned_orig
from spin_dynamics.workflows import run_ideal_cpmg_train, run_tuned_cpmg_train


class AbsolutePhaseTests(unittest.TestCase):
    def test_absolute_phase_spec_wraps_pulse_phase(self) -> None:
        spec = AbsolutePhaseSpec(rf_frequency_hz=1.0, rf_phase_at_zero_rad=0.25)

        phase = spec.pulse_phase(1.0, 0.5)

        self.assertAlmostEqual(phase, 0.75)

    def test_cpmg_refocus_start_times_follow_echo_spacing(self) -> None:
        starts = cpmg_refocus_start_times(
            excitation_start_seconds=0.0,
            excitation_duration_seconds=10e-6,
            correction_delay_seconds=-2e-6,
            pre_refocus_delay_seconds=20e-6,
            echo_spacing_seconds=50e-6,
            num_echoes=3,
        )

        np.testing.assert_allclose(starts, [28e-6, 78e-6, 128e-6])

    def test_finite_cpmg_phase_schedule_tracks_all_pulses(self) -> None:
        spec = AbsolutePhaseSpec(rf_frequency_hz=0.25 / 50e-6)

        schedule = build_finite_cpmg_phase_schedule(
            spec=spec,
            excitation_start_seconds=1e-6,
            excitation_duration_seconds=10e-6,
            correction_delay_seconds=-2e-6,
            pre_refocus_delay_seconds=20e-6,
            echo_spacing_seconds=50e-6,
            num_echoes=3,
        )

        self.assertEqual(
            schedule.pulse_kind,
            ("excitation", "excitation", "refocusing", "refocusing", "refocusing"),
        )
        np.testing.assert_allclose(
            schedule.refocus_start_seconds,
            [29e-6, 79e-6, 129e-6],
        )
        np.testing.assert_allclose(
            schedule.pulse_start_seconds,
            [1e-6, 1e-6, 29e-6, 79e-6, 129e-6],
        )
        self.assertAlmostEqual(schedule.delta_refocus_phase_cycles, 0.25)

    def test_finite_cpmg_pulse_plan_allocates_per_echo_refocusing(self) -> None:
        plan = build_finite_cpmg_pulse_plan(3, per_echo_refocusing=True)

        self.assertEqual(plan.pulse_matrix_count, 5)
        np.testing.assert_array_equal(plan.excitation_cycle_one, [1, 0])
        np.testing.assert_array_equal(plan.excitation_cycle_two, [2, 0])
        np.testing.assert_array_equal(
            plan.refocus_cycle,
            [0, 3, 0, 0, 4, 0, 0, 5, 0],
        )

    def test_sinusoidal_perturbation_supports_half_periodicity(self) -> None:
        shape = PulseShape(
            duration=np.array([1.0]),
            phase=np.array([0.0]),
            amplitude=np.array([1.0]),
        )
        model = SinusoidalTransientPerturbation(
            phase_amplitude_rad=0.1,
            amplitude_fraction=0.2,
            periodicity="half",
            applies_to="refocusing",
        )

        first = model.apply(shape, np.pi / 4, "refocusing")
        second = model.apply(shape, np.pi / 4 + np.pi, "refocusing")

        np.testing.assert_allclose(first.phase, second.phase)
        np.testing.assert_allclose(first.amplitude, second.amplitude)

    def test_longitudinal_phase_kick_leaves_shape_and_returns_z_phase(self) -> None:
        shape = PulseShape(
            duration=np.array([1.0]),
            phase=np.array([0.0]),
            amplitude=np.array([1.0]),
        )
        model = LongitudinalPhaseKick(phase_amplitude_rad=0.043)

        kicked_shape = model.apply(shape, np.pi / 2, "refocusing")

        np.testing.assert_allclose(kicked_shape.phase, shape.phase)
        np.testing.assert_allclose(kicked_shape.amplitude, shape.amplitude)
        self.assertAlmostEqual(model.phase_kick(np.pi / 2, "refocusing"), 0.043)

    def test_pulse_shape_library_interpolates_complex_drive(self) -> None:
        library = PulseShapeLibrary(
            absolute_phase_rad=np.array([0.0, np.pi]),
            shapes={
                "refocusing": [
                    PulseShape(
                        duration=np.array([1.0]),
                        phase=np.array([np.pi - 0.1]),
                        amplitude=np.array([1.0]),
                    ),
                    PulseShape(
                        duration=np.array([1.0]),
                        phase=np.array([-np.pi + 0.1]),
                        amplitude=np.array([1.0]),
                    ),
                ]
            },
        )
        model = InterpolatedPulseShapeModel(library)
        base = PulseShape(
            duration=np.array([1.0]),
            phase=np.array([0.0]),
            amplitude=np.array([1.0]),
        )

        interpolated = model.apply(base, np.pi / 2, "refocusing")

        self.assertGreater(interpolated.amplitude[0], 0.9)
        self.assertGreater(abs(interpolated.phase[0]), 3.0)

    def test_ideal_cpmg_absolute_phase_tracking_preserves_default(self) -> None:
        baseline = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=3,
            rephase_action="ignore",
        )
        tracked = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=3,
            rephase_action="ignore",
            absolute_phase={"rf_frequency_hz": 1.0e6},
        )

        self.assertIsNotNone(tracked.absolute_phase)
        self.assertEqual(tracked.absolute_phase.pulse_matrix_count, 5)
        self.assertEqual(
            tracked.absolute_phase.pulse_kind,
            ("excitation", "excitation", "refocusing", "refocusing", "refocusing"),
        )
        self.assertIsNotNone(tracked.absolute_phase.pulse_absolute_phase_rad)
        np.testing.assert_array_equal(
            tracked.absolute_phase.excitation_matrix_indices,
            [1, 0],
        )
        np.testing.assert_array_equal(
            tracked.absolute_phase.refocus_matrix_indices,
            [0, 3, 0, 0, 4, 0, 0, 5, 0],
        )
        np.testing.assert_allclose(tracked.mrx, baseline.mrx, atol=1e-12, rtol=1e-12)

    def test_ideal_cpmg_absolute_phase_transient_changes_train(self) -> None:
        baseline = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
        )
        perturbed = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
            absolute_phase={
                "rf_frequency_hz": 0.25 / 200e-6,
                "transient_model": {
                    "kind": "sinusoidal",
                    "phase_amplitude_rad": 0.35,
                    "periodicity": "full",
                },
            },
        )

        self.assertIsNotNone(perturbed.absolute_phase)
        self.assertAlmostEqual(
            perturbed.absolute_phase.delta_refocus_phase_cycles,
            0.25,
        )
        self.assertFalse(np.allclose(perturbed.mrx, baseline.mrx))

    def test_tuned_cpmg_phase_resolved_shapes_have_pi_periodicity(
        self,
    ) -> None:
        _params, _sp, pp = set_params_tuned_orig(numpts=9)
        echo_spacing = float(np.sum(pp.tref))
        first_refocus = float(np.ravel(pp.texc)[0] + pp.tcorr + pp.tref[0])

        def absolute_phase(step_cycles: float) -> dict[str, float]:
            rf_frequency_hz = (
                1.0 if step_cycles == 0.0 else float(step_cycles)
            ) / echo_spacing
            return {
                "rf_frequency_hz": rf_frequency_hz,
                "rf_phase_at_zero_rad": -2.0 * np.pi * rf_frequency_hz * first_refocus,
            }

        synchronized = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=12,
            t1_seconds=1.0e9,
            t2_seconds=1.0e9,
            rephase_action="ignore",
            absolute_phase=absolute_phase(0.0),
        )
        half_cycle = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=12,
            t1_seconds=1.0e9,
            t2_seconds=1.0e9,
            rephase_action="ignore",
            absolute_phase=absolute_phase(0.5),
        )
        quarter_cycle = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=12,
            t1_seconds=1.0e9,
            t2_seconds=1.0e9,
            rephase_action="ignore",
            absolute_phase=absolute_phase(0.25),
        )

        self.assertIsNotNone(synchronized.absolute_phase)
        self.assertIsNotNone(half_cycle.absolute_phase)
        self.assertIsNotNone(quarter_cycle.absolute_phase)
        self.assertAlmostEqual(
            synchronized.absolute_phase.delta_refocus_phase_cycles,
            0.0,
        )
        self.assertAlmostEqual(
            half_cycle.absolute_phase.delta_refocus_phase_cycles,
            0.5,
        )
        self.assertAlmostEqual(
            quarter_cycle.absolute_phase.delta_refocus_phase_cycles,
            0.25,
        )
        denominator = np.trapezoid(
            np.abs(synchronized.echo) ** 2,
            synchronized.tvect,
            axis=1,
        )
        half_ratio = np.abs(
            np.trapezoid(
                half_cycle.echo * np.conj(synchronized.echo),
                half_cycle.tvect,
                axis=1,
            )
            / denominator
        )
        quarter_ratio = np.abs(
            np.trapezoid(
                quarter_cycle.echo * np.conj(synchronized.echo),
                quarter_cycle.tvect,
                axis=1,
            )
            / denominator
        )

        np.testing.assert_allclose(half_ratio, 1.0, atol=5e-3)
        self.assertGreater(float(np.max(np.abs(quarter_ratio - 1.0))), 0.02)

    def test_ideal_cpmg_accepts_absolute_phase_pulse_library(self) -> None:
        baseline = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
        )
        with_library = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
            absolute_phase={
                "rf_frequency_hz": 0.25 / 200e-6,
                "transient_model": {
                    "kind": "library",
                    "absolute_phase_rad": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2],
                    "shapes": {
                        "refocusing": {
                            "duration": [np.pi],
                            "phase": [[0.0], [0.15], [0.0], [-0.15]],
                            "amplitude": [[1.0], [0.8], [1.0], [1.2]],
                        }
                    },
                },
            },
        )

        self.assertIsNotNone(with_library.absolute_phase)
        self.assertEqual(
            with_library.absolute_phase.transient_model,
            "InterpolatedPulseShapeModel",
        )
        self.assertFalse(np.allclose(with_library.mrx, baseline.mrx))

    def test_nonresonant_circuit_builder_returns_finite_shapes(self) -> None:
        library = build_nonresonant_circuit_pulse_library(
            absolute_phase_rad=np.linspace(0.0, 2 * np.pi, 5, endpoint=False),
            rf_frequency_hz=1.0e6,
            pulse_duration_seconds=10e-6,
            post_delay_seconds=2e-6,
            time_scale_rad_per_s=1.0e5,
            tau_seconds=1.0e-6,
        )

        shape0 = library.shape("refocusing", 0.0)
        shape1 = library.shape("refocusing", np.pi / 2)

        self.assertGreater(shape0.duration.size, 1)
        self.assertTrue(np.all(np.isfinite(shape0.phase)))
        self.assertTrue(np.all(np.isfinite(shape0.amplitude)))
        self.assertFalse(np.allclose(shape0.amplitude, shape1.amplitude))

    def test_tuned_resonator_builder_has_half_period_symmetry(self) -> None:
        library = build_tuned_resonator_pulse_library(
            absolute_phase_rad=np.array([0.0, np.pi / 2]),
            rf_frequency_hz=1.0e6,
            resonant_frequency_hz=1.0e6,
            quality_factor=20.0,
            pulse_duration_seconds=8e-6,
            time_scale_rad_per_s=1.0e5,
        )

        shape0 = library.shape("refocusing", 0.25)
        shape_pi = library.shape("refocusing", 0.25 + np.pi)

        np.testing.assert_allclose(shape0.amplitude, shape_pi.amplitude)
        np.testing.assert_allclose(
            np.exp(1j * shape0.phase),
            np.exp(1j * shape_pi.phase),
        )

    def test_nonresonant_dc_phase_perturbation_uses_paper_estimate(self) -> None:
        model = nonresonant_dc_phase_perturbation(
            nutation_rate_rad_s=2.0e5,
            tau_seconds=4.0e-6,
            longitudinal_fraction=0.025,
        )

        self.assertAlmostEqual(model.phase_amplitude_rad, 0.04)
        self.assertEqual(model.periodicity, "full")
        self.assertIsInstance(model, LongitudinalPhaseKick)

    def test_ideal_cpmg_accepts_generated_nonresonant_library(self) -> None:
        baseline = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
        )
        library = build_nonresonant_circuit_pulse_library(
            absolute_phase_rad=np.linspace(0.0, 2 * np.pi, 8, endpoint=False),
            rf_frequency_hz=0.25 / 200e-6,
            pulse_duration_seconds=50e-6,
            time_scale_rad_per_s=(np.pi / 2) / 25e-6,
            tau_seconds=12e-6,
        )
        spec = AbsolutePhaseSpec(
            rf_frequency_hz=0.25 / 200e-6,
            transient_model=InterpolatedPulseShapeModel(library),
        )

        with_library = run_ideal_cpmg_train(
            numpts=17,
            num_echoes=4,
            rephase_action="ignore",
            absolute_phase=spec,
        )

        self.assertIsNotNone(with_library.absolute_phase)
        self.assertFalse(np.allclose(with_library.mrx, baseline.mrx))


if __name__ == "__main__":
    unittest.main()
