"""Optimization and MATLAB result-layout fixture checks."""

from tests.octave_fixtures.support import *


class OctaveOptimizationFixtureTests(OctaveFixtureBase):
    def test_spa_pulse_catalog_matches_matlab(self) -> None:
        pulses = spa_pulse_list()
        self.assertEqual(len(pulses), 10)
        self.assertEqual([pulse.phases.size for pulse in pulses], [9, 10, 13, 20, 21, 31, 35, 39, 47, 55])
        np.testing.assert_allclose(pulses[0].phases / np.pi, [1, 1, 0, 1, 0, 1, 0, 1, 1])
        np.testing.assert_allclose(pulses[-1].pulse_length_t180, 5.5)
        np.testing.assert_allclose(rectangular_refocusing_lengths(), [0.6, 0.8, 1.0])
    def test_spa_metrics_match_matlab_normalization(self) -> None:
        spa_snr = np.linspace(0.7, 1.6, 10)
        rect_snr = np.array([0.75, 0.9, 1.0])
        metrics = evaluate_spa_metrics(spa_snr, rect_snr)

        expected_lengths = np.array([0.6, 0.8, 0.9, 1.0, 1.3, 2.0, 2.1, 3.1, 3.5, 3.9, 4.7, 5.5])
        expected_echo = 6.0 + expected_lengths
        expected_snr = np.concatenate([rect_snr[:2], spa_snr])
        expected_fom_time = expected_echo / expected_snr**2 / 7.0
        expected_fom_energy = expected_echo * expected_lengths / expected_snr**2 / 7.0

        np.testing.assert_allclose(metrics.pulse_length_t180, expected_lengths)
        np.testing.assert_allclose(metrics.echo_spacing_t180, expected_echo)
        np.testing.assert_allclose(metrics.snr, expected_snr)
        np.testing.assert_allclose(metrics.fom_time, expected_fom_time)
        np.testing.assert_allclose(metrics.fom_energy, expected_fom_energy)
        self.assertEqual(metrics.labels[0], "rect0.6")
        self.assertEqual(metrics.labels[-1], "spa10")
    def test_tuned_refocusing_evaluation_matches_lower_level_call(self) -> None:
        result = evaluate_tuned_refocusing_pulse(np.zeros(6), numpts=17)
        params, sp, pp = set_params_tuned_spa(numpts=17)
        texc = pp.T_90 / 6.0
        params = params.__class__(
            **{
                **params.__dict__,
                "aexc": np.array([6.0], dtype=np.float64),
                "texc": np.array([texc], dtype=np.float64),
                "pref": np.zeros(6),
                "aref": np.ones(6),
                "tref": pp.T_180 * 0.1 * np.ones(6),
            }
        )
        pp = pp.__class__(**{**pp.__dict__, "tcorr": -(2 / np.pi) * texc})
        expected_mrx, expected_masy, expected_snr = calc_masy_tuned_probe_lp_orig(
            params,
            sp,
            pp,
        )

        np.testing.assert_allclose(result.del_w, sp.del_w)
        np.testing.assert_allclose(result.mrx, expected_mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.masy, expected_masy, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.snr, expected_snr, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.pulse_length_t180, 0.6)
        self.assertEqual(result.echo.shape, result.tvect.shape)
    def test_tuned_refocusing_evaluation_accepts_spa_catalog_pulse(self) -> None:
        pulse = spa_pulse_list()[0]
        result = evaluate_tuned_refocusing_pulse(pulse.phases, numpts=17)

        self.assertEqual(result.mrx.shape, (17,))
        self.assertEqual(result.masy.shape, (17,))
        self.assertTrue(np.isfinite(result.snr))
        self.assertGreater(result.snr, 0)
        np.testing.assert_allclose(result.pulse_length_t180, pulse.pulse_length_t180)
    def test_untuned_refocusing_evaluation_matches_lower_level_call(self) -> None:
        result = evaluate_untuned_refocusing_pulse(np.zeros(6), numpts=17)
        params, sp, pp = set_params_untuned_spa(numpts=17)
        texc = pp.T_90 / 6.0
        params = params.__class__(
            **{
                **params.__dict__,
                "aexc": np.array([6.0], dtype=np.float64),
                "texc": np.array([texc], dtype=np.float64),
                "pref": np.zeros(6),
                "aref": np.ones(6),
                "tref": pp.T_180 * 0.1 * np.ones(6),
            }
        )
        pp = pp.__class__(**{**pp.__dict__, "tcorr": -(2 / np.pi) * texc})
        expected_mrx, expected_masy, expected_snr = calc_masy_untuned_probe_lp(
            params,
            sp,
            pp,
        )

        np.testing.assert_allclose(result.del_w, sp.del_w)
        np.testing.assert_allclose(result.mrx, expected_mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.masy, expected_masy, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.snr, expected_snr, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.pulse_length_t180, 0.6)
        self.assertEqual(result.echo.shape, result.tvect.shape)
    def test_untuned_refocusing_evaluation_accepts_spa_catalog_pulse(self) -> None:
        pulse = spa_pulse_list()[0]
        result = evaluate_untuned_refocusing_pulse(pulse.phases, numpts=17)

        self.assertEqual(result.mrx.shape, (17,))
        self.assertEqual(result.masy.shape, (17,))
        self.assertTrue(np.isfinite(result.snr))
        self.assertGreater(result.snr, 0)
        np.testing.assert_allclose(result.pulse_length_t180, pulse.pulse_length_t180)
    def test_matched_refocusing_evaluation_matches_lower_level_call(self) -> None:
        result = evaluate_matched_refocusing_pulse(np.zeros(6), numpts=9)
        sp, pp = set_params_matched_spa(numpts=9)
        texc = pp.T_90 / 6.0
        pp = pp.__class__(
            **{
                **pp.__dict__,
                "aexc": np.array([6.0], dtype=np.float64),
                "texc": np.array([texc], dtype=np.float64),
                "tcorr": -(2 / np.pi) * texc,
                "pref": np.zeros(6, dtype=np.float64),
                "aref": np.ones(6, dtype=np.float64),
                "tref": pp.T_180 * 0.1 * np.ones(6, dtype=np.float64),
            }
        )
        expected_mrx, expected_masy, expected_snr = calc_masy_matched_probe_orig(sp, pp)

        np.testing.assert_allclose(result.del_w, sp.del_w)
        np.testing.assert_allclose(result.mrx, expected_mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.masy, expected_masy, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.snr, expected_snr, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.pulse_length_t180, 0.6)
        self.assertEqual(result.echo.shape, result.tvect.shape)
    def test_matched_refocusing_evaluation_accepts_spa_catalog_pulse(self) -> None:
        pulse = spa_pulse_list()[0]
        result = evaluate_matched_refocusing_pulse(pulse.phases, numpts=9)

        self.assertEqual(result.mrx.shape, (9,))
        self.assertEqual(result.masy.shape, (9,))
        self.assertTrue(np.isfinite(result.snr))
        self.assertGreater(result.snr, 0)
        np.testing.assert_allclose(result.pulse_length_t180, pulse.pulse_length_t180)
    def test_tuned_spa_summary_returns_matlab_style_metrics(self) -> None:
        result = summarize_tuned_spa_refocusing(numpts=9, pulse_indices=[1, 2])

        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.rectangular_snr.shape, (3,))
        self.assertEqual(result.spa_snr.shape, (2,))
        np.testing.assert_allclose(result.pulse_indices, [1, 2])
        self.assertEqual(result.metrics.labels, ("rect0.6", "rect0.8", "spa1", "spa2"))
        np.testing.assert_allclose(
            result.metrics.snr[:2],
            result.rectangular_snr[:2] / result.rectangular_snr[-1],
        )
        np.testing.assert_allclose(
            result.metrics.snr[2:],
            result.spa_snr / result.rectangular_snr[-1],
        )
        self.assertTrue(np.all(np.isfinite(result.metrics.fom_time)))
    def test_untuned_spa_summary_returns_matlab_style_metrics(self) -> None:
        result = summarize_untuned_spa_refocusing(numpts=9, pulse_indices=[1])

        self.assertEqual(result.probe, "untuned")
        self.assertEqual(result.rectangular_snr.shape, (3,))
        self.assertEqual(result.spa_snr.shape, (1,))
        self.assertEqual(result.metrics.labels, ("rect0.6", "rect0.8", "spa1"))
        self.assertTrue(np.all(np.isfinite(result.metrics.snr)))
        self.assertTrue(np.all(result.metrics.fom_energy > 0))
    def test_matched_spa_summary_accepts_selected_catalog_subset(self) -> None:
        result = summarize_matched_spa_refocusing(numpts=5, pulse_indices=[1])

        self.assertEqual(result.probe, "matched")
        self.assertEqual(result.rectangular_snr.shape, (3,))
        self.assertEqual(result.spa_snr.shape, (1,))
        self.assertEqual(result.metrics.labels, ("rect0.6", "rect0.8", "spa1"))
        self.assertTrue(np.all(np.isfinite(result.metrics.snr)))
    def test_spa_phase_optimizer_improves_synthetic_objective(self) -> None:
        target = np.array([0.0, np.pi, np.pi, 0.0])

        def score(phases: np.ndarray) -> float:
            return -float(np.sum((phases - target) ** 2))

        result = optimize_spa_phase_program(np.zeros(4), score, max_passes=2)

        np.testing.assert_allclose(result.best_phases, target)
        self.assertGreater(result.best_score, result.history_scores[0])
        self.assertTrue(result.improved)
        self.assertGreater(result.iterations, 0)
    def test_tuned_refocusing_phase_optimizer_runs_small_search(self) -> None:
        initial = np.array([0.0, 0.35])
        result = optimize_tuned_refocusing_phases(
            initial,
            numpts=7,
            initial_step=0.2,
            max_passes=1,
            optimizer="pattern",
        )

        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.best_phases.shape, initial.shape)
        self.assertEqual(result.best_evaluation.phases.shape, initial.shape)
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.snr)
    def test_untuned_refocusing_phase_optimizer_respects_bounds(self) -> None:
        result = optimize_untuned_refocusing_phases(
            [-10.0, 10.0],
            numpts=7,
            bounds=(-0.5, 0.5),
            initial_step=0.2,
            max_passes=1,
            optimizer="pattern",
        )

        self.assertEqual(result.probe, "untuned")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertTrue(np.all(result.best_phases >= -0.5))
        self.assertTrue(np.all(result.best_phases <= 0.5))
        np.testing.assert_allclose(result.bounds, (-0.5, 0.5))
    def test_matched_refocusing_phase_optimizer_runs_tiny_search(self) -> None:
        original = refocusing_module.evaluate_matched_refocusing_pulse

        def fake_evaluator(phases: np.ndarray, **kwargs: object) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                phases=phase_arr,
                snr=1.0 - float(np.sum((phase_arr - 0.1) ** 2)),
            )

        refocusing_module.evaluate_matched_refocusing_pulse = fake_evaluator
        try:
            result = optimize_matched_refocusing_phases(
                [0.0],
                numpts=5,
                initial_step=0.1,
                max_passes=1,
                optimizer="pattern",
            )
        finally:
            refocusing_module.evaluate_matched_refocusing_pulse = original

        self.assertEqual(result.probe, "matched")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertEqual(result.best_phases.shape, (1,))
        self.assertEqual(result.best_evaluation.phases.shape, (1,))
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.snr)
    def test_ideal_v0crit_refocusing_evaluation_returns_metrics(self) -> None:
        phases = np.array([0.0, np.pi / 2, np.pi])
        result = evaluate_ideal_v0crit_refocusing_pulse(
            phases,
            numpts=21,
            segment_fraction=0.2,
        )

        self.assertEqual(result.del_w.shape, (21,))
        self.assertEqual(result.neff.shape, (3, 21))
        self.assertEqual(result.alpha.shape, (21,))
        self.assertEqual(result.v0crit.shape, (21,))
        self.assertEqual(result.masy.shape, (21,))
        self.assertEqual(result.phases.shape, phases.shape)
        self.assertGreater(result.axis_rms, 0)
        self.assertTrue(np.isfinite(result.v0crit_average))
        np.testing.assert_allclose(
            result.score,
            result.axis_rms + result.v0crit_average,
        )
        np.testing.assert_allclose(result.snr, result.score)
        np.testing.assert_allclose(result.pulse_length_t180, 0.6)
    def test_ideal_time_varying_excitation_vector_returns_shape(self) -> None:
        mexc = ideal_time_varying_excitation_vector(numpts=21, maxoffs=4.0)

        self.assertEqual(mexc.shape, (3, 21))
        self.assertTrue(np.all(np.isfinite(mexc)))
        self.assertGreater(float(np.max(np.abs(mexc[0, :]))), 0.5)
    def test_ideal_v0crit_excited_refocusing_evaluation_returns_metrics(self) -> None:
        phases = np.array([0.0, np.pi / 2, np.pi])
        result = evaluate_ideal_v0crit_excited_refocusing_pulse(
            phases,
            numpts=21,
            segment_fraction=0.2,
        )
        mexc = ideal_time_varying_excitation_vector(numpts=21, maxoffs=4.0)
        baseline = evaluate_ideal_v0crit_refocusing_pulse(
            phases,
            numpts=21,
            maxoffs=4.0,
            segment_fraction=0.2,
            excitation_vector=mexc,
        )

        self.assertEqual(result.del_w.shape, (21,))
        self.assertEqual(result.neff.shape, (3, 21))
        self.assertEqual(result.masy.shape, (21,))
        self.assertGreater(result.axis_rms, 0)
        self.assertTrue(np.isfinite(result.v0crit_average))
        np.testing.assert_allclose(result.masy, baseline.masy)
        np.testing.assert_allclose(result.score, baseline.score)
    def test_ideal_v0crit_excited_refocusing_uses_matlab_dot_convention(
        self,
    ) -> None:
        phases = np.array([0.0, np.pi / 2])
        mexc = np.ones((3, 9), dtype=np.complex128)
        mexc[0, :] += 1j * np.linspace(-1.0, 1.0, 9)
        result = evaluate_ideal_v0crit_refocusing_pulse(
            phases,
            numpts=9,
            maxoffs=2.0,
            segment_fraction=0.25,
            excitation_vector=mexc,
        )
        window = refocusing_module._normalized_sinc_window(result.del_w, 3 * np.pi)
        transverse = result.neff[0, :] + 1j * result.neff[1, :]
        expected_raw = np.sum(np.conj(mexc) * result.neff, axis=0) * transverse
        expected = np.convolve(expected_raw, window, mode="same")

        np.testing.assert_allclose(result.masy, expected)
    def test_ideal_v0crit_refocusing_optimizer_runs_small_search(self) -> None:
        initial = np.array([0.0, np.pi])
        result = optimize_ideal_v0crit_refocusing_phases(
            initial,
            numpts=21,
            segment_fraction=0.2,
            initial_step=0.2,
            max_passes=1,
            optimizer="pattern",
        )

        self.assertEqual(result.probe, "ideal_v0crit")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.best_phases.shape, initial.shape)
        self.assertEqual(result.best_evaluation.phases.shape, initial.shape)
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.score)
    def test_ideal_v0crit_excited_refocusing_optimizer_runs_small_search(self) -> None:
        initial = np.array([0.0, np.pi])
        result = optimize_ideal_v0crit_excited_refocusing_phases(
            initial,
            numpts=21,
            segment_fraction=0.2,
            initial_step=0.2,
            max_passes=1,
            optimizer="pattern",
        )

        self.assertEqual(result.probe, "ideal_v0crit_excited")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.best_phases.shape, initial.shape)
        self.assertEqual(result.best_evaluation.phases.shape, initial.shape)
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.score)
    def test_ideal_time_varying_refocusing_evaluation_returns_metrics(self) -> None:
        phases = np.array([0.0, np.pi / 2])
        result = evaluate_ideal_time_varying_refocusing_pulse(
            phases,
            field_offsets=np.array([0.0, 0.5, 0.0]),
            numpts=9,
            segment_fraction=0.5,
        )

        self.assertEqual(result.del_w.shape, (9,))
        self.assertEqual(result.field_offsets.shape, (3,))
        self.assertEqual(result.mrx.shape, (9,))
        self.assertEqual(result.echo.shape, result.tvect.shape)
        self.assertEqual(result.reference_echo.shape, result.tvect.shape)
        self.assertEqual(result.phases.shape, phases.shape)
        self.assertTrue(np.isfinite(result.score))
        self.assertTrue(np.isfinite(result.matched_signal))
        np.testing.assert_allclose(result.snr, result.score)
        np.testing.assert_allclose(result.pulse_length_t180, 1.0)
    def test_ideal_time_varying_refocusing_optimizer_runs_small_search(self) -> None:
        initial = np.array([0.0, np.pi])
        result = optimize_ideal_time_varying_refocusing_phases(
            initial,
            field_offsets=np.array([0.0, 0.5, 0.0]),
            numpts=9,
            segment_fraction=0.5,
            initial_step=0.2,
            max_passes=1,
            optimizer="pattern",
        )

        self.assertEqual(result.probe, "ideal_time_varying")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertTrue(result.optimizer_success)
        self.assertEqual(result.best_phases.shape, initial.shape)
        self.assertEqual(result.best_evaluation.phases.shape, initial.shape)
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.score)
    def test_tuned_excitation_evaluation_returns_finite_snr(self) -> None:
        del_w = np.linspace(-10.0, 10.0, 7)
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        result = evaluate_tuned_excitation_pulse([0.0], neff, numpts=7)

        self.assertEqual(result.del_w.shape, (7,))
        self.assertEqual(result.mrx.shape, (7,))
        self.assertEqual(result.masy.shape, (7,))
        self.assertEqual(result.phases.shape, (1,))
        self.assertEqual(result.echo.shape, result.tvect.shape)
        self.assertTrue(np.isfinite(result.snr))
        np.testing.assert_allclose(result.pulse_length_t180, 0.1)
        np.testing.assert_allclose(result.neff, neff)
    def test_tuned_excitation_phase_shift_matches_matlab_fixture(self) -> None:
        table = np.loadtxt(
            FIXTURES / "optimization_tuned_excitation_phase_shift.csv",
            delimiter=",",
        )
        del_w = table[:, 0]
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        phases = np.array([0.2, 1.1, 2.4])
        shifted_phases = np.mod(phases + np.pi, 2 * np.pi)

        result = evaluate_tuned_excitation_pulse(phases, neff, numpts=del_w.size)
        shifted = evaluate_tuned_excitation_pulse(
            shifted_phases,
            neff,
            numpts=del_w.size,
        )
        mrx_ref = table[:, 3] + 1j * table[:, 4]
        shifted_mrx_ref = table[:, 7] + 1j * table[:, 8]

        np.testing.assert_allclose(result.del_w, del_w)
        np.testing.assert_allclose(
            result.masy,
            table[:, 1] + 1j * table[:, 2],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(result.mrx, mrx_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.snr, table[0, 9], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            shifted.masy,
            table[:, 5] + 1j * table[:, 6],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            shifted.mrx,
            shifted_mrx_ref,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(shifted.snr, table[0, 10], rtol=1e-14, atol=1e-14)
        residual_ratio = trapezoid(np.abs(result.mrx + shifted.mrx), del_w)
        residual_ratio /= trapezoid(np.abs(result.mrx), del_w)
        self.assertGreater(residual_ratio, 1.0)
    def test_tuned_excitation_optimizer_matches_compact_matlab_result(self) -> None:
        table = np.loadtxt(
            FIXTURES / "optimization_tuned_excitation_result.csv",
            delimiter=",",
        )
        del_w = np.linspace(-10.0, 10.0, 9)
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        initial_phases = table[0, 2:]
        matlab_best_phases = table[1, 2:]

        initial_eval = evaluate_tuned_excitation_pulse(
            initial_phases,
            neff,
            numpts=del_w.size,
        )
        matlab_best_eval = evaluate_tuned_excitation_pulse(
            matlab_best_phases,
            neff,
            numpts=del_w.size,
        )
        result = optimize_tuned_excitation_phases(
            initial_phases,
            neff,
            numpts=del_w.size,
            optimizer="pattern",
            max_passes=8,
        )

        np.testing.assert_allclose(initial_eval.snr, table[0, 1], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            matlab_best_eval.snr,
            table[1, 1],
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertGreater(matlab_best_eval.snr, initial_eval.snr)
        self.assertGreater(result.best_score, result.initial_score)
        self.assertGreaterEqual(result.best_score, table[1, 1] - 2e-4)
    def test_tuned_inverse_optimizer_matches_compact_matlab_objective(self) -> None:
        exc_table = np.loadtxt(
            FIXTURES / "optimization_tuned_excitation_result.csv",
            delimiter=",",
        )
        inv_table = np.loadtxt(
            FIXTURES / "optimization_tuned_inverse_result.csv",
            delimiter=",",
        )
        del_w = np.linspace(-10.0, 10.0, 9)
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        target = evaluate_tuned_excitation_pulse(
            exc_table[1, 2:],
            neff,
            numpts=del_w.size,
        )
        initial_phases = inv_table[0, 4:]
        matlab_best_phases = inv_table[1, 4:]

        initial_eval = evaluate_tuned_inverse_excitation_pulse(
            initial_phases,
            neff,
            target.mrx,
            target.snr,
            numpts=del_w.size,
        )
        matlab_best_eval = evaluate_tuned_inverse_excitation_pulse(
            matlab_best_phases,
            neff,
            target.mrx,
            target.snr,
            numpts=del_w.size,
        )
        result = optimize_tuned_inverse_excitation_phases(
            initial_phases,
            neff,
            target.mrx,
            target.snr,
            numpts=del_w.size,
            optimizer="pattern",
            max_passes=8,
        )

        initial_ratio = trapezoid(np.abs(target.mrx + initial_eval.excitation.mrx), del_w)
        initial_ratio /= trapezoid(np.abs(target.mrx), del_w)
        matlab_best_ratio = trapezoid(
            np.abs(target.mrx + matlab_best_eval.excitation.mrx),
            del_w,
        )
        matlab_best_ratio /= trapezoid(np.abs(target.mrx), del_w)

        np.testing.assert_allclose(
            initial_eval.mismatch,
            inv_table[0, 1],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(initial_ratio, inv_table[0, 2], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            matlab_best_eval.mismatch,
            inv_table[1, 1],
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            matlab_best_ratio,
            inv_table[1, 2],
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertLess(matlab_best_eval.mismatch, initial_eval.mismatch)
        self.assertLess(result.best_evaluation.mismatch, initial_eval.mismatch)
        self.assertLessEqual(result.best_evaluation.mismatch, inv_table[1, 1] + 2e-3)
    def test_tuned_inverse_excitation_evaluation_matches_objective(self) -> None:
        del_w = np.linspace(-10.0, 10.0, 7)
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        target = evaluate_tuned_excitation_pulse([0.0], neff, numpts=7)
        result = evaluate_tuned_inverse_excitation_pulse(
            [np.pi],
            neff,
            target.mrx,
            target.snr,
            numpts=7,
        )
        expected = trapezoid(np.abs(target.mrx + result.excitation.mrx), result.excitation.del_w)
        expected += 0.8 * abs(result.snr - target.snr)

        self.assertEqual(result.phases.shape, (1,))
        self.assertEqual(result.target_mrx.shape, (7,))
        self.assertTrue(np.isfinite(result.mismatch))
        np.testing.assert_allclose(result.mismatch, expected)
        np.testing.assert_allclose(result.neff, neff)
    def test_tuned_excitation_phase_optimizer_runs_small_search(self) -> None:
        original = excitation_module.evaluate_tuned_excitation_pulse

        def fake_evaluator(
            phases: np.ndarray,
            neff: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                phases=phase_arr,
                neff=np.asarray(neff, dtype=np.complex128),
                snr=1.0 - float(np.sum((phase_arr - 0.2) ** 2)),
            )

        excitation_module.evaluate_tuned_excitation_pulse = fake_evaluator
        try:
            neff = np.zeros((3, 5), dtype=np.float64)
            neff[0, :] = 1.0
            result = optimize_tuned_excitation_phases(
                [0.0],
                neff,
                numpts=5,
                initial_step=0.2,
                max_passes=1,
                optimizer="pattern",
            )
        finally:
            excitation_module.evaluate_tuned_excitation_pulse = original

        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertEqual(result.best_phases.shape, (1,))
        self.assertEqual(result.best_evaluation.phases.shape, (1,))
        self.assertGreaterEqual(result.best_score, result.initial_score)
        self.assertEqual(result.history_scores.size, result.iterations + 1)
        self.assertGreater(result.iterations, 0)
        self.assertTrue(np.all(np.isfinite(result.history_scores)))
        np.testing.assert_allclose(result.best_score, result.best_evaluation.snr)
    def test_tuned_inverse_excitation_optimizer_runs_small_search(self) -> None:
        original = excitation_module.evaluate_tuned_inverse_excitation_pulse

        def fake_evaluator(
            phases: np.ndarray,
            neff: np.ndarray,
            target_mrx: np.ndarray,
            target_snr: float,
            **kwargs: object,
        ) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                phases=phase_arr,
                neff=np.asarray(neff, dtype=np.complex128),
                target_mrx=np.asarray(target_mrx, dtype=np.complex128),
                target_snr=float(target_snr),
                snr=1.0,
                mismatch=float(np.sum((phase_arr - 0.3) ** 2)),
            )

        excitation_module.evaluate_tuned_inverse_excitation_pulse = fake_evaluator
        try:
            neff = np.zeros((3, 5), dtype=np.float64)
            neff[0, :] = 1.0
            result = optimize_tuned_inverse_excitation_phases(
                [0.0],
                neff,
                np.zeros(5, dtype=np.complex128),
                1.0,
                numpts=5,
                initial_step=0.3,
                max_passes=1,
                optimizer="pattern",
            )
        finally:
            excitation_module.evaluate_tuned_inverse_excitation_pulse = original

        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.optimizer_method, "pattern")
        self.assertGreaterEqual(result.best_score, result.initial_score)
        np.testing.assert_allclose(result.best_phases, [0.3])
        np.testing.assert_allclose(result.best_score, -result.best_evaluation.mismatch)
    def test_tuned_excitation_optimizer_rejects_invalid_inputs(self) -> None:
        neff = np.zeros((3, 5), dtype=np.float64)
        with self.assertRaises(ValueError):
            optimize_tuned_excitation_phases([], neff, numpts=5)
        with self.assertRaises(ValueError):
            optimize_tuned_excitation_phases([0.0], neff, numpts=5, bounds=(1.0, 1.0))
        with self.assertRaises(ValueError):
            optimize_tuned_excitation_phases([0.0], np.zeros((3, 5)), numpts=4)
        with self.assertRaises(ValueError):
            optimize_tuned_inverse_excitation_phases(
                [0.0],
                neff,
                np.zeros(4, dtype=np.complex128),
                1.0,
                numpts=5,
            )
    def test_scipy_optimizer_backend_requires_optional_dependency(self) -> None:
        if importlib.util.find_spec("scipy") is not None:
            self.skipTest("SciPy is installed in this environment")
        with self.assertRaises(ImportError):
            optimize_tuned_refocusing_phases(
                [0.0],
                numpts=7,
                optimizer="scipy",
            )
    def test_random_phase_starts_are_seeded_and_bounded(self) -> None:
        starts_a = random_phase_starts(3, 2, bounds=(-0.25, 0.5), seed=123)
        starts_b = random_phase_starts(3, 2, bounds=(-0.25, 0.5), seed=123)
        default_starts = random_phase_starts(8, 3, seed=456)

        self.assertEqual(starts_a.shape, (3, 2))
        np.testing.assert_allclose(starts_a, starts_b)
        self.assertTrue(np.all(starts_a >= -0.25))
        self.assertTrue(np.all(starts_a <= 0.5))
        self.assertTrue(np.all(default_starts >= 0.0))
        self.assertTrue(np.all(default_starts <= 2 * np.pi))
        with self.assertRaises(ValueError):
            random_phase_starts(0, 2)
        with self.assertRaises(ValueError):
            random_phase_starts(1, 2, seed=1, rng=np.random.default_rng(1))
    def test_tuned_refocusing_multistart_selects_best_result(self) -> None:
        original = driver_module.refocusing_module.optimize_tuned_refocusing_phases

        def fake_optimizer(phases: np.ndarray, **kwargs: object) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                best_score=float(np.sum(phase_arr)),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.refocusing_module.optimize_tuned_refocusing_phases = fake_optimizer
        try:
            starts = np.array([[0.0, 0.1], [0.3, 0.4], [-0.1, 0.2]])
            result = run_tuned_refocusing_multistart(
                2,
                initial_phases=starts,
                bounds=(-1.0, 1.0),
            )
        finally:
            driver_module.refocusing_module.optimize_tuned_refocusing_phases = original

        self.assertEqual(result.pulse_kind, "refocusing")
        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.best_index, 1)
        self.assertEqual(len(result.results), 3)
        np.testing.assert_allclose(result.initial_phases, starts)
        np.testing.assert_allclose(result.best_score, 0.7)
        np.testing.assert_allclose(result.best_result.best_phases, starts[1])
    def test_ideal_v0crit_refocusing_multistart_selects_best_result(self) -> None:
        original = driver_module.refocusing_module.optimize_ideal_v0crit_refocusing_phases

        def fake_optimizer(phases: np.ndarray, **kwargs: object) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                best_score=-float(np.sum((phase_arr - 0.4) ** 2)),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.refocusing_module.optimize_ideal_v0crit_refocusing_phases = (
            fake_optimizer
        )
        try:
            starts = np.array([[0.0], [0.4], [0.8]])
            result = run_ideal_v0crit_refocusing_multistart(
                1,
                initial_phases=starts,
                bounds=(-1.0, 1.0),
            )
        finally:
            driver_module.refocusing_module.optimize_ideal_v0crit_refocusing_phases = (
                original
            )

        self.assertEqual(result.pulse_kind, "refocusing")
        self.assertEqual(result.probe, "ideal_v0crit")
        self.assertEqual(result.best_index, 1)
        np.testing.assert_allclose(result.bounds, (-1.0, 1.0))
        np.testing.assert_allclose(result.best_result.best_phases, [0.4])
    def test_ideal_v0crit_excited_refocusing_multistart_selects_best_result(
        self,
    ) -> None:
        original = (
            driver_module.refocusing_module
            .optimize_ideal_v0crit_excited_refocusing_phases
        )

        def fake_optimizer(phases: np.ndarray, **kwargs: object) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                best_score=-float(np.sum((phase_arr - 0.4) ** 2)),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.refocusing_module.optimize_ideal_v0crit_excited_refocusing_phases = (
            fake_optimizer
        )
        try:
            starts = np.array([[0.0], [0.4], [0.8]])
            result = run_ideal_v0crit_excited_refocusing_multistart(
                1,
                initial_phases=starts,
                bounds=(-1.0, 1.0),
            )
        finally:
            driver_module.refocusing_module.optimize_ideal_v0crit_excited_refocusing_phases = (
                original
            )

        self.assertEqual(result.pulse_kind, "refocusing")
        self.assertEqual(result.probe, "ideal_v0crit_excited")
        self.assertEqual(result.best_index, 1)
        np.testing.assert_allclose(result.bounds, (-1.0, 1.0))
        np.testing.assert_allclose(result.best_result.best_phases, [0.4])
    def test_ideal_time_varying_refocusing_multistart_selects_best_result(self) -> None:
        original = (
            driver_module.refocusing_module.optimize_ideal_time_varying_refocusing_phases
        )

        def fake_optimizer(phases: np.ndarray, **kwargs: object) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                best_score=-float(np.sum((phase_arr - 0.25) ** 2)),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.refocusing_module.optimize_ideal_time_varying_refocusing_phases = (
            fake_optimizer
        )
        try:
            starts = np.array([[0.0], [0.25], [0.5]])
            result = run_ideal_time_varying_refocusing_multistart(
                1,
                initial_phases=starts,
                bounds=(-1.0, 1.0),
            )
        finally:
            driver_module.refocusing_module.optimize_ideal_time_varying_refocusing_phases = (
                original
            )

        self.assertEqual(result.pulse_kind, "refocusing")
        self.assertEqual(result.probe, "ideal_time_varying")
        self.assertEqual(result.best_index, 1)
        np.testing.assert_allclose(result.bounds, (-1.0, 1.0))
        np.testing.assert_allclose(result.best_result.best_phases, [0.25])
    def test_multistart_refocusing_export_uses_matlab_cell_shape(self) -> None:
        results = (
            SimpleNamespace(
                best_score=1.0,
                best_phases=np.array([0.1, 0.2]),
                initial_phases=np.array([0.0, 0.0]),
                best_evaluation=SimpleNamespace(
                    pulse_length_t180=0.4,
                    v0crit_average=10.0,
                ),
                bounds=(0.0, 2 * np.pi),
                history_scores=np.array([0.5, 1.0]),
                optimizer_method="pattern",
                optimizer_success=True,
                optimizer_message="ok",
            ),
            SimpleNamespace(
                best_score=2.0,
                best_phases=np.array([0.3, 0.4]),
                initial_phases=np.array([0.2, 0.2]),
                best_evaluation=SimpleNamespace(
                    pulse_length_t180=0.4,
                    v0crit_average=20.0,
                ),
                bounds=(0.0, 2 * np.pi),
                history_scores=np.array([1.5, 2.0]),
                optimizer_method="pattern",
                optimizer_success=True,
                optimizer_message="ok",
            ),
        )
        multistart = SimpleNamespace(
            pulse_kind="refocusing",
            probe="ideal_v0crit",
            initial_phases=np.array([[0.0, 0.0], [0.2, 0.2]]),
            results=results,
            best_index=1,
            best_result=results[1],
            best_score=2.0,
            bounds=(0.0, 2 * np.pi),
        )

        cells = multistart_to_matlab_results(multistart, free_precession_t180=1.5)
        summary = multistart_summary_arrays(multistart)

        self.assertEqual(cells.shape, (2, 1))
        self.assertEqual(cells[0, 0].shape, (1, 8))
        np.testing.assert_allclose(cells[0, 0][0, 0], [1.5, 0.2, 0.2, 1.5])
        np.testing.assert_allclose(cells[0, 0][0, 1], [0.0, 0.1, 0.2, 0.0])
        self.assertEqual(float(cells[1, 0][0, 4]), 20.0)
        self.assertEqual(cells[1, 0][0, 5]["best_index"], 2)
        np.testing.assert_allclose(summary["scores"], [1.0, 2.0])
        np.testing.assert_allclose(summary["best_phases"], [0.3, 0.4])

        cell_summary = summarize_matlab_results(cells, pulse_number=1)
        selected = select_matlab_result_program(cells, pulse_number=2)
        np.testing.assert_allclose(cell_summary.scores, [1.0, 2.0])
        np.testing.assert_allclose(cell_summary.secondary_scores, [10.0, 20.0])
        self.assertEqual(cell_summary.best_index, 1)
        self.assertEqual(cell_summary.selected_index, 0)
        self.assertEqual(selected.pulse_kind, "refocusing")
        self.assertEqual(selected.pulse_number, 2)
        self.assertEqual(selected.score, 2.0)
        self.assertEqual(selected.secondary_score, 20.0)
        self.assertIsNone(selected.excitation)
        self.assertIsNotNone(selected.refocusing)
        np.testing.assert_allclose(selected.refocusing.phases, [0.0, 0.3, 0.4, 0.0])
    def test_matlab_result_layout_catalog_maps_plot_scripts(self) -> None:
        layouts = matlab_result_layouts()
        tuned = get_matlab_result_layout("plot_opt_ref_results_tuned.m")
        v0crit = get_matlab_result_layout("ideal_v0crit")
        excitation = get_matlab_result_layout("plot_opt_exc_results_tuned")

        self.assertIn("tuned_refocusing", layouts)
        self.assertEqual(tuned.score_index, 3)
        self.assertEqual(tuned.params_index, 4)
        self.assertEqual(v0crit.secondary_index, 4)
        self.assertEqual(v0crit.params_index, 5)
        self.assertEqual(excitation.pulse_kind, "excitation")
        self.assertEqual(excitation.score_index, 6)
    def test_analyze_matlab_optimization_results_uses_script_layout(self) -> None:
        cell1 = np.empty((1, 8), dtype=object)
        cell1[0, 0] = np.array([0.0, 0.1, 0.1, 0.0])
        cell1[0, 1] = np.array([0.0, 0.2, 0.4, 0.0])
        cell1[0, 2] = np.array([0.0, 1.0, 1.0, 0.0])
        cell1[0, 3] = 1.5
        cell1[0, 4] = 9.0
        cell1[0, 5] = {"params": "first"}
        cell1[0, 6] = {"sp": "first"}
        cell1[0, 7] = {"pp": "first"}
        cell2 = np.empty((1, 8), dtype=object)
        cell2[0, 0] = np.array([0.0, 0.2, 0.2, 0.0])
        cell2[0, 1] = np.array([0.0, 0.3, 0.5, 0.0])
        cell2[0, 2] = np.array([0.0, 1.0, 1.0, 0.0])
        cell2[0, 3] = 2.5
        cell2[0, 4] = 12.0
        cell2[0, 5] = {"params": "selected"}
        cell2[0, 6] = {"sp": "selected"}
        cell2[0, 7] = {"pp": "selected"}
        cells = np.empty((2, 1), dtype=object)
        cells[0, 0] = cell1
        cells[1, 0] = cell2

        analysis = analyze_matlab_optimization_results(
            cells,
            layout="plot_opt_ref_results_ideal_v0crit.m",
            pulse_number=2,
        )

        self.assertEqual(analysis.layout.name, "ideal_v0crit_refocusing")
        self.assertEqual(analysis.score_label, "Optimized SNR")
        self.assertEqual(analysis.secondary_label, "Average v0crit")
        self.assertEqual(analysis.summary.best_index, 1)
        self.assertEqual(analysis.selected_program.pulse_number, 2)
        self.assertEqual(analysis.selected_program.score, 2.5)
        self.assertEqual(analysis.selected_program.secondary_score, 12.0)
        self.assertEqual(analysis.params["params"], "selected")
        self.assertEqual(analysis.sp["sp"], "selected")
        self.assertEqual(analysis.pp["pp"], "selected")
        np.testing.assert_allclose(
            analysis.selected_program.refocusing.phases,
            [0.0, 0.3, 0.5, 0.0],
        )
    def test_matlab_tuned_excitation_result_fixture_matches_csv(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")
        table = np.loadtxt(
            FIXTURES / "optimization_tuned_excitation_result.csv",
            delimiter=",",
        )

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_tuned_excitation_results.mat",
            layout="plot_opt_exc_results_tuned.m",
        )

        np.testing.assert_allclose(analysis.summary.scores, table[:, 1])
        self.assertEqual(analysis.summary.best_index, 1)
        self.assertEqual(analysis.selected_program.pulse_number, 2)
        self.assertEqual(analysis.selected_program.score, table[1, 1])
        self.assertIsNotNone(analysis.selected_program.excitation)
        np.testing.assert_allclose(
            analysis.selected_program.excitation.phases,
            table[1, 2:],
        )
    def test_matlab_tuned_inverse_result_pair_fixture_matches_csv(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")
        exc_table = np.loadtxt(
            FIXTURES / "optimization_tuned_excitation_result.csv",
            delimiter=",",
        )
        inv_table = np.loadtxt(
            FIXTURES / "optimization_tuned_inverse_result.csv",
            delimiter=",",
        )

        pair = analyze_tuned_inverse_result_files(
            FIXTURES / "optimization_tuned_excitation_results.mat",
        )

        np.testing.assert_allclose(pair.original.summary.scores, exc_table[:, 1])
        np.testing.assert_allclose(pair.inverse.summary.scores, inv_table[:, 1])
        self.assertEqual(pair.original.selected_program.pulse_number, 2)
        self.assertEqual(pair.inverse.selected_program.pulse_number, 2)
        self.assertEqual(pair.original_score, exc_table[1, 1])
        self.assertEqual(pair.inverse_score, inv_table[1, 1])
        self.assertLess(pair.inverse_score, inv_table[0, 1])
        self.assertIsNotNone(pair.inverse.selected_program.excitation)
        np.testing.assert_allclose(
            pair.inverse.selected_program.excitation.phases,
            inv_table[1, 4:],
        )
    def test_matlab_tuned_inverse_fixture_matches_residual_spectra(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")
        inv_table = np.loadtxt(
            FIXTURES / "optimization_tuned_inverse_result.csv",
            delimiter=",",
        )
        spectra = np.loadtxt(
            FIXTURES / "optimization_tuned_inverse_spectra.csv",
            delimiter=",",
        )

        del_w = spectra[:, 0]
        target_mrx = spectra[:, 1] + 1j * spectra[:, 2]
        initial_mrx = spectra[:, 3] + 1j * spectra[:, 4]
        optimized_mrx = spectra[:, 5] + 1j * spectra[:, 6]
        neff = calc_rot_axis_arba3(np.array([np.pi]), np.array([0.0]), np.ones(1), del_w)
        pair = analyze_tuned_inverse_result_files(
            FIXTURES / "optimization_tuned_excitation_results.mat",
        )

        target_eval = evaluate_tuned_excitation_pulse(
            pair.original.selected_program.excitation.phases,
            neff,
            numpts=del_w.size,
        )
        initial_eval = evaluate_tuned_inverse_excitation_pulse(
            inv_table[0, 4:],
            neff,
            target_eval.mrx,
            target_eval.snr,
            numpts=del_w.size,
        )
        optimized_eval = evaluate_tuned_inverse_excitation_pulse(
            pair.inverse.selected_program.excitation.phases,
            neff,
            target_eval.mrx,
            target_eval.snr,
            numpts=del_w.size,
        )

        np.testing.assert_allclose(target_eval.mrx, target_mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            initial_eval.excitation.mrx,
            initial_mrx,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            optimized_eval.excitation.mrx,
            optimized_mrx,
            rtol=1e-13,
            atol=1e-13,
        )

        target_norm = trapezoid(np.abs(target_eval.mrx), del_w)
        initial_ratio = (
            trapezoid(np.abs(target_eval.mrx + initial_eval.excitation.mrx), del_w)
            / target_norm
        )
        optimized_ratio = (
            trapezoid(np.abs(target_eval.mrx + optimized_eval.excitation.mrx), del_w)
            / target_norm
        )
        np.testing.assert_allclose(initial_eval.mismatch, inv_table[0, 1])
        np.testing.assert_allclose(optimized_eval.mismatch, inv_table[1, 1])
        np.testing.assert_allclose(initial_ratio, inv_table[0, 2])
        np.testing.assert_allclose(optimized_ratio, inv_table[1, 2])
        self.assertLess(optimized_eval.mismatch, initial_eval.mismatch)
        self.assertLess(optimized_ratio, initial_ratio)
    def test_matlab_ideal_v0crit_result_fixture_exposes_secondary_metric(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_ideal_v0crit_results.mat",
            layout="plot_opt_ref_results_ideal_v0crit.m",
        )

        self.assertEqual(analysis.layout.name, "ideal_v0crit_refocusing")
        self.assertEqual(analysis.summary.scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(analysis.summary.secondary_scores)))
        self.assertEqual(analysis.selected_program.pulse_kind, "refocusing")
        self.assertIsNotNone(analysis.selected_program.refocusing)
        self.assertIsNotNone(analysis.selected_program.secondary_score)
        np.testing.assert_allclose(
            analysis.selected_program.secondary_score,
            analysis.summary.secondary_scores[analysis.summary.selected_index],
        )
    def test_matlab_tuned_refocusing_result_fixture_matches_python_eval(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_tuned_refocusing_results.mat",
            layout="plot_opt_ref_results_tuned.m",
        )

        self.assertEqual(analysis.layout.name, "tuned_refocusing")
        self.assertEqual(analysis.summary.scores.shape, (2,))
        self.assertEqual(analysis.summary.best_index, 0)
        self.assertEqual(analysis.selected_program.pulse_number, 1)
        self.assertIsNone(analysis.selected_program.secondary_score)
        self.assertIsNotNone(analysis.selected_program.refocusing)
        np.testing.assert_allclose(
            analysis.selected_program.refocusing.phases,
            [0.1, 0.4],
        )
        evaluated = evaluate_tuned_refocusing_pulse(
            analysis.selected_program.refocusing.phases,
            numpts=9,
        )
        np.testing.assert_allclose(
            evaluated.snr,
            analysis.selected_program.score,
            rtol=1e-5,
            atol=1e-6,
        )
    def test_matlab_untuned_refocusing_result_fixture_matches_python_eval(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_untuned_refocusing_results.mat",
            layout="plot_opt_ref_results_untuned.m",
        )

        self.assertEqual(analysis.layout.name, "untuned_refocusing")
        self.assertEqual(analysis.summary.scores.shape, (2,))
        self.assertEqual(analysis.summary.best_index, 0)
        self.assertEqual(analysis.selected_program.pulse_number, 1)
        self.assertIsNone(analysis.selected_program.secondary_score)
        self.assertIsNotNone(analysis.selected_program.refocusing)
        np.testing.assert_allclose(
            analysis.selected_program.refocusing.phases,
            [0.1, 0.4],
        )
        evaluated = evaluate_untuned_refocusing_pulse(
            analysis.selected_program.refocusing.phases,
            numpts=9,
        )
        np.testing.assert_allclose(
            evaluated.snr,
            analysis.selected_program.score,
            rtol=1e-5,
            atol=1e-6,
        )
    def test_matlab_matched_refocusing_result_fixture_matches_python_eval(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_matched_refocusing_results.mat",
            layout="plot_opt_ref_results_matched.m",
        )

        self.assertEqual(analysis.layout.name, "matched_refocusing")
        self.assertEqual(analysis.summary.scores.shape, (2,))
        self.assertEqual(analysis.summary.best_index, 0)
        self.assertEqual(analysis.selected_program.pulse_number, 1)
        self.assertIsNone(analysis.selected_program.secondary_score)
        self.assertIsNotNone(analysis.selected_program.refocusing)
        np.testing.assert_allclose(
            analysis.selected_program.refocusing.phases,
            [0.1, 0.4],
        )
        evaluated = evaluate_matched_refocusing_pulse(
            analysis.selected_program.refocusing.phases,
            numpts=9,
        )
        np.testing.assert_allclose(
            evaluated.snr,
            analysis.selected_program.score,
            rtol=1e-5,
            atol=1e-6,
        )
    def test_matlab_matched_refocusing_csv_fixture_matches_python_eval(self) -> None:
        table = np.loadtxt(
            FIXTURES / "optimization_matched_refocusing_result.csv",
            delimiter=",",
        )
        selected = table[0]

        self.assertEqual(int(selected[0]), 0)
        np.testing.assert_allclose(selected[2:], [0.1, 0.4])
        evaluated = evaluate_matched_refocusing_pulse(selected[2:], numpts=9)
        np.testing.assert_allclose(
            evaluated.snr,
            selected[1],
            rtol=1e-5,
            atol=1e-6,
        )
    def test_matlab_ideal_time_varying_result_fixture_matches_python_eval(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")

        analysis = analyze_optimization_result_file(
            FIXTURES / "optimization_ideal_tv_results.mat",
            layout="plot_opt_ref_results_ideal_tv.m",
        )

        self.assertEqual(analysis.layout.name, "ideal_time_varying_refocusing")
        self.assertEqual(analysis.summary.scores.shape, (2,))
        self.assertEqual(analysis.summary.best_index, 0)
        self.assertEqual(analysis.selected_program.pulse_number, 1)
        self.assertIsNone(analysis.selected_program.secondary_score)
        self.assertIsNotNone(analysis.selected_program.refocusing)
        np.testing.assert_allclose(
            analysis.selected_program.refocusing.phases,
            [0.0, 0.1, 0.4, 0.0],
        )
        full_cycle_offsets = 1.5 * np.sin(2 * np.pi * np.linspace(0, 1, 16))
        evaluated = evaluate_ideal_time_varying_refocusing_pulse(
            analysis.selected_program.refocusing.phases[1:-1],
            field_offsets=full_cycle_offsets,
            numpts=9,
            maxoffs=0.1,
            num_echoes=16,
            num_workers=1,
        )
        np.testing.assert_allclose(
            evaluated.score,
            analysis.selected_program.score,
            rtol=1e-5,
            atol=1e-10,
        )
    def test_multistart_npz_export_round_trips_matlab_cells(self) -> None:
        result = SimpleNamespace(
            best_score=1.25,
            best_phases=np.array([0.6]),
            initial_phases=np.array([0.5]),
            best_evaluation=SimpleNamespace(pulse_length_t180=0.1),
            bounds=(0.0, 2 * np.pi),
            history_scores=np.array([1.0, 1.25]),
            optimizer_method="pattern",
            optimizer_success=True,
            optimizer_message="ok",
        )
        multistart = SimpleNamespace(
            pulse_kind="refocusing",
            probe="tuned",
            initial_phases=np.array([[0.5]]),
            results=(result,),
            best_index=0,
            best_result=result,
            best_score=1.25,
            bounds=(0.0, 2 * np.pi),
        )
        outdir = ROOT / ".tmp" / "tests"
        outfile = outdir / "multistart_export_test.npz"

        save_multistart_results_npz(multistart, outfile)
        loaded = load_multistart_results_npz(outfile)

        self.assertIn("results", loaded)
        self.assertEqual(loaded["results"].shape, (1, 1))
        np.testing.assert_allclose(loaded["scores"], [1.25])
        np.testing.assert_allclose(loaded["best_phases"], [0.6])
        self.assertEqual(str(loaded["pulse_kind"][0]), "refocusing")
        self.assertEqual(str(loaded["probe"][0]), "tuned")

        loaded_cells = load_optimization_results(outfile)
        self.assertEqual(loaded_cells.shape, (1, 1))
        np.testing.assert_allclose(loaded_cells[0, 0][0, 1], [0.0, 0.6, 0.0])
        with self.assertRaises(ValueError):
            load_optimization_results(outfile.with_suffix(".txt"))
    def test_matlab_result_loader_requires_scipy_for_mat_files(self) -> None:
        if importlib.util.find_spec("scipy") is not None:
            self.skipTest("SciPy is installed in this environment")
        with self.assertRaises(ImportError):
            load_matlab_results_mat(ROOT / ".tmp" / "missing_results.mat")
        with self.assertRaises(ImportError):
            save_multistart_results_mat(SimpleNamespace(), ROOT / ".tmp" / "out.mat")
    def test_matlab_result_mat_round_trip_when_scipy_is_available(self) -> None:
        if importlib.util.find_spec("scipy") is None:
            self.skipTest("SciPy is not installed in this environment")
        result = SimpleNamespace(
            best_score=2.5,
            best_phases=np.array([0.2, 0.4]),
            initial_phases=np.array([0.1, 0.3]),
            best_evaluation=SimpleNamespace(
                pulse_length_t180=0.2,
                v0crit_average=11.0,
            ),
            bounds=(0.0, 2 * np.pi),
            history_scores=np.array([1.0, 2.5]),
            optimizer_method="pattern",
            optimizer_success=True,
            optimizer_message="ok",
        )
        multistart = SimpleNamespace(
            pulse_kind="refocusing",
            probe="ideal_v0crit",
            initial_phases=np.array([[0.1, 0.3]]),
            results=(result,),
            best_index=0,
            best_result=result,
            best_score=2.5,
            bounds=(0.0, 2 * np.pi),
        )
        outfile = ROOT / ".tmp" / "tests" / "mat_round_trip.mat"

        save_multistart_results_mat(
            multistart,
            outfile,
            free_precession_t180=1.5,
        )
        cells = load_optimization_results(outfile)
        summary = summarize_matlab_results(cells)
        selected = select_matlab_result_program(cells)

        self.assertEqual(cells.shape, (1, 1))
        self.assertEqual(summary.best_score, 2.5)
        self.assertEqual(selected.secondary_score, 11.0)
        self.assertIsNotNone(selected.refocusing)
        np.testing.assert_allclose(selected.refocusing.times, [1.5, 0.1, 0.1, 1.5])
        np.testing.assert_allclose(selected.refocusing.phases, [0.0, 0.2, 0.4, 0.0])
    def test_multistart_excitation_export_uses_ten_cell_layout(self) -> None:
        result = SimpleNamespace(
            best_score=0.75,
            best_phases=np.array([0.1, 0.9]),
            initial_phases=np.array([0.0, 0.8]),
            best_evaluation=SimpleNamespace(pulse_length_t180=0.2),
            bounds=(0.0, 2 * np.pi),
            history_scores=np.array([0.5, 0.75]),
            optimizer_method="pattern",
            optimizer_success=True,
            optimizer_message="ok",
        )
        multistart = SimpleNamespace(
            pulse_kind="excitation",
            probe="tuned",
            initial_phases=np.array([[0.0, 0.8]]),
            results=(result,),
            best_index=0,
            best_result=result,
            best_score=0.75,
            bounds=(0.0, 2 * np.pi),
        )

        cells = multistart_to_matlab_results(multistart, free_precession_t180=1.0)

        self.assertEqual(cells.shape, (1, 1))
        self.assertEqual(cells[0, 0].shape, (1, 10))
        np.testing.assert_allclose(cells[0, 0][0, 0], [0.1, 0.1])
        np.testing.assert_allclose(cells[0, 0][0, 1], [0.1, 0.9])
        self.assertEqual(float(cells[0, 0][0, 6]), 0.75)

        summary = summarize_matlab_results(cells)
        selected = select_matlab_result_program(cells)
        self.assertEqual(summary.pulse_kind, "excitation")
        self.assertEqual(summary.best_index, 0)
        self.assertEqual(selected.pulse_kind, "excitation")
        self.assertEqual(selected.pulse_number, 1)
        self.assertEqual(selected.score, 0.75)
        self.assertIsNotNone(selected.excitation)
        self.assertIsNotNone(selected.refocusing)
        np.testing.assert_allclose(selected.excitation.phases, [0.1, 0.9])
        np.testing.assert_allclose(selected.refocusing.times, [1.0, 1.0])
    def test_analyze_tuned_inverse_result_pair_uses_matching_pulse_number(self) -> None:
        def excitation_cell(score: float, phases: np.ndarray, label: str) -> np.ndarray:
            cell = np.empty((1, 10), dtype=object)
            cell[0, 0] = np.array([0.1, 0.1])
            cell[0, 1] = phases
            cell[0, 2] = np.ones(2)
            cell[0, 3] = np.array([0.0, 0.2, 0.2, 0.0])
            cell[0, 4] = np.array([0.0, 0.3, 0.4, 0.0])
            cell[0, 5] = np.array([0.0, 1.0, 1.0, 0.0])
            cell[0, 6] = score
            cell[0, 7] = {"params": label}
            cell[0, 8] = {"sp": label}
            cell[0, 9] = {"pp": label}
            return cell

        original = np.empty((2, 1), dtype=object)
        inverse = np.empty((2, 1), dtype=object)
        original[0, 0] = excitation_cell(0.5, np.array([0.1, 0.2]), "orig-1")
        original[1, 0] = excitation_cell(1.2, np.array([0.3, 0.4]), "orig-2")
        inverse[0, 0] = excitation_cell(0.8, np.array([3.1, 3.2]), "inv-1")
        inverse[1, 0] = excitation_cell(0.9, np.array([3.3, 3.4]), "inv-2")

        pair = analyze_tuned_inverse_result_pair(
            original,
            inverse,
            pulse_number=2,
        )

        self.assertEqual(pair.original.layout.name, "tuned_excitation")
        self.assertEqual(pair.inverse.layout.name, "tuned_inverse_excitation")
        self.assertEqual(pair.original.selected_program.pulse_kind, "excitation")
        self.assertEqual(
            pair.inverse.selected_program.pulse_kind,
            "inverse_excitation",
        )
        self.assertEqual(pair.original_score, 1.2)
        self.assertEqual(pair.inverse_score, 0.9)
        self.assertAlmostEqual(pair.score_difference, -0.3)
        self.assertEqual(pair.original.params["params"], "orig-2")
        self.assertEqual(pair.inverse.params["params"], "inv-2")
        np.testing.assert_allclose(
            pair.inverse.selected_program.excitation.phases,
            [3.3, 3.4],
        )
    def test_tuned_excitation_inverse_pipeline_uses_refocusing_neff(self) -> None:
        original_exc = pipeline_module.drivers.run_tuned_excitation_multistart
        original_inv = pipeline_module.drivers.run_tuned_inverse_excitation_multistart
        del_w = np.linspace(-1.0, 1.0, 5)
        neff = np.zeros((3, 5), dtype=np.complex128)
        neff[0, :] = 1.0
        refocusing = SimpleNamespace(
            pulse_kind="refocusing",
            probe="ideal_v0crit",
            results=(
                SimpleNamespace(
                    best_score=1.0,
                    best_phases=np.array([0.1, 0.2]),
                    best_evaluation=SimpleNamespace(del_w=del_w, neff=neff),
                ),
            ),
            best_index=0,
        )
        refocusing.best_result = refocusing.results[0]

        def fake_excitation(
            num_segments: int,
            neff_arg: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            self.assertEqual(num_segments, 2)
            np.testing.assert_allclose(neff_arg, neff)
            self.assertEqual(kwargs["numpts"], 5)
            target = SimpleNamespace(
                del_w=del_w,
                mrx=np.ones(5, dtype=np.complex128),
                snr=1.0,
                phases=np.array([0.3, 0.4]),
            )
            result = SimpleNamespace(best_evaluation=target, best_score=2.0)
            return SimpleNamespace(
                pulse_kind="excitation",
                probe="tuned",
                results=(result,),
                best_index=0,
                best_result=result,
                best_score=2.0,
            )

        def fake_inverse(
            num_segments: int,
            neff_arg: np.ndarray,
            target_mrx: np.ndarray,
            target_snr: float,
            reference_phases: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            self.assertEqual(num_segments, 2)
            np.testing.assert_allclose(neff_arg, neff)
            np.testing.assert_allclose(target_mrx, np.ones(5))
            self.assertEqual(target_snr, 1.0)
            np.testing.assert_allclose(reference_phases, [0.3, 0.4])
            inverse_evals = (
                SimpleNamespace(excitation=SimpleNamespace(mrx=-0.2 * np.ones(5))),
                SimpleNamespace(excitation=SimpleNamespace(mrx=-1.0 * np.ones(5))),
            )
            results = tuple(
                SimpleNamespace(best_evaluation=evaluation, best_score=float(index))
                for index, evaluation in enumerate(inverse_evals)
            )
            return SimpleNamespace(
                pulse_kind="inverse_excitation",
                probe="tuned",
                results=results,
                best_index=1,
                best_result=results[1],
                best_score=1.0,
            )

        pipeline_module.drivers.run_tuned_excitation_multistart = fake_excitation
        pipeline_module.drivers.run_tuned_inverse_excitation_multistart = fake_inverse
        try:
            result = run_tuned_excitation_inverse_pipeline(
                refocusing,
                excitation_segments=2,
                excitation_starts=1,
                inverse_starts=2,
                seed=5,
                numpts=5,
            )
        finally:
            pipeline_module.drivers.run_tuned_excitation_multistart = original_exc
            pipeline_module.drivers.run_tuned_inverse_excitation_multistart = (
                original_inv
            )

        self.assertIsNone(result.selected_refocusing)
        np.testing.assert_allclose(result.del_w, del_w)
        np.testing.assert_allclose(result.neff, neff)
        np.testing.assert_allclose(result.inverse_residual_ratios, [0.8, 0.0])
        self.assertEqual(result.residual_best_index, 1)
        self.assertEqual(result.residual_best_ratio, 0.0)
    def test_tuned_excitation_inverse_pipeline_reconstructs_cell_refocusing_axis(
        self,
    ) -> None:
        original_exc = pipeline_module.drivers.run_tuned_excitation_multistart
        original_inv = pipeline_module.drivers.run_tuned_inverse_excitation_multistart
        cell = np.empty((1, 7), dtype=object)
        cell[0, 0] = np.array([0.2, 0.2])
        cell[0, 1] = np.array([0.0, np.pi / 2])
        cell[0, 2] = np.ones(2)
        cell[0, 3] = 1.0
        cell[0, 4] = {}
        cell[0, 5] = {}
        cell[0, 6] = {}
        cells = np.empty((1, 1), dtype=object)
        cells[0, 0] = cell

        def fake_excitation(
            _num_segments: int,
            neff_arg: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            self.assertEqual(neff_arg.shape, (3, 7))
            self.assertEqual(kwargs["numpts"], 7)
            target = SimpleNamespace(
                del_w=np.linspace(-2.0, 2.0, 7),
                mrx=np.ones(7, dtype=np.complex128),
                snr=1.0,
                phases=np.array([0.1]),
            )
            result = SimpleNamespace(best_evaluation=target, best_score=1.0)
            return SimpleNamespace(results=(result,), best_result=result, best_score=1.0)

        def fake_inverse(*_args: object, **_kwargs: object) -> SimpleNamespace:
            evaluation = SimpleNamespace(
                excitation=SimpleNamespace(mrx=-np.ones(7, dtype=np.complex128))
            )
            result = SimpleNamespace(best_evaluation=evaluation, best_score=1.0)
            return SimpleNamespace(
                results=(result,),
                best_index=0,
                best_result=result,
                best_score=1.0,
            )

        pipeline_module.drivers.run_tuned_excitation_multistart = fake_excitation
        pipeline_module.drivers.run_tuned_inverse_excitation_multistart = fake_inverse
        try:
            result = run_tuned_excitation_inverse_pipeline(
                cells,
                excitation_segments=1,
                numpts=7,
                maxoffs=2.0,
            )
        finally:
            pipeline_module.drivers.run_tuned_excitation_multistart = original_exc
            pipeline_module.drivers.run_tuned_inverse_excitation_multistart = (
                original_inv
            )

        self.assertIsNotNone(result.selected_refocusing)
        self.assertEqual(result.selected_refocusing.pulse_number, 1)
        np.testing.assert_allclose(result.del_w, np.linspace(-2.0, 2.0, 7))
        self.assertEqual(result.neff.shape, (3, 7))
        self.assertEqual(result.residual_best_ratio, 0.0)
    def test_tuned_excitation_multistart_forwards_neff_and_selects_best(self) -> None:
        original = driver_module.excitation_module.optimize_tuned_excitation_phases
        calls = []

        def fake_optimizer(
            phases: np.ndarray,
            neff: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            calls.append(np.asarray(neff, dtype=np.complex128).copy())
            return SimpleNamespace(
                best_score=-float(np.sum((phase_arr - 0.2) ** 2)),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.excitation_module.optimize_tuned_excitation_phases = fake_optimizer
        try:
            neff = np.zeros((3, 5), dtype=np.float64)
            neff[0, :] = 1.0
            starts = np.array([[0.0], [0.2], [0.5]])
            result = run_tuned_excitation_multistart(
                1,
                neff,
                initial_phases=starts,
                bounds=(-1.0, 1.0),
                numpts=5,
            )
        finally:
            driver_module.excitation_module.optimize_tuned_excitation_phases = original

        self.assertEqual(result.pulse_kind, "excitation")
        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.best_index, 1)
        np.testing.assert_allclose(result.bounds, (-1.0, 1.0))
        self.assertEqual(len(calls), 3)
        np.testing.assert_allclose(calls[0], neff)
        np.testing.assert_allclose(result.best_result.best_phases, [0.2])
    def test_tuned_excitation_multistart_defaults_to_matlab_phase_bounds(self) -> None:
        original = driver_module.excitation_module.optimize_tuned_excitation_phases

        def fake_optimizer(
            phases: np.ndarray,
            neff: np.ndarray,
            **kwargs: object,
        ) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            return SimpleNamespace(
                best_score=float(phase_arr[0]),
                best_phases=phase_arr,
                bounds=kwargs["bounds"],
            )

        driver_module.excitation_module.optimize_tuned_excitation_phases = fake_optimizer
        try:
            neff = np.zeros((3, 5), dtype=np.float64)
            neff[0, :] = 1.0
            result = run_tuned_excitation_multistart(
                1,
                neff,
                num_starts=2,
                seed=321,
                numpts=5,
            )
        finally:
            driver_module.excitation_module.optimize_tuned_excitation_phases = original

        np.testing.assert_allclose(result.bounds, (0.0, 2 * np.pi))
        self.assertTrue(np.all(result.initial_phases >= 0.0))
        self.assertTrue(np.all(result.initial_phases <= 2 * np.pi))
    def test_tuned_inverse_excitation_multistart_tracks_best_seed(self) -> None:
        original = driver_module.excitation_module.optimize_tuned_inverse_excitation_phases
        calls = []
        target = np.arange(5, dtype=np.float64).astype(np.complex128)

        def fake_optimizer(
            phases: np.ndarray,
            neff: np.ndarray,
            target_mrx: np.ndarray,
            target_snr: float,
            **kwargs: object,
        ) -> SimpleNamespace:
            phase_arr = np.asarray(phases, dtype=np.float64)
            calls.append(
                (
                    phase_arr.copy(),
                    np.asarray(neff, dtype=np.complex128).copy(),
                    np.asarray(target_mrx, dtype=np.complex128).copy(),
                    float(target_snr),
                    kwargs["bounds"],
                )
            )
            return SimpleNamespace(
                best_score=float(len(calls)),
                best_phases=phase_arr + 0.1,
                bounds=kwargs["bounds"],
            )

        driver_module.excitation_module.optimize_tuned_inverse_excitation_phases = (
            fake_optimizer
        )
        try:
            neff = np.zeros((3, 5), dtype=np.float64)
            neff[0, :] = 1.0
            result = run_tuned_inverse_excitation_multistart(
                1,
                neff,
                target,
                2.0,
                [0.25],
                num_starts=3,
                random_fraction=0.0,
                bounds=(-10.0, 10.0),
                numpts=5,
            )
        finally:
            driver_module.excitation_module.optimize_tuned_inverse_excitation_phases = (
                original
            )

        self.assertEqual(result.pulse_kind, "inverse_excitation")
        self.assertEqual(result.probe, "tuned")
        self.assertEqual(result.best_index, 2)
        self.assertEqual(len(calls), 3)
        expected_first = np.mod(np.pi + 0.25, 2 * np.pi)
        np.testing.assert_allclose(calls[0][0], [expected_first])
        np.testing.assert_allclose(calls[1][0], [expected_first + 0.1])
        np.testing.assert_allclose(calls[2][0], [expected_first + 0.2])
        np.testing.assert_allclose(calls[0][1], neff)
        np.testing.assert_allclose(calls[0][2], target)
        self.assertEqual(calls[0][3], 2.0)
        self.assertEqual(calls[0][4], (-10.0, 10.0))
        np.testing.assert_allclose(result.best_result.best_phases, [expected_first + 0.3])
        with self.assertRaises(ValueError):
            run_tuned_inverse_excitation_multistart(
                1,
                neff,
                target,
                2.0,
                [0.25],
                random_fraction=1.5,
            )
    def test_refocusing_phase_optimizer_rejects_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            optimize_tuned_refocusing_phases([], numpts=7)
        with self.assertRaises(ValueError):
            optimize_tuned_refocusing_phases([0.0], numpts=7, bounds=(1.0, 1.0))
        with self.assertRaises(ValueError):
            optimize_tuned_refocusing_phases([0.0], numpts=7, step_decay=1.0)
        with self.assertRaises(ValueError):
            optimize_tuned_refocusing_phases([0.0], numpts=7, optimizer="unknown")
