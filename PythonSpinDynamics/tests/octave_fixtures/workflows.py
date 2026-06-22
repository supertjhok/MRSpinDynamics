"""High-level workflow, diffusion, imaging, and train fixture checks."""

from tests.octave_fixtures.support import *


class OctaveWorkflowFixtureTests(OctaveFixtureBase):
    def test_cpmg_workflow_result_shapes(self) -> None:
        runners = [
            run_ideal_cpmg,
            run_tuned_cpmg,
            run_untuned_cpmg,
            run_matched_cpmg,
        ]
        for runner in runners:
            with self.subTest(runner=runner.__name__):
                result = runner(numpts=11, maxoffs=4)
                self.assertEqual(result.del_w.shape, (11,))
                self.assertEqual(result.masy.shape, (11,))
                self.assertEqual(result.mrx.shape, (11,))
                self.assertGreater(result.echo.size, 0)
                self.assertEqual(result.echo.shape, result.tvect.shape)
                if result.probe == "ideal":
                    self.assertIsNone(result.snr)
                else:
                    self.assertIsInstance(result.snr, float)
    def test_run_ideal_cpmg_train_matches_octave(self) -> None:
        mrx_table = np.loadtxt(FIXTURES / "run_ideal_cpmg_train_mrx.csv", delimiter=",")
        echo_table = np.loadtxt(FIXTURES / "run_ideal_cpmg_train_echo.csv", delimiter=",")
        int_table = np.loadtxt(
            FIXTURES / "run_ideal_cpmg_train_integrals.csv",
            delimiter=",",
        )

        num_echoes = int(np.max(mrx_table[:, 0]))
        numpts = int(np.max(mrx_table[:, 1]))
        mrx_ref = np.zeros((num_echoes, numpts), dtype=np.complex128)
        for row in mrx_table:
            mrx_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]

        nacq = int(np.max(echo_table[:, 1]))
        echo_ref = np.zeros((num_echoes, nacq), dtype=np.complex128)
        tvect_ref = np.zeros(nacq, dtype=np.float64)
        for row in echo_table:
            echo_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]
            tvect_ref[int(row[1]) - 1] = row[4]

        echo_int_ref = int_table[:, 1] + 1j * int_table[:, 2]

        result = run_ideal_cpmg_train(
            numpts=numpts,
            maxoffs=5,
            num_echoes=num_echoes,
            t1_seconds=1.7,
            t2_seconds=1.1,
            rephase_action="ignore",
        )

        np.testing.assert_allclose(result.mrx, mrx_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.echo, echo_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(result.tvect, tvect_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            result.echo_integrals,
            echo_int_ref,
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertEqual(result.probe, "ideal")
    def test_run_ideal_cpmg_train_can_auto_refine_grid(self) -> None:
        coarse = run_ideal_cpmg_train(
            numpts=5,
            maxoffs=5,
            num_echoes=1,
            rephase_action="ignore",
        )
        refined = run_ideal_cpmg_train(
            numpts=5,
            maxoffs=5,
            num_echoes=1,
            auto_refine_grid=True,
            rephase_action="raise",
        )
        self.assertEqual(coarse.del_w.size, 5)
        self.assertGreater(refined.del_w.size, coarse.del_w.size)
    def test_probe_parameter_sweeps_return_expected_shapes(self) -> None:
        cases = [
            run_tuned_q_sweep(q_values=[20, 50], numpts=17),
            run_tuned_mistuning_sweep(offsets=[-1, 0, 1], numpts=17),
            run_matched_q_sweep(q_values=[20, 50], numpts=16),
            run_matched_mistuning_sweep(offsets=[-1, 0, 1], numpts=16),
        ]
        for result in cases:
            self.assertEqual(result.mrx.shape, (result.values.size, result.del_w.size))
            self.assertEqual(result.echo.shape, (result.values.size, result.tvect.size))
            self.assertEqual(result.snr.shape, (result.values.size,))
            self.assertTrue(np.all(np.isfinite(result.snr)))

        z_result = run_matched_z_magnetization_q_sweep(q_values=[20, 50], numpts=9)
        self.assertEqual(z_result.mz.shape, (z_result.values.size, z_result.del_w.size))
        self.assertGreater(z_result.tvect.size, 0)
        self.assertTrue(np.all(np.isfinite(z_result.mz)))
    def test_tuned_q_sweep_parallel_matches_serial(self) -> None:
        serial = run_tuned_q_sweep(q_values=[20, 50, 80], numpts=17, num_workers=1)
        parallel = run_tuned_q_sweep(q_values=[20, 50, 80], numpts=17, num_workers=2)
        np.testing.assert_allclose(parallel.values, serial.values)
        np.testing.assert_allclose(parallel.mrx, serial.mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.snr, serial.snr, rtol=1e-13, atol=1e-13)
    def test_matched_z_magnetization_q_sweep_parallel_matches_serial(self) -> None:
        serial = run_matched_z_magnetization_q_sweep(
            q_values=[20, 50],
            numpts=9,
            num_workers=1,
        )
        parallel = run_matched_z_magnetization_q_sweep(
            q_values=[20, 50],
            numpts=9,
            num_workers=2,
        )
        np.testing.assert_allclose(parallel.values, serial.values)
        np.testing.assert_allclose(parallel.mz, serial.mz, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.tvect, serial.tvect, rtol=1e-13, atol=1e-13)
    def test_ideal_time_varying_cpmg_final_returns_expected_shapes(self) -> None:
        waveform = sinusoidal_field_waveform(4)
        result = run_ideal_time_varying_cpmg_final(
            0.5 * waveform,
            numpts=17,
            pulse_name="rect180",
            rephase_action="ignore",
        )
        self.assertEqual(result.field_offsets.shape, (4,))
        self.assertEqual(result.mrx.shape, (17,))
        self.assertEqual(result.echo.shape, result.tvect.shape)
        self.assertTrue(np.isfinite(result.echo_integral))
    def test_time_varying_cpmg_rephasing_checks_and_refines(self) -> None:
        waveform = np.zeros(2)
        with self.assertRaises(RuntimeError):
            run_ideal_time_varying_cpmg_final(
                waveform,
                numpts=5,
                maxoffs=1,
                rephase_action="raise",
            )
        refined = run_ideal_time_varying_cpmg_final(
            waveform,
            numpts=5,
            maxoffs=1,
            auto_refine_grid=True,
            rephase_action="raise",
        )
        self.assertGreater(refined.del_w.size, 5)

        with self.assertRaises(RuntimeError):
            run_tuned_time_varying_cpmg_final(
                waveform,
                numpts=5,
                maxoffs=1,
                rephase_action="raise",
            )
        refined_probe = run_tuned_time_varying_cpmg_final(
            waveform,
            numpts=5,
            maxoffs=1,
            auto_refine_grid=True,
            rephase_action="raise",
        )
        self.assertGreater(refined_probe.del_w.size, 5)
    def test_ideal_time_varying_amplitude_sweep_parallel_matches_serial(self) -> None:
        waveform = sinusoidal_field_waveform(4)
        serial = run_ideal_time_varying_amplitude_sweep(
            amplitudes=[0.0, 0.5],
            waveform=waveform,
            numpts=17,
            num_workers=1,
            rephase_action="ignore",
        )
        parallel = run_ideal_time_varying_amplitude_sweep(
            amplitudes=[0.0, 0.5],
            waveform=waveform,
            numpts=17,
            num_workers=2,
            rephase_action="ignore",
        )
        np.testing.assert_allclose(parallel.amplitudes, serial.amplitudes)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.matched_signal,
            serial.matched_signal,
            rtol=1e-13,
            atol=1e-13,
        )
    def test_time_varying_amplitude_sweep_forwards_rephasing_options(self) -> None:
        waveform = np.zeros(2)
        with self.assertRaises(RuntimeError):
            run_ideal_time_varying_amplitude_sweep(
                amplitudes=[0.0],
                waveform=waveform,
                numpts=5,
                maxoffs=1,
                rephase_action="raise",
            )
        refined = run_ideal_time_varying_amplitude_sweep(
            amplitudes=[0.0],
            waveform=waveform,
            numpts=5,
            maxoffs=1,
            auto_refine_grid=True,
            rephase_action="raise",
        )
        self.assertGreater(refined.del_w.size, 5)
    def test_probe_time_varying_cpmg_final_returns_expected_shapes(self) -> None:
        waveform = sinusoidal_field_waveform(2)
        for runner, probe in [
            (run_tuned_time_varying_cpmg_final, "tuned"),
            (run_untuned_time_varying_cpmg_final, "untuned"),
            (run_matched_time_varying_cpmg_final, "matched"),
        ]:
            with self.subTest(probe=probe):
                result = runner(
                    0.25 * waveform,
                    numpts=9,
                    maxoffs=4,
                    num_workers=1,
                    rephase_action="ignore",
                )
                self.assertEqual(result.probe, probe)
                self.assertEqual(result.field_offsets.shape, (2,))
                self.assertEqual(result.mrx.shape, (9,))
                self.assertEqual(result.echo.shape, result.tvect.shape)
                self.assertTrue(np.isfinite(result.echo_integral))
    def test_probe_time_varying_amplitude_sweep_parallel_matches_serial(self) -> None:
        waveform = sinusoidal_field_waveform(2)
        for runner, probe in [
            (run_tuned_time_varying_amplitude_sweep, "tuned"),
            (run_untuned_time_varying_amplitude_sweep, "untuned"),
            (run_matched_time_varying_amplitude_sweep, "matched"),
        ]:
            with self.subTest(probe=probe):
                serial = runner(
                    amplitudes=[0.0, 0.25],
                    waveform=waveform,
                    numpts=9,
                    maxoffs=4,
                    num_workers=1,
                    rephase_action="ignore",
                )
                parallel = runner(
                    amplitudes=[0.0, 0.25],
                    waveform=waveform,
                    numpts=9,
                    maxoffs=4,
                    num_workers=2,
                    rephase_action="ignore",
                )
                np.testing.assert_allclose(parallel.amplitudes, serial.amplitudes)
                np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
                np.testing.assert_allclose(
                    parallel.matched_signal,
                    serial.matched_signal,
                    rtol=1e-13,
                    atol=1e-13,
                )
    def test_matched_cpmg_ir_train_returns_expected_shapes(self) -> None:
        result = run_matched_cpmg_ir_train(
            num_echoes=2,
            tauvect=[0.5e-3, 1.0e-3],
            numpts=9,
            rephase_action="ignore",
        )
        self.assertEqual(result.mrx.shape, (2, 2, 9))
        self.assertEqual(result.echo.shape[:2], (2, 2))
        self.assertEqual(result.echo.shape[2], result.tvect.size)
        self.assertEqual(result.echo_integrals.shape, (2, 2))
        self.assertEqual(result.sequence_time.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.echo_integrals)))
    def test_nonmatched_cpmg_ir_train_returns_expected_shapes(self) -> None:
        for runner, probe in [
            (run_ideal_cpmg_ir_train, "ideal"),
            (run_tuned_cpmg_ir_train, "tuned"),
            (run_untuned_cpmg_ir_train, "untuned"),
        ]:
            with self.subTest(probe=probe):
                result = runner(
                    num_echoes=2,
                    tauvect=[0.5e-3, 1.0e-3],
                    numpts=9,
                    maxoffs=4,
                    rephase_action="ignore",
                )
                self.assertEqual(result.probe, probe)
                self.assertEqual(result.mrx.shape, (2, 2, result.del_w.size))
                self.assertEqual(result.echo.shape[:2], (2, 2))
                self.assertEqual(result.echo.shape[2], result.tvect.size)
                self.assertEqual(result.echo_integrals.shape, (2, 2))
                self.assertEqual(result.sequence_time.shape, (2,))
                self.assertTrue(np.all(np.isfinite(result.echo_integrals)))
    def test_ideal_cpmg_ir_train_tau_parallel_matches_serial(self) -> None:
        serial = run_ideal_cpmg_ir_train(
            num_echoes=2,
            tauvect=[0.5e-3, 1.0e-3],
            numpts=9,
            tau_workers=1,
            rephase_action="ignore",
        )
        parallel = run_ideal_cpmg_ir_train(
            num_echoes=2,
            tauvect=[0.5e-3, 1.0e-3],
            numpts=9,
            tau_workers=2,
            rephase_action="ignore",
        )
        np.testing.assert_allclose(parallel.tauvect, serial.tauvect)
        np.testing.assert_allclose(parallel.mrx, serial.mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.echo_integrals,
            serial.echo_integrals,
            rtol=1e-13,
            atol=1e-13,
        )
    def test_matched_cpmg_ir_train_tau_parallel_matches_serial(self) -> None:
        serial = run_matched_cpmg_ir_train(
            num_echoes=2,
            tauvect=[0.5e-3, 1.0e-3],
            numpts=9,
            num_workers=1,
            tau_workers=1,
            rephase_action="ignore",
        )
        parallel = run_matched_cpmg_ir_train(
            num_echoes=2,
            tauvect=[0.5e-3, 1.0e-3],
            numpts=9,
            num_workers=1,
            tau_workers=2,
            rephase_action="ignore",
        )
        np.testing.assert_allclose(parallel.tauvect, serial.tauvect)
        np.testing.assert_allclose(parallel.mrx, serial.mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.echo_integrals,
            serial.echo_integrals,
            rtol=1e-13,
            atol=1e-13,
        )
    def test_finite_probe_parameter_sweeps_return_expected_shapes(self) -> None:
        cases = [
            run_tuned_finite_q_sweep([20, 50], numpts=9, num_echoes=2, rephase_action="ignore"),
            run_untuned_finite_q_sweep([20, 50], numpts=9, num_echoes=2, rephase_action="ignore"),
            run_matched_finite_q_sweep([20, 50], numpts=9, num_echoes=2, rephase_action="ignore"),
            run_tuned_finite_mistuning_sweep(
                [-1, 1],
                numpts=9,
                num_echoes=2,
                rephase_action="ignore",
            ),
            run_untuned_finite_mistuning_sweep(
                [-1, 1],
                numpts=9,
                num_echoes=2,
                rephase_action="ignore",
            ),
            run_matched_finite_mistuning_sweep(
                [-1, 1],
                numpts=9,
                num_echoes=2,
                rephase_action="ignore",
            ),
        ]
        for result in cases:
            with self.subTest(probe=result.probe, sweep=result.sweep):
                self.assertEqual(result.mrx.shape[:2], (result.values.size, 2))
                self.assertEqual(result.echo.shape[:2], (result.values.size, 2))
                self.assertEqual(result.echo.shape[2], result.tvect.size)
                self.assertEqual(result.echo_integrals.shape, (result.values.size, 2))
                self.assertGreaterEqual(result.del_w.size, 9)
                self.assertTrue(np.all(np.isfinite(result.echo_integrals)))
    def test_finite_probe_parameter_sweep_parallel_matches_serial(self) -> None:
        serial = run_matched_finite_q_sweep(
            [20, 50],
            numpts=9,
            num_echoes=2,
            num_workers=1,
            sweep_workers=1,
            rephase_action="ignore",
        )
        parallel = run_matched_finite_q_sweep(
            [20, 50],
            numpts=9,
            num_echoes=2,
            num_workers=1,
            sweep_workers=2,
            rephase_action="ignore",
        )
        np.testing.assert_allclose(parallel.values, serial.values)
        np.testing.assert_allclose(parallel.mrx, serial.mrx, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.echo_integrals,
            serial.echo_integrals,
            rtol=1e-13,
            atol=1e-13,
        )
    def test_arb10_diffusion_zero_diffusion_matches_arb10(self) -> None:
        sp, _pp = set_params_ideal(numpts=9)
        del_w = np.linspace(-2, 2, 9)
        rtot = [
            calc_rotation_matrix(
                del_w,
                np.ones_like(del_w),
                np.array([np.pi / 2]),
                np.array([np.pi / 2]),
                np.array([1.0]),
            )
        ]
        params = {
            "tp": np.array([np.pi / 2, 1.0]),
            "pul": np.array([1, 0]),
            "amp": np.array([1.0, 0.0]),
            "acq": np.array([0, 1]),
            "grad": np.array([0.0, 0.0]),
            "Rtot": rtot,
            "del_w": del_w,
            "del_wg": np.zeros_like(del_w),
            "w_1": np.ones_like(del_w),
            "T1n": 1000 * np.ones_like(del_w),
            "T2n": 1000 * np.ones_like(del_w),
            "m0": sp.m0 * np.ones_like(del_w),
            "mth": sp.mth * np.ones_like(del_w),
        }
        diff_params = {
            **params,
            "gamma": 1.0,
            "gradient": 1.0,
            "diffusion_coefficient": 0.0,
            "time_scale": 1.0,
        }
        np.testing.assert_allclose(
            sim_spin_dynamics_arb10_diffusion(diff_params),
            sim_spin_dynamics_arb10(params),
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            sim_spin_dynamics_arb10_diffusion_chunked(diff_params, num_workers=2),
            sim_spin_dynamics_arb10_diffusion(diff_params),
            rtol=1e-13,
            atol=1e-13,
        )
    def test_matched_diffusion_cpmg_returns_expected_shapes(self) -> None:
        result = run_matched_diffusion_cpmg(
            num_echoes=2,
            numpts=17,
            q_value=20,
            rephase_action="ignore",
        )
        self.assertEqual(result.mrx.shape, (2, 17))
        self.assertEqual(result.echo.shape[0], 2)
        self.assertEqual(result.echo.shape[1], result.tvect.size)
        self.assertEqual(result.echo_integrals.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.echo_integrals)))
    def test_matched_diffusion_cpmg_rephasing_checks_and_refines(self) -> None:
        with self.assertRaises(RuntimeError):
            run_matched_diffusion_cpmg(
                num_echoes=1,
                numpts=5,
                dz=1e-4,
                q_value=20,
                rephase_action="raise",
                q_stability_action="ignore",
            )
        refined = run_matched_diffusion_cpmg(
            num_echoes=1,
            numpts=5,
            dz=1e-4,
            q_value=20,
            auto_refine_grid=True,
            rephase_action="raise",
            q_stability_action="ignore",
        )
        self.assertGreater(refined.del_w.size, 5)
    def test_matched_diffusion_q_stability_boundary(self) -> None:
        self.assertTrue(check_matched_diffusion_q_stability(VALIDATED_MATCHED_DIFFUSION_Q_MAX))
        with self.assertWarns(RuntimeWarning):
            self.assertFalse(check_matched_diffusion_q_stability(2500))
        with self.assertRaises(RuntimeError):
            check_matched_diffusion_q_stability(2500, action="raise")
    def test_matched_diffusion_q_sweep_parallel_matches_serial(self) -> None:
        serial = run_matched_diffusion_q_sweep(
            [20, 50],
            num_echoes=2,
            numpts=17,
            sweep_workers=1,
            rephase_action="ignore",
        )
        parallel = run_matched_diffusion_q_sweep(
            [20, 50],
            num_echoes=2,
            numpts=17,
            sweep_workers=2,
            rephase_action="ignore",
        )
        np.testing.assert_allclose(parallel.values, serial.values)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.echo_integrals,
            serial.echo_integrals,
            rtol=1e-13,
            atol=1e-13,
        )
    def test_matched_diffusion_q_sweep_forwards_rephasing_options(self) -> None:
        with self.assertRaises(RuntimeError):
            run_matched_diffusion_q_sweep(
                [20],
                num_echoes=1,
                numpts=5,
                rephase_action="raise",
                q_stability_action="ignore",
            )
        refined = run_matched_diffusion_q_sweep(
            [20],
            num_echoes=1,
            numpts=5,
            auto_refine_grid=True,
            rephase_action="raise",
            q_stability_action="ignore",
        )
        self.assertGreater(refined.del_w.size, 5)
    def test_ideal_cpmg_imaging_returns_expected_shapes(self) -> None:
        rho = np.eye(3)
        result = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=5,
            num_workers=1,
            phase_workers=1,
        )
        self.assertEqual(result.kspace.shape, (3, 3, 1))
        self.assertEqual(result.image.shape, (3, 3, 1))
        self.assertEqual(result.magnitude.shape, (3, 3, 1))
        self.assertEqual(result.echo_integrals.shape, (3, 3, 1))
        self.assertTrue(np.all(np.isfinite(result.kspace)))
        self.assertTrue(np.all(np.isfinite(result.magnitude)))
    def test_ideal_cpmg_imaging_white_noise_is_added_in_kspace(self) -> None:
        rho = np.eye(2)
        clean = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )
        noisy1 = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
            noise=NoiseSpec(sigma=1e-3, seed=9),
        )
        noisy2 = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
            noise=NoiseSpec(sigma=1e-3, seed=9),
        )

        np.testing.assert_allclose(noisy1.kspace, clean.kspace)
        np.testing.assert_allclose(noisy1.image, clean.image)
        self.assertIsNotNone(noisy1.kspace_noisy)
        self.assertIsNotNone(noisy1.image_noisy)
        self.assertIsNotNone(noisy1.magnitude_noisy)
        np.testing.assert_allclose(noisy1.kspace_noisy, noisy2.kspace_noisy)
        np.testing.assert_allclose(noisy1.image_noisy, noisy2.image_noisy)
        np.testing.assert_allclose(
            noisy1.image_noisy[:, :, 0],
            reconstruct_image_from_kspace(noisy1.kspace_noisy, echo_index=0),
        )
        self.assertGreater(
            float(np.max(np.abs(noisy1.kspace_noisy - noisy1.kspace))),
            0.0,
        )
        with self.assertRaises(ValueError):
            run_ideal_cpmg_imaging(
                rho,
                num_echoes=1,
                ny=3,
                noise=NoiseSpec(model="probe", seed=1),
            )
        with self.assertRaises(ValueError):
            run_ideal_cpmg_imaging(
                rho,
                num_echoes=1,
                ny=3,
                noise=NoiseSpec(sigma=1e-3, seed=1, domain="time"),
            )
    def test_imaging_noise_statistics_summarize_repeated_trials(self) -> None:
        rho = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float64)
        trials = [
            run_ideal_cpmg_imaging(
                rho,
                num_echoes=1,
                ny=3,
                num_workers=1,
                phase_workers=1,
                noise=NoiseSpec(sigma=1e-3, seed=seed),
            )
            for seed in range(8)
        ]
        signal_mask = rho > 0
        background_mask = rho == 0

        stats = summarize_imaging_noise_trials(
            trials,
            mode="single",
            signal_mask=signal_mask,
            background_mask=background_mask,
        )

        self.assertEqual(stats.num_trials, 8)
        self.assertEqual(stats.clean_image.shape, rho.shape)
        self.assertEqual(stats.noisy_mean.shape, rho.shape)
        self.assertEqual(stats.noise_std.shape, rho.shape)
        self.assertGreater(stats.background_noise_rms, 0.0)
        self.assertGreater(stats.signal_mean, 0.0)
        self.assertGreater(stats.snr, 0.0)
        with self.assertRaises(ValueError):
            summarize_imaging_noise_trials([])
        with self.assertRaises(ValueError):
            summarize_imaging_noise_trials([run_ideal_cpmg_imaging(rho, num_echoes=1, ny=3)])
    def test_imaging_echo_formation_modes_use_expected_weighting(self) -> None:
        echo_times = np.array([0.1e-3, 0.2e-3, 0.3e-3], dtype=np.float64)
        rho = np.array([[2.0, 4.0], [1.5, 3.0]], dtype=np.float64)
        t2 = np.array([[0.8e-3, 1.2e-3], [1.6e-3, 2.4e-3]], dtype=np.float64)
        magnitude = rho[:, :, np.newaxis] * np.exp(
            -echo_times.reshape(1, 1, -1) / t2[:, :, np.newaxis]
        )
        magnitude_noisy = 1.5 * magnitude
        result = SimpleNamespace(
            image=magnitude.astype(np.complex128),
            magnitude=magnitude,
            magnitude_noisy=magnitude_noisy,
            sequence_time=echo_times,
        )

        np.testing.assert_allclose(
            form_imaging_image(result, mode="single", echo_index=1),
            magnitude[:, :, 1],
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="echo_sum"),
            np.sum(magnitude, axis=2),
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="fit_rho"),
            rho,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="fit_t2"),
            t2,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="single", echo_index=1, use_noisy=True),
            magnitude_noisy[:, :, 1],
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="echo_sum", use_noisy=True),
            np.sum(magnitude_noisy, axis=2),
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="fit_rho", use_noisy=True),
            1.5 * rho,
            rtol=1e-13,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            form_imaging_image(result, mode="fit_t2", use_noisy=True),
            t2,
            rtol=1e-13,
            atol=1e-13,
        )
        with self.assertRaises(ValueError):
            form_imaging_image(
                SimpleNamespace(magnitude=magnitude, sequence_time=echo_times),
                mode="single",
                use_noisy=True,
            )
    def test_imaging_echo_decay_fit_reports_maps_and_rejects_bad_inputs(self) -> None:
        echo_times = np.array([0.1e-3, 0.2e-3, 0.3e-3], dtype=np.float64)
        rho = np.array([[2.0]], dtype=np.float64)
        t2 = np.array([[0.8e-3]], dtype=np.float64)
        magnitude = rho[:, :, np.newaxis] * np.exp(
            -echo_times.reshape(1, 1, -1) / t2[:, :, np.newaxis]
        )
        magnitude_noisy = 2.0 * magnitude
        result = SimpleNamespace(
            image=magnitude.astype(np.complex128),
            magnitude=magnitude,
            magnitude_noisy=magnitude_noisy,
            sequence_time=echo_times,
        )

        fit = fit_imaging_echo_decay(result)

        np.testing.assert_allclose(fit.rho_map, rho, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.t2_map, t2, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.fitted_magnitude, magnitude, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.residual_norm, 0.0, atol=1e-13)
        self.assertTrue(bool(fit.mask[0, 0]))
        noisy_fit = fit_imaging_echo_decay(result, use_noisy=True)
        np.testing.assert_allclose(noisy_fit.rho_map, 2.0 * rho, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(noisy_fit.t2_map, t2, rtol=1e-13, atol=1e-13)
        with self.assertRaises(ValueError):
            fit_imaging_echo_decay(
                SimpleNamespace(
                    image=magnitude[:, :, :1].astype(np.complex128),
                    magnitude=magnitude[:, :, :1],
                    sequence_time=echo_times[:1],
                )
            )
        with self.assertRaises(ValueError):
            form_imaging_image(result, mode="unknown")
        with self.assertRaises(ValueError):
            fit_imaging_echo_decay(
                SimpleNamespace(magnitude=magnitude, sequence_time=echo_times),
                use_noisy=True,
            )
    def test_phase_encoded_imaging_names_match_legacy_aliases(self) -> None:
        rho = np.eye(2, dtype=np.float64)
        sentinel = object()

        self.assertIs(
            run_ideal_phase_encoded_cpmg_imaging,
            imaging_module.run_ideal_phase_encoded_cpmg_imaging,
        )
        with patch.object(
            imaging_module,
            "run_ideal_phase_encoded_cpmg_imaging",
            return_value=sentinel,
        ) as canonical:
            self.assertIs(run_ideal_cpmg_imaging(rho, num_echoes=1), sentinel)
            canonical.assert_called_once()

        self.assertIs(
            run_tuned_phase_encoded_cpmg_imaging,
            imaging_module.run_tuned_phase_encoded_cpmg_imaging,
        )
        with patch.object(
            imaging_module,
            "run_tuned_phase_encoded_cpmg_imaging",
            return_value=sentinel,
        ) as canonical:
            self.assertIs(run_tuned_cpmg_imaging(rho, num_echoes=1), sentinel)
            canonical.assert_called_once()

        self.assertIs(
            run_matched_phase_encoded_cpmg_imaging,
            imaging_module.run_matched_phase_encoded_cpmg_imaging,
        )
        with patch.object(
            imaging_module,
            "run_matched_phase_encoded_cpmg_imaging",
            return_value=sentinel,
        ) as canonical:
            self.assertIs(run_matched_cpmg_imaging(rho, num_echoes=1), sentinel)
            canonical.assert_called_once()
    def test_make_imaging_field_maps_accepts_custom_b0_b1_maps(self) -> None:
        rho = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=np.float64)
        b0_map = np.array([[0.1, -0.2], [0.0, 0.3]], dtype=np.float64)
        b1_tx_map = np.array([[1.0, 0.8], [0.6, 0.4]], dtype=np.float64)
        b1_rx_map = np.array([[0.9, 0.7], [0.5, 0.3]], dtype=np.float64)

        maps = make_imaging_field_maps(
            rho,
            b0_map=b0_map,
            b1_tx_map=b1_tx_map,
            b1_rx_map=b1_rx_map,
        )
        kernel = maps.kernel_maps(ny=3, maxoffs=2.0)

        np.testing.assert_allclose(maps.b0_map, b0_map)
        np.testing.assert_allclose(maps.b1_tx_map, b1_tx_map)
        np.testing.assert_allclose(maps.b1_rx_map, b1_rx_map)
        self.assertEqual(kernel["del_w"].shape, (12,))
        np.testing.assert_allclose(
            kernel["del_w"],
            np.concatenate([offset + b0_map.reshape(-1) for offset in [-2.0, 0.0, 2.0]]),
        )
        np.testing.assert_allclose(kernel["w_1"][:4], b1_tx_map.reshape(-1))
        np.testing.assert_allclose(kernel["w_1r"][:4], b1_rx_map.reshape(-1))
        preserve = maps.kernel_maps(
            ny=3,
            maxoffs=2.0,
            density_normalization="preserve",
        )
        self.assertAlmostEqual(float(np.sum(preserve["m0"])), float(np.sum(rho)))
        self.assertAlmostEqual(float(np.sum(preserve["mth"])), float(np.sum(rho)))
        self.assertAlmostEqual(float(np.sum(kernel["m0"])), 3.0 * float(np.sum(rho)))
        with self.assertRaises(ValueError):
            maps.kernel_maps(ny=3, maxoffs=2.0, density_normalization="bad")
    def test_imaging_vector_b1_maps_are_projected_transverse_to_b0(self) -> None:
        rho = np.ones((2, 2), dtype=np.float64)
        b0_vector = np.array(
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            ],
            dtype=np.float64,
        )
        b1_tx_vector = np.array(
            [
                [[3.0, 4.0, 5.0], [2.0, 6.0, 0.0]],
                [[1.0, 7.0, 2.0], [3.0, 1.0, 4.0]],
            ],
            dtype=np.float64,
        )
        b1_rx_vector = 2.0 * b1_tx_vector

        maps = make_imaging_field_maps(
            rho,
            b0_vector_map=b0_vector,
            b1_tx_vector_map=b1_tx_vector,
            b1_rx_vector_map=b1_rx_vector,
        )

        expected = np.array(
            [[5.0, 6.0], [np.sqrt(5.0), np.sqrt(18.0)]],
            dtype=np.float64,
        )
        np.testing.assert_allclose(maps.b1_tx_map, expected)
        np.testing.assert_allclose(maps.b1_rx_map, 2.0 * expected)
    def test_make_imaging_field_maps_rejects_invalid_maps(self) -> None:
        rho = np.ones((2, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            make_imaging_field_maps(rho, b1_tx_map=-np.ones_like(rho))
        with self.assertRaises(ValueError):
            make_imaging_field_maps(rho, b0_map=np.ones((2, 3)))
        with self.assertRaises(ValueError):
            make_imaging_field_maps(
                rho,
                b1_tx_map=np.ones_like(rho),
                b1_tx_vector_map=np.ones((2, 2, 3), dtype=np.float64),
            )
        with self.assertRaises(ValueError):
            make_imaging_field_maps(
                rho,
                b1_tx_vector_map=np.ones((2, 2, 3), dtype=np.float64),
            )
    def test_load_imaging_field_maps_npz_round_trips_custom_maps(self) -> None:
        rho = np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float64)
        b0_map = np.array([[0.0, 0.2], [-0.1, 0.4]], dtype=np.float64)
        b1_tx_map = np.array([[1.0, 0.9], [0.7, 0.5]], dtype=np.float64)
        b1_rx_map = np.array([[0.8, 0.6], [0.4, 0.2]], dtype=np.float64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "field_maps.npz"
            np.savez(
                path,
                rho=rho,
                b0_map=b0_map,
                b1_tx_map=b1_tx_map,
                b1_rx_map=b1_rx_map,
            )
            maps = load_imaging_field_maps_npz(path)

        np.testing.assert_allclose(maps.rho, rho)
        np.testing.assert_allclose(maps.b0_map, b0_map)
        np.testing.assert_allclose(maps.b1_tx_map, b1_tx_map)
        np.testing.assert_allclose(maps.b1_rx_map, b1_rx_map)
    def test_load_imaging_field_maps_npz_projects_vector_b1_maps(self) -> None:
        rho = np.ones((1, 2), dtype=np.float64)
        b0_vector = np.array(
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
            dtype=np.float64,
        )
        b1_tx_vector = np.array(
            [[[3.0, 4.0, 5.0], [2.0, 6.0, 0.0]]],
            dtype=np.float64,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "field_maps.npz"
            np.savez(
                path,
                rho=rho,
                b0_vector_map=b0_vector,
                b1_tx_vector_map=b1_tx_vector,
            )
            maps = load_imaging_field_maps_npz(path)

        np.testing.assert_allclose(maps.b1_tx_map, [[5.0, 6.0]])
        np.testing.assert_allclose(maps.b1_rx_map, [[5.0, 6.0]])
    def test_ideal_cpmg_imaging_accepts_custom_field_maps(self) -> None:
        rho = np.array([[1.0, 0.0], [0.5, 0.25]], dtype=np.float64)
        b0_map = np.array([[0.0, 0.1], [-0.1, 0.2]], dtype=np.float64)
        b1_tx_map = np.array([[1.0, 0.8], [0.6, 0.4]], dtype=np.float64)
        maps = make_imaging_field_maps(rho, b0_map=b0_map, b1_tx_map=b1_tx_map)

        result = run_ideal_cpmg_imaging(
            maps,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )

        self.assertEqual(result.kspace.shape, (2, 2, 1))
        np.testing.assert_allclose(result.rho, rho)
        np.testing.assert_allclose(result.b0_map, b0_map)
        np.testing.assert_allclose(result.b1_tx_map, b1_tx_map)
        np.testing.assert_allclose(result.b1_rx_map, b1_tx_map)
        self.assertTrue(np.all(np.isfinite(result.kspace)))
    def test_ideal_cpmg_imaging_can_preserve_density_across_aux_offsets(self) -> None:
        rho = np.array([[1.0, 0.5], [0.25, 0.75]], dtype=np.float64)
        legacy = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )
        preserve = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
            density_normalization="preserve",
        )

        np.testing.assert_allclose(preserve.kspace, legacy.kspace / 3.0)
    def test_t1_encoded_cpmg_imaging_adds_synthetic_t1_contrast(self) -> None:
        short_t1 = make_imaging_field_maps(
            np.ones((1, 1), dtype=np.float64),
            t1_map=0.25e-3 * np.ones((1, 1), dtype=np.float64),
            t2_map=50e-3 * np.ones((1, 1), dtype=np.float64),
            b1_tx_map=np.ones((1, 1), dtype=np.float64),
            b1_rx_map=np.ones((1, 1), dtype=np.float64),
        )
        near_null_t1 = make_imaging_field_maps(
            np.ones((1, 1), dtype=np.float64),
            t1_map=0.72e-3 * np.ones((1, 1), dtype=np.float64),
            t2_map=50e-3 * np.ones((1, 1), dtype=np.float64),
            b1_tx_map=np.ones((1, 1), dtype=np.float64),
            b1_rx_map=np.ones((1, 1), dtype=np.float64),
        )
        short_result = run_t1_encoded_phase_encoded_cpmg_imaging(
            short_t1,
            inversion_time_seconds=0.5e-3,
            num_echoes=3,
            ny=1,
            maxoffs=0.0,
            num_workers=1,
            phase_workers=1,
        )
        near_null_result = run_t1_encoded_phase_encoded_cpmg_imaging(
            near_null_t1,
            inversion_time_seconds=0.5e-3,
            num_echoes=3,
            ny=1,
            maxoffs=0.0,
            num_workers=1,
            phase_workers=1,
        )
        short_signal = float(form_imaging_image(short_result, mode="single")[0, 0])
        near_null_signal = float(form_imaging_image(near_null_result, mode="single")[0, 0])

        self.assertGreater(short_signal, 100 * near_null_signal)

        rho = np.ones((2, 2), dtype=np.float64)
        t1_map = np.array(
            [
                [0.25e-3, 0.72e-3],
                [0.25e-3, 0.72e-3],
            ],
            dtype=np.float64,
        )
        t2_map = 50e-3 * np.ones_like(rho)
        maps = make_imaging_field_maps(
            rho,
            t1_map=t1_map,
            t2_map=t2_map,
            b1_tx_map=np.ones_like(rho),
            b1_rx_map=np.ones_like(rho),
        )

        encoded = run_t1_encoded_phase_encoded_cpmg_imaging(
            maps,
            inversion_time_seconds=0.5e-3,
            num_echoes=3,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )
        encoded_single = form_imaging_image(encoded, mode="single", echo_index=0)
        encoded_echo_sum = form_imaging_image(encoded, mode="echo_sum")
        encoded_fit = fit_imaging_echo_decay(encoded)

        self.assertEqual(encoded_single.shape, rho.shape)
        self.assertEqual(encoded_echo_sum.shape, rho.shape)
        self.assertEqual(encoded_fit.rho_map.shape, rho.shape)
        self.assertEqual(encoded_fit.t2_map.shape, rho.shape)
        self.assertGreater(float(np.max(encoded_single)), float(np.min(encoded_single)))
        self.assertTrue(np.all(np.isfinite(encoded_echo_sum)))
        self.assertTrue(np.any(encoded_fit.mask))
        with self.assertRaises(ValueError):
            run_t1_encoded_phase_encoded_cpmg_imaging(
                maps,
                inversion_time_seconds=0.0,
                num_echoes=1,
                ny=1,
            )
    def test_matched_cpmg_imaging_applies_custom_receive_map(self) -> None:
        rho = np.ones((1, 1), dtype=np.float64)
        maps = make_imaging_field_maps(
            rho,
            b1_tx_map=np.ones_like(rho),
            b1_rx_map=np.zeros_like(rho),
        )

        result = run_matched_cpmg_imaging(
            maps,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
        )

        np.testing.assert_allclose(result.kspace, 0.0, atol=1e-14)
        np.testing.assert_allclose(result.echo_integrals, 0.0, atol=1e-14)
    def test_tuned_cpmg_imaging_raw_receive_mode_matches_matlab_convention(self) -> None:
        rho = np.ones((1, 1), dtype=np.float64)
        zero_rx = make_imaging_field_maps(
            rho,
            b1_tx_map=np.ones_like(rho),
            b1_rx_map=np.zeros_like(rho),
        )
        one_rx = make_imaging_field_maps(
            rho,
            b1_tx_map=np.ones_like(rho),
            b1_rx_map=np.ones_like(rho),
        )

        zero_result = run_tuned_cpmg_imaging(
            zero_rx,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
        )
        one_result = run_tuned_cpmg_imaging(
            one_rx,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
        )

        np.testing.assert_allclose(zero_result.kspace, one_result.kspace, rtol=1e-13, atol=1e-13)
        self.assertGreater(float(np.max(np.abs(zero_result.kspace))), 0.0)
    def test_tuned_cpmg_imaging_weighted_receive_mode_applies_receive_map(self) -> None:
        rho = np.ones((1, 1), dtype=np.float64)
        maps = make_imaging_field_maps(
            rho,
            b1_tx_map=np.ones_like(rho),
            b1_rx_map=np.zeros_like(rho),
        )

        result = run_tuned_cpmg_imaging(
            maps,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
            receive_mode="weighted",
        )

        np.testing.assert_allclose(result.kspace, 0.0, atol=1e-14)
        np.testing.assert_allclose(result.echo_integrals, 0.0, atol=1e-14)
    def test_tuned_cpmg_imaging_probe_noise_uses_weighted_receive_mode(self) -> None:
        rho = np.ones((1, 1), dtype=np.float64)
        clean = run_tuned_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
            receive_mode="weighted",
        )
        noisy1 = run_tuned_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
            receive_mode="weighted",
            noise=NoiseSpec(model="probe", target_snr=15.0, seed=7),
        )
        noisy2 = run_tuned_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=1,
            num_workers=1,
            phase_workers=1,
            receive_mode="weighted",
            noise=NoiseSpec(model="probe", target_snr=15.0, seed=7),
        )

        np.testing.assert_allclose(noisy1.kspace, clean.kspace)
        np.testing.assert_allclose(noisy1.image, clean.image)
        self.assertIsNotNone(noisy1.kspace_noisy)
        self.assertIsNotNone(noisy1.image_noisy)
        self.assertEqual(noisy1.noise.model, "probe")
        np.testing.assert_allclose(noisy1.kspace_noisy, noisy2.kspace_noisy)
        self.assertGreater(
            float(np.max(np.abs(noisy1.kspace_noisy - noisy1.kspace))),
            0.0,
        )
        with self.assertRaises(ValueError):
            run_tuned_cpmg_imaging(
                rho,
                num_echoes=1,
                ny=1,
                num_workers=1,
                phase_workers=1,
                noise=NoiseSpec(model="probe", seed=7),
            )
    def test_tuned_cpmg_imaging_rejects_unknown_receive_mode(self) -> None:
        with self.assertRaises(ValueError):
            run_tuned_cpmg_imaging(
                np.ones((1, 1), dtype=np.float64),
                num_echoes=1,
                ny=1,
                receive_mode="receiver",
            )
    def test_ideal_cpmg_imaging_field_maps_match_legacy_inputs(self) -> None:
        rho = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        maps = make_imaging_field_maps(rho)
        legacy = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )
        container = run_ideal_cpmg_imaging(
            maps,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )

        np.testing.assert_allclose(container.kspace, legacy.kspace, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(container.image, legacy.image, rtol=1e-13, atol=1e-13)
    def test_ideal_cpmg_imaging_phase_parallel_matches_serial(self) -> None:
        rho = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        serial = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=5,
            num_workers=1,
            phase_workers=1,
        )
        parallel = run_ideal_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=5,
            num_workers=1,
            phase_workers=2,
        )
        np.testing.assert_allclose(parallel.kspace, serial.kspace, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.image, serial.image, rtol=1e-13, atol=1e-13)
    def test_probe_cpmg_imaging_returns_expected_shapes(self) -> None:
        rho = np.eye(2)
        for runner, probe in [
            (run_tuned_cpmg_imaging, "tuned"),
            (run_matched_cpmg_imaging, "matched"),
        ]:
            with self.subTest(probe=probe):
                result = runner(
                    rho,
                    num_echoes=1,
                    ny=3,
                    num_workers=1,
                    phase_workers=1,
                )
                self.assertEqual(result.probe, probe)
                self.assertEqual(result.kspace.shape, (2, 2, 1))
                self.assertEqual(result.image.shape, (2, 2, 1))
                self.assertEqual(result.magnitude.shape, (2, 2, 1))
                self.assertTrue(np.all(np.isfinite(result.kspace)))
                self.assertTrue(np.all(np.isfinite(result.magnitude)))
    def test_tuned_cpmg_imaging_phase_parallel_matches_serial(self) -> None:
        rho = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        serial = run_tuned_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=1,
        )
        parallel = run_tuned_cpmg_imaging(
            rho,
            num_echoes=1,
            ny=3,
            num_workers=1,
            phase_workers=2,
        )
        np.testing.assert_allclose(parallel.kspace, serial.kspace, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(parallel.image, serial.image, rtol=1e-13, atol=1e-13)
    def test_run_ideal_cpmg_imaging_matches_matlab(self) -> None:
        self._assert_imaging_fixture(
            "run_ideal_cpmg_imaging",
            run_ideal_cpmg_imaging,
            rtol=1e-13,
            atol=1e-10,
        )
    def test_run_tuned_cpmg_imaging_matches_matlab(self) -> None:
        self._assert_imaging_fixture(
            "run_tuned_cpmg_imaging",
            run_tuned_cpmg_imaging,
            rtol=1e-11,
            atol=1e-11,
        )
    def test_run_matched_cpmg_imaging_matches_matlab(self) -> None:
        self._assert_imaging_fixture(
            "run_matched_cpmg_imaging",
            run_matched_cpmg_imaging,
            rtol=1e-6,
            atol=3e-5,
        )
    def test_run_tuned_cpmg_train_matches_octave(self) -> None:
        self._assert_train_fixture(
            "run_tuned_cpmg_train",
            run_tuned_cpmg_train,
            numpts_expected=None,
            maxoffs=5,
            rtol=1e-11,
            atol=1e-11,
        )
    def test_run_untuned_cpmg_train_matches_octave(self) -> None:
        self._assert_train_fixture(
            "run_untuned_cpmg_train",
            run_untuned_cpmg_train,
            numpts_expected=None,
            maxoffs=5,
            rtol=1e-11,
            atol=1e-11,
        )
    def test_run_matched_cpmg_train_matches_matlab(self) -> None:
        self._assert_train_fixture(
            "run_matched_cpmg_train",
            run_matched_cpmg_train,
            numpts_expected=None,
            maxoffs=4,
            rtol=8e-2,
            atol=5e-2,
        )
