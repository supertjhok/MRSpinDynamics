"""Core numerical, parameter, probe, pulse, and WURST fixture checks."""

from tests.octave_fixtures.support import *


class OctaveCoreFixtureTests(OctaveFixtureBase):
    def test_numpy_compatibility_helpers(self) -> None:
        y = np.array([0.0, 1.0, 0.0])
        x = np.array([0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(trapezoid(y, x)), 0.5)
    def test_received_white_noise_is_seeded_and_scaled(self) -> None:
        signal = np.zeros(20000, dtype=np.complex128)
        spec = NoiseSpec(sigma=0.2, seed=123)

        noisy1, metadata1 = add_received_noise(signal, spec)
        noisy2, metadata2 = add_received_noise(signal, spec)

        np.testing.assert_allclose(noisy1, noisy2)
        self.assertEqual(metadata1, metadata2)
        self.assertEqual(metadata1.model, "white")
        self.assertEqual(metadata1.domain, "spectrum")
        self.assertGreater(metadata1.noise_rms, 0.0)
        self.assertEqual(metadata1.signal_rms, 0.0)
        self.assertEqual(metadata1.realized_snr, 0.0)
        self.assertAlmostEqual(float(np.std(np.real(noisy1))), 0.2, delta=0.01)
        self.assertAlmostEqual(float(np.std(np.imag(noisy1))), 0.2, delta=0.01)
    def test_cpmg_noise_preserves_clean_signal_and_repeats_with_seed(self) -> None:
        clean = run_ideal_cpmg(numpts=33, maxoffs=5)
        noisy1 = run_ideal_cpmg(
            numpts=33,
            maxoffs=5,
            noise=NoiseSpec(sigma=1e-3, seed=123),
        )
        noisy2 = run_ideal_cpmg(
            numpts=33,
            maxoffs=5,
            noise=NoiseSpec(sigma=1e-3, seed=123),
        )

        self.assertIsNone(clean.mrx_noisy)
        self.assertIsNone(clean.echo_noisy)
        np.testing.assert_allclose(noisy1.mrx, clean.mrx)
        np.testing.assert_allclose(noisy1.echo, clean.echo)
        self.assertIsNotNone(noisy1.mrx_noisy)
        self.assertIsNotNone(noisy1.echo_noisy)
        np.testing.assert_allclose(noisy1.mrx_noisy, noisy2.mrx_noisy)
        np.testing.assert_allclose(noisy1.echo_noisy, noisy2.echo_noisy)
        self.assertGreater(float(np.max(np.abs(noisy1.mrx_noisy - noisy1.mrx))), 0.0)
        self.assertGreater(noisy1.noise.noise_rms, 0.0)
        self.assertGreater(noisy1.noise.signal_rms, 0.0)
        self.assertGreater(noisy1.noise.realized_snr, 0.0)
    def test_cpmg_time_domain_noise_adds_echo_noise_only(self) -> None:
        clean = run_ideal_cpmg(numpts=33, maxoffs=5)
        noisy = run_ideal_cpmg(
            numpts=33,
            maxoffs=5,
            noise=NoiseSpec(sigma=1e-3, seed=456, domain="time"),
        )

        np.testing.assert_allclose(noisy.mrx, clean.mrx)
        np.testing.assert_allclose(noisy.echo, clean.echo)
        self.assertIsNone(noisy.mrx_noisy)
        self.assertIsNotNone(noisy.echo_noisy)
        self.assertEqual(noisy.noise.domain, "time")
        self.assertGreater(float(np.max(np.abs(noisy.echo_noisy - noisy.echo))), 0.0)

        train = run_ideal_cpmg_train(
            numpts=17,
            maxoffs=4,
            num_echoes=2,
            rephase_action="ignore",
            noise=NoiseSpec(sigma=1e-3, seed=456, domain="time"),
        )
        self.assertIsNone(train.mrx_noisy)
        self.assertIsNotNone(train.echo_noisy)
        self.assertIsNotNone(train.echo_integrals_noisy)
        self.assertEqual(train.noise.domain, "time")
        with self.assertRaises(ValueError):
            run_tuned_cpmg(
                numpts=17,
                maxoffs=4,
                noise=NoiseSpec(model="probe", target_snr=10.0, seed=1, domain="time"),
            )
    def test_probe_noise_uses_receiver_density_without_overwriting_clean_signal(self) -> None:
        clean = run_tuned_cpmg(numpts=17, maxoffs=4)
        noisy = run_tuned_cpmg(
            numpts=17,
            maxoffs=4,
            noise=NoiseSpec(model="probe", target_snr=20.0, seed=4),
        )

        np.testing.assert_allclose(noisy.mrx, clean.mrx)
        np.testing.assert_allclose(noisy.echo, clean.echo)
        self.assertIsNotNone(noisy.mrx_noisy)
        self.assertIsNotNone(noisy.echo_noisy)
        self.assertEqual(noisy.noise.model, "probe")
        self.assertGreater(float(np.max(np.abs(noisy.mrx_noisy - noisy.mrx))), 0.0)
    def test_matched_filter_snr_estimate_matches_probe_noise_prediction(self) -> None:
        numpts = 41
        maxoffs = 6.0
        clean = run_tuned_cpmg(numpts=numpts, maxoffs=maxoffs)
        params, sp, pp = set_params_tuned_orig(numpts=numpts)
        del_w = np.linspace(-maxoffs, maxoffs, numpts)
        sp = replace(
            sp,
            numpts=numpts,
            maxoffs=maxoffs,
            del_w=del_w,
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
        )
        pnoise, frequencies = tuned_probe_output_noise_density(sp, pp)

        noisy_rows = []
        noise_scale = None
        for seed in range(300):
            noisy, metadata = add_received_noise(
                clean.mrx,
                NoiseSpec(model="probe", target_snr=25.0, seed=seed),
                pnoise=pnoise,
                frequencies=frequencies,
                sample_axis=clean.del_w,
            )
            noisy_rows.append(noisy)
            noise_scale = metadata.scale

        snr = estimate_matched_filter_snr(
            clean.mrx,
            np.asarray(noisy_rows),
            pnoise=pnoise,
            frequencies=frequencies,
            offsets=clean.del_w,
            noise_scale=noise_scale,
        )

        self.assertIsNotNone(snr.predicted_snr)
        self.assertIsNotNone(snr.predicted_noise_rms)
        self.assertGreater(snr.measured_snr, 0.0)
        self.assertLess(
            abs(snr.measured_noise_rms - snr.predicted_noise_rms)
            / snr.predicted_noise_rms,
            0.18,
        )
        self.assertLess(
            abs(snr.measured_snr - snr.predicted_snr) / snr.predicted_snr,
            0.18,
        )
    def test_rephasing_analysis_recommends_finer_grid(self) -> None:
        del_w = np.linspace(-5, 5, 11)
        analysis = analyze_rephasing(del_w, max_time=12.0, safety_factor=1.25)
        self.assertFalse(analysis.ok)
        self.assertGreaterEqual(
            analysis.recommended_numpts,
            recommended_numpts_for_rephasing(5, 12.0, safety_factor=1.25),
        )
        with self.assertWarns(RuntimeWarning):
            check_rephasing(del_w, max_time=12.0, safety_factor=1.25, action="warn")
    def test_calc_time_domain_echo_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_time_domain_echo.csv", delimiter=",")
        echo_ref = table[:, 0] + 1j * table[:, 1]
        tvect_ref = table[:, 2]

        del_w = np.linspace(-4, 4, 17)
        spect = np.exp(-0.25 * del_w**2) * np.exp(1j * 0.2 * del_w)
        echo, tvect = calc_time_domain_echo(spect, del_w)

        np.testing.assert_allclose(echo, echo_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-13, atol=1e-13)
    def test_calc_time_domain_echo_arb_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_time_domain_echo_arb.csv", delimiter=",")
        echo_ref = table[:, 0] + 1j * table[:, 1]
        tvect_ref = table[:, 2]

        del_w = np.linspace(-4, 4, 17)
        mrx = np.exp(-0.2 * del_w**2) * np.exp(1j * (0.3 * del_w + 0.05 * del_w**2))
        tacq = 4 * np.pi
        tdw = tacq / 32
        echo, tvect = calc_time_domain_echo_arb(mrx, del_w, tacq, tdw)

        np.testing.assert_allclose(echo, echo_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-13, atol=1e-13)
    def test_sim_spin_dynamics_asymp_mag3_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "sim_spin_dynamics_asymp_mag3.csv", delimiter=",")
        masy_ref = table[:, 0] + 1j * table[:, 1]
        del_w = table[:, 2]

        tp = np.array([np.pi / 2, 0.5, np.pi / 3])
        phi = np.array([np.pi / 2, 0, np.pi / 4])
        amp = np.array([1, 0, 0.75])
        t_acq = 2 * np.pi
        neff = np.zeros((3, del_w.size))
        neff[0, :] = np.cos(0.15 * del_w)
        neff[1, :] = np.sin(0.15 * del_w)
        neff[2, :] = 0.25
        neff = neff / np.sqrt(np.sum(neff**2, axis=0))

        masy = sim_spin_dynamics_asymp_mag3(tp, phi, amp, neff, del_w, t_acq)

        np.testing.assert_allclose(masy, masy_ref, rtol=1e-13, atol=1e-13)
    def test_sim_spin_dynamics_exc_returns_excitation_vector(self) -> None:
        del_w = np.linspace(-2.0, 2.0, 9)
        mvect = sim_spin_dynamics_exc(
            np.array([np.pi / 4, -0.5]),
            np.array([np.pi / 2, 0.0]),
            np.array([2.0, 0.0]),
            del_w,
        )

        self.assertEqual(mvect.shape, (3, 9))
        self.assertTrue(np.all(np.isfinite(mvect)))
        center = mvect[:, del_w.size // 2]
        np.testing.assert_allclose(center[0], 1.0, atol=1e-14)
        np.testing.assert_allclose(center[1], 0.0, atol=1e-14)
        np.testing.assert_allclose(center[2], 0.0, atol=1e-14)
    def test_set_params_ideal_matches_octave(self) -> None:
        values = np.loadtxt(FIXTURES / "set_params_ideal.csv", delimiter=",")
        sp, pp = set_params_ideal()

        actual = np.array(
            [
                sp.k,
                sp.T,
                sp.gamma,
                sp.grad,
                sp.D,
                sp.f0,
                sp.fin,
                sp.m0,
                sp.mth,
                sp.numpts,
                sp.maxoffs,
                sp.del_w[0],
                sp.del_w[-1],
                sp.mf_type,
                sp.plt_tx,
                sp.plt_rx,
                sp.plt_sequence,
                sp.plt_axis,
                sp.plt_mn,
                sp.plt_echo,
                pp.N,
                pp.T_90,
                pp.T_180,
                pp.psi,
                pp.preDelay,
                pp.postDelay,
                pp.texc[0],
                pp.pexc[0],
                pp.aexc[0],
                pp.tcorr,
                *pp.tref,
                *pp.pref,
                *pp.aref,
                pp.pcycle,
                pp.tacq[0],
                pp.tdw,
                pp.amp_zero,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, values, rtol=1e-14, atol=1e-14)
    def test_set_params_ideal_fid_matches_octave(self) -> None:
        values = np.loadtxt(FIXTURES / "set_params_ideal_fid.csv", delimiter=",")
        sp, pp = set_params_ideal_fid()

        actual = np.array(
            [
                sp.k,
                sp.T,
                sp.f0,
                sp.fin,
                sp.m0,
                sp.mth,
                sp.numpts,
                sp.maxoffs,
                sp.del_w[0],
                sp.del_w[-1],
                sp.w_1[0],
                sp.w_1r[0],
                sp.T1[0],
                sp.T2[0],
                sp.mf_type,
                sp.plt_tx,
                sp.plt_rx,
                sp.plt_sequence,
                sp.plt_axis,
                sp.plt_mn,
                sp.plt_echo,
                pp.N,
                pp.T_90,
                pp.acqDelay,
                pp.acqTpTime,
                pp.psi,
                pp.tacq,
                pp.tdw,
                pp.amp_zero,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, values, rtol=1e-14, atol=1e-14)
    def test_set_params_tuned_orig_matches_octave(self) -> None:
        values = np.loadtxt(FIXTURES / "set_params_tuned_orig.csv", delimiter=",")
        params, sp, pp = set_params_tuned_orig()

        actual = np.array(
            [
                sp.k,
                sp.T,
                sp.gamma,
                sp.f0,
                sp.fin,
                sp.w0,
                sp.L,
                sp.Q,
                sp.R,
                sp.C,
                sp.Rs,
                sp.Vs,
                sp.Rin,
                sp.Cin,
                sp.Rd,
                sp.NF,
                sp.vn,
                sp.in_,
                sp.m0,
                sp.mth,
                sp.numpts,
                sp.maxoffs,
                sp.del_w[0],
                sp.del_w[-1],
                sp.mf_type,
                sp.plt_tx,
                sp.plt_rx,
                sp.plt_sequence,
                sp.plt_axis,
                sp.plt_mn,
                sp.plt_echo,
                sp.sens,
                pp.w,
                pp.N,
                pp.T_90,
                pp.T_180,
                pp.psi,
                pp.preDelay,
                pp.postDelay,
                pp.texc[0],
                pp.pexc[0],
                pp.aexc[0],
                pp.tcorr,
                pp.tqs,
                pp.trd,
                *pp.tref,
                *pp.pref,
                *pp.aref,
                *pp.Rsref,
                pp.pcycle,
                pp.tacq[0],
                pp.tdw,
                pp.amp_zero,
                params.texc[0],
                params.pexc[0],
                params.aexc[0],
                params.trd,
                params.tref[0],
                params.pref[0],
                params.aref[0],
                params.tfp,
                params.tqs,
                params.tacq[0],
                *params.Rs,
                params.pcycle,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, values, rtol=1e-14, atol=1e-14)
    def test_set_params_untuned_orig_matches_octave(self) -> None:
        values = np.loadtxt(FIXTURES / "set_params_untuned_orig.csv", delimiter=",")
        params, sp, pp = set_params_untuned_orig()

        actual = np.array(
            [
                sp.k,
                sp.T,
                sp.gamma,
                sp.f0,
                sp.fin,
                sp.w0,
                sp.L,
                sp.Q,
                sp.R,
                sp.C,
                sp.Rs,
                sp.Vs,
                sp.Rin,
                sp.Cin,
                sp.Rd,
                sp.Rdup,
                sp.Nrx,
                sp.krx,
                sp.L1,
                sp.R1,
                sp.L2,
                sp.R2,
                sp.NF,
                sp.vn,
                sp.in_,
                sp.m0,
                sp.mth,
                sp.numpts,
                sp.maxoffs,
                sp.del_w[0],
                sp.del_w[-1],
                sp.mf_type,
                sp.plt_tx,
                sp.plt_rx,
                sp.plt_sequence,
                sp.plt_axis,
                sp.plt_mn,
                sp.plt_echo,
                sp.sens,
                pp.w,
                pp.N,
                pp.T_90,
                pp.T_180,
                pp.psi,
                pp.preDelay,
                pp.postDelay,
                pp.texc[0],
                pp.pexc[0],
                pp.aexc[0],
                pp.tcorr,
                pp.tqs,
                pp.trd,
                *pp.tref,
                *pp.pref,
                *pp.aref,
                *pp.Rsref,
                pp.tacq[0],
                pp.tdw,
                pp.amp_zero,
                params.texc[0],
                params.pexc[0],
                params.aexc[0],
                params.trd,
                params.tref[0],
                params.pref[0],
                params.aref[0],
                params.tfp,
                params.tqs,
                params.tacq[0],
                *params.Rs,
                params.pcycle,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, values, rtol=1e-14, atol=1e-14)
    def test_set_params_matched_orig_matches_matlab(self) -> None:
        values = np.loadtxt(FIXTURES / "set_params_matched_orig.csv", delimiter=",")
        sp, pp = set_params_matched_orig()

        actual = np.array(
            [
                sp.k,
                sp.T,
                sp.gamma,
                sp.grad,
                sp.D,
                sp.f0,
                sp.fin,
                sp.L,
                sp.Q,
                sp.R,
                sp.Rs,
                sp.Rin,
                sp.NF,
                sp.m0,
                sp.mth,
                sp.numpts,
                sp.maxoffs,
                sp.del_w[0],
                sp.del_w[-1],
                sp.mf_type,
                sp.plt_tx,
                sp.plt_rx,
                sp.plt_sequence,
                sp.plt_axis,
                sp.plt_mn,
                sp.plt_echo,
                pp.N,
                pp.T_90,
                pp.T_180,
                pp.psi,
                pp.preDelay,
                pp.postDelay,
                pp.texc[0],
                pp.pexc[0],
                pp.aexc[0],
                pp.tcorr,
                pp.trd,
                *pp.tref,
                *pp.pref,
                *pp.aref,
                pp.tacq[0],
                pp.tdw,
                pp.amp_zero,
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(actual, values, rtol=1e-14, atol=1e-14)
    def test_jmr_parameter_constructors_return_expected_defaults(self) -> None:
        tuned_sp, tuned_pp = set_params_tuned_jmr(numpts=17)
        untuned_sp, untuned_pp = set_params_untuned_jmr(numpts=17)
        matched_sp, matched_pp = set_params_matched_jmr(numpts=17)

        self.assertEqual(tuned_sp.numpts, 17)
        self.assertEqual(untuned_sp.numpts, 17)
        self.assertEqual(matched_sp.numpts, 17)
        np.testing.assert_allclose(tuned_pp.tref, [75e-6, 50e-6, 75e-6])
        np.testing.assert_allclose(untuned_pp.tref, [20e-6, 50e-6, 50e-6])
        np.testing.assert_allclose(matched_pp.tref, [20e-6, 50e-6, 70e-6])
        self.assertEqual(tuned_sp.f0, 0.5e6)
        self.assertEqual(untuned_sp.vn, 0.45e-9)
        self.assertEqual(matched_sp.Rs, 50.0)
    def test_tuned_spa_parameter_constructor_matches_matlab_defaults(self) -> None:
        params, sp, pp = set_params_tuned_spa(numpts=17)

        self.assertEqual(sp.numpts, 17)
        self.assertEqual(sp.f0, 8e6)
        self.assertEqual(sp.fin, 8e6)
        self.assertEqual(sp.Q, 50.0)
        np.testing.assert_allclose(sp.L, 10e-6 * (1e6 / 8e6))
        np.testing.assert_allclose(pp.T_90, 24e-6)
        np.testing.assert_allclose(pp.preDelay, 144e-6)
        np.testing.assert_allclose(pp.postDelay, 144e-6)
        np.testing.assert_allclose(pp.tacq, [4 * pp.T_180])
        np.testing.assert_allclose(params.tfp, pp.preDelay)
        np.testing.assert_allclose(params.Rs, [2.0, 2.0, 20.0])
    def test_untuned_spa_parameter_constructor_matches_matlab_defaults(self) -> None:
        params, sp, pp = set_params_untuned_spa(numpts=17)

        self.assertEqual(sp.numpts, 17)
        self.assertEqual(sp.f0, 8e6)
        self.assertEqual(sp.fin, 8e6)
        self.assertEqual(sp.Q, 50.0)
        np.testing.assert_allclose(sp.L, 10e-6 * (1e6 / 8e6))
        np.testing.assert_allclose(sp.C, 1 / ((2 * np.pi * 10 * sp.f0) ** 2 * sp.L))
        np.testing.assert_allclose(pp.T_90, 24e-6)
        np.testing.assert_allclose(pp.preDelay, 144e-6)
        np.testing.assert_allclose(pp.postDelay, 144e-6)
        np.testing.assert_allclose(pp.tacq, [4 * pp.T_180])
        np.testing.assert_allclose(params.tfp, pp.preDelay)
        np.testing.assert_allclose(params.Rs, [2.0, 2.0, 20.0])
    def test_matched_spa_parameter_constructor_matches_matlab_defaults(self) -> None:
        sp, pp = set_params_matched_spa(numpts=9)

        self.assertEqual(sp.numpts, 9)
        self.assertEqual(sp.f0, 8e6)
        self.assertEqual(sp.fin, 8e6)
        self.assertEqual(sp.Q, 50.0)
        np.testing.assert_allclose(sp.L, 10e-6 * (1e6 / 8e6))
        np.testing.assert_allclose(pp.T_90, 24e-6)
        np.testing.assert_allclose(pp.preDelay, 144e-6)
        np.testing.assert_allclose(pp.postDelay, 144e-6)
        np.testing.assert_allclose(pp.trd, 3 * pp.T_90)
        np.testing.assert_allclose(pp.tacq, [4 * pp.T_180])
    def test_calc_rotation_matrix_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_rotation_matrix.csv", delimiter=",")
        del_w = table[:, 0]
        w_1 = table[:, -1]
        tp = np.array([0.25, 0.5, 0.75])
        phi = np.array([0, np.pi / 3, -np.pi / 5])
        amp = np.array([1.0, 0.6, 1.2])

        rtot = calc_rotation_matrix(del_w, w_1, tp, phi, amp)
        names = [
            "R_00",
            "R_0p",
            "R_0m",
            "R_p0",
            "R_m0",
            "R_pp",
            "R_mm",
            "R_pm",
            "R_mp",
        ]
        for idx, name in enumerate(names):
            ref = table[:, 1 + 2 * idx] + 1j * table[:, 2 + 2 * idx]
            np.testing.assert_allclose(
                getattr(rtot, name),
                ref,
                rtol=1e-13,
                atol=1e-13,
                err_msg=name,
            )
    def test_sim_spin_dynamics_arb10_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "sim_spin_dynamics_arb10.csv", delimiter=",")
        acq_count = int(np.max(table[:, 0]))
        numpts = int(np.max(table[:, 1]))
        macq_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        for row in table:
            macq_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]

        del_w = np.linspace(-6, 6, numpts)
        w_1 = 1 + 0.08 * np.sin(del_w / 2)
        rtot = [
            calc_rotation_matrix(
                del_w,
                w_1,
                np.array([np.pi / 2]),
                np.array([np.pi / 2]),
                np.array([1]),
            ),
            calc_rotation_matrix(
                del_w,
                w_1,
                np.array([0.4 * np.pi, 0.6 * np.pi]),
                np.array([0, np.pi / 4]),
                np.array([0.8, 1.1]),
            ),
        ]
        params = {
            "tp": np.array(
                [
                    np.pi / 2,
                    1.2 * np.pi,
                    0.7 * np.pi,
                    0.9 * np.pi,
                    0.5 * np.pi,
                    1.1 * np.pi,
                ]
            ),
            "pul": np.array([1, 0, 2, 0, 0, 2]),
            "Rtot": rtot,
            "amp": np.array([1, 0, 1, 0, 0, 1]),
            "acq": np.array([0, 1, 0, 1, 1, 0]),
            "grad": np.array([0, 0.2, 0, -0.15, 0.1, 0]),
            "del_w": del_w,
            "del_wg": np.linspace(-1, 1, numpts),
            "T1n": 120 + 10 * np.cos(del_w / 3),
            "T2n": 45 + 5 * np.sin(del_w / 4),
            "m0": 0.9 + 0.05 * np.cos(del_w),
            "mth": 1.1 + 0.03 * np.sin(del_w),
        }

        macq = sim_spin_dynamics_arb10(params)
        macq_chunked = sim_spin_dynamics_arb10_chunked(
            params,
            num_workers=2,
            min_chunk_size=4,
        )

        np.testing.assert_allclose(macq, macq_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(macq_chunked, macq, rtol=1e-13, atol=1e-13)
    def test_calc_macq_ideal_probe_relax4_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_macq_ideal_probe_relax4.csv", delimiter=",")
        acq_count = int(np.max(table[:, 0]))
        numpts = int(np.max(table[:, 1]))
        macq_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        for row in table:
            macq_ref[int(row[0]) - 1, int(row[1]) - 1] = row[2] + 1j * row[3]

        del_w = np.linspace(-5, 5, numpts)
        sp = {
            "del_w": del_w,
            "del_wg": np.linspace(-0.75, 0.75, numpts),
            "w_1": 0.95 + 0.07 * np.cos(del_w / 2),
            "T1": 1.4 + 0.08 * np.cos(del_w / 3),
            "T2": 0.9 + 0.05 * np.sin(del_w / 4),
            "m0": 0.8 + 0.04 * np.cos(del_w),
            "mth": 1.05 + 0.02 * np.sin(del_w),
        }
        rtot = [
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([np.pi / 2]),
                np.array([np.pi / 2]),
                np.array([1.0]),
            ),
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([0.45 * np.pi, 0.45 * np.pi]),
                np.array([0, np.pi / 4]),
                np.array([0.8, 1.05]),
            ),
            calc_rotation_matrix(
                del_w,
                sp["w_1"],
                np.array([1.1 * np.pi]),
                np.array([-np.pi / 5]),
                np.array([0.9]),
            ),
        ]
        pp = {
            "T_90": 25e-6,
            "tp": np.array(
                [
                    np.pi / 2,
                    0.35 * np.pi,
                    0.9 * np.pi,
                    0.4 * np.pi,
                    0.55 * np.pi,
                    1.1 * np.pi,
                ]
            ),
            "amp": np.array([1, 0, 1, 0, 0, 1]),
            "acq": np.array([0, 1, 0, 1, 1, 0]),
            "grad": np.array([0, 0.25, 0, -0.2, 0.15, 0]),
            "pul": np.array([1, 0, 2, 0, 0, 3]),
            "Rtot": rtot,
        }

        macq = calc_macq_ideal_probe_relax4(sp, pp)

        np.testing.assert_allclose(macq, macq_ref, rtol=1e-13, atol=1e-13)
    def test_calc_macq_tuned_probe_relax4_matches_octave(self) -> None:
        macq_ref, mrx_ref = self._load_probe_relax4_fixture(
            "calc_macq_tuned_probe_relax4.csv"
        )
        sp, pp = self._probe_relax4_inputs(macq_ref.shape[1])
        del_w = sp["del_w"]
        sp = {
            **sp,
            "tf": (0.8 + 0.03 * del_w) * np.exp(1j * 0.15 * del_w),
            "w_1r": 0.9 + 0.02 * np.cos(del_w),
        }

        macq, mrx = calc_macq_tuned_probe_relax4(sp, pp)

        np.testing.assert_allclose(macq, macq_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(mrx, mrx_ref, rtol=1e-13, atol=1e-13)
    def test_calc_macq_matched_probe_relax4_matches_octave(self) -> None:
        macq_ref, mrx_ref = self._load_probe_relax4_fixture(
            "calc_macq_matched_probe_relax4.csv"
        )
        sp, pp = self._probe_relax4_inputs(macq_ref.shape[1])
        del_w = sp["del_w"]
        sp = {
            **sp,
            "tf2": (0.7 - 0.02 * del_w) * np.exp(-1j * 0.11 * del_w),
            "w_1r": 1.0 + 0.03 * np.sin(del_w / 2),
        }

        macq, mrx = calc_macq_matched_probe_relax4(sp, pp)

        np.testing.assert_allclose(macq, macq_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(mrx, mrx_ref, rtol=1e-13, atol=1e-13)
    def test_calc_macq_untuned_probe_relax4_applies_receiver(self) -> None:
        sp, pp = self._probe_relax4_inputs(17)
        del_w = sp["del_w"]
        sp = {
            **sp,
            "tf": (0.65 + 0.01 * del_w) * np.exp(1j * 0.07 * del_w),
            "w_1r": 0.85 + 0.04 * np.cos(del_w / 3),
        }

        macq, mrx = calc_macq_untuned_probe_relax4(sp, pp)

        expected = macq * sp["tf"][np.newaxis, :] * sp["w_1r"][np.newaxis, :]
        np.testing.assert_allclose(mrx, expected, rtol=1e-13, atol=1e-13)
    def test_calc_rot_axis_arba_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_rot_axis_arba.csv", delimiter=",")
        del_w = table[:, 0]
        n3_ref = table[:, 1:4].T
        n4_ref = table[:, 4:7].T
        alpha_ref = table[:, 7]

        tp = np.array([0.8, np.pi, 0.4, np.pi / 3])
        phi = np.array([0, np.pi / 2, 0, np.pi / 5])
        amp = np.array([0, 1, 0, 0.6])

        n3 = calc_rot_axis_arba3(tp, phi, amp, del_w)
        n4, alpha = calc_rot_axis_arba4(tp, phi, amp, del_w)

        np.testing.assert_allclose(n3, n3_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(n4, n4_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(alpha, alpha_ref, rtol=1e-13, atol=1e-13)
    def test_calc_v0crit_returns_centered_axis_derivative(self) -> None:
        del_w = np.linspace(-2.0, 2.0, 9)
        neff, alpha = calc_rot_axis_arba4(
            np.array([0.2, np.pi, 0.2]),
            np.array([0.0, 0.3, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            del_w,
        )

        v0crit = calc_v0crit(del_w, neff, alpha)

        self.assertEqual(v0crit.shape, del_w.shape)
        self.assertTrue(np.all(np.isfinite(v0crit)))
        self.assertTrue(np.all(v0crit > 0))
    def test_calc_masy_ideal_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_masy_ideal.csv", delimiter=",")
        masy_ref = table[:, 0] + 1j * table[:, 1]
        del_w = table[:, 2]

        sp = {
            "del_w": del_w,
            "plt_axis": 0,
            "plt_rx": 0,
        }
        pp = {
            "T_90": 25e-6,
            "tref": np.array([75e-6, 50e-6, 75e-6]),
            "pref": np.array([0, 0, 0]),
            "aref": np.array([0, 1, 0]),
            "texc": np.array([25e-6]),
            "pexc": np.array([np.pi / 2]),
            "aexc": np.array([1]),
            "tacq": np.array([150e-6]),
        }

        masy = calc_masy_ideal(sp, pp)

        np.testing.assert_allclose(masy, masy_ref, rtol=1e-13, atol=1e-13)
    def test_sim_fid_ideal_matches_octave(self) -> None:
        macq_table = np.loadtxt(FIXTURES / "sim_fid_ideal_macq.csv", delimiter=",")
        echo_table = np.loadtxt(FIXTURES / "sim_fid_ideal_echo.csv", delimiter=",")

        macq_ref = macq_table[:, 0] + 1j * macq_table[:, 1]
        del_w = macq_table[:, 2]
        echo_ref = echo_table[:, 0] + 1j * echo_table[:, 1]
        tvect_ref = echo_table[:, 2]

        sp = {
            "del_w": del_w,
            "w_1": 0.9 + 0.1 * np.cos(del_w / 2),
            "T1": 1.5 + 0.1 * np.cos(del_w / 3),
            "T2": 1.2 + 0.1 * np.sin(del_w / 4),
            "m0": 1.0,
            "mth": 1.0,
        }
        pp = {
            "T_90": 25e-6,
            "acqDelay": 25e-6 / 10,
            "tacq": 25e-6,
            "tdw": 0.5e-6,
        }

        macq, echo, tvect = sim_fid_ideal(sp, pp)

        np.testing.assert_allclose(macq[0, :], macq_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(echo, echo_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-13, atol=1e-13)
    def test_tuned_probe_lp_orig_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "tuned_probe_lp_orig.csv", delimiter=",")
        tvect_ref = table[:, 0]
        icr_ref = table[:, 1] + 1j * table[:, 2]

        _params, sp, pp = set_params_tuned_orig(numpts=21)
        sp = replace(
            sp,
            numpts=21,
            del_w=np.linspace(-5, 5, 21),
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
        )
        tvect, icr, _tvect_raw, _ic = tuned_probe_lp_orig(sp, pp)

        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(icr, icr_ref, rtol=1e-12, atol=1e-12)
    def test_calc_masy_tuned_probe_lp_orig_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_masy_tuned_probe_lp_orig.csv", delimiter=",")
        del_w = table[:, 0]
        masy_ref = table[:, 1] + 1j * table[:, 2]
        mrx_ref = table[:, 3] + 1j * table[:, 4]
        snr_ref = table[0, 5]

        params, sp, pp = set_params_tuned_orig(numpts=del_w.size)
        sp = replace(
            sp,
            numpts=del_w.size,
            del_w=del_w,
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
        )

        mrx, masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)

        np.testing.assert_allclose(masy, masy_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(mrx, mrx_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(snr, snr_ref, rtol=1e-11, atol=1e-11)
    def test_untuned_probe_lp_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "untuned_probe_lp.csv", delimiter=",")
        tvect_ref = table[:, 0]
        icr_ref = table[:, 1] + 1j * table[:, 2]

        _params, sp, pp = set_params_untuned_orig(numpts=21)
        sp = replace(
            sp,
            numpts=21,
            del_w=np.linspace(-5, 5, 21),
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
            plt_axis=0,
        )
        tvect, icr, _tvect_raw, _ic = untuned_probe_lp(sp, pp)

        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(icr, icr_ref, rtol=1e-12, atol=1e-12)
    def test_calc_masy_untuned_probe_lp_matches_octave(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_masy_untuned_probe_lp.csv", delimiter=",")
        del_w = table[:, 0]
        masy_ref = table[:, 1] + 1j * table[:, 2]
        mrx_ref = table[:, 3] + 1j * table[:, 4]
        snr_ref = table[0, 5]

        params, sp, pp = set_params_untuned_orig(numpts=del_w.size)
        sp = replace(
            sp,
            numpts=del_w.size,
            del_w=del_w,
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
            plt_axis=0,
        )

        mrx, masy, snr = calc_masy_untuned_probe_lp(params, sp, pp)

        np.testing.assert_allclose(masy, masy_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(mrx, mrx_ref, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(snr, snr_ref, rtol=1e-11, atol=1e-11)
    def test_find_coil_current_matched_matches_matlab(self) -> None:
        table = np.loadtxt(FIXTURES / "find_coil_current_matched.csv", delimiter=",")
        tvect_ref = table[:, 0]
        icr_ref = table[:, 1] + 1j * table[:, 2]

        sp, pp = set_params_matched_orig(numpts=11)
        sp = replace(
            sp,
            numpts=11,
            del_w=np.linspace(-4, 4, 11),
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
            plt_axis=0,
            plt_mn=0,
        )
        c1, c2 = matching_network_design2(sp.L, sp.Q, sp.f0, sp.Rs)
        sp_curr = {**sp.__dict__, "C1": c1, "C2": c2}
        pp_curr = {
            **pp.__dict__,
            "tp": np.concatenate([pp.texc, [pp.trd]]),
            "phi": np.concatenate([pp.pexc, [0.0]]),
            "amp": np.concatenate([pp.aexc, [0.0]]),
        }
        tvect, icr, _tf1, _tf2 = find_coil_current(sp_curr, pp_curr)

        np.testing.assert_allclose(tvect, tvect_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(icr, icr_ref, rtol=7e-2, atol=7e-3)
    def test_matching_network_design_high_q_fallback_is_finite(self) -> None:
        c1, c2 = matching_network_design2(10e-6, 50_000, 1e6, 50)
        self.assertGreater(c1, 0)
        self.assertGreater(c2, 0)
        self.assertTrue(np.isfinite(c1))
        self.assertTrue(np.isfinite(c2))
    def test_calc_masy_matched_probe_orig_matches_matlab(self) -> None:
        table = np.loadtxt(FIXTURES / "calc_masy_matched_probe_orig.csv", delimiter=",")
        del_w = table[:, 0]
        masy_ref = table[:, 1] + 1j * table[:, 2]
        mrx_ref = table[:, 3] + 1j * table[:, 4]
        snr_ref = table[0, 5]
        c1_ref = table[0, 6]
        c2_ref = table[0, 7]

        sp, pp = set_params_matched_orig(numpts=del_w.size)
        sp = replace(
            sp,
            numpts=del_w.size,
            del_w=del_w,
            plt_tx=0,
            plt_rx=0,
            plt_echo=0,
            plt_axis=0,
            plt_mn=0,
        )

        c1, c2 = matching_network_design2(sp.L, sp.Q, sp.f0, sp.Rs)
        mrx, masy, snr = calc_masy_matched_probe_orig(sp, pp)

        np.testing.assert_allclose(c1, c1_ref, rtol=1e-7, atol=1e-18)
        np.testing.assert_allclose(c2, c2_ref, rtol=1e-7, atol=1e-18)
        np.testing.assert_allclose(masy, masy_ref, rtol=8e-2, atol=2e-2)
        np.testing.assert_allclose(mrx, mrx_ref, rtol=8e-2, atol=2e-2)
        np.testing.assert_allclose(snr, snr_ref, rtol=8e-2, atol=2e-2)
    def test_quantize_phase_matches_matlab(self) -> None:
        table = np.loadtxt(FIXTURES / "pulse_quantize_phase.csv", delimiter=",")
        actual = quantize_phase(table[:, 0], num_phases=8)
        np.testing.assert_allclose(actual, table[:, 1], rtol=1e-14, atol=1e-14)
    def test_untuned_segment_adjustment_matches_matlab(self) -> None:
        table = np.loadtxt(FIXTURES / "pulse_untuned_segment_adjust.csv", delimiter=",")
        meta = np.loadtxt(FIXTURES / "pulse_untuned_segment_adjust_meta.csv", delimiter=",")

        result = adjust_untuned_segment_lengths(table[:, 0], table[:, 1], num_phases=8)

        np.testing.assert_allclose(result.segment_lengths, table[:, 2], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(result.phases, table[:, 3], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(result.phase_rotation, meta[0], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(result.clock_period, meta[1], rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(result.steady_state_phase, meta[2], rtol=1e-14, atol=1e-14)
    def test_tuned_rectangular_pulse_response_matches_matlab(self) -> None:
        self._assert_pulse_response_fixture(
            "pulse_tuned_rectangular",
            tuned_rectangular_pulse_response(numpts=17),
            rtol=1e-13,
            atol=1e-13,
        )
    def test_untuned_rectangular_pulse_response_matches_matlab(self) -> None:
        self._assert_pulse_response_fixture(
            "pulse_untuned_rectangular",
            untuned_rectangular_pulse_response(numpts=17),
            rtol=1e-13,
            atol=1e-13,
        )
    def test_matched_rectangular_pulse_response_matches_matlab(self) -> None:
        table = np.loadtxt(FIXTURES / "pulse_matched_rectangular.csv", delimiter=",")
        result = matched_rectangular_pulse_response(numpts=17)
        rot_idx = self._fixture_sample_indices(result.rotating_time.size)
        tf_idx = table[:, 3]
        tf_idx = tf_idx[np.isfinite(tf_idx)].astype(int) - 1

        np.testing.assert_allclose(result.rotating_time[rot_idx], table[: rot_idx.size, 0], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            result.rotating_current[rot_idx],
            table[: rot_idx.size, 1] + 1j * table[: rot_idx.size, 2],
            rtol=7e-3,
            atol=7e-4,
        )
        np.testing.assert_allclose(
            result.receiver_tf[tf_idx],
            table[: tf_idx.size, 4] + 1j * table[: tf_idx.size, 5],
            rtol=2e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            result.receiver_tf_signal[tf_idx],
            table[: tf_idx.size, 6] + 1j * table[: tf_idx.size, 7],
            rtol=2e-6,
            atol=1e-6,
        )
    def test_create_wurst_pulse_returns_symmetric_envelope_and_chirp(self) -> None:
        pulse = create_wurst_pulse(
            duration_seconds=200e-6,
            sweep_width_rad_per_s=2 * np.pi * 100e3,
            num_steps=33,
            order=20,
        )

        self.assertEqual(pulse.duration.shape, (33,))
        self.assertAlmostEqual(float(np.sum(pulse.duration)), 200e-6)
        self.assertAlmostEqual(float(pulse.amplitude[0]), 0.0)
        self.assertAlmostEqual(float(pulse.amplitude[-1]), 0.0)
        self.assertAlmostEqual(float(np.max(pulse.amplitude)), 1.0)
        np.testing.assert_allclose(pulse.amplitude, pulse.amplitude[::-1])
        self.assertLess(pulse.frequency_offset[0], 0.0)
        self.assertGreater(pulse.frequency_offset[-1], 0.0)
        self.assertTrue(np.all(np.diff(pulse.frequency_offset) > 0))
        self.assertTrue(np.all(np.isfinite(pulse.phase)))
    def test_matched_wurst_pulse_response_returns_finite_current(self) -> None:
        pulse = create_wurst_pulse(
            duration_seconds=80e-6,
            sweep_width_rad_per_s=2 * np.pi * 50e3,
            num_steps=16,
        )
        result = matched_wurst_pulse_response(pulse, numpts=17)

        self.assertEqual(result.probe, "matched_wurst")
        self.assertGreater(result.rotating_time.size, pulse.duration.size)
        self.assertEqual(result.rotating_current.shape, result.rotating_time.shape)
        self.assertEqual(result.receiver_tf.shape, (17,))
        self.assertEqual(result.receiver_tf_signal.shape, (17,))
        self.assertTrue(np.all(np.isfinite(result.rotating_current)))
    def test_ideal_wurst_inversion_has_central_inversion_band(self) -> None:
        result = run_ideal_wurst_inversion(numpts=41, maxoffs=10, num_steps=128)
        center = result.mz[result.mz.size // 2]

        self.assertEqual(result.probe, "ideal")
        self.assertLess(center, -0.5)
        self.assertGreater(result.mz[0], 0.5)
        self.assertGreater(result.mz[-1], 0.5)
        self.assertTrue(np.all(np.isfinite(result.transverse)))
    def test_matched_wurst_inversion_returns_finite_maps(self) -> None:
        result = run_matched_wurst_inversion(
            numpts=17,
            num_steps=16,
            duration_seconds=80e-6,
            sweep_width_normalized=8,
        )

        self.assertEqual(result.probe, "matched")
        self.assertEqual(result.mz.shape, (17,))
        self.assertGreater(result.rotating_current.size, 0)
        self.assertEqual(result.receiver_tf_signal.shape, (17,))
        self.assertTrue(np.all(np.isfinite(result.mz)))
    def test_matched_wurst_cpmg_returns_expected_shapes(self) -> None:
        result = run_matched_wurst_cpmg(
            num_echoes=1,
            numpts=17,
            num_steps=16,
            duration_seconds=80e-6,
            sweep_width_normalized=8,
            rephase_action="ignore",
        )

        self.assertEqual(result.probe, "matched_wurst")
        self.assertEqual(result.mrx.shape, (1, 17))
        self.assertEqual(result.echo.shape[0], 1)
        self.assertEqual(result.echo.shape[1], result.tvect.size)
        self.assertEqual(result.echo_integrals.shape, (1,))
        self.assertGreater(result.rotating_current.size, 0)
        self.assertTrue(np.all(np.isfinite(result.echo_integrals)))
