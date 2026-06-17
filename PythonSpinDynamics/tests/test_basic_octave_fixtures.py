from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from dataclasses import replace
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core.echo import calc_time_domain_echo, calc_time_domain_echo_arb
from spin_dynamics.core.isochromats import (
    analyze_rephasing,
    check_rephasing,
    recommended_numpts_for_rephasing,
)
from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_chunked,
    sim_spin_dynamics_arb10_diffusion,
    sim_spin_dynamics_arb10_diffusion_chunked,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    calc_v0crit,
    calc_rotation_matrix,
    calc_rot_axis_arba3,
    calc_rot_axis_arba4,
    sim_spin_dynamics_asymp_mag3,
    sim_spin_dynamics_exc,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_ideal_fid,
    set_params_matched_jmr,
    set_params_matched_orig,
    set_params_matched_spa,
    set_params_tuned_jmr,
    set_params_tuned_orig,
    set_params_tuned_spa,
    set_params_untuned_jmr,
    set_params_untuned_orig,
    set_params_untuned_spa,
)
from spin_dynamics.optimization import (
    analyze_matlab_optimization_results,
    analyze_optimization_result_file,
    analyze_tuned_inverse_result_files,
    analyze_tuned_inverse_result_pair,
    evaluate_ideal_time_varying_refocusing_pulse,
    evaluate_ideal_v0crit_excited_refocusing_pulse,
    evaluate_ideal_v0crit_refocusing_pulse,
    evaluate_tuned_excitation_pulse,
    evaluate_tuned_inverse_excitation_pulse,
    evaluate_matched_refocusing_pulse,
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
    evaluate_spa_metrics,
    ideal_time_varying_excitation_vector,
    optimize_ideal_time_varying_refocusing_phases,
    optimize_ideal_v0crit_excited_refocusing_phases,
    optimize_ideal_v0crit_refocusing_phases,
    optimize_matched_refocusing_phases,
    optimize_spa_phase_program,
    optimize_tuned_excitation_phases,
    optimize_tuned_inverse_excitation_phases,
    optimize_tuned_refocusing_phases,
    optimize_untuned_refocusing_phases,
    random_phase_starts,
    rectangular_refocusing_lengths,
    get_matlab_result_layout,
    load_optimization_results,
    load_matlab_results_mat,
    load_multistart_results_npz,
    matlab_result_layouts,
    multistart_summary_arrays,
    multistart_to_matlab_results,
    run_ideal_time_varying_refocusing_multistart,
    run_ideal_v0crit_excited_refocusing_multistart,
    run_ideal_v0crit_refocusing_multistart,
    run_tuned_excitation_inverse_pipeline,
    run_tuned_excitation_multistart,
    run_tuned_inverse_excitation_multistart,
    run_tuned_refocusing_multistart,
    save_multistart_results_mat,
    save_multistart_results_npz,
    select_matlab_result_program,
    spa_pulse_list,
    summarize_matlab_results,
    summarize_matched_spa_refocusing,
    summarize_tuned_spa_refocusing,
    summarize_untuned_spa_refocusing,
)
from spin_dynamics.pulses import (
    adjust_untuned_segment_lengths,
    matched_rectangular_pulse_response,
    quantize_phase,
    tuned_rectangular_pulse_response,
    untuned_rectangular_pulse_response,
)
import spin_dynamics.optimization.drivers as driver_module
import spin_dynamics.optimization.excitation as excitation_module
import spin_dynamics.optimization.pipeline as pipeline_module
import spin_dynamics.optimization.refocusing as refocusing_module
import spin_dynamics.workflows.imaging as imaging_module
from spin_dynamics.probes.tuned import (
    calc_masy_tuned_probe_lp_orig,
    tuned_probe_lp_orig,
)
from spin_dynamics.probes.untuned import (
    calc_masy_untuned_probe_lp,
    untuned_probe_lp,
)
from spin_dynamics.probes.matched import (
    calc_masy_matched_probe_orig,
    find_coil_current,
    matching_network_design2,
)
from spin_dynamics.workflows.cpmg import calc_masy_ideal
from spin_dynamics.workflows import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
    fit_imaging_echo_decay,
    form_imaging_image,
    load_imaging_field_maps_npz,
    make_imaging_field_maps,
    run_ideal_cpmg,
    run_ideal_cpmg_imaging,
    run_ideal_cpmg_ir_train,
    run_ideal_phase_encoded_cpmg_imaging,
    run_ideal_cpmg_train,
    run_ideal_time_varying_amplitude_sweep,
    run_ideal_time_varying_cpmg_final,
    run_t1_encoded_phase_encoded_cpmg_imaging,
    run_matched_cpmg,
    run_matched_cpmg_imaging,
    run_matched_cpmg_ir_train,
    run_matched_phase_encoded_cpmg_imaging,
    run_matched_cpmg_train,
    run_matched_diffusion_cpmg,
    run_matched_diffusion_q_sweep,
    run_matched_finite_mistuning_sweep,
    run_matched_finite_q_sweep,
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_matched_time_varying_amplitude_sweep,
    run_matched_time_varying_cpmg_final,
    run_matched_z_magnetization_q_sweep,
    run_tuned_cpmg,
    run_tuned_cpmg_imaging,
    run_tuned_cpmg_ir_train,
    run_tuned_phase_encoded_cpmg_imaging,
    run_tuned_cpmg_train,
    run_tuned_finite_mistuning_sweep,
    run_tuned_finite_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
    run_tuned_time_varying_amplitude_sweep,
    run_tuned_time_varying_cpmg_final,
    run_untuned_cpmg,
    run_untuned_cpmg_ir_train,
    run_untuned_cpmg_train,
    run_untuned_finite_mistuning_sweep,
    run_untuned_finite_q_sweep,
    run_untuned_time_varying_amplitude_sweep,
    run_untuned_time_varying_cpmg_final,
    VALIDATED_MATCHED_DIFFUSION_Q_MAX,
    check_matched_diffusion_q_stability,
    sinusoidal_field_waveform,
)
from spin_dynamics.workflows.fid import sim_fid_ideal


FIXTURES = ROOT / "validation" / "fixtures"


class OctaveFixtureTests(unittest.TestCase):
    def test_numpy_compatibility_helpers(self) -> None:
        y = np.array([0.0, 1.0, 0.0])
        x = np.array([0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(trapezoid(y, x)), 0.5)

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

    def _probe_relax4_inputs(
        self,
        numpts: int,
    ) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float | list]]:
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
        return sp, pp

    def _load_probe_relax4_fixture(
        self,
        filename: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        table = np.loadtxt(FIXTURES / filename, delimiter=",")
        acq_count = int(np.max(table[:, 0]))
        numpts = int(np.max(table[:, 1]))
        macq_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        mrx_ref = np.zeros((acq_count, numpts), dtype=np.complex128)
        for row in table:
            idx = (int(row[0]) - 1, int(row[1]) - 1)
            macq_ref[idx] = row[2] + 1j * row[3]
            mrx_ref[idx] = row[4] + 1j * row[5]
        return macq_ref, mrx_ref

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
        )
        self.assertEqual(result.field_offsets.shape, (4,))
        self.assertEqual(result.mrx.shape, (17,))
        self.assertEqual(result.echo.shape, result.tvect.shape)
        self.assertTrue(np.isfinite(result.echo_integral))

    def test_ideal_time_varying_amplitude_sweep_parallel_matches_serial(self) -> None:
        waveform = sinusoidal_field_waveform(4)
        serial = run_ideal_time_varying_amplitude_sweep(
            amplitudes=[0.0, 0.5],
            waveform=waveform,
            numpts=17,
            num_workers=1,
        )
        parallel = run_ideal_time_varying_amplitude_sweep(
            amplitudes=[0.0, 0.5],
            waveform=waveform,
            numpts=17,
            num_workers=2,
        )
        np.testing.assert_allclose(parallel.amplitudes, serial.amplitudes)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.matched_signal,
            serial.matched_signal,
            rtol=1e-13,
            atol=1e-13,
        )

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
                )
                parallel = runner(
                    amplitudes=[0.0, 0.25],
                    waveform=waveform,
                    numpts=9,
                    maxoffs=4,
                    num_workers=2,
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
            "diffusion_time": 1.0,
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
        result = run_matched_diffusion_cpmg(num_echoes=2, numpts=17, q_value=20)
        self.assertEqual(result.mrx.shape, (2, 17))
        self.assertEqual(result.echo.shape[0], 2)
        self.assertEqual(result.echo.shape[1], result.tvect.size)
        self.assertEqual(result.echo_integrals.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.echo_integrals)))

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
        )
        parallel = run_matched_diffusion_q_sweep(
            [20, 50],
            num_echoes=2,
            numpts=17,
            sweep_workers=2,
        )
        np.testing.assert_allclose(parallel.values, serial.values)
        np.testing.assert_allclose(parallel.echo, serial.echo, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            parallel.echo_integrals,
            serial.echo_integrals,
            rtol=1e-13,
            atol=1e-13,
        )

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

    def test_imaging_echo_formation_modes_use_expected_weighting(self) -> None:
        echo_times = np.array([0.1e-3, 0.2e-3, 0.3e-3], dtype=np.float64)
        rho = np.array([[2.0, 4.0], [1.5, 3.0]], dtype=np.float64)
        t2 = np.array([[0.8e-3, 1.2e-3], [1.6e-3, 2.4e-3]], dtype=np.float64)
        magnitude = rho[:, :, np.newaxis] * np.exp(
            -echo_times.reshape(1, 1, -1) / t2[:, :, np.newaxis]
        )
        result = SimpleNamespace(
            image=magnitude.astype(np.complex128),
            magnitude=magnitude,
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

    def test_imaging_echo_decay_fit_reports_maps_and_rejects_bad_inputs(self) -> None:
        echo_times = np.array([0.1e-3, 0.2e-3, 0.3e-3], dtype=np.float64)
        rho = np.array([[2.0]], dtype=np.float64)
        t2 = np.array([[0.8e-3]], dtype=np.float64)
        magnitude = rho[:, :, np.newaxis] * np.exp(
            -echo_times.reshape(1, 1, -1) / t2[:, :, np.newaxis]
        )
        result = SimpleNamespace(
            image=magnitude.astype(np.complex128),
            magnitude=magnitude,
            sequence_time=echo_times,
        )

        fit = fit_imaging_echo_decay(result)

        np.testing.assert_allclose(fit.rho_map, rho, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.t2_map, t2, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.fitted_magnitude, magnitude, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(fit.residual_norm, 0.0, atol=1e-13)
        self.assertTrue(bool(fit.mask[0, 0]))
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

    def test_make_imaging_field_maps_rejects_invalid_maps(self) -> None:
        rho = np.ones((2, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            make_imaging_field_maps(rho, b1_tx_map=-np.ones_like(rho))
        with self.assertRaises(ValueError):
            make_imaging_field_maps(rho, b0_map=np.ones((2, 3)))

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

    def _assert_train_fixture(
        self,
        fixture_stem: str,
        runner,
        numpts_expected: int | None,
        maxoffs: float,
        rtol: float,
        atol: float,
    ) -> None:
        mrx_table = np.loadtxt(FIXTURES / f"{fixture_stem}_mrx.csv", delimiter=",")
        echo_table = np.loadtxt(FIXTURES / f"{fixture_stem}_echo.csv", delimiter=",")
        int_table = np.loadtxt(FIXTURES / f"{fixture_stem}_integrals.csv", delimiter=",")

        num_echoes = int(np.max(mrx_table[:, 0]))
        numpts = int(np.max(mrx_table[:, 1]))
        if numpts_expected is not None:
            self.assertEqual(numpts, numpts_expected)
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

        result = runner(
            numpts=numpts,
            maxoffs=maxoffs,
            num_echoes=num_echoes,
            t1_seconds=1.7,
            t2_seconds=1.1,
            rephase_action="ignore",
        )

        np.testing.assert_allclose(result.mrx, mrx_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(result.echo, echo_ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(result.tvect, tvect_ref, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(
            result.echo_integrals,
            echo_int_ref,
            rtol=rtol,
            atol=atol,
        )
        self.assertEqual(result.probe, fixture_stem.removeprefix("run_").removesuffix("_cpmg_train"))

    def _assert_imaging_fixture(
        self,
        fixture_stem: str,
        runner,
        rtol: float,
        atol: float,
    ) -> None:
        table = np.loadtxt(FIXTURES / f"{fixture_stem}_kspace.csv", delimiter=",")
        px = int(np.max(table[:, 0]))
        pz = int(np.max(table[:, 1]))
        num_echoes = int(np.max(table[:, 2]))
        kspace_ref = np.zeros((px, pz, num_echoes), dtype=np.complex128)
        for row in table:
            kspace_ref[int(row[0]) - 1, int(row[1]) - 1, int(row[2]) - 1] = row[3] + 1j * row[4]

        rho = np.array([[0.0, 1.0], [1.0, 0.35]], dtype=np.float64)
        relaxation = 5e-3 * np.ones_like(rho)
        result = runner(
            rho,
            t1_map=relaxation,
            t2_map=relaxation,
            num_echoes=num_echoes,
            echo_spacing_seconds=0.2e-3,
            gradient_duration_seconds=0.5e-3,
            fov=(20.0, 20.0),
            ny=400,
            num_workers=1,
            phase_workers=1,
        )

        np.testing.assert_allclose(result.kspace, kspace_ref, rtol=rtol, atol=atol)

    def _assert_pulse_response_fixture(
        self,
        fixture_stem: str,
        result,
        rtol: float,
        atol: float,
    ) -> None:
        table = np.loadtxt(FIXTURES / f"{fixture_stem}.csv", delimiter=",")
        rot_idx = self._fixture_sample_indices(result.rotating_time.size)
        raw_idx = self._fixture_sample_indices(result.raw_time.size)
        tf_idx = table[:, 6]
        tf_idx = tf_idx[np.isfinite(tf_idx)].astype(int) - 1

        np.testing.assert_allclose(result.rotating_time[rot_idx], table[: rot_idx.size, 0], rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            result.rotating_current[rot_idx],
            table[: rot_idx.size, 1] + 1j * table[: rot_idx.size, 2],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(result.raw_time[raw_idx], table[: raw_idx.size, 3], rtol=rtol, atol=atol)
        np.testing.assert_allclose(
            result.raw_current[raw_idx],
            table[: raw_idx.size, 4] + 1j * table[: raw_idx.size, 5],
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            result.receiver_tf[tf_idx],
            table[: tf_idx.size, 7] + 1j * table[: tf_idx.size, 8],
            rtol=rtol,
            atol=atol,
        )

    @staticmethod
    def _fixture_sample_indices(size: int) -> np.ndarray:
        return np.unique(np.rint(np.linspace(1, size, min(8, size))).astype(int)) - 1


if __name__ == "__main__":
    unittest.main()
