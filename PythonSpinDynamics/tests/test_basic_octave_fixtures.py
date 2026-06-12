from __future__ import annotations

import unittest
from pathlib import Path
import sys
from dataclasses import replace

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
    calc_rotation_matrix,
    calc_rot_axis_arba3,
    calc_rot_axis_arba4,
    sim_spin_dynamics_asymp_mag3,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_ideal_fid,
    set_params_matched_jmr,
    set_params_matched_orig,
    set_params_tuned_jmr,
    set_params_tuned_orig,
    set_params_tuned_spa,
    set_params_untuned_jmr,
    set_params_untuned_orig,
    set_params_untuned_spa,
)
from spin_dynamics.optimization import (
    evaluate_tuned_refocusing_pulse,
    evaluate_untuned_refocusing_pulse,
    evaluate_spa_metrics,
    rectangular_refocusing_lengths,
    spa_pulse_list,
)
from spin_dynamics.pulses import (
    adjust_untuned_segment_lengths,
    matched_rectangular_pulse_response,
    quantize_phase,
    tuned_rectangular_pulse_response,
    untuned_rectangular_pulse_response,
)
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
    run_ideal_cpmg,
    run_ideal_cpmg_imaging,
    run_ideal_cpmg_train,
    run_ideal_time_varying_amplitude_sweep,
    run_ideal_time_varying_cpmg_final,
    run_matched_cpmg,
    run_matched_cpmg_imaging,
    run_matched_cpmg_ir_train,
    run_matched_cpmg_train,
    run_matched_diffusion_cpmg,
    run_matched_diffusion_q_sweep,
    run_matched_finite_mistuning_sweep,
    run_matched_finite_q_sweep,
    run_matched_mistuning_sweep,
    run_matched_q_sweep,
    run_matched_z_magnetization_q_sweep,
    run_tuned_cpmg,
    run_tuned_cpmg_imaging,
    run_tuned_cpmg_train,
    run_tuned_finite_mistuning_sweep,
    run_tuned_finite_q_sweep,
    run_tuned_mistuning_sweep,
    run_tuned_q_sweep,
    run_untuned_cpmg,
    run_untuned_cpmg_train,
    run_untuned_finite_mistuning_sweep,
    run_untuned_finite_q_sweep,
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
            self.assertFalse(check_matched_diffusion_q_stability(200))
        with self.assertRaises(RuntimeError):
            check_matched_diffusion_q_stability(200, action="raise")

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
