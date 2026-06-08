from __future__ import annotations

import unittest
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core.echo import calc_time_domain_echo, calc_time_domain_echo_arb
from spin_dynamics.core.kernels import sim_spin_dynamics_arb10
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
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
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
    run_ideal_cpmg_train,
    run_matched_cpmg,
    run_tuned_cpmg,
    run_untuned_cpmg,
)
from spin_dynamics.workflows.fid import sim_fid_ideal


FIXTURES = ROOT / "validation" / "fixtures"


class OctaveFixtureTests(unittest.TestCase):
    def test_numpy_compatibility_helpers(self) -> None:
        y = np.array([0.0, 1.0, 0.0])
        x = np.array([0.0, 0.5, 1.0])
        self.assertAlmostEqual(float(trapezoid(y, x)), 0.5)

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

        np.testing.assert_allclose(macq, macq_ref, rtol=1e-13, atol=1e-13)

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


if __name__ == "__main__":
    unittest.main()
