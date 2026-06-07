from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo, calc_time_domain_echo_arb
from spin_dynamics.core.kernels import sim_spin_dynamics_arb10
from spin_dynamics.core.rotations import (
    calc_rotation_matrix,
    calc_rot_axis_arba3,
    calc_rot_axis_arba4,
    sim_spin_dynamics_asymp_mag3,
)
from spin_dynamics.parameters import set_params_ideal
from spin_dynamics.workflows.cpmg import calc_masy_ideal


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "validation" / "fixtures"


class OctaveFixtureTests(unittest.TestCase):
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
            calc_rotation_matrix(del_w, w_1, np.array([np.pi / 2]), np.array([np.pi / 2]), np.array([1])),
            calc_rotation_matrix(
                del_w,
                w_1,
                np.array([0.4 * np.pi, 0.6 * np.pi]),
                np.array([0, np.pi / 4]),
                np.array([0.8, 1.1]),
            ),
        ]
        params = {
            "tp": np.array([np.pi / 2, 1.2 * np.pi, 0.7 * np.pi, 0.9 * np.pi, 0.5 * np.pi, 1.1 * np.pi]),
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


if __name__ == "__main__":
    unittest.main()
