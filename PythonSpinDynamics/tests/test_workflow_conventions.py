"""Cross-workflow convention regressions (review items M-12, M-14..M-16, M-44)."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.parameters import (  # noqa: E402
    set_params_ideal,
    set_params_matched_orig,
    set_params_tuned_orig,
)
from spin_dynamics.workflows import (  # noqa: E402
    make_imaging_field_maps,
    reconstruct_image_from_kspace,
    run_ideal_cpmg_imaging,
    run_ideal_cpmg_ir_train,
    run_ideal_cpmg_train,
    run_matched_cpmg_imaging,
    run_matched_cpmg_train,
    run_matched_diffusion_cpmg,
    run_tuned_cpmg_imaging,
    run_tuned_diffusion_cpmg,
)
from spin_dynamics.workflows.cpmg import _calc_tuned_pulse_shape  # noqa: E402


class EchoTimeBookkeepingTests(unittest.TestCase):
    """M-15: echo-centre times measured from the excitation's time origin.

    The reference is the documented pulse-program geometry: every refocusing
    block [pre, 180, post] ends on its echo, so echo n of a train sits at
    (n + 1) block lengths after the excitation, plus any encoding block.
    """

    def test_ideal_and_probe_trains_report_full_echo_periods(self) -> None:
        _sp, pp = set_params_ideal(numpts=3)
        period = pp.preDelay + pp.T_180 + pp.postDelay
        ideal = run_ideal_cpmg_train(numpts=11, num_echoes=3, rephase_action="ignore")
        # Was (n + 0.5) * period = [100, 300, 500] us.
        np.testing.assert_allclose(ideal.sequence_time, period * np.array([1, 2, 3]))

        _msp, mpp = set_params_matched_orig(numpts=3)
        matched = run_matched_cpmg_train(numpts=11, num_echoes=2, rephase_action="ignore")
        np.testing.assert_allclose(
            matched.sequence_time,
            (mpp.preDelay + mpp.T_180 + mpp.postDelay) * np.array([1, 2]),
        )

    def test_cpmg_and_cpmg_ir_twins_share_the_echo_time_convention(self) -> None:
        _sp, pp = set_params_ideal(numpts=3)
        period = pp.preDelay + pp.T_180 + pp.postDelay
        cpmg = run_ideal_cpmg_train(numpts=11, num_echoes=3, rephase_action="ignore")
        ir = run_ideal_cpmg_ir_train(
            num_echoes=3, echo_spacing_seconds=period, tauvect=[1e-3],
            numpts=11, rephase_action="ignore",
        )
        np.testing.assert_allclose(ir.sequence_time, cpmg.sequence_time)

    def test_phase_encoded_imaging_counts_the_encoding_block(self) -> None:
        gradient, spacing = 0.5e-3, 0.2e-3
        _sp, pp = set_params_ideal(numpts=3)
        result = run_ideal_cpmg_imaging(
            np.ones((3, 3)), num_echoes=2, echo_spacing_seconds=spacing,
            gradient_duration_seconds=gradient, ny=1,
        )
        # [gradient, 180, gradient] encoding spin echo, then the CPMG train.
        offset = 2 * gradient + 2 * pp.T_90
        np.testing.assert_allclose(
            result.sequence_time, offset + spacing * np.array([1, 2])
        )

    def test_diffusion_trains_count_the_encoding_block(self) -> None:
        delta, spacing, t90 = 1.0e-3, 1.0e-3, 100e-6
        for runner in (run_matched_diffusion_cpmg, run_tuned_diffusion_cpmg):
            result = runner(
                num_echoes=2, echo_spacing_seconds=spacing, diffusion_time=delta,
                t90_seconds=t90, numpts=11, rephase_action="ignore",
            )
            # Encoding block [gap, 180, gap] with gap = delta - T90/2 - T180/2.
            np.testing.assert_allclose(
                result.sequence_time, 2 * delta - t90 + spacing * np.array([1, 2])
            )


class PhaseCycleNormalizationTests(unittest.TestCase):
    def test_matched_diffusion_combine_respects_the_magnetization_bound(self) -> None:
        # M-16: without the receiver, mrx is magnetization, so |mrx| <= M0 = 1.
        # The matched twin used the unnormalized PAP combine (max 1.9995).
        kw = dict(num_echoes=2, numpts=41, rephase_action="ignore",
                  diffusion_coefficient=0.0, t1_seconds=10.0, t2_seconds=10.0,
                  apply_receiver=False)
        for runner in (run_matched_diffusion_cpmg, run_tuned_diffusion_cpmg):
            peak = float(np.max(np.abs(runner(**kw).mrx)))
            self.assertLessEqual(peak, 1.0 + 1e-9)
        matched_peak = float(np.max(np.abs(run_matched_diffusion_cpmg(**kw).mrx)))
        self.assertGreater(matched_peak, 0.9)

    def test_cpmg_ir_cpmg_convention_recovers_the_cpmg_train(self) -> None:
        # M-14: after full recovery (tau >> T1, T2) an IR train is exactly the
        # CPMG train with the same timing. Reference: run_ideal_cpmg_train.
        _sp, pp = set_params_ideal(numpts=3)
        period = pp.preDelay + pp.T_180 + pp.postDelay
        kw = dict(num_echoes=3, echo_spacing_seconds=period, tauvect=[0.2],
                  t1_seconds=5e-3, t2_seconds=5e-3, numpts=21, rephase_action="ignore")
        cpmg = run_ideal_cpmg_train(
            numpts=21, num_echoes=3, t1_seconds=5e-3, t2_seconds=5e-3,
            rephase_action="ignore",
        )
        aligned = run_ideal_cpmg_ir_train(signal_convention="cpmg", **kw)
        np.testing.assert_allclose(
            aligned.echo_integrals[0], cpmg.echo_integrals, rtol=1e-9, atol=1e-12
        )
        # The default keeps the MATLAB relax4 combine: -2x the CPMG convention.
        legacy = run_ideal_cpmg_ir_train(**kw)
        np.testing.assert_allclose(legacy.mrx, -2.0 * aligned.mrx, rtol=1e-12, atol=1e-15)
        with self.assertRaises(ValueError):
            run_ideal_cpmg_ir_train(signal_convention="bogus", **kw)


class TunedPulseNormalizationTests(unittest.TestCase):
    def test_long_tuned_pulse_plateau_is_unit_for_any_gate_on_phase(self) -> None:
        # M-44: shapes were peak-normalized, and the gate-on overshoot depends
        # on the absolute RF phase, so the nominal amplitude moved with phase
        # (plateau 0.58 at psi = pi/2). Reference: the steady state actually
        # reached by a long simulated pulse must map to amplitude 1.
        _params, sp0, pp = set_params_tuned_orig(numpts=9)
        sp0 = replace(sp0, R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q)
        sp = {**sp0.__dict__, "plt_tx": 0, "plt_rx": 0}
        duration = 20 * pp.T_180
        plateaus = []
        for psi in (0.0, np.pi / 4, np.pi / 2):
            tp, _phi, amp = _calc_tuned_pulse_shape(
                sp, pp, duration, 0.0, 1.0, 2 * pp.T_90, psi=psi
            )
            seconds = np.cumsum(tp[:-1]) * pp.T_90 / (np.pi / 2)
            window = (seconds > 0.8 * duration) & (seconds < 0.95 * duration)
            plateaus.append(float(np.mean(amp[:-1][window])))
            # The overshoot itself is phase dependent; it is not the reference.
        np.testing.assert_allclose(plateaus, 1.0, atol=2e-3)


class PhaseEncodedGridTests(unittest.TestCase):
    def test_even_matrix_point_reconstructs_to_one_pixel(self) -> None:
        # M-12: for even sizes the encode grid missed k=0 and the voxel grid
        # had a half-integer centre, so a point spread over a 2x2 block (peak
        # 0.125 of the image sum). Reference: with the FOV matched to the DFT
        # (fov = n**3 / (n - 1)**2), a point voxel must reconstruct to a delta.
        for n, index in ((4, (1, 2)), (5, (1, 3))):
            fov = n**3 / (n - 1) ** 2
            rho = np.zeros((n, n))
            rho[index] = 1.0
            result = run_ideal_cpmg_imaging(
                rho, fov=(fov, fov), num_echoes=1,
                t1_map=np.ones((n, n)), t2_map=np.ones((n, n)),
            )
            image = np.abs(reconstruct_image_from_kspace(result.kspace, 0))
            self.assertEqual(np.unravel_index(int(np.argmax(image)), image.shape), index)
            self.assertGreater(image.max() / image.sum(), 0.999)


class ImagingCombineConventionTests(unittest.TestCase):
    def test_cpmg_convention_matches_the_refocused_magnetization(self) -> None:
        # L-54: phase-encoded imaging keeps the MATLAB unnormalized PAP combine
        # by default (2x the CPMG trains); signal_convention="cpmg" normalizes
        # it. Reference: an on-resonance, non-relaxing unit voxel is fully
        # refocused, so its k=0 echo integral is the normalized acquisition
        # window length (pi/2) * tacq / T90.
        rho = np.zeros((3, 3))
        rho[1, 1] = 1.0
        ones = np.ones_like(rho)
        fields = make_imaging_field_maps(
            rho, t1_map=1e3 * ones, t2_map=1e3 * ones, b1_tx_map=ones, b1_rx_map=ones
        )
        spacing = 0.2e-3
        _sp, pp = set_params_ideal(numpts=1)
        tacq = min(float(np.ravel(pp.tacq)[0]), spacing - 2 * pp.T_90)
        window = (np.pi / 2) * tacq / pp.T_90
        kw = dict(num_echoes=1, ny=1, maxoffs=1e-3, echo_spacing_seconds=spacing)
        normalized = run_ideal_cpmg_imaging(fields, signal_convention="cpmg", **kw)
        legacy = run_ideal_cpmg_imaging(fields, **kw)
        self.assertAlmostEqual(abs(normalized.kspace[1, 1, 0]) / window, 1.0, delta=1e-4)
        np.testing.assert_allclose(legacy.kspace, 2.0 * normalized.kspace, rtol=1e-12)
        # Matched halves; tuned acquires one branch (already normalized).
        matched = run_matched_cpmg_imaging(fields, num_echoes=1, ny=3)
        matched_cpmg = run_matched_cpmg_imaging(
            fields, num_echoes=1, ny=3, signal_convention="cpmg"
        )
        np.testing.assert_allclose(matched_cpmg.kspace, 0.5 * matched.kspace, rtol=1e-12)
        tuned = run_tuned_cpmg_imaging(fields, num_echoes=1, ny=3)
        tuned_cpmg = run_tuned_cpmg_imaging(fields, num_echoes=1, ny=3, signal_convention="cpmg")
        np.testing.assert_array_equal(tuned_cpmg.kspace, tuned.kspace)
        with self.assertRaises(ValueError):
            run_ideal_cpmg_imaging(fields, signal_convention="bogus", **kw)


if __name__ == "__main__":
    unittest.main()
