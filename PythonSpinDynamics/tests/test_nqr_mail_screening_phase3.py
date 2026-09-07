"""Phase 3 physical invariants, protected cancellation and receiver rejection."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "studies/nqr_mail_screening"
sys.path[:0] = [str(ROOT / "phase1"), str(ROOT / "phase3")]
from pulsed_study import Config, simulate  # noqa: E402
from acquisition import Receiver, receiver, motion_timeline, disorder_grid_points  # noqa: E402
from environment import cancellation_trial  # noqa: E402
from spin_dynamics.interference.cancellers import adaptive_lms_canceller  # noqa: E402

sys.path.remove(str(ROOT / "phase1"))
sys.path.remove(str(ROOT / "phase3"))


class Phase3Tests(unittest.TestCase):
    def config(self, **kwargs):
        return replace(
            Config(echoes=2, powder_theta=2, powder_phi=4, offset_points=9), **kwargs
        )

    def test_fid_rf_events_and_blanking(self):
        cfg = self.config(sequence="FID")
        row, t, v, valid, d = simulate(cfg, return_details=True)
        self.assertEqual(d["rf_events_s"].shape, (1, 2))
        self.assertAlmostEqual(
            row["rf_train_s"], cfg.excitation_s + cfg.echoes * cfg.spacing_s
        )
        self.assertTrue(np.all(t[valid] > cfg.excitation_s + cfg.blank_s))
        self.assertGreater(np.linalg.norm(v), 0)

    def test_carrier_handoff_retains_state_and_recovery_relaxes_it(self):
        cfg = self.config(sequence="SORC")
        *_, old = simulate(cfg, return_details=True)
        next_cfg = replace(cfg, line_id="y", current_peak_a=0)
        *_, new = simulate(next_cfg, return_details=True, initial_state=old["history"])
        np.testing.assert_array_equal(new["start_state"], old["history"]["state"])
        *_, recovered = simulate(
            next_cfg, return_details=True, initial_state=old["history"], recovery_s=0.3
        )
        before = np.linalg.norm(
            new["start_state"][:, :9] - new["equilibrium_state"][:, :9]
        )
        after = np.linalg.norm(
            recovered["start_state"][:, :9] - recovered["equilibrium_state"][:, :9]
        )
        self.assertLess(after, before * np.exp(-0.3 / 0.042) * 1.01)
        with self.assertRaises(ValueError):
            simulate(replace(next_cfg, temperature_k=300), initial_state=old["history"])

    def test_vector_powder_reduces_to_axial_model(self):
        cfg = self.config()

        def scalar(p):
            return np.full(len(p), 1.2e-5)

        def vector(p):
            return np.tile([0, 0, 1.2e-5], (len(p), 1))

        _, t, a, mask = simulate(cfg, field_profile=scalar)
        _, tv, b, mv = simulate(cfg, field_profile=vector)
        np.testing.assert_array_equal(t, tv)
        np.testing.assert_array_equal(mask, mv)
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-25)

    def test_overload_extends_invalid_window(self):
        cfg = Receiver()
        x = np.zeros(1000, complex)
        x[20:30] = 1
        _, valid, margin = receiver(x, np.ones(1000, bool), cfg)
        self.assertGreater(margin["adc_clipped_samples"], 0)
        self.assertFalse(valid[30])
        self.assertTrue(valid[-1])
        self.assertLess(margin["receiver_rail_margin_v"], 0)

    def test_protected_adaptation_and_leaky_reference_are_distinct(self):
        t = np.arange(500) / 10000
        refs = np.exp(2j * np.pi * 100 * t)[None, :]
        y = 0.7 * refs[0]
        fit = np.arange(500) < 200
        injection = np.where(fit, 0, 1e-3 * np.exp(2j * np.pi * 700 * t))
        a = adaptive_lms_canceller(y, refs, fit).cleaned
        b = adaptive_lms_canceller(y + injection, refs, fit).cleaned
        np.testing.assert_allclose(b - a, injection, atol=1e-15)
        leaked = adaptive_lms_canceller(
            y + injection, refs + injection[None, :], fit
        ).cleaned
        self.assertGreater(
            np.linalg.norm(leaked - a - injection), 0.5 * np.linalg.norm(injection)
        )

    def test_unused_reference_overload_does_not_reject_passive_record(self):
        t = np.arange(4000) / 200000
        signal = np.where(t < 0.005, 0.0, 1e-10).astype(complex)
        record = {
            "time_s": t,
            "signal_v": signal,
            "receive_mask": t >= 0.005,
            "fit_mask": t < 0.004,
            "tx_leakage_before_isolation_v": np.zeros_like(t),
            "candidate": "axial_gradiometer",
            "temperature_k": 293.15,
            "blocks": [{"loaded_equivalent_resistance_ohm": 7.0}],
        }
        passive, _ = cancellation_trial(record, "overload", "none")
        active, _ = cancellation_trial(record, "overload", "multi_reference")
        self.assertGreater(passive["reference_clipped_samples"], 0)
        self.assertTrue(passive["receiver_feasible"])
        self.assertFalse(active["receiver_feasible"])

    def test_stopped_acquisition_has_constant_position_and_coupling(self):
        cfg = self.config(
            motion_mode="stop_transfer_stop", preparation_s=0.7, coil_dwell_s=0.03
        )
        row, _, _, _, details = simulate(
            cfg, return_details=True, post_acquisition_s=0.002
        )
        self.assertAlmostEqual(row["record_duration_s"] - row["rf_train_s"], 0.002)
        self.assertEqual(
            row["readout_start_z_relative_coil_m"], row["readout_end_z_relative_coil_m"]
        )
        np.testing.assert_allclose(
            details["receive_beta_t_per_a"], row["beta_t_per_a"], rtol=1e-14
        )
        timeline = motion_timeline(cfg, 0.01, 0.015)
        self.assertAlmostEqual(timeline["cycle_s"], row["cycle_s"])
        for stage in timeline["stages"]:
            if stage["stage"] in ("magnet_dwell", "coil_dwell"):
                self.assertEqual(stage["start_z_m"], stage["end_z_m"])
        with self.assertRaises(ValueError):
            simulate(replace(cfg, coil_dwell_s=0.0001))

    def test_dwell_builds_polarization_then_relaxes_at_coil_stop(self):
        from pulsed_study import prepared_density

        cfg = self.config(
            motion_mode="stop_transfer_stop",
            prepolarization=True,
            preparation_s=0.0,
            settling_s=0.0,
        )
        _, short = prepared_density(cfg)
        _, long = prepared_density(replace(cfg, preparation_s=1.0))
        _, relaxed = prepared_density(replace(cfg, preparation_s=1.0, settling_s=0.02))
        self.assertGreater(
            long["after_magnet_dwell_proton_polarization_fraction"],
            short["after_magnet_dwell_proton_polarization_fraction"],
        )
        stop = (
            cfg.magnet_coil_spacing_m
            + (cfg.readout_start_fraction - 0.5) * cfg.coil_length_m
        )
        self.assertAlmostEqual(long["transfer_stop_z_relative_magnet_m"], stop)
        self.assertAlmostEqual(relaxed["transfer_stop_z_relative_magnet_m"], stop)
        self.assertAlmostEqual(
            long["transfer_time_s"], stop / cfg.transport_velocity_m_s
        )
        self.assertAlmostEqual(
            relaxed["enhancement"] - 1,
            (long["enhancement"] - 1) * np.exp(-0.02 / 0.039),
            places=12,
        )

    def test_disorder_grid_has_no_spurious_revival_during_dwell(self):
        sigma = 360.0
        for duration in (0.015, 0.070):
            offsets = np.linspace(
                -4 * sigma, 4 * sigma, disorder_grid_points(sigma, duration)
            )
            weights = np.exp(-0.5 * (offsets / sigma) ** 2)
            weights /= weights.sum()
            t = np.linspace(0, duration, 1001)
            discrete = weights @ np.exp(2j * np.pi * offsets[:, None] * t)
            analytic = np.exp(-2 * np.pi**2 * sigma**2 * t**2)
            # The normalized +/-4 sigma truncation alone permits twice the
            # omitted Gaussian tail probability (~1.27e-4).
            self.assertLess(np.max(abs(discrete - analytic)), 1.3e-4)
