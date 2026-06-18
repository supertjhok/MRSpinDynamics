from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.radiation_damping import (
    MU0,
    RadiationDampingProbe,
    analytic_radiation_damping_envelope,
    initial_state_from_flip_angle,
    hyperpolarized_proton_sample,
    normalized_radiation_damping_weights,
    proton_thermal_magnetization_density,
    radiation_damping_probe_from_matched,
    radiation_damping_probe_from_tuned,
    radiation_damping_time,
    simulate_nmr_maser,
    simulate_radiation_damping_fid,
    water_proton_sample,
)
from spin_dynamics.workflows import run_radiation_damping_fid, run_tuned_cpmg_train


class RadiationDampingTests(unittest.TestCase):
    def test_radiation_damping_time_matches_textbook_formula(self) -> None:
        gamma = 2.0
        eta = 0.5
        mth = 3.0
        q = 4.0
        expected = 2.0 / (gamma * MU0 * eta * mth * q)
        self.assertAlmostEqual(radiation_damping_time(gamma, eta, mth, q), expected)

    def test_probe_constructors_use_existing_probe_q(self) -> None:
        _params, tuned_sp, _tuned_pp = set_params_tuned_orig(numpts=9)
        matched_sp, _matched_pp = set_params_matched_orig(numpts=9)
        tuned = radiation_damping_probe_from_tuned(
            tuned_sp,
            fill_factor=0.6,
            equilibrium_magnetization=0.01,
        )
        matched = radiation_damping_probe_from_matched(
            matched_sp,
            fill_factor=0.6,
            equilibrium_magnetization=0.01,
        )
        self.assertEqual(tuned.name, "tuned")
        self.assertEqual(matched.name, "matched")
        self.assertAlmostEqual(tuned.q, tuned_sp.Q)
        self.assertAlmostEqual(matched.q, matched_sp.Q)
        self.assertGreater(tuned.trd, 0.0)
        self.assertGreater(matched.resonator_time_constant, 0.0)

    def test_initial_state_from_x_flip_points_along_minus_y(self) -> None:
        mxy, mz = initial_state_from_flip_angle(np.pi / 2)
        np.testing.assert_allclose(mxy, -1j)
        self.assertAlmostEqual(mz, 0.0, places=15)

    def test_instant_fid_matches_analytic_envelope(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1.0,
            omega0=100.0,
            q=10.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 3.0 * probe.trd, 501)
        theta = np.deg2rad(45.0)
        result = simulate_radiation_damping_fid(time, probe, flip_angle=theta)
        expected = analytic_radiation_damping_envelope(time, theta, probe.trd)
        np.testing.assert_allclose(result.envelope, expected, rtol=2e-5, atol=2e-7)
        np.testing.assert_allclose(
            result.envelope**2 + result.mz**2,
            np.ones_like(time),
            rtol=2e-5,
            atol=2e-7,
        )

    def test_circuit_model_lags_instant_feedback_for_high_q_probe(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1e6,
            omega0=1e5,
            q=50.0,
            fill_factor=0.7,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 0.2 * probe.trd, 101)
        circuit = simulate_radiation_damping_fid(time, probe, model="circuit")
        instant = simulate_radiation_damping_fid(time, probe, model="instant")
        self.assertLess(circuit.envelope[-1], circuit.envelope[0])
        self.assertGreater(circuit.envelope[-1], instant.envelope[-1])

    def test_proton_magnetization_density_scales_with_field(self) -> None:
        low = proton_thermal_magnetization_density(1.0)
        high = proton_thermal_magnetization_density(2.0)
        self.assertGreater(low, 0.0)
        self.assertAlmostEqual(high / low, 2.0)

    def test_sample_presets_scale_magnetization(self) -> None:
        water = water_proton_sample(1.0)
        boosted = hyperpolarized_proton_sample(1.0, polarization_scale=100.0)
        self.assertEqual(water.name, "water protons")
        self.assertGreater(water.equilibrium_magnetization, 0.0)
        self.assertAlmostEqual(
            boosted.equilibrium_magnetization / water.equilibrium_magnetization,
            100.0,
        )

    def test_normalized_radiation_damping_weights_use_sensitivity_squared(self) -> None:
        density = np.array([1.0, 1.0, 2.0])
        sensitivity = np.array([1.0, 2.0, 0.5])
        weights = normalized_radiation_damping_weights(density, sensitivity)
        expected = density * np.abs(sensitivity) ** 2
        expected = expected / np.sum(expected)
        np.testing.assert_allclose(weights, expected)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0)

    def test_tuned_cpmg_train_accepts_radiation_damping_mapping(self) -> None:
        clean = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=2,
            rephase_action="ignore",
        )
        damped = run_tuned_cpmg_train(
            numpts=9,
            num_echoes=2,
            rephase_action="ignore",
            radiation_damping={
                "fill_factor": 0.7,
                "equilibrium_magnetization": 0.8,
            },
        )
        self.assertIsNotNone(damped.radiation_damping)
        self.assertEqual(damped.mrx.shape, clean.mrx.shape)
        self.assertGreater(damped.radiation_damping.trd, 0.0)
        self.assertGreater(
            float(np.max(np.abs(damped.mrx - clean.mrx))),
            0.0,
        )

    def test_tuned_cpmg_train_can_apply_radiation_damping_during_pulses(self) -> None:
        common = {
            "numpts": 9,
            "num_echoes": 2,
            "rephase_action": "ignore",
            "radiation_damping": {
                "fill_factor": 0.7,
                "equilibrium_magnetization": 0.8,
            },
        }
        free_only = run_tuned_cpmg_train(**common)
        during_pulses = run_tuned_cpmg_train(
            **{
                **common,
                "radiation_damping": {
                    **common["radiation_damping"],
                    "apply_during_pulses": True,
                },
            }
        )
        self.assertIsNotNone(during_pulses.radiation_damping)
        self.assertTrue(during_pulses.radiation_damping.apply_during_pulses)
        self.assertEqual(during_pulses.mrx.shape, free_only.mrx.shape)
        self.assertGreater(
            float(np.max(np.abs(during_pulses.mrx - free_only.mrx))),
            0.0,
        )

    def test_radiation_damping_fid_workflow_matches_analytic_envelope(self) -> None:
        result = run_radiation_damping_fid(
            probe="matched",
            equilibrium_magnetization=0.8,
            fill_factor=0.7,
            flip_angle=np.deg2rad(60.0),
            num_points=301,
        )
        self.assertEqual(result.time_seconds.shape, result.mxy.shape)
        self.assertGreater(result.probe.trd, 0.0)
        np.testing.assert_allclose(
            result.envelope,
            result.analytic_envelope,
            rtol=3e-5,
            atol=3e-7,
        )

    def test_radiation_damping_fid_workflow_supports_tuned_probe(self) -> None:
        result = run_radiation_damping_fid(
            probe="tuned",
            equilibrium_magnetization=0.8,
            fill_factor=0.7,
            flip_angle=np.deg2rad(45.0),
            num_points=31,
        )
        self.assertEqual(result.probe.name, "tuned")
        self.assertEqual(result.normalized_time.shape, result.time_seconds.shape)
        self.assertGreater(result.envelope[0], result.envelope[-1])

    def test_radiation_damping_fid_workflow_exposes_circuit_detuning(self) -> None:
        common = {
            "probe": "matched",
            "equilibrium_magnetization": 0.8,
            "fill_factor": 0.7,
            "flip_angle": np.deg2rad(60.0),
            "model": "circuit",
            "num_points": 81,
        }
        tuned = run_radiation_damping_fid(**common)
        detuned = run_radiation_damping_fid(
            **{
                **common,
                "phase": 0.25,
                "detuning": 1.0e5,
            }
        )
        self.assertAlmostEqual(detuned.probe.phase, 0.25)
        self.assertAlmostEqual(detuned.probe.detuning, 1.0e5)
        self.assertGreater(
            float(np.max(np.abs(detuned.mxy - tuned.mxy))),
            0.0,
        )

    def test_nmr_maser_threshold_growth(self) -> None:
        probe = RadiationDampingProbe(
            gamma=1.0,
            omega0=100.0,
            q=10.0,
            fill_factor=0.5,
            equilibrium_magnetization=1.0,
        )
        time = np.linspace(0.0, 20.0 * probe.trd, 501)
        t2 = 10.0 * probe.trd
        t1 = 10.0 * probe.trd
        below = simulate_nmr_maser(
            time,
            probe,
            seed_mxy=-1e-5j,
            initial_mz=-0.05,
            pump_mz=-0.05,
            t1=t1,
            t2=t2,
            model="instant",
        )
        above = simulate_nmr_maser(
            time,
            probe,
            seed_mxy=-1e-5j,
            initial_mz=-0.8,
            pump_mz=-0.8,
            t1=t1,
            t2=t2,
            model="instant",
        )
        self.assertLess(below.envelope[-1], below.envelope[0])
        self.assertGreater(above.envelope[-1], 1e3 * above.envelope[0])
        self.assertGreater(above.mz[-1], -0.8)


if __name__ == "__main__":
    unittest.main()
