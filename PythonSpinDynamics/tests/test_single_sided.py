from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.magnetostatics import GAMMA_PROTON, nmr_mouse_magnets
from spin_dynamics.motion import (
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence
from spin_dynamics.workflows.single_sided import (
    LayeredSample,
    SampleLayer,
    measure_diffusion_at_depth,
    mouse_depth_profile,
    resonant_depth,
)

GAMMA = GAMMA_PROTON


class WalkerEngineTrustTest(unittest.TestCase):
    """The foundation: the moving-walker engine must reproduce the exact
    constant-gradient Carr-Purcell law. If this holds, the real-field MOUSE
    simulation (same engine, spatially varying B0) is trustworthy."""

    def test_uniform_gradient_diffusion_matches_carr_purcell(self):
        G, D = 3.0, 2.3e-9          # MOUSE-scale gradient
        tE, N = 1.5e-4, 12
        x_axis = np.linspace(-2e-4, 2e-4, 3)
        y_axis = np.linspace(-2e-3, 2e-3, 121)
        zero = np.zeros((x_axis.size, y_axis.size))
        fields = make_motion_field_maps_2d(x_axis, y_axis, b0_map=zero,
                                           b1_tx_map=zero + 1, b1_rx_map=zero + 1)
        trains = []
        for seed in range(3):
            ens = initialize_ensemble_from_density(zero + 1, x_axis, y_axis,
                                                   walkers_per_cell=40,
                                                   diffusion_coefficient=D,
                                                   seed=seed, jitter=True)
            res = run_motion_cpmg_sequence(
                ens, fields, num_echoes=N, echo_spacing=tE,
                excitation_duration=8e-6, refocusing_duration=8e-6,
                gradient=(0.0, GAMMA * G), t2=1e9, boundary="reflect",
                substeps_per_interval=4, rng=np.random.default_rng(seed),
            )
            a = np.abs(res.signal)
            trains.append(a / a[0])
        sim = np.mean(trains, axis=0)
        n = np.arange(1, N + 1)
        rate = (1.0 / 12.0) * GAMMA**2 * G**2 * D * tE**2
        analytic = np.exp(-rate * tE * (n - 1))
        sim_rate = -np.polyfit(tE * n, np.log(sim), 1)[0]
        ana_rate = -np.polyfit(tE * n, np.log(analytic), 1)[0]
        self.assertTrue(0.8 < sim_rate / ana_rate < 1.2,
                        f"walker diffusion rate off: {sim_rate/ana_rate:.2f}")


class LayeredSampleTests(unittest.TestCase):
    def test_properties_lookup(self):
        s = LayeredSample([
            SampleLayer(0.02, 0.03, rho=1.0, t2=0.05, diffusion=2e-9),
            SampleLayer(0.03, 0.04, rho=0.0, t2=0.05, diffusion=0.0),
        ])
        p = s.properties(np.array([0.025, 0.035, 0.05]))
        np.testing.assert_allclose(p["rho"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(p["diffusion"], [2e-9, 0.0, 0.0])


class MouseMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.bars, self.yoke = nmr_mouse_magnets(
            magnet_width=0.02, magnet_height=0.02, gap=0.012, remanence=1.30
        )

    def test_resonant_depth_decreases_with_frequency(self):
        d_low = resonant_depth(self.bars, 8e6, yoke_y=self.yoke)
        d_high = resonant_depth(self.bars, 16e6, yoke_y=self.yoke)
        self.assertLess(d_high, d_low)  # higher frequency -> shallower (stronger B0)

    def test_depth_profile_detects_a_gap(self):
        sample = LayeredSample([
            SampleLayer(0.022, 0.030, rho=1.0, t2=0.05, diffusion=0.0),
            SampleLayer(0.030, 0.034, rho=0.0, t2=0.05, diffusion=0.0),  # gap
            SampleLayer(0.034, 0.044, rho=1.0, t2=0.05, diffusion=0.0),
        ])
        from spin_dynamics.fields.magnetostatics import bar_array_b0
        ys = np.array([0.026, 0.032, 0.039])  # material, gap, material
        fr = GAMMA * np.hypot(*bar_array_b0(np.zeros_like(ys), ys, self.bars,
                                            yoke_y=self.yoke)) / (2 * np.pi)
        prof = mouse_depth_profile(
            self.bars, sample, fr, yoke_y=self.yoke, echo_time=2e-4, num_echoes=6,
            depth_halfwidth=0.4e-3, n_depth=41, walkers_per_cell=4,
            substeps_per_interval=2, seed=0,
        )
        # signal present in the two material layers, ~zero in the gap.
        self.assertGreater(prof.signal[0], 1.0)
        self.assertGreater(prof.signal[2], 1.0)
        self.assertLess(prof.signal[1], 0.01 * prof.signal[0])

    def test_diffusion_measurement_recovers_order_of_magnitude(self):
        sample = LayeredSample([
            SampleLayer(0.020, 0.050, rho=1.0, t2=0.08, diffusion=2.3e-9),
        ])
        from spin_dynamics.fields.magnetostatics import bar_array_b0
        f = GAMMA * np.hypot(*bar_array_b0([0.0], [0.026], self.bars,
                                           yoke_y=self.yoke))[0] / (2 * np.pi)
        r = measure_diffusion_at_depth(
            self.bars, sample, float(f), echo_time=1.3e-4, num_echoes=24,
            n_seeds=3, depth_halfwidth=0.7e-3, n_depth=81, walkers_per_cell=12,
            substeps_per_interval=4,
        )
        # stochastic: require the right order of magnitude and a sane gradient.
        self.assertGreater(r.local_gradient, 10.0)
        self.assertTrue(1.0e-9 < r.diffusion < 4.0e-9,
                        f"D out of range: {r.diffusion:.2e}")


if __name__ == "__main__":
    unittest.main()
