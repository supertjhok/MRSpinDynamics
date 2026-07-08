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
from spin_dynamics.fields.nonlinear_magnetostatics import linear_material, ndfeb
from spin_dynamics.fields.scalar_potential_3d import ReducedScalarPotential3D
from spin_dynamics.workflows.single_sided import (
    AnalyticMouseField,
    LayeredSample,
    SampleLayer,
    SolvedMouseField,
    measure_diffusion_at_depth,
    mouse_depth_profile,
    resonant_depth,
)

GAMMA = GAMMA_PROTON


def _solved_mouse_field():
    """A coarse 3-D NMR-MOUSE solve (magnets + iron yoke) wrapped for the workflow."""

    mm = 1e-3

    def axis(lo, hi, h=2.5 * mm):
        return np.linspace(lo, hi, int(round((hi - lo) / h)) + 1)

    x, y, z = axis(-50 * mm, 50 * mm), axis(-70 * mm, 50 * mm), axis(-90 * mm, 90 * mm)
    prob = ReducedScalarPotential3D(x, y, z)
    bar_x = (-20 * mm, 20 * mm)
    prob.add_material(prob.box(bar_x, (-32 * mm, 0.0), (-32.5 * mm, -6.5 * mm)),
                      ndfeb(1.2), remanence_direction=(0, 1, 0))
    prob.add_material(prob.box(bar_x, (-32 * mm, 0.0), (6.5 * mm, 32.5 * mm)),
                      ndfeb(1.2), remanence_direction=(0, -1, 0))
    prob.add_material(prob.box(bar_x, (-47 * mm, -32 * mm), (-32.5 * mm, 32.5 * mm)),
                      linear_material(1000.0))
    return SolvedMouseField(prob.solve(), depth_range=(0.5e-3, 12.0e-3))


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


class SolvedFieldMouseTests(unittest.TestCase):
    """The single-sided workflow driven by a full 3-D magnetostatic solve
    (magnets + iron yoke) instead of the analytic bar-magnet model."""

    @classmethod
    def setUpClass(cls):
        cls.field = _solved_mouse_field()

    def test_bars_wrap_into_a_field_source(self):
        # Backward compatibility: passing bars still works (auto-wrapped).
        bars, yoke = nmr_mouse_magnets(gap=0.012, remanence=1.30)
        d = resonant_depth(bars, 12e6, yoke_y=yoke)
        self.assertEqual(
            d, resonant_depth(AnalyticMouseField(bars, yoke), 12e6)
        )

    def test_resonant_depth_decreases_with_frequency(self):
        # The solved-field on-axis profile is monotonic like the analytic one.
        d_lo, f_lo = self.field.larmor_profile(0.5e-3, 12e-3, GAMMA)
        f_shallow, f_deep = np.interp([2.5e-3, 6.0e-3], d_lo, f_lo)
        self.assertLess(resonant_depth(self.field, f_shallow),
                        resonant_depth(self.field, f_deep))

    def test_depth_profile_detects_a_gap_in_the_solved_field(self):
        # Choose carrier frequencies that resonate at bulk / gap / bulk depths of
        # the *solved* field, then confirm the density gap reads near-zero signal.
        depths, larmor = self.field.larmor_profile(0.5e-3, 12e-3, GAMMA)
        targets = np.array([2.5e-3, 4.0e-3, 5.5e-3])  # material, gap, material
        freqs = np.interp(targets, depths, larmor)
        sample = LayeredSample([
            SampleLayer(0.0, 3.2e-3, rho=1.0, t2=0.05),
            SampleLayer(3.2e-3, 4.8e-3, rho=0.0),        # gap
            SampleLayer(4.8e-3, 20.0e-3, rho=1.0, t2=0.05),
        ])
        prof = mouse_depth_profile(
            self.field, sample, freqs, echo_time=2e-4, num_echoes=4,
            depth_halfwidth=0.4e-3, n_depth=21, walkers_per_cell=4,
            substeps_per_interval=2, depth_range=(0.5e-3, 12e-3), seed=0,
        )
        bulk = min(prof.signal[0], prof.signal[2])
        self.assertGreater(bulk, 0.0)
        self.assertLess(prof.signal[1], 0.2 * bulk)  # gap is a hole


if __name__ == "__main__":
    unittest.main()
