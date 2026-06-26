from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.workflows.bipolar import (
    GAMMA,
    ToggleInterval,
    cotts_thirteen_interval_intervals,
    monopolar_pgste_intervals,
    run_cotts_thirteen_interval_moment,
    run_cotts_thirteen_interval_walkers,
    run_monopolar_pgste_moment,
    toggling_frame_moments,
)
from spin_dynamics.workflows import pgse_b_value


def _apparent_d_from_sweep(runner, *, background_gradient, diffusion, gradients):
    """Recover D from the slope of ln(E) vs applied b-value over a g-sweep."""

    b_values = []
    log_signal = []
    for g in gradients:
        result = runner(
            gradient_amplitude=float(g),
            diffusion_coefficient=diffusion,
            background_gradient=background_gradient,
        )
        b_values.append(result.b_value)
        log_signal.append(np.log(result.diffusion_attenuation))
    slope = np.polyfit(b_values, log_signal, 1)[0]
    return -slope


class ToggleIntervalValidationTests(unittest.TestCase):
    def test_sign_must_be_valid(self) -> None:
        with self.assertRaises(ValueError):
            ToggleInterval(1e-3, 0.1, sign=2)

    def test_negative_duration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ToggleInterval(-1e-3, 0.1, sign=1)


class TogglingFrameMomentTests(unittest.TestCase):
    def test_single_bipolar_pair_matches_hand_integral(self) -> None:
        # +g (sign +1) then +g (sign -1): effective bipolar, q goes 0 -> q0 -> 0
        delta = 2e-3
        g = 0.1
        intervals = [
            ToggleInterval(delta, g, +1),
            ToggleInterval(delta, g, -1),
        ]
        moments = toggling_frame_moments(intervals)
        # analytic: q(t)=gamma g t on [0,delta]; then gamma g(2delta - t) on [delta,2delta]
        # integral of q^2 over the pair = 2 * (gamma g)^2 delta^3 / 3
        expected = 2.0 * (GAMMA * g) ** 2 * delta**3 / 3.0
        self.assertAlmostEqual(moments.b_applied / expected, 1.0, places=10)
        self.assertAlmostEqual(moments.residual_applied, 0.0, places=6)

    def test_storage_parks_the_wavevector(self) -> None:
        delta = 2e-3
        g = 0.1
        storage = 50e-3
        q0 = GAMMA * g * delta
        intervals = [
            ToggleInterval(delta, g, +1),
            ToggleInterval(storage, 0.0, 0),  # parked at q0
            ToggleInterval(delta, g, -1),
        ]
        moments = toggling_frame_moments(intervals)
        # storage contributes q0^2 * storage to b_applied
        storage_contribution = q0 * q0 * storage
        self.assertGreater(moments.b_applied, storage_contribution)
        self.assertAlmostEqual(moments.residual_applied, 0.0, places=6)


class EchoAndSuppressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = dict(
            gradient_amplitude=0.1,
            gradient_duration=2e-3,
            half_echo_time=6e-3,
            storage_time=40e-3,
        )

    def test_thirteen_interval_refocuses_static_spins(self) -> None:
        moments = toggling_frame_moments(
            cotts_thirteen_interval_intervals(**self.kwargs)
        )
        self.assertTrue(moments.refocuses_static_spins)

    def test_thirteen_interval_cancels_cross_term(self) -> None:
        moments = toggling_frame_moments(
            cotts_thirteen_interval_intervals(**self.kwargs)
        )
        self.assertLess(abs(moments.cross_coefficient), 1e-6 * moments.b_applied)

    def test_monopolar_has_nonzero_cross_term(self) -> None:
        moments = toggling_frame_moments(monopolar_pgste_intervals(**self.kwargs))
        self.assertTrue(moments.refocuses_static_spins)
        self.assertGreater(abs(moments.cross_coefficient), 0.1 * moments.b_applied)

    def test_monopolar_b_value_is_stejskal_tanner_order(self) -> None:
        # one lobe per half separated by ~storage; b ~ (gamma g delta)^2 (Delta - delta/3)
        moments = toggling_frame_moments(monopolar_pgste_intervals(**self.kwargs))
        delta = self.kwargs["gradient_duration"]
        tau = self.kwargs["half_echo_time"]
        big_delta = self.kwargs["storage_time"] + 2.0 * tau
        approx = pgse_b_value(
            self.kwargs["gradient_amplitude"], delta, big_delta, gamma=GAMMA
        )
        self.assertAlmostEqual(moments.b_applied / approx, 1.0, delta=0.2)

    def test_b_value_scales_with_gradient_squared(self) -> None:
        base = toggling_frame_moments(
            cotts_thirteen_interval_intervals(**self.kwargs)
        ).b_applied
        doubled = toggling_frame_moments(
            cotts_thirteen_interval_intervals(
                **{**self.kwargs, "gradient_amplitude": 0.2}
            )
        ).b_applied
        self.assertAlmostEqual(doubled / base, 4.0, places=6)


class ApparentDiffusionTests(unittest.TestCase):
    def test_thirteen_interval_diffusion_unbiased_by_background(self) -> None:
        gradients = np.linspace(0.0, 0.2, 9)
        diffusion = 2.3e-9
        for g0 in (0.0, 0.02, 0.05):
            apparent = _apparent_d_from_sweep(
                run_cotts_thirteen_interval_moment,
                background_gradient=g0,
                diffusion=diffusion,
                gradients=gradients,
            )
            self.assertAlmostEqual(apparent / diffusion, 1.0, places=4)

    def test_monopolar_diffusion_biased_by_background(self) -> None:
        gradients = np.linspace(0.0, 0.2, 9)
        diffusion = 2.3e-9
        unbiased = _apparent_d_from_sweep(
            run_monopolar_pgste_moment,
            background_gradient=0.0,
            diffusion=diffusion,
            gradients=gradients,
        )
        self.assertAlmostEqual(unbiased / diffusion, 1.0, places=4)
        biased = _apparent_d_from_sweep(
            run_monopolar_pgste_moment,
            background_gradient=0.05,
            diffusion=diffusion,
            gradients=gradients,
        )
        # the background cross-term inflates the apparent diffusion coefficient
        self.assertGreater(biased / diffusion, 1.3)

    def test_cross_term_bias_reported_per_run(self) -> None:
        c = run_cotts_thirteen_interval_moment(
            gradient_amplitude=0.1, background_gradient=0.05
        )
        m = run_monopolar_pgste_moment(
            gradient_amplitude=0.1, background_gradient=0.05
        )
        self.assertLess(abs(c.cross_term_bias), 1e-6)
        self.assertGreater(abs(m.cross_term_bias), 0.1)
        self.assertEqual(c.label, "cotts_13_interval")
        self.assertEqual(m.label, "monopolar_pgste")

    def test_thirteen_interval_exposes_diff_stebp_phase_cycle(self) -> None:
        result = run_cotts_thirteen_interval_moment(gradient_amplitude=0.1)
        self.assertEqual(result.phase_cycle.name, "diff_stebp")
        self.assertEqual(result.phase_cycle.num_steps, 16)


class PathwayConsistencyTests(unittest.TestCase):
    def test_interval_signs_are_the_selected_coherence_pathway(self) -> None:
        # the diff_stebp phase cycle selects p = (+1, -1) on the two sides of each
        # 180 and p = 0 during storage; the interval signs must follow that, and
        # the gradient polarity must alternate between the two encoding periods
        intervals = cotts_thirteen_interval_intervals(
            gradient_amplitude=0.1,
            gradient_duration=2e-3,
            half_echo_time=6e-3,
            storage_time=40e-3,
        )
        lobes = [iv for iv in intervals if iv.applied_gradient != 0.0]
        self.assertEqual(len(lobes), 4)
        signs = [iv.sign for iv in lobes]
        self.assertEqual(signs, [+1, -1, +1, -1])  # coherence order across each 180
        polarities = [np.sign(iv.applied_gradient) for iv in lobes]
        self.assertEqual(polarities, [+1, +1, -1, -1])  # alternates between periods
        # storage interval carries coherence order 0 (parked)
        storage = [iv for iv in intervals if iv.sign == 0]
        self.assertEqual(len(storage), 1)


class WalkerRunnerTests(unittest.TestCase):
    COMMON = dict(
        gradient_duration=2e-3,
        half_echo_time=3e-3,
        storage_time=15e-3,
        excitation_duration=20e-6,
        refocusing_duration=40e-6,
        walkers_per_cell=3000,
        substeps_per_interval=6,
        seed=7,
    )

    def test_walker_reports_diff_stebp_phase_cycle(self) -> None:
        result = run_cotts_thirteen_interval_walkers(
            gradient_amplitude=0.05, **self.COMMON
        )
        self.assertEqual(result.phase_cycle.name, "diff_stebp")
        self.assertEqual(result.phase_cycle.num_steps, 16)
        self.assertEqual(result.label, "cotts_13_interval_walkers")

    def test_stationary_spins_form_an_unattenuated_echo(self) -> None:
        result = run_cotts_thirteen_interval_walkers(
            gradient_amplitude=0.1, diffusion_coefficient=0.0, **self.COMMON
        )
        # the bipolar pair refocuses, so a stationary ensemble keeps full echo
        self.assertGreater(float(np.abs(result.signal[-1])), 0.95)

    def test_free_diffusion_attenuation_matches_moment_b_value(self) -> None:
        diffusion = 2.3e-9
        for g in (0.05, 0.1, 0.15):
            wet = run_cotts_thirteen_interval_walkers(
                gradient_amplitude=g, diffusion_coefficient=diffusion, **self.COMMON
            )
            dry = run_cotts_thirteen_interval_walkers(
                gradient_amplitude=g, diffusion_coefficient=0.0, **self.COMMON
            )
            attenuation = float(np.abs(wet.signal[-1]) / np.abs(dry.signal[-1]))
            predicted = float(np.exp(-wet.b_value * diffusion))
            self.assertAlmostEqual(attenuation / predicted, 1.0, delta=0.03)

    def test_background_gradient_does_not_bias_apparent_diffusion(self) -> None:
        diffusion = 2.3e-9
        gradients = np.array([0.0, 0.06, 0.10, 0.14])

        def apparent(background: float) -> float:
            base = np.abs(
                run_cotts_thirteen_interval_walkers(
                    gradient_amplitude=0.0,
                    diffusion_coefficient=diffusion,
                    background_gradient=background,
                    **self.COMMON,
                ).signal[-1]
            )
            b_values = []
            log_signal = []
            for g in gradients:
                r = run_cotts_thirteen_interval_walkers(
                    gradient_amplitude=float(g),
                    diffusion_coefficient=diffusion,
                    background_gradient=background,
                    **self.COMMON,
                )
                b_values.append(r.b_value)
                log_signal.append(np.log(np.abs(r.signal[-1]) / base))
            return -np.polyfit(b_values, log_signal, 1)[0]

        unbiased = apparent(0.0) / diffusion
        with_background = apparent(0.04) / diffusion
        self.assertAlmostEqual(unbiased, 1.0, delta=0.06)
        self.assertAlmostEqual(with_background, 1.0, delta=0.06)

    def test_walker_rejects_bad_parameters(self) -> None:
        with self.assertRaises(ValueError):
            run_cotts_thirteen_interval_walkers(
                gradient_amplitude=0.05, storage_time=0.0, **{
                    k: v for k, v in self.COMMON.items() if k != "storage_time"
                }
            )
        with self.assertRaises(ValueError):
            run_cotts_thirteen_interval_walkers(
                gradient_amplitude=0.05, half_echo_time=1e-3, **{
                    k: v for k, v in self.COMMON.items() if k != "half_echo_time"
                }
            )


if __name__ == "__main__":
    unittest.main()
