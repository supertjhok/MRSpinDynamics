"""Analytic limits and independent six-population collision integration."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from spin_dynamics.dnp import NitroxideSaturationModel, bloch_saturation


class NitroxideSaturationTests(unittest.TestCase):
    def setUp(self):
        self.model = NitroxideSaturationModel()
        self.parameters = dict(
            field_tesla=0.35, concentration_molar=0.005, nutation_rad_per_s=2e7
        )

    def test_no_drive_thermal_and_population_normalization(self):
        model = replace(
            self.model,
            exchange_coefficient_per_molar_second=2e9,
            nitrogen_transition_per_second=1e5,
        )
        result = model.steady_state(**{**self.parameters, "nutation_rad_per_s": 0})
        assert_allclose(result.saturation_factor, 0, atol=2e-15)
        assert_allclose(result.nitrogen_populations, np.full(3, 1 / 3), atol=1e-16)
        assert_allclose(
            result.electron_polarization,
            result.thermal_electron_polarization,
            rtol=2e-13,
        )
        assert_allclose(
            model.population_derivative(
                result.populations, **{**self.parameters, "nutation_rad_per_s": 0}
            ),
            0,
            atol=3e-10,
        )

    def test_independent_lines_reproduce_bloch(self):
        for spin in (0.5, 1):
            model = replace(self.model, nitrogen_spin=spin)
            result = model.steady_state(**self.parameters)
            expected = bloch_saturation(
                2e7,
                model.electron_t1_seconds,
                model.electron_t2_seconds,
                detuning_rad_per_s=2 * np.pi * model.hyperfine_hz * model.nitrogen_m,
            )
            assert_allclose(result.line_saturation, expected, rtol=2e-13)
            assert_allclose(result.saturation_factor, expected.mean(), rtol=2e-13)

    def test_selective_line_limit_one_third_or_one_half(self):
        for spin in (0.5, 1):
            # Separate the line width, drive saturation and hyperfine scales.
            model = NitroxideSaturationModel(
                hyperfine_hz=1e9,
                electron_t1_seconds=1,
                electron_t2_seconds=1e-3,
                nitrogen_spin=spin,
            )
            result = model.steady_state(
                field_tesla=1,
                concentration_molar=0,
                nutation_rad_per_s=1e4,
                detuning_rad_per_s=-2 * np.pi * 1e9 * spin,
            )
            self.assertAlmostEqual(
                float(result.saturation_factor), 1 / (2 * spin + 1), delta=6e-6
            )

    def test_exchange_central_line_analytic_solution(self):
        # Three equal populations, only central line driven: each outer line
        # s_outer = k*s_center/(3*R+k), and s_total = s_center*(R+k)/(3*R+k).
        model = NitroxideSaturationModel(
            hyperfine_hz=1e9,
            electron_t1_seconds=0.01,
            electron_t2_seconds=1e-3,
            exchange_coefficient_per_molar_second=6e4,
        )
        result = model.steady_state(
            field_tesla=1, concentration_molar=0.01, nutation_rad_per_s=1e3
        )
        rate, exchange, drive = 100, 600, 1000  # drive = 2W
        center = drive / (rate + drive + 2 * exchange * rate / (3 * rate + exchange))
        outer = exchange * center / (3 * rate + exchange)
        assert_allclose(result.line_saturation, [outer, center, outer], atol=1e-10)

    def test_fast_exchange_and_nitrogen_relaxation_mix_lines(self):
        parameters = {
            **self.parameters,
            "concentration_molar": np.array([0, 0.001, 0.01]),
        }
        model = replace(self.model, exchange_coefficient_per_molar_second=1e11)
        result = model.steady_state(**parameters)
        self.assertTrue(np.all(np.diff(result.saturation_factor) > 0))
        self.assertLess(
            np.ptp(result.line_saturation[-1]), np.ptp(result.line_saturation[0])
        )
        # In the rapid mixing limit, drive rates are averaged before saturation.
        fast = replace(model, exchange_coefficient_per_molar_second=1e15)
        mixed = fast.steady_state(**self.parameters)
        drive = 2 * mixed.microwave_transition_rates_per_second.mean()
        assert_allclose(
            mixed.saturation_factor,
            drive / (1 / fast.electron_t1_seconds + drive),
            rtol=3e-6,
        )
        nitrogen = replace(self.model, nitrogen_transition_per_second=1e12)
        nitrogen_result = nitrogen.steady_state(**self.parameters)
        assert_allclose(
            nitrogen_result.saturation_factor,
            drive / (1 / fast.electron_t1_seconds + drive),
            rtol=1e-5,
        )

    def test_collision_term_matches_pair_enumeration_and_conserves_marginals(self):
        model = replace(self.model, exchange_coefficient_per_molar_second=3e8)
        p = np.array([[0.03, 0.12], [0.25, 0.15], [0.30, 0.15]])
        collision = model.population_derivative(
            p, **self.parameters
        ) - self.model.population_derivative(p, **self.parameters)
        independent = np.zeros_like(p)
        k = 3e8 * self.parameters["concentration_molar"]
        for m in range(3):
            for n in range(m + 1, 3):
                flux = k * (p[m, 0] * p[n, 1] - p[m, 1] * p[n, 0])
                independent[m] += [-flux, flux]
                independent[n] += [flux, -flux]
        assert_allclose(collision, independent, atol=1e-10)
        assert_allclose(collision.sum(axis=0), 0, atol=1e-10)
        assert_allclose(collision.sum(axis=1), 0, atol=1e-10)

    def test_independent_six_population_integration(self):
        model = NitroxideSaturationModel(
            hyperfine_hz=1e3,
            electron_t1_seconds=0.01,
            electron_t2_seconds=0.001,
            nitrogen_transition_per_second=30,
            exchange_coefficient_per_molar_second=2e4,
        )
        params = dict(
            field_tesla=0.35,
            concentration_molar=0.01,
            nutation_rad_per_s=700,
            detuning_rad_per_s=1500,
        )
        result = model.steady_state(**params)
        pe = float(result.thermal_electron_polarization)
        w = result.microwave_transition_rates_per_second

        def rhs(_, vector):
            p = vector.reshape(3, 2)
            dp = np.zeros_like(p)
            for m in range(3):
                flux = (50 * (1 + pe) + w[m]) * p[m, 0] - (50 * (1 - pe) + w[m]) * p[
                    m, 1
                ]
                dp[m] += [-flux, flux]
                if m < 2:
                    flow = 30 * (p[m + 1] - p[m])
                    dp[m] += flow
                    dp[m + 1] -= flow
                for n in range(m + 1, 3):
                    flux = 200 * (p[m, 0] * p[n, 1] - p[m, 1] * p[n, 0])
                    dp[m] += [-flux, flux]
                    dp[n] += [flux, -flux]
            return dp.ravel()

        initial = np.tile([(1 - pe) / 6, (1 + pe) / 6], (3, 1))
        solution = solve_ivp(
            rhs, (0, 0.5), initial.ravel(), method="DOP853", rtol=1e-11, atol=1e-13
        )
        self.assertTrue(solution.success)
        self.assertGreaterEqual(solution.y.min(), 0)
        assert_allclose(solution.y.sum(axis=0), 1, atol=2e-14)
        assert_allclose(solution.y[:, -1].reshape(3, 2), result.populations, atol=3e-12)
        assert_allclose(
            model.population_derivative(result.populations, **params), 0, atol=1e-13
        )

    def test_broadcasting_broadening_and_exchange_units(self):
        model = replace(
            self.model,
            exchange_coefficient_per_molar_second=2e9,
            transverse_broadening_per_molar_second=1e9,
        )
        c = np.array([0, 0.001, 0.01])
        result = model.steady_state(
            field_tesla=np.array([0.1, 0.35])[:, None],
            concentration_molar=c,
            nutation_rad_per_s=2e7,
        )
        self.assertEqual(result.populations.shape, (2, 3, 3, 2))
        assert_allclose(result.exchange_rate_per_second[0], 2e9 * c)
        assert_allclose(1 / result.t2e_seconds[0], 1e7 + 1e9 * c)
        assert_allclose(result.populations.sum(axis=(-1, -2)), 1, atol=2e-16)
        self.assertTrue(np.all(result.populations >= 0))
        self.assertTrue(
            np.all((result.saturation_factor >= 0) & (result.saturation_factor <= 1))
        )

    def test_off_recovery_from_nonuniform_nitrogen_and_population_boundary(self):
        model = NitroxideSaturationModel(
            hyperfine_hz=1000,
            electron_t1_seconds=0.01,
            electron_t2_seconds=0.001,
            nitrogen_transition_per_second=100,
            exchange_coefficient_per_molar_second=2e4,
        )
        params = dict(field_tesla=0.35, concentration_molar=0.01, nutation_rad_per_s=0)
        initial = np.array([[0, 0.1], [0.4, 0.2], [0.2, 0.1]])
        derivative = model.population_derivative(initial, **params)
        self.assertGreaterEqual(derivative[0, 0], 0)
        self.assertAlmostEqual(float(derivative.sum()), 0, delta=1e-13)
        solution = solve_ivp(
            lambda _, p: model.population_derivative(p.reshape(3, 2), **params).ravel(),
            (0, 0.5),
            initial.ravel(),
            method="DOP853",
            rtol=1e-10,
            atol=1e-13,
            max_step=0.002,
        )
        self.assertTrue(solution.success)
        thermal = model.steady_state(**params)
        assert_allclose(
            solution.y[:, -1].reshape(3, 2), thermal.populations, atol=1e-12
        )
        assert_allclose(thermal.saturation_factor, 0, atol=0)

    def test_invalid_parameters_rejected(self):
        for options in (
            {"nitrogen_spin": 1.5},
            {"electron_t1_seconds": 0},
            {"electron_t2_seconds": 2e-6},
            {"hyperfine_hz": np.nan},
            {"exchange_coefficient_per_molar_second": -1},
            {"nitrogen_transition_per_second": [1, 2]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                NitroxideSaturationModel(**options)
        for options in (
            {"field_tesla": 0.001},
            {"field_tesla": 100},
            {"concentration_molar": -1},
            {"nutation_rad_per_s": np.inf},
            {"temperature_kelvin": 0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.model.steady_state(**{**self.parameters, **options})
        with self.assertRaises(ValueError):
            self.model.population_derivative(np.ones((3, 2)), **self.parameters)
        with self.assertRaises(ValueError):
            replace(
                self.model, exchange_coefficient_per_molar_second=1e30
            ).steady_state(**self.parameters)


if __name__ == "__main__":
    unittest.main()
