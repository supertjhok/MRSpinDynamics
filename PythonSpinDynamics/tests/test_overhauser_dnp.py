"""Physical limits, measured-condition workflows, and CW DNP integration."""

import json
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose

from spin_dynamics.dnp import (
    OverhauserRates,
    TranslationalDiffusionModel,
    bloch_saturation,
    cw_overhauser,
    overhauser_from_factors,
    polarization_buildup,
    power_saturation,
    translational_relaxation_rates,
)
from spin_dynamics.workflows.dnp import run_overhauser_power_sweep


class OverhauserDNPTests(unittest.TestCase):
    def setUp(self):
        self.motion = TranslationalDiffusionModel(3e-10, 3e-9)

    def cw(self, rates, **kwargs):
        return cw_overhauser(
            rates,
            field_tesla=0.35,
            saturation_factor=kwargs.pop("saturation_factor", 1),
            **kwargs,
        )

    def test_no_pumping_no_radicals_and_zero_rate_degeneracy(self):
        for rates, saturation in [
            (OverhauserRates(1, 2, 1), 0),
            (OverhauserRates(1, 0, 0), 1),
            (OverhauserRates(0, 0, 0), 1),
            (OverhauserRates(1, 2, 0), 1),
        ]:
            result = self.cw(rates, saturation_factor=saturation)
            self.assertEqual(result.enhancement, 1)
            self.assertTrue(np.isfinite(result.thermal_polarization))
        result = self.cw(OverhauserRates(0, 0, 0))
        self.assertTrue(np.isinf(result.rates.t1_seconds))
        assert_allclose(
            polarization_buildup(
                result, [0, 1, 100], initial_polarization=0.1
            ).polarization,
            0.1,
        )

    def test_dipolar_and_scalar_limits_and_cancellation(self):
        rates = translational_relaxation_rates(1e-12, self.motion, 0.001)
        assert_allclose(rates.coupling_factor, 0.5, rtol=1e-6)
        ratio = 28.0249514242e9 / 42.57747892e6
        self.assertAlmostEqual(
            float(self.cw(OverhauserRates(0, 2, 1)).enhancement), 1 - ratio / 2
        )
        self.assertAlmostEqual(
            float(self.cw(OverhauserRates(0, 1, -1)).enhancement), 1 + ratio
        )
        dipolar = translational_relaxation_rates(0.35, self.motion, 0.001)
        mixed = translational_relaxation_rates(
            0.35,
            self.motion,
            0.001,
            scalar_self_rate_per_second=dipolar.cross_relaxation_per_second,
        )
        self.assertEqual(mixed.cross_relaxation_per_second, 0)
        self.assertEqual(self.cw(mixed).enhancement, 1)

    def test_concentration_scaling_and_leakage(self):
        low = translational_relaxation_rates(
            0.35, self.motion, 0.001, background_r1_per_second=0.4
        )
        high = translational_relaxation_rates(
            0.35, self.motion, 0.002, background_r1_per_second=0.4
        )
        assert_allclose(
            high.self_relaxation_per_second, 2 * low.self_relaxation_per_second
        )
        assert_allclose(high.coupling_factor, low.coupling_factor)
        self.assertGreater(high.leakage_factor, low.leakage_factor)
        assert_allclose(low.leakage_factor, 1 - low.t1_seconds / 2.5)

    def test_measured_relaxivity_and_factors_agree(self):
        rates = OverhauserRates.from_relaxivities(
            0.001, 300, 90, background_r1_per_second=0.2
        )
        result = self.cw(rates, saturation_factor=0.6)
        factors = overhauser_from_factors(
            coupling_factor=0.3,
            leakage_factor=0.6,
            t1_seconds=2,
            field_tesla=0.35,
            saturation_factor=0.6,
        )
        assert_allclose(result.enhancement, factors.enhancement)
        assert_allclose(result.rates.t1_seconds, factors.rates.t1_seconds)

    def test_heating_dependent_measured_t1_conditions(self):
        rates = OverhauserRates.from_t1([1, 1.5], [2, 3], coupling_factor=0.27)
        result = cw_overhauser(
            rates,
            field_tesla=0.35,
            saturation_factor=[0, 0.8],
            temperature_kelvin=[298, 310],
        )
        assert_allclose(result.leakage_factor, 0.5)
        assert_allclose(result.rates.t1_seconds, [1, 1.5])
        self.assertEqual(result.enhancement[0], 1)
        self.assertNotEqual(
            result.thermal_polarization[0], result.thermal_polarization[1]
        )

    def test_buildup_recovery_and_time_leading_broadcast(self):
        result = self.cw(OverhauserRates(1, [1, 2], [0.4, 0.8]))
        times = [0, 0.1, 100]
        buildup = polarization_buildup(result, times)
        self.assertEqual(buildup.polarization.shape, (3, 2))
        assert_allclose(buildup.polarization[0], result.thermal_polarization)
        assert_allclose(buildup.polarization[-1], result.steady_state_polarization)
        recovered = polarization_buildup(
            result, times, pumping=False, off_t1_seconds=0.5
        )
        assert_allclose(recovered.polarization[0], result.steady_state_polarization)
        assert_allclose(recovered.polarization[-1], result.thermal_polarization)
        assert_allclose(
            (recovered.polarization[1] - result.thermal_polarization)
            / (result.steady_state_polarization - result.thermal_polarization),
            np.exp(-0.1 / 0.5),
        )

    def test_saturation_calibration_and_detuning(self):
        assert_allclose(power_saturation([0, 1, 1e20], 1, maximum=0.8), [0, 0.4, 0.8])
        self.assertEqual(bloch_saturation(0, 1, 1), 0)
        self.assertEqual(bloch_saturation(1, 1, 1), 0.5)
        assert_allclose(bloch_saturation(1, 1, 1, detuning_rad_per_s=1), 1 / 3)
        assert_allclose(bloch_saturation(1e100, 1, 1, maximum=0.7), 0.7)
        assert_allclose(power_saturation(1e308, 1e308), 0.5)
        assert_allclose(bloch_saturation(1e308, 1e308, 1e308), 1)
        assert_allclose(bloch_saturation(0, 1e308, 1e308), 0)

    def test_nano_mr_adapter_preserves_absolute_polarization(self):
        from spin_dynamics.nano_mr.high_resolution import dnp_polarization

        result = self.cw(OverhauserRates(0.4, 0.3, 0.09))
        times = np.array([0, 0.5, 1, 10])
        legacy = result.to_nano_mr_model()
        for pumping in [True, False]:
            initial = (
                result.thermal_polarization
                if pumping
                else result.steady_state_polarization
            )
            expected = polarization_buildup(result, times, pumping=pumping)
            actual = dnp_polarization(
                legacy,
                times,
                float(result.thermal_polarization),
                pumping=pumping,
                initial_polarization=float(initial),
            )
            assert_allclose(actual.polarization, expected.polarization)
        with self.assertRaises(ValueError):
            self.cw(OverhauserRates(0, 0, 0)).to_nano_mr_model()
        with self.assertRaises(ValueError):
            self.cw(OverhauserRates(1, [1, 2], 0)).to_nano_mr_model()

    def test_power_field_workflow_and_numerical_broadcast(self):
        sweep = run_overhauser_power_sweep(
            [0.1, 0.35, 1],
            [0, 0.1, 1, 10],
            model=self.motion,
            concentration_molar=0.001,
            half_saturation_power_watts=0.5,
            maximum_saturation=0.8,
            background_r1_per_second=0.4,
        )
        self.assertEqual(sweep.cw.enhancement.shape, (4, 3))
        assert_allclose(sweep.cw.enhancement[0], 1)
        self.assertTrue(np.all(np.diff(sweep.cw.enhancement, axis=0) < 0))
        # Higher field lowers the coupling factor, not necessarily absolute P.
        self.assertTrue(np.all(np.diff(sweep.cw.coupling_factor[0]) < 0))
        self.assertEqual(
            polarization_buildup(sweep.cw, [0, 1]).polarization.shape, (2, 4, 3)
        )

    def test_literature_diffusion_curve_reference(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "validation/experimental/overhauser_ffhs_reference.json"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        # Hwang-Freed Table I: half of J(0)*d*D/n = 4/27.
        assert_allclose(
            ffhs_zero := self.motion.spectral_density(0),
            data["normalized_one_sided_j0_over_tau"]
            * self.motion.diffusion_time_seconds,
        )
        self.assertGreater(ffhs_zero, 0)
        for row in data["diffusion_curve"]:
            rates = translational_relaxation_rates(
                row["field_tesla"],
                TranslationalDiffusionModel(3e-10, 9e-20 / row["tau_d_seconds"]),
                0.001,
            )
            assert_allclose(rates.coupling_factor, row["coupling_factor"], atol=2e-9)

    def test_invalid_inputs_and_validity_boundary(self):
        for args in [(0, 0, 1), (-1, 1, 0), (0, 1, -2), (np.nan, 1, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                OverhauserRates(*args)
        for kwargs in [
            dict(saturation_factor=1.01),
            dict(temperature_kelvin=0),
            dict(temperature_kelvin=0.1),
            dict(gamma_n_hz_per_t=-1),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.cw(OverhauserRates(1, 1, 0.3), **kwargs)
        with self.assertRaises(ValueError):
            cw_overhauser(
                OverhauserRates(1, 1, 0.3), field_tesla=0, saturation_factor=1
            )
        with self.assertRaises(ValueError):
            OverhauserRates.from_t1(3, 2, coupling_factor=0.3)
        with self.assertRaises(ValueError):
            OverhauserRates.from_relaxivities(0, 1, 2)
        with self.assertRaises(ValueError):
            polarization_buildup(self.cw(OverhauserRates(1, 1, 0)), [-1])
        with self.assertRaises(ValueError):
            polarization_buildup(
                self.cw(OverhauserRates(1, 1, 0)), [0], off_t1_seconds=1
            )


if __name__ == "__main__":
    unittest.main()
