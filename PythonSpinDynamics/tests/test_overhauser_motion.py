"""Absolute molecular-motion rates, independent spin prefactors and composition."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from spin_dynamics.dnp import (
    MotionBasedCWModel,
    RotationalDipolarModel,
    ScalarContactModel,
    TranslationalDiffusionModel,
    translational_relaxation_rates,
)
from spin_dynamics.workflows.dnp import run_overhauser_power_sweep


def _spin_operators():
    operators = [
        np.array([[0, 1], [1, 0]]) / 2,
        np.array([[0, -1j], [1j, 0]]) / 2,
        np.diag([0.5, -0.5]),
    ]
    return (
        [np.kron(op, np.eye(2)) for op in operators],
        [np.kron(np.eye(2), op) for op in operators],
    )


def _rates_from_transitions(squared_elements, field, tau, nuclear_gamma):
    electron, nucleus = _spin_operators()
    wn, we = 2 * np.pi * field * np.array([nuclear_gamma, 28.0249514242e9])
    energies = np.diag(we * electron[2] - wn * nucleus[2]).real
    frequency = energies[:, None] - energies[None, :]
    # Fermi's golden-rule rate uses two-sided 2*J; no production rate formula.
    generator = 2 * squared_elements * tau / (1 + (frequency * tau) ** 2)
    np.fill_diagonal(generator, 0)
    generator -= np.diag(generator.sum(axis=0))
    iz, ez = np.diag(nucleus[2]).real, np.diag(electron[2]).real
    return -iz @ generator @ iz / (iz @ iz), -iz @ generator @ ez / (iz @ iz)


class OverhauserMotionTests(unittest.TestCase):
    def test_rotational_rates_against_angular_hamiltonian_average(self):
        electron, nucleus = _spin_operators()
        squared = np.zeros((4, 4))
        z_values, weights = np.polynomial.legendre.leggauss(6)
        for z, weight in zip(z_values, weights):
            for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                r = [
                    np.sqrt(1 - z * z) * np.cos(phi),
                    np.sqrt(1 - z * z) * np.sin(phi),
                    z,
                ]
                h = sum(s @ i for s, i in zip(electron, nucleus)) - 3 * (
                    sum(v * s for v, s in zip(r, electron))
                    @ sum(v * i for v, i in zip(r, nucleus))
                )
                squared += weight / 24 * np.abs(h) ** 2
        motion = RotationalDipolarModel(0.9e-9, 130e-12, pair_count=0.03)
        for gamma in [42.57747892e6, 10.7084e6]:
            b = (
                1.00000000055e-7
                * 6.62607015e-34
                / (2 * np.pi)
                * (2 * np.pi * gamma)
                * (2 * np.pi * 28.0249514242e9)
                / motion.distance_m**3
            )
            for field in [0.01, 0.35, 3.4]:
                reference = _rates_from_transitions(
                    squared * b * b * motion.pair_count,
                    field,
                    motion.correlation_time_seconds,
                    gamma,
                )
                rates = motion.relaxation_rates(field, gamma_n_hz_per_t=gamma)
                assert_allclose(
                    [
                        rates.self_relaxation_per_second,
                        rates.cross_relaxation_per_second,
                    ],
                    reference,
                    rtol=2e-12,
                    atol=1e-15,
                )

    def test_scalar_rates_against_isotropic_contact_hamiltonian(self):
        electron, nucleus = _spin_operators()
        contact = ScalarContactModel(2.3e6, 8e-12, contact_fraction=0.07)
        h = (
            2
            * np.pi
            * contact.rms_hyperfine_hz
            * sum(s @ i for s, i in zip(electron, nucleus))
        )
        for gamma in [42.57747892e6, 10.7084e6]:
            for field in [0.01, 0.35, 3.4]:
                reference = _rates_from_transitions(
                    np.abs(h) ** 2 * contact.contact_fraction,
                    field,
                    contact.correlation_time_seconds,
                    gamma,
                )
                rates = contact.relaxation_rates(field, gamma_n_hz_per_t=gamma)
                assert_allclose(
                    [
                        rates.self_relaxation_per_second,
                        rates.cross_relaxation_per_second,
                    ],
                    reference,
                    rtol=3e-14,
                )
                self.assertEqual(rates.coupling_factor, -1)

    def test_rotational_limits_distance_scaling_and_population_weights(self):
        first = RotationalDipolarModel(1e-9, 10e-12, 0.2)
        larger = RotationalDipolarModel(2e-9, 10e-12, 0.4)
        low = first.relaxation_rates(1e-8)
        assert_allclose(low.coupling_factor, 0.5, rtol=1e-12)
        a = first.relaxation_rates([0.1, 1])
        b = larger.relaxation_rates([0.1, 1])
        assert_allclose(b.self_relaxation_per_second, a.self_relaxation_per_second / 32)
        assert_allclose(b.coupling_factor, a.coupling_factor)
        self.assertEqual(
            RotationalDipolarModel(1e-9, 1e-10, 0)
            .relaxation_rates(1)
            .total_r1_per_second,
            0,
        )
        high = first.relaxation_rates(10)
        self.assertLess(high.coupling_factor, a.coupling_factor[-1])

    def test_scalar_covariance_scaling_zero_contact_and_frequency(self):
        first = ScalarContactModel(1e6, 1e-11, 0.1)
        second = ScalarContactModel(2e6, 1e-11, 0.2)
        r1 = first.relaxation_rates([0.1, 1, 10])
        r2 = second.relaxation_rates([0.1, 1, 10])
        assert_allclose(
            r2.self_relaxation_per_second, 8 * r1.self_relaxation_per_second
        )
        self.assertTrue(np.all(np.diff(r1.self_relaxation_per_second) < 0))
        self.assertEqual(
            ScalarContactModel(0, 1e-11).relaxation_rates(1).total_r1_per_second, 0
        )
        self.assertEqual(
            ScalarContactModel(1e6, 1e-11, 0).relaxation_rates(1).total_r1_per_second, 0
        )

    def test_ffhs_wrapper_parity_and_no_background_double_counting(self):
        diffusion = TranslationalDiffusionModel(0.39e-9, 2.8e-9)
        model = MotionBasedCWModel(diffusion, 0.01)
        fields = np.array([0.1, 0.35, 1])
        old = translational_relaxation_rates(
            fields, diffusion, 0.01, background_r1_per_second=0.4
        )
        new = model.relaxation_rates(fields, background_r1_per_second=0.4)
        for name in [
            "background_r1_per_second",
            "self_relaxation_per_second",
            "cross_relaxation_per_second",
        ]:
            assert_allclose(getattr(old, name), getattr(new, name), rtol=1e-15)
        self.assertEqual(
            set(model.rate_components(fields)), {"background", "translational"}
        )

    def test_composition_cancellation_and_vectorized_conditions(self):
        # Choose scalar RMS to cancel the rotational cross rate at a known field.
        rotation = RotationalDipolarModel(1e-9, 30e-12)
        unit = ScalarContactModel(1, 5e-12)
        target = rotation.relaxation_rates(0.35).cross_relaxation_per_second
        amplitude = np.sqrt(
            target / unit.relaxation_rates(0.35).self_relaxation_per_second
        )
        scalar = ScalarContactModel(float(amplitude), 5e-12)
        model = MotionBasedCWModel(rotational_models=[rotation], scalar_models=[scalar])
        rates = model.relaxation_rates(0.35, background_r1_per_second=0.4)
        assert_allclose(rates.cross_relaxation_per_second, 0, atol=1e-15)
        cw = model.cw(0.35, saturation_factor=0.8, background_r1_per_second=0.4)
        assert_allclose(cw.enhancement, 1, atol=1e-12)
        fields = np.array([[0.1, 0.35, 1]])
        result = model.cw(
            fields,
            saturation_factor=np.array([[0], [0.8]]),
            temperature_kelvin=np.array([[298], [310]]),
            background_r1_per_second=np.array([[0.3], [0.4]]),
        )
        self.assertEqual(result.enhancement.shape, (2, 3))
        assert_allclose(result.enhancement[0], 1)
        assert_allclose(result.rates.background_r1_per_second, [[0.3] * 3, [0.4] * 3])
        self.assertIsInstance(model.rotational_models, tuple)

    def test_power_workflow_mixed_channels_and_conflicting_inputs(self):
        model = MotionBasedCWModel(scalar_models=(ScalarContactModel(1e6, 5e-12),))
        sweep = run_overhauser_power_sweep(
            [0.1, 0.35, 1],
            [0, 1],
            model=model,
            half_saturation_power_watts=0.3,
            background_r1_per_second=0.4,
        )
        self.assertEqual(sweep.cw.enhancement.shape, (2, 3))
        assert_allclose(sweep.cw.enhancement[0], 1)
        self.assertTrue(np.all(sweep.cw.enhancement[1] > 1))
        for kwargs in [
            dict(concentration_molar=0.01),
            dict(scalar_self_rate_per_second=1),
        ]:
            with self.assertRaises(ValueError):
                run_overhauser_power_sweep(
                    [1], [1], model=model, half_saturation_power_watts=1, **kwargs
                )

    def test_invalid_physical_parameters_and_zero_channel_model(self):
        for args in [(0, 1e-11), (1e-9, -1), (1e-9, 1e-11, -1), ([1, 2], 1e-11)]:
            with self.assertRaises(ValueError):
                RotationalDipolarModel(*args)
        for args in [(-1, 1e-11), (1, np.inf), (1, 1e-11, 1.1), (1, 1e-11, [0.5])]:
            with self.assertRaises(ValueError):
                ScalarContactModel(*args)
        with self.assertRaises(ValueError):
            MotionBasedCWModel(concentration_molar=0.1)
        with self.assertRaises(TypeError):
            MotionBasedCWModel(rotational_models=(ScalarContactModel(1, 1),))
        empty = MotionBasedCWModel().cw(0.35, saturation_factor=1)
        self.assertEqual(empty.enhancement, 1)
        self.assertTrue(np.isinf(empty.rates.t1_seconds))


if __name__ == "__main__":
    unittest.main()
