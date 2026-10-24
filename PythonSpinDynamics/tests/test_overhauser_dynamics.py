"""Thermal balance, coherent driving, physical states and the CW limiting case."""

from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from spin_dynamics.dnp import (
    CoupledElectronNuclearModel,
    ElectronNuclearTransitionRates,
    MicrowaveDrive,
    RotationalDipolarModel,
    ScalarContactModel,
    cw_overhauser,
)

try:
    import scipy.linalg  # noqa: F401
except ImportError:
    HAS_SCIPY = False
else:
    HAS_SCIPY = True


class OverhauserDynamicsTests(unittest.TestCase):
    def setUp(self):
        self.transitions = ElectronNuclearTransitionRates(0.05, 0.3, 0.1, 0.02)
        self.model = CoupledElectronNuclearModel(0.35, transitions=self.transitions)

    def test_exact_thermal_polarizations_and_finite_temperature(self):
        model = replace(self.model, temperature_kelvin=3.0)
        density = model.thermal_density()
        electron, nucleus = model.polarizations(density)
        factor = (
            6.62607015e-34
            * model.field_tesla
            / (2 * 1.380649e-23 * model.temperature_kelvin)
        )
        assert_allclose(
            [electron, nucleus],
            np.tanh(
                factor * np.array([model.gamma_e_hz_per_t, model.gamma_n_hz_per_t])
            ),
            rtol=2e-12,
        )
        self.assertAlmostEqual(np.trace(density).real, 1)
        self.assertGreater(np.linalg.eigvalsh(density).min(), 0)

    def test_gibbs_stationarity_detailed_balance_and_secular_hyperfine(self):
        model = replace(
            self.model,
            temperature_kelvin=2.0,
            electron_t1_seconds=0.02,
            secular_hyperfine_hz=2e5,
        )
        density = model.thermal_density()
        generator = model.liouvillian()
        vector = density.reshape(-1, order="F")
        assert_allclose(generator @ vector, 0, atol=3e-13)
        indices = np.arange(4) * 5
        population_generator = generator[np.ix_(indices, indices)].real
        populations = np.diag(density).real
        for source in range(4):
            for destination in range(4):
                if source != destination:
                    assert_allclose(
                        population_generator[destination, source] * populations[source],
                        population_generator[source, destination]
                        * populations[destination],
                        rtol=3e-14,
                        atol=1e-15,
                    )
        assert_allclose(model.steady_state().density, density, atol=1e-13)
        # Baths use laboratory gaps even if the microwave frame is detuned.
        assert_allclose(
            model.liouvillian(MicrowaveDrive(detuning_rad_per_s=2e6)) @ vector,
            0,
            atol=3e-13,
        )

    def test_trace_generator_and_high_temperature_solomon_coefficients(self):
        model = replace(self.model, temperature_kelvin=1e10, electron_t1_seconds=0.1)
        generator = model.liouvillian(MicrowaveDrive(100, 30, 0.7))
        trace = np.eye(4).reshape(-1, order="F")
        assert_allclose(trace @ generator, 0, atol=1e-14)
        indices = np.arange(4) * 5
        pop = model.liouvillian()[np.ix_(indices, indices)].real
        iz = np.array([0.5, -0.5, 0.5, -0.5])
        sz = np.array([0.5, 0.5, -0.5, -0.5])
        assert_allclose(-iz @ pop @ iz, 0.4 + 0.05 + 2 * 0.1 + 0.3, rtol=2e-13)
        assert_allclose(-iz @ pop @ sz, 0.3 - 0.05, rtol=2e-13)
        assert_allclose(-sz @ pop @ sz, 10 + 0.05 + 0.3 + 2 * 0.02, rtol=2e-13)

    def test_uncoupled_driven_electron_matches_bloch_saturation(self):
        model = CoupledElectronNuclearModel(
            0.35,
            electron_t1_seconds=0.2,
            nuclear_t1_seconds=0.7,
            electron_dephasing_per_second=3,
        )
        t2 = 1 / (
            1 / (2 * model.electron_t1_seconds) + model.electron_dephasing_per_second
        )
        drive = MicrowaveDrive(8, 3)
        strength = drive.nutation_rad_per_s**2 * model.electron_t1_seconds * t2
        expected_saturation = strength / (
            1 + (drive.detuning_rad_per_s * t2) ** 2 + strength
        )
        steady = model.steady_state(drive)
        assert_allclose(steady.electron_saturation, expected_saturation, rtol=2e-12)
        assert_allclose(
            steady.nuclear_polarization,
            model.polarizations(model.thermal_density())[1],
            rtol=2e-9,
        )
        self.assertLess(steady.residual_per_second, 1e-12)
        phased = model.steady_state(replace(drive, phase_rad=np.pi / 2))
        assert_allclose(
            phased.electron_polarization, steady.electron_polarization, atol=1e-14
        )

    def test_weak_coupling_stationary_limit_matches_cw_with_actual_saturation(self):
        for field in [0.035, 0.35]:
            model = replace(self.model, field_tesla=field)
            steady = model.steady_state(MicrowaveDrive(2e5))
            rates = model.transitions.nuclear_cw_rates(
                background_r1_per_second=1 / model.nuclear_t1_seconds
            )
            cw = cw_overhauser(
                rates,
                field_tesla=field,
                temperature_kelvin=model.temperature_kelvin,
                saturation_factor=steady.electron_saturation,
            )
            assert_allclose(steady.nuclear_enhancement, cw.enhancement, rtol=3e-6)
            self.assertLess(steady.nuclear_polarization, 0)
            self.assertTrue(0 < steady.electron_saturation < 1)

    @unittest.skipUnless(HAS_SCIPY, "propagation uses optional SciPy")
    def test_coherent_rabi_oscillations_and_no_relaxation(self):
        model = CoupledElectronNuclearModel(
            0.35,
            electron_t1_seconds=np.inf,
            nuclear_t1_seconds=np.inf,
            electron_dephasing_per_second=0,
        )
        initial = np.diag([0, 0, 1, 0]).astype(complex)
        times = np.linspace(0, 0.3, 31)
        result = model.simulate(
            times, drive=MicrowaveDrive(20), initial_density=initial
        )
        assert_allclose(result.electron_polarization, np.cos(20 * times), atol=2e-14)
        assert_allclose(result.nuclear_polarization, 1, atol=1e-14)
        self.assertPhysical(result.density_matrices)

    @unittest.skipUnless(HAS_SCIPY, "propagation uses optional SciPy")
    def test_pumping_and_microwave_off_thermal_recovery(self):
        drive = MicrowaveDrive(5e4)
        pumped = self.model.simulate([0, 0.001, 1, 5, 15], drive=drive)
        assert_allclose(
            pumped.density_matrices[0], self.model.thermal_density(), atol=0
        )
        self.assertLess(pumped.nuclear_polarization[-1], 0)
        assert_allclose(
            pumped.density_matrices[-1],
            self.model.steady_state(drive).density,
            atol=3e-10,
        )
        recovered = self.model.simulate(
            [0, 0.001, 1, 5, 25], initial_density=pumped.density_matrices[-1]
        )
        assert_allclose(
            recovered.density_matrices[0], pumped.density_matrices[-1], atol=0
        )
        assert_allclose(
            recovered.density_matrices[-1], self.model.thermal_density(), atol=3e-10
        )
        self.assertPhysical(pumped.density_matrices)
        self.assertPhysical(recovered.density_matrices)

    @unittest.skipUnless(HAS_SCIPY, "propagation uses optional SciPy")
    def test_longitudinal_relaxation_and_random_density_remain_physical(self):
        from scipy.linalg import expm

        rng = np.random.default_rng(42)
        matrix = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        density = matrix @ matrix.conj().T
        density /= np.trace(density)
        model = replace(
            self.model,
            electron_t1_seconds=0.02,
            secular_hyperfine_hz=2e4,
            temperature_kelvin=4,
        )
        result = model.simulate(
            [0, 0, 0.01, 0.05, 0.2, 1],
            drive=MicrowaveDrive(3e4, 1e4, 0.3),
            initial_density=density,
        )
        self.assertPhysical(result.density_matrices)
        # Check the trace-preserving coordinate propagation against a direct
        # full Liouville exponential over a well-conditioned short interval.
        drive = MicrowaveDrive(300, 100, 0.7)
        short = model.simulate([0.003], drive=drive, initial_density=density)
        direct = (
            expm(model.liouvillian(drive) * 0.003) @ density.reshape(-1, order="F")
        ).reshape((4, 4), order="F")
        assert_allclose(short.density_matrices[0], direct, atol=3e-13, rtol=0)
        # Independent T1 recovery is analytic when cross channels are absent.
        uncoupled = CoupledElectronNuclearModel(
            0.35, electron_t1_seconds=0.2, nuclear_t1_seconds=0.7
        )
        times = np.array([0.03, 0.2, 0.8])
        initial = np.diag([0, 0, 1, 0]).astype(complex)
        recovery = uncoupled.simulate(times, initial_density=initial)
        pe0, pn0 = uncoupled.polarizations(uncoupled.thermal_density())
        assert_allclose(
            recovery.electron_polarization,
            pe0 + (1 - pe0) * np.exp(-times / 0.2),
            atol=3e-12,
        )
        assert_allclose(
            recovery.nuclear_polarization,
            pn0 + (1 - pn0) * np.exp(-times / 0.7),
            atol=3e-12,
        )

    def test_transition_factories_match_motion_based_nuclear_rates(self):
        for field in [0.1, 0.35, 3.4]:
            for motion, factory in [
                (
                    RotationalDipolarModel(1e-9, 50e-12),
                    ElectronNuclearTransitionRates.from_rotational_model,
                ),
                (
                    ScalarContactModel(1e6, 5e-12, 0.1),
                    ElectronNuclearTransitionRates.from_scalar_model,
                ),
            ]:
                transitions = factory(motion, field)
                rates = transitions.nuclear_cw_rates()
                reference = motion.relaxation_rates(field)
                assert_allclose(
                    rates.self_relaxation_per_second,
                    reference.self_relaxation_per_second,
                    rtol=1e-15,
                )
                assert_allclose(
                    rates.cross_relaxation_per_second,
                    reference.cross_relaxation_per_second,
                    rtol=1e-15,
                )
        with self.assertRaises(ValueError):
            ElectronNuclearTransitionRates.from_rotational_model(
                RotationalDipolarModel(1e-9, 1e-11, 0.5), 0.35
            )

    @unittest.skipUnless(HAS_SCIPY, "propagation uses optional SciPy")
    def test_completely_positive_trace_preserving_semigroup(self):
        from scipy.linalg import expm

        model = replace(self.model, electron_t1_seconds=0.02, temperature_kelvin=0.2)
        generator = model.liouvillian(MicrowaveDrive(120, 30, 0.4))
        evolution = expm(generator * 0.01)
        choi = np.zeros((16, 16), dtype=complex)
        for row in range(4):
            for col in range(4):
                choi[4 * row : 4 * (row + 1), 4 * col : 4 * (col + 1)] = evolution[
                    :, row + 4 * col
                ].reshape((4, 4), order="F")
        assert_allclose(choi, choi.conj().T, atol=2e-14)
        self.assertGreater(np.linalg.eigvalsh(choi).min(), -2e-13)
        assert_allclose(
            np.trace(choi.reshape(4, 4, 4, 4), axis1=1, axis2=3), np.eye(4), atol=2e-13
        )
        cold = model.steady_state(MicrowaveDrive(2e4))
        self.assertGreater(np.linalg.eigvalsh(cold.density).min(), -1e-12)
        self.assertLessEqual(abs(cold.nuclear_polarization), 1)

    def test_microwave_off_retains_frame_reference(self):
        drive = MicrowaveDrive(300, 123, 0.7)
        off = drive.off()
        self.assertEqual(off.nutation_rad_per_s, 0)
        self.assertEqual(off.detuning_rad_per_s, drive.detuning_rad_per_s)
        self.assertEqual(off.phase_rad, drive.phase_rad)

    def test_nonunique_stationary_state_and_invalid_physical_inputs(self):
        disconnected = CoupledElectronNuclearModel(0.35, nuclear_t1_seconds=np.inf)
        with self.assertRaises(ValueError):
            disconnected.steady_state()
        for kwargs in [
            dict(field_tesla=0),
            dict(temperature_kelvin=-1),
            dict(electron_t1_seconds=0),
            dict(electron_dephasing_per_second=-1),
            dict(gamma_n_hz_per_t=-1),
            dict(transitions="rates"),
        ]:
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises((ValueError, TypeError)),
            ):
                CoupledElectronNuclearModel(**(dict(field_tesla=0.35) | kwargs))
        with self.assertRaises(ValueError):
            ElectronNuclearTransitionRates(zero_quantum_per_second=-1)
        with self.assertRaises(ValueError):
            MicrowaveDrive(-1)

    @unittest.skipUnless(HAS_SCIPY, "propagation uses optional SciPy")
    def test_invalid_states_and_time_axes_are_rejected(self):
        nonhermitian = np.eye(4) / 4
        nonhermitian[0, 1] = 0.1
        for density in [
            np.eye(4),
            np.diag([1.1, -0.1, 0, 0]),
            np.eye(2),
            np.eye(4) / 4 + np.diag([np.nan, 0, 0, 0]),
            nonhermitian,
        ]:
            with self.assertRaises(ValueError):
                self.model.simulate([0, 1], initial_density=density)
        for times in [[], [1, 0], [-1], [0, np.nan]]:
            with self.assertRaises(ValueError):
                self.model.simulate(times)

    def assertPhysical(self, densities):
        assert_allclose(densities, densities.conj().transpose(0, 2, 1), atol=5e-11)
        assert_allclose(np.trace(densities, axis1=1, axis2=2), 1, atol=5e-10, rtol=0)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(densities).min()), -5e-11)


if __name__ == "__main__":
    unittest.main()
