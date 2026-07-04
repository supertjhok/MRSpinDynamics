from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.relaxation import (
    BOLTZMANN,
    BPP_T1_MINIMUM_OMEGA_TAU,
    BPPRelaxationModel,
    PrepolarizedT1rhoExperiment,
    WallCollisionRelaxationModel,
    apply_relaxation_to_parameters,
    arrhenius_correlation_time,
    bpp_relaxation_rates,
    gas_mean_speed_m_per_s,
    liouville_superoperator,
    matrix_exponential,
    simulate_prepolarized_t1rho,
    single_spin_matrices,
    sphere_surface_to_volume_per_m,
    spectral_density_lorentzian,
    stokes_einstein_debye_correlation_time,
    t1rho_relaxation_rate,
    tau_c_from_t1_minimum,
    wall_collision_rate_per_second,
)


class RelaxationTests(unittest.TestCase):
    def test_lorentzian_spectral_density_has_expected_limits(self) -> None:
        tau = 2.0e-9

        j0 = spectral_density_lorentzian(0.0, tau)
        j_corner = spectral_density_lorentzian(1.0 / tau, tau)

        self.assertAlmostEqual(float(j0), 2.0 * tau)
        self.assertAlmostEqual(float(j_corner), tau)

    def test_arrhenius_correlation_time_matches_reference_and_temperature_trend(self) -> None:
        tau = arrhenius_correlation_time(
            [280.0, 300.0, 330.0],
            tau_ref_seconds=1.0e-9,
            reference_temperature_kelvin=300.0,
            activation_energy_j_per_mol=12_000.0,
        )

        self.assertAlmostEqual(float(tau[1]), 1.0e-9)
        self.assertGreater(tau[0], tau[1])
        self.assertGreater(tau[1], tau[2])

    def test_stokes_einstein_debye_matches_closed_form_and_scaling(self) -> None:
        radius = 1.45e-10
        viscosity = 0.89e-3
        temperature = 298.15

        tau = stokes_einstein_debye_correlation_time(radius, viscosity, temperature)
        expected = (
            4.0 * np.pi * viscosity * radius**3 / (3.0 * BOLTZMANN * temperature)
        )
        self.assertAlmostEqual(float(tau), expected)
        # Water rank-2 reorientation is a couple of picoseconds.
        self.assertLess(1.0e-12, float(tau))
        self.assertLess(float(tau), 5.0e-12)

        # tau scales linearly with the slip factor and with viscosity.
        half = stokes_einstein_debye_correlation_time(
            radius, viscosity, temperature, slip_factor=0.5
        )
        self.assertAlmostEqual(float(half), 0.5 * float(tau))
        sweep = stokes_einstein_debye_correlation_time(
            radius, np.array([viscosity, 2.0 * viscosity]), temperature
        )
        self.assertAlmostEqual(float(sweep[1]), 2.0 * float(sweep[0]))

    def test_stokes_einstein_debye_rejects_nonpositive_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "hydrodynamic_radius_m"):
            stokes_einstein_debye_correlation_time(0.0, 1.0e-3, 300.0)
        with self.assertRaisesRegex(ValueError, "viscosity_pa_s"):
            stokes_einstein_debye_correlation_time(1.0e-10, -1.0, 300.0)
        with self.assertRaisesRegex(ValueError, "slip_factor"):
            stokes_einstein_debye_correlation_time(
                1.0e-10, 1.0e-3, 300.0, slip_factor=0.0
            )

    def test_tau_c_from_t1_minimum_locates_the_r1_maximum(self) -> None:
        self.assertAlmostEqual(BPP_T1_MINIMUM_OMEGA_TAU, 0.6158, places=3)

        omega = 2.0 * np.pi * 20.0e6
        tau_min = tau_c_from_t1_minimum(omega)
        self.assertAlmostEqual(omega * tau_min, BPP_T1_MINIMUM_OMEGA_TAU)

        # T1 is minimal (R1 maximal) at the returned correlation time.
        neighbors = tau_min * np.array([0.5, 1.0, 2.0])
        r1 = bpp_relaxation_rates(
            angular_frequency_rad_per_s=omega,
            correlation_time_seconds=neighbors,
            coupling_scale_per_second2=1.0e10,
        ).r1_per_second
        self.assertGreater(r1[1], r1[0])
        self.assertGreater(r1[1], r1[2])

    def test_tau_c_from_t1_minimum_rejects_invalid_frequency(self) -> None:
        with self.assertRaisesRegex(ValueError, "angular_frequency_rad_per_s"):
            tau_c_from_t1_minimum(0.0)

    def test_bpp_rates_use_j0_jw_and_j2w_coefficients(self) -> None:
        omega = 4.0e6
        tau = 3.0e-7
        scale = 2.0e12
        rates = bpp_relaxation_rates(
            angular_frequency_rad_per_s=omega,
            correlation_time_seconds=tau,
            coupling_scale_per_second2=scale,
            r1_coefficients=(1.0, 2.0, 3.0),
            r2_coefficients=(4.0, 5.0, 6.0),
            baseline_r1_per_second=0.5,
            baseline_r2_per_second=0.25,
        )

        j0 = spectral_density_lorentzian(0.0, tau)
        jw = spectral_density_lorentzian(omega, tau)
        j2w = spectral_density_lorentzian(2.0 * omega, tau)
        expected_r1 = scale * (j0 + 2.0 * jw + 3.0 * j2w) + 0.5
        expected_r2 = scale * (4.0 * j0 + 5.0 * jw + 6.0 * j2w) + 0.25
        self.assertAlmostEqual(float(rates.r1_per_second), float(expected_r1))
        self.assertAlmostEqual(float(rates.r2_per_second), float(expected_r2))
        self.assertAlmostEqual(float(rates.t1_seconds), 1.0 / float(expected_r1))
        self.assertAlmostEqual(float(rates.t2_seconds), 1.0 / float(expected_r2))

    def test_bpp_model_broadcasts_temperature_and_frequency_arrays(self) -> None:
        model = BPPRelaxationModel(
            angular_frequency_rad_per_s=np.array([1.0e6, 2.0e6]),
            tau_ref_seconds=1.0e-8,
            coupling_scale_per_second2=1.0e10,
            reference_temperature_kelvin=300.0,
            activation_energy_j_per_mol=8_000.0,
        )

        rates = model.rates(np.array([290.0, 320.0]))

        self.assertEqual(rates.t1_seconds.shape, (2,))
        self.assertEqual(rates.t2_seconds.shape, (2,))
        self.assertGreater(rates.correlation_time_seconds[0], rates.correlation_time_seconds[1])
        self.assertTrue(np.all(rates.r1_per_second > 0.0))
        self.assertTrue(np.all(rates.r2_per_second > 0.0))

    def test_zero_coupling_and_baseline_rates_are_supported(self) -> None:
        rates = bpp_relaxation_rates(
            angular_frequency_rad_per_s=1.0,
            correlation_time_seconds=1.0e-9,
            coupling_scale_per_second2=0.0,
        )
        self.assertTrue(np.isinf(float(rates.t1_seconds)))
        self.assertTrue(np.isinf(float(rates.t2_seconds)))

        baseline = bpp_relaxation_rates(
            angular_frequency_rad_per_s=1.0,
            correlation_time_seconds=1.0e-9,
            coupling_scale_per_second2=0.0,
            baseline_r1_per_second=2.0,
            baseline_r2_per_second=4.0,
        )
        self.assertAlmostEqual(float(baseline.t1_seconds), 0.5)
        self.assertAlmostEqual(float(baseline.t2_seconds), 0.25)

    def test_rates_return_workflow_parameter_fields(self) -> None:
        rates = bpp_relaxation_rates(
            angular_frequency_rad_per_s=[1.0, 2.0],
            correlation_time_seconds=[1.0e-8, 2.0e-8],
            coupling_scale_per_second2=1.0e9,
        )

        params = rates.as_parameters()
        np.testing.assert_allclose(params["T1"], rates.t1_seconds)
        np.testing.assert_allclose(params["T2"], rates.t2_seconds)

    def test_apply_relaxation_to_mapping_or_dataclass(self) -> None:
        rates = bpp_relaxation_rates(
            angular_frequency_rad_per_s=[1.0, 2.0],
            correlation_time_seconds=[1.0e-8, 2.0e-8],
            coupling_scale_per_second2=1.0e9,
        )

        mapped = apply_relaxation_to_parameters({"T1": 1.0, "other": 3}, rates)
        self.assertEqual(mapped["other"], 3)
        np.testing.assert_allclose(mapped["T2"], rates.t2_seconds)

        @dataclass(frozen=True)
        class Params:
            del_w: np.ndarray
            T1: float
            T2: float

        copied = apply_relaxation_to_parameters(
            Params(del_w=np.array([0.0, 1.0]), T1=1.0, T2=1.0),
            rates,
        )
        np.testing.assert_allclose(copied["del_w"], [0.0, 1.0])
        np.testing.assert_allclose(copied["T1"], rates.t1_seconds)

    def test_wall_collision_rate_uses_kinetic_surface_sampling(self) -> None:
        mean_speed = float(gas_mean_speed_m_per_s(300.0, 128.9047808611))
        surface_to_volume = sphere_surface_to_volume_per_m([0.002, 0.010])

        rates = wall_collision_rate_per_second(
            surface_to_volume,
            temperature_kelvin=300.0,
            mass_amu=128.9047808611,
            accommodation_probability=0.5,
        )

        np.testing.assert_allclose(rates, 0.5 * mean_speed * surface_to_volume / 4.0)
        self.assertGreater(rates[0], rates[1])

    def test_wall_collision_liouvillian_damps_traceless_spin_modes(self) -> None:
        ops = single_spin_matrices(0.5)
        model = WallCollisionRelaxationModel(
            spin=0.5,
            collision_rate_per_second=2.0e4,
            depolarization_probability=1.5e-8,
        )
        generator = liouville_superoperator(np.zeros((2, 2)), model)

        duration = 1000.0
        propagated_identity = matrix_exponential(
            generator,
            duration,
        ) @ ops.identity.reshape(-1, order="F")
        propagated_ix = matrix_exponential(generator, duration) @ ops.ix.reshape(
            -1,
            order="F",
        )

        np.testing.assert_allclose(
            propagated_identity.reshape((2, 2), order="F"),
            ops.identity,
            atol=1.0e-12,
        )
        expected = np.exp(-model.relaxation_rate_per_second * duration) * ops.ix
        np.testing.assert_allclose(
            propagated_ix.reshape((2, 2), order="F"),
            expected,
            rtol=1.0e-9,
            atol=1.0e-12,
        )

    def test_t1rho_weak_lock_limit_equals_transverse_rate(self) -> None:
        omega0 = 2.0 * np.pi * 20.0e6
        tau = 8.0e-9
        scale = 5.0e7

        r1rho_zero = t1rho_relaxation_rate(
            0.0, omega0, tau, coupling_scale_per_second2=scale
        )
        r2 = bpp_relaxation_rates(
            angular_frequency_rad_per_s=omega0,
            correlation_time_seconds=tau,
            coupling_scale_per_second2=scale,
        ).r2_per_second
        # On-resonance spin lock at vanishing B1 reduces to transverse relaxation
        # (Stanford RAD226b Lecture 11, slide "T1rho versus T2").
        self.assertAlmostEqual(float(r1rho_zero), float(r2))

    def test_t1rho_matches_closed_form_and_probes_two_omega1(self) -> None:
        omega0 = 2.0 * np.pi * 30.0e6
        omega1 = 2.0 * np.pi * 500.0
        tau = 3.0e-4
        scale = 4.0e4

        rate = t1rho_relaxation_rate(
            omega1,
            omega0,
            tau,
            coupling_scale_per_second2=scale,
            coefficients=(1.5, 2.5, 1.0),
        )
        j_lock = spectral_density_lorentzian(2.0 * omega1, tau)
        j_w0 = spectral_density_lorentzian(omega0, tau)
        j_2w0 = spectral_density_lorentzian(2.0 * omega0, tau)
        expected = scale * (1.5 * j_lock + 2.5 * j_w0 + j_2w0)
        self.assertAlmostEqual(float(rate), float(expected))

        # lock_harmonic selects the spectral-density argument of the lock term.
        rate_1w = t1rho_relaxation_rate(
            omega1, omega0, tau, coupling_scale_per_second2=scale, lock_harmonic=1.0
        )
        expected_1w = scale * (
            1.5 * spectral_density_lorentzian(omega1, tau) + 2.5 * j_w0 + j_2w0
        )
        self.assertAlmostEqual(float(rate_1w), float(expected_1w))

    def test_t1rho_disperses_downward_only_for_slow_tumblers(self) -> None:
        omega0 = 2.0 * np.pi * 20.0e6
        scale = 2.0e6
        lock_hz = np.geomspace(20.0, 20_000.0, 32)
        omega1 = 2.0 * np.pi * lock_hz

        # Slow tumbler: strong monotone dispersion across the kHz spin-lock band.
        slow = t1rho_relaxation_rate(
            omega1, omega0, 3.0e-4, coupling_scale_per_second2=scale
        )
        self.assertTrue(np.all(np.diff(slow) <= 1.0e-9))
        self.assertGreater(slow[0] / slow[-1], 10.0)

        # Fast tumbler: essentially flat, no dispersion in the same band.
        fast = t1rho_relaxation_rate(
            omega1, omega0, 1.0e-11, coupling_scale_per_second2=scale
        )
        self.assertLess(abs(fast[0] - fast[-1]) / fast[0], 1.0e-3)

    def test_t1rho_low_lock_rate_grows_with_molecule_size(self) -> None:
        omega0 = 2.0 * np.pi * 20.0e6
        radii = np.array([0.5e-9, 6.0e-9, 20.0e-9])
        tau = stokes_einstein_debye_correlation_time(radii, 0.1, 310.0)

        rate = t1rho_relaxation_rate(
            2.0 * np.pi * 100.0, omega0, tau, coupling_scale_per_second2=2.0e6
        )
        # Larger hydrodynamic radius -> longer tau_c -> larger R1rho at weak lock.
        self.assertTrue(np.all(np.diff(rate) > 0.0))

    def test_simulate_prepolarized_t1rho_builds_size_by_lock_grid(self) -> None:
        omega0 = 2.0 * np.pi * 20.0e6
        tau = np.array([1.0e-8, 2.0e-5, 5.0e-4])
        lock_hz = np.geomspace(20.0, 20_000.0, 12)
        t1 = np.array([1.0, 0.2, 5.0])

        experiment = simulate_prepolarized_t1rho(
            spin_lock_angular_rad_per_s=2.0 * np.pi * lock_hz,
            larmor_angular_rad_per_s=omega0,
            correlation_time_seconds=tau,
            spin_lock_time_seconds=0.03,
            coupling_scale_per_second2=2.0e6,
            polarizing_field_tesla=0.3,
            detection_field_tesla=0.02,
            prepolarization_time_seconds=3.0,
            t1_seconds=t1,
        )

        self.assertIsInstance(experiment, PrepolarizedT1rhoExperiment)
        self.assertEqual(experiment.r1rho_per_second.shape, (3, 12))
        self.assertEqual(experiment.locked_signal.shape, (3, 12))
        self.assertEqual(experiment.prepared_m0.shape, (3,))

        # Locked signal is prepared magnetization damped by exp(-TSL * R1rho).
        expected = experiment.prepared_m0[:, np.newaxis] * np.exp(
            -experiment.spin_lock_time_seconds * experiment.r1rho_per_second
        )
        np.testing.assert_allclose(experiment.locked_signal, expected)
        np.testing.assert_allclose(
            experiment.t1rho_seconds, 1.0 / experiment.r1rho_per_second
        )
        # A stronger polarizing field lifts the prepared magnetization above the
        # detection-field thermal value for the fast, short-T1 pool.
        self.assertGreater(experiment.prepared_m0[0], 1.0)
        # The slow pool disperses: strong-lock survival exceeds weak-lock survival.
        fraction = np.exp(
            -experiment.spin_lock_time_seconds * experiment.r1rho_per_second
        )
        self.assertGreater(fraction[2, -1], fraction[2, 0])

    def test_t1rho_rejects_invalid_inputs(self) -> None:
        omega0 = 2.0 * np.pi * 20.0e6
        with self.assertRaisesRegex(ValueError, "larmor_angular_rad_per_s"):
            t1rho_relaxation_rate(100.0, 0.0, 1.0e-6)
        with self.assertRaisesRegex(ValueError, "spin_lock_angular_rad_per_s"):
            t1rho_relaxation_rate(-1.0, omega0, 1.0e-6)
        with self.assertRaisesRegex(ValueError, "coupling_scale_per_second2"):
            t1rho_relaxation_rate(
                100.0, omega0, 1.0e-6, coupling_scale_per_second2=-1.0
            )
        with self.assertRaisesRegex(ValueError, "lock_harmonic"):
            t1rho_relaxation_rate(100.0, omega0, 1.0e-6, lock_harmonic=0.0)

    def test_invalid_inputs_raise_clear_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "correlation_time_seconds"):
            spectral_density_lorentzian(1.0, 0.0)
        with self.assertRaisesRegex(ValueError, "temperature_kelvin"):
            arrhenius_correlation_time(-1.0, tau_ref_seconds=1.0)
        with self.assertRaisesRegex(ValueError, "r1_coefficients"):
            bpp_relaxation_rates(
                angular_frequency_rad_per_s=1.0,
                correlation_time_seconds=1.0,
                r1_coefficients=(1.0, 2.0),
            )


if __name__ == "__main__":
    unittest.main()
