"""Validation of field-dependent Gibbs equilibrium and relaxation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    FieldDependentDaviesRelaxationModel,
    FieldDependentNonsecularRelaxationModel,
    FieldDependentRelaxationModel,
    QuadrupolarSite,
    field_dependent_equilibrium,
    nqr_hamiltonian,
    simulate_field_relaxation,
    simulate_lab_frame_rf,
    simulate_crossover_slse,
)


class FieldEquilibriumTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.17e6,
        )

    def test_zero_field_kramers_partners_have_equal_populations(self) -> None:
        result = field_dependent_equilibrium(
            self.site,
            temperature_kelvin=5.0e-5,
        )
        self.assertAlmostEqual(result.populations[0], result.populations[1])
        self.assertAlmostEqual(result.populations[2], result.populations[3])
        self.assertAlmostEqual(float(np.trace(result.density_matrix_pas).real), 1.0)
        np.testing.assert_allclose(
            result.density_matrix_pas,
            result.density_matrix_pas.conj().T,
            atol=1.0e-14,
        )

    def test_field_reversal_reverses_equilibrium_spin(self) -> None:
        positive = field_dependent_equilibrium(
            self.site,
            (0.0, 0.0, 0.2),
            temperature_kelvin=1.0e-4,
        )
        negative = field_dependent_equilibrium(
            self.site,
            (0.0, 0.0, -0.2),
            temperature_kelvin=1.0e-4,
        )
        np.testing.assert_allclose(
            positive.spin_expectation_pas,
            -negative.spin_expectation_pas,
            atol=1.0e-12,
        )
        self.assertGreater(positive.spin_expectation_pas[2], 0.0)

    def test_populations_obey_boltzmann_detailed_balance(self) -> None:
        temperature = 8.0e-5
        result = field_dependent_equilibrium(
            self.site,
            (0.1, 0.05, 0.2),
            temperature_kelvin=temperature,
        )
        planck = 6.62607015e-34
        boltzmann = 1.380649e-23
        gap_hz = result.levels_hz[-1] - result.levels_hz[0]
        expected = np.exp(-planck * gap_hz / (boltzmann * temperature))
        self.assertAlmostEqual(
            result.populations[-1] / result.populations[0],
            expected,
            places=13,
        )


class FieldDependentRelaxationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.0,
            isotope="14N",
            quadrupole_frequency_hz=900.0e3,
            eta=0.3,
            gamma_hz_per_t=3.0766e6,
        )
        self.b0 = np.array([0.06, 0.03, 0.08])
        self.model = FieldDependentRelaxationModel(
            temperature_kelvin=6.0e-5,
            thermalization_time_seconds=2.0e-3,
            dephasing_time_seconds=0.8e-3,
        )

    def test_gibbs_state_is_stationary_and_generator_preserves_trace(self) -> None:
        hamiltonian = nqr_hamiltonian(self.site, self.b0)
        equilibrium = self.model.equilibrium_density(hamiltonian)
        generator = self.model.superoperator(hamiltonian)
        stationary = generator @ equilibrium.reshape(-1, order="F")
        trace_vector = np.eye(self.site.dimension).reshape(-1, order="F")
        np.testing.assert_allclose(stationary, 0.0, atol=1.0e-11)
        np.testing.assert_allclose(trace_vector @ generator, 0.0, atol=1.0e-11)

    def test_pure_thermalization_has_closed_form(self) -> None:
        model = FieldDependentRelaxationModel(
            temperature_kelvin=6.0e-5,
            thermalization_time_seconds=2.0e-3,
        )
        initial = np.eye(self.site.dimension) / self.site.dimension
        times = np.array([0.0, 1.0e-3, 4.0e-3])
        result = simulate_field_relaxation(
            self.site,
            self.b0,
            times,
            relaxation=model,
            initial_density=initial,
        )
        equilibrium = result.equilibrium.density_matrix_pas
        for time, density in zip(times, result.density_matrices_pas):
            expected = equilibrium + (initial - equilibrium) * np.exp(
                -time / model.thermalization_time_seconds
            )
            np.testing.assert_allclose(density, expected, atol=2.0e-12)

    def test_relaxation_preserves_physical_density_matrix(self) -> None:
        state = np.ones(self.site.dimension, dtype=np.complex128)
        state /= np.linalg.norm(state)
        initial = np.outer(state, state.conj())
        result = simulate_field_relaxation(
            self.site,
            self.b0,
            np.linspace(0.0, 8.0e-3, 9),
            relaxation=self.model,
            initial_density=initial,
        )
        for density in result.density_matrices_pas:
            self.assertAlmostEqual(float(np.trace(density).real), 1.0, places=11)
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(density))), -1e-11)
        final_error = np.linalg.norm(
            result.density_matrices_pas[-1]
            - result.equilibrium.density_matrix_pas
        )
        initial_error = np.linalg.norm(
            initial - result.equilibrium.density_matrix_pas
        )
        self.assertLess(final_error, 0.03 * initial_error)

    def test_lab_frame_uses_static_field_as_the_thermal_reference(self) -> None:
        initial = np.eye(self.site.dimension) / self.site.dimension
        result = simulate_lab_frame_rf(
            self.site,
            self.b0,
            [10.0e-3],
            [[0.0, 0.0, 0.0]],
            initial_density=initial,
            relaxation=self.model,
        )
        expected = field_dependent_equilibrium(
            self.site,
            self.b0,
            temperature_kelvin=self.model.temperature_kelvin,
        ).density_matrix_pas
        self.assertLess(
            np.linalg.norm(result.density_matrices[-1] - expected),
            0.01 * np.linalg.norm(initial - expected),
        )


class FieldDependentDaviesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.17e6,
        )

    def test_finite_temperature_gibbs_state_is_stationary(self) -> None:
        model = FieldDependentDaviesRelaxationModel(
            spin=self.site.spin,
            temperature_kelvin=8.0e-5,
            magnetic_rate_per_second=100.0,
            efg_rate_per_second=30.0,
            correlation_time_seconds=2.0e-7,
            secular_tolerance_hz=1.0e-4,
        )
        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        field = self.site.quadrupole_frequency_hz / self.site.gamma_hz_per_t
        hamiltonian = nqr_hamiltonian(self.site, field * direction)
        generator = model.superoperator(hamiltonian)
        equilibrium = model.equilibrium_density(hamiltonian)
        trace_vector = np.eye(self.site.dimension).reshape(-1, order="F")
        np.testing.assert_allclose(
            generator @ equilibrium.reshape(-1, order="F"),
            0.0,
            atol=2.0e-11,
        )
        np.testing.assert_allclose(trace_vector @ generator, 0.0, atol=2.0e-11)
        self.assertLess(float(np.max(np.real(np.linalg.eigvals(generator)))), 1.0e-10)

    def test_lorentzian_bath_suppresses_high_field_transitions(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.17e6,
        )
        model = FieldDependentDaviesRelaxationModel(
            spin=site.spin,
            temperature_kelvin=300.0,
            magnetic_rate_per_second=100.0,
            correlation_time_seconds=1.0e-6,
        )

        def excited_state_decay(interaction_ratio: float) -> float:
            field = interaction_ratio * site.quadrupole_frequency_hz / (
                site.gamma_hz_per_t
            )
            hamiltonian = nqr_hamiltonian(site, (0.0, 0.0, field))
            _, vectors = np.linalg.eigh(hamiltonian)
            excited = np.outer(vectors[:, -1], vectors[:, -1].conj())
            derivative = model.superoperator(hamiltonian) @ excited.reshape(
                -1, order="F"
            )
            return float(np.linalg.norm(derivative))

        self.assertLess(
            excited_state_decay(10.0),
            0.1 * excited_state_decay(0.01),
        )


class FieldDependentNonsecularTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(
            spin=1.0,
            quadrupole_frequency_hz=1.0e6,
            eta=0.3,
            gamma_hz_per_t=3.0766e6,
        )
        direction = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        self.b0 = (
            self.site.quadrupole_frequency_hz
            / self.site.gamma_hz_per_t
            * direction
        )
        self.hamiltonian = nqr_hamiltonian(self.site, self.b0)
        self.common = dict(
            spin=self.site.spin,
            temperature_kelvin=300.0,
            magnetic_rate_per_second=100.0,
            efg_rate_per_second=30.0,
            correlation_time_seconds=2.0e-7,
            secular_tolerance_hz=1.0e-3,
        )

    def test_zero_cluster_width_recovers_secular_davies_generator(self) -> None:
        secular = FieldDependentDaviesRelaxationModel(**self.common)
        unified = FieldDependentNonsecularRelaxationModel(
            **self.common,
            frequency_cluster_width_hz=self.common["secular_tolerance_hz"],
        )
        np.testing.assert_allclose(
            unified.superoperator(self.hamiltonian),
            secular.superoperator(self.hamiltonian),
            atol=1.0e-12,
        )

    def test_unresolved_frequency_cluster_retains_cross_terms_and_is_cp(self) -> None:
        secular = FieldDependentDaviesRelaxationModel(**self.common)
        unified = FieldDependentNonsecularRelaxationModel(
            **self.common,
            frequency_cluster_width_hz=200.0e3,
        )
        secular_generator = secular.superoperator(self.hamiltonian)
        unified_generator = unified.superoperator(self.hamiltonian)
        self.assertGreater(
            np.linalg.norm(unified_generator - secular_generator),
            1.0,
        )
        self.assertLess(
            float(np.max(np.real(np.linalg.eigvals(unified_generator)))),
            1.0e-10,
        )
        relative_gibbs_residual = unified.gibbs_stationarity_error(
            self.hamiltonian
        ) / np.linalg.norm(unified_generator)
        self.assertLess(relative_gibbs_residual, 1.0e-8)

        initial = np.zeros((self.site.dimension, self.site.dimension))
        initial[0, 0] = 1.0
        result = simulate_field_relaxation(
            self.site,
            self.b0,
            [0.0, 1.0e-3, 10.0e-3],
            relaxation=unified,
            initial_density=initial,
        )
        for density in result.density_matrices_pas:
            self.assertGreaterEqual(float(np.min(np.linalg.eigvalsh(density))), -1e-11)

    def test_exactly_degenerate_clusters_keep_the_low_temperature_gibbs_state(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.17e6,
        )
        model = FieldDependentNonsecularRelaxationModel(
            spin=site.spin,
            temperature_kelvin=8.0e-5,
            magnetic_rate_per_second=100.0,
            efg_rate_per_second=30.0,
            frequency_cluster_width_hz=100.0e3,
        )
        self.assertLess(
            model.gibbs_stationarity_error(nqr_hamiltonian(site)),
            1.0e-10,
        )

    def test_exact_pulse_slse_preserves_trace_with_nonsecular_free_decay(self) -> None:
        model = FieldDependentNonsecularRelaxationModel(
            **self.common,
            frequency_cluster_width_hz=200.0e3,
        )
        nutation = 0.02 * self.site.quadrupole_frequency_hz
        result = simulate_crossover_slse(
            self.site,
            (0.0, 0.0, 0.0),
            nutation_hz=nutation,
            excitation_duration_seconds=0.15 / nutation,
            refocus_duration_seconds=0.30 / nutation,
            echo_spacing_seconds=200.0e-6,
            num_echoes=4,
            relaxation=model,
            b1_direction_pas=(1.0, -1.0, 0.0),
            floquet_sidebands=5,
        )
        self.assertEqual(result.echo_amplitudes.shape, (4,))
        self.assertTrue(np.all(np.isfinite(result.echo_amplitudes)))
        self.assertLess(result.excitation_pulse_unitarity_error, 1.0e-6)
        self.assertLess(result.refocus_pulse_unitarity_error, 1.0e-6)
        np.testing.assert_allclose(
            np.trace(result.density_matrices_pas, axis1=1, axis2=2),
            1.0,
            atol=2.0e-7,
        )


if __name__ == "__main__":
    unittest.main()
