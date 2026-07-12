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
    OrientationSample,
    QuadrupolarSite,
    b0_b1_powder_average_halton,
    diagonalize_site,
    field_dependent_equilibrium,
    matched_filter_echo_waveforms,
    nqr_hamiltonian,
    powder_carrier_frequency_hz,
    powder_average_grid,
    select_powder_frequency_slice,
    simulate_field_relaxation,
    simulate_lab_frame_rf,
    simulate_crossover_slse,
    simulate_crossover_slse_powder,
    simulate_full_slse,
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
            acquisition_offsets_seconds=(-20.0e-6, 0.0, 20.0e-6),
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
        np.testing.assert_allclose(
            result.echo_waveforms[:, 1], result.echo_amplitudes, atol=1.0e-14
        )

    def test_powder_slse_uses_normalized_weights_and_one_global_carrier(self) -> None:
        model = FieldDependentNonsecularRelaxationModel(
            **self.common,
            frequency_cluster_width_hz=200.0e3,
        )
        nutation = 0.02 * self.site.quadrupole_frequency_hz
        field = self.site.quadrupole_frequency_hz / self.site.gamma_hz_per_t
        result = simulate_crossover_slse_powder(
            self.site,
            field,
            nutation_hz=nutation,
            excitation_duration_seconds=0.15 / nutation,
            refocus_duration_seconds=0.30 / nutation,
            echo_spacing_seconds=200.0e-6,
            num_echoes=3,
            relaxation=model,
            n_theta=2,
            n_phi=4,
            n_chi=2,
            floquet_sidebands=4,
            acquisition_duration_seconds=100.0e-6,
            acquisition_points=9,
            receiver_bandwidth_hz=50.0e3,
            orientations=b0_b1_powder_average_halton(16),
        )
        self.assertEqual(result.local_echo_amplitudes.shape, (16, 3))
        self.assertAlmostEqual(float(np.sum(result.orientation_weights)), 1.0)
        for sample in b0_b1_powder_average_halton(16):
            self.assertAlmostEqual(
                float(np.dot(sample.b0_direction_pas, sample.b1_direction_pas)),
                0.0,
                places=12,
            )
        self.assertTrue(np.all(np.isfinite(result.echo_amplitudes)))
        self.assertEqual(result.echo_waveforms.shape, (3, 9))
        self.assertAlmostEqual(result.matched_echo_amplitudes[0].real, 1.0)
        self.assertAlmostEqual(result.matched_echo_amplitudes[0].imag, 0.0)
        np.testing.assert_allclose(
            [local.rf_frequency_hz for local in result.local_results],
            result.rf_frequency_hz,
        )
        parallel = simulate_crossover_slse_powder(
            self.site,
            field,
            nutation_hz=nutation,
            excitation_duration_seconds=0.15 / nutation,
            refocus_duration_seconds=0.30 / nutation,
            echo_spacing_seconds=200.0e-6,
            num_echoes=3,
            relaxation=model,
            rf_frequency_hz=result.rf_frequency_hz,
            floquet_sidebands=4,
            acquisition_duration_seconds=100.0e-6,
            acquisition_points=9,
            receiver_bandwidth_hz=50.0e3,
            orientations=b0_b1_powder_average_halton(16),
            num_workers=2,
            parallel_backend="process",
            retain_local_results=False,
        )
        self.assertEqual(parallel.local_results, ())
        np.testing.assert_allclose(
            parallel.local_echo_amplitudes, result.local_echo_amplitudes
        )
        np.testing.assert_allclose(parallel.echo_waveforms, result.echo_waveforms)
        np.testing.assert_allclose(
            parallel.unfiltered_echo_waveforms,
            result.unfiltered_echo_waveforms,
        )
        np.testing.assert_allclose(
            parallel.matched_echo_amplitudes, result.matched_echo_amplitudes
        )
        np.testing.assert_allclose(
            parallel.prefix_matched_echo_amplitudes,
            result.prefix_matched_echo_amplitudes,
        )
        with self.assertRaisesRegex(ValueError, "sampling rate"):
            matched_filter_echo_waveforms(
                result.echo_waveforms,
                result.acquisition_offsets_seconds,
                receiver_bandwidth_hz=1.0e9,
            )

    def test_frequency_slice_reports_retained_full_powder_weight(self) -> None:
        orientations = b0_b1_powder_average_halton(64)
        field = self.site.quadrupole_frequency_hz / self.site.gamma_hz_per_t
        carrier = powder_carrier_frequency_hz(
            self.site,
            field,
            orientations,
            nutation_hz=0.1 * self.site.quadrupole_frequency_hz,
        )
        selected, retained = select_powder_frequency_slice(
            self.site,
            field,
            orientations,
            carrier_frequency_hz=carrier,
            half_width_hz=0.2 * self.site.quadrupole_frequency_hz,
        )
        self.assertGreater(len(selected), 0)
        self.assertGreater(retained, 0.0)
        self.assertLessEqual(retained, 1.0)
        self.assertAlmostEqual(sum(sample.weight for sample in selected), 1.0)

    def test_matched_powder_decay_is_stable_to_receiver_grid_and_slice(self) -> None:
        model = FieldDependentNonsecularRelaxationModel(
            **self.common,
            frequency_cluster_width_hz=20.0e3,
        )
        field = self.site.quadrupole_frequency_hz / self.site.gamma_hz_per_t
        candidates = b0_b1_powder_average_halton(256)
        carrier = powder_carrier_frequency_hz(
            self.site,
            field,
            candidates,
            nutation_hz=100.0e3,
        )
        nominal_90 = 1.0 / (4.0 * np.sqrt(2.0) * 100.0e3)
        common = dict(
            nutation_hz=100.0e3,
            excitation_duration_seconds=nominal_90,
            refocus_duration_seconds=2.0 * nominal_90,
            echo_spacing_seconds=2.0e-3,
            num_echoes=4,
            relaxation=model,
            rf_frequency_hz=carrier,
            pulse_model="rwa",
            acquisition_duration_seconds=100.0e-6,
            receiver_bandwidth_hz=200.0e3,
            retain_local_results=False,
        )
        selected, _ = select_powder_frequency_slice(
            self.site,
            field,
            candidates,
            carrier_frequency_hz=carrier,
            half_width_hz=300.0e3,
        )
        coarse = simulate_crossover_slse_powder(
            self.site,
            field,
            orientations=selected,
            acquisition_points=129,
            **common,
        )
        fine = simulate_crossover_slse_powder(
            self.site,
            field,
            orientations=selected,
            acquisition_points=257,
            **common,
        )
        time_grid_error = np.linalg.norm(
            fine.matched_echo_amplitudes - coarse.matched_echo_amplitudes
        ) / np.linalg.norm(fine.matched_echo_amplitudes)
        self.assertLess(float(time_grid_error), 0.03)

        _, narrow_receiver = matched_filter_echo_waveforms(
            fine.unfiltered_echo_waveforms,
            fine.acquisition_offsets_seconds,
            receiver_bandwidth_hz=150.0e3,
        )
        _, wide_receiver = matched_filter_echo_waveforms(
            fine.unfiltered_echo_waveforms,
            fine.acquisition_offsets_seconds,
            receiver_bandwidth_hz=250.0e3,
        )
        bandwidth_error = np.linalg.norm(
            wide_receiver - narrow_receiver
        ) / np.linalg.norm(fine.matched_echo_amplitudes)
        self.assertLess(float(bandwidth_error), 0.05)

        slice_shapes = []
        for half_width in (250.0e3, 350.0e3):
            sliced, _ = select_powder_frequency_slice(
                self.site,
                field,
                candidates,
                carrier_frequency_hz=carrier,
                half_width_hz=half_width,
            )
            result = simulate_crossover_slse_powder(
                self.site,
                field,
                orientations=sliced,
                acquisition_points=65,
                **common,
            )
            slice_shapes.append(result.matched_echo_amplitudes)
        slice_error = np.linalg.norm(slice_shapes[1] - slice_shapes[0]) / np.linalg.norm(
            slice_shapes[1]
        )
        self.assertLess(float(slice_error), 0.05)

    def test_zero_field_exact_slse_approaches_rwa_at_noninteger_carrier_phase(self) -> None:
        model = FieldDependentDaviesRelaxationModel(
            spin=self.site.spin,
            temperature_kelvin=300.0,
        )
        carrier = self.site.quadrupole_frequency_hz
        nutation = 0.001 * carrier
        pulse_duration = 0.02 / nutation
        directions = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
        )
        for direction in directions:
            exact = simulate_crossover_slse(
                self.site,
                (0.0, 0.0, 0.0),
                nutation_hz=nutation,
                excitation_duration_seconds=pulse_duration,
                refocus_duration_seconds=pulse_duration,
                echo_spacing_seconds=253.7e-6,
                num_echoes=5,
                relaxation=model,
                rf_frequency_hz=carrier,
                b1_direction_pas=direction,
                floquet_sidebands=5,
            )
            rwa = simulate_full_slse(
                self.site,
                nutation_hz=nutation,
                excitation_duration_seconds=pulse_duration,
                refocus_duration_seconds=pulse_duration,
                echo_spacing_seconds=253.7e-6,
                num_echoes=5,
                rf_frequency_hz=carrier,
                orientations=[OrientationSample(direction)],
            )
            rwa_backend = simulate_crossover_slse(
                self.site,
                (0.0, 0.0, 0.0),
                nutation_hz=nutation,
                excitation_duration_seconds=pulse_duration,
                refocus_duration_seconds=pulse_duration,
                echo_spacing_seconds=253.7e-6,
                num_echoes=5,
                relaxation=model,
                rf_frequency_hz=carrier,
                b1_direction_pas=direction,
                pulse_model="rwa",
            )
            rwa_scale = np.vdot(
                rwa.echo_amplitudes, rwa_backend.echo_amplitudes
            ) / np.vdot(rwa.echo_amplitudes, rwa.echo_amplitudes)
            np.testing.assert_allclose(
                rwa_backend.echo_amplitudes,
                rwa_scale * rwa.echo_amplitudes,
                rtol=1.0e-5,
                atol=1.0e-5 * np.max(np.abs(rwa_backend.echo_amplitudes)),
            )
            scale = np.vdot(rwa.echo_amplitudes, exact.echo_amplitudes) / np.vdot(
                rwa.echo_amplitudes, rwa.echo_amplitudes
            )
            np.testing.assert_allclose(
                exact.echo_amplitudes,
                scale * rwa.echo_amplitudes,
                rtol=6.0e-2,
                atol=6.0e-2 * np.max(np.abs(exact.echo_amplitudes)),
                err_msg=f"direction={direction}",
            )

    def test_zero_field_powder_slse_matches_established_spherical_average(self) -> None:
        model = FieldDependentDaviesRelaxationModel(
            spin=self.site.spin,
            temperature_kelvin=300.0,
        )
        carrier = self.site.quadrupole_frequency_hz
        nutation = 0.01 * carrier
        pulse_duration = 0.2 / nutation
        sequence = dict(
            nutation_hz=nutation,
            excitation_duration_seconds=pulse_duration,
            refocus_duration_seconds=pulse_duration,
            echo_spacing_seconds=250.0e-6,
            num_echoes=5,
        )
        exact = simulate_crossover_slse_powder(
            self.site,
            0.0,
            relaxation=model,
            n_theta=4,
            n_phi=8,
            n_chi=3,
            floquet_sidebands=5,
            **sequence,
        )
        rwa = simulate_full_slse(
            self.site,
            orientations=powder_average_grid(4, 8),
            rf_frequency_hz=exact.rf_frequency_hz,
            **sequence,
        )
        self.assertEqual(exact.local_echo_amplitudes.shape, (32, 5))
        scale = np.vdot(rwa.echo_amplitudes, exact.echo_amplitudes) / np.vdot(
            rwa.echo_amplitudes, rwa.echo_amplitudes
        )
        np.testing.assert_allclose(
            exact.echo_amplitudes,
            scale * rwa.echo_amplitudes,
            rtol=2.0e-2,
            atol=2.0e-2 * np.max(np.abs(exact.echo_amplitudes)),
        )

    def test_nano2_zero_field_powder_decay_is_converged_and_exponential(self) -> None:
        site = QuadrupolarSite(
            spin=1.0,
            isotope="14N",
            quadrupole_frequency_hz=3.7755e6,
            eta=0.1119,
            gamma_hz_per_t=3.0766e6,
        )
        model = FieldDependentNonsecularRelaxationModel(
            spin=site.spin,
            temperature_kelvin=300.0,
            magnetic_rate_per_second=3.48,
            efg_rate_per_second=0.87,
            correlation_time_seconds=0.2e-6,
            frequency_cluster_width_hz=20.0e3,
        )
        sequence = dict(
            nutation_hz=3.3056e3,
            excitation_duration_seconds=50.0e-6,
            refocus_duration_seconds=50.0e-6,
            echo_spacing_seconds=2.0e-3,
            num_echoes=24,
            relaxation=model,
            rf_frequency_hz=diagonalize_site(site).transition("y").frequency_hz,
            floquet_sidebands=5,
        )
        coarse = simulate_crossover_slse_powder(
            site, 0.0, n_theta=8, n_phi=16, n_chi=4, **sequence
        )
        fine = simulate_crossover_slse_powder(
            site, 0.0, n_theta=10, n_phi=20, n_chi=4, **sequence
        )
        coarse_shape = np.abs(coarse.echo_amplitudes)
        fine_shape = np.abs(fine.echo_amplitudes)
        coarse_shape /= coarse_shape[1]
        fine_shape /= fine_shape[1]
        np.testing.assert_allclose(coarse_shape, fine_shape, rtol=1.0e-2, atol=2.0e-3)

        times = coarse.echo_times_seconds[1:]
        log_signal = np.log(coarse_shape[1:])
        slope, intercept = np.polyfit(times, log_signal, 1)
        residual = log_signal - (slope * times + intercept)
        self.assertLess(float(np.sqrt(np.mean(residual**2))), 3.0e-2)
        self.assertLess(slope, 0.0)


if __name__ == "__main__":
    unittest.main()
