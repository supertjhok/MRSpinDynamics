from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    EFGDistribution,
    EFGIsochromat,
    NQRRelaxationModel,
    OrientationSample,
    QuadrupolarSite,
    SelectivePulse,
    apply_selective_pulse,
    b0_b1_powder_average_grid,
    b0_powder_average_grid,
    check_efg_rephasing,
    diagonalize_site,
    efg_line_spectrum,
    gaussian_efg_distribution,
    powder_average_grid,
    propagate_density_liouville,
    quadrupole_frequency_scale_hz,
    simulate_fid_efg_distribution,
    simulate_population_transfer,
    simulate_slse,
    simulate_slse_acquisition_spectrum,
    simulate_slse_efg_distribution,
    simulate_slse_offset_sweep,
    simulate_slse_spacing_sweep,
    simulate_weak_b0_spectrum,
    slse_sequence,
    spin_matrices,
    weak_field_ratio,
    zeeman_frequency_hz,
)


class NQRTests(unittest.TestCase):
    def test_spin_one_operators_use_standard_commutator(self) -> None:
        ops = spin_matrices(1)

        np.testing.assert_allclose(
            ops.ix @ ops.iy - ops.iy @ ops.ix,
            1j * ops.iz,
            atol=1e-14,
        )

    def test_spin_one_quadrupole_transitions_match_xyz_convention(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)

        eigensystem = diagonalize_site(site)
        by_label = {transition.label: transition for transition in eigensystem.transitions}

        self.assertEqual(set(by_label), {"x", "y", "z"})
        self.assertAlmostEqual(by_label["x"].frequency_hz, 990.0)
        self.assertAlmostEqual(by_label["y"].frequency_hz, 810.0)
        self.assertAlmostEqual(by_label["z"].frequency_hz, 180.0)
        np.testing.assert_allclose(
            np.abs(by_label["x"].dipole_vector),
            [1.0, 0.0, 0.0],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.abs(by_label["y"].dipole_vector),
            [0.0, 1.0, 0.0],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            np.abs(by_label["z"].dipole_vector),
            [0.0, 0.0, 1.0],
            atol=1e-14,
        )

    def test_spin_three_halves_nqr_line_uses_chlorine_convention(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            isotope="35Cl",
            quadrupole_frequency_hz=900.0,
            eta=0.3,
        )

        eigensystem = diagonalize_site(site)
        expected = 900.0 * np.sqrt(1.0 + site.eta**2 / 3.0)

        self.assertAlmostEqual(quadrupole_frequency_scale_hz(site), 150.0)
        self.assertGreaterEqual(len(eigensystem.transitions), 1)
        for transition in eigensystem.transitions:
            self.assertGreater(transition.frequency_hz, 0.0)
            self.assertAlmostEqual(transition.frequency_hz, expected)

    def test_powder_grid_weights_are_normalized(self) -> None:
        grid = powder_average_grid(n_theta=6, n_phi=12)

        self.assertAlmostEqual(sum(sample.weight for sample in grid), 1.0)
        self.assertEqual(len(grid), 72)

    def test_b0_powder_grid_weights_and_static_field_directions(self) -> None:
        grid = b0_powder_average_grid(n_theta=4, n_phi=6)

        self.assertAlmostEqual(sum(sample.weight for sample in grid), 1.0)
        self.assertTrue(all(sample.b0_direction_pas is not None for sample in grid))
        self.assertEqual(len(grid), 24)

    def test_correlated_b0_b1_powder_grid_preserves_lab_angle(self) -> None:
        grid = b0_b1_powder_average_grid(
            n_theta=3,
            n_phi=4,
            n_chi=5,
            b1_b0_angle=np.pi / 2.0,
        )

        self.assertAlmostEqual(sum(sample.weight for sample in grid), 1.0)
        self.assertEqual(len(grid), 60)
        dots = [
            float(np.dot(sample.b0_direction_pas, sample.b1_direction_pas))
            for sample in grid
        ]
        np.testing.assert_allclose(dots, 0.0, atol=1e-14)

    def test_selective_pi_pulse_swaps_selected_transition_populations(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        transition = diagonalize_site(site).transition("x")
        density = np.diag([1.0, 0.25, -1.0]).astype(np.complex128)

        final = apply_selective_pulse(
            density,
            transition,
            SelectivePulse("x", duration_seconds=0.5, nutation_hz=1.0),
            b1_direction_pas=(1.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(final[transition.lower, transition.lower].real, -1.0)
        self.assertAlmostEqual(final[transition.upper, transition.upper].real, 1.0)
        self.assertAlmostEqual(final[1, 1].real, 0.25)

    def test_selective_pulse_leaves_orthogonal_transition_dark(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        transition = diagonalize_site(site).transition("x")
        density = np.diag([1.0, 0.25, -1.0]).astype(np.complex128)

        final = apply_selective_pulse(
            density,
            transition,
            SelectivePulse("x", duration_seconds=0.5, nutation_hz=1.0),
            b1_direction_pas=(0.0, 1.0, 0.0),
        )

        np.testing.assert_allclose(final, density, atol=1e-14)

    def test_slse_accepts_initial_coherence_and_applies_t2e_decay(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        transition = diagonalize_site(site).transition("x")
        density = np.zeros((3, 3), dtype=np.complex128)
        density[transition.upper, transition.lower] = 1.0
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            echo_spacing_seconds=0.1,
            num_echoes=3,
        )

        result = simulate_slse(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            t2e_seconds=0.2,
            initial_density=density,
        )

        expected = np.exp(-result.echo_times / 0.2)
        np.testing.assert_allclose(result.echo_amplitudes.real, expected)

    def test_spin_three_halves_selective_pulses_require_manifold_model(self) -> None:
        site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=900.0, eta=0.0)
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=1,
        )

        with self.assertRaises(NotImplementedError):
            simulate_slse(
                site,
                sequence,
                orientations=[OrientationSample((1.0, 0.0, 0.0))],
            )

    def test_weak_b0_static_spectrum_supports_spin_one_and_three_halves(self) -> None:
        spin_one = QuadrupolarSite(
            spin=1,
            quadrupole_frequency_hz=900.0,
            eta=0.3,
            gamma_hz_per_t=3.0e6,
        )
        spin_three_halves = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=900.0,
            eta=0.1,
            gamma_hz_per_t=4.0e6,
        )

        one = simulate_weak_b0_spectrum(
            spin_one,
            1e-6,
            transition_label="x",
            orientations=[OrientationSample((1.0, 0.0, 0.0), b0_direction_pas=(0.0, 0.0, 1.0))],
            broadening_hz=1.0,
            points=65,
            weak_ratio_action="ignore",
        )
        three_halves = simulate_weak_b0_spectrum(
            spin_three_halves,
            1e-6,
            orientations=[OrientationSample((1.0, 0.0, 0.0), b0_direction_pas=(0.0, 0.0, 1.0))],
            broadening_hz=1.0,
            points=65,
            weak_ratio_action="ignore",
        )

        self.assertGreater(len(one.transitions), 0)
        self.assertGreater(len(three_halves.transitions), 0)
        self.assertLess(one.max_perturbation_ratio, 0.05)
        self.assertLess(three_halves.max_perturbation_ratio, 0.05)
        self.assertEqual(one.spectrum.shape, one.offsets_hz.shape)
        self.assertEqual(three_halves.spectrum.shape, three_halves.offsets_hz.shape)

    def test_spin_three_halves_weak_b0_axial_limit_has_two_rf_lines(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.0e6,
        )

        result = simulate_weak_b0_spectrum(
            site,
            1e-3,
            orientations=[
                OrientationSample(
                    (1.0, 0.0, 0.0),
                    b0_direction_pas=(0.0, 0.0, 1.0),
                )
            ],
            broadening_hz=1.0,
            points=65,
            weak_ratio_action="ignore",
        )

        frequencies = sorted(round(item.frequency_hz) for item in result.transitions)
        self.assertEqual(frequencies, [996000, 1004000])
        self.assertEqual(len(result.transitions), 2)

    def test_weak_b0_ratio_helpers_and_warning(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.0e6,
        )

        self.assertAlmostEqual(zeeman_frequency_hz(site, [0.0, 0.0, 1e-3]), 4.0e3)
        self.assertAlmostEqual(weak_field_ratio(site, 1e-3), 0.004)
        with self.assertWarns(RuntimeWarning):
            simulate_weak_b0_spectrum(
                site,
                0.1,
                broadening_hz=10.0,
                points=17,
                weak_ratio_threshold=0.01,
            )

    def test_liouville_relaxation_preserves_trace_and_damps_coherence(self) -> None:
        density = np.array(
            [
                [1.0, 0.5, 0.0],
                [0.25, -0.5, 0.0],
                [0.0, 0.0, -0.5],
            ],
            dtype=np.complex128,
        )
        hamiltonian = np.zeros((3, 3), dtype=np.complex128)

        final = propagate_density_liouville(
            density,
            hamiltonian,
            0.1,
            relaxation=NQRRelaxationModel(t1_seconds=0.2, t2_seconds=0.5),
        )

        self.assertAlmostEqual(np.trace(final).real, np.trace(density).real)
        np.testing.assert_allclose(final[0, 1], density[0, 1] * np.exp(-0.1 / 0.5))
        self.assertLess(
            np.linalg.norm(np.diag(final)),
            np.linalg.norm(np.diag(density)),
        )

    def test_relaxing_slse_uses_liouville_t2_and_reports_effective_decay(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        transition = diagonalize_site(site).transition("x")
        density = np.zeros((3, 3), dtype=np.complex128)
        density[transition.upper, transition.lower] = 1.0
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            echo_spacing_seconds=0.1,
            num_echoes=3,
        )

        result = simulate_slse(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            initial_density=density,
            relaxation=NQRRelaxationModel(t2_seconds=0.2),
        )

        expected = np.exp(-result.echo_times / 0.2)
        np.testing.assert_allclose(result.echo_amplitudes.real, expected)
        self.assertIsNotNone(result.local_effective_t2eff_seconds)
        self.assertIsNotNone(result.local_cycle_eigenvalues)
        np.testing.assert_allclose(result.local_effective_t2eff_seconds, [0.2])

    def test_relaxing_slse_offset_sweep_reports_amplitude_and_t2eff(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)

        result = simulate_slse_offset_sweep(
            site,
            "x",
            [-20.0, 0.0, 20.0],
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            echo_spacing_seconds=0.1,
            num_echoes=2,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            relaxation=NQRRelaxationModel(t2_seconds=0.2),
        )

        self.assertEqual(result.sweep_name, "offset_hz")
        self.assertEqual(result.selected_echo_amplitudes.shape, (3,))
        self.assertEqual(result.effective_t2eff_seconds.shape, (3,))
        np.testing.assert_allclose(result.effective_t2eff_seconds, 0.2)

    def test_relaxing_slse_spacing_sweep_tracks_cycle_duration(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)

        result = simulate_slse_spacing_sweep(
            site,
            "x",
            [0.05, 0.1, 0.2],
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            num_echoes=2,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            relaxation=NQRRelaxationModel(t2_seconds=0.2),
        )

        self.assertEqual(result.sweep_name, "echo_spacing_seconds")
        np.testing.assert_allclose(result.sweep_values, [0.05, 0.1, 0.2])
        np.testing.assert_allclose(result.effective_t2eff_seconds, 0.2)

    def test_gaussian_efg_distribution_normalizes_weights(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)

        distribution = gaussian_efg_distribution(
            site,
            quadrupole_std_hz=5.0,
            samples=5,
        )

        self.assertEqual(len(distribution.isochromats), 5)
        self.assertAlmostEqual(float(np.sum(distribution.weights)), 1.0)

    def test_single_sample_gaussian_efg_distribution_uses_center(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)

        distribution = gaussian_efg_distribution(
            site,
            quadrupole_std_hz=5.0,
            eta_std=0.1,
            samples=1,
        )

        only_site = distribution.isochromats[0].site
        self.assertAlmostEqual(only_site.quadrupole_frequency_hz, 900.0)
        self.assertAlmostEqual(only_site.eta, 0.3)

    def test_efg_distribution_fid_dephases_and_returns_spectrum(self) -> None:
        broad = EFGDistribution(
            (
                EFGIsochromat(
                    QuadrupolarSite(
                        spin=1,
                        quadrupole_frequency_hz=890.0,
                        eta=0.3,
                    ),
                    0.5,
                ),
                EFGIsochromat(
                    QuadrupolarSite(
                        spin=1,
                        quadrupole_frequency_hz=910.0,
                        eta=0.3,
                    ),
                    0.5,
                ),
            )
        )
        times = np.linspace(0.0, 0.05, 64)

        result = simulate_fid_efg_distribution(
            broad,
            "x",
            times,
            excitation=SelectivePulse("x", duration_seconds=0.00025, nutation_hz=1e3),
            carrier_frequency_hz=990.0,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            rephase_action="ignore",
            window="none",
        )

        self.assertEqual(result.signal.shape, times.shape)
        self.assertEqual(result.isochromat_frequencies_hz.shape, (2,))
        self.assertEqual(result.spectrum.shape, result.spectrum_frequencies_hz.shape)
        self.assertLess(abs(result.signal[-1]), abs(result.signal[0]))

    def test_efg_line_spectrum_zero_width_has_centered_single_peak(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        distribution = gaussian_efg_distribution(site, samples=1)
        carrier = diagonalize_site(site).transition("x").frequency_hz

        axis, spectrum = efg_line_spectrum(
            distribution,
            "x",
            carrier_frequency_hz=carrier,
            linewidth_hz=10.0,
            points=129,
        )

        self.assertAlmostEqual(axis[int(np.argmax(spectrum))], 0.0)
        local_maxima = np.where(
            (spectrum[1:-1] > spectrum[:-2])
            & (spectrum[1:-1] > spectrum[2:])
        )[0]
        self.assertEqual(local_maxima.size, 1)

    def test_slse_acquisition_requires_window_shorter_than_spacing(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        distribution = gaussian_efg_distribution(site, samples=1)
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=1,
        )

        with self.assertRaises(ValueError):
            simulate_slse_acquisition_spectrum(
                distribution,
                sequence,
                acquisition_duration_seconds=0.1,
                orientations=[OrientationSample((1.0, 0.0, 0.0))],
            )

    def test_slse_acquisition_zero_width_spectrum_is_centered(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        distribution = gaussian_efg_distribution(site, samples=1)
        carrier = diagonalize_site(site).transition("x").frequency_hz
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=2,
            rf_frequency_hz=carrier,
        )

        result = simulate_slse_acquisition_spectrum(
            distribution,
            sequence,
            acquisition_duration_seconds=0.02,
            acquisition_points=64,
            echo_index=0,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            zero_fill_factor=2,
            rephase_action="ignore",
        )

        peak = int(np.argmax(np.abs(result.spectrum)))
        self.assertAlmostEqual(result.spectrum_frequencies_hz[peak], 0.0)
        self.assertEqual(
            np.count_nonzero(np.abs(result.spectrum) == abs(result.spectrum[peak])),
            1,
        )

    def test_slse_acquisition_noise_and_deconvolution_across_snr(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        distribution = gaussian_efg_distribution(
            site,
            quadrupole_std_hz=20.0,
            samples=5,
        )
        carrier = diagonalize_site(site).transition("x").frequency_hz
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=2,
            rf_frequency_hz=carrier,
        )

        noise_rms = []
        for snr in (5.0, 20.0, 80.0):
            result = simulate_slse_acquisition_spectrum(
                distribution,
                sequence,
                acquisition_duration_seconds=0.02,
                acquisition_points=32,
                echo_index=0,
                orientations=[OrientationSample((1.0, 0.0, 0.0))],
                zero_fill_factor=1,
                noise={"target_snr": snr, "seed": 123},
                deconvolution_strength=1e-2,
                rephase_action="ignore",
            )

            self.assertIsNotNone(result.noise_metadata)
            self.assertIsNotNone(result.deconvolution)
            self.assertEqual(result.noise_metadata.domain, "time")
            self.assertTrue(np.all(np.isfinite(result.spectrum)))
            self.assertTrue(
                np.all(np.isfinite(result.deconvolution.deconvolved_spectrum))
            )
            noise_rms.append(result.noise_metadata.noise_rms)

        self.assertGreater(noise_rms[0], noise_rms[1])
        self.assertGreater(noise_rms[1], noise_rms[2])

    def test_efg_rephasing_check_warns_for_coarse_grid(self) -> None:
        with self.assertWarns(RuntimeWarning):
            check_efg_rephasing([0.0, 100.0, 200.0], max_time_seconds=0.02)

    def test_slse_efg_distribution_matches_single_site_when_width_is_zero(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        distribution = gaussian_efg_distribution(site, samples=1)
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=2,
        )

        single = simulate_slse(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
        )
        distributed = simulate_slse_efg_distribution(
            distribution,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
        )

        np.testing.assert_allclose(distributed.echo_amplitudes, single.echo_amplitudes)

    def test_powder_slse_signal_does_not_cancel_projection_signs(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        sequence = slse_sequence(
            "x",
            pulse_duration_seconds=1.0 / 3.0,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=1,
        )

        result = simulate_slse(site, sequence, orientations=powder_average_grid(6, 12))

        self.assertGreater(abs(result.echo_amplitudes[0]), 0.1)

    def test_population_transfer_changes_detection_echo(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        orientation = OrientationSample((1.0, 1.0, 0.0))
        detection = slse_sequence(
            "y",
            pulse_duration_seconds=0.25,
            nutation_hz=1.0,
            echo_spacing_seconds=0.1,
            num_echoes=1,
        )

        result = simulate_population_transfer(
            site,
            SelectivePulse("x", duration_seconds=0.5, nutation_hz=1.0),
            detection,
            orientations=[orientation],
        )

        self.assertGreater(abs(result.normalized_difference[0]), 0.05)

    def test_weak_b0_perturbs_transition_frequencies(self) -> None:
        site = QuadrupolarSite(
            spin=1,
            quadrupole_frequency_hz=900.0,
            eta=0.3,
            gamma_hz_per_t=3.0e6,
        )

        zero_field = diagonalize_site(site).transition("x").frequency_hz
        weak_field = diagonalize_site(site, [0.0, 0.0, 1e-5]).transition("x").frequency_hz

        self.assertNotAlmostEqual(zero_field, weak_field)


if __name__ == "__main__":
    unittest.main()
