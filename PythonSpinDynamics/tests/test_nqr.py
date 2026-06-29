from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    EFGDistribution,
    EFGIsochromat,
    NQRRelaxationModel,
    OrientationSample,
    PROTON_GAMMA_HZ_PER_T,
    CylindricalSampleGeometry,
    GAMMA_14N_HZ_PER_T,
    LinearTransportMotion,
    PolarizationEnhancedNQRSample,
    QuadrupolarSite,
    SelectivePulse,
    apply_selective_pulse,
    b0_b1_powder_average_grid,
    b0_powder_average_grid,
    check_efg_rephasing,
    diagonalize_site,
    dipolar_coupling_hz,
    efg_line_spectrum,
    equilibrium_density,
    fid_powder_theory_signal,
    gaussian_efg_distribution,
    ideal_spin1_enhancement_factors,
    estimate_proton_dipolar_couplings_from_cif,
    powder_average_grid,
    propagate_density_liouville,
    quadrupole_frequency_scale_hz,
    relaxation_superoperator,
    transition_drive_scale,
    zeeman_hamiltonian,
    simulate_fid_efg_distribution,
    simulate_adiabatic_polarization_transfer,
    simulate_population_transfer,
    simulate_slse,
    simulate_slse_acquisition_spectrum,
    simulate_slse_efg_distribution,
    simulate_slse_offset_sweep,
    simulate_slse_spacing_sweep,
    simulate_sorc,
    simulate_weak_b0_spectrum,
    slse_sequence,
    sorc_powder_theory_signal,
    sorc_sequence,
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

    def test_glickstein_spin_one_enhancement_matches_melamine_example(self) -> None:
        enhancements = ideal_spin1_enhancement_factors(
            (2.766e6, 2.034e6, 0.732e6),
            max_b0_tesla=0.65,
            protons_per_molecule=6.0,
            nitrogens_per_molecule=6.0,
        )

        self.assertAlmostEqual(enhancements[0], 8.24, delta=0.1)
        self.assertGreater(enhancements[2], enhancements[0])

    def test_adiabatic_transfer_improves_at_lower_velocity(self) -> None:
        def fringe_field(points):
            z = np.asarray(points)[..., 2]
            return 0.70 - 20.0 * z

        sample = PolarizationEnhancedNQRSample(
            name="melamine-like",
            line_frequencies_hz=(2.766e6, 2.034e6, 0.732e6),
            protons_per_molecule=6.0,
            nitrogens_per_molecule=6.0,
            proton_t1_seconds=50.0,
            nitrogen_t1_seconds=5.0,
            proton_linewidth_hz=25.0e3,
            proton_nitrogen_coupling_hz=300.0,
        )
        geometry = CylindricalSampleGeometry(
            length=1.0e-3,
            diameter=1.0e-3,
            axial_points=3,
            radial_rings=0,
        )
        slow = simulate_adiabatic_polarization_transfer(
            fringe_field,
            sample,
            geometry,
            LinearTransportMotion(0.0, 0.04, velocity=0.005),
            prepolarization_time_seconds=np.inf,
            path_points=101,
        )
        fast = simulate_adiabatic_polarization_transfer(
            fringe_field,
            sample,
            geometry,
            LinearTransportMotion(0.0, 0.04, velocity=0.10),
            prepolarization_time_seconds=np.inf,
            path_points=101,
        )

        self.assertTrue(np.all(np.isfinite(slow.crossing_positions)))
        np.testing.assert_allclose(
            slow.crossing_fields_tesla,
            slow.line_frequencies_hz / PROTON_GAMMA_HZ_PER_T,
        )
        self.assertGreater(slow.practical_enhancement[0],
                           fast.practical_enhancement[0])
        self.assertGreater(slow.transfer_efficiency[0], fast.transfer_efficiency[0])

    def test_finite_prepolarization_time_reduces_enhancement(self) -> None:
        def fringe_field(points):
            z = np.asarray(points)[..., 2]
            return 0.70 - 20.0 * z

        sample = PolarizationEnhancedNQRSample(
            line_frequencies_hz=(2.766e6,),
            line_labels=("line",),
            protons_per_molecule=6.0,
            nitrogens_per_molecule=1.0,
            proton_t1_seconds=50.0,
            proton_linewidth_hz=25.0e3,
            proton_nitrogen_coupling_hz=300.0,
        )
        geometry = CylindricalSampleGeometry(
            length=1.0e-3,
            diameter=1.0e-3,
            axial_points=1,
            radial_rings=0,
        )
        motion = LinearTransportMotion(0.0, 0.04, velocity=0.005)

        saturated = simulate_adiabatic_polarization_transfer(
            fringe_field,
            sample,
            geometry,
            motion,
            prepolarization_time_seconds=np.inf,
            path_points=101,
        )
        finite = simulate_adiabatic_polarization_transfer(
            fringe_field,
            sample,
            geometry,
            motion,
            prepolarization_time_seconds=10.0,
            path_points=101,
        )

        self.assertGreater(saturated.practical_enhancement[0],
                           finite.practical_enhancement[0])
        self.assertAlmostEqual(
            finite.proton_polarization_factor,
            1.0 - np.exp(-10.0 / 50.0),
        )

    def test_dipolar_coupling_formula_matches_one_angstrom_nh_pair(self) -> None:
        coupling = dipolar_coupling_hz(
            1.0,
            gamma_a_hz_per_t=GAMMA_14N_HZ_PER_T,
            gamma_b_hz_per_t=PROTON_GAMMA_HZ_PER_T,
        )

        self.assertTrue(8.0e3 < coupling < 9.0e3)

    def test_cif_nearby_proton_coupling_estimator(self) -> None:
        cif = """data_test
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_symmetry_equiv_pos_as_xyz
x,y,z
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
N1 N 0 0 0
H1 H 0.1 0 0
H2 H 0.5 0 0
"""
        tmpdir = ROOT / ".tmp"
        tmpdir.mkdir(exist_ok=True)
        path = tmpdir / "test_coupling.cif"
        path.write_text(cif, encoding="utf-8")

        estimate = estimate_proton_dipolar_couplings_from_cif(
            path,
            "N1",
            proton_radius_angstrom=1.5,
        )

        self.assertEqual(estimate.target_label, "N1")
        self.assertEqual(len(estimate.proton_couplings), 1)
        self.assertEqual(estimate.proton_couplings[0].proton_label, "H1")
        self.assertAlmostEqual(estimate.proton_couplings[0].distance_angstrom, 1.0)
        self.assertAlmostEqual(
            estimate.effective_rms_hz,
            estimate.proton_couplings[0].coupling_hz,
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

    def test_sorc_powder_theory_has_konnai_offset_zeros(self) -> None:
        tau = 0.8e-3
        zeros = sorc_powder_theory_signal(
            [0.0, 1.0 / (2.0 * tau), 2.0 / (2.0 * tau)],
            tau,
            0.66 * np.pi,
        )
        antinode = sorc_powder_theory_signal(
            [0.5 / (2.0 * tau)],
            tau,
            0.66 * np.pi,
        )

        np.testing.assert_allclose(zeros, 0.0, atol=1e-12)
        self.assertGreater(float(antinode[0]), 0.1)

    def test_fid_powder_theory_peaks_near_spin_one_powder_pulse(self) -> None:
        flip_angles = np.linspace(0.05, 4.0, 400)
        signal = fid_powder_theory_signal(flip_angles)
        peak_angle = float(flip_angles[int(np.argmax(signal))])

        self.assertGreater(peak_angle, 1.9)
        self.assertLess(peak_angle, 2.2)

    def test_sorc_accepts_initial_coherence_and_applies_t2e_decay(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900.0, eta=0.3)
        transition = diagonalize_site(site).transition("x")
        density = np.zeros((3, 3), dtype=np.complex128)
        density[transition.upper, transition.lower] = 1.0
        sequence = sorc_sequence(
            "x",
            pulse_duration_seconds=0.0,
            nutation_hz=0.0,
            half_spacing_seconds=0.05,
            num_pulses=3,
        )

        result = simulate_sorc(
            site,
            sequence,
            orientations=[OrientationSample((1.0, 0.0, 0.0))],
            t2e_seconds=0.2,
            initial_density=density,
        )

        expected = np.exp(-result.observation_times / 0.2)
        np.testing.assert_allclose(result.signal_amplitudes.real, expected)

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


class NQRAuditAdditionTests(unittest.TestCase):
    """Coverage added during the physics audit (operators, equilibrium,
    relaxation generator, drive selectivity, and the t2e/relaxation guard)."""

    def test_operators_satisfy_casimir_and_ladder_relations(self) -> None:
        for spin, casimir in ((1.0, 2.0), (1.5, 3.75)):
            ops = spin_matrices(spin)
            i_squared = ops.ix @ ops.ix + ops.iy @ ops.iy + ops.iz @ ops.iz
            np.testing.assert_allclose(
                i_squared, casimir * np.eye(ops.iz.shape[0]), atol=1e-12
            )
            np.testing.assert_allclose(
                ops.iy @ ops.iz - ops.iz @ ops.iy, 1j * ops.ix, atol=1e-12
            )
        # <3/2|I+|1/2> = sqrt(I(I+1) - m(m+1)) = sqrt(3.75 - 0.75) = sqrt(3).
        self.assertAlmostEqual(abs(spin_matrices(1.5).i_plus[0, 1]), np.sqrt(3.0),
                               places=12)

    def test_three_line_frequencies_track_eta_parametrically(self) -> None:
        nu_q = 1.2e6
        for eta in (0.0, 0.15, 0.4, 0.85):
            site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=nu_q, eta=eta)
            freqs = {t.label: t.frequency_hz for t in diagonalize_site(site).transitions}
            self.assertAlmostEqual(freqs["x"], nu_q * (1 + eta / 3), places=2)
            if eta > 0:  # nu_0 line is dropped at eta = 0
                self.assertAlmostEqual(freqs["y"], nu_q * (1 - eta / 3), places=2)
                self.assertAlmostEqual(freqs["z"], (2.0 / 3.0) * nu_q * eta, places=2)

    def test_label_axis_is_the_dominant_drive_axis(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        axis_of = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
        for transition in diagonalize_site(site).transitions:
            self.assertAlmostEqual(
                transition_drive_scale(transition, axis_of[transition.label]),
                1.0, places=6,
            )
            for label, direction in axis_of.items():
                if label != transition.label:
                    self.assertLess(transition_drive_scale(transition, direction), 1e-9)

    def test_equilibrium_density_is_traceless_hermitian_boltzmann(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        eig = diagonalize_site(site)
        rho = equilibrium_density(eig.levels_hz)
        self.assertAlmostEqual(np.trace(rho).real, 0.0, places=12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
        # High-temperature limit: the lowest-energy level is the most populated.
        self.assertEqual(int(np.argmax(np.diag(rho).real)),
                         int(np.argmin(eig.levels_hz)))

    def test_relaxation_generator_is_traceless_and_t1_mixes_to_uniform(self) -> None:
        model = NQRRelaxationModel(t1_seconds=2e-3, t2_seconds=4e-3)
        gen = relaxation_superoperator(3, model)
        rho = np.diag([0.2, 0.3, 0.5]).astype(np.complex128)
        drho = (gen @ rho.reshape(-1, order="F")).reshape((3, 3), order="F")
        self.assertAlmostEqual(np.trace(drho).real, 0.0, places=12)
        # Pure T1 over many time constants equalizes populations, conserving trace.
        relaxed = propagate_density_liouville(
            np.diag([1.0, 0.0, 0.0]).astype(np.complex128),
            np.zeros((3, 3), dtype=np.complex128),
            1.0,
            relaxation=NQRRelaxationModel(t1_seconds=1e-3, t2_seconds=np.inf),
        )
        self.assertAlmostEqual(np.trace(relaxed).real, 1.0, places=10)
        np.testing.assert_allclose(np.diag(relaxed).real, np.full(3, 1.0 / 3.0),
                                   atol=1e-6)

    def test_zeeman_hamiltonian_hermitian_and_linear_in_field(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3,
                               gamma_hz_per_t=3.08e6)
        h1 = zeeman_hamiltonian(site, [0.0, 0.0, 1e-3])
        h2 = zeeman_hamiltonian(site, [0.0, 0.0, 2e-3])
        np.testing.assert_allclose(h1, h1.conj().T, atol=1e-9)
        np.testing.assert_allclose(h2, 2.0 * h1, atol=1e-9)

    def test_simulate_slse_warns_when_t2e_and_relaxation_both_set(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        seq = slse_sequence("x", pulse_duration_seconds=25e-6, nutation_hz=10e3,
                            echo_spacing_seconds=1e-3, num_echoes=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate_slse(site, seq, orientations="single", t2e_seconds=20e-3,
                          relaxation=NQRRelaxationModel(t2_seconds=20e-3))
        self.assertTrue(any("double-counted" in str(w.message) for w in caught))

    def test_simulate_slse_does_not_warn_for_relaxation_only(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        seq = slse_sequence("x", pulse_duration_seconds=25e-6, nutation_hz=10e3,
                            echo_spacing_seconds=1e-3, num_echoes=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate_slse(site, seq, orientations="single",
                          relaxation=NQRRelaxationModel(t2_seconds=20e-3))
        self.assertFalse(any("double-counted" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
