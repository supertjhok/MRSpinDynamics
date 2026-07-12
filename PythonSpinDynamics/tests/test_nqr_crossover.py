from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    CrossoverOrientation,
    NQREigensystem,
    QuadrupolarSite,
    boltzmann_populations,
    crossover_transitions_from_eigensystem,
    diagonalize_site,
    simulate_crossover_spectrum,
    simulate_crossover_powder_sweep,
    track_crossover_field_sweep,
)


class NQRCrossoverTests(unittest.TestCase):
    def test_boltzmann_populations_are_normalized_and_energy_ordered(self) -> None:
        populations = boltzmann_populations([-1.0e6, 0.0, 2.0e6], 300.0)

        self.assertAlmostEqual(float(np.sum(populations)), 1.0)
        self.assertGreater(populations[0], populations[1])
        self.assertGreater(populations[1], populations[2])

    def test_zero_field_spin_three_halves_aggregates_kramers_doublets(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.3,
            gamma_hz_per_t=4.0e6,
        )

        result = simulate_crossover_spectrum(
            site,
            0.0,
            broadening_hz=10.0,
            points=33,
        )

        self.assertEqual(len(result.transitions), 1)
        transition = result.transitions[0]
        self.assertEqual(transition.lower_levels, (0, 1))
        self.assertEqual(transition.upper_levels, (2, 3))
        expected = 1.0e6 * np.sqrt(1.0 + 0.3**2 / 3.0)
        self.assertAlmostEqual(transition.frequency_hz, expected, places=7)
        self.assertGreater(transition.intensity, 0.0)

    def test_zero_field_manifold_response_is_basis_invariant(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.4,
        )
        eigensystem = diagonalize_site(site)
        orientation = CrossoverOrientation(
            transmit_direction_pas=(1.0, 2.0, -0.5),
        )
        reference = crossover_transitions_from_eigensystem(
            eigensystem,
            orientation,
        )

        lower_rotation = np.array(
            [[1.0, 1.0j], [1.0j, 1.0]],
            dtype=np.complex128,
        ) / np.sqrt(2.0)
        upper_rotation = np.array(
            [[1.0, -1.0], [1.0, 1.0]],
            dtype=np.complex128,
        ) / np.sqrt(2.0)
        rotated_vectors = eigensystem.eigenvectors.copy()
        rotated_vectors[:, :2] = rotated_vectors[:, :2] @ lower_rotation
        rotated_vectors[:, 2:] = rotated_vectors[:, 2:] @ upper_rotation
        rotated = NQREigensystem(
            site=site,
            levels_hz=eigensystem.levels_hz,
            eigenvectors=rotated_vectors,
            transitions=eigensystem.transitions,
        )
        transformed = crossover_transitions_from_eigensystem(
            rotated,
            orientation,
        )

        self.assertEqual(len(reference), 1)
        self.assertEqual(len(transformed), 1)
        self.assertAlmostEqual(
            reference[0].frequency_hz,
            transformed[0].frequency_hz,
        )
        np.testing.assert_allclose(
            reference[0].local_amplitude,
            transformed[0].local_amplitude,
            rtol=1e-9,
            atol=1e-15,
        )

    def test_high_field_axial_spin_three_halves_recovers_nmr_triplet(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.0e6,
        )

        result = simulate_crossover_spectrum(
            site,
            2.5,
            broadening_hz=10.0,
            points=33,
        )

        self.assertEqual(result.zeeman_to_quadrupole_ratio, 10.0)
        np.testing.assert_allclose(
            [item.frequency_hz for item in result.transitions],
            [9.0e6, 10.0e6, 11.0e6],
            atol=1e-7,
        )
        matrix_element_strengths = [
            item.excitation_strength / item.population_difference
            for item in result.transitions
        ]
        np.testing.assert_allclose(
            matrix_element_strengths,
            [0.75, 1.0, 0.75],
            atol=1e-9,
        )

    def test_intermediate_tilted_field_exposes_all_spin_three_halves_lines(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.3,
            gamma_hz_per_t=4.0e6,
        )
        orientation = CrossoverOrientation(
            b0_direction_pas=(1.0, 1.0, 1.0),
            transmit_direction_pas=(1.0, -1.0, 0.0),
        )

        result = simulate_crossover_spectrum(
            site,
            0.25,
            orientations=[orientation],
            broadening_hz=10.0,
            points=33,
        )

        self.assertAlmostEqual(result.zeeman_to_quadrupole_ratio, 1.0)
        self.assertEqual(len(result.transitions), 6)
        self.assertTrue(all(item.intensity > 0.0 for item in result.transitions))
        np.testing.assert_allclose(
            result.transitions[0].b0_vector_tesla_pas,
            np.full(3, 0.25 / np.sqrt(3.0)),
        )

    def test_reversing_static_field_preserves_reciprocal_coil_spectrum(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.3,
            gamma_hz_per_t=4.0e6,
        )
        forward = CrossoverOrientation(
            b0_direction_pas=(1.0, 2.0, 3.0),
            transmit_direction_pas=(2.0, -1.0, 0.5),
        )
        reverse = CrossoverOrientation(
            b0_direction_pas=(-1.0, -2.0, -3.0),
            transmit_direction_pas=(2.0, -1.0, 0.5),
        )

        positive = simulate_crossover_spectrum(
            site,
            0.25,
            orientations=[forward],
            broadening_hz=10.0,
            points=33,
        )
        negative = simulate_crossover_spectrum(
            site,
            0.25,
            orientations=[reverse],
            broadening_hz=10.0,
            points=33,
        )

        np.testing.assert_allclose(
            [item.frequency_hz for item in positive.transitions],
            [item.frequency_hz for item in negative.transitions],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            [item.intensity for item in positive.transitions],
            [item.intensity for item in negative.transitions],
            rtol=1e-8,
            atol=1e-15,
        )

    def test_circular_polarization_selects_high_field_helicity(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.0,
            gamma_hz_per_t=4.0e6,
        )
        active = CrossoverOrientation(
            transmit_direction_pas=(1.0, -1.0j, 0.0),
        )
        inactive = CrossoverOrientation(
            transmit_direction_pas=(1.0, 1.0j, 0.0),
        )

        result = simulate_crossover_spectrum(
            site,
            2.5,
            orientations=[active],
            broadening_hz=10.0,
            points=33,
        )

        self.assertEqual(len(result.transitions), 3)
        with self.assertRaisesRegex(ValueError, "no observable transitions"):
            simulate_crossover_spectrum(
                site,
                2.5,
                orientations=[inactive],
                broadening_hz=10.0,
                points=33,
            )

    def test_orientation_weights_are_normalized_before_summation(self) -> None:
        site = QuadrupolarSite(
            spin=1,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=3.0e6,
        )
        orientations = [
            CrossoverOrientation(weight=2.0),
            CrossoverOrientation(
                b0_direction_pas=(1.0, 0.0, 0.0),
                transmit_direction_pas=(0.0, 1.0, 0.0),
                weight=1.0,
            ),
        ]

        result = simulate_crossover_spectrum(
            site,
            0.1,
            orientations=orientations,
            broadening_hz=10.0,
            points=33,
        )

        weights = {
            item.orientation_index: item.orientation_weight
            for item in result.transitions
        }
        self.assertAlmostEqual(weights[0], 2.0 / 3.0)
        self.assertAlmostEqual(weights[1], 1.0 / 3.0)

    def test_reciprocal_coil_spectrum_is_real_and_normalizable(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.0e6,
        )

        result = simulate_crossover_spectrum(
            site,
            0.25,
            broadening_hz=1.0e3,
            points=257,
            normalize=True,
        )

        np.testing.assert_allclose(result.spectrum.imag, 0.0, atol=1e-15)
        self.assertAlmostEqual(float(np.max(result.spectrum.real)), 1.0)

    def test_field_sweep_tracks_complete_orthonormal_eigenbasis(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.25,
            gamma_hz_per_t=4.0e6,
        )
        ratios = np.logspace(-3.0, 1.0, 81)
        fields = ratios * site.quadrupole_frequency_hz / site.gamma_hz_per_t

        result = track_crossover_field_sweep(site, fields)

        self.assertEqual(result.levels_hz.shape, (81, 4))
        self.assertEqual(result.transition_frequencies_hz.shape, (81, 6))
        self.assertGreater(float(np.min(result.minimum_state_overlap[1:])), 0.5)
        for index in (0, 40, 80):
            np.testing.assert_allclose(
                result.eigenvectors[index].conj().T @ result.eigenvectors[index],
                np.eye(4),
                atol=1e-12,
            )
            direct = diagonalize_site(
                site,
                [0.0, 0.0, fields[index]],
            )
            np.testing.assert_allclose(
                np.sort(result.levels_hz[index]),
                direct.levels_hz,
                atol=1e-8,
            )

    def test_high_field_tracked_states_recover_zeeman_quantum_numbers(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.0e6,
        )
        ratios = np.logspace(-2.0, 2.0, 101)
        fields = ratios * site.quadrupole_frequency_hz / site.gamma_hz_per_t

        result = track_crossover_field_sweep(site, fields)

        np.testing.assert_allclose(
            np.sort(result.magnetic_quantum_expectation[-1]),
            [-1.5, -0.5, 0.5, 1.5],
            atol=1e-3,
        )

    def test_powder_sweep_returns_common_normalized_frequency_map(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.0e6,
        )

        result = simulate_crossover_powder_sweep(
            site,
            [0.05, 0.25],
            n_theta=3,
            n_phi=6,
            n_chi=3,
            broadening_hz=20.0e3,
            frequency_points=129,
        )

        self.assertEqual(result.orientation_count, 54)
        self.assertEqual(result.spectra.shape, (2, 129))
        np.testing.assert_allclose(np.max(np.abs(result.spectra), axis=1), 1.0)
        np.testing.assert_allclose(result.spectra.imag, 0.0, atol=1e-15)
        np.testing.assert_allclose(
            result.zeeman_to_quadrupole_ratio,
            [0.2, 1.0],
        )

    def test_powder_integrated_intensity_converges_with_grid(self) -> None:
        site = QuadrupolarSite(
            spin=1.5,
            quadrupole_frequency_hz=1.0e6,
            eta=0.2,
            gamma_hz_per_t=4.0e6,
        )
        common = dict(
            b0_tesla=[0.25],
            broadening_hz=20.0e3,
            frequency_points=65,
            normalize_each_field=False,
        )

        medium = simulate_crossover_powder_sweep(
            site,
            n_theta=3,
            n_phi=6,
            n_chi=3,
            **common,
        )
        fine = simulate_crossover_powder_sweep(
            site,
            n_theta=4,
            n_phi=8,
            n_chi=4,
            **common,
        )

        self.assertAlmostEqual(
            medium.integrated_intensity[0] / fine.integrated_intensity[0],
            1.0,
            delta=0.02,
        )


if __name__ == "__main__":
    unittest.main()
