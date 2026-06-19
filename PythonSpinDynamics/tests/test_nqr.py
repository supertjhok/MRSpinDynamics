from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    OrientationSample,
    QuadrupolarSite,
    SelectivePulse,
    apply_selective_pulse,
    diagonalize_site,
    powder_average_grid,
    simulate_population_transfer,
    simulate_slse,
    slse_sequence,
    spin_matrices,
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
        np.testing.assert_allclose(np.abs(by_label["x"].dipole_vector), [1.0, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(by_label["y"].dipole_vector), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(np.abs(by_label["z"].dipole_vector), [0.0, 0.0, 1.0])

    def test_powder_grid_weights_are_normalized(self) -> None:
        grid = powder_average_grid(n_theta=6, n_phi=12)

        self.assertAlmostEqual(sum(sample.weight for sample in grid), 1.0)
        self.assertEqual(len(grid), 72)

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
