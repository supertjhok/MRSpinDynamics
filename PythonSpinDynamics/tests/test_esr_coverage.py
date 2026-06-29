"""Broadened coverage for previously under-tested ESR helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.evolution import evolve_density  # noqa: E402
from spin_dynamics.esr import (  # noqa: E402
    ESRRelaxationModel,
    ESRSpinSystem,
    as_g_tensor,
    diagonalize_system,
    effective_g_vector,
    flip_angle_duration,
    gaussian_lineshape,
    normalize_orientations,
    propagate_density_liouville,
    resonance_field_tesla,
    resonance_frequency_hz,
    simulate_fid,
    single_crystal_orientation,
    spectrum_from_lines,
)
from spin_dynamics.esr.orientations import spherical_direction
from spin_dynamics.esr.pulsed import (  # noqa: E402
    detection_operator,
    equilibrium_density,
    rf_operator_eigenbasis,
    rotating_indices,
)
from spin_dynamics.esr.relaxation import (  # noqa: E402
    liouville_hamiltonian,
    matrix_exponential,
    relaxation_superoperator,
)


class GTensorTests(unittest.TestCase):
    def test_scalar_vector_and_matrix_forms(self) -> None:
        np.testing.assert_allclose(as_g_tensor(2.0), 2.0 * np.eye(3))
        np.testing.assert_allclose(as_g_tensor([2.0, 2.1, 2.2]), np.diag([2.0, 2.1, 2.2]))
        full = np.array([[2.0, 0.1, 0.0], [0.1, 2.1, 0.0], [0.0, 0.0, 2.2]])
        np.testing.assert_allclose(as_g_tensor(full), full)

    def test_invalid_shapes_and_values_raise(self) -> None:
        with self.assertRaises(ValueError):
            as_g_tensor([1.0, 2.0])
        with self.assertRaises(ValueError):
            as_g_tensor(0.0)
        with self.assertRaises(ValueError):
            as_g_tensor([np.nan, 2.0, 2.0])

    def test_effective_g_vector_is_g_transpose_direction(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.0, 2.1, 2.2])

        np.testing.assert_allclose(effective_g_vector(system, [0.0, 0.0, 5.0]), [0.0, 0.0, 2.2])

    def test_resonance_field_inverts_resonance_frequency(self) -> None:
        system = ESRSpinSystem(g_tensor=[2.0, 2.05, 2.1])
        direction = (0.0, 0.0, 1.0)
        field = resonance_field_tesla(system, 9.5e9, direction)

        recovered = resonance_frequency_hz(system, field * np.asarray(direction))
        self.assertAlmostEqual(recovered, 9.5e9, places=2)


class RelaxationSuperoperatorTests(unittest.TestCase):
    def test_liouville_propagation_matches_unitary_without_relaxation(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        hamiltonian = diagonalize_system(system, [0.0, 0.0, 0.35]).eigenvectors
        # Use a generic Hermitian Hamiltonian and a coherence-bearing density.
        h = np.array([[1.0e7, 2.0e6], [2.0e6, -1.0e7]], dtype=np.complex128)
        rho = np.array([[0.6, 0.3 + 0.2j], [0.3 - 0.2j, 0.4]], dtype=np.complex128)

        liouville = propagate_density_liouville(rho, h, 3.0e-8, relaxation=None)
        unitary = evolve_density(rho, h, 3.0e-8)

        np.testing.assert_allclose(liouville, unitary, atol=1e-12)
        self.assertEqual(hamiltonian.shape, (2, 2))

    def test_t1_preserves_trace_and_drives_toward_uniform_populations(self) -> None:
        density = np.diag([1.0, 0.0]).astype(np.complex128)
        zero_h = np.zeros((2, 2), dtype=np.complex128)
        model = ESRRelaxationModel(t1_seconds=1.0e-6)

        relaxed = propagate_density_liouville(density, zero_h, 50.0e-6, relaxation=model)

        self.assertAlmostEqual(np.trace(relaxed).real, 1.0, places=10)
        np.testing.assert_allclose(np.diag(relaxed).real, [0.5, 0.5], atol=1e-6)

    def test_t2_damps_coherences_but_not_populations(self) -> None:
        density = np.array([[0.7, 0.5], [0.5, 0.3]], dtype=np.complex128)
        zero_h = np.zeros((2, 2), dtype=np.complex128)
        t2 = 2.0e-6
        model = ESRRelaxationModel(t2_seconds=t2)

        relaxed = propagate_density_liouville(density, zero_h, t2, relaxation=model)

        np.testing.assert_allclose(np.diag(relaxed).real, [0.7, 0.3], atol=1e-12)
        self.assertAlmostEqual(abs(relaxed[0, 1]), 0.5 * np.exp(-1.0), places=12)

    def test_relaxation_superoperator_rejects_bad_dimension(self) -> None:
        with self.assertRaises(ValueError):
            relaxation_superoperator(0, ESRRelaxationModel(t1_seconds=1.0))

    def test_matrix_exponential_identity_and_square_validation(self) -> None:
        np.testing.assert_allclose(
            matrix_exponential(np.array([[3.0, 1.0], [0.0, -2.0]]), 0.0),
            np.eye(2),
        )
        diagonal = np.diag([2.0, -1.0]).astype(np.complex128)
        np.testing.assert_allclose(
            np.diag(matrix_exponential(diagonal, 0.5)),
            [np.exp(1.0), np.exp(-0.5)],
        )
        with self.assertRaises(ValueError):
            liouville_hamiltonian(np.zeros((2, 3)))


class PulsedHelperTests(unittest.TestCase):
    def test_equilibrium_density_is_traceless_with_lower_level_more_populated(self) -> None:
        rho = equilibrium_density(np.array([0.0, 4.0e9]))

        self.assertAlmostEqual(np.trace(rho).real, 0.0, places=12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-14)
        self.assertGreater(rho[0, 0].real, rho[1, 1].real)

    def test_rotating_indices_label_two_level_ladder(self) -> None:
        np.testing.assert_array_equal(
            rotating_indices(np.array([0.0, 9.5e9]), 9.5e9), [0, 1]
        )

    def test_rf_operator_is_hermitian_detector_is_not(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        eigensystem = diagonalize_system(system, [0.0, 0.0, 0.35])
        carrier = eigensystem.transitions[0].frequency_hz

        rf = rf_operator_eigenbasis(eigensystem, (1.0, 0.0, 0.0))
        detector = detection_operator(eigensystem, carrier, (1.0, 0.0, 0.0))

        np.testing.assert_allclose(rf, rf.conj().T, atol=1e-14)
        self.assertFalse(np.allclose(detector, detector.conj().T))

    def test_pi_pulse_produces_no_transverse_signal(self) -> None:
        system = ESRSpinSystem(g_tensor=2.0)
        b0 = [0.0, 0.0, 0.35]
        carrier = resonance_frequency_hz(system, b0)
        nutation_hz = 1.0e6

        result = simulate_fid(
            system,
            b0,
            nutation_hz=nutation_hz,
            pulse_duration_seconds=flip_angle_duration(np.pi, nutation_hz),
            times_seconds=np.array([0.0, 1.0e-6]),
            rf_frequency_hz=carrier,
        )

        np.testing.assert_allclose(np.abs(result.signal), 0.0, atol=1e-9)


class OrientationLineshapeTests(unittest.TestCase):
    def test_spherical_direction_is_unit_and_hits_known_axes(self) -> None:
        np.testing.assert_allclose(spherical_direction(0.0, 0.0), [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            spherical_direction(0.0, np.pi / 2.0), [1.0, 0.0, 0.0], atol=1e-12
        )
        vec = spherical_direction(0.7, 1.2)
        self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=12)

    def test_single_crystal_orientation_requires_paired_b1_angles(self) -> None:
        with self.assertRaises(ValueError):
            single_crystal_orientation(0.0, 0.0, b1_alpha=0.3)

    def test_normalize_orientations_sums_to_one(self) -> None:
        samples = single_crystal_orientation(0.2, 0.4) * 3
        normalized = normalize_orientations(samples)

        self.assertAlmostEqual(sum(s.weight for s in normalized), 1.0, places=12)

    def test_gaussian_lineshape_peak_and_derivative_zero_at_center(self) -> None:
        axis = np.array([-1.0, 0.0, 1.0])

        absorption = gaussian_lineshape(axis, 0.0, 1.0)
        derivative = gaussian_lineshape(axis, 0.0, 1.0, derivative=True)

        self.assertAlmostEqual(absorption[1], 1.0, places=12)
        self.assertAlmostEqual(derivative[1], 0.0, places=12)

    def test_spectrum_from_lines_rejects_unknown_modes(self) -> None:
        with self.assertRaises(ValueError):
            spectrum_from_lines(np.array([0.0]), [0.0], [1.0], width=1.0, lineshape="box")
        with self.assertRaises(ValueError):
            spectrum_from_lines(
                np.array([0.0]), [0.0], [1.0], width=1.0, detection_mode="phase"
            )


if __name__ == "__main__":
    unittest.main()
