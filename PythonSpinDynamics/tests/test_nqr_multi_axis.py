"""Tests for multi-axis (circular / tri-axial) NQR excitation primitives.

Covers the superposition pulse Hamiltonian, the circular / quadrature helpers,
the SO(3) powder frame grid, and the physical handedness of circular
excitation + matched quadrature detection (the wrong sense must null).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.evolution import propagator  # noqa: E402
from spin_dynamics.nqr import (  # noqa: E402
    CoilDrive,
    OrientationFrame,
    QuadrupolarSite,
    circular_pulse_hamiltonian,
    detection_operator,
    diagonalize_site,
    equilibrium_density,
    multi_axis_pulse_hamiltonian,
    powder_frame_grid,
    pulse_hamiltonian,
    quadrature_detection_operator,
    static_hamiltonian_rotating,
)


def _site() -> QuadrupolarSite:
    return QuadrupolarSite(spin=1, isotope="14N",
                           quadrupole_frequency_hz=3.0e6, eta=0.3)


def _carrier(eig) -> float:
    return max(t.frequency_hz for t in eig.transitions)


class MultiAxisPulseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = _site()
        self.eig = diagonalize_site(self.site)
        self.carrier = _carrier(self.eig)

    def test_single_coil_matches_pulse_hamiltonian(self) -> None:
        direction = (0.6, 0.8, 0.0)
        multi = multi_axis_pulse_hamiltonian(
            self.eig, [CoilDrive(20e3, 0.3, direction)], rf_frequency_hz=self.carrier)
        ref = pulse_hamiltonian(self.eig, nutation_hz=20e3, rf_frequency_hz=self.carrier,
                                phase=0.3, b1_direction_pas=direction)
        np.testing.assert_allclose(multi, ref, atol=1e-12)

    def test_zero_amplitude_second_coil_reduces_to_linear(self) -> None:
        h = multi_axis_pulse_hamiltonian(
            self.eig,
            [CoilDrive(20e3, 0.0, (1, 0, 0)), CoilDrive(0.0, np.pi / 2, (0, 1, 0))],
            rf_frequency_hz=self.carrier)
        ref = pulse_hamiltonian(self.eig, nutation_hz=20e3, rf_frequency_hz=self.carrier,
                                phase=0.0, b1_direction_pas=(1, 0, 0))
        np.testing.assert_allclose(h, ref, atol=1e-12)

    def test_circular_is_hermitian(self) -> None:
        h = circular_pulse_hamiltonian(
            self.eig, nutation_hz=20e3, rf_frequency_hz=self.carrier,
            axis1_pas=(1, 0, 0), axis2_pas=(0, 1, 0))
        np.testing.assert_allclose(h, h.conj().T, atol=1e-12)

    def test_circular_equals_two_quadrature_coils(self) -> None:
        h = circular_pulse_hamiltonian(
            self.eig, nutation_hz=20e3, rf_frequency_hz=self.carrier,
            axis1_pas=(1, 0, 0), axis2_pas=(0, 1, 0), helicity=1, phase=0.2)
        manual = multi_axis_pulse_hamiltonian(
            self.eig,
            [CoilDrive(20e3, 0.2, (1, 0, 0)),
             CoilDrive(20e3, 0.2 + np.pi / 2, (0, 1, 0))],
            rf_frequency_hz=self.carrier)
        np.testing.assert_allclose(h, manual, atol=1e-12)

    def test_quadrature_detector_sign(self) -> None:
        dp = quadrature_detection_operator(self.eig, self.carrier, (1, 0, 0), (0, 1, 0),
                                           helicity=1)
        dm = quadrature_detection_operator(self.eig, self.carrier, (1, 0, 0), (0, 1, 0),
                                           helicity=-1)
        d1 = detection_operator(self.eig, self.carrier, (1, 0, 0))
        d2 = detection_operator(self.eig, self.carrier, (0, 1, 0))
        np.testing.assert_allclose(dp, d1 + 1j * d2, atol=1e-12)
        np.testing.assert_allclose(dm, d1 - 1j * d2, atol=1e-12)


class HandednessTests(unittest.TestCase):
    """Circular excitation + matched quadrature detection: the wrong sense nulls.

    The null is a *powder* cancellation -- across the SO(3) orientation average
    the counter-rotating (wrong-sense) coherence interferes destructively -- so
    it must be checked on the powder, not a single crystallite.
    """

    def setUp(self) -> None:
        self.site = _site()
        self.eig = diagonalize_site(self.site)
        self.carrier = _carrier(self.eig)
        self.free = static_hamiltonian_rotating(self.eig, self.carrier)
        self.rho0 = equilibrium_density(self.eig.levels_hz)
        self.frames = powder_frame_grid(6, 12, 4)

    def _powder_slse_first_echo(self, detect_helicity: int) -> complex:
        nut = 20e3
        t_ex = 0.15 / (2 * nut)
        t_ref = 2 * t_ex
        tau = 200e-6
        u_free = propagator(self.free, 0.5 * (tau - t_ref))
        total = 0.0 + 0.0j
        for frame in self.frames:
            a1, a2 = frame.x, frame.y
            excite = circular_pulse_hamiltonian(
                self.eig, nutation_hz=nut, rf_frequency_hz=self.carrier,
                axis1_pas=a1, axis2_pas=a2, helicity=1, phase=0.0)
            refocus = circular_pulse_hamiltonian(
                self.eig, nutation_hz=nut, rf_frequency_hz=self.carrier,
                axis1_pas=a1, axis2_pas=a2, helicity=1, phase=np.pi / 2)
            det = quadrature_detection_operator(self.eig, self.carrier, a1, a2,
                                                helicity=detect_helicity)
            u_ex = propagator(excite, t_ex)
            u_ref = propagator(refocus, t_ref)
            rho = u_ex @ self.rho0 @ u_ex.conj().T
            rho = u_free @ rho @ u_free.conj().T
            rho = u_ref @ rho @ u_ref.conj().T
            rho = u_free @ rho @ u_free.conj().T
            total += frame.weight * np.trace(rho @ det)
        return total

    def test_wrong_sense_detection_nulls(self) -> None:
        matched = abs(self._powder_slse_first_echo(+1))
        wrong = abs(self._powder_slse_first_echo(-1))
        self.assertGreater(matched, 0.1)
        self.assertLess(wrong, 1e-6)


class PowderFrameGridTests(unittest.TestCase):
    def test_count_and_weights(self) -> None:
        frames = powder_frame_grid(4, 6, 3)
        self.assertEqual(len(frames), 4 * 6 * 3)
        self.assertAlmostEqual(sum(f.weight for f in frames), 1.0, places=12)

    def test_frames_orthonormal(self) -> None:
        for frame in powder_frame_grid(3, 4, 2):
            np.testing.assert_allclose(frame.axes.T @ frame.axes, np.eye(3), atol=1e-12)

    def test_column_accessors(self) -> None:
        frame = powder_frame_grid(2, 2, 2)[0]
        np.testing.assert_allclose(frame.x, frame.axes[:, 0])
        np.testing.assert_allclose(frame.y, frame.axes[:, 1])
        np.testing.assert_allclose(frame.z, frame.axes[:, 2])

    def test_rejects_non_orthonormal(self) -> None:
        with self.assertRaises(ValueError):
            OrientationFrame(axes=np.ones((3, 3)), weight=1.0)


if __name__ == "__main__":
    unittest.main()
