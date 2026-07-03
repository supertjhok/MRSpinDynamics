from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (
    QUADRUPOLAR_ISOTOPES,
    QuadrupolarSite,
    cq_hz_from_nu_q,
    diagonalize_site,
    nu_q_from_cq_hz,
    quadrupolar_site,
    simulate_full_echo,
    simulate_full_fid,
)
from spin_dynamics.nqr.hamiltonians import quadrupole_hamiltonian
from spin_dynamics.relaxation import NQRRelaxationModel, liouville_superoperator


def _line_freqs(site: QuadrupolarSite) -> list[float]:
    """Distinct zero-field transition frequencies (Hz), ascending."""

    eig = diagonalize_site(site)
    return sorted({round(t.frequency_hz) for t in eig.transitions})


class HigherSpinLadderTests(unittest.TestCase):
    def test_axial_transition_ratios(self) -> None:
        for spin, ratios in ((2.5, [1, 2]), (3.5, [1, 2, 3]), (4.5, [1, 2, 3, 4])):
            site = QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1.0e6, eta=0.0)
            lines = _line_freqs(site)
            self.assertEqual(len(lines), len(ratios))
            self.assertEqual([round(f / lines[0]) for f in lines], ratios)

    def test_fundamental_line_equals_nu_q(self) -> None:
        for spin in (1.0, 1.5, 2.5, 3.5, 4.5):
            site = QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1.3e6, eta=0.0)
            self.assertAlmostEqual(min(_line_freqs(site)), 1.3e6, delta=1.0)

    def test_hilbert_dimension_scales_with_spin(self) -> None:
        for spin, dim in ((2.5, 6), (3.5, 8), (4.5, 10)):
            self.assertEqual(
                QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1e6).dimension, dim
            )

    def test_kramers_doublets_persist(self) -> None:
        # Half-integer spins keep doubly-degenerate levels (Kramers theorem), even
        # with eta != 0; so there are (2I+1)/2 distinct energies.
        for spin, distinct in ((2.5, 3), (3.5, 4), (4.5, 5)):
            eig = diagonalize_site(
                QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1e6, eta=0.4)
            )
            energies = {round(float(e), 3) for e in eig.levels_hz}
            self.assertEqual(len(energies), distinct)

    def test_eta_shifts_lines_smoothly(self) -> None:
        # Axial (eta=0) spin-5/2 has exactly the 1:2 line pair; eta != 0 mixes the
        # eigenstates, shifting the lines and switching on a weak combination line.
        axial = _line_freqs(QuadrupolarSite(spin=2.5, quadrupole_frequency_hz=1e6, eta=0.0))
        self.assertEqual(len(axial), 2)
        self.assertAlmostEqual(axial[1] / axial[0], 2.0, delta=1e-6)

        prev_fundamental = None
        for eta in np.linspace(0.0, 0.8, 9):
            lines = _line_freqs(QuadrupolarSite(spin=2.5, quadrupole_frequency_hz=1e6, eta=float(eta)))
            fundamental = min(lines)
            self.assertTrue(np.all(np.isfinite(lines)) and fundamental > 0)
            if prev_fundamental is not None:  # smooth, small step-to-step change
                self.assertLess(abs(fundamental - prev_fundamental), 0.4e6)
            prev_fundamental = fundamental
        # eta != 0 moves the fundamental line away from the axial nu_Q.
        skewed = min(_line_freqs(QuadrupolarSite(spin=2.5, quadrupole_frequency_hz=1e6, eta=0.6)))
        self.assertGreater(abs(skewed - 1.0e6), 1.0e3)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_spin_one_xyz_convention_unchanged(self) -> None:
        # nu_+ = nu_Q(1 + eta/3), nu_- = nu_Q(1 - eta/3), nu_0 = nu_+ - nu_-.
        site = QuadrupolarSite(spin=1.0, quadrupole_frequency_hz=900e3, eta=0.3)
        self.assertEqual(_line_freqs(site), [180000, 810000, 990000])

    def test_spin_three_halves_single_line_unchanged(self) -> None:
        eig = diagonalize_site(QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=1e6, eta=0.0))
        self.assertEqual(eig.levels_hz.size, 4)
        self.assertTrue(all(np.isclose(t.frequency_hz, 1e6) for t in eig.transitions))


class ConversionAndPresetTests(unittest.TestCase):
    def test_cq_to_nu_q_matches_standard_factors(self) -> None:
        cq = 20.0e6
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 1.0), 0.75 * cq, delta=1.0)
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 1.5), 0.5 * cq, delta=1.0)
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 2.5), 3.0 * cq / 20.0, delta=1.0)
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 3.5), cq / 14.0, delta=1.0)
        self.assertAlmostEqual(nu_q_from_cq_hz(cq, 4.5), cq / 24.0, delta=1.0)

    def test_cq_nu_q_roundtrip(self) -> None:
        for spin in (1.0, 1.5, 2.5, 3.5, 4.5):
            self.assertAlmostEqual(
                cq_hz_from_nu_q(nu_q_from_cq_hz(7.0e6, spin), spin), 7.0e6, delta=1.0
            )

    def test_quadrupolar_site_from_cq_matches_fundamental_line(self) -> None:
        site = quadrupolar_site("27Al", cq_hz=3.0e6, eta=0.0)
        self.assertEqual(site.spin, 2.5)
        self.assertGreater(site.gamma_hz_per_t, 0.0)
        self.assertAlmostEqual(min(_line_freqs(site)), 3.0 * 3.0e6 / 20.0, delta=1.0)

    def test_registry_isotopes_instantiate(self) -> None:
        for iso in ("14N", "35Cl", "27Al", "51V", "93Nb", "209Bi"):
            self.assertIn(iso, QUADRUPOLAR_ISOTOPES)
            self.assertGreater(quadrupolar_site(iso, nu_q_hz=1e6).gamma_hz_per_t, 0.0)

    def test_requires_exactly_one_of_cq_or_nu_q(self) -> None:
        with self.assertRaises(ValueError):
            quadrupolar_site("27Al", cq_hz=1e6, nu_q_hz=1e6)
        with self.assertRaises(ValueError):
            quadrupolar_site("27Al")


class HigherSpinFullDynamicsTests(unittest.TestCase):
    def test_fid_baseband_tracks_offset_on_fundamental(self) -> None:
        site = quadrupolar_site("27Al", nu_q_hz=1.0e6, eta=0.0)  # spin-5/2
        nu1 = min(_line_freqs(site))
        offset = 40.0e3
        times = np.linspace(0.0, 300e-6, 1500)
        fid = simulate_full_fid(
            site, nutation_hz=5e3, pulse_duration_seconds=30e-6,
            times_seconds=times, rf_frequency_hz=nu1 + offset,
        )
        sig = fid.signal - fid.signal.mean()
        freqs = np.fft.fftfreq(times.size, d=times[1] - times[0])
        peak = abs(freqs[np.argmax(np.abs(np.fft.fft(sig)))])
        self.assertAlmostEqual(peak, offset, delta=3e3)

    def test_zero_pulse_gives_no_signal(self) -> None:
        site = quadrupolar_site("51V", nu_q_hz=1.0e6)  # spin-7/2
        times = np.linspace(0.0, 40e-6, 64)
        quiet = simulate_full_fid(
            site, nutation_hz=5e3, pulse_duration_seconds=0.0, times_seconds=times,
            rf_frequency_hz=1.0e6,
        )
        self.assertLess(np.max(np.abs(quiet.signal)), 1e-9)

    def test_echo_runs_and_relaxation_damps(self) -> None:
        site = quadrupolar_site("27Al", nu_q_hz=1.0e6)  # spin-5/2
        times = np.linspace(0.0, 120e-6, 128)
        common = dict(
            nutation_hz=5e3, excitation_duration_seconds=20e-6,
            refocus_duration_seconds=40e-6, echo_spacing_seconds=140e-6,
            times_seconds=times, rf_frequency_hz=1.0e6,
        )
        bare = simulate_full_echo(site, **common)
        damped = simulate_full_echo(
            site, relaxation=NQRRelaxationModel(t2_seconds=30e-6), **common
        )
        self.assertTrue(np.all(np.isfinite(bare.signal)))
        self.assertLess(np.max(np.abs(damped.signal)), np.max(np.abs(bare.signal)))

    def test_liouville_superoperator_dimension_scales(self) -> None:
        model = NQRRelaxationModel(t1_seconds=1e-3, t2_seconds=1e-3)
        for spin, dim2 in ((2.5, 36), (3.5, 64), (4.5, 100)):
            hamiltonian = quadrupole_hamiltonian(
                QuadrupolarSite(spin=spin, quadrupole_frequency_hz=1e6)
            )
            liouvillian = liouville_superoperator(hamiltonian, model)
            self.assertEqual(liouvillian.shape, (dim2, dim2))


class RWAValidityGuardTests(unittest.TestCase):
    def test_driving_fundamental_line_does_not_warn(self) -> None:
        site = quadrupolar_site("51V", nu_q_hz=1.0e6)  # spin-7/2, lines at 1,2,3 MHz
        times = np.linspace(0.0, 40e-6, 64)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            simulate_full_fid(
                site, nutation_hz=5e3, pulse_duration_seconds=20e-6,
                times_seconds=times, rf_frequency_hz=1.0e6,
            )
        self.assertFalse([w for w in caught if issubclass(w.category, RuntimeWarning)])

    def test_driving_satellite_line_warns(self) -> None:
        site = quadrupolar_site("51V", nu_q_hz=1.0e6)  # second line at 2 MHz
        times = np.linspace(0.0, 40e-6, 64)
        with self.assertWarns(RuntimeWarning):
            simulate_full_fid(
                site, nutation_hz=5e3, pulse_duration_seconds=20e-6,
                times_seconds=times, rf_frequency_hz=2.0e6,
            )


if __name__ == "__main__":
    unittest.main()
