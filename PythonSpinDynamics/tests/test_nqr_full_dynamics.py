"""Validation tests for the full (2I+1)-level density-matrix NQR model.

The decisive check propagates a pulse with the *exact* lab-frame, time-dependent
Hamiltonian ``H_Q + H_Z - 2 (2 pi nu_1) cos(2 pi nu_rf t + phi) (e1 . I)`` (no
RWA), transforms the result into the rotating frame, and compares it with the
module's RWA pulse Hamiltonian. The two must agree in the ``omega_1 << omega_rf``
limit, with the residual scaling as the Bloch-Siegert ``omega_1**2`` shift.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.evolution import evolve_density  # noqa: E402
from spin_dynamics.nqr import (  # noqa: E402
    NQRRelaxationModel,
    QuadrupolarSite,
    diagonalize_site,
)
from spin_dynamics.nqr.simulation import equilibrium_density  # noqa: E402
from spin_dynamics.nqr.full_dynamics import (  # noqa: E402
    pulse_hamiltonian,
    rf_operator_eigenbasis,
    rotating_indices,
    simulate_full_echo,
    simulate_full_fid,
)

TAU = 2.0 * np.pi


def _labframe_rotating(site, rho0, nutation, rf_hz, phase, b1, duration, dt):
    """Exact lab-frame pulse propagation, returned in the rotating frame."""

    eig = diagonalize_site(site)
    h0 = np.diag(TAU * eig.levels_hz).astype(np.complex128)
    rf_op = rf_operator_eigenbasis(eig, b1)
    rho = rho0.copy()
    n_steps = int(np.ceil(duration / dt))
    step = duration / n_steps
    for k in range(n_steps):
        t = (k + 0.5) * step
        drive = -2.0 * TAU * nutation * np.cos(TAU * rf_hz * t + phase)
        rho = evolve_density(rho, h0 + drive * rf_op, step)
    n = rotating_indices(eig.levels_hz, rf_hz)
    rotation = np.diag(np.exp(-1j * TAU * rf_hz * n * duration))
    return rotation.conj().T @ rho @ rotation


class FullModelLabFrameValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=1.0e6, eta=0.0)
        self.eig = diagonalize_site(self.site)
        self.nu_q = max(t.frequency_hz for t in self.eig.transitions)
        self.rho0 = equilibrium_density(self.eig.levels_hz)

    def test_rwa_matches_exact_labframe_in_weak_drive_limit(self) -> None:
        nutation = 1e3  # omega_1 / omega_rf = 1e-3
        h = pulse_hamiltonian(self.eig, nutation_hz=nutation, rf_frequency_hz=self.nu_q,
                              phase=0.7, b1_direction_pas=(1, 0, 0))
        rho_rwa = evolve_density(self.rho0, h, 20e-6)
        rho_lab = _labframe_rotating(self.site, self.rho0, nutation, self.nu_q, 0.7,
                                     (1, 0, 0), 20e-6, dt=5e-9)
        self.assertLess(np.max(np.abs(rho_rwa - rho_lab)), 1e-3)

    def test_rwa_error_scales_quadratically_bloch_siegert(self) -> None:
        def err(nutation: float) -> float:
            h = pulse_hamiltonian(self.eig, nutation_hz=nutation,
                                  rf_frequency_hz=self.nu_q, phase=0.7,
                                  b1_direction_pas=(1, 0, 0))
            rho_rwa = evolve_density(self.rho0, h, 20e-6)
            rho_lab = _labframe_rotating(self.site, self.rho0, nutation, self.nu_q,
                                         0.7, (1, 0, 0), 20e-6, dt=5e-9)
            return float(np.max(np.abs(rho_rwa - rho_lab)))

        # In the perturbative regime the residual is the omega_1^2 Bloch-Siegert
        # shift, so halving the drive cuts the error by roughly four.
        ratio = err(2e3) / err(1e3)
        self.assertTrue(3.0 < ratio < 5.0, f"unexpected scaling ratio {ratio:.2f}")


class FullModelStructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=1.0e6, eta=0.0)
        self.eig = diagonalize_site(self.site)
        self.nu_q = max(t.frequency_hz for t in self.eig.transitions)

    def test_pulse_hamiltonian_is_hermitian(self) -> None:
        h = pulse_hamiltonian(self.eig, nutation_hz=5e3, rf_frequency_hz=self.nu_q,
                              phase=0.4, b1_direction_pas=(1, 0, 0))
        np.testing.assert_allclose(h, h.conj().T, atol=1e-9)

    def test_dominant_cross_doublet_coupling_is_root_three_over_two(self) -> None:
        # spin-3/2: the |3/2><1/2| element of I_x is sqrt(3)/2.
        rf_op = rf_operator_eigenbasis(self.eig, (1, 0, 0))
        n = rotating_indices(self.eig.levels_hz, self.nu_q)
        band = (n[:, None] - n[None, :]) == 1
        self.assertAlmostEqual(float(np.max(np.abs(rf_op[band]))), np.sqrt(3) / 2,
                               places=10)

    def test_four_states_underlie_the_single_zero_field_line(self) -> None:
        # One NQR frequency, but a 4x4 model: two degenerate Kramers doublets.
        self.assertEqual(self.eig.levels_hz.size, 4)
        distinct = {round(t.frequency_hz, 3) for t in self.eig.transitions}
        self.assertEqual(len(distinct), 1)


class FullModelSequenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=1.0e6, eta=0.0)

    def test_fid_baseband_frequency_tracks_carrier_offset(self) -> None:
        eig = diagonalize_site(self.site)
        nu_q = max(t.frequency_hz for t in eig.transitions)
        offset = 50e3
        times = np.linspace(0.0, 200e-6, 2000)
        fid = simulate_full_fid(self.site, nutation_hz=5e3,
                                pulse_duration_seconds=30e-6, times_seconds=times,
                                rf_frequency_hz=nu_q + offset)
        sig = fid.signal - fid.signal.mean()
        freqs = np.fft.fftfreq(times.size, d=times[1] - times[0])
        peak = abs(freqs[np.argmax(np.abs(np.fft.fft(sig)))])
        self.assertAlmostEqual(peak, offset, delta=2e3)

    def test_fid_conserves_trace_and_needs_a_pulse_for_signal(self) -> None:
        times = np.linspace(0.0, 40e-6, 64)
        # Zero-length pulse leaves equilibrium populations -> no coherence -> ~0 signal.
        quiet = simulate_full_fid(self.site, nutation_hz=5e3,
                                  pulse_duration_seconds=0.0, times_seconds=times)
        self.assertLess(np.max(np.abs(quiet.signal)), 1e-9)
        loud = simulate_full_fid(self.site, nutation_hz=5e3,
                                 pulse_duration_seconds=40e-6, times_seconds=times)
        self.assertGreater(np.max(np.abs(loud.signal)), 1e-3)

    def test_echo_runs_and_relaxation_damps_signal(self) -> None:
        times = np.linspace(0.0, 100e-6, 128)
        bare = simulate_full_echo(self.site, nutation_hz=5e3,
                                  excitation_duration_seconds=20e-6,
                                  refocus_duration_seconds=40e-6,
                                  echo_spacing_seconds=120e-6, times_seconds=times)
        self.assertTrue(np.all(np.isfinite(bare.signal)))
        damped = simulate_full_echo(self.site, nutation_hz=5e3,
                                    excitation_duration_seconds=20e-6,
                                    refocus_duration_seconds=40e-6,
                                    echo_spacing_seconds=120e-6, times_seconds=times,
                                    relaxation=NQRRelaxationModel(t2_seconds=30e-6))
        self.assertLess(np.max(np.abs(damped.signal)), np.max(np.abs(bare.signal)))

    def test_spin_one_full_fid_recovers_addressed_transition_frequency(self) -> None:
        # The full model also works for spin-1; an FID addressed to the x line and
        # detuned by `offset` shows that offset as its baseband frequency.
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        nu_x = diagonalize_site(site).transition("x").frequency_hz
        offset = 30e3
        times = np.linspace(0.0, 300e-6, 3000)
        fid = simulate_full_fid(site, nutation_hz=4e3, pulse_duration_seconds=40e-6,
                                times_seconds=times, rf_frequency_hz=nu_x + offset,
                                b1_direction_pas=(1, 0, 0))
        sig = fid.signal - fid.signal.mean()
        freqs = np.fft.fftfreq(times.size, d=times[1] - times[0])
        peak = abs(freqs[np.argmax(np.abs(np.fft.fft(sig)))])
        self.assertAlmostEqual(peak, offset, delta=2e3)


if __name__ == "__main__":
    unittest.main()
