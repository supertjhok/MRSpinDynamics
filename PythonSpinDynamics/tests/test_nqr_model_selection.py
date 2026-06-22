"""Tests for the reduced-vs-full NQR model-selection layer.

The recommendation must follow the physics of the technical note, not the spin
quantum number: it depends on the static Hamiltonian, the RF polarization, and
the pulse parameters. Cases cover spin-1 isolation, eta=0 degeneracy,
polarization selectivity, spin-3/2 Kramers doublets, Zeeman-resolved spin-3/2,
and RF-dark targets.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    diagonalize_site,
    select_nqr_model,
)


class Spin1ModelSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)

    def test_isolated_line_weak_pulse_is_reduced(self) -> None:
        sel = select_nqr_model(self.site, "x", nutation_hz=1e3,
                               pulse_duration_seconds=200e-6,
                               b1_direction_pas=(1, 1, 1))
        self.assertEqual(sel.recommended_model, "reduced")
        self.assertTrue(sel.reduced_is_valid)
        self.assertEqual(sel.active_states, (0, 2))
        self.assertGreaterEqual(sel.isolation_ratio, sel.isolation_threshold)

    def test_broadband_general_polarization_is_full(self) -> None:
        # A 2 us pulse (500 kHz bandwidth) cannot resolve the 180 kHz line gap
        # when every line is RF-active.
        sel = select_nqr_model(self.site, "x", nutation_hz=1e3,
                               pulse_duration_seconds=2e-6,
                               b1_direction_pas=(1, 1, 1))
        self.assertEqual(sel.recommended_model, "full")
        self.assertLess(sel.isolation_ratio, sel.isolation_threshold)
        self.assertGreater(len(sel.active_states), 2)

    def test_polarization_isolation_stays_reduced_even_broadband(self) -> None:
        # B1 || x makes the y and z lines RF-dark, so the x line is isolated by
        # polarization regardless of pulse bandwidth.
        sel = select_nqr_model(self.site, "x", nutation_hz=1e3,
                               pulse_duration_seconds=2e-6,
                               b1_direction_pas=(1, 0, 0))
        self.assertEqual(sel.recommended_model, "reduced")
        self.assertEqual(sel.active_states, (0, 2))
        self.assertIsNone(sel.nearest_competing_label)

    def test_eta_zero_is_full(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.0)
        label = diagonalize_site(site).transitions[0].label
        sel = select_nqr_model(site, label, nutation_hz=1e3,
                               pulse_duration_seconds=200e-6,
                               b1_direction_pas=(1, 1, 1))
        self.assertEqual(sel.recommended_model, "full")
        self.assertTrue(sel.degenerate_target)
        self.assertEqual(sel.isolation_hz, 0.0)

    def test_rf_dark_target_is_flagged_and_full(self) -> None:
        sel = select_nqr_model(self.site, "x", nutation_hz=1e3,
                               pulse_duration_seconds=200e-6,
                               b1_direction_pas=(0, 0, 1))
        self.assertTrue(sel.target_is_rf_dark)
        self.assertEqual(sel.recommended_model, "full")
        self.assertEqual(sel.target_coupling, 0.0)


class Spin3HalvesModelSelection(unittest.TestCase):
    def test_zero_field_kramers_doublets_force_full(self) -> None:
        site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=30e6, eta=0.1)
        label = diagonalize_site(site).transitions[0].label
        sel = select_nqr_model(site, label, nutation_hz=5e3,
                               pulse_duration_seconds=50e-6,
                               b1_direction_pas=(1, 1, 1))
        self.assertEqual(sel.recommended_model, "full")
        self.assertTrue(sel.degenerate_target)
        self.assertEqual(len(sel.active_states), 4)  # both doublets

    def test_zeeman_resolved_line_can_be_reduced(self) -> None:
        # A strong static field splits the doublets into well-separated lines;
        # the lowest, isolated transition becomes a valid reduced case.
        site = QuadrupolarSite(spin=1.5, quadrupole_frequency_hz=30e6, eta=0.0,
                               gamma_hz_per_t=4.17e6)
        b0 = [0.0, 0.0, 0.5]
        label = diagonalize_site(site, b0).transitions[0].label
        sel = select_nqr_model(site, label, nutation_hz=5e3,
                               pulse_duration_seconds=50e-6,
                               b1_direction_pas=(1, 1, 1), b0_vector_tesla_pas=b0)
        self.assertFalse(sel.degenerate_target)
        self.assertEqual(sel.recommended_model, "reduced")
        self.assertEqual(len(sel.active_states), 2)


class ModelSelectionDiagnostics(unittest.TestCase):
    def test_linewidth_can_tip_recommendation_to_full(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        kwargs = dict(nutation_hz=1e3, pulse_duration_seconds=200e-6,
                      b1_direction_pas=(1, 1, 1))
        narrow = select_nqr_model(site, "x", linewidth_hz=1e3, **kwargs)
        broad = select_nqr_model(site, "x", linewidth_hz=100e3, **kwargs)
        self.assertEqual(narrow.recommended_model, "reduced")
        # A 100 kHz linewidth is comparable to the 180 kHz line spacing.
        self.assertEqual(broad.recommended_model, "full")
        self.assertGreater(broad.broadening_hz, narrow.broadening_hz)

    def test_describe_reports_key_quantities(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        text = select_nqr_model(site, "x", nutation_hz=1e3,
                                pulse_duration_seconds=200e-6,
                                b1_direction_pas=(1, 1, 1)).describe()
        for token in ("recommended model", "isolation ratio", "broadening",
                      "addressed states", "reasons"):
            self.assertIn(token, text)

    def test_invalid_inputs_raise(self) -> None:
        site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
        with self.assertRaises(ValueError):
            select_nqr_model(site, "x", nutation_hz=-1.0,
                             pulse_duration_seconds=1e-4)
        with self.assertRaises(ValueError):
            select_nqr_model(site, "x", nutation_hz=1e3,
                             pulse_duration_seconds=1e-4, isolation_threshold=0.0)


if __name__ == "__main__":
    unittest.main()
