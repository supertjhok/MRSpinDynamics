"""Tests for the nonresonant field-reversal (CSAR) spin-echo engine.

Validates the Bloch rotation core, the basic-vs-CSAR physics (background-field
dephasing, even-echo refocusing, the ~pi net rotation and ~1/2 refocused fraction),
and that a finite reversal time accelerates the odd-even modulation -- the key claims
of Brill et al., Science 297, 369 (2002).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nonresonant import (  # noqa: E402
    NonresonantFieldModel,
    basic_reversal_sequence,
    csar_double_reversal_sequence,
    csar_sequence,
    csar_supercycle_sequence,
    effective_rotation,
    rodrigues_rotate,
    sample_isochromats,
    sequence_waveform,
    simulate_field_reversal_echoes,
)


def _model(background_tesla=50e-6):
    return NonresonantFieldModel(coil_a_tesla=2.14e-3, coil_b_tesla=2.14e-3,
                                 background_tesla=background_tesla)


class RotationCoreTests(unittest.TestCase):
    def test_rodrigues_quarter_turn(self):
        v = np.array([[1.0, 0.0, 0.0]])
        axis = np.array([[0.0, 0.0, 1.0]])
        out = rodrigues_rotate(v, axis, np.pi / 2)
        np.testing.assert_allclose(out[0], [0.0, 1.0, 0.0], atol=1e-12)

    def test_rodrigues_preserves_norm(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((20, 3))
        axis = v / np.linalg.norm(v, axis=1)[:, None]
        out = rodrigues_rotate(v, axis, rng.uniform(0, 2 * np.pi, 20))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1),
                                   np.linalg.norm(v, axis=1), atol=1e-12)

    def test_axis_component_is_invariant(self):
        # A rotation leaves the component along its axis unchanged.
        v = np.array([[0.3, 0.4, 0.5]])
        axis = np.array([[0.0, 0.0, 1.0]])
        out = rodrigues_rotate(v, axis, 1.234)
        self.assertAlmostEqual(out[0, 2], v[0, 2], places=12)


class EnsembleTests(unittest.TestCase):
    def test_sample_shapes_and_weights(self):
        ens = sample_isochromats(_model(), 128, seed=1)
        self.assertEqual(ens.size, 128)
        self.assertEqual(ens.coil_a_dir.shape, (128, 3))
        self.assertAlmostEqual(float(ens.weight.sum()), 1.0, places=12)
        np.testing.assert_allclose(np.linalg.norm(ens.coil_b_dir, axis=1), 1.0, atol=1e-12)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.spacing = 2.5e-3
        self.ens = sample_isochromats(_model(), 400, b_inhomogeneity=0.25,
                                      direction_tilt_deg=15.0, seed=0)
        self.ens_nobg = sample_isochromats(_model(0.0), 400, b_inhomogeneity=0.25,
                                           direction_tilt_deg=15.0, seed=0)

    def test_background_dephases_basic_sequence(self):
        seq = basic_reversal_sequence(_model(), echo_spacing_seconds=self.spacing)
        with_bg = simulate_field_reversal_echoes(_model(), self.ens, seq,
                                                 num_echoes=40, t2_seconds=np.inf)
        no_bg = simulate_field_reversal_echoes(_model(0.0), self.ens_nobg, seq,
                                               num_echoes=40, t2_seconds=np.inf)
        # With no background the basic reversal echoes stay refocused; with the
        # background they decay strongly.
        self.assertGreater(no_bg.magnitude[-1] / no_bg.magnitude[0], 0.9)
        self.assertLess(with_bg.magnitude[-1] / with_bg.magnitude[0], 0.7)

    def test_csar_refocuses_even_echoes(self):
        seq = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                            tau_rev_seconds=0.0)
        r = simulate_field_reversal_echoes(_model(), self.ens, seq, num_echoes=12,
                                           t2_seconds=np.inf)
        mag = r.magnitude
        even = mag[1:8:2]  # echoes 2,4,6,8
        odd = mag[0:8:2]   # echoes 1,3,5,7
        self.assertGreater(even.min(), 5.0 * odd.max())

    def test_effective_rotation_is_near_pi_and_half_refocused(self):
        seq = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                            tau_rev_seconds=0.0)
        angles = []
        fracs = []
        for k in range(0, self.ens.size, 20):
            axis, angle = effective_rotation(self.ens, seq, k)
            angles.append(np.rad2deg(angle))
            fracs.append(axis[0] ** 2)
        self.assertGreater(np.mean(angles), 170.0)
        self.assertLess(np.mean(angles), 185.0)
        self.assertAlmostEqual(float(np.mean(fracs)), 0.5, delta=0.12)

    def test_finite_reversal_accelerates_odd_echo_growth(self):
        ideal = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                              tau_rev_seconds=0.0)
        imperfect = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                                  tau_rev_seconds=40e-6)
        r_ideal = simulate_field_reversal_echoes(_model(), self.ens, ideal,
                                                 num_echoes=30, t2_seconds=np.inf)
        r_imp = simulate_field_reversal_echoes(_model(), self.ens, imperfect,
                                               num_echoes=30, t2_seconds=np.inf)
        # odd echoes (energy leaked from even) grow faster with a finite reversal
        odd_ideal = r_ideal.magnitude[10:30:2].mean()
        odd_imp = r_imp.magnitude[10:30:2].mean()
        self.assertGreater(odd_imp, odd_ideal)

    def test_t2_envelope_applied(self):
        seq = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                            tau_rev_seconds=0.0)
        r = simulate_field_reversal_echoes(_model(), self.ens, seq, num_echoes=20,
                                           t2_seconds=100e-3)
        ratio = r.echo_amplitudes / r.coherent_amplitudes
        np.testing.assert_allclose(np.abs(ratio), np.exp(-r.echo_times / 100e-3),
                                   atol=1e-12)


class SequenceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.spacing = 2.5e-3
        self.ens = sample_isochromats(_model(), 300, b_inhomogeneity=0.25,
                                      direction_tilt_deg=18.0, seed=0)

    def test_double_reversal_refocuses_both_parities(self):
        seq = csar_double_reversal_sequence(_model(), echo_spacing_seconds=self.spacing,
                                            tau_rev_seconds=0.0)
        r = simulate_field_reversal_echoes(_model(), self.ens, seq, num_echoes=8,
                                           t2_seconds=np.inf)
        # unlike the pi (single-reversal) unit, odd echoes are NOT pinned near zero
        odd = r.magnitude[0:6:2]
        self.assertGreater(odd.min(), 0.3 * r.magnitude.max())

    def test_supercycle_sustains_better_than_single_reversal(self):
        pi_seq = csar_sequence(_model(), echo_spacing_seconds=self.spacing,
                               tau_rev_seconds=30e-6)
        sc_seq = csar_supercycle_sequence(_model(), echo_spacing_seconds=self.spacing,
                                          tau_rev_seconds=30e-6)
        r_pi = simulate_field_reversal_echoes(_model(), self.ens, pi_seq,
                                              num_echoes=80, t2_seconds=np.inf)
        r_sc = simulate_field_reversal_echoes(_model(), self.ens, sc_seq,
                                              num_echoes=80, t2_seconds=np.inf)

        def envelope(m):  # parity-independent refocused-signal envelope, normalized
            npr = 2 * (m.size // 2)
            e = np.sqrt(m[0:npr:2] ** 2 + m[1:npr:2] ** 2)
            return e / e[0]

        # the supercycle suppresses the imperfection decay, so its late-echo signal
        # envelope stays higher than the single-reversal (pi) sequence's
        late_pi = envelope(r_pi.magnitude)[30:40].mean()
        late_sc = envelope(r_sc.magnitude)[30:40].mean()
        self.assertGreater(late_sc, 1.12 * late_pi)

    def test_supercycle_accepts_unit_cycle(self):
        sc_seq = csar_supercycle_sequence(_model(), echo_spacing_seconds=self.spacing)
        self.assertEqual(len(sc_seq), 4)  # four units cycled
        r = simulate_field_reversal_echoes(_model(), self.ens, sc_seq, num_echoes=12,
                                           t2_seconds=np.inf)
        self.assertEqual(r.magnitude.shape, (12,))


class WaveformTests(unittest.TestCase):
    def test_sequence_waveform_shapes_and_values(self):
        seq = csar_sequence(_model(), echo_spacing_seconds=2.5e-3)
        t, i_a, i_b = sequence_waveform(seq, repeats=2)
        self.assertEqual(t.shape, i_a.shape)
        self.assertEqual(t.shape, i_b.shape)
        self.assertTrue(np.all(np.diff(t) >= 0))  # non-decreasing time
        # coil B holds at its nominal field during free precession
        self.assertAlmostEqual(np.max(np.abs(i_b)), _model().coil_b_tesla, places=6)

    def test_waveform_handles_supercycle(self):
        sc = csar_supercycle_sequence(_model(), echo_spacing_seconds=2.5e-3)
        t, i_a, i_b = sequence_waveform(sc, repeats=1)
        self.assertGreater(t.size, 0)


if __name__ == "__main__":
    unittest.main()
