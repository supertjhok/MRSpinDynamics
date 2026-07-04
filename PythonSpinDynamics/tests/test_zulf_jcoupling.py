from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.coupling.isotopes import (
    larmor_frequency_hz,
    nuclear_isotope,
)
from spin_dynamics.coupling.mixed_operators import (
    dot_product_operator,
    embedded_operator,
    hilbert_dimension,
    product_operator,
)
from spin_dynamics.coupling.multinuclear import (
    multinuclear_hamiltonian,
    multinuclear_system,
    per_spin_relaxation_superoperator,
)
from spin_dynamics.coupling.zulf import simulate_zulf_fid, simulate_zulf_spectrum
from spin_dynamics.relaxation import matrix_exponential


def _resolved_lines(frequencies, spectrum, lo, hi, rel_threshold=0.25):
    mask = (frequencies >= lo) & (frequencies <= hi)
    mag = np.abs(spectrum[mask])
    if mag.size < 3 or mag.max() <= 0:
        return 0, 0.0
    thr = rel_threshold * mag.max()
    interior = mag[1:-1]
    peaks = int(
        np.sum((interior > thr) & (interior > mag[:-2]) & (interior > mag[2:]))
    )
    return peaks, float(mag.max())


class IsotopeTests(unittest.TestCase):
    def test_registry_lookup_and_larmor(self) -> None:
        proton = nuclear_isotope("1H")
        self.assertEqual(proton.spin, 0.5)
        # Larmor at 1 T equals |gamma|/2pi; 14N (spin-1) is in the merged registry.
        self.assertAlmostEqual(larmor_frequency_hz("1H", 1.0), proton.gamma_hz_per_t)
        self.assertEqual(nuclear_isotope("14N").spin, 1.0)
        self.assertAlmostEqual(larmor_frequency_hz("1H", 50e-6), 2128.87, places=1)

    def test_unknown_isotope_raises(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown isotope"):
            nuclear_isotope("999Zz")


class MixedOperatorTests(unittest.TestCase):
    def test_hilbert_dimension_mixes_spins(self) -> None:
        self.assertEqual(hilbert_dimension([0.5, 0.5, 1.0]), 12)
        self.assertEqual(hilbert_dimension([0.5]), 2)

    def test_embedded_operator_obeys_su2_commutator(self) -> None:
        spins = [0.5, 1.0]
        ix = embedded_operator(spins, 1, "x")
        iy = embedded_operator(spins, 1, "y")
        iz = embedded_operator(spins, 1, "z")
        commutator = ix @ iy - iy @ ix
        np.testing.assert_allclose(commutator, 1j * iz, atol=1e-12)

    def test_dot_product_is_symmetric_and_hermitian(self) -> None:
        spins = [0.5, 0.5]
        dot = dot_product_operator(spins, 0, 1)
        np.testing.assert_allclose(dot, dot.conj().T, atol=1e-12)
        np.testing.assert_allclose(dot, dot_product_operator(spins, 1, 0), atol=1e-12)

    def test_product_operator_rejects_two_terms_on_one_spin(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most one term per spin"):
            product_operator([0.5, 0.5], [(0, "x"), (0, "y")])


class MultinuclearSystemTests(unittest.TestCase):
    def test_system_fills_registry_spins_and_larmor(self) -> None:
        system = multinuclear_system(
            ["1H", "19F", "14N"], np.zeros((3, 3)), 50e-6
        )
        self.assertEqual(system.spins, (0.5, 0.5, 1.0))
        self.assertEqual(system.dimension, 12)
        np.testing.assert_allclose(
            system.larmor_hz, system.gammas_hz_per_t * 50e-6
        )
        self.assertEqual(system.indices_for_isotope("19F"), (1,))

    def test_zeeman_only_hamiltonian_matches_larmor_frequencies(self) -> None:
        system = multinuclear_system(["1H", "19F"], np.zeros((2, 2)), 50e-6)
        hamiltonian = multinuclear_hamiltonian(system, coupling="secular")
        # With J = 0 the Hamiltonian is pure Zeeman; single-spin-flip gaps are
        # 2 pi nu_i. The four eigenvalues are +-pi(nu_H +- nu_F).
        eigenvalues = np.sort(np.linalg.eigvalsh(hamiltonian).real)
        nu_h, nu_f = system.larmor_hz
        expected = np.sort(
            2.0 * np.pi * np.array(
                [
                    -0.5 * (nu_h + nu_f),
                    -0.5 * (nu_h - nu_f),
                    0.5 * (nu_h - nu_f),
                    0.5 * (nu_h + nu_f),
                ]
            )
        )
        np.testing.assert_allclose(eigenvalues, expected, atol=1e-6)

    def test_isotropic_adds_flip_flop_to_secular(self) -> None:
        system = multinuclear_system(["1H", "19F"], [[0, 5.0], [5.0, 0]], 50e-6)
        iso = multinuclear_hamiltonian(system, coupling="isotropic", include_zeeman=False)
        sec = multinuclear_hamiltonian(system, coupling="secular", include_zeeman=False)
        spins = system.spins
        flip_flop = 2.0 * np.pi * 5.0 * (
            product_operator(spins, [(0, "x"), (1, "x")])
            + product_operator(spins, [(0, "y"), (1, "y")])
        )
        np.testing.assert_allclose(iso - sec, flip_flop, atol=1e-12)

    def test_invalid_coupling_matrix_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "symmetric"):
            multinuclear_system(["1H", "19F"], [[0, 1.0], [2.0, 0]], 50e-6)


class PerSpinRelaxationTests(unittest.TestCase):
    def _decay_rate(self, generator, operator, t):
        vec = operator.reshape(-1, order="F")
        evolved = (matrix_exponential(generator, t) @ vec).reshape(
            operator.shape, order="F"
        )
        overlap = np.trace(evolved @ operator.conj().T) / np.trace(
            operator @ operator.conj().T
        )
        return -np.log(abs(overlap)) / t

    def test_spin_half_rates_are_exact(self) -> None:
        spins = [0.5]
        r1, r2 = 3.0, 7.0
        generator = per_spin_relaxation_superoperator(spins, r1, r2)
        iz = embedded_operator(spins, 0, "z")
        i_plus = embedded_operator(spins, 0, "+")
        self.assertAlmostEqual(self._decay_rate(generator, iz, 0.1), r1, places=4)
        self.assertAlmostEqual(self._decay_rate(generator, i_plus, 0.1), r2, places=4)

    def test_relaxation_is_spin_local(self) -> None:
        spins = [0.5, 0.5]
        # Only spin 0 relaxes; spin 1 transverse magnetization is preserved.
        generator = per_spin_relaxation_superoperator(spins, [5.0, 0.0], [5.0, 0.0])
        i1_plus = embedded_operator(spins, 1, "+")
        self.assertAlmostEqual(self._decay_rate(generator, i1_plus, 0.2), 0.0, places=6)
        i0_plus = embedded_operator(spins, 0, "+")
        self.assertAlmostEqual(self._decay_rate(generator, i0_plus, 0.2), 5.0, places=4)

    def test_rate_bound_is_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least r1_per_second / 2"):
            per_spin_relaxation_superoperator([0.5], 10.0, 1.0)


class ZulfSpectrumTests(unittest.TestCase):
    def _spectrum(self, system, r14n, detect, n=8192):
        r = np.full(system.nspin, 0.2)
        r[-1] = r14n
        return simulate_zulf_spectrum(
            system,
            r1_per_second=r,
            r2_per_second=r,
            dwell_seconds=2e-4,
            n_points=n,
            detect_indices=detect,
            apodization_hz=0.8,
        )

    def test_heteronuclear_doublet_spacing_equals_j(self) -> None:
        # Weak-coupling AX pair (19F reporter, 1H) -> 1H doublet split by J(H,F).
        j = np.zeros((3, 3))
        j[0, 1] = j[1, 0] = 10.0
        system = multinuclear_system(["1H", "19F", "14N"], j, 50e-6)
        spec = self._spectrum(system, 0.0, system.indices_for_isotope("1H"))
        freqs, mag = spec.frequencies_hz, np.abs(spec.spectrum)
        nu_h = system.larmor_hz[0]
        window = (freqs > nu_h - 30) & (freqs < nu_h + 30)
        fw, mw = freqs[window], mag[window]
        lines = fw[1:-1][
            (mw[1:-1] > 0.3 * mw.max()) & (mw[1:-1] > mw[:-2]) & (mw[1:-1] > mw[2:])
        ]
        self.assertEqual(lines.size, 2)
        self.assertAlmostEqual(float(np.ptp(lines)), 10.0, delta=1.0)
        # Doublet centred on the proton Larmor frequency.
        self.assertAlmostEqual(float(np.mean(lines)), nu_h, delta=1.0)

    def test_quadrupolar_relaxation_collapses_and_self_decouples(self) -> None:
        j = np.zeros((3, 3))
        j[1, 2] = j[2, 1] = 37.0  # strong 19F-14N
        j[0, 1] = j[1, 0] = 8.0  # 1H-19F
        system = multinuclear_system(["1H", "19F", "14N"], j, 50e-6)
        f_idx = system.indices_for_isotope("19F")
        nu_f = system.larmor_hz[1]
        lo, hi = nu_f - 80.0, nu_f + 80.0

        s0 = self._spectrum(system, 0.0, f_idx)
        s_coal = self._spectrum(system, 250.0, f_idx)
        s_narrow = self._spectrum(system, 1e5, f_idx)
        lines0, amp0 = _resolved_lines(s0.frequencies_hz, s0.spectrum, lo, hi)
        lines_coal, amp_coal = _resolved_lines(s_coal.frequencies_hz, s_coal.spectrum, lo, hi)
        _, amp_narrow = _resolved_lines(s_narrow.frequencies_hz, s_narrow.spectrum, lo, hi)

        # The multiplet is resolved with no 14N relaxation and collapses at the
        # coalescence rate, then recovers amplitude in the extreme-narrowing limit.
        self.assertGreaterEqual(lines0, 3)
        self.assertLess(lines_coal, lines0)
        self.assertLess(amp_coal, amp0)
        self.assertGreater(amp_narrow, amp_coal)

    def test_fid_shape_and_validation(self) -> None:
        system = multinuclear_system(["1H", "19F"], [[0, 5.0], [5.0, 0]], 50e-6)
        times, fid = simulate_zulf_fid(
            system,
            r1_per_second=0.5,
            r2_per_second=0.5,
            dwell_seconds=1e-3,
            n_points=256,
        )
        self.assertEqual(times.shape, (256,))
        self.assertEqual(fid.shape, (256,))
        self.assertTrue(np.all(np.isfinite(fid)))
        with self.assertRaisesRegex(ValueError, "dwell_seconds"):
            simulate_zulf_fid(
                system, r1_per_second=0.5, r2_per_second=0.5,
                dwell_seconds=0.0, n_points=16,
            )


if __name__ == "__main__":
    unittest.main()
