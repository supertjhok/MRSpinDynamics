"""Phase 4: batched NQR diagonalization parity with per-orientation diagonalize_site."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.nqr import _jax_eigh as je  # noqa: E402
from spin_dynamics.nqr.hamiltonians import (  # noqa: E402
    diagonalize_site,
    diagonalize_sites_over_b0,
)
from spin_dynamics.nqr.systems import QuadrupolarSite  # noqa: E402
from spin_dynamics.nqr.zeeman import simulate_weak_b0_spectrum  # noqa: E402


def _powder_b0_vectors(n: int, magnitude: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return magnitude * directions


SITE_SPIN1 = QuadrupolarSite(
    spin=1.0, quadrupole_frequency_hz=3.75e6, eta=0.2, gamma_hz_per_t=3.077e6
)
SITE_SPIN32 = QuadrupolarSite(
    spin=1.5, quadrupole_frequency_hz=30.0e6, eta=0.1, gamma_hz_per_t=4.17e6
)


class BatchedDiagonalizationTests(unittest.TestCase):
    def _assert_matches_loop(self, site: QuadrupolarSite) -> None:
        b0_vectors = _powder_b0_vectors(40, magnitude=1.0e-3)
        batched = diagonalize_sites_over_b0(site, b0_vectors)
        self.assertEqual(len(batched), b0_vectors.shape[0])
        for idx, eig in enumerate(batched):
            ref = diagonalize_site(site, b0_vectors[idx])
            np.testing.assert_allclose(eig.levels_hz, ref.levels_hz, rtol=1e-9, atol=1e-3)
            got = np.array([t.frequency_hz for t in eig.transitions])
            want = np.array([t.frequency_hz for t in ref.transitions])
            self.assertEqual(got.shape, want.shape)
            np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-3)
            got_s = np.array([t.strength for t in eig.transitions])
            want_s = np.array([t.strength for t in ref.transitions])
            np.testing.assert_allclose(got_s, want_s, rtol=1e-8, atol=1e-10)

    def test_batched_matches_loop_spin1(self) -> None:
        self._assert_matches_loop(SITE_SPIN1)

    def test_batched_matches_loop_spin32(self) -> None:
        self._assert_matches_loop(SITE_SPIN32)

    @unittest.skipUnless(je.JAX_AVAILABLE, "jax not installed")
    def test_jax_backend_matches_numpy(self) -> None:
        b0_vectors = _powder_b0_vectors(24, magnitude=1.0e-3)
        ref = diagonalize_sites_over_b0(SITE_SPIN1, b0_vectors, backend="numpy")
        jax_batched = diagonalize_sites_over_b0(SITE_SPIN1, b0_vectors, backend="jax")
        for a, b in zip(ref, jax_batched):
            np.testing.assert_allclose(a.levels_hz, b.levels_hz, rtol=1e-6, atol=1e-2)
            fa = np.array([t.frequency_hz for t in a.transitions])
            fb = np.array([t.frequency_hz for t in b.transitions])
            np.testing.assert_allclose(fa, fb, rtol=1e-6, atol=1e-1)

    def test_weak_b0_spectrum_runs_through_batched_path(self) -> None:
        result = simulate_weak_b0_spectrum(
            SITE_SPIN1,
            1.0e-3,
            orientations="powder",
            points=128,
            weak_ratio_action="ignore",
        )
        self.assertEqual(result.spectrum.shape, (128,))
        self.assertGreater(float(np.max(result.spectrum)), 0.0)

    @unittest.skipUnless(je.JAX_AVAILABLE, "jax not installed")
    def test_weak_b0_spectrum_jax_matches_numpy(self) -> None:
        kw = dict(orientations="powder", points=128, weak_ratio_action="ignore")
        ref = simulate_weak_b0_spectrum(SITE_SPIN1, 1.0e-3, **kw)
        got = simulate_weak_b0_spectrum(SITE_SPIN1, 1.0e-3, backend="jax", **kw)
        np.testing.assert_allclose(got.spectrum, ref.spectrum, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
