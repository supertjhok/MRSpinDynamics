"""Phase 2 acceleration tests: JAX arb10 backend parity with NumPy.

The JAX path is skipped where ``jax`` is not installed. Parity is checked
within the same interpreter (jax vs numpy), so it is independent of NumPy
version, and against an end-to-end CPMG train so workflow assembly is covered.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import _perf_scenarios as scenarios  # noqa: E402

from spin_dynamics.core import _jax_kernels as jk  # noqa: E402
from spin_dynamics.core.kernels import (  # noqa: E402
    set_arb10_backend,
    sim_spin_dynamics_arb10,
    sim_spin_dynamics_arb10_batched,
)
from spin_dynamics.workflows import run_ideal_cpmg_train  # noqa: E402


class JaxBackendTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_arb10_backend("numpy")

    def test_jax_is_a_valid_backend_selection(self) -> None:
        # Selecting jax must succeed even if jax is absent; the error (if any)
        # surfaces only when a simulation is actually run.
        set_arb10_backend("jax")
        set_arb10_backend("numpy")

    @unittest.skipIf(jk.JAX_AVAILABLE, "jax is installed")
    def test_jax_backend_errors_without_jax(self) -> None:
        params = scenarios.tiny_arb10_params()
        with self.assertRaises(ImportError):
            sim_spin_dynamics_arb10(params, backend="jax")

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_jax_backend_matches_numpy_raw_kernel(self) -> None:
        params = scenarios.tiny_arb10_params(numpts=33)
        reference = sim_spin_dynamics_arb10(params, backend="numpy")
        compiled = sim_spin_dynamics_arb10(params, backend="jax")
        np.testing.assert_allclose(compiled, reference, rtol=1e-9, atol=1e-11)

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_jax_x64_is_enabled(self) -> None:
        import jax.numpy as jnp

        # Without x64, this would be float32; complex128 parity depends on it.
        self.assertEqual(jnp.asarray(1.0).dtype, np.float64)

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_jax_backend_matches_numpy_cpmg_workflow(self) -> None:
        kwargs = dict(
            numpts=129,
            maxoffs=8.0,
            num_echoes=6,
            t1_seconds=1.7,
            t2_seconds=1.1,
            num_workers=1,
            auto_refine_grid=False,
            rephase_action="ignore",
        )
        reference = run_ideal_cpmg_train(**kwargs)
        set_arb10_backend("jax")
        compiled = run_ideal_cpmg_train(**kwargs)
        np.testing.assert_allclose(
            compiled.mrx, reference.mrx, rtol=1e-9, atol=1e-11
        )
        np.testing.assert_allclose(
            compiled.echo_integrals, reference.echo_integrals, rtol=1e-9, atol=1e-11
        )


class JaxBatchedTests(unittest.TestCase):
    @staticmethod
    def _cases():
        base = scenarios.tiny_arb10_params(numpts=17)
        cases = []
        for factor in (1.0, 0.7, 1.3):
            case = dict(base)
            case["T2n"] = base["T2n"] * factor
            case["m0"] = base["m0"] * factor
            cases.append(case)
        return cases

    @unittest.skipIf(jk.JAX_AVAILABLE, "jax is installed")
    def test_batched_errors_without_jax(self) -> None:
        with self.assertRaises(ImportError):
            sim_spin_dynamics_arb10_batched(self._cases())

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_batched_matches_looped_singles(self) -> None:
        cases = self._cases()
        batched = sim_spin_dynamics_arb10_batched(cases)
        self.assertEqual(batched.shape[0], len(cases))
        for idx, case in enumerate(cases):
            single = sim_spin_dynamics_arb10(case, backend="numpy")
            np.testing.assert_allclose(
                batched[idx], single, rtol=1e-9, atol=1e-11, err_msg=f"case {idx}"
            )

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_batched_rejects_mismatched_structure(self) -> None:
        cases = self._cases()
        bad = dict(cases[0])
        bad["tp"] = cases[0]["tp"] * 2.0  # different segment timing
        with self.assertRaises(ValueError):
            sim_spin_dynamics_arb10_batched([cases[0], bad])


if __name__ == "__main__":
    unittest.main()
