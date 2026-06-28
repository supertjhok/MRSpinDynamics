"""Phase 2b: the batched ideal CPMG relaxation sweep must match looped singles."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core import _jax_kernels as jk  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    run_ideal_cpmg_relaxation_sweep,
    run_ideal_cpmg_train,
)

T1 = [2.0, 1.5, 3.0]
T2 = [1.0, 0.8, 1.2]
KW = dict(numpts=65, maxoffs=10.0, num_echoes=4, rephase_action="ignore")


class BatchedSweepTests(unittest.TestCase):
    @unittest.skipIf(jk.JAX_AVAILABLE, "jax is installed")
    def test_sweep_errors_without_jax(self) -> None:
        with self.assertRaises(ImportError):
            run_ideal_cpmg_relaxation_sweep(T1, T2, **KW)

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_sweep_matches_looped_single_trains(self) -> None:
        sweep = run_ideal_cpmg_relaxation_sweep(T1, T2, **KW)
        self.assertEqual(sweep.mrx.shape[0], len(T1))
        for idx, (t1, t2) in enumerate(zip(T1, T2)):
            single = run_ideal_cpmg_train(
                t1_seconds=t1, t2_seconds=t2, num_workers=1, **KW
            )
            np.testing.assert_allclose(
                sweep.mrx[idx], single.mrx, rtol=1e-8, atol=1e-10, err_msg=f"mrx {idx}"
            )
            np.testing.assert_allclose(
                sweep.echo_integrals[idx],
                single.echo_integrals,
                rtol=1e-8,
                atol=1e-10,
                err_msg=f"integrals {idx}",
            )

    @unittest.skipUnless(jk.JAX_AVAILABLE, "jax not installed")
    def test_sweep_validates_shapes(self) -> None:
        with self.assertRaises(ValueError):
            run_ideal_cpmg_relaxation_sweep([2.0, 1.0], [1.0], **KW)


if __name__ == "__main__":
    unittest.main()
