"""Phase 1 acceleration tests: Numba arb10 backend and batched matrix power.

Two layers of validation:

* ``test_stacked_core_matches_numpy`` runs the compiled-kernel *algorithm* in
  pure Python (numba absent) on a tiny grid, so the stacked representation and
  segment-loop semantics are validated on every host;
* ``test_numba_backend_matches_numpy`` exercises the actual JIT path and is
  skipped where numba is unavailable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.core import _numba_kernels as nk  # noqa: E402
from spin_dynamics.core.kernels import (  # noqa: E402
    _matrix_elements_power,
    _stack_matrix_elements,
    get_arb10_backend,
    set_arb10_backend,
    sim_spin_dynamics_arb10,
)
from spin_dynamics.core.rotations import MatrixElements, rf_matrix_elements  # noqa: E402


def _tiny_arb10_params(numpts: int = 9) -> dict:
    """A small two-pulse acquisition with free-precession gradients."""

    del_w = np.linspace(-5.0, 5.0, numpts)
    exc = rf_matrix_elements(del_w, w1=1.0, tp=np.pi / 2, phi=0.0)
    ref = rf_matrix_elements(del_w, w1=1.0, tp=np.pi, phi=np.pi / 2)
    return {
        "tp": np.array([np.pi / 2, 1.0, np.pi, 1.3]),
        "pul": np.array([1, 0, 2, 0]),
        "amp": np.array([1.0, 0.0, 1.0, 0.0]),
        "acq": np.array([False, False, False, True]),
        "grad": np.array([0.0, 0.2, 0.0, 0.1]),
        "Rtot": [exc, ref],
        "del_w": del_w,
        "del_wg": np.ones(numpts),
        "w_1": np.ones(numpts),
        "T1n": np.full(numpts, 100.0),
        "T2n": np.full(numpts, 50.0),
        "m0": np.ones(numpts, dtype=np.complex128),
        "mth": np.zeros(numpts, dtype=np.complex128),
    }


class BatchedMatrixPowerTests(unittest.TestCase):
    def test_matrix_elements_power_matches_per_isochromat_reference(self) -> None:
        del_w = np.linspace(-4.0, 4.0, 11)
        mat = rf_matrix_elements(del_w, w1=1.3, tp=0.9, phi=0.4)
        exponent = 1.0 / 3.0
        powered = _matrix_elements_power(mat, exponent)

        # Independent per-isochromat reference.
        size = del_w.size
        expected = {name: np.empty(size, dtype=np.complex128) for name in (
            "R_00", "R_0m", "R_0p", "R_m0", "R_mm", "R_mp", "R_p0", "R_pm", "R_pp"
        )}
        for idx in range(size):
            full = np.array(
                [
                    [mat.R_00[idx], mat.R_0m[idx], mat.R_0p[idx]],
                    [mat.R_m0[idx], mat.R_mm[idx], mat.R_mp[idx]],
                    [mat.R_p0[idx], mat.R_pm[idx], mat.R_pp[idx]],
                ],
                dtype=np.complex128,
            )
            vals, vecs = np.linalg.eig(full)
            ref = vecs @ np.diag(vals**exponent) @ np.linalg.inv(vecs)
            for i, row in enumerate(("0", "m", "p")):
                for k, col in enumerate(("0", "m", "p")):
                    expected[f"R_{row}{col}"][idx] = ref[i, k]

        for name, ref_arr in expected.items():
            np.testing.assert_allclose(
                getattr(powered, name), ref_arr, rtol=1e-10, atol=1e-12, err_msg=name
            )

    def test_matrix_power_one_is_identity_transform(self) -> None:
        del_w = np.linspace(-3.0, 3.0, 7)
        mat = rf_matrix_elements(del_w, w1=0.8, tp=1.1, phi=0.2)
        powered = _matrix_elements_power(mat, 1.0)
        for name in ("R_00", "R_pp", "R_mm", "R_pm", "R_0p", "R_m0"):
            np.testing.assert_allclose(
                getattr(powered, name), getattr(mat, name), rtol=1e-9, atol=1e-12
            )


class StackedCoreTests(unittest.TestCase):
    def test_stack_matrix_elements_round_trips(self) -> None:
        del_w = np.linspace(-2.0, 2.0, 5)
        mat = rf_matrix_elements(del_w, w1=1.0, tp=1.0, phi=0.3)
        stack = _stack_matrix_elements([mat], del_w.size)
        self.assertEqual(stack.shape, (1, 3, 3, del_w.size))
        np.testing.assert_array_equal(stack[0, 0, 0], np.asarray(mat.R_00, np.complex128))
        np.testing.assert_array_equal(stack[0, 1, 2], np.asarray(mat.R_mp, np.complex128))
        np.testing.assert_array_equal(stack[0, 2, 0], np.asarray(mat.R_p0, np.complex128))

    def test_stacked_core_matches_numpy(self) -> None:
        """Validate the compiled-kernel algorithm (pure-Python on every host)."""

        params = _tiny_arb10_params()
        reference = sim_spin_dynamics_arb10(params, backend="numpy")
        rstack = _stack_matrix_elements(params["Rtot"], params["del_w"].size)
        core_out = nk.arb10_core(
            params["tp"],
            params["pul"].astype(np.int64),
            params["amp"],
            params["acq"].astype(np.uint8),
            params["grad"],
            params["del_w"],
            params["del_wg"],
            params["T1n"],
            params["T2n"],
            params["m0"],
            params["mth"],
            rstack,
            int(np.sum(params["acq"])),
        )
        np.testing.assert_allclose(core_out, reference, rtol=1e-12, atol=1e-14)


class NumbaBackendTests(unittest.TestCase):
    def tearDown(self) -> None:
        set_arb10_backend("numpy")

    def test_backend_selector_round_trips(self) -> None:
        self.assertEqual(get_arb10_backend(), "numpy")
        set_arb10_backend("numba")
        self.assertEqual(get_arb10_backend(), "numba")
        set_arb10_backend("numpy")
        with self.assertRaises(ValueError):
            set_arb10_backend("gpu")

    @unittest.skipUnless(nk.NUMBA_AVAILABLE, "numba not installed")
    def test_numba_backend_matches_numpy(self) -> None:
        params = _tiny_arb10_params(numpts=33)
        reference = sim_spin_dynamics_arb10(params, backend="numpy")
        compiled = sim_spin_dynamics_arb10(params, backend="numba")
        np.testing.assert_allclose(compiled, reference, rtol=1e-10, atol=1e-12)

    @unittest.skipUnless(nk.NUMBA_AVAILABLE, "numba not installed")
    def test_global_default_backend_is_honored(self) -> None:
        params = _tiny_arb10_params(numpts=17)
        reference = sim_spin_dynamics_arb10(params, backend="numpy")
        set_arb10_backend("numba")
        through_default = sim_spin_dynamics_arb10(params)
        np.testing.assert_allclose(through_default, reference, rtol=1e-10, atol=1e-12)

    @unittest.skipIf(nk.NUMBA_AVAILABLE, "numba is installed")
    def test_numba_backend_errors_without_numba(self) -> None:
        params = _tiny_arb10_params()
        with self.assertRaises(ImportError):
            sim_spin_dynamics_arb10(params, backend="numba")


if __name__ == "__main__":
    unittest.main()
