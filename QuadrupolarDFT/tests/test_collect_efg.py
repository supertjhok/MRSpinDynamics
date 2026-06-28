"""Collect mode curvatures from real-format ABINIT EFG outputs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from quadrupolar_dft import (
    PhononMode,
    collect_efg_outputs,
    vibrational_modes_from_collected,
)
from quadrupolar_dft.constants import EFG_AU_TO_SI


def _abo(total_efg_au: np.ndarray) -> str:
    """A minimal ABINIT .abo EFG block in the real output format (atom 1)."""

    rows = "\n".join(
        f"      total efg :   {r[0]: .8f}   {r[1]: .8f}   {r[2]: .8f}"
        for r in total_efg_au
    )
    return (
        " Electric Field Gradient Calculation \n\n"
        " Atom   1, typat   1: Cq =     1.000000 MHz     eta =      0.100000\n\n"
        f"{rows}\n"
    )


class CollectTests(unittest.TestCase):
    def test_collect_and_recover_curvature_from_abo(self):
        delta_q = 1.0e-24
        v_eq = np.diag([0.40, -0.15, -0.25])
        delta = np.array(
            [[2.0e-3, 5.0e-4, 0.0], [5.0e-4, -1.0e-3, 0.0], [0.0, 0.0, -1.0e-3]]
        )
        # Pure quadratic: V(+/-) = V_eq + 1/2 C q^2, identical for +/- ; here we
        # write that perturbation directly as `delta` = 1/2 C q^2 (in au).
        v_disp = v_eq + delta

        manifest = {
            "target_atom_index": 0,
            "jobs": [
                {"name": "equilibrium", "mode_index": -1, "sign": 0, "delta_q_si": 0.0},
                {"name": "mode000_plus", "mode_index": 0, "sign": 1, "delta_q_si": delta_q},
                {"name": "mode000_minus", "mode_index": 0, "sign": -1, "delta_q_si": delta_q},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "equilibrium.abo").write_text(_abo(v_eq), encoding="utf-8")
            (directory / "mode000_plus.abo").write_text(_abo(v_disp), encoding="utf-8")
            (directory / "mode000_minus.abo").write_text(_abo(v_disp), encoding="utf-8")

            efg_by_job = collect_efg_outputs(manifest, directory)
            mode = PhononMode(150.0, np.array([[1.0, 0.0, 0.0]]))
            vib = vibrational_modes_from_collected([mode], manifest, efg_by_job)

        # Central difference of identical +/- displacements: (V+ - 2 V0 + V-)/q^2
        # = (2 * delta) / q^2, converted au -> SI.
        expected = (2.0 * delta) * EFG_AU_TO_SI / delta_q**2
        np.testing.assert_allclose(vib[0].efg_curvature_si, expected, rtol=1e-6)

    def test_missing_output_raises(self):
        manifest = {
            "target_atom_index": 0,
            "jobs": [{"name": "equilibrium", "mode_index": -1, "sign": 0, "delta_q_si": 0.0}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                collect_efg_outputs(manifest, tmp)


if __name__ == "__main__":
    unittest.main()
