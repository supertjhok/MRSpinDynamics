"""Convergence-study driver: ABINIT input rewriting + eta/C_Q tabulation."""

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

_EXAMPLE = (
    Path(__file__).resolve().parents[1] / "examples" / "abinit" / "efg_convergence.py"
)
_spec = importlib.util.spec_from_file_location("efg_convergence", _EXAMPLE)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

BASE_ABI = """# starter
ecut 25
pawecutdg 50
ngkpt 4 4 4
nsym 1
nucefg 2
"""


def _abo(atom: int, typat: int, total_efg_au: np.ndarray) -> str:
    """A minimal ABINIT .abo EFG block (compact atom header + total tensor)."""

    rows = "\n".join(
        f"      total efg :   {r[0]: .8f}   {r[1]: .8f}   {r[2]: .8f}"
        for r in total_efg_au
    )
    # Cq/eta on the header line are cross-check values only; collect recomputes
    # from the total tensor, so their exact values here do not matter.
    return (
        " Electric Field Gradient Calculation \n\n"
        f" Atom   {atom}, typat   {typat}: Cq =     1.000000 MHz     eta =      0.100000\n\n"
        f"{rows}\n"
    )


class RewriteTests(unittest.TestCase):
    def test_set_scalar_replaces_only_target_not_substring(self):
        out = conv.set_scalar(BASE_ABI, "ecut", 35)
        self.assertIn("ecut 35", out)
        # pawecutdg must be untouched even though it contains "ecut".
        self.assertIn("pawecutdg 50", out)
        self.assertEqual(out.count("ecut 35"), 1)

    def test_set_scalar_preserves_inline_comment(self):
        text = "ecut 25 # starter cutoff\n"
        out = conv.set_scalar(text, "ecut", 40)
        self.assertEqual(out, "ecut 40 # starter cutoff\n")

    def test_set_scalar_appends_when_absent(self):
        out = conv.set_scalar("nsym 1\n", "pawecutdg", 80)
        self.assertIn("pawecutdg 80", out)

    def test_set_and_get_ngkpt(self):
        out = conv.set_ngkpt(BASE_ABI, 6, 6, 6)
        self.assertIn("ngkpt 6 6 6", out)
        self.assertEqual(conv.get_ngkpt(out), [6, 6, 6])

    def test_get_scalar_reads_baseline(self):
        self.assertEqual(conv.get_scalar(BASE_ABI, "ecut"), 25.0)
        self.assertEqual(conv.get_scalar(BASE_ABI, "pawecutdg"), 50.0)
        self.assertIsNone(conv.get_scalar(BASE_ABI, "nstep"))

    def test_parse_ngkpt_token(self):
        self.assertEqual(conv._parse_ngkpt_token("6"), (6, 6, 6))
        self.assertEqual(conv._parse_ngkpt_token("4x6x6"), (4, 6, 6))
        with self.assertRaises(ValueError):
            conv._parse_ngkpt_token("4x6")


class GenerateCollectTests(unittest.TestCase):
    def test_generate_dedupes_baseline_and_sets_cutoffs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.abi"
            base.write_text(BASE_ABI, encoding="utf-8")
            out = Path(tmp) / "conv"
            conv.cmd_generate(
                argparse.Namespace(
                    base=str(base), target=2, out=str(out),
                    ecut="25,35", pawecutdg="50,80", ngkpt="4,6",
                )
            )
            names = sorted(p.stem for p in out.glob("*.abi"))
            # baseline once; the ecut=25/pawecutdg=50/ngkpt=4 baseline values are
            # deduped, leaving one off-baseline variant per knob.
            self.assertEqual(
                names, ["baseline", "ecut_35", "ngkpt_6x6x6", "pawecutdg_80"]
            )
            ecut35 = (out / "ecut_35.abi").read_text(encoding="utf-8")
            self.assertIn("ecut 35", ecut35)
            self.assertIn("pawecutdg 50", ecut35)  # held at baseline
            ngkpt6 = (out / "ngkpt_6x6x6.abi").read_text(encoding="utf-8")
            self.assertIn("ngkpt 6 6 6", ngkpt6)

    def test_collect_round_trip_tabulates_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.abi"
            base.write_text(BASE_ABI, encoding="utf-8")
            out = Path(tmp) / "conv"
            conv.cmd_generate(
                argparse.Namespace(
                    base=str(base), target=2, out=str(out),
                    ecut=None, pawecutdg="50,80", ngkpt=None,
                )
            )
            # Two jobs: baseline (pawecutdg 50) and pawecutdg_80. Give them
            # different traceless tensors so eta differs across the ladder.
            eta_low = np.diag([0.40, -0.16, -0.24])          # eta = 0.20
            eta_high = np.diag([0.40, -0.10, -0.30])         # eta = 0.50
            for name, tensor in (("baseline", eta_low), ("pawecutdg_80", eta_high)):
                (out / f"{name}.abo").write_text(
                    _abo(3, 2, tensor), encoding="utf-8"  # atom index 3 = target 2 + 1
                )
            report_path = out / "convergence.md"
            csv_path = out / "convergence.csv"
            conv.cmd_collect(
                argparse.Namespace(
                    workdir=str(out), quadmom=0.02044, suffix=".abo",
                    eta_tol=0.01, allow_missing=False,
                    out=str(report_path), csv=str(csv_path),
                )
            )
            report = report_path.read_text(encoding="utf-8")
            self.assertIn("## Sweep: pawecutdg", report)
            self.assertIn("0.2000", report)  # baseline eta
            self.assertIn("0.5000", report)  # pawecutdg 80 eta
            self.assertIn("+0.3000", report)  # d(eta) across the ladder
            self.assertIn("has **not** settled", report)  # 0.30 step > tol
            csv_text = csv_path.read_text(encoding="utf-8")
            self.assertIn("pawecutdg_80", csv_text)


if __name__ == "__main__":
    unittest.main()
