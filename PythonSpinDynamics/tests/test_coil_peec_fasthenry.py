"""FastHenry interop: .inp export format (always) + a live comparison (Windows only).

The exporter tests run everywhere. The live comparison runs FastHenry through its COM
automation server and is skipped unless pywin32 and a registered ``FastHenry2.Document`` are
available (a Windows FastFieldSolvers install), so the suite stays green on CI.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields.coil_peec import Conductor  # noqa: E402
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER  # noqa: E402
from spin_dynamics.fields.fasthenry_interop import to_fasthenry_inp  # noqa: E402


def _fasthenry_available() -> bool:
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False
    try:
        doc = win32com.client.Dispatch("FastHenry2.Document")
        try:
            doc.Quit
        except Exception:
            pass
        return True
    except Exception:
        return False


class ExportFormatTests(unittest.TestCase):
    def test_round_wire_deck_structure(self) -> None:
        c = Conductor(np.array([[0, 0, 0], [0, 0, 0.05], [0.01, 0, 0.05]]),
                      wire_radius=0.5e-3, material=ANNEALED_COPPER)
        deck = to_fasthenry_inp(c, [1e6])
        self.assertIn(".units m", deck)
        self.assertIn(".Default sigma=", deck)
        self.assertIn(".external N0 N2", deck)         # first and last node are the port
        self.assertIn(".freq fmin=1000000", deck)
        self.assertTrue(deck.rstrip().endswith(".end"))
        self.assertEqual(deck.count("\nN"), 3)          # 3 nodes for 2 segments
        self.assertEqual(deck.count("\nE"), 2)          # 2 segments

    def test_round_wire_exports_equal_area_square(self) -> None:
        r = 0.5e-3
        c = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), wire_radius=r, material=ANNEALED_COPPER)
        deck = to_fasthenry_inp(c, [1e6])
        side = float(np.sqrt(np.pi) * r)  # equal cross-sectional area
        self.assertIn(f"w={side:.10g}", deck)
        self.assertIn(f"h={side:.10g}", deck)

    def test_rect_conductor_exported_exactly(self) -> None:
        c = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                      cross_section="rect", width=2e-3, height=0.5e-3, n_width=8, n_height=4)
        deck = to_fasthenry_inp(c, [1e5, 1e7])
        self.assertIn("w=0.002", deck)
        self.assertIn("h=0.0005", deck)
        self.assertIn("nwinc=8 nhinc=4", deck)
        self.assertIn(".freq fmin=100000 fmax=10000000", deck)


@unittest.skipUnless(_fasthenry_available(), "FastHenry2 COM automation not available")
class LiveComparisonTests(unittest.TestCase):
    def test_straight_bar_matches_fasthenry(self) -> None:
        from spin_dynamics.fields.fasthenry_interop import compare_with_fasthenry

        c = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                      cross_section="rect", width=1e-3, height=1e-3, n_width=8, n_height=8)
        freqs = [1e4, 1e5]
        peec, fh = compare_with_fasthenry(c, freqs)
        # Low-frequency inductance within ~8%, resistance within ~3% of FastHenry.
        for i in range(len(freqs)):
            self.assertLess(abs(peec.inductance[i] - fh.inductance[i]) / fh.inductance[i], 0.08)
            self.assertLess(abs(peec.resistance[i] - fh.resistance[i]) / fh.resistance[i], 0.03)


if __name__ == "__main__":
    unittest.main()
