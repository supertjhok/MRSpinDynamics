"""FasterCap interop: panel export format (always) + a live capacitance comparison (Windows).

The panelizer tests run everywhere. The live comparison runs FasterCap through its COM
automation server and is skipped unless pywin32 and a registered ``FasterCap.Document`` are
available, so the suite stays green on CI.
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
from spin_dynamics.fields.fastercap_interop import to_fastercap_panels  # noqa: E402


def _fastercap_available() -> bool:
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False
    try:
        doc = win32com.client.gencache.EnsureDispatch("FasterCap.Document")
        # Method names only resolve if the type library is registered.
        _ = doc.IsRunning
        try:
            doc.Quit()
        except Exception:
            pass
        return True
    except Exception:
        return False


def _straight_wire(a=1e-3, length=1.0, n=200):
    pts = np.column_stack([np.zeros(n), np.zeros(n), np.linspace(0, length, n)])
    return Conductor(pts, wire_radius=a, material=ANNEALED_COPPER)


class PanelExportTests(unittest.TestCase):
    def test_panel_deck_structure(self) -> None:
        c = _straight_wire(n=6)
        deck = to_fastercap_panels(c, n_theta=8, name="wire")
        lines = deck.strip().splitlines()
        self.assertTrue(lines[0].startswith("0 "))          # comment header
        quads = [x for x in lines if x.startswith("Q wire")]
        tris = [x for x in lines if x.startswith("T wire")]
        # 5 path segments x 8 facets side quads; 2 x 8 end-cap triangles.
        self.assertEqual(len(quads), 5 * 8)
        self.assertEqual(len(tris), 2 * 8)
        # A quad line is 'Q name' + 12 coordinates.
        self.assertEqual(len(quads[0].split()) - 2, 12)
        self.assertEqual(len(tris[0].split()) - 2, 9)

    def test_rect_conductor_rejected(self) -> None:
        c = Conductor(np.array([[0, 0, 0], [0, 0, 0.1]]), material=ANNEALED_COPPER,
                      cross_section="rect", width=1e-3, height=1e-3)
        with self.assertRaises(NotImplementedError):
            to_fastercap_panels(c)


@unittest.skipUnless(_fastercap_available(), "FasterCap COM automation not available")
class LiveCapacitanceTests(unittest.TestCase):
    def test_wire_capacitance_matches_fastercap(self) -> None:
        from spin_dynamics.fields.fastercap_interop import compare_capacitance_with_fastercap

        c = _straight_wire(a=1e-3, length=1.0)
        peec_c, fc_c = compare_capacitance_with_fastercap(c, n_theta=24)
        # PEEC thin-wire self-capacitance within ~5% of the FasterCap surface solve.
        self.assertLess(abs(peec_c - fc_c) / fc_c, 0.05)


if __name__ == "__main__":
    unittest.main()
