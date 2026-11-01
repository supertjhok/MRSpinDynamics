from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "validation" / "audit_nqr_relaxation.py"
SPEC = importlib.util.spec_from_file_location("audit_nqr_relaxation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


class NQRDatabaseRelaxationAuditTests(unittest.TestCase):
    def test_inventory_and_basic_plausibility(self) -> None:
        report = AUDIT.audit_relaxation_records()
        coverage = report["coverage"]
        plausibility = report["plausibility"]

        self.assertEqual(coverage["relaxation_records"], 89)
        self.assertEqual(coverage["observable_ranges"]["t1_s"]["count"], 84)
        self.assertEqual(coverage["observable_ranges"]["t2_s"]["count"], 50)
        self.assertEqual(coverage["observable_ranges"]["t2_star_s"]["count"], 3)
        self.assertEqual(plausibility["nonpositive_values"], [])
        self.assertEqual(plausibility["t2_greater_than_t1_line_ids"], [])

    def test_no_cross_source_scale_conflicts_remain(self) -> None:
        # The CWRU HTML sodium-nitrite T1/T2 cells carry no unit and are in
        # milliseconds; they used to be normalized as seconds, a factor-1000
        # conflict with the NRL summary (6 values). NQRDatabase now applies the
        # CWRU millisecond convention, so the duplicate sources agree.
        report = AUDIT.audit_relaxation_records()
        conflicts = report["plausibility"]["cross_source_scale_conflicts"]

        self.assertEqual(conflicts, [])


if __name__ == "__main__":
    unittest.main()
