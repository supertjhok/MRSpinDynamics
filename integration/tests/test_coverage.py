"""Tests for the structure -> DFT -> database target survey."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mr_integration import (
    compare_available_dft,
    format_target_survey,
    load_dft_summary,
    survey_integration_targets,
)
from mr_integration.database import default_database_path

_HAS_DB = default_database_path().exists()


class DFTSummaryTests(unittest.TestCase):
    def test_load_minimal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.write_text(
                "case_id,title,isotope,spin,atoms,mean_cq_mhz,"
                "mean_abs_cq_mhz,mean_eta\n"
                "case_a,Example,14N,1,3 4,-5.0,5.0,0.2\n",
                encoding="utf-8",
            )
            records = load_dft_summary(path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].atom_indices, (3, 4))
        self.assertEqual(records[0].mean_cq_hz, -5.0e6)
        self.assertEqual(records[0].mean_abs_cq_hz, 5.0e6)
        self.assertAlmostEqual(records[0].mean_eta, 0.2)

    def test_missing_summary_is_empty(self) -> None:
        self.assertEqual(load_dft_summary("does-not-exist.csv"), ())


@unittest.skipUnless(_HAS_DB, "NQR SQLite export not available")
class RepositoryCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.targets = survey_integration_targets()

    def test_repository_target_statuses(self) -> None:
        by_name = {target.spec.structure_name: target for target in self.targets}
        self.assertEqual(len(by_name), 10)
        self.assertEqual(by_name["NaNO2"].status, "comparison-ready")
        self.assertEqual(by_name["Melamine"].status, "dft-needed")
        self.assertEqual(by_name["Acetaminophen"].status, "dft-needed")
        self.assertEqual(
            by_name["Caffeine"].status,
            "isotope-metadata-needed",
        )
        self.assertEqual(
            by_name["Hexamethylenetetramine"].status,
            "measurement-needed",
        )

    def test_ready_queue_contains_five_new_materials(self) -> None:
        queued = {
            target.spec.structure_name
            for target in self.targets
            if target.status == "dft-needed"
        }
        self.assertEqual(
            queued,
            {"Acetaminophen", "Famotidine", "Glycine", "L-Proline", "Melamine"},
        )

    def test_available_summary_rows_run_end_to_end(self) -> None:
        comparisons = compare_available_dft(self.targets)
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(
            {comparison.dft_record.case_id for comparison in comparisons},
            {"nano2_icsd82857_efg", "nano2_starter"},
        )
        for comparison in comparisons:
            self.assertEqual(comparison.report.compound, "Sodium Nitrite")
            self.assertEqual(comparison.report.isotope, "14N")
            self.assertEqual(len(comparison.report.measured), 3)

    def test_survey_format_exposes_next_action(self) -> None:
        table = format_target_survey(self.targets)
        self.assertIn("Sodium Nitrite", table)
        self.assertIn("comparison-ready", table)
        self.assertIn("dft-needed", table)
        self.assertIn("isotope-metadata-needed", table)


if __name__ == "__main__":
    unittest.main()
