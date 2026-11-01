"""Tests for the structure -> DFT -> database target survey."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mr_integration import (
    IntegrationTargetCoverage,
    IntegrationTargetSpec,
    MeasuredLine,
    compare_available_dft,
    format_target_survey,
    load_dft_summary,
    survey_integration_targets,
)
from mr_integration.database import default_database_path

_SUMMARY_HEADER = (
    "case_id,title,isotope,spin,atoms,mean_cq_mhz,mean_abs_cq_mhz,mean_eta,"
    "nqr_transitions_mhz\n"
)

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

    def test_missing_summary_raises_instead_of_returning_empty(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_dft_summary("does-not-exist.csv")

    def test_missing_default_summary_names_the_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {"MR_INTEGRATION_REPO_ROOT": directory}):
                with self.assertRaises(FileNotFoundError) as caught:
                    load_dft_summary()
                with self.assertRaises(FileNotFoundError):
                    survey_integration_targets()
        self.assertIn("MR_INTEGRATION_DFT_SUMMARY", str(caught.exception))
        self.assertIn("MR_INTEGRATION_REPO_ROOT", str(caught.exception))

    def test_env_override_points_at_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elsewhere.csv"
            path.write_text(_SUMMARY_HEADER + "case_a,Example,14N,1,3,-5.0,5.0,0.2,\n",
                            encoding="utf-8")
            with mock.patch.dict(os.environ, {"MR_INTEGRATION_DFT_SUMMARY": str(path)}):
                records = load_dft_summary()
        self.assertEqual([record.case_id for record in records], ["case_a"])

    def test_per_atom_lines_identify_non_equivalent_sites(self) -> None:
        rows = (
            # Two 14N atoms agreeing to ~75 Hz: one site (unsymmetrized noise).
            'eq,Eq,14N,1,3 4,-5.03,5.03,0.11,"0.281699, 3.634708, 3.916407; '
            '0.281643, 3.634689, 3.916332"\n'
            # Two 17O atoms differing by ~2%: averaging two different sites.
            'neq,Neq,17O,2.5,5 6,11.1,11.1,0.44,"1.928148, 3.288072, 5.216220; '
            '2.047416, 3.157246, 5.204662"\n'
            # No per-atom data: cannot be checked.
            "unk,Unk,14N,1,3,-5.0,5.0,0.2,\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.csv"
            path.write_text(_SUMMARY_HEADER + rows, encoding="utf-8")
            records = {record.case_id: record for record in load_dft_summary(path)}

        self.assertTrue(records["eq"].sites_equivalent)
        self.assertEqual(len(records["eq"].per_atom_lines_hz), 2)
        self.assertFalse(records["neq"].sites_equivalent)
        self.assertIsNone(records["unk"].sites_equivalent)

        target = IntegrationTargetCoverage(
            spec=IntegrationTargetSpec("X", "X"),
            structure_files=(),
            measured=(
                MeasuredLine("X", "14N", None, 3.6e6, None, None, None, None),
                MeasuredLine("X", "17O", None, 3.2e6, None, None, None, None),
            ),
            dft_records=tuple(records.values()),
        )
        self.assertEqual(
            {record.case_id for record in target.comparison_ready_records},
            {"eq", "unk"},
        )


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
