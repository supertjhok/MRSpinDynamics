"""Checks for the structured validation-evidence registry."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "docs" / "generate_validation_matrix.py"
SPEC = importlib.util.spec_from_file_location("generate_validation_matrix", GENERATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class ValidationEvidenceTests(unittest.TestCase):
    def test_registry_is_valid_and_has_all_evidence_levels(self) -> None:
        data = GENERATOR.load_registry()
        self.assertGreaterEqual(len(data["records"]), 15)
        self.assertEqual(set(data["evidence_levels"]), {"A", "B", "C", "D", "E", "R"})

    def test_generated_documents_are_current(self) -> None:
        self.assertEqual(GENERATOR.main(["--check"]), 0)


if __name__ == "__main__":
    unittest.main()
