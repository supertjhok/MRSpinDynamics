"""Regression checks for the 14N NQR mail-screening Phase 0 artifacts."""

from __future__ import annotations

import importlib.util
import copy
from unittest.mock import patch
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT / "studies" / "nqr_mail_screening" / "phase0" / "validate_phase0.py"
)
SPEC = importlib.util.spec_from_file_location(
    "validate_nqr_mail_screening_phase0", VALIDATOR_PATH
)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class NQRMailScreeningPhase0Tests(unittest.TestCase):
    def test_study_defaults_are_valid_and_ready(self) -> None:
        VALIDATOR.validate_schemas()
        VALIDATOR.validate_evidence()
        unresolved = VALIDATOR.validate_requirements(require_ready=False)
        VALIDATOR.validate_materials()
        VALIDATOR.validate_result_example()

        self.assertEqual(unresolved, [])
        VALIDATOR.validate_study_readiness()
        self.assertTrue(VALIDATOR.validate_approval())

    def test_readiness_check_rejects_unresolved_requirements(self) -> None:
        docs = self._documents()
        docs["requirements.json"]["requirements"][0].update(
            status="unresolved", value=None
        )
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "Gate 0 is not ready"
            ):
                VALIDATOR.validate_requirements(require_ready=True)

    def test_provisional_input_requires_provenance(self):
        docs = self._documents()
        docs["requirements.json"]["requirements"][0]["origin"]["reference"] = None
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with self.assertRaises(VALIDATOR.ValidationError):
                VALIDATOR.validate_requirements(True)

    def test_study_snapshot_rejects_changed_artifact(self):
        with patch.object(VALIDATOR, "_artifact_sha256", return_value="0" * 64):
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "study snapshot stale"
            ):
                VALIDATOR.validate_study_readiness()

    def test_study_snapshot_requires_complete_artifact_set(self):
        docs = self._documents()
        docs["gate0_approval.json"]["artifact_sha256"] = {}
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "artifact set mismatch"
            ):
                VALIDATOR.validate_study_readiness()

    def _documents(self):
        names = [
            "requirements.json",
            "materials.json",
            "result_record.example.json",
            "gate0_approval.json",
        ]
        return {name: VALIDATOR.load_json(name) for name in names}

    def _loader(self, documents):
        original = VALIDATOR.load_json
        return lambda name: (
            copy.deepcopy(documents[name]) if name in documents else original(name)
        )

    def test_schema_rejects_missing_field_unknown_field_and_bad_date(self):
        for mutation in ("missing", "extra", "date"):
            with self.subTest(mutation=mutation):
                docs = self._documents()
                record = docs["result_record.example.json"]
                if mutation == "missing":
                    del record["scenario"]["random_seed"]
                elif mutation == "extra":
                    record["decision"]["unsupported"] = True
                else:
                    record["created_utc"] = "yesterday"
                with patch.object(
                    VALIDATOR, "load_json", side_effect=self._loader(docs)
                ):
                    with self.assertRaises(VALIDATOR.ValidationError):
                        VALIDATOR.validate_schemas()

    def test_semantics_reject_bad_limits_and_removed_blockers(self):
        for mutation in (
            "negative",
            "boolean",
            "zero_confidence",
            "remove",
            "nonblocking",
            "units",
        ):
            with self.subTest(mutation=mutation):
                docs = self._documents()
                records = docs["requirements.json"]["requirements"]
                by_id = {r["id"]: r for r in records}
                record = by_id["req.hardware.max_coil_current"]
                if mutation in {"negative", "boolean"}:
                    record.update(
                        status="provisional",
                        value=-1 if mutation == "negative" else True,
                    )
                elif mutation == "zero_confidence":
                    by_id["req.performance.confidence_level"].update(
                        status="provisional", value=0
                    )
                elif mutation == "remove":
                    records.pop()
                elif mutation == "nonblocking":
                    record["blocking_gate_0"] = False
                else:
                    record["units"] = "kW"
                with patch.object(
                    VALIDATOR, "load_json", side_effect=self._loader(docs)
                ):
                    with self.assertRaises(VALIDATOR.ValidationError):
                        VALIDATOR.validate_requirements(False)

    def test_approval_requires_signoff_and_rejects_stale_artifacts(self):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "Gate 0 is not ready"):
            VALIDATOR.validate_approval(True)
        docs = self._documents()
        approval = docs["gate0_approval.json"]
        approval.update(
            status="approved",
            approved_by="test-fixture-only",
            approved_utc="2026-09-05T12:00:00Z",
            approval_reference="fixture",
        )
        approval["artifact_sha256"] = {
            name: "0" * 64 for name in approval["artifact_sha256"]
        }
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "approval stale"):
                VALIDATOR.validate_approval(True)

    def test_approved_fixture_passes_then_changed_artifact_fails(self):
        # Pure in-memory synthetic approval; never writes stakeholder values.
        docs = self._documents()
        worksheet = docs["requirements.json"]
        worksheet["status"] = "approved"
        worksheet["reference_concept"]["status"] = "approved"
        values = {
            "number": 1,
            "probability": 0.5,
            "string": "fixture-only",
            "vector3": [1, 1, 1],
            "range": [1, 2],
            "list": ["fixture"],
        }
        for record in worksheet["requirements"]:
            record.update(status="approved", value=values[record["value_type"]])
            record["origin"].update(owner="fixture", reference="fixture")
        docs["materials.json"]["status"] = "frozen"
        approval = docs["gate0_approval.json"]
        approval.update(
            status="approved",
            approved_by="fixture",
            approved_utc="2026-09-05T12:00:00Z",
            approval_reference="fixture",
        )
        approval["artifact_sha256"] = {
            name: "a" * 64 for name in approval["artifact_sha256"]
        }
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with patch.object(VALIDATOR, "_artifact_sha256", return_value="a" * 64):
                self.assertEqual(VALIDATOR.validate_approval(True), [])
            with patch.object(VALIDATOR, "_artifact_sha256", return_value="b" * 64):
                with self.assertRaisesRegex(
                    VALIDATOR.ValidationError, "approval stale"
                ):
                    VALIDATOR.validate_approval(True)

    def test_literature_line_mismatch_is_rejected(self):
        docs = self._documents()
        docs["materials.json"]["materials"][0]["candidate_lines"][0][
            "frequency_hz"
        ] += 1000
        with patch.object(VALIDATOR, "load_json", side_effect=self._loader(docs)):
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "literature values mismatch"
            ):
                VALIDATOR.validate_materials()


if __name__ == "__main__":
    unittest.main()
