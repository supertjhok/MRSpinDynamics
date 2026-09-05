"""Validate Phase 0 study inputs (install the project's dev extra)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PHASE0 = Path(__file__).resolve().parent
REPOSITORY = PHASE0.parents[3]
EVIDENCE_IDS = {
    "predicted",
    "literature",
    "measured_calibration",
    "fitted",
    "held_out_validation",
}

REQUIREMENT_CONTRACT = {
    "req.parcel.max_outer_dimensions": ["vector3", "m", True],
    "req.operation.mode": ["string", None, True],
    "req.operation.max_scan_time": ["number", "s", False],
    "req.performance.minimum_probability_detection": ["probability", "1", False],
    "req.performance.maximum_probability_false_alarm": ["probability", "1", False],
    "req.performance.confidence_level": ["probability", "1", True],
    "req.material.minimum_target_amount": ["number", "kg", True],
    "req.environment.temperature_range": ["range", "K", True],
    "req.hardware.max_peak_rf_power": ["number", "W", False],
    "req.hardware.max_average_rf_power": ["number", "W", False],
    "req.hardware.max_coil_current": ["number", "A", True],
    "req.hardware.max_terminal_voltage": ["number", "V", True],
    "req.receiver.max_recovery_time": ["number", "s", True],
    "req.thermal.max_component_temperature": ["number", "K", True],
    "req.thermal.max_parcel_temperature_rise": ["number", "K", True],
    "req.rf.applicable_safety_and_emc_standards": ["list", None, True],
    "req.validation.approved_material_and_facility_protocol": ["string", None, True],
    "req.population.target_scope": ["string", None, True],
    "req.population.benign_scope": ["string", None, True],
    "req.validation.test_set_policy": ["string", None, True],
    "req.parcel.clearance": ["number", "m", True],
    "req.objective.design": ["string", None, True],
    "req.hardware.rf_power_policy": ["string", None, True],
    "req.protocol.pulse_families": ["list", None, True],
    "req.protocol.prepolarization": ["string", None, True],
}


class ValidationError(RuntimeError):
    """A Phase 0 artifact is internally inconsistent."""


def load_json(name: str) -> dict[str, Any]:
    path = PHASE0 / name
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must contain a JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def unique_ids(records: list[dict[str, Any]], label: str) -> None:
    identifiers = [record.get("id") for record in records]
    require(None not in identifiers, f"{label} contains a record without an id")
    require(
        len(identifiers) == len(set(identifiers)),
        f"{label} contains duplicate ids",
    )


def validate_schemas() -> None:
    from jsonschema import Draft202012Validator, FormatChecker

    for name in (
        "requirements.schema.json",
        "materials.schema.json",
        "result_record.schema.json",
    ):
        schema = load_json(name)
        require(
            schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
            f"{name} must declare JSON Schema draft 2020-12",
        )
        require(schema.get("type") == "object", f"{name} root must be an object")
        Draft202012Validator.check_schema(schema)
        document = name.replace(".schema.json", ".json")
        if name == "result_record.schema.json":
            document = "result_record.example.json"
        errors = list(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(
                load_json(document)
            )
        )
        require(
            not errors,
            f"{document}: "
            + "; ".join(f"{error.json_path}: {error.message}" for error in errors),
        )


def validate_evidence() -> None:
    data = load_json("evidence_classes.json")
    records = data.get("classes")
    require(isinstance(records, list), "evidence classes must be a list")
    unique_ids(records, "evidence classes")
    actual = {record["id"] for record in records}
    require(actual == EVIDENCE_IDS, f"unexpected evidence classes: {actual}")
    for record in records:
        require(record.get("definition"), f"{record['id']} needs a definition")
        require(record.get("allowed_roles"), f"{record['id']} needs allowed roles")
        require(record.get("restrictions"), f"{record['id']} needs restrictions")


def _valid_value(requirement: dict[str, Any]) -> bool:
    value = requirement["value"]
    value_type = requirement["value_type"]
    if value is None:
        return False
    if value_type in {"number", "probability"}:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isfinite(float(value)):
            return False
        return (
            0.0 < float(value) < 1.0
            if value_type == "probability"
            else float(value) > 0.0
        )
    if value_type == "string":
        return isinstance(value, str) and bool(value.strip())
    if value_type == "vector3":
        return (
            isinstance(value, list)
            and len(value) == 3
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                and float(item) > 0
                for item in value
            )
        )
    if value_type == "range":
        return (
            isinstance(value, list)
            and len(value) == 2
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in value
            )
            and 0.0 < float(value[0]) < float(value[1])
        )
    if value_type == "list":
        return isinstance(value, list) and len(value) > 0
    return False


def validate_requirements(require_ready: bool) -> list[str]:
    data = load_json("requirements.json")
    require(data.get("schema_version") == "0.1.0", "requirements schema mismatch")
    require(data.get("study_id") == "nqr-mail-screening", "wrong study id")
    records = data.get("requirements")
    require(isinstance(records, list) and records, "requirements must be non-empty")
    unique_ids(records, "requirements")
    by_id = {record["id"]: record for record in records}
    require(
        REQUIREMENT_CONTRACT.keys() <= by_id.keys(), "missing mandatory requirements"
    )
    for identifier, (kind, units, blocking) in REQUIREMENT_CONTRACT.items():
        record = by_id[identifier]
        require(
            record.get("blocking_gate_0") is blocking,
            f"{identifier}: blocking flag mismatch",
        )
        require(
            record.get("value_type") == kind and record.get("units") == units,
            f"{identifier}: requirement type/units mismatch",
        )
    unresolved: list[str] = []
    for record in records:
        identifier = record["id"]
        require(identifier.startswith("req."), f"invalid requirement id {identifier}")
        status = record.get("status")
        require(
            status in {"unresolved", "provisional", "approved"},
            f"invalid status for {identifier}",
        )
        origin = record.get("origin")
        require(isinstance(origin, dict), f"{identifier} needs an origin object")
        evidence = origin.get("evidence_class")
        require(
            evidence is None or evidence in EVIDENCE_IDS,
            f"{identifier} has invalid evidence class",
        )
        has_value = _valid_value(record)
        if status == "unresolved":
            require(
                record["value"] is None, f"{identifier}: unresolved value must be null"
            )
        else:
            require(has_value, f"{identifier}: invalid requirement value")
        if status == "approved":
            require(has_value, f"approved requirement {identifier} has no valid value")
            require(
                origin.get("owner"), f"approved requirement {identifier} needs an owner"
            )
            require(
                origin.get("reference"),
                f"approved requirement {identifier} needs a reference",
            )
        if record.get("blocking_gate_0") and (status != "approved" or not has_value):
            unresolved.append(identifier)
    if require_ready and unresolved:
        joined = "\n  - ".join(unresolved)
        raise ValidationError(
            f"Gate 0 is not ready; unresolved requirements:\n  - {joined}"
        )
    return unresolved


def validate_approval(require_ready: bool = False) -> list[str]:
    """Bind stakeholder sign-off to artifact bytes; never infer approval."""
    data = load_json("gate0_approval.json")
    require(data.get("schema_version") == "0.1.0", "approval schema mismatch")
    require(data.get("status") in {"pending", "approved"}, "invalid approval status")
    expected = {
        "requirements.json",
        "materials.json",
        "evidence_classes.json",
        "test_set_policy.md",
        "requirements.schema.json",
        "materials.schema.json",
        "result_record.schema.json",
        "literature_supplement.json",
        "user_requirements.md",
        "envelope_scope.md",
    }
    require(
        set(data.get("artifact_sha256", {})) == expected,
        "approval artifact set mismatch",
    )
    pending = []
    if data["status"] != "approved":
        pending.append("stakeholder approval of requirements and material/test set")
    else:
        for field in ("approved_by", "approved_utc", "approval_reference"):
            require(
                isinstance(data.get(field), str) and data[field].strip(),
                f"approval needs {field}",
            )
        from datetime import datetime

        stamp = datetime.fromisoformat(data["approved_utc"].replace("Z", "+00:00"))
        require(stamp.utcoffset() is not None, "approval timestamp needs timezone")
        for name, digest in data["artifact_sha256"].items():
            require(digest == _sha256(PHASE0 / name), f"approval stale for {name}")
        worksheet = load_json("requirements.json")
        require(worksheet["status"] == "approved", "worksheet is not approved")
        require(
            worksheet["reference_concept"]["status"] == "approved",
            "reference concept is not approved",
        )
        require(
            load_json("materials.json")["status"] == "frozen",
            "material set is not frozen",
        )
        validate_requirements(require_ready=True)
    if require_ready and pending:
        raise ValidationError("Gate 0 is not ready; " + "; ".join(pending))
    return pending


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identifier = record.get("id")
            if not identifier:
                raise ValidationError(f"{path}:{line_number} has no id")
            records[str(identifier)] = record
    return records


def _same_optional(actual: Any, expected: Any, scale: float = 1.0) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    return math.isclose(
        float(actual) * scale, float(expected), rel_tol=1e-12, abs_tol=1e-9
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_materials() -> None:
    data = load_json("materials.json")
    require(data.get("schema_version") == "0.1.0", "materials schema mismatch")
    require(data.get("study_id") == "nqr-mail-screening", "wrong materials study id")
    records = data.get("materials")
    require(isinstance(records, list) and records, "materials must be non-empty")
    unique_ids(records, "materials")

    roles = [record.get("role") for record in records]
    require(
        roles.count("target") >= 2, "development library needs at least two targets"
    )
    require(
        roles.count("benign_confuser") >= 2,
        "development library needs at least two benign confusers",
    )
    require(roles.count("null") >= 1, "development library needs at least one null")

    source = data["source_database"]
    database = REPOSITORY / source["path"]
    require(database.is_file(), f"NQR database not found: {database}")
    require(
        _sha256(database) == source["sha256"],
        "NQR database SHA-256 does not match materials.json",
    )

    normalized = REPOSITORY / "NQRDatabase" / "data" / "normalized"
    compounds = _load_jsonl(normalized / "compounds.jsonl")
    samples = _load_jsonl(normalized / "samples.jsonl")
    sites = _load_jsonl(normalized / "sites.jsonl")
    lines = _load_jsonl(normalized / "lines.jsonl")

    for material in records:
        identifier = material["id"]
        role = material["role"]
        line_records = material["candidate_lines"]
        if role == "null":
            require(
                material["isotope"] is None, f"{identifier}: null isotope must be null"
            )
            require(
                material["database_compound_id"] is None,
                f"{identifier}: null database compound must be null",
            )
            require(not line_records, f"{identifier}: null class must not claim lines")
            continue

        require(material["isotope"] == "14N", f"{identifier}: expected isotope 14N")
        compound_id = material["database_compound_id"]
        if material.get("source_reference"):
            supplement = load_json("literature_supplement.json")
            sources = {item["id"]: item for item in supplement["sources"]}
            require(
                material["source_reference"] in sources, f"{identifier}: unknown source"
            )
            require(
                material["physical_test_status"] == "simulation_only",
                f"{identifier}: literature targets remain simulation-only",
            )
            source_lines = {item["line_id"]: item for item in supplement["materials"]}
            for candidate in line_records:
                source_line = source_lines.get(candidate["line_id"])
                require(
                    source_line is not None, f"{identifier}: missing literature line"
                )
                require(
                    source_line["source_id"] == material["source_reference"],
                    f"{identifier}: literature source mismatch",
                )
                require(
                    candidate == source_line["values"],
                    f"{identifier}: literature values mismatch",
                )
            continue
        require(
            compound_id in compounds, f"{identifier}: unknown compound {compound_id}"
        )
        require(line_records, f"{identifier}: expected at least one candidate line")
        if role == "target":
            require(
                material["physical_test_status"] == "simulation_only",
                f"{identifier}: Phase 0 targets must remain simulation-only",
            )

        for candidate in line_records:
            line_id = candidate["line_id"]
            require(line_id in lines, f"{identifier}: unknown line {line_id}")
            source_line = lines[line_id]
            site_id = candidate["site_id"]
            require(source_line["site_id"] == site_id, f"{line_id}: site mismatch")
            require(site_id in sites, f"{line_id}: missing source site")
            source_site = sites[site_id]
            require(source_site["isotope"] == "14N", f"{line_id}: source is not 14N")
            sample_id = source_site["sample_id"]
            require(sample_id in samples, f"{line_id}: missing source sample")
            require(
                samples[sample_id]["compound_id"] == compound_id,
                f"{line_id}: compound mismatch",
            )
            comparisons = (
                ("frequency_khz", "frequency_hz", 1000.0),
                ("fwhm_khz", "fwhm_hz", 1000.0),
                ("t1_s", "t1_s", 1.0),
                ("t2_s", "t2_s", 1.0),
                ("t2_star_s", "t2_star_s", 1.0),
                ("dnu_dt_khz_per_c", "dnu_dt_hz_per_k", 1000.0),
                ("temperature_k", "reference_temperature_k", 1.0),
            )
            for source_key, candidate_key, scale in comparisons:
                require(
                    _same_optional(
                        source_line.get(source_key),
                        candidate.get(candidate_key),
                        scale,
                    ),
                    f"{line_id}: {candidate_key} does not match normalized source",
                )
            require(
                candidate["evidence_class"] == "literature",
                f"{line_id}: database line must be literature evidence",
            )


def validate_result_example() -> None:
    data = load_json("result_record.example.json")
    require(data.get("schema_version") == "0.1.0", "result example schema mismatch")
    require(data.get("study_id") == "nqr-mail-screening", "wrong result study id")
    require(data.get("evidence_class") in EVIDENCE_IDS, "invalid result evidence")
    require(
        data.get("decision", {}).get("status") == "not_evaluated",
        "Phase 0 result example must not claim a decision",
    )
    require(
        data.get("acquisition", {}).get("physical_time_s") is None,
        "Phase 0 result example must not claim scan time",
    )
    require(
        data.get("constraints") == [],
        "Phase 0 result example must not claim constraint checks",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Fail unless all Gate-0-blocking requirements are approved.",
    )
    args = parser.parse_args(argv)
    try:
        validate_schemas()
        validate_evidence()
        unresolved = validate_requirements(False)
        validate_materials()
        validate_result_example()
        pending = validate_approval(False)
        if args.require_ready and (unresolved or pending):
            raise ValidationError(
                "Gate 0 is not ready:\n  - " + "\n  - ".join(unresolved + pending)
            )
    except (
        ImportError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ValidationError,
    ) as exc:
        parser.exit(1, f"Phase 0 validation failed: {exc}\n")

    print("Phase 0 artifacts are structurally valid.")
    if unresolved or pending:
        print(
            f"Gate 0 remains open: {len(unresolved)} blocking requirements unresolved."
        )
        for item in pending:
            print(f"Pending: {item}")
    else:
        print("Gate 0 passed: requirements and material/test set approved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
