"""Validate the evidence registry and generate Markdown/LaTeX summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "validation" / "evidence.json"
MARKDOWN_OUTPUT = ROOT / "docs" / "validation_matrix.md"
LATEX_OUTPUT = ROOT / "docs" / "generated" / "validation_summary.tex"

REQUIRED_FIELDS = {
    "id",
    "component",
    "claim",
    "evidence_types",
    "basis",
    "parameter_range",
    "metric",
    "tolerance",
    "status",
    "reproducers",
    "references",
    "limitations",
}
VALID_STATUSES = {"validated", "partially_validated", "regression_only"}


def load_registry(path: Path = REGISTRY) -> dict[str, Any]:
    """Load and validate the JSON validation-evidence registry."""

    data = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_registry(data, root=ROOT)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"invalid validation evidence registry:\n{joined}")
    return data


def validate_registry(data: dict[str, Any], *, root: Path) -> list[str]:
    """Return human-readable schema and reference errors."""

    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    levels = data.get("evidence_levels")
    if not isinstance(levels, dict) or not levels:
        errors.append("evidence_levels must be a non-empty object")
        levels = {}
    records = data.get("records")
    if not isinstance(records, list) or not records:
        errors.append("records must be a non-empty array")
        return errors

    seen: set[str] = set()
    for index, record in enumerate(records):
        label = record.get("id", f"record[{index}]") if isinstance(record, dict) else f"record[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{label}: must be an object")
            continue
        missing = sorted(REQUIRED_FIELDS - record.keys())
        if missing:
            errors.append(f"{label}: missing fields {', '.join(missing)}")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{label}: id must be a non-empty string")
        elif record_id in seen:
            errors.append(f"{label}: duplicate id")
        else:
            seen.add(record_id)

        for field in ("component", "claim", "basis", "parameter_range", "metric", "tolerance"):
            if not isinstance(record.get(field), str) or not record.get(field, "").strip():
                errors.append(f"{label}: {field} must be a non-empty string")
        evidence_types = record.get("evidence_types")
        if not isinstance(evidence_types, list) or not evidence_types:
            errors.append(f"{label}: evidence_types must be a non-empty array")
        else:
            unknown = sorted(set(evidence_types) - set(levels))
            if unknown:
                errors.append(f"{label}: unknown evidence levels {', '.join(unknown)}")
        if record.get("status") not in VALID_STATUSES:
            errors.append(f"{label}: status must be one of {sorted(VALID_STATUSES)}")

        for field in ("reproducers", "references", "limitations"):
            values = record.get(field)
            if not isinstance(values, list) or not values or not all(
                isinstance(value, str) and value.strip() for value in values
            ):
                errors.append(f"{label}: {field} must be a non-empty string array")

        for reproducer in record.get("reproducers", []):
            if not isinstance(reproducer, str):
                continue
            path_text, _, symbol = reproducer.partition("::")
            target = root / path_text
            if not target.is_file():
                errors.append(f"{label}: reproducer does not exist: {path_text}")
            elif symbol and symbol not in target.read_text(encoding="utf-8", errors="replace"):
                errors.append(f"{label}: symbol {symbol!r} not found in {path_text}")
    return errors


def _md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _md_link(reproducer: str) -> str:
    path_text, _, symbol = reproducer.partition("::")
    label = f"{path_text}::{symbol}" if symbol else path_text
    anchor = ""
    if symbol:
        lines = (ROOT / path_text).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        line_number = next(
            index for index, line in enumerate(lines, start=1) if symbol in line
        )
        anchor = f"#L{line_number}"
    return f"[`{label}`](../{path_text}{anchor})"


def render_markdown(data: dict[str, Any]) -> str:
    """Render the detailed human-readable evidence matrix."""

    records = data["records"]
    status_counts = Counter(record["status"] for record in records)
    lines = [
        "# Validation Evidence Matrix",
        "",
        "<!-- Generated by docs/generate_validation_matrix.py; do not edit by hand. -->",
        "",
        "This matrix distinguishes independent physical validation from regression",
        "coverage. The authoritative records are in `validation/evidence.json`.",
        "A passing test protects behavior; only the evidence levels below describe",
        "what supports the underlying scientific or numerical claim.",
        "",
        "## Evidence Levels",
        "",
        "| Level | Meaning |",
        "| --- | --- |",
    ]
    lines.extend(
        f"| **{code}** | {_md_cell(description)} |"
        for code, description in data["evidence_levels"].items()
    )
    lines.extend(
        [
            "",
            "## Coverage Summary",
            "",
            f"- {len(records)} capability-level claims",
            f"- {status_counts['validated']} validated",
            f"- {status_counts['partially_validated']} partially validated",
            f"- {status_counts['regression_only']} regression-only",
            "",
            "| Component | Claim | Evidence | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for record in records:
        evidence = ", ".join(f"**{code}**" for code in record["evidence_types"])
        status = record["status"].replace("_", " ")
        lines.append(
            f"| {_md_cell(record['component'])} | {_md_cell(record['claim'])} | "
            f"{evidence} | {status} |"
        )

    lines.extend(["", "## Detailed Evidence", ""])
    for record in records:
        evidence = ", ".join(record["evidence_types"])
        lines.extend(
            [
                f"### {record['component']}",
                "",
                f"- **Claim:** {record['claim']}",
                f"- **Evidence:** {evidence} ({record['status'].replace('_', ' ')})",
                f"- **Basis:** {record['basis']}",
                f"- **Tested range:** {record['parameter_range']}",
                f"- **Metric:** {record['metric']}",
                f"- **Tolerance:** {record['tolerance']}",
                f"- **References:** {'; '.join(record['references'])}",
                "- **Reproduce:** " + "; ".join(_md_link(item) for item in record["reproducers"]),
                "- **Limitations:** " + " ".join(record["limitations"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_latex(data: dict[str, Any]) -> str:
    """Render the compact matrix included by the LaTeX user manual."""

    records = data["records"]
    lines = [
        "% Generated by docs/generate_validation_matrix.py; do not edit by hand.",
        r"\section{Capability Evidence Matrix}",
        "",
        (
            "The table below is generated from "
            r"\code{validation/evidence.json}. Evidence letters are defined in the "
            "preceding section; multiple letters indicate complementary checks. "
            "The detailed tested ranges, tolerances, reproducers, references, and "
            r"limitations are published in \code{docs/validation\_matrix.md}."
        ),
        "",
        r"\begin{longtable}{p{0.25\linewidth}p{0.12\linewidth}p{0.16\linewidth}p{0.37\linewidth}}",
        r"\toprule",
        r"Capability & Evidence & Status & Recorded claim \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Capability & Evidence & Status & Recorded claim \\",
        r"\midrule",
        r"\endhead",
    ]
    for record in records:
        evidence = ", ".join(record["evidence_types"])
        status = {
            "validated": "validated",
            "partially_validated": "partial",
            "regression_only": "regression only",
        }[record["status"]]
        lines.append(
            f"{_tex(record['component'])} & {_tex(evidence)} & {_tex(status)} & "
            f"{_tex(record['claim'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    return "\n".join(lines)


def _check_output(path: Path, expected: str) -> str | None:
    if not path.is_file():
        return f"missing generated file: {path.relative_to(ROOT)}"
    if path.read_text(encoding="utf-8") != expected:
        return f"stale generated file: {path.relative_to(ROOT)}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and fail if generated outputs are missing or stale.",
    )
    args = parser.parse_args(argv)
    try:
        data = load_registry()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    markdown = render_markdown(data)
    latex = render_latex(data)
    if args.check:
        errors = [
            error
            for error in (
                _check_output(MARKDOWN_OUTPUT, markdown),
                _check_output(LATEX_OUTPUT, latex),
            )
            if error is not None
        ]
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print(f"validated {len(data['records'])} evidence records and generated outputs")
        return 0

    MARKDOWN_OUTPUT.write_text(markdown, encoding="utf-8")
    LATEX_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    LATEX_OUTPUT.write_text(latex, encoding="utf-8")
    print(f"wrote {MARKDOWN_OUTPUT.relative_to(ROOT)}")
    print(f"wrote {LATEX_OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
