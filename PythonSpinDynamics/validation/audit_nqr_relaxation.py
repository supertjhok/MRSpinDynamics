"""Audit normalized NQRDatabase relaxation records without modifying them."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = REPO_ROOT / "NQRDatabase" / "data" / "normalized"
DEFAULT_OUTPUT = Path(__file__).with_name("nqr_relaxation_audit.json")
OBSERVABLES = ("t1_s", "t2_s", "t2_star_s")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def audit_relaxation_records(database: Path = DEFAULT_DATABASE) -> dict[str, Any]:
    """Return coverage and conservative plausibility flags for relaxation data."""

    lines = _read_jsonl(database / "lines.jsonl")
    sites = {row["id"]: row for row in _read_jsonl(database / "sites.jsonl")}
    samples = {row["id"]: row for row in _read_jsonl(database / "samples.jsonl")}
    compounds = {
        row["id"]: row for row in _read_jsonl(database / "compounds.jsonl")
    }
    records = [
        row
        for row in lines
        if any(row.get(observable) is not None for observable in OBSERVABLES)
    ]

    isotope_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    uncertainty_records = 0
    explicit_temperature = 0
    nonpositive: list[dict[str, Any]] = []
    t2_greater_than_t1: list[str] = []
    comparable: defaultdict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)

    for line in records:
        site = sites[line["site_id"]]
        sample = samples[site["sample_id"]]
        compound = compounds[sample["compound_id"]]
        isotope_counts[site.get("isotope") or "unassigned"] += 1
        source_counts[line["source_id"]] += 1
        if line.get("temperature_k") is not None:
            explicit_temperature += 1
        originals = " ".join(
            str(line.get(name) or "")
            for name in ("t1_original", "t2_original", "t2_star_original")
        )
        if re.search(r"(?:±|\+/-)", originals):
            uncertainty_records += 1
        for observable in OBSERVABLES:
            value = line.get(observable)
            if value is not None and value <= 0.0:
                nonpositive.append(
                    {"line_id": line["id"], "observable": observable, "value": value}
                )
        if (
            line.get("t1_s") is not None
            and line.get("t2_s") is not None
            and line["t2_s"] > line["t1_s"]
        ):
            t2_greater_than_t1.append(line["id"])
        comparable[(compound["canonical_name"], float(line["frequency_khz"]))].append(line)

    cross_source_scale_conflicts: list[dict[str, Any]] = []
    for (compound, frequency_khz), group in comparable.items():
        sources = {row["source_id"] for row in group}
        if len(sources) < 2:
            continue
        for observable in OBSERVABLES:
            values = [float(row[observable]) for row in group if row.get(observable) is not None]
            if len(values) < 2 or min(values) <= 0.0:
                continue
            ratio = max(values) / min(values)
            if ratio >= 100.0:
                cross_source_scale_conflicts.append(
                    {
                        "compound": compound,
                        "frequency_khz": frequency_khz,
                        "observable": observable,
                        "max_to_min_ratio": round(ratio, 6),
                        "records": [
                            {
                                "line_id": row["id"],
                                "source_id": row["source_id"],
                                "normalized_seconds": row.get(observable),
                                "original": row.get(observable.replace("_s", "_original")),
                            }
                            for row in group
                            if row.get(observable) is not None
                        ],
                    }
                )

    ranges = {}
    for observable in OBSERVABLES:
        values = [float(row[observable]) for row in records if row.get(observable) is not None]
        ranges[observable] = {
            "count": len(values),
            "minimum_seconds": min(values),
            "maximum_seconds": max(values),
        }

    return {
        "schema_version": 1,
        "database_path": "NQRDatabase/data/normalized",
        "record_definition": "A line with at least one normalized T1, T2, or T2* value.",
        "coverage": {
            "relaxation_records": len(records),
            "explicit_temperature_records": explicit_temperature,
            "records_with_reported_uncertainty": uncertainty_records,
            "by_isotope": dict(sorted(isotope_counts.items())),
            "by_source": dict(sorted(source_counts.items())),
            "observable_ranges": ranges,
        },
        "plausibility": {
            "nonpositive_values": nonpositive,
            "t2_greater_than_t1_line_ids": sorted(t2_greater_than_t1),
            "cross_source_scale_conflicts": cross_source_scale_conflicts,
        },
        "interpretation": {
            "microscopic_fit_eligible": 0,
            "reason": (
                "No record set combines a relaxation observable, explicit temperature "
                "series, uncertainty, pulse-sequence definition, and sufficient model inputs."
            ),
            "sodium_nitrite": (
                "The CWRU HTML and NRL summary duplicate the same three lines, but the "
                "HTML-normalized T1/T2 values are exactly 1000 times the NRL values. "
                "Treat the HTML entries as unresolved unit conflicts, not independent data."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(audit_relaxation_records(args.database), indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"stale relaxation audit: run {Path(__file__).name}")
        return 0
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
