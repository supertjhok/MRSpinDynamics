"""Create a compact checked-in snapshot from the full reproducible gate report."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform


def main():
    raw = Path(".tmp/nqr_phase2_gate/gate2_report.json")
    report = json.loads(raw.read_text())
    for name, expected in report["source_sha256"].items():
        actual = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit("source changed after gate run: " + name)
    summary = {
        key: report[key]
        for key in (
            "phase",
            "gate2_passed",
            "numerical_domain_passed",
            "limits",
            "supported_domain",
            "analytic_loading",
            "outside_validated_scope",
            "source_sha256",
        )
    }
    summary["software"] = {
        "python": platform.python_version(),
        **{n: version(n) for n in ("numpy", "scipy", "magpylib")},
    }
    summary["full_report_sha256"] = hashlib.sha256(raw.read_bytes()).hexdigest()
    summary["full_report_path"] = str(raw)
    summary["candidates"] = {}
    for name, data in report["candidates"].items():
        rows = data["spatial_loading"]["circuit_cases"]
        ranges = {
            key: [
                min(r["circuit_corner_ranges"][key][0] for r in rows),
                max(r["circuit_corner_ranges"][key][1] for r in rows),
            ]
            for key in rows[0]["circuit_corner_ranges"]
        }
        summary["candidates"][name] = {
            key: data[key]
            for key in ("passed", "checks", "network_refinements", "field_refinements")
        }
        summary["candidates"][name].update(
            circuit_case_count=len(rows),
            circuit_corner_envelope=ranges,
            range_scope="evaluated corners; not certified extrema",
            spatial_refinements=data["spatial_loading"]["refinements"],
        )
    summary["material_benchmark_cases"] = len(report["material_benchmarks"])
    summary["material_benchmark_specifications"] = [
        {k: r[k] for k in ("materials", "translation_z_m", "frequency_hz")}
        for r in report["material_benchmarks"]
    ]
    Path(__file__).with_name("gate2_report.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print("Saved checked-in Gate 2 snapshot; passed:", summary["gate2_passed"])


if __name__ == "__main__":
    main()
