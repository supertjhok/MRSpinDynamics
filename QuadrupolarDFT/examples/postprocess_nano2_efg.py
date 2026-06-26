"""Post-process NaNO2 ABINIT EFG runs into Markdown and CSV records."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
import re
from pathlib import Path

import numpy as np

from quadrupolar_dft import AbinitEFGRecord, nqr_frequencies_hz, parse_abinit_efg


@dataclass(frozen=True)
class IsotopeSpec:
    label: str
    spin: float


@dataclass(frozen=True)
class IsotopeSummary:
    isotope: str
    spin: float
    atoms: tuple[int, ...]
    mean_cq_mhz: float
    mean_abs_cq_mhz: float
    mean_eta: float
    transitions_by_atom_mhz: tuple[tuple[float, ...], ...]


ISOTOPES_BY_TYPAT = {
    1: IsotopeSpec("23Na", 1.5),
    2: IsotopeSpec("14N", 1.0),
    3: IsotopeSpec("17O", 2.5),
}

SUMMARY_FIELDS = [
    "case_id",
    "title",
    "analysis_date",
    "input_file",
    "output_file",
    "abinit_version",
    "isotope",
    "spin",
    "atoms",
    "mean_cq_mhz",
    "mean_abs_cq_mhz",
    "mean_eta",
    "nqr_transitions_mhz",
]

START_RE = re.compile(
    r"^<!-- quadrupolar-dft result:start case_id=(?P<case_id>[^ ]+) -->$",
    re.MULTILINE,
)


def main() -> None:
    args = _parse_args()
    project_root = args.project_root.resolve()
    run_dir = _resolve(project_root, args.run_dir)
    output_path, records = _find_parseable_output(run_dir)
    summaries = _summarize_records(records)

    input_path = (
        _resolve(project_root, args.input) if args.input else _infer_input(run_dir)
    )
    output_md = _resolve(project_root, args.output_md)
    summary_csv = _resolve(project_root, args.summary_csv)
    analysis_date = args.analysis_date or date.today().isoformat()
    abinit_version = _extract_abinit_version(output_path.read_text(encoding="utf-8"))

    section = _format_markdown_section(
        case_id=args.case_id,
        title=args.title,
        analysis_date=analysis_date,
        input_path=input_path,
        output_path=output_path,
        abinit_version=abinit_version,
        project_root=project_root,
        records=records,
        summaries=summaries,
        note=args.note,
    )
    _upsert_markdown_section(output_md, args.case_id, section)
    _upsert_summary_csv(
        summary_csv,
        case_id=args.case_id,
        title=args.title,
        analysis_date=analysis_date,
        input_path=input_path,
        output_path=output_path,
        abinit_version=abinit_version,
        project_root=project_root,
        summaries=summaries,
    )

    print(f"Analyzed {len(records)} EFG records from {_rel(output_path, project_root)}")
    print(f"Updated {_rel(output_md, project_root)}")
    print(f"Updated {_rel(summary_csv, project_root)}")


def _parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze a NaNO2 ABINIT EFG run and update result records."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=script_root,
        help="QuadrupolarDFT project root.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing ABINIT .abo/.stdout files.",
    )
    parser.add_argument("--case-id", required=True, help="Stable result identifier.")
    parser.add_argument("--title", required=True, help="Human-readable result title.")
    parser.add_argument("--input", type=Path, help="ABINIT input file to record.")
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/nano2_efg_results.md"),
        help="Markdown results file to update.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/nano2_efg_summary.csv"),
        help="CSV summary file to update.",
    )
    parser.add_argument(
        "--analysis-date",
        help="ISO analysis date. Defaults to today's date in the runtime environment.",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional note to include in the generated Markdown section.",
    )
    return parser.parse_args()


def _find_parseable_output(run_dir: Path) -> tuple[Path, list[AbinitEFGRecord]]:
    candidates = [
        path
        for path in run_dir.iterdir()
        if path.is_file()
        and "dryrun" not in path.name.lower()
        and (
            ".abo" in path.name
            or path.suffix.lower() in {".out", ".stdout", ".log"}
        )
    ]
    parsed: list[tuple[int, float, Path, list[AbinitEFGRecord]]] = []
    for path in candidates:
        try:
            records = parse_abinit_efg(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        if records:
            parsed.append((len(records), path.stat().st_mtime, path, records))
    if not parsed:
        raise SystemExit(f"No parseable ABINIT EFG records found in {run_dir}")
    _, _, path, records = max(parsed, key=lambda item: (item[0], item[1]))
    return path, records


def _summarize_records(records: list[AbinitEFGRecord]) -> list[IsotopeSummary]:
    summaries = []
    for typat, isotope in ISOTOPES_BY_TYPAT.items():
        group = [record for record in records if record.typat == typat]
        if not group:
            continue
        transitions = tuple(
            tuple(
                float(value) / 1e6
                for value in nqr_frequencies_hz(
                    spin=isotope.spin,
                    cq_hz=record.cq_mhz * 1e6,
                    eta=record.eta,
                )
            )
            for record in group
        )
        summaries.append(
            IsotopeSummary(
                isotope=isotope.label,
                spin=isotope.spin,
                atoms=tuple(record.atom_index for record in group),
                mean_cq_mhz=float(np.mean([record.cq_mhz for record in group])),
                mean_abs_cq_mhz=float(np.mean([abs(record.cq_mhz) for record in group])),
                mean_eta=float(np.mean([record.eta for record in group])),
                transitions_by_atom_mhz=transitions,
            )
        )
    return summaries


def _format_markdown_section(
    *,
    case_id: str,
    title: str,
    analysis_date: str,
    input_path: Path,
    output_path: Path,
    abinit_version: str,
    project_root: Path,
    records: list[AbinitEFGRecord],
    summaries: list[IsotopeSummary],
    note: str,
) -> str:
    lines = [
        f"<!-- quadrupolar-dft result:start case_id={case_id} -->",
        f"## {title}",
        "",
        f"Analysis date: {analysis_date}",
        "",
        f"Case ID: `{case_id}`  ",
        f"Input: `{_rel(input_path, project_root)}`  ",
        f"Output analyzed: `{_rel(output_path, project_root)}`  ",
        f"ABINIT version: {abinit_version}",
        "",
    ]
    if note:
        lines.extend([note, ""])

    lines.extend(
        [
            "### ABINIT EFG Summary",
            "",
            "`C_Q` signs are ABINIT signs from the supplied quadrupole moments. "
            "NQR frequencies use absolute energy differences from the quadrupolar "
            "Hamiltonian.",
            "",
            "| Isotope | ABINIT atoms | Mean `C_Q` (MHz) | Mean `|C_Q|` (MHz) | "
            "Mean `eta` | NQR transitions (MHz) |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary.isotope}, I={_format_spin(summary.spin)} | "
            f"{', '.join(str(atom) for atom in summary.atoms)} | "
            f"{summary.mean_cq_mhz:.6f} | "
            f"{summary.mean_abs_cq_mhz:.6f} | "
            f"{summary.mean_eta:.6f} | "
            f"{_format_transition_groups(summary.transitions_by_atom_mhz)} |"
        )

    n14 = next((summary for summary in summaries if summary.isotope == "14N"), None)
    if n14 is not None:
        lines.extend(["", *_format_n14_comparison(n14)])

    lines.extend(
        [
            "",
            "### Atom-Level EFG Results",
            "",
            "| Atom | typat | Isotope | `C_Q` (MHz) | `eta` | NQR transitions (MHz) |",
            "|---:|---:|---|---:|---:|---|",
        ]
    )
    for record in sorted(records, key=lambda item: item.atom_index):
        isotope = ISOTOPES_BY_TYPAT[record.typat]
        transitions = nqr_frequencies_hz(
            spin=isotope.spin,
            cq_hz=record.cq_mhz * 1e6,
            eta=record.eta,
        )
        lines.append(
            f"| {record.atom_index} | {record.typat} | "
            f"{isotope.label}, I={_format_spin(isotope.spin)} | "
            f"{record.cq_mhz:.6f} | {record.eta:.6f} | "
            f"{_format_values(transitions / 1e6)} |"
        )

    lines.extend(
        [
            f"<!-- quadrupolar-dft result:end case_id={case_id} -->",
            "",
        ]
    )
    return "\n".join(lines)


def _format_n14_comparison(summary: IsotopeSummary) -> list[str]:
    lit_fq_mhz = 4.1
    lit_eta = 0.38
    lit_abs_cq_mhz = 4.0 * lit_fq_mhz / 3.0
    lit_transitions = nqr_frequencies_hz(
        spin=1.0,
        cq_hz=lit_abs_cq_mhz * 1e6,
        eta=lit_eta,
    ) / 1e6
    calc_fq_mhz = 3.0 * summary.mean_abs_cq_mhz / 4.0
    calc_transitions = np.mean(np.asarray(summary.transitions_by_atom_mhz), axis=0)
    return [
        "### 14N Literature Comparison",
        "",
        "For room-temperature NaNO2, the literature values used here are "
        "`f_Q = 4.1 MHz` and `eta = 0.38`; with `f_Q = 3 |C_Q| / 4`, "
        "this corresponds to `|C_Q| = 5.466667 MHz`.",
        "",
        "| Quantity | Literature | This run | Difference |",
        "|---|---:|---:|---:|",
        _comparison_row("f_Q (MHz)", lit_fq_mhz, calc_fq_mhz),
        _comparison_row("|C_Q| (MHz)", lit_abs_cq_mhz, summary.mean_abs_cq_mhz),
        _comparison_row("eta", lit_eta, summary.mean_eta),
        _comparison_row(
            "Low 14N transition (MHz)",
            lit_transitions[0],
            calc_transitions[0],
        ),
        _comparison_row(
            "Middle 14N transition (MHz)",
            lit_transitions[1],
            calc_transitions[1],
        ),
        _comparison_row(
            "High 14N transition (MHz)",
            lit_transitions[2],
            calc_transitions[2],
        ),
    ]


def _comparison_row(label: str, reference: float, value: float) -> str:
    percent = 100.0 * (value - reference) / reference
    return f"| `{label}` | {reference:.6f} | {value:.6f} | {percent:+.2f}% |"


def _upsert_markdown_section(path: Path, case_id: str, section: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        text = path.read_text(encoding="utf-8").rstrip() + "\n\n"
    else:
        text = "# NaNO2 ABINIT EFG Results\n\n"

    match = next(
        (
            candidate
            for candidate in START_RE.finditer(text)
            if candidate.group("case_id") == case_id
        ),
        None,
    )
    if match is None:
        path.write_text(text + section, encoding="utf-8")
        return

    end_marker = f"<!-- quadrupolar-dft result:end case_id={case_id} -->"
    end_index = text.find(end_marker, match.end())
    if end_index == -1:
        raise SystemExit(f"Found start marker for {case_id} without an end marker.")
    end_index += len(end_marker)
    updated = text[: match.start()] + section.rstrip() + text[end_index:]
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def _upsert_summary_csv(
    path: Path,
    *,
    case_id: str,
    title: str,
    analysis_date: str,
    input_path: Path,
    output_path: Path,
    abinit_version: str,
    project_root: Path,
    summaries: list[IsotopeSummary],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_existing_summary(path)
    rows = [
        row
        for row in rows
        if not (
            row["case_id"] == case_id
            and row["isotope"] in {summary.isotope for summary in summaries}
        )
    ]
    for summary in summaries:
        rows.append(
            {
                "case_id": case_id,
                "title": title,
                "analysis_date": analysis_date,
                "input_file": _rel(input_path, project_root),
                "output_file": _rel(output_path, project_root),
                "abinit_version": abinit_version,
                "isotope": summary.isotope,
                "spin": f"{summary.spin:g}",
                "atoms": " ".join(str(atom) for atom in summary.atoms),
                "mean_cq_mhz": f"{summary.mean_cq_mhz:.6f}",
                "mean_abs_cq_mhz": f"{summary.mean_abs_cq_mhz:.6f}",
                "mean_eta": f"{summary.mean_eta:.6f}",
                "nqr_transitions_mhz": _format_transition_groups(
                    summary.transitions_by_atom_mhz
                ),
            }
        )
    rows.sort(key=lambda row: (row["case_id"], row["isotope"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            return []
        if reader.fieldnames == SUMMARY_FIELDS:
            return rows
        return [_migrate_legacy_row(row) for row in rows]


def _migrate_legacy_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "case_id": "nano2_starter",
        "title": "NaNO2 ABINIT EFG Starter Run",
        "analysis_date": "",
        "input_file": "examples/abinit/nano2_efg.abi",
        "output_file": "runs/nano2_efg/nano2_efg.abo",
        "abinit_version": "9.10.4",
        "isotope": row.get("isotope", ""),
        "spin": row.get("spin", ""),
        "atoms": row.get("atoms", ""),
        "mean_cq_mhz": row.get("mean_cq_mhz", ""),
        "mean_abs_cq_mhz": row.get("mean_abs_cq_mhz", ""),
        "mean_eta": row.get("mean_eta", ""),
        "nqr_transitions_mhz": row.get("nqr_transitions_mhz", ""),
    }


def _infer_input(run_dir: Path) -> Path:
    abi_files = sorted(run_dir.glob("*.abi"))
    if not abi_files:
        raise SystemExit(f"No .abi input found in {run_dir}; pass --input.")
    return abi_files[0]


def _extract_abinit_version(text: str) -> str:
    match = re.search(r"\.Version\s+(?P<version>\S+)\s+of ABINIT", text)
    if match:
        return match.group("version")
    match = re.search(r"ABINIT\s+(?P<version>\d+(?:\.\d+)+)", text)
    if match:
        return match.group("version")
    return "unknown"


def _resolve(project_root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _rel(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _format_spin(spin: float) -> str:
    if spin == 1.5:
        return "3/2"
    if spin == 2.5:
        return "5/2"
    return f"{spin:g}"


def _format_values(values: np.ndarray | tuple[float, ...]) -> str:
    return ", ".join(f"{float(value):.6f}" for value in values)


def _format_transition_groups(groups: tuple[tuple[float, ...], ...]) -> str:
    return "; ".join(_format_values(group) for group in groups)


if __name__ == "__main__":
    main()
