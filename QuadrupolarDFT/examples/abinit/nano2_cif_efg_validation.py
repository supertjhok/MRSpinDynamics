"""Prepare and collect NaNO2 CIF EFG validation runs.

This helper stages static ABINIT EFG calculations for the best-ranked bundled
NaNO2 CIFs, then collects completed outputs into a simulation-vs-reference
report.  The companion shell script ``nano2_cif_efg_validation.sh`` performs the
actual ABINIT launches through the MPI-aware ``run_static_efg_wsl.sh`` runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np

from quadrupolar_dft import nqr_frequencies_hz, parse_abinit_efg
from quadrupolar_dft.strain_coupling import (
    _apply_symmetry_operation,
    _cif_lattice,
    _cif_symmetry_operations,
    _float_cif,
    _parse_cif_tokens,
)


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from check_nano2_cifs import DEFAULT_CIF_DIR, assess_cif  # noqa: E402


QUADMOM_BY_Z = {11: 0.104, 7: 0.02044, 8: -0.02558}
PSEUDO_BY_Z = {11: "Na.xml", 7: "N.xml", 8: "O.xml"}
SPECIES_ORDER = (11, 7, 8)
ELEMENT_Z = {"Na": 11, "N": 7, "O": 8}

REFERENCE_ABS_CQ_MHZ = 5.466667
REFERENCE_ETA = 0.38
REFERENCE_SPIN = 1.0


@dataclass(frozen=True)
class ExpandedAtom:
    label: str
    element: str
    z: int
    fractional: np.ndarray
    symmetry_index: int


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    icsd_code: str
    cif_file: str
    status: str
    metadata_score: float
    metadata_verdict: str
    temperature_k: float | None
    space_group: str | None
    mean_cq_mhz: float | None
    mean_abs_cq_mhz: float | None
    mean_eta: float | None
    eta_error: float | None
    cq_percent_error: float | None
    rms_line_error_khz: float | None
    transitions_mhz: tuple[float, ...]
    notes: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Stage ABINIT EFG inputs.")
    prepare.add_argument("--workdir", type=Path, required=True)
    prepare.add_argument("--cif-dir", type=Path, default=DEFAULT_CIF_DIR)
    prepare.add_argument("--target-temperature", type=float, default=295.0)
    prepare.add_argument("--top", type=int, default=5)
    prepare.add_argument("--paraelectric-controls", type=int, default=2)
    prepare.add_argument(
        "--include-partial-occupancy",
        action="store_true",
        help=(
            "Also stage partial-occupancy/disordered CIFs. These often need an "
            "ordered approximant and can trigger PAW sphere-overlap aborts."
        ),
    )
    prepare.add_argument(
        "--include",
        action="append",
        default=[],
        help="Additional ICSD code(s), comma-separated or repeated.",
    )
    prepare.add_argument("--ecut", type=float, default=25.0)
    prepare.add_argument("--pawecutdg", type=float, default=50.0)
    prepare.add_argument("--ngkpt", default="4,4,4")
    prepare.add_argument(
        "--pseudo-dir",
        default="Pseudodojo_paw_pbe_standard",
        help="Subdirectory under ABI_PSPDIR containing Na/N/O PAW XML files.",
    )
    prepare.set_defaults(func=cmd_prepare)

    list_inputs = subparsers.add_parser(
        "list-inputs",
        help="Print staged .abi paths from an existing manifest.",
    )
    list_inputs.add_argument("--workdir", type=Path, required=True)
    list_inputs.set_defaults(func=cmd_list_inputs)

    collect = subparsers.add_parser(
        "collect",
        help="Collect completed ABINIT EFG outputs and rank simulation accuracy.",
    )
    collect.add_argument("--workdir", type=Path, required=True)
    collect.add_argument("--allow-missing", action="store_true")
    collect.add_argument("--reference-cq-mhz", type=float, default=REFERENCE_ABS_CQ_MHZ)
    collect.add_argument("--reference-eta", type=float, default=REFERENCE_ETA)
    collect.add_argument("--csv", type=Path)
    collect.add_argument("--markdown", type=Path)
    collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


def cmd_prepare(args: argparse.Namespace) -> None:
    workdir = args.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    ngkpt = _parse_int_triplet(args.ngkpt)
    assessments = _selected_assessments(
        args.cif_dir,
        target_temperature_k=args.target_temperature,
        top=args.top,
        paraelectric_controls=args.paraelectric_controls,
        include_codes=_parse_include_codes(args.include),
        include_partial_occupancy=args.include_partial_occupancy,
    )
    cases = []
    for assessment in assessments:
        case_id = f"nano2_icsd{assessment.icsd_code}_efg"
        input_path = workdir / f"{case_id}.abi"
        text, atoms = nano2_static_efg_input_from_cif(
            assessment.path,
            ecut=args.ecut,
            pawecutdg=args.pawecutdg,
            ngkpt=ngkpt,
            pseudo_dir=args.pseudo_dir,
            case_id=case_id,
        )
        input_path.write_text(text, encoding="utf-8")
        n_atoms_zero_based = [
            index for index, atom in enumerate(atoms) if atom.z == ELEMENT_Z["N"]
        ]
        cases.append(
            {
                "case_id": case_id,
                "icsd_code": assessment.icsd_code,
                "cif_file": _rel(assessment.path),
                "input_file": _rel(input_path),
                "output_file": _rel(workdir / f"{case_id}.abo"),
                "space_group": assessment.space_group,
                "temperature_k": assessment.temperature_k,
                "metadata_score": assessment.score,
                "metadata_verdict": assessment.verdict,
                "metadata_reasons": list(assessment.reasons),
                "nitrogen_atom_indices_zero_based": n_atoms_zero_based,
                "nitrogen_atom_indices_abinit": [
                    index + 1 for index in n_atoms_zero_based
                ],
                "nitrogen_typat": SPECIES_ORDER.index(ELEMENT_Z["N"]) + 1,
                "natom": len(atoms),
            }
        )

    manifest = {
        "kind": "nano2_cif_efg_validation",
        "target_temperature_k": args.target_temperature,
        "reference": {
            "abs_cq_mhz": REFERENCE_ABS_CQ_MHZ,
            "eta": REFERENCE_ETA,
            "spin": REFERENCE_SPIN,
        },
        "settings": {
            "ecut": args.ecut,
            "pawecutdg": args.pawecutdg,
            "ngkpt": list(ngkpt),
            "pseudo_dir": args.pseudo_dir,
        },
        "cases": cases,
    }
    manifest_path = workdir / "nano2_cif_efg_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Staged {len(cases)} NaNO2 CIF EFG input(s) in {_rel(workdir)}")
    print(f"Manifest: {_rel(manifest_path)}")
    print("Cases:")
    for case in cases:
        print(
            f"  {case['case_id']}: ICSD {case['icsd_code']} "
            f"score={case['metadata_score']:.1f} "
            f"SG={case['space_group']} T={case['temperature_k']}"
        )


def cmd_list_inputs(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.workdir)
    for case in manifest["cases"]:
        print(_resolve(case["input_file"]))


def cmd_collect(args: argparse.Namespace) -> None:
    manifest = _read_manifest(args.workdir)
    reference_transitions = nqr_frequencies_hz(
        spin=REFERENCE_SPIN,
        cq_hz=args.reference_cq_mhz * 1e6,
        eta=args.reference_eta,
    )
    results = [
        _collect_case(
            case,
            reference_cq_mhz=args.reference_cq_mhz,
            reference_eta=args.reference_eta,
            reference_transitions_hz=reference_transitions,
            allow_missing=args.allow_missing,
        )
        for case in manifest["cases"]
    ]
    results = sorted(
        results,
        key=lambda item: (
            item.status != "ok",
            item.rms_line_error_khz if item.rms_line_error_khz is not None else 1e99,
            abs(item.eta_error) if item.eta_error is not None else 1e99,
        ),
    )

    _print_collection(results)
    if args.csv:
        _write_csv(args.csv, results)
        print(f"Wrote CSV report: {_rel(args.csv)}")
    if args.markdown:
        _write_markdown(
            args.markdown,
            results,
            reference_cq_mhz=args.reference_cq_mhz,
            reference_eta=args.reference_eta,
            reference_transitions_hz=reference_transitions,
        )
        print(f"Wrote Markdown report: {_rel(args.markdown)}")


def nano2_static_efg_input_from_cif(
    path: Path,
    *,
    ecut: float,
    pawecutdg: float,
    ngkpt: tuple[int, int, int],
    pseudo_dir: str,
    case_id: str,
) -> tuple[str, list[ExpandedAtom]]:
    text = path.read_text(encoding="utf-8")
    items, loops = _parse_cif_tokens(text)
    atoms = _cif_atom_sites(loops)
    sym_ops = _cif_symmetry_operations(loops) or ("x,y,z",)
    expanded = _expand_atoms(atoms, sym_ops)
    species = [z for z in SPECIES_ORDER if any(atom.z == z for atom in expanded)]
    typat_by_z = {z: index + 1 for index, z in enumerate(species)}
    lattice = _cif_lattice(items)
    space_group = _item(
        items,
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
    )

    lines = [
        f"# Static EFG input generated from {path.name}",
        f"# Case: {case_id}",
        f"# ICSD: {_item(items, '_database_code_ICSD') or 'unknown'}",
        f"# Space group: {space_group or 'unknown'}",
        "# Generated for NaNO2 CIF EFG validation; converge before publication.",
        "",
        "nucefg 2",
        "quadmom " + " ".join(f"{QUADMOM_BY_Z[z]:.8g}" for z in species),
        "",
        "acell 1.0 1.0 1.0 Angstrom",
        "rprim",
    ]
    lines.extend(f"  {row[0]:.12f}  {row[1]:.12f}  {row[2]:.12f}" for row in lattice)
    lines.extend(
        [
            "chkprim 0",
            "nsym 1",
            "",
            f"ntypat {len(species)}",
            "znucl " + " ".join(str(z) for z in species),
            f'pp_dirpath "$ABI_PSPDIR/{pseudo_dir}"',
            'pseudos "' + ", ".join(PSEUDO_BY_Z[z] for z in species) + '"',
            "",
            f"natom {len(expanded)}",
            "typat",
        ]
    )
    typat_values = [typat_by_z[atom.z] for atom in expanded]
    lines.extend(_format_int_rows(typat_values))
    lines.extend(["", "xred"])
    for index, atom in enumerate(expanded):
        frac = atom.fractional
        comment = (
            f"# {index:02d} {atom.label} {atom.element} "
            f"sym={atom.symmetry_index}"
        )
        lines.append(
            f"  {frac[0]:.12f}  {frac[1]:.12f}  {frac[2]:.12f}  {comment}"
        )
    lines.extend(
        [
            "",
            "# Initial numerical settings; converge EFG tensor components.",
            f"ecut {ecut:g}",
            f"pawecutdg {pawecutdg:g}",
            "ngkpt " + " ".join(str(value) for value in ngkpt),
            "nshiftk 1",
            "shiftk 0.0 0.0 0.0",
            "",
            "occopt 1",
            "iscf 17",
            "nstep 80",
            "tolvrs 1.0d-14",
            "diemac 4.0",
            "",
            "prtden 0",
            "prtwf 0",
            "prteig 0",
            "prtvol 2",
            "",
        ]
    )
    return "\n".join(lines), expanded


def _selected_assessments(
    cif_dir: Path,
    *,
    target_temperature_k: float,
    top: int,
    paraelectric_controls: int,
    include_codes: set[str],
    include_partial_occupancy: bool,
) -> list:
    assessments = sorted(
        (
            assess_cif(path, target_temperature_k=target_temperature_k)
            for path in sorted(cif_dir.glob("*.cif"))
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    selected: list = []
    selected.extend(
        item
        for item in assessments
        if _is_ferroelectric(item.space_group)
        and (include_partial_occupancy or _is_full_occupancy(item))
    )
    selected = selected[: max(top, 0)]
    controls = [
        item
        for item in assessments
        if _is_paraelectric(item.space_group)
        and (include_partial_occupancy or _is_full_occupancy(item))
    ][: max(paraelectric_controls, 0)]
    selected.extend(controls)
    for code in include_codes:
        match = next((item for item in assessments if item.icsd_code == code), None)
        if match is None:
            raise SystemExit(f"Requested ICSD code {code} was not found.")
        selected.append(match)

    deduped = []
    seen: set[str] = set()
    for item in selected:
        if item.icsd_code in seen:
            continue
        seen.add(item.icsd_code)
        deduped.append(item)
    return deduped


def _collect_case(
    case: dict,
    *,
    reference_cq_mhz: float,
    reference_eta: float,
    reference_transitions_hz: np.ndarray,
    allow_missing: bool,
) -> CaseResult:
    output_path = _resolve(case["output_file"])
    if not output_path.exists():
        failure_note = _case_failure_note(case)
        if failure_note:
            return _non_ok_case(case, "failed", failure_note)
        if allow_missing:
            return _non_ok_case(
                case,
                "missing",
                f"missing output: {case['output_file']}",
            )
        raise FileNotFoundError(f"Missing output for {case['case_id']}: {output_path}")

    records = parse_abinit_efg(output_path.read_text(encoding="utf-8"))
    nitrogen_typat = int(case["nitrogen_typat"])
    nitrogen_records = [record for record in records if record.typat == nitrogen_typat]
    if not nitrogen_records:
        dry_run_note = _case_dry_run_note(case)
        if dry_run_note and allow_missing:
            return _non_ok_case(case, "missing", dry_run_note)
        failure_note = _case_failure_note(case) or "no nitrogen EFG records parsed"
        if allow_missing:
            return _non_ok_case(case, "failed", failure_note)
        raise ValueError(f"No nitrogen EFG records in {output_path}")

    mean_cq_mhz = float(np.mean([record.cq_mhz for record in nitrogen_records]))
    mean_abs_cq_mhz = float(
        np.mean([abs(record.cq_mhz) for record in nitrogen_records])
    )
    mean_eta = float(np.mean([record.eta for record in nitrogen_records]))
    transitions_by_atom = [
        nqr_frequencies_hz(
            spin=REFERENCE_SPIN,
            cq_hz=record.cq_mhz * 1e6,
            eta=record.eta,
        )
        for record in nitrogen_records
    ]
    transitions_hz = np.mean(np.asarray(transitions_by_atom), axis=0)
    rms_line_error_khz = float(
        np.sqrt(np.mean((transitions_hz - reference_transitions_hz) ** 2)) / 1e3
    )
    return CaseResult(
        case_id=case["case_id"],
        icsd_code=case["icsd_code"],
        cif_file=case["cif_file"],
        status="ok",
        metadata_score=float(case["metadata_score"]),
        metadata_verdict=case["metadata_verdict"],
        temperature_k=case["temperature_k"],
        space_group=case["space_group"],
        mean_cq_mhz=mean_cq_mhz,
        mean_abs_cq_mhz=mean_abs_cq_mhz,
        mean_eta=mean_eta,
        eta_error=mean_eta - reference_eta,
        cq_percent_error=100.0 * (mean_abs_cq_mhz - reference_cq_mhz)
        / reference_cq_mhz,
        rms_line_error_khz=rms_line_error_khz,
        transitions_mhz=tuple(float(value) / 1e6 for value in transitions_hz),
        notes="",
    )


def _non_ok_case(case: dict, status: str, note: str) -> CaseResult:
    return CaseResult(
        case_id=case["case_id"],
        icsd_code=case["icsd_code"],
        cif_file=case["cif_file"],
        status=status,
        metadata_score=float(case["metadata_score"]),
        metadata_verdict=case["metadata_verdict"],
        temperature_k=case["temperature_k"],
        space_group=case["space_group"],
        mean_cq_mhz=None,
        mean_abs_cq_mhz=None,
        mean_eta=None,
        eta_error=None,
        cq_percent_error=None,
        rms_line_error_khz=None,
        transitions_mhz=(),
        notes=note,
    )


def _write_csv(path: Path, results: list[CaseResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "case_id",
        "icsd_code",
        "cif_file",
        "status",
        "metadata_score",
        "metadata_verdict",
        "temperature_k",
        "space_group",
        "mean_cq_mhz",
        "mean_abs_cq_mhz",
        "mean_eta",
        "eta_error",
        "cq_percent_error",
        "rms_line_error_khz",
        "transitions_mhz",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(_result_row(rank, result))


def _write_markdown(
    path: Path,
    results: list[CaseResult],
    *,
    reference_cq_mhz: float,
    reference_eta: float,
    reference_transitions_hz: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NaNO2 CIF EFG Validation",
        "",
        (
            "Static ABINIT EFG simulations ranked against the room-temperature "
            "14N NaNO2 reference."
        ),
        "",
        f"Reference `|C_Q|`: {reference_cq_mhz:.6f} MHz",
        f"Reference `eta`: {reference_eta:.6f}",
        "Reference lines (MHz): "
        + ", ".join(f"{value / 1e6:.6f}" for value in reference_transitions_hz),
        "",
        "| Rank | Case | Status | Metadata score | T (K) | SG | "
        "`|C_Q|` MHz | `eta` | RMS line error (kHz) | Lines (MHz) | Notes |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---|---|",
    ]
    for rank, result in enumerate(results, start=1):
        lines.append(
            "| "
            f"{rank} | `{result.case_id}` | {result.status} | "
            f"{result.metadata_score:.1f} | {_fmt(result.temperature_k)} | "
            f"{result.space_group or ''} | {_fmt(result.mean_abs_cq_mhz)} | "
            f"{_fmt(result.mean_eta)} | {_fmt(result.rms_line_error_khz)} | "
            f"{_format_lines(result.transitions_mhz)} | {result.notes} |"
        )
    ok_results = [result for result in results if result.status == "ok"]
    if ok_results:
        best = ok_results[0]
        lines.extend(
            [
                "",
                (
                    f"Best simulated agreement: `{best.case_id}` "
                    f"(ICSD {best.icsd_code}), RMS line error "
                    f"{best.rms_line_error_khz:.1f} kHz."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_collection(results: list[CaseResult]) -> None:
    ok_results = [result for result in results if result.status == "ok"]
    if ok_results:
        best = ok_results[0]
        print(
            f"Best simulated agreement: {best.case_id} "
            f"(ICSD {best.icsd_code}), RMS line error "
            f"{best.rms_line_error_khz:.1f} kHz, eta={best.mean_eta:.4f}"
        )
    else:
        print("No completed EFG outputs were collected.")
    print("Collected cases:")
    for result in results:
        print(
            f"  {result.case_id:<22} {result.status:<8} "
            f"eta={_fmt(result.mean_eta):>8} "
            f"|C_Q|={_fmt(result.mean_abs_cq_mhz):>8} "
            f"rms={_fmt(result.rms_line_error_khz):>8} kHz"
        )


def _result_row(rank: int, result: CaseResult) -> dict:
    return {
        "rank": rank,
        "case_id": result.case_id,
        "icsd_code": result.icsd_code,
        "cif_file": result.cif_file,
        "status": result.status,
        "metadata_score": f"{result.metadata_score:.3f}",
        "metadata_verdict": result.metadata_verdict,
        "temperature_k": _fmt(result.temperature_k),
        "space_group": result.space_group,
        "mean_cq_mhz": _fmt(result.mean_cq_mhz),
        "mean_abs_cq_mhz": _fmt(result.mean_abs_cq_mhz),
        "mean_eta": _fmt(result.mean_eta),
        "eta_error": _fmt(result.eta_error),
        "cq_percent_error": _fmt(result.cq_percent_error),
        "rms_line_error_khz": _fmt(result.rms_line_error_khz),
        "transitions_mhz": _format_lines(result.transitions_mhz),
        "notes": result.notes,
    }


def _cif_atom_sites(loops: list[tuple[list[str], list[list[str]]]]) -> list[dict]:
    for tags, rows in loops:
        if "_atom_site_label" not in tags:
            continue
        label_i = tags.index("_atom_site_label")
        element_i = tags.index("_atom_site_type_symbol")
        fx_i = tags.index("_atom_site_fract_x")
        fy_i = tags.index("_atom_site_fract_y")
        fz_i = tags.index("_atom_site_fract_z")
        atoms = []
        for row in rows:
            element = _element(row[element_i])
            atoms.append(
                {
                    "label": row[label_i],
                    "element": element,
                    "z": ELEMENT_Z[element],
                    "fractional": np.array(
                        [
                            _float_cif(row[fx_i]),
                            _float_cif(row[fy_i]),
                            _float_cif(row[fz_i]),
                        ],
                        dtype=float,
                    ),
                }
            )
        return atoms
    raise ValueError("CIF contains no atom-site loop")


def _expand_atoms(atoms: list[dict], sym_ops: Iterable[str]) -> list[ExpandedAtom]:
    expanded: list[ExpandedAtom] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    for sym_index, operation in enumerate(sym_ops, start=1):
        for atom in atoms:
            fractional = np.mod(
                _apply_symmetry_operation(operation, atom["fractional"]),
                1.0,
            )
            rounded = tuple(float(value) for value in np.round(fractional, 10))
            key = (atom["label"], rounded)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                ExpandedAtom(
                    label=atom["label"],
                    element=atom["element"],
                    z=atom["z"],
                    fractional=fractional,
                    symmetry_index=sym_index,
                )
            )
    return expanded


def _read_manifest(workdir: Path) -> dict:
    manifest_path = workdir / "nano2_cif_efg_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _parse_include_codes(values: list[str]) -> set[str]:
    codes: set[str] = set()
    for value in values:
        for item in value.split(","):
            stripped = item.strip()
            if stripped:
                codes.add(stripped)
    return codes


def _case_failure_note(case: dict) -> str:
    input_path = _resolve(case["input_file"])
    name = input_path.stem
    job_dir = input_path.parent / f"{name}.run"
    texts = []
    for suffix in ("stdout", "stderr"):
        path = job_dir / f"{name}.{suffix}"
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(texts)
    if not text:
        return ""
    if "--- !ERROR" not in text and "abinit_abort" not in text:
        return ""
    if "PAW SPHERES ARE OVERLAPPING" in text:
        return (
            "ABINIT aborted: PAW spheres overlap. This usually means the CIF "
            "has partial/disordered split sites and needs an ordered approximant."
        )
    if "PAW COMPENSATION DENSITIES ARE OVERLAPPING" in text:
        return "ABINIT aborted: PAW compensation densities overlap."
    action_match = re.search(r"Action:\s*(?P<message>.+)", text)
    if action_match:
        return "ABINIT aborted: " + action_match.group("message").strip()
    error_match = re.search(r"--- !ERROR(?P<body>.*?)(?:\n---|\Z)", text, re.S)
    if error_match:
        body = " ".join(error_match.group("body").split())
        return f"ABINIT error: {body[:240]}"
    if "MPI_ABORT" in text:
        return "ABINIT/MPI abort; inspect the job stdout/stderr."
    return ""


def _case_dry_run_note(case: dict) -> str:
    input_path = _resolve(case["input_file"])
    name = input_path.stem
    job_dir = input_path.parent / f"{name}.run"
    texts = []
    for suffix in ("stdout", "stderr"):
        path = job_dir / f"{name}.{suffix}"
        if path.exists():
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    text = "\n".join(texts)
    if "debugging mode => will skip driver" in text:
        return "ABINIT dry-run/debugging output only; full EFG was not run."
    return ""


def _parse_int_triplet(value: str) -> tuple[int, int, int]:
    items = tuple(int(item.strip()) for item in value.split(","))
    if len(items) != 3:
        raise SystemExit("--ngkpt must contain exactly three comma-separated ints")
    return items


def _is_full_occupancy(assessment) -> bool:
    return assessment.min_occupancy is None or assessment.min_occupancy >= 0.999


def _is_ferroelectric(space_group: str | None) -> bool:
    return _normalize_space_group(space_group) == "im2m"


def _is_paraelectric(space_group: str | None) -> bool:
    return _normalize_space_group(space_group) == "immm"


def _normalize_space_group(space_group: str | None) -> str:
    return re.sub(r"\s+", "", (space_group or "").strip("'\"")).lower()


def _item(items: dict[str, str], *tags: str) -> str | None:
    for tag in tags:
        value = items.get(tag)
        if value not in {None, "?", "."}:
            return value
    return None


def _element(value: str) -> str:
    match = re.search(r"[A-Z][a-z]?", str(value).strip().strip("'\""))
    if match is None:
        raise ValueError(f"could not infer element from atom type {value!r}")
    return match.group(0)


def _format_int_rows(values: list[int], columns: int = 12) -> list[str]:
    rows = []
    for start in range(0, len(values), columns):
        rows.append(
            "  " + " ".join(str(value) for value in values[start:start + columns])
        )
    return rows


def _resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


def _format_lines(values: tuple[float, ...]) -> str:
    return ", ".join(f"{value:.6f}" for value in values)


if __name__ == "__main__":
    main()
