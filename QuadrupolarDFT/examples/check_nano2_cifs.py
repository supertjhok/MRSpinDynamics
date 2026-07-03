"""Rank bundled NaNO2 CIFs as EFG/NQR structure candidates.

This script is a metadata and geometry screen, not a replacement for a static
EFG calculation.  It asks: among the NaNO2 CIFs already bundled in
``structures/NaNO2``, which file is the best crystallographic starting point for
a room-temperature ferroelectric 14N EFG/NQR run?

Example
-------
PYTHONPATH=src python examples/check_nano2_cifs.py \
    --csv results/nano2_cif_screen.csv \
    --markdown results/nano2_cif_screen.md
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np

from quadrupolar_dft.strain_coupling import (
    _apply_symmetry_operation,
    _cif_lattice,
    _cif_symmetry_operations,
    _float_cif,
    _parse_cif_tokens,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CIF_DIR = REPO_ROOT / "structures" / "NaNO2"

# The ICSD 82857 CIF does not carry _diffrn_ambient_temperature, but the local
# source note identifies it as the preferred room-temperature ferroelectric
# structure.  Keep that assumption explicit so the score is inspectable.
TEMPERATURE_OVERRIDES_K = {
    "82857": (295.0, "assumed room-temperature entry from source.md"),
}

TARGET_SPACE_GROUP = "I m 2 m"
PARAELECTRIC_SPACE_GROUP = "I m m m"


@dataclass(frozen=True)
class AtomSite:
    label: str
    element: str
    fractional: np.ndarray
    occupancy: float | None
    raw_fractional: tuple[str, str, str]


@dataclass(frozen=True)
class CifAssessment:
    path: Path
    icsd_code: str
    temperature_k: float | None
    temperature_note: str
    space_group: str | None
    citation_year: int | None
    citation_title: str | None
    citation_journal: str | None
    method_hint: str
    has_anisotropic_displacements: bool
    min_occupancy: float | None
    mean_coordinate_uncertainty: float | None
    cell_lengths: tuple[float, float, float]
    volume: float | None
    n_o_distance_angstrom: float | None
    o_n_o_angle_deg: float | None
    score: float
    verdict: str
    reasons: tuple[str, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=DEFAULT_CIF_DIR,
        help="Directory containing EntryWithCollCode*.cif files.",
    )
    parser.add_argument(
        "--target-temperature",
        type=float,
        default=295.0,
        help="Target experimental temperature in K for the ranking.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV report path.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional Markdown report path.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
        help="Number of ranked entries to print to stdout.",
    )
    args = parser.parse_args()

    assessments = sorted(
        (
            assess_cif(path, target_temperature_k=args.target_temperature)
            for path in sorted(args.cif_dir.glob("*.cif"))
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    if not assessments:
        raise SystemExit(f"No CIF files found in {args.cif_dir}")

    _print_summary(
        assessments,
        top=args.top,
        target_temperature_k=args.target_temperature,
    )
    if args.csv:
        _write_csv(args.csv, assessments)
        print(f"\nWrote CSV report: {args.csv}")
    if args.markdown:
        _write_markdown(
            args.markdown,
            assessments,
            target_temperature_k=args.target_temperature,
        )
        print(f"Wrote Markdown report: {args.markdown}")


def assess_cif(path: Path, *, target_temperature_k: float) -> CifAssessment:
    text = path.read_text(encoding="utf-8")
    items, loops = _parse_cif_tokens(text)
    atoms = _atom_sites(loops)
    lattice = _cif_lattice(items)
    icsd_code = _item(items, "_database_code_ICSD") or _code_from_filename(path)
    temperature_k, temperature_note = _temperature(items, icsd_code)
    cell_lengths = tuple(
        _float_cif(items[tag])
        for tag in ("_cell_length_a", "_cell_length_b", "_cell_length_c")
    )
    space_group = _item(
        items,
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
    )
    citation_year, citation_journal = _citation_year_journal(loops)
    citation_title = _item(items, "_citation_title")
    method_hint = _method_hint(citation_title, citation_journal)
    has_aniso = any(
        any(tag.startswith("_atom_site_aniso_") for tag in tags)
        for tags, _rows in loops
    )
    min_occupancy = _min_occupancy(atoms)
    mean_uncertainty = _mean_coordinate_uncertainty(atoms)
    no_distance, ono_angle = _nitrite_geometry(atoms, loops, lattice)
    volume = _optional_float(items, "_cell_volume")

    score, verdict, reasons = _score(
        space_group=space_group,
        temperature_k=temperature_k,
        temperature_note=temperature_note,
        target_temperature_k=target_temperature_k,
        citation_year=citation_year,
        method_hint=method_hint,
        has_anisotropic_displacements=has_aniso,
        min_occupancy=min_occupancy,
        mean_coordinate_uncertainty=mean_uncertainty,
        citation_title=citation_title,
    )

    return CifAssessment(
        path=path,
        icsd_code=icsd_code,
        temperature_k=temperature_k,
        temperature_note=temperature_note,
        space_group=space_group,
        citation_year=citation_year,
        citation_title=citation_title,
        citation_journal=citation_journal,
        method_hint=method_hint,
        has_anisotropic_displacements=has_aniso,
        min_occupancy=min_occupancy,
        mean_coordinate_uncertainty=mean_uncertainty,
        cell_lengths=cell_lengths,
        volume=volume,
        n_o_distance_angstrom=no_distance,
        o_n_o_angle_deg=ono_angle,
        score=score,
        verdict=verdict,
        reasons=tuple(reasons),
    )


def _score(
    *,
    space_group: str | None,
    temperature_k: float | None,
    temperature_note: str,
    target_temperature_k: float,
    citation_year: int | None,
    method_hint: str,
    has_anisotropic_displacements: bool,
    min_occupancy: float | None,
    mean_coordinate_uncertainty: float | None,
    citation_title: str | None,
) -> tuple[float, str, list[str]]:
    score = 0.0
    reasons: list[str] = []
    sg = _normalize_space_group(space_group)

    if sg == _normalize_space_group(TARGET_SPACE_GROUP):
        score += 30.0
        reasons.append("ferroelectric Im2m model")
    elif sg == _normalize_space_group(PARAELECTRIC_SPACE_GROUP):
        score -= 8.0
        reasons.append("paraelectric Immm model; likely symmetry-averages eta")
    else:
        score += 6.0
        reasons.append("unknown/nonstandard space group for this screen")

    title = (citation_title or "").lower()
    if any(word in title for word in ("disordered", "paraelectric", "diffuse")):
        score -= 8.0
        reasons.append("title indicates disorder/paraelectric diffuse scattering")
    if "modulated" in title or "incommensurate" in title:
        score -= 4.0
        reasons.append("modulated/incommensurate entry is not a simple static cell")

    if temperature_k is None:
        score += 8.0
        reasons.append("missing explicit temperature")
    else:
        delta_t = abs(temperature_k - target_temperature_k)
        if delta_t <= 30.0:
            score += 25.0
            reasons.append(f"temperature close to target ({temperature_note})")
        elif delta_t <= 100.0:
            score += 17.0
            reasons.append(f"temperature moderately close ({temperature_note})")
        elif delta_t <= 180.0:
            score += 10.0
            reasons.append(f"temperature comparison case ({temperature_note})")
        else:
            score += 4.0
            reasons.append(f"temperature far from target ({temperature_note})")

    if method_hint == "neutron":
        score += 15.0
        reasons.append("neutron refinement/source")
    elif method_hint == "electron_density":
        score += 13.0
        reasons.append("electron/deformation-density study")
    elif method_hint == "xray":
        score += 8.0
        reasons.append("X-ray refinement/source")
    else:
        score += 6.0
        reasons.append("method not explicit in CIF title")

    if mean_coordinate_uncertainty is None:
        score += 3.0
        reasons.append("coordinate uncertainties not reported")
    elif mean_coordinate_uncertainty <= 1.0e-4:
        score += 15.0
        reasons.append("high fractional-coordinate precision")
    elif mean_coordinate_uncertainty <= 5.0e-4:
        score += 12.0
        reasons.append("good fractional-coordinate precision")
    elif mean_coordinate_uncertainty <= 1.5e-3:
        score += 8.0
        reasons.append("moderate fractional-coordinate precision")
    else:
        score += 4.0
        reasons.append("coarse fractional-coordinate precision")

    if has_anisotropic_displacements:
        score += 5.0
        reasons.append("anisotropic displacement parameters present")

    if min_occupancy is not None and min_occupancy >= 0.999:
        score += 10.0
        reasons.append("complete ordered Na/N/O occupancies")
    elif min_occupancy is not None:
        score -= 5.0
        reasons.append("partial occupancies present")

    if citation_year is not None:
        if citation_year >= 1990:
            score += 5.0
            reasons.append("modern refinement")
        elif citation_year >= 1960:
            score += 2.0
            reasons.append("older but post-1960 source")

    score = max(0.0, min(100.0, score))
    if score >= 90.0:
        verdict = "best target"
    elif score >= 75.0:
        verdict = "strong comparison"
    elif score >= 50.0:
        verdict = "diagnostic comparison"
    else:
        verdict = "poor RT-ferroelectric target"
    return score, verdict, reasons


def _atom_sites(loops: list[tuple[list[str], list[list[str]]]]) -> list[AtomSite]:
    for tags, rows in loops:
        if "_atom_site_label" not in tags:
            continue
        label_i = tags.index("_atom_site_label")
        element_i = tags.index("_atom_site_type_symbol")
        fx_i = tags.index("_atom_site_fract_x")
        fy_i = tags.index("_atom_site_fract_y")
        fz_i = tags.index("_atom_site_fract_z")
        occupancy_i = (
            tags.index("_atom_site_occupancy")
            if "_atom_site_occupancy" in tags
            else None
        )
        atoms: list[AtomSite] = []
        for row in rows:
            raw_fractional = (row[fx_i], row[fy_i], row[fz_i])
            occupancy = (
                _safe_float(row[occupancy_i])
                if occupancy_i is not None
                else None
            )
            atoms.append(
                AtomSite(
                    label=row[label_i],
                    element=_element(row[element_i]),
                    fractional=np.array(
                        [_float_cif(value) for value in raw_fractional],
                        dtype=float,
                    ),
                    occupancy=occupancy,
                    raw_fractional=raw_fractional,
                )
            )
        return atoms
    raise ValueError("CIF contains no atom-site loop")


def _nitrite_geometry(
    atoms: list[AtomSite],
    loops: list[tuple[list[str], list[list[str]]]],
    lattice: np.ndarray,
) -> tuple[float | None, float | None]:
    sym_ops = _cif_symmetry_operations(loops) or ("x,y,z",)
    expanded = _expanded_sites(atoms, sym_ops)
    nitrogens = [atom for atom in expanded if atom.element == "N"]
    oxygens = [atom for atom in expanded if atom.element == "O"]
    if not nitrogens or len(oxygens) < 2:
        return None, None

    distances: list[float] = []
    angles: list[float] = []
    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    for nitrogen in nitrogens:
        vectors: list[tuple[float, np.ndarray]] = []
        for oxygen in oxygens:
            for shift in shifts:
                delta = oxygen.fractional + shift - nitrogen.fractional
                cart = delta @ lattice
                length = float(np.linalg.norm(cart))
                if length > 1.0e-8:
                    vectors.append((length, cart))
        nearest = sorted(vectors, key=lambda item: item[0])[:2]
        if len(nearest) < 2:
            continue
        distances.extend([nearest[0][0], nearest[1][0]])
        v1 = nearest[0][1]
        v2 = nearest[1][1]
        cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angles.append(float(np.degrees(np.arccos(cos_angle))))
    if not distances or not angles:
        return None, None
    return float(np.mean(distances)), float(np.mean(angles))


def _expanded_sites(
    atoms: list[AtomSite],
    sym_ops: Iterable[str],
) -> list[AtomSite]:
    expanded: list[AtomSite] = []
    seen: set[tuple[str, tuple[float, float, float]]] = set()
    for operation in sym_ops:
        for atom in atoms:
            fractional = np.mod(
                _apply_symmetry_operation(operation, atom.fractional),
                1.0,
            )
            rounded = tuple(float(value) for value in np.round(fractional, 10))
            key = (atom.element, rounded)
            if key in seen:
                continue
            seen.add(key)
            expanded.append(
                AtomSite(
                    label=atom.label,
                    element=atom.element,
                    fractional=fractional,
                    occupancy=atom.occupancy,
                    raw_fractional=atom.raw_fractional,
                )
            )
    return expanded


def _write_csv(path: Path, assessments: list[CifAssessment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "file",
        "icsd_code",
        "score",
        "verdict",
        "temperature_k",
        "temperature_note",
        "space_group",
        "citation_year",
        "method_hint",
        "has_anisotropic_displacements",
        "min_occupancy",
        "mean_coordinate_uncertainty",
        "cell_a",
        "cell_b",
        "cell_c",
        "volume",
        "n_o_distance_angstrom",
        "o_n_o_angle_deg",
        "citation_title",
        "reasons",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, assessment in enumerate(assessments, start=1):
            writer.writerow(_row(rank, assessment))


def _write_markdown(
    path: Path,
    assessments: list[CifAssessment],
    *,
    target_temperature_k: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = assessments[0]
    lines = [
        "# NaNO2 CIF Screen",
        "",
        f"Target: ferroelectric `I m 2 m` NaNO2 near {target_temperature_k:g} K.",
        "",
        (
            "This ranking screens crystallographic inputs for an EFG/NQR workflow. "
            "It does not prove that the resulting DFT `eta` will be accurate; "
            "ABINIT EFG runs are still required."
        ),
        "",
        f"Best candidate: `{best.path.name}` (ICSD {best.icsd_code}), "
        f"score {best.score:.1f}/100.",
        "",
        "| Rank | CIF | Score | Verdict | T (K) | SG | Method | "
        "N-O (A) | O-N-O (deg) | Key reasons |",
        "|---:|---|---:|---|---:|---|---|---:|---:|---|",
    ]
    for rank, assessment in enumerate(assessments, start=1):
        lines.append(
            "| "
            f"{rank} | `{assessment.path.name}` | {assessment.score:.1f} | "
            f"{assessment.verdict} | {_fmt(assessment.temperature_k)} | "
            f"{assessment.space_group or ''} | {assessment.method_hint} | "
            f"{_fmt(assessment.n_o_distance_angstrom)} | "
            f"{_fmt(assessment.o_n_o_angle_deg)} | "
            f"{'; '.join(assessment.reasons[:4])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(
    assessments: list[CifAssessment],
    *,
    top: int,
    target_temperature_k: float,
) -> None:
    best = assessments[0]
    print(
        "NaNO2 CIF screen: target ferroelectric Im2m "
        f"near {target_temperature_k:g} K"
    )
    print(
        f"Best candidate: {best.path.name} (ICSD {best.icsd_code}), "
        f"score {best.score:.1f}/100"
    )
    print(f"Verdict: {best.verdict}")
    print("Reasons:")
    for reason in best.reasons:
        print(f"  - {reason}")
    print("\nTop candidates:")
    for rank, assessment in enumerate(assessments[:top], start=1):
        print(
            f"{rank:2d}. {assessment.path.name:<30} "
            f"score={assessment.score:5.1f} "
            f"T={_fmt(assessment.temperature_k):>6} "
            f"SG={assessment.space_group or '':<7} "
            f"{assessment.verdict}"
        )


def _row(rank: int, assessment: CifAssessment) -> dict:
    return {
        "rank": rank,
        "file": assessment.path.name,
        "icsd_code": assessment.icsd_code,
        "score": f"{assessment.score:.3f}",
        "verdict": assessment.verdict,
        "temperature_k": _fmt(assessment.temperature_k),
        "temperature_note": assessment.temperature_note,
        "space_group": assessment.space_group,
        "citation_year": assessment.citation_year,
        "method_hint": assessment.method_hint,
        "has_anisotropic_displacements": assessment.has_anisotropic_displacements,
        "min_occupancy": _fmt(assessment.min_occupancy),
        "mean_coordinate_uncertainty": _fmt(assessment.mean_coordinate_uncertainty),
        "cell_a": f"{assessment.cell_lengths[0]:.6g}",
        "cell_b": f"{assessment.cell_lengths[1]:.6g}",
        "cell_c": f"{assessment.cell_lengths[2]:.6g}",
        "volume": _fmt(assessment.volume),
        "n_o_distance_angstrom": _fmt(assessment.n_o_distance_angstrom),
        "o_n_o_angle_deg": _fmt(assessment.o_n_o_angle_deg),
        "citation_title": assessment.citation_title,
        "reasons": "; ".join(assessment.reasons),
    }


def _temperature(
    items: dict[str, str],
    icsd_code: str,
) -> tuple[float | None, str]:
    if icsd_code in TEMPERATURE_OVERRIDES_K:
        value, note = TEMPERATURE_OVERRIDES_K[icsd_code]
        return value, note
    for tag in ("_diffrn_ambient_temperature", "_cell_measurement_temperature"):
        if tag in items:
            return _float_cif(items[tag]), tag
    return None, "not reported"


def _citation_year_journal(
    loops: list[tuple[list[str], list[list[str]]]],
) -> tuple[int | None, str | None]:
    for tags, rows in loops:
        if "_citation_year" not in tags or not rows:
            continue
        year = int(float(rows[0][tags.index("_citation_year")]))
        journal = None
        if "_citation_journal_full" in tags:
            journal = rows[0][tags.index("_citation_journal_full")]
        return year, journal
    return None, None


def _method_hint(title: str | None, journal: str | None) -> str:
    text = f"{title or ''} {journal or ''}".lower()
    if "neutron" in text:
        return "neutron"
    if "electron density" in text or "deformation density" in text:
        return "electron_density"
    if "x-ray" in text or "x ray" in text:
        return "xray"
    return "unspecified"


def _mean_coordinate_uncertainty(atoms: list[AtomSite]) -> float | None:
    uncertainties = [
        uncertainty
        for atom in atoms
        for value in atom.raw_fractional
        if (uncertainty := _cif_uncertainty(value)) is not None
    ]
    if not uncertainties:
        return None
    return float(np.mean(uncertainties))


def _cif_uncertainty(value: str) -> float | None:
    text = str(value).strip().strip("'\"")
    match = re.match(r"[-+]?(?P<number>\d+(?:\.\d*)?|\.\d+)\((?P<digits>\d+)\)$", text)
    if match is None:
        return None
    number = match.group("number")
    decimals = len(number.split(".", maxsplit=1)[1]) if "." in number else 0
    return int(match.group("digits")) * 10.0 ** (-decimals)


def _min_occupancy(atoms: list[AtomSite]) -> float | None:
    occupancies = [atom.occupancy for atom in atoms if atom.occupancy is not None]
    if not occupancies:
        return None
    return min(occupancies)


def _item(items: dict[str, str], *tags: str) -> str | None:
    for tag in tags:
        value = items.get(tag)
        if value not in {None, "?", "."}:
            return value
    return None


def _optional_float(items: dict[str, str], tag: str) -> float | None:
    value = items.get(tag)
    if value in {None, "?", "."}:
        return None
    return _float_cif(value)


def _safe_float(value: str | None) -> float | None:
    if value in {None, "?", "."}:
        return None
    try:
        return _float_cif(str(value))
    except ValueError:
        return None


def _element(value: str) -> str:
    match = re.search(r"[A-Z][a-z]?", str(value).strip().strip("'\""))
    if match is None:
        raise ValueError(f"could not infer element from atom type {value!r}")
    return match.group(0)


def _code_from_filename(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else path.stem


def _normalize_space_group(space_group: str | None) -> str:
    return re.sub(r"\s+", "", (space_group or "").strip("'\"")).lower()


def _fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()
