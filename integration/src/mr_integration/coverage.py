"""Survey and execute the repository's cross-project NQR integration targets.

This module joins structure directories from ``QuadrupolarDFT``, completed EFG
summary rows, and isotope-tagged measured lines from ``NQRDatabase``.  The result
is both an actionable DFT queue and a data-driven way to execute every currently
available DFT -> simulator -> measurement comparison.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .database import MeasuredLine, measured_lines
from .pipeline import ComparisonReport, compare_dft_to_measured

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_STRUCTURES = _REPOSITORY_ROOT / "QuadrupolarDFT" / "structures"
_DEFAULT_DFT_SUMMARY = (
    _REPOSITORY_ROOT / "QuadrupolarDFT" / "results" / "nano2_efg_summary.csv"
)


@dataclass(frozen=True)
class IntegrationTargetSpec:
    """Map one DFT structure directory to its database identity and DFT cases."""

    structure_name: str
    database_compound: str
    dft_case_prefixes: tuple[str, ...] = ()


# Explicit aliases prevent fuzzy matching from joining the wrong compound.
DEFAULT_TARGET_SPECS = (
    IntegrationTargetSpec("Acetaminophen", "Paracetamol"),
    IntegrationTargetSpec("Benzocaine", "Benzocaine"),
    IntegrationTargetSpec("Caffeine", "Caffeine"),
    IntegrationTargetSpec("Famotidine", "Famotidine - Pepcid - Prescription"),
    IntegrationTargetSpec("Glycine", "Glycine"),
    IntegrationTargetSpec("Hexamethylenetetramine", "Hexamethylenetetramine"),
    IntegrationTargetSpec("L-Proline", "L-Proline"),
    IntegrationTargetSpec("Melamine", "Melamine"),
    IntegrationTargetSpec("NaNO2", "Sodium Nitrite", ("nano2_",)),
    IntegrationTargetSpec("Nicotinamide", "Nicotinamide"),
)


@dataclass(frozen=True)
class DFTSummaryRecord:
    """One isotope-level row from a curated QuadrupolarDFT result summary."""

    case_id: str
    title: str
    isotope: str
    spin: float
    atom_indices: tuple[int, ...]
    mean_cq_hz: float
    mean_abs_cq_hz: float
    mean_eta: float
    source_path: Path


@dataclass(frozen=True)
class IntegrationTargetCoverage:
    """Available inputs and current readiness for one compound target."""

    spec: IntegrationTargetSpec
    structure_files: tuple[Path, ...]
    measured: tuple[MeasuredLine, ...]
    dft_records: tuple[DFTSummaryRecord, ...]

    @property
    def isotope_line_counts(self) -> tuple[tuple[str, int], ...]:
        """Return sorted measured-line counts for records with known isotopes."""

        counts: dict[str, int] = {}
        for line in self.measured:
            if line.isotope is not None:
                counts[line.isotope] = counts.get(line.isotope, 0) + 1
        return tuple(sorted(counts.items()))

    @property
    def comparison_ready_records(self) -> tuple[DFTSummaryRecord, ...]:
        """DFT rows with a calibrated spin and matching measured isotope lines."""

        measured_isotopes = {isotope for isotope, _ in self.isotope_line_counts}
        return tuple(
            record
            for record in self.dft_records
            if record.isotope in measured_isotopes and record.spin in (1.0, 1.5)
        )

    @property
    def status(self) -> str:
        """Return a readiness label ordered by the next missing input."""

        if self.comparison_ready_records:
            return "comparison-ready"
        if self.isotope_line_counts:
            return "dft-needed"
        if self.measured:
            return "isotope-metadata-needed"
        return "measurement-needed"


@dataclass(frozen=True)
class AvailableDFTComparison:
    """A DFT summary record and its simulator/database comparison."""

    target: IntegrationTargetCoverage
    dft_record: DFTSummaryRecord
    report: ComparisonReport


def load_dft_summary(path: str | Path | None = None) -> tuple[DFTSummaryRecord, ...]:
    """Load curated isotope-level EFG summary rows from a CSV file."""

    summary_path = Path(path) if path is not None else _DEFAULT_DFT_SUMMARY
    if not summary_path.exists():
        return ()

    records: list[DFTSummaryRecord] = []
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            atom_indices = tuple(
                int(token) for token in str(row.get("atoms", "")).split() if token
            )
            records.append(
                DFTSummaryRecord(
                    case_id=str(row["case_id"]),
                    title=str(row["title"]),
                    isotope=str(row["isotope"]),
                    spin=float(row["spin"]),
                    atom_indices=atom_indices,
                    mean_cq_hz=float(row["mean_cq_mhz"]) * 1.0e6,
                    mean_abs_cq_hz=float(row["mean_abs_cq_mhz"]) * 1.0e6,
                    mean_eta=float(row["mean_eta"]),
                    source_path=summary_path,
                )
            )
    return tuple(records)


def _records_for_target(
    spec: IntegrationTargetSpec,
    records: tuple[DFTSummaryRecord, ...],
) -> tuple[DFTSummaryRecord, ...]:
    return tuple(
        record
        for record in records
        if any(record.case_id.startswith(prefix) for prefix in spec.dft_case_prefixes)
    )


def survey_integration_targets(
    *,
    structures_path: str | Path | None = None,
    dft_summary_path: str | Path | None = None,
    database_path: str | Path | None = None,
    target_specs: tuple[IntegrationTargetSpec, ...] = DEFAULT_TARGET_SPECS,
) -> tuple[IntegrationTargetCoverage, ...]:
    """Join structure, DFT-result, and measured-line coverage for all targets."""

    root = Path(structures_path) if structures_path is not None else _DEFAULT_STRUCTURES
    dft_records = load_dft_summary(dft_summary_path)
    targets: list[IntegrationTargetCoverage] = []
    for spec in target_specs:
        target_dir = root / spec.structure_name
        structure_files = (
            tuple(
                sorted(
                    path
                    for path in target_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in {".cif", ".abi"}
                )
            )
            if target_dir.exists()
            else ()
        )
        targets.append(
            IntegrationTargetCoverage(
                spec=spec,
                structure_files=structure_files,
                measured=tuple(
                    measured_lines(
                        spec.database_compound,
                        database_path=database_path,
                    )
                ),
                dft_records=_records_for_target(spec, dft_records),
            )
        )
    order = {
        "comparison-ready": 0,
        "dft-needed": 1,
        "isotope-metadata-needed": 2,
        "measurement-needed": 3,
    }
    targets.sort(
        key=lambda target: (order[target.status], target.spec.database_compound)
    )
    return tuple(targets)


def compare_available_dft(
    targets: tuple[IntegrationTargetCoverage, ...],
    *,
    database_path: str | Path | None = None,
) -> tuple[AvailableDFTComparison, ...]:
    """Execute every comparison-ready DFT summary row in ``targets``."""

    comparisons: list[AvailableDFTComparison] = []
    for target in targets:
        for record in target.comparison_ready_records:
            report = compare_dft_to_measured(
                compound=target.spec.database_compound,
                cq_hz=record.mean_cq_hz,
                eta=record.mean_eta,
                spin=record.spin,
                isotope=record.isotope,
                database_path=database_path,
            )
            comparisons.append(
                AvailableDFTComparison(
                    target=target,
                    dft_record=record,
                    report=report,
                )
            )
    return tuple(comparisons)


def format_target_survey(targets: tuple[IntegrationTargetCoverage, ...]) -> str:
    """Format a compact, stable table for humans and CI logs."""

    rows = [
        "compound                              structures  isotope lines  DFT rows  status",
        (
            "------------------------------------  ----------  -------------  --------  "
            "-----------------------"
        ),
    ]
    for target in targets:
        isotope_counts = ",".join(
            f"{isotope}:{count}" for isotope, count in target.isotope_line_counts
        )
        line_summary = isotope_counts or str(len(target.measured))
        rows.append(
            f"{target.spec.database_compound[:36]:36}  "
            f"{len(target.structure_files):10d}  "
            f"{line_summary:>13}  "
            f"{len(target.dft_records):8d}  {target.status}"
        )
    return "\n".join(rows)
