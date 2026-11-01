"""Survey and execute the repository's cross-project NQR integration targets.

This module joins structure directories from ``QuadrupolarDFT``, completed EFG
summary rows, and isotope-tagged measured lines from ``NQRDatabase``.  The result
is both an actionable DFT queue and a data-driven way to execute every currently
available DFT -> simulator -> measurement comparison.

Data locations default to an MRSpinDynamics checkout and can be redirected with
the ``MR_INTEGRATION_*`` environment variables (see :mod:`mr_integration.paths`).
Missing inputs raise :class:`FileNotFoundError` rather than silently producing
an empty survey.

**Site averaging.**  Each summary row carries ``mean_cq_mhz``/``mean_eta``
averaged over the row's ``atoms``.  That average only describes a physical site
when those atoms are crystallographically equivalent.  The row's per-atom
``nqr_transitions_mhz`` lists let us check this: a row whose atoms' lines differ
by more than :data:`SITE_EQUIVALENCE_RTOL` (relative to the largest line) is
treated as averaging *non-equivalent* sites and is excluded from
:attr:`IntegrationTargetCoverage.comparison_ready_records`, because no single
``(C_Q, eta)`` reproduces such a row.  Rows without per-atom lines cannot be
checked and are used as-is (``sites_equivalent is None``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .conversions import nu_q_from_cq_hz
from .database import MeasuredLine, measured_lines
from .paths import (
    DFT_SUMMARY_ENV,
    STRUCTURES_ENV,
    default_dft_summary_path,
    default_structures_path,
    missing_data_message,
)
from .pipeline import ComparisonReport, compare_dft_to_measured

#: Per-atom NQR lines that agree to this relative tolerance are treated as one
#: crystallographic site.  Unsymmetrized ABINIT runs leave ~1e-5-1e-3 noise
#: between symmetry-equivalent atoms (NaNO2 14N: 2e-5; 23Na: 1.3e-3), while
#: genuinely different EFGs differ by percent (NaNO2 starter 17O: 2.3%).
SITE_EQUIVALENCE_RTOL = 5.0e-3


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
    #: Per-atom NQR lines (Hz) from ``nqr_transitions_mhz``, in ``atoms`` order;
    #: empty when the summary does not provide them.
    per_atom_lines_hz: tuple[tuple[float, ...], ...] = ()

    @property
    def sites_equivalent(self) -> bool | None:
        """Whether the averaged atoms are one site (``None`` if unknown).

        True when every atom's line list matches the first atom's within
        :data:`SITE_EQUIVALENCE_RTOL` of the largest line; ``None`` when the
        summary has no per-atom lines.
        """

        if not self.per_atom_lines_hz:
            return None
        reference = self.per_atom_lines_hz[0]
        scale = max((abs(value) for value in reference), default=0.0)
        for lines in self.per_atom_lines_hz[1:]:
            if len(lines) != len(reference):
                return False
            for value, ref in zip(sorted(lines), sorted(reference)):
                if abs(value - ref) > SITE_EQUIVALENCE_RTOL * scale:
                    return False
        return True


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
        """DFT rows that can be compared against measured lines.

        A row qualifies when its isotope has measured lines, its spin is
        quadrupolar (any ``I >= 1``), and its averaged atoms are not known to be
        non-equivalent sites (see :attr:`DFTSummaryRecord.sites_equivalent`).
        """

        measured_isotopes = {isotope for isotope, _ in self.isotope_line_counts}
        return tuple(
            record
            for record in self.dft_records
            if record.isotope in measured_isotopes
            and _quadrupolar_spin(record.spin)
            and record.sites_equivalent is not False
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


def _quadrupolar_spin(spin: float) -> bool:
    try:
        nu_q_from_cq_hz(1.0, spin)
    except ValueError:
        return False
    return True


def _per_atom_lines(text: str | None) -> tuple[tuple[float, ...], ...]:
    """Parse ``"a, b, c; d, e, f"`` (MHz, one group per atom) into Hz tuples."""

    groups: list[tuple[float, ...]] = []
    for group in str(text or "").split(";"):
        values = tuple(
            float(token) * 1.0e6 for token in group.split(",") if token.strip()
        )
        if values:
            groups.append(values)
    return tuple(groups)


def load_dft_summary(path: str | Path | None = None) -> tuple[DFTSummaryRecord, ...]:
    """Load curated isotope-level EFG summary rows from a CSV file.

    Raises :class:`FileNotFoundError` when the summary is missing (it used to
    return ``()``, which silently turned an installed package's survey into
    "no DFT rows").
    """

    if path is not None:
        summary_path = Path(path)
        if not summary_path.is_file():
            raise FileNotFoundError(f"DFT summary not found at {summary_path}")
    else:
        summary_path = default_dft_summary_path()
        if not summary_path.is_file():
            raise FileNotFoundError(
                missing_data_message("DFT summary CSV", summary_path, DFT_SUMMARY_ENV)
            )

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
                    per_atom_lines_hz=_per_atom_lines(row.get("nqr_transitions_mhz")),
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

    if structures_path is not None:
        root = Path(structures_path)
        if not root.is_dir():
            raise FileNotFoundError(f"structure directory not found at {root}")
    else:
        root = default_structures_path()
        if not root.is_dir():
            raise FileNotFoundError(
                missing_data_message("DFT structure directory", root, STRUCTURES_ENV)
            )
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
