"""Cross-project integration layer for MRSpinDynamics.

Bridges three subprojects into one ``predict -> simulate -> validate`` loop:

- ``quadrupolar_dft`` (ab initio EFG -> C_Q, eta);
- ``spin_dynamics`` (pulsed NQR simulation);
- the ``NQRDatabase`` SQLite export (measured frequencies).
"""

from __future__ import annotations

from .conversions import (
    cq_hz_from_nu_q,
    nu_q_from_cq_hz,
    quadrupolar_site_from_cq,
    quadrupolar_site_from_efg_record,
    spin1_parameters_from_lines,
)
from .cross_validation import PredictedLines, match_lines, predicted_lines
from .database import (
    MeasuredLine,
    SiteRecord,
    default_database_path,
    measured_lines,
    sites_with_parameters,
)
from .database_validation import (
    SiteConsistencyReport,
    check_site,
    describe,
    summarize,
    validate_database,
)
from .flag_export import FlagExportSummary, write_consistency_flags
from .landolt_review_export import (
    LandoltReviewSummary,
    write_landolt_review_flags,
)
from .landolt_validation import (
    LandoltConsistencyReport,
    LandoltSetRecord,
    check_landolt_set,
    describe_landolt,
    parse_nucleus,
    validate_landolt_sets,
)
from .pipeline import ComparisonReport, compare_dft_to_measured

__all__ = [
    "ComparisonReport",
    "FlagExportSummary",
    "LandoltConsistencyReport",
    "LandoltReviewSummary",
    "LandoltSetRecord",
    "MeasuredLine",
    "PredictedLines",
    "SiteConsistencyReport",
    "SiteRecord",
    "check_landolt_set",
    "check_site",
    "compare_dft_to_measured",
    "cq_hz_from_nu_q",
    "default_database_path",
    "describe",
    "describe_landolt",
    "match_lines",
    "measured_lines",
    "nu_q_from_cq_hz",
    "parse_nucleus",
    "predicted_lines",
    "quadrupolar_site_from_cq",
    "quadrupolar_site_from_efg_record",
    "sites_with_parameters",
    "spin1_parameters_from_lines",
    "summarize",
    "validate_database",
    "validate_landolt_sets",
    "write_consistency_flags",
    "write_landolt_review_flags",
]
