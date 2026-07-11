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
from .coverage import (
    DEFAULT_TARGET_SPECS,
    AvailableDFTComparison,
    DFTSummaryRecord,
    IntegrationTargetCoverage,
    IntegrationTargetSpec,
    compare_available_dft,
    format_target_survey,
    load_dft_summary,
    survey_integration_targets,
)
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
from .temperature import (
    MeasuredTemperatureCoefficient,
    TemperatureCoefficientComparison,
    TemperatureCoefficientMatch,
    compare_temperature_coefficients,
    measured_temperature_coefficients,
    slopes_from_temperature_points,
)
from .uncertainty import (
    PredictedLineInterval,
    QuadrupolarParameterDistribution,
    UncertainComparisonReport,
    UncertainLineMatch,
    UncertainLinePrediction,
    compare_uncertain_dft_to_measured,
    propagate_parameter_uncertainty,
)

__all__ = [
    "AvailableDFTComparison",
    "ComparisonReport",
    "DEFAULT_TARGET_SPECS",
    "DFTSummaryRecord",
    "FlagExportSummary",
    "LandoltConsistencyReport",
    "LandoltReviewSummary",
    "LandoltSetRecord",
    "IntegrationTargetCoverage",
    "IntegrationTargetSpec",
    "MeasuredLine",
    "MeasuredTemperatureCoefficient",
    "PredictedLines",
    "PredictedLineInterval",
    "QuadrupolarParameterDistribution",
    "TemperatureCoefficientComparison",
    "TemperatureCoefficientMatch",
    "UncertainComparisonReport",
    "UncertainLineMatch",
    "UncertainLinePrediction",
    "SiteConsistencyReport",
    "SiteRecord",
    "check_landolt_set",
    "check_site",
    "compare_available_dft",
    "compare_dft_to_measured",
    "compare_temperature_coefficients",
    "compare_uncertain_dft_to_measured",
    "cq_hz_from_nu_q",
    "default_database_path",
    "describe",
    "describe_landolt",
    "format_target_survey",
    "load_dft_summary",
    "match_lines",
    "measured_lines",
    "measured_temperature_coefficients",
    "nu_q_from_cq_hz",
    "slopes_from_temperature_points",
    "parse_nucleus",
    "predicted_lines",
    "propagate_parameter_uncertainty",
    "quadrupolar_site_from_cq",
    "quadrupolar_site_from_efg_record",
    "sites_with_parameters",
    "spin1_parameters_from_lines",
    "summarize",
    "survey_integration_targets",
    "validate_database",
    "validate_landolt_sets",
    "write_consistency_flags",
    "write_landolt_review_flags",
]
