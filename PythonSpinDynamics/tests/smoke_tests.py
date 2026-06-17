"""Fast smoke-test suite for routine Python port development.

Run with:
    python -m unittest tests.smoke_tests

The full validation suite remains:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from tests.test_basic_octave_fixtures import OctaveFixtureTests
from tests.test_examples import ExampleSmokeTests


FAST_FIXTURE_TESTS = [
    "test_numpy_compatibility_helpers",
    "test_rephasing_analysis_recommends_finer_grid",
    "test_calc_time_domain_echo_matches_octave",
    "test_set_params_ideal_matches_octave",
    "test_jmr_parameter_constructors_return_expected_defaults",
    "test_tuned_spa_parameter_constructor_matches_matlab_defaults",
    "test_untuned_spa_parameter_constructor_matches_matlab_defaults",
    "test_matched_spa_parameter_constructor_matches_matlab_defaults",
    "test_quantize_phase_matches_matlab",
    "test_tuned_rectangular_pulse_response_matches_matlab",
    "test_spa_pulse_catalog_matches_matlab",
    "test_ideal_v0crit_refocusing_evaluation_returns_metrics",
    "test_ideal_v0crit_excited_refocusing_evaluation_returns_metrics",
    "test_ideal_time_varying_refocusing_evaluation_returns_metrics",
    "test_multistart_refocusing_export_uses_matlab_cell_shape",
    "test_analyze_matlab_optimization_results_uses_script_layout",
    "test_multistart_npz_export_round_trips_matlab_cells",
    "test_matlab_result_mat_round_trip_when_scipy_is_available",
    "test_matlab_tuned_excitation_result_fixture_matches_csv",
    "test_matlab_tuned_refocusing_result_fixture_matches_python_eval",
    "test_matlab_ideal_time_varying_result_fixture_matches_python_eval",
    "test_tuned_excitation_inverse_pipeline_uses_refocusing_neff",
    "test_tuned_refocusing_evaluation_accepts_spa_catalog_pulse",
    "test_untuned_refocusing_evaluation_accepts_spa_catalog_pulse",
    "test_tuned_spa_summary_returns_matlab_style_metrics",
    "test_spa_phase_optimizer_improves_synthetic_objective",
    "test_ideal_time_varying_cpmg_final_returns_expected_shapes",
    "test_matched_cpmg_ir_train_returns_expected_shapes",
    "test_nonmatched_cpmg_ir_train_returns_expected_shapes",
    "test_matched_diffusion_q_stability_boundary",
]

FAST_EXAMPLE_TESTS = [
    "test_examples_run_from_examples_directory",
    "test_plot_examples_expose_cli_without_matplotlib",
]


def load_tests(
    loader: unittest.TestLoader,
    _standard_tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    for name in FAST_FIXTURE_TESTS:
        suite.addTest(OctaveFixtureTests(name))
    for name in FAST_EXAMPLE_TESTS:
        suite.addTest(ExampleSmokeTests(name))
    return suite


if __name__ == "__main__":
    unittest.main()
