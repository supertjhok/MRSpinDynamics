"""Fast smoke-test suite for routine Python port development.

Run with:
    python -m unittest tests.smoke_tests

The full validation suite remains:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from tests.test_basic_octave_fixtures import OctaveFixtureTests
from tests.test_bloch_siegert import BlochSiegertWorkflowTests
from tests.test_composition import (
    test_compiled_timeline_aligns_channels_and_applies_typed_hardware,
    test_flow_field_interpolates_space_and_time_in_si_units,
)
from tests.test_examples import ExampleSmokeTests
from tests.test_electropermanent import ElectropermanentRodFieldTests
from tests.test_electropermanent_array import (
    ArraySynthesisTests,
    HybridArrayGeometryTests,
)
from tests.test_electropermanent_hysteresis import (
    NeighborCouplingTests,
    PlayHysteresisTests,
)
from tests.test_electropermanent_imaging import (
    NonlinearEPMImagingTests,
    TissuePhantomTests,
)
from tests.test_electropermanent_transport import (
    MagnetophoreticTransportTests,
    ParticlePhysicsTests,
)
from tests.test_electropermanent_pulses import (
    EmpiricalProgrammingTests,
    PulseArchiveRegressionTests,
)
from tests.test_electropermanent_transient import (
    MutualProgrammingCircuitTests,
    TransientProgrammingTests,
)
from tests.test_gradient_coils import CylindricalGradientDesignTests
from tests.test_gradient_shielding import ActivelyShieldedGradientTests
from tests.test_gradient_windings import SyntheticContourTests
from tests.test_hyperpolarization_singlet import SingletStateTests
from tests.test_hyperpolarization_workflows import LongLivedSingletWorkflowTests
from tests.test_phase_cycling import PhaseCyclingTests
from tests.test_sequence_ir import PulseqImportTests, SequenceIRTests
from tests.test_workflow_validation import (
    test_inverse_excitation_validation_recognizes_exact_broadband_cancellation,
)


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
]

# The exhaustive ``--help`` check launches every plotting script and therefore
# grows with the example catalog. Keep it in ``tests.example_tests`` and the
# full suite; the change-aware runner checks only examples modified in an edit.

FAST_SEQUENCE_TESTS = [
    (SequenceIRTests, "test_compile_preserves_concurrent_rf_gradient_and_adc_timing"),
    (SequenceIRTests, "test_motion_adapter_converts_cycles_to_angular_units"),
    (SequenceIRTests, "test_plot_data_uses_shared_time_axis_and_block_metadata"),
    (PulseqImportTests, "test_imports_pulseq_15_rf_adc_and_compiles"),
    (PulseqImportTests, "test_decompresses_run_length_encoded_shape"),
    (
        PhaseCyclingTests,
        "test_arbitrary_sequence_ir_cycle_programs_named_pulses",
    ),
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
    for case, name in FAST_SEQUENCE_TESTS:
        suite.addTest(case(name))
    suite.addTest(
        unittest.FunctionTestCase(
            test_flow_field_interpolates_space_and_time_in_si_units
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            test_compiled_timeline_aligns_channels_and_applies_typed_hardware
        )
    )
    suite.addTest(
        unittest.FunctionTestCase(
            test_inverse_excitation_validation_recognizes_exact_broadband_cancellation
        )
    )
    suite.addTest(
        CylindricalGradientDesignTests(
            "test_numpy_solve_enforces_kcl_and_fits_linear_target"
        )
    )
    suite.addTest(
        ActivelyShieldedGradientTests(
            "test_joint_solve_closes_both_surfaces_and_suppresses_fringe_field"
        )
    )
    suite.addTest(
        SyntheticContourTests(
            "test_periodic_seam_is_stitched_into_closed_cylindrical_loops"
        )
    )
    suite.addTest(
        BlochSiegertWorkflowTests(
            "test_common_phase_is_counter_rotating_and_inverts_larmor"
        )
    )
    suite.addTest(
        SingletStateTests("test_statistical_hydrogen_has_no_deviation_order")
    )
    suite.addTest(
        LongLivedSingletWorkflowTests("test_slic_prepare_store_readout_follows_ts")
    )
    suite.addTest(
        ElectropermanentRodFieldTests(
            "test_published_rod_matches_reported_surface_field_scale"
        )
    )
    suite.addTest(
        HybridArrayGeometryTests(
            "test_reference_hierarchy_has_two_panels_and_72_controls"
        )
    )
    suite.addTest(
        ArraySynthesisTests(
            "test_numpy_bounded_solver_recovers_representable_target"
        )
    )
    suite.addTest(
        TissuePhantomTests(
            "test_simple_phantom_contains_off_center_target_and_relaxation_maps"
        )
    )
    suite.addTest(
        NonlinearEPMImagingTests(
            "test_noisy_tissue_reconstruction_is_reproducible_and_accurate"
        )
    )
    suite.addTest(
        ParticlePhysicsTests(
            "test_langevin_force_recovers_linear_low_field_and_saturates"
        )
    )
    suite.addTest(
        MagnetophoreticTransportTests(
            "test_seeded_transport_is_reproducible_and_captures_target"
        )
    )
    suite.addTest(
        PulseArchiveRegressionTests(
            "test_archived_peak_current_cases_match_configuration_specific_fits"
        )
    )
    suite.addTest(
        EmpiricalProgrammingTests(
            "test_published_protocol_zero_crossing_updates_state_and_history"
        )
    )
    suite.addTest(
        PlayHysteresisTests(
            "test_return_to_nested_reversal_restores_operator_state_exactly"
        )
    )
    suite.addTest(
        NeighborCouplingTests(
            "test_saturated_neighbor_changes_partial_programming_result"
        )
    )
    suite.addTest(
        MutualProgrammingCircuitTests(
            "test_positive_mutual_inductance_opposes_neighbor_current_ramp"
        )
    )
    suite.addTest(
        TransientProgrammingTests(
            "test_transient_crosstalk_disturbs_neighbor_return_point_state"
        )
    )
    return suite


if __name__ == "__main__":
    unittest.main()
