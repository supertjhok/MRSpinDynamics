"""Phase-8 integration tests for nano-MR experiment and design facades."""

from __future__ import annotations

import time

import numpy as np
import pytest

from spin_dynamics.design import (
    ExperimentPredictor,
    NanoMRQdyneAdapter,
    NanoMRQdyneDesign,
)
from spin_dynamics.experiment import (
    Experiment,
    ExperimentPlanError,
    Hardware,
    NanoMRBathComponent,
    NanoMRLayer,
    NanoMROpticalReadout,
    NanoMRQdyne,
    NanoMRSensor,
    NanoMRStatisticalSpectrum,
    Sample,
    load_config,
    load_run,
    save_config,
)
from spin_dynamics.experiment.nano_mr_adapter import qdyne_kwargs, statistical_kwargs
from spin_dynamics.nano_mr import simulate_qdyne, simulate_statistical_spectrum


def _statistical_experiment(*, points: int = 257) -> Experiment:
    return Experiment(
        sequence=NanoMRStatisticalSpectrum(num_points=points),
        sample=Sample(nano_mr_layer=NanoMRLayer()),
        hardware=Hardware(nano_mr_sensor=NanoMRSensor()),
    )


def _qdyne_experiment(*, shots: int = 128, optical: bool = False) -> Experiment:
    return Experiment(
        sequence=NanoMRQdyne(
            signal_frequency_hz=2.0e6,
            reference_frequency_hz=1.99e6,
            shot_count=shots,
            sensing_duration_seconds=2.0e-6,
            repetition_interval_seconds=20.0e-6,
            seed=7,
        ),
        hardware=Hardware(
            nano_mr_sensor=NanoMRSensor(depth_nm=6.0),
            nano_mr_optical_readout=(
                NanoMROpticalReadout(bright_count_rate_hz=2.0e6)
                if optical
                else None
            ),
        ),
    )


@pytest.mark.smoke
def test_statistical_facade_plans_runs_and_round_trips_result(tmp_path) -> None:
    experiment = _statistical_experiment()
    plan = experiment.plan(estimate=False)
    assert plan.ok
    assert plan.workflow == "simulate_statistical_spectrum"

    record = experiment.run()
    direct = simulate_statistical_spectrum(**statistical_kwargs(experiment))
    assert record.result.component_psd_t2_s.shape == (1, 257)
    np.testing.assert_array_equal(
        record.result.component_psd_t2_s, direct.component_psd_t2_s
    )
    assert record.result.rms_field_tesla > 0.0
    peak = np.argmax(record.result.component_psd_t2_s[0])
    peak_frequency = record.result.angular_frequencies_rad_s[peak]
    assert abs(peak_frequency) == pytest.approx(
        record.result.larmor_angular_frequencies_rad_s[0], rel=0.02
    )

    path = tmp_path / "statistical.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.result_matches is True
    np.testing.assert_array_equal(
        loaded.result.component_psd_t2_s,
        record.result.component_psd_t2_s,
    )


def test_nano_mr_planner_rejects_missing_physical_inputs() -> None:
    statistical = Experiment(sequence=NanoMRStatisticalSpectrum(num_points=11))
    assert any(
        "nano_mr_sensor" in item
        for item in statistical.plan(estimate=False).errors
    )
    assert any(
        "nano_mr_layer" in item
        for item in statistical.plan(estimate=False).errors
    )
    with pytest.raises(ExperimentPlanError):
        statistical.run()

    qdyne = Experiment(sequence=NanoMRQdyne(shot_count=8))
    assert any(
        "nano_mr_sensor" in item for item in qdyne.plan(estimate=False).errors
    )

    short_cycle = Experiment(
        sequence=NanoMRQdyne(
            shot_count=8,
            sensing_duration_seconds=2.0e-6,
            repetition_interval_seconds=2.1e-6,
        ),
        hardware=Hardware(
            nano_mr_sensor=NanoMRSensor(),
            nano_mr_optical_readout=NanoMROpticalReadout(
                readout_seconds=300.0e-9
            ),
        ),
    )
    assert any(
        "optical cycle" in item
        for item in short_cycle.plan(estimate=False).errors
    )


def test_multi_isotope_layer_round_trips_friendly_toml(tmp_path) -> None:
    layer = NanoMRLayer(
        components=(
            NanoMRBathComponent(isotope="1H", number_density_m3=6.7e28),
            NanoMRBathComponent(
                isotope="19F",
                number_density_m3=3.0e28,
                correlation_time_seconds=50.0e-6,
            ),
        ),
        thickness_nm=8.0,
    )
    experiment = Experiment(
        sequence=NanoMRStatisticalSpectrum(num_points=101),
        sample=Sample(nano_mr_layer=layer),
        hardware=Hardware(nano_mr_sensor=NanoMRSensor(preset="sic_pl6")),
    )
    path = tmp_path / "nano_mr.toml"
    save_config(experiment, path)
    text = path.read_text(encoding="utf-8")
    assert "[[sample.nano_mr_layer.components]]" in text
    loaded = load_config(path)
    assert loaded == experiment
    assert Experiment.from_json(experiment.to_json()) == experiment


@pytest.mark.smoke
def test_qdyne_optical_result_round_trips_in_npz(tmp_path) -> None:
    experiment = _qdyne_experiment(optical=True)
    record = experiment.run()
    direct = simulate_qdyne(**qdyne_kwargs(experiment))
    assert record.result.expected_beat_frequency_hz == pytest.approx(10.0e3)
    np.testing.assert_array_equal(
        record.result.normalized_signal, direct.normalized_signal
    )
    np.testing.assert_array_equal(
        record.result.optical_readout.sampled_counts,
        direct.optical_readout.sampled_counts,
    )
    assert record.result.optical_readout is not None
    assert record.result.optical_readout.sampled_counts.shape == (128,)

    path = tmp_path / "qdyne.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.unsaved_result_fields == ()
    assert loaded.result_matches is True
    np.testing.assert_array_equal(
        loaded.result.optical_readout.sampled_counts,
        record.result.optical_readout.sampled_counts,
    )


def test_qdyne_bayesian_adapter_binds_particle_and_action() -> None:
    template = _qdyne_experiment(shots=64)
    adapter = NanoMRQdyneAdapter(
        template,
        nominal_parameters={
            "signal_frequency_hz": 2.0e6,
            "field_amplitude_tesla": 20.0e-9,
        },
        sample_stride=8,
    )
    action = NanoMRQdyneDesign(
        reference_frequency_hz=1.995e6,
        sensing_duration_seconds=1.5e-6,
    )
    assert adapter.plan(action).ok
    predictor = ExperimentPredictor(adapter)
    predicted = predictor(
        {
            "signal_frequency_hz": np.array([1.999e6, 2.001e6]),
            "field_amplitude_tesla": np.array([10.0e-9, 20.0e-9]),
        },
        action,
    )
    assert predicted.shape == (2, 8)
    assert not np.array_equal(predicted[0], predicted[1])
    assert adapter.physical_seconds(action) == pytest.approx(64 * 20.0e-6)


def test_representative_nano_mr_workloads_stay_linear_and_fast() -> None:
    statistical = _statistical_experiment(points=4001)
    statistical_plan = statistical.plan()
    assert statistical_plan.estimate is not None
    assert statistical_plan.estimate.work_units == 4001
    assert statistical_plan.estimate.memory_bytes < 200_000

    qdyne = _qdyne_experiment(shots=4096)
    qdyne_plan = qdyne.plan()
    assert qdyne_plan.estimate is not None
    assert qdyne_plan.estimate.memory_bytes < 400_000
    started = time.perf_counter()
    qdyne.run()
    elapsed = time.perf_counter() - started
    assert elapsed < 5.0, "loose regression ceiling, not a cross-host benchmark"
