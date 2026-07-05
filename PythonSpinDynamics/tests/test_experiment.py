"""Tests for the declarative experiment facade (spin_dynamics.experiment).

The core guarantee is bit-for-bit parity: an ``Experiment`` must reproduce
the direct ``run_*`` workflow call exactly, for every registered CPMG-family
combination.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from spin_dynamics import workflows
from spin_dynamics.experiment import (
    CPMG,
    Acquisition,
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    ExperimentPlanError,
    Hardware,
    Sample,
    SerializationError,
    available_workflows,
    load_run,
)
from spin_dynamics.noise import NoiseSpec

PROBES = ("ideal", "tuned", "untuned", "matched")

_ACQ = Acquisition(numpts=31, maxoffs=8.0, rephase_action="ignore")
_TRAIN_SAMPLE = Sample(t1_seconds=0.05, t2_seconds=0.04)
_IR_TAUS = (0.5e-3, 1.5e-3)


def _assert_results_equal(facade_result, direct_result) -> None:
    assert type(facade_result) is type(direct_result)
    for spec_field in dataclasses.fields(direct_result):
        expected = getattr(direct_result, spec_field.name)
        actual = getattr(facade_result, spec_field.name)
        if isinstance(expected, np.ndarray):
            assert np.array_equal(actual, expected), spec_field.name
        elif isinstance(expected, float):
            assert actual == pytest.approx(expected, rel=0, abs=0), spec_field.name


@pytest.mark.smoke
def test_ideal_cpmg_parity() -> None:
    experiment = Experiment(sequence=CPMG(), acquisition=Acquisition(numpts=31, maxoffs=8.0))
    record = experiment.run()
    direct = workflows.run_ideal_cpmg(numpts=31, maxoffs=8.0)
    _assert_results_equal(record.result, direct)
    assert record.provenance["workflow"] == "run_ideal_cpmg"


@pytest.mark.parametrize("probe", PROBES)
def test_cpmg_parity(probe: str) -> None:
    experiment = Experiment(
        sequence=CPMG(),
        hardware=Hardware(probe=probe),
        acquisition=Acquisition(numpts=31, maxoffs=8.0),
    )
    direct_func = getattr(workflows, f"run_{probe}_cpmg")
    _assert_results_equal(experiment.run().result, direct_func(numpts=31, maxoffs=8.0))


@pytest.mark.parametrize("probe", PROBES)
def test_cpmg_train_parity(probe: str) -> None:
    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=2),
        sample=_TRAIN_SAMPLE,
        hardware=Hardware(probe=probe),
        acquisition=_ACQ,
    )
    direct_func = getattr(workflows, f"run_{probe}_cpmg_train")
    direct = direct_func(
        numpts=31,
        maxoffs=8.0,
        num_echoes=2,
        t1_seconds=0.05,
        t2_seconds=0.04,
        rephase_action="ignore",
    )
    _assert_results_equal(experiment.run().result, direct)


@pytest.mark.parametrize("probe", PROBES)
def test_cpmg_ir_train_parity(probe: str) -> None:
    experiment = Experiment(
        sequence=CPMGIRTrain(num_echoes=2, tauvect=_IR_TAUS),
        hardware=Hardware(probe=probe),
        acquisition=_ACQ,
    )
    direct_func = getattr(workflows, f"run_{probe}_cpmg_ir_train")
    direct = direct_func(
        num_echoes=2,
        tauvect=np.asarray(_IR_TAUS),
        numpts=31,
        maxoffs=8.0,
        rephase_action="ignore",
    )
    _assert_results_equal(experiment.run().result, direct)


@pytest.mark.smoke
def test_noise_parity_with_seed() -> None:
    noise = NoiseSpec(sigma=0.05, seed=42)
    experiment = Experiment(
        sequence=CPMG(),
        acquisition=Acquisition(numpts=31, maxoffs=8.0, noise=noise),
    )
    direct = workflows.run_ideal_cpmg(numpts=31, maxoffs=8.0, noise=noise)
    _assert_results_equal(experiment.run().result, direct)


@pytest.mark.smoke
def test_tuned_q_value_and_mistuning_forwarded() -> None:
    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=2),
        sample=_TRAIN_SAMPLE,
        hardware=Hardware(probe="tuned", q_value=15.0, mistuning_offset=0.5),
        acquisition=_ACQ,
    )
    direct = workflows.run_tuned_cpmg_train(
        numpts=31,
        maxoffs=8.0,
        num_echoes=2,
        t1_seconds=0.05,
        t2_seconds=0.04,
        q_value=15.0,
        mistuning_offset=0.5,
        rephase_action="ignore",
    )
    _assert_results_equal(experiment.run().result, direct)


@pytest.mark.smoke
def test_plan_reports_resolved_workflow() -> None:
    plan = Experiment(sequence=CPMGTrain(), hardware=Hardware(probe="matched")).plan()
    assert plan.ok
    assert plan.workflow == "run_matched_cpmg_train"
    assert "run_matched_cpmg_train" in plan.report()


@pytest.mark.smoke
def test_plan_unknown_probe_is_error() -> None:
    plan = Experiment(sequence=CPMG(), hardware=Hardware(probe="squid")).plan()
    assert not plan.ok
    assert any("unknown probe" in msg for msg in plan.errors)
    with pytest.raises(ExperimentPlanError):
        Experiment(sequence=CPMG(), hardware=Hardware(probe="squid")).run()


@pytest.mark.smoke
def test_plan_warns_on_ignored_fields() -> None:
    experiment = Experiment(
        sequence=CPMG(),
        hardware=Hardware(q_value=25.0),
        sample=Sample(t1_seconds=1.0),
    )
    plan = experiment.plan()
    assert plan.ok
    ignored = "\n".join(plan.warnings)
    assert "hardware.q_value" in ignored
    assert "sample.t1_seconds" in ignored
    with pytest.warns(UserWarning) as caught:
        experiment.run()
    messages = "\n".join(str(w.message) for w in caught)
    assert "hardware.q_value" in messages
    assert "sample.t1_seconds" in messages


@pytest.mark.smoke
def test_plan_sanity_errors() -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=0),
        acquisition=Acquisition(numpts=-1),
    ).plan()
    assert not plan.ok
    joined = "\n".join(plan.errors)
    assert "num_echoes" in joined
    assert "numpts" in joined


@pytest.mark.smoke
def test_execution_kwargs_are_validated() -> None:
    with pytest.raises(TypeError, match="num_workers"):
        Experiment(sequence=CPMG()).run(num_workers=2)
    record = Experiment(
        sequence=CPMGTrain(num_echoes=2), sample=_TRAIN_SAMPLE, acquisition=_ACQ
    ).run(num_workers=1)
    assert record.provenance["execution"] == {"num_workers": 1}


@pytest.mark.smoke
def test_experiment_json_round_trip() -> None:
    experiment = Experiment(
        sequence=CPMGIRTrain(num_echoes=3, tauvect=[1e-3, 2e-3]),
        sample=Sample(t1_seconds=0.01, t2_seconds=0.008),
        hardware=Hardware(probe="untuned", q_value=20.0),
        acquisition=Acquisition(numpts=51, noise=NoiseSpec(sigma=0.1, seed=7)),
    )
    assert Experiment.from_json(experiment.to_json()) == experiment
    assert experiment.sequence.tauvect == (1e-3, 2e-3)


@pytest.mark.smoke
def test_unserializable_spec_raises() -> None:
    noise = NoiseSpec(sigma=0.1, rng=np.random.default_rng(3))
    experiment = Experiment(sequence=CPMG(), acquisition=Acquisition(noise=noise))
    with pytest.raises(SerializationError):
        experiment.to_dict()


@pytest.mark.smoke
def test_save_load_round_trip(tmp_path) -> None:
    noise = NoiseSpec(sigma=0.02, seed=11)
    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=2),
        sample=_TRAIN_SAMPLE,
        acquisition=Acquisition(
            numpts=31, maxoffs=8.0, rephase_action="ignore", noise=noise
        ),
    )
    record = experiment.run()
    path = tmp_path / "run1.npz"
    record.save(str(path))

    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    assert loaded.provenance["workflow"] == "run_ideal_cpmg_train"
    assert np.array_equal(loaded.arrays["mrx"], record.result.mrx)

    reconstructed = loaded.result
    _assert_results_equal(reconstructed, record.result)
    assert reconstructed.noise == record.result.noise
    assert reconstructed.phase_cycle == record.result.phase_cycle

    rerun = loaded.experiment.run()
    assert np.array_equal(rerun.result.mrx, record.result.mrx)


@pytest.mark.smoke
def test_registry_entries_point_at_public_workflows() -> None:
    entries = available_workflows()
    assert len(entries) == 12
    public = set(workflows.STABLE_WORKFLOW_API) | set(workflows.EXTENDED_WORKFLOW_API)
    for entry in entries:
        assert entry.name in public, entry.name
        assert getattr(workflows, entry.name) is entry.func
