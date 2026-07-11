"""Tests for the declarative experiment facade (spin_dynamics.experiment).

The core guarantee is bit-for-bit parity: an ``Experiment`` must reproduce
the direct ``run_*`` workflow call exactly, for every registered CPMG-family
combination.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from spin_dynamics import workflows
from spin_dynamics.experiment import (
    CPMG,
    Acquisition,
    CPMGImaging,
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    ExperimentPlanError,
    Hardware,
    ImagingPlane,
    Phantom,
    PGSE,
    PGSEWalkers,
    Sample,
    SequenceDomain,
    SequenceIRExecution,
    SerializationError,
    SolenoidCoil,
    TxCoil,
    TransportDomain2D,
    UniformFlow2D,
    available_workflows,
    experiment_fingerprint,
    load_run,
    result_fingerprint,
)
from spin_dynamics.noise import NoiseSpec
from spin_dynamics.motion import (
    initialize_ensemble_from_domain,
    make_motion_field_maps,
)
from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.sequences import (
    ADCEvent,
    GradientWaveform,
    HardwareEffectsPolicy,
    RFPulse,
    SequenceBlock,
    SequenceIR,
    compile_sequence,
    compiled_to_motion_steps,
    parse_pulseq,
    run_motion_sequence,
    serialize_pulseq,
)

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
def test_provenance_fingerprints_are_stable_and_specific(tmp_path) -> None:
    experiment = Experiment(
        sequence=CPMG(), acquisition=Acquisition(numpts=31, maxoffs=8.0)
    )
    first = experiment.run()
    second = experiment.run()
    assert first.provenance["schema_version"] == 2
    assert first.provenance["experiment_sha256"] == experiment_fingerprint(experiment)
    assert first.provenance["result_sha256"] == second.provenance["result_sha256"]
    assert first.provenance["result_sha256"] == result_fingerprint(first.result)
    assert first.provenance["randomness"]["status"] == "deterministic"
    assert first.provenance["implementation"]["module_sha256"]
    assert first.provenance["environment"]["sha256"]
    assert first.provenance["source"]["package_source_sha256"]
    assert experiment_fingerprint(experiment) != experiment_fingerprint(
        dataclasses.replace(
            experiment, acquisition=Acquisition(numpts=33, maxoffs=8.0)
        )
    )

    path = tmp_path / "reproducible.npz"
    first.save(str(path))
    loaded = load_run(str(path))
    assert loaded.format_version == 2
    assert loaded.specification_matches is True
    assert loaded.result_matches is True
    report = loaded.verify_reproduction()
    assert report.matches
    assert report.archive_result_matches is True
    assert report.implementation_matches is True
    assert report.environment_matches is True
    report.require_match()


@pytest.mark.smoke
def test_provenance_classifies_seeded_and_unseeded_randomness() -> None:
    seeded = Experiment(
        sequence=CPMG(),
        acquisition=Acquisition(
            numpts=31, maxoffs=8.0, noise=NoiseSpec(sigma=0.01, seed=7)
        ),
    ).run()
    assert seeded.provenance["randomness"]["status"] == "seeded"
    assert seeded.provenance["randomness"]["sources"][0]["seed"] == 7

    unseeded = Experiment(
        sequence=CPMG(),
        acquisition=Acquisition(numpts=31, maxoffs=8.0, noise=0.01),
    ).run()
    assert unseeded.provenance["randomness"]["status"] == "unseeded"


def test_version_one_run_archive_remains_readable(tmp_path) -> None:
    path = tmp_path / "current.npz"
    Experiment(sequence=CPMG(), acquisition=Acquisition(numpts=15)).run().save(
        str(path)
    )
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        arrays = {
            key: archive[key].copy()
            for key in archive.files
            if key != "__meta__"
        }
    meta["format_version"] = 1
    for key in (
        "schema_version",
        "experiment_sha256",
        "result_sha256",
        "implementation",
        "environment",
        "source",
        "randomness",
    ):
        meta["provenance"].pop(key, None)
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, __meta__=json.dumps(meta), **arrays)
    loaded = load_run(str(legacy))
    assert loaded.format_version == 1
    assert loaded.specification_matches is None
    assert loaded.result_matches is None
    report = loaded.verify_reproduction()
    assert not report.matches
    assert "no result fingerprint" in report.notes[0]


def test_archive_fingerprints_detect_spec_and_result_tampering(tmp_path) -> None:
    original = tmp_path / "original.npz"
    Experiment(sequence=CPMG(), acquisition=Acquisition(numpts=15)).run().save(
        str(original)
    )
    with np.load(original, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        arrays = {
            key: archive[key].copy()
            for key in archive.files
            if key != "__meta__"
        }

    array_key = next(iter(arrays))
    arrays[array_key].flat[0] += 1.0
    changed_result = tmp_path / "changed-result.npz"
    np.savez(changed_result, __meta__=json.dumps(meta), **arrays)
    assert load_run(str(changed_result)).result_matches is False

    with np.load(original, allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in archive.files
            if key != "__meta__"
        }
    meta["experiment"]["acquisition"]["numpts"] = 17
    changed_spec = tmp_path / "changed-spec.npz"
    np.savez(changed_spec, __meta__=json.dumps(meta), **arrays)
    assert load_run(str(changed_spec)).specification_matches is False


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
def test_rephasing_rule_warns_on_coarse_grid() -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=40),
        acquisition=Acquisition(numpts=11, maxoffs=10.0),
    ).plan()
    assert plan.ok  # warn, not error
    rephasing = [f for f in plan.findings if f.rule == "rephasing"]
    assert len(rephasing) == 1
    assert rephasing[0].severity == "warn"
    assert rephasing[0].details["recommended_numpts"] > 11
    assert any("rephas" in w for w in plan.warnings)


@pytest.mark.smoke
def test_rephasing_rule_clean_on_fine_grid() -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=2),
        acquisition=Acquisition(numpts=201, maxoffs=5.0),
    ).plan()
    assert plan.ok
    assert not [f for f in plan.findings if f.rule == "rephasing"]


@pytest.mark.smoke
def test_rephasing_action_raise_blocks_run() -> None:
    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=40),
        acquisition=Acquisition(numpts=11, maxoffs=10.0, rephase_action="raise"),
    )
    plan = experiment.plan()
    assert not plan.ok
    assert any("rephas" in e for e in plan.errors)
    with pytest.raises(ExperimentPlanError):
        experiment.run()


@pytest.mark.smoke
def test_rephasing_action_ignore_skips_rule() -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=40),
        acquisition=Acquisition(numpts=11, maxoffs=10.0, rephase_action="ignore"),
    ).plan()
    assert not [f for f in plan.findings if f.rule == "rephasing"]


@pytest.mark.smoke
def test_rephasing_auto_refine_is_informational() -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=40),
        acquisition=Acquisition(numpts=11, maxoffs=10.0, auto_refine_grid=True),
    ).plan()
    assert plan.ok
    rephasing = [f for f in plan.findings if f.rule == "rephasing"]
    assert len(rephasing) == 1
    assert rephasing[0].severity == "ok"


@pytest.mark.smoke
def test_asymptotic_cpmg_has_no_rephasing_rule() -> None:
    plan = Experiment(sequence=CPMG(), acquisition=Acquisition(numpts=11)).plan()
    assert not [f for f in plan.findings if f.rule == "rephasing"]


@pytest.mark.parametrize(
    ("numpts", "num_echoes", "expect_workflow_warns"),
    [(11, 40, True), (201, 2, False)],
)
def test_plan_rephasing_matches_workflow(
    numpts: int, num_echoes: int, expect_workflow_warns: bool
) -> None:
    """The plan-time verdict must match the workflow's own run-time check."""

    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=num_echoes),
        sample=Sample(t1_seconds=0.05, t2_seconds=0.04),
        acquisition=Acquisition(numpts=numpts, maxoffs=10.0),
    )
    plan_warns = bool([f for f in experiment.plan().findings if f.rule == "rephasing"])
    assert plan_warns == expect_workflow_warns

    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        workflows.run_ideal_cpmg_train(
            numpts=numpts,
            maxoffs=10.0,
            num_echoes=num_echoes,
            t1_seconds=0.05,
            t2_seconds=0.04,
        )
    workflow_warns = any("rephase" in str(w.message).lower() for w in caught)
    assert workflow_warns == expect_workflow_warns
    assert plan_warns == workflow_warns


@pytest.mark.smoke
def test_noise_spec_rule_rejects_time_nonwhite() -> None:
    plan = Experiment(
        sequence=CPMG(),
        acquisition=Acquisition(noise=NoiseSpec(domain="time", model="probe", sigma=0.1)),
    ).plan()
    assert not plan.ok
    assert any("time-domain noise" in e for e in plan.errors)


@pytest.fixture
def pinned_calibration(monkeypatch):
    """Pin the runtime-estimate constants so tests are timing-independent."""

    from spin_dynamics.experiment import estimate as estimate_module

    monkeypatch.setattr(
        estimate_module,
        "_CALIBRATION",
        estimate_module._Calibration(
            overhead_seconds=0.0,
            seconds_per_unit=1e-6,
            backend="pinned",
            manual=True,
        ),
    )


@pytest.mark.smoke
def test_estimate_present_only_for_finite_workflows(pinned_calibration) -> None:
    train_plan = Experiment(sequence=CPMGTrain(num_echoes=4)).plan()
    assert train_plan.estimate is not None
    assert train_plan.estimate.backend == "pinned"
    assert train_plan.estimate.memory_bytes > 0
    assert "estimate:" in train_plan.report()

    ir_plan = Experiment(sequence=CPMGIRTrain(num_echoes=4)).plan()
    assert ir_plan.estimate is not None

    assert Experiment(sequence=CPMG()).plan().estimate is None
    assert Experiment(sequence=CPMGTrain()).plan(estimate=False).estimate is None


@pytest.mark.smoke
def test_estimate_scales_with_work(pinned_calibration) -> None:
    def seconds(num_echoes: int, numpts: int = 101) -> float:
        plan = Experiment(
            sequence=CPMGTrain(num_echoes=num_echoes),
            acquisition=Acquisition(numpts=numpts, rephase_action="ignore"),
        ).plan()
        return plan.estimate.seconds

    # units = 2 * numpts * (2 + 3 * num_echoes) with zero pinned overhead
    assert seconds(16) / seconds(8) == pytest.approx((2 + 3 * 16) / (2 + 3 * 8))
    assert seconds(8, numpts=202) / seconds(8, numpts=101) == pytest.approx(2.0)

    def ir_seconds(num_taus: int) -> float:
        plan = Experiment(
            sequence=CPMGIRTrain(num_echoes=4, tauvect=tuple([1e-3] * num_taus)),
            acquisition=Acquisition(rephase_action="ignore"),
        ).plan()
        return plan.estimate.seconds

    assert ir_seconds(6) / ir_seconds(3) == pytest.approx(2.0)


@pytest.mark.smoke
def test_estimate_memory_scales_with_numpts(pinned_calibration) -> None:
    def memory(numpts: int) -> int:
        plan = Experiment(
            sequence=CPMGTrain(num_echoes=4),
            acquisition=Acquisition(numpts=numpts, rephase_action="ignore"),
        ).plan()
        return plan.estimate.memory_bytes

    assert memory(2002) == pytest.approx(2 * memory(1001), rel=1e-6)


@pytest.mark.smoke
def test_probe_estimate_carries_overhead_note(pinned_calibration) -> None:
    plan = Experiment(
        sequence=CPMGTrain(num_echoes=4), hardware=Hardware(probe="tuned")
    ).plan()
    assert any("probe-circuit" in note for note in plan.estimate.notes)


@pytest.mark.smoke
def test_run_skips_estimate(pinned_calibration) -> None:
    record = Experiment(
        sequence=CPMGTrain(num_echoes=2), sample=_TRAIN_SAMPLE, acquisition=_ACQ
    ).run()
    assert record.plan.estimate is None


def test_estimate_accuracy_against_real_run() -> None:
    """Advisory accuracy: the estimate must be the right order of magnitude.

    Uses a real calibration and compares against a warm (second) run to avoid
    cold-start outliers; the window is deliberately wide because CI hosts are
    noisy.
    """

    import time

    from spin_dynamics.experiment import calibrate

    calibrate(force=True)
    experiment = Experiment(
        sequence=CPMGTrain(num_echoes=8),
        sample=Sample(t1_seconds=0.05, t2_seconds=0.04),
        acquisition=Acquisition(numpts=2001, rephase_action="ignore"),
    )
    predicted = experiment.plan().estimate.seconds
    experiment.run()  # warm caches
    start = time.perf_counter()
    experiment.run()
    actual = time.perf_counter() - start
    assert predicted > 0
    assert 1 / 20 < actual / predicted < 20


def _disc_phantom(n: int = 8) -> Phantom:
    x = np.linspace(-1.0, 1.0, n)
    rho = ((x[:, None] ** 2 + x[None, :] ** 2) < 0.8).astype(np.float64)
    return Phantom(rho=rho)


_XCOIL = TxCoil(
    geometry=SolenoidCoil(radius_m=0.015, length_m=0.03, turns=10, axis="x")
)
_ZCOIL = TxCoil(
    geometry=SolenoidCoil(radius_m=0.015, length_m=0.03, turns=10, axis="z")
)


@pytest.mark.parametrize("probe", ("ideal", "tuned", "matched"))
def test_imaging_parity(probe: str) -> None:
    phantom = _disc_phantom()
    experiment = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=phantom, t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(probe=probe),
    )
    direct_func = getattr(workflows, f"run_{probe}_cpmg_imaging")
    direct = direct_func(
        phantom.rho,
        t1_map=5e-3 * np.ones_like(phantom.rho),
        t2_map=5e-3 * np.ones_like(phantom.rho),
        ny=5,
    )
    _assert_results_equal(experiment.run().result, direct)


@pytest.mark.smoke
def test_imaging_requires_phantom() -> None:
    plan = Experiment(sequence=CPMGImaging()).plan()
    assert not plan.ok
    assert any("sample.phantom" in e for e in plan.errors)


@pytest.mark.smoke
def test_imaging_relaxation_ambiguity_is_error() -> None:
    phantom = Phantom(
        rho=np.ones((4, 4)), t1_map=1e-3 * np.ones((4, 4))
    )
    plan = Experiment(
        sequence=CPMGImaging(), sample=Sample(phantom=phantom, t1_seconds=2e-3)
    ).plan()
    assert not plan.ok
    assert any("t1_map" in e for e in plan.errors)


@pytest.mark.smoke
def test_tx_coil_replaces_synthetic_b1() -> None:
    phantom = _disc_phantom()
    base = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=phantom, t1_seconds=5e-3, t2_seconds=5e-3),
    )
    wired = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=phantom, t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(tx_coil=_XCOIL, plane=ImagingPlane()),
    )
    plan = wired.plan()
    assert plan.ok
    wiring_findings = [f for f in plan.findings if f.rule == "hardware_wiring"]
    assert wiring_findings and wiring_findings[0].severity == "ok"
    assert wiring_findings[0].details["tx_transverse_fraction"] > 0.9

    base_result = base.run().result
    wired_result = wired.run().result
    assert not np.allclose(wired_result.b1_tx_map, base_result.b1_tx_map)
    # rho-weighted mean normalization: nominal flip calibrated at the sample
    weights = np.abs(phantom.rho)
    mean_b1 = float(
        np.sum(wired_result.b1_tx_map * weights) / np.sum(weights)
    )
    assert mean_b1 == pytest.approx(1.0)
    # rx defaults to tx when no rx coil is given
    assert np.array_equal(wired_result.b1_rx_map, wired_result.b1_tx_map)


@pytest.mark.smoke
def test_axial_coil_warns_low_transverse_fraction() -> None:
    plan = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=_disc_phantom()),
        hardware=Hardware(tx_coil=_ZCOIL, plane=ImagingPlane()),
    ).plan()
    assert plan.ok  # warns, does not block
    assert any("transverse to B0" in w for w in plan.warnings)


@pytest.mark.smoke
def test_field_solve_is_cached(monkeypatch) -> None:
    from spin_dynamics.experiment import wiring

    calls = {"n": 0}
    real = wiring.biot_savart

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(wiring, "biot_savart", counting)
    monkeypatch.setattr(wiring, "_SOLVE_CACHE", {})

    experiment = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=_disc_phantom(), t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(tx_coil=_XCOIL, plane=ImagingPlane()),
    )
    experiment.plan()  # solves + caches
    first = calls["n"]
    assert first == 1
    experiment.plan()
    experiment.run()
    assert calls["n"] == first  # all subsequent uses hit the cache


@pytest.mark.smoke
def test_imaging_experiment_json_round_trip() -> None:
    experiment = Experiment(
        sequence=CPMGImaging(num_echoes=3, ny=7),
        sample=Sample(phantom=_disc_phantom(6), t1_seconds=4e-3, t2_seconds=3e-3),
        hardware=Hardware(tx_coil=_XCOIL, plane=ImagingPlane(extent_m=(0.03, 0.02))),
    )
    assert Experiment.from_json(experiment.to_json()) == experiment


@pytest.mark.smoke
def test_imaging_save_load_round_trip(tmp_path) -> None:
    experiment = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=_disc_phantom(), t1_seconds=5e-3, t2_seconds=5e-3),
        hardware=Hardware(tx_coil=_XCOIL, plane=ImagingPlane()),
    )
    record = experiment.run()
    path = tmp_path / "imaging.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    _assert_results_equal(loaded.result, record.result)
    rerun = loaded.experiment.run()
    assert np.array_equal(rerun.result.kspace, record.result.kspace)


@pytest.mark.smoke
def test_imaging_execution_kwargs() -> None:
    experiment = Experiment(
        sequence=CPMGImaging(ny=5),
        sample=Sample(phantom=_disc_phantom(), t1_seconds=5e-3, t2_seconds=5e-3),
    )
    record = experiment.run(num_workers=1, phase_workers=1)
    assert record.provenance["execution"] == {"num_workers": 1, "phase_workers": 1}
    with pytest.raises(TypeError, match="tau_workers"):
        experiment.run(tau_workers=2)


@pytest.mark.smoke
def test_pgse_moment_parity_and_plan() -> None:
    spec = PGSE(
        num_echoes=3,
        gradient_amplitude=0.035,
        gradient_duration=1.5e-3,
        diffusion_time=18e-3,
        first_echo_time_seconds=40e-3,
        echo_spacing_seconds=12e-3,
    )
    sample = Sample(t2_seconds=0.12, diffusion_coefficient=1.4e-9)
    experiment = Experiment(sequence=spec, sample=sample)

    plan = experiment.plan()
    assert plan.ok
    assert plan.workflow == "run_pgse_moment"
    assert plan.estimate is not None

    record = experiment.run()
    direct = workflows.run_pgse_moment(
        num_echoes=3,
        gradient_amplitude=0.035,
        gradient_duration=1.5e-3,
        diffusion_time=18e-3,
        diffusion_coefficient=1.4e-9,
        t2_seconds=0.12,
        first_echo_time_seconds=40e-3,
        echo_spacing_seconds=12e-3,
    )
    _assert_results_equal(record.result, direct)
    assert record.provenance["workflow"] == "run_pgse_moment"


@pytest.mark.smoke
def test_pgse_json_config_and_result_round_trip(tmp_path) -> None:
    experiment = Experiment(
        sequence=PGSE(gradient_amplitude=0.02, diffusion_time=15e-3),
        sample=Sample(diffusion_coefficient=2.0e-9, t2_seconds=0.08),
    )
    assert Experiment.from_json(experiment.to_json()) == experiment

    record = experiment.run()
    path = tmp_path / "pgse.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    _assert_results_equal(loaded.result, record.result)


@pytest.mark.smoke
def test_pgse_plan_rejects_invalid_diffusion_timing() -> None:
    plan = Experiment(
        sequence=PGSE(gradient_duration=3e-3, diffusion_time=2e-3),
        sample=Sample(diffusion_coefficient=-1e-9),
    ).plan()
    assert not plan.ok
    errors = "\n".join(plan.errors)
    assert "diffusion_time" in errors
    assert "diffusion_coefficient" in errors


def _transport_domain() -> TransportDomain2D:
    return TransportDomain2D(
        rho=np.array([[1.0, 0.5], [1.0, 1.0], [0.5, 1.0]]),
        x_axis=np.array([-1e-3, 0.0, 1e-3]),
        z_axis=np.array([-0.5e-3, 0.5e-3]),
    )


@pytest.mark.smoke
def test_pgse_walkers_with_flow_matches_direct_workflow() -> None:
    spec = PGSEWalkers(
        num_echoes=2,
        gradient_amplitude=0.025,
        gradient_duration=1e-3,
        diffusion_time=10e-3,
        walkers_per_cell=8,
        seed=17,
        jitter=True,
        echo_spacing_seconds=24e-3,
        boundary="periodic",
        substeps_per_interval=2,
    )
    sample = Sample(
        diffusion_coefficient=1.2e-9,
        t1_seconds=0.5,
        t2_seconds=0.15,
        transport_domain=_transport_domain(),
        flow=UniformFlow2D((2e-3, -0.5e-3)),
    )
    experiment = Experiment(sequence=spec, sample=sample)

    plan = experiment.plan()
    assert plan.ok
    assert plan.workflow == "run_pgse_walkers"
    assert plan.estimate is not None
    transport = [finding for finding in plan.findings if finding.rule == "transport"]
    assert len(transport) == 1
    assert transport[0].severity == "ok"
    assert transport[0].details["boundary"] == "periodic"

    record = experiment.run()
    domain = sample.transport_domain
    direct = workflows.run_pgse_walkers(
        rho=domain.rho,
        x_axis=domain.x_axis,
        z_axis=domain.z_axis,
        num_echoes=2,
        gradient_amplitude=0.025,
        gradient_duration=1e-3,
        diffusion_time=10e-3,
        diffusion_coefficient=1.2e-9,
        walkers_per_cell=8,
        seed=17,
        jitter=True,
        echo_spacing_seconds=24e-3,
        t1_seconds=0.5,
        t2_seconds=0.15,
        velocity=np.array([2e-3, -0.5e-3]),
        boundary="periodic",
        substeps_per_interval=2,
    )
    _assert_results_equal(record.result, direct)
    assert np.array_equal(
        record.result.sequence.final_ensemble.positions,
        direct.sequence.final_ensemble.positions,
    )


@pytest.mark.smoke
def test_pgse_walkers_config_json_and_primary_result_round_trip(tmp_path) -> None:
    experiment = Experiment(
        sequence=PGSEWalkers(
            walkers_per_cell=4,
            seed=3,
            boundary="periodic",
            substeps_per_interval=2,
        ),
        sample=Sample(
            diffusion_coefficient=0.0,
            transport_domain=_transport_domain(),
            flow=UniformFlow2D((1e-3, 0.0)),
        ),
    )
    assert Experiment.from_json(experiment.to_json()) == experiment

    record = experiment.run()
    path = tmp_path / "pgse_walkers.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    assert loaded.unsaved_result_fields == ("sequence", "initial_ensemble")
    assert np.array_equal(loaded.result.signal, record.result.signal)
    assert np.array_equal(loaded.result.echo_times, record.result.echo_times)


@pytest.mark.smoke
def test_pgse_walkers_plan_requires_domain_and_valid_boundary() -> None:
    plan = Experiment(
        sequence=PGSEWalkers(boundary="open", seed=-1),
    ).plan()
    assert not plan.ok
    errors = "\n".join(plan.errors)
    assert "transport_domain" in errors
    assert "boundary" in errors
    assert "seed" in errors


@pytest.mark.smoke
def test_pgse_walkers_warns_when_flow_reflects() -> None:
    plan = Experiment(
        sequence=PGSEWalkers(walkers_per_cell=2, seed=1),
        sample=Sample(
            transport_domain=_transport_domain(),
            flow=UniformFlow2D((1e-3, 0.0)),
        ),
    ).plan()
    assert plan.ok
    assert any("not through-flow" in warning for warning in plan.warnings)


from spin_dynamics.experiment import (  # noqa: E402
    NQRFID,
    NQRPopulationTransfer,
    NQRSLSE,
    NQRSORC,
)
from spin_dynamics.nqr import (  # noqa: E402
    QuadrupolarSite,
    SelectivePulse,
    SLSESequence,
    SORCSequence,
    diagonalize_site,
    simulate_full_slse,
    simulate_full_fid,
    simulate_population_transfer,
    simulate_slse,
    simulate_sorc,
)

_SPIN1 = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
_SPIN32 = QuadrupolarSite(spin=1.5, isotope="35Cl", quadrupole_frequency_hz=30e6, eta=0.1)
# Soft selective 90-degree pulse: nutation * duration = 0.25
_SOFT_SLSE = NQRSLSE(
    pulse_duration_seconds=100e-6,
    nutation_hz=2.5e3,
    echo_spacing_seconds=1e-3,
    num_echoes=4,
    orientations="single",
)


@pytest.mark.smoke
def test_nqr_auto_dispatch() -> None:
    plan1 = Experiment(sequence=_SOFT_SLSE, sample=Sample(site=_SPIN1)).plan()
    assert plan1.ok
    assert plan1.workflow == "simulate_slse"

    hard = NQRSLSE(
        pulse_duration_seconds=10e-6,
        nutation_hz=25e3,
        echo_spacing_seconds=0.5e-3,
        num_echoes=3,
        orientations="single",
    )
    plan2 = Experiment(sequence=hard, sample=Sample(site=_SPIN32)).plan()
    assert plan2.ok
    assert plan2.workflow == "simulate_full_slse"
    selection = [f for f in plan2.findings if f.rule == "nqr_model"][0]
    assert selection.details["recommended_model"] == "full"


def test_nqr_slse_reduced_parity() -> None:
    record = Experiment(sequence=_SOFT_SLSE, sample=Sample(site=_SPIN1)).run()
    assert record.provenance["workflow"] == "simulate_slse"
    direct = simulate_slse(
        _SPIN1,
        SLSESequence(
            detection=SelectivePulse(
                transition_label="x",
                duration_seconds=100e-6,
                nutation_hz=2.5e3,
            ),
            echo_spacing_seconds=1e-3,
            num_echoes=4,
        ),
        orientations="single",
    )
    assert np.array_equal(record.result.echo_amplitudes, direct.echo_amplitudes)
    assert record.result.transition.label == "x"


def test_nqr_slse_full_parity_with_bare_nutation() -> None:
    """The facade must convert effective -> bare nutation for the full engine."""

    spec = NQRSLSE(
        pulse_duration_seconds=10e-6,
        nutation_hz=25e3,
        echo_spacing_seconds=0.5e-3,
        num_echoes=3,
        transition="x",
        orientations="single",
    )
    record = Experiment(sequence=spec, sample=Sample(site=_SPIN32)).run()
    assert record.provenance["workflow"] == "simulate_full_slse"

    transition = diagonalize_site(_SPIN32).transition("x")
    bare = 25e3 / (2.0 * transition.strength)
    direct = simulate_full_slse(
        _SPIN32,
        nutation_hz=bare,
        excitation_duration_seconds=10e-6,
        refocus_duration_seconds=10e-6,
        echo_spacing_seconds=0.5e-3,
        num_echoes=3,
        rf_frequency_hz=transition.frequency_hz,
        excitation_phase=0.0,
        refocus_phase=np.pi / 2.0,
        orientations="single",
    )
    assert np.array_equal(record.result.echo_amplitudes, direct.echo_amplitudes)


def test_nqr_sorc_parity() -> None:
    spec = NQRSORC(
        pulse_duration_seconds=100e-6,
        nutation_hz=2.5e3,
        half_spacing_seconds=0.5e-3,
        num_pulses=6,
        orientations="single",
    )
    record = Experiment(sequence=spec, sample=Sample(site=_SPIN1)).run()
    direct = simulate_sorc(
        _SPIN1,
        SORCSequence(
            detection=SelectivePulse(
                transition_label="x",
                duration_seconds=100e-6,
                nutation_hz=2.5e3,
            ),
            half_spacing_seconds=0.5e-3,
            num_pulses=6,
        ),
        orientations="single",
    )
    assert np.array_equal(record.result.signal_amplitudes, direct.signal_amplitudes)


@pytest.mark.smoke
def test_nqr_fid_parity() -> None:
    spec = NQRFID(
        nutation_hz=10e3,
        pulse_duration_seconds=25e-6,
        acquisition_seconds=100e-6,
        num_points=32,
    )
    record = Experiment(sequence=spec, sample=Sample(site=_SPIN1)).run()
    direct = simulate_full_fid(
        _SPIN1,
        nutation_hz=10e3,
        pulse_duration_seconds=25e-6,
        times_seconds=np.linspace(0.0, 100e-6, 32),
    )
    assert np.array_equal(record.result.signal, direct.signal)


@pytest.mark.smoke
def test_nqr_population_transfer_parity_and_spin_guard() -> None:
    spec = NQRPopulationTransfer(
        perturbation_duration_seconds=100e-6,
        perturbation_nutation_hz=2.5e3,
        detection_duration_seconds=100e-6,
        detection_nutation_hz=2.5e3,
        echo_spacing_seconds=1e-3,
        num_echoes=3,
        perturbation_transition="x",
        detection_transition="y",
        orientations="single",
    )
    record = Experiment(sequence=spec, sample=Sample(site=_SPIN1)).run()
    direct = simulate_population_transfer(
        _SPIN1,
        SelectivePulse("x", 100e-6, 2.5e3),
        SLSESequence(SelectivePulse("y", 100e-6, 2.5e3), 1e-3, 3),
        orientations="single",
    )
    assert np.array_equal(record.result.normalized_difference, direct.normalized_difference)
    plan = Experiment(sequence=spec, sample=Sample(site=_SPIN32)).plan()
    assert not plan.ok
    assert any("spin-1" in error for error in plan.errors)


@pytest.mark.smoke
def test_nqr_bare_nutation_conversion() -> None:
    from spin_dynamics.experiment import nqr_adapter

    for site, label in ((_SPIN1, "x"), (_SPIN32, "x")):
        strength = diagonalize_site(site).transition(label).strength
        assert nqr_adapter.bare_nutation_hz(site, label, 10e3) == pytest.approx(
            10e3 / (2.0 * strength)
        )


@pytest.mark.smoke
def test_nqr_forced_model_override_warns() -> None:
    forced = Experiment(
        sequence=dataclasses.replace(_SOFT_SLSE, model="full"),
        sample=Sample(site=_SPIN1),
    )
    plan = forced.plan()
    assert plan.ok
    assert plan.workflow == "simulate_full_slse"
    assert any("overrides" in w for w in plan.warnings)


@pytest.mark.smoke
def test_nqr_forced_reduced_on_higher_spin_errors() -> None:
    plan = Experiment(
        sequence=dataclasses.replace(_SOFT_SLSE, model="reduced"),
        sample=Sample(site=_SPIN32),
    ).plan()
    assert not plan.ok
    assert any("spin-1 only" in e for e in plan.errors)


@pytest.mark.smoke
def test_nqr_requires_site() -> None:
    plan = Experiment(sequence=_SOFT_SLSE).plan()
    assert not plan.ok
    assert any("sample.site" in e for e in plan.errors)


@pytest.mark.smoke
def test_nqr_json_and_save_round_trip(tmp_path) -> None:
    experiment = Experiment(sequence=_SOFT_SLSE, sample=Sample(site=_SPIN1))
    assert Experiment.from_json(experiment.to_json()) == experiment

    record = experiment.run()
    path = tmp_path / "nqr.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    assert loaded.unsaved_result_fields == ()
    assert np.array_equal(
        loaded.result.echo_amplitudes, record.result.echo_amplitudes
    )
    rerun = loaded.experiment.run()
    assert np.array_equal(
        rerun.result.echo_amplitudes, record.result.echo_amplitudes
    )


from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    HyperfineCoupling,
    davies_endor_spectrum,
    hyscore_signal,
    hyscore_spectrum,
    mims_endor_spectrum,
    simulate_deer,
    simulate_fid,
    simulate_field_sweep,
    simulate_hahn_echo,
    three_pulse_eseem,
    two_pulse_eseem,
)
from spin_dynamics.experiment import (  # noqa: E402
    DEERDistribution,
    ESRCWSweep,
    ESRDEER,
    ESRDaviesENDOR,
    ESRFID,
    ESRHYSCORE,
    ESRHahnEcho,
    ESRMimsENDOR,
    ESRThreePulseESEEM,
    ESRTwoPulseESEEM,
    UniformB0,
)

_ESR_SYSTEM = ESRSpinSystem()
_ESR_HW = Hardware(b0=UniformB0(field_tesla=0.35))
_ESR_FID = ESRFID(
    nutation_hz=25e6,
    pulse_duration_seconds=10e-9,
    acquisition_seconds=200e-9,
    num_points=128,
)


@pytest.mark.smoke
def test_esr_fid_parity() -> None:
    record = Experiment(
        sequence=_ESR_FID, sample=Sample(esr_system=_ESR_SYSTEM), hardware=_ESR_HW
    ).run()
    assert record.provenance["workflow"] == "simulate_fid"
    direct = simulate_fid(
        _ESR_SYSTEM,
        (0.0, 0.0, 0.35),
        nutation_hz=25e6,
        pulse_duration_seconds=10e-9,
        times_seconds=np.linspace(0.0, 200e-9, 128),
    )
    assert np.array_equal(record.result.signal, direct.signal)
    assert record.result.rf_frequency_hz == direct.rf_frequency_hz


@pytest.mark.smoke
def test_esr_hahn_echo_parity_with_defaults() -> None:
    """Default refocus = 2x excitation and window = 2*tau must match a
    hand-built direct call."""

    spec = ESRHahnEcho(
        nutation_hz=25e6,
        excitation_duration_seconds=10e-9,
        tau_seconds=200e-9,
        num_points=128,
    )
    record = Experiment(
        sequence=spec, sample=Sample(esr_system=_ESR_SYSTEM), hardware=_ESR_HW
    ).run()
    assert record.provenance["workflow"] == "simulate_hahn_echo"
    direct = simulate_hahn_echo(
        _ESR_SYSTEM,
        (0.0, 0.0, 0.35),
        nutation_hz=25e6,
        excitation_duration_seconds=10e-9,
        refocus_duration_seconds=20e-9,
        tau_seconds=200e-9,
        times_seconds=np.linspace(0.0, 400e-9, 128),
    )
    assert np.array_equal(record.result.signal, direct.signal)
    assert record.result.echo_center_seconds == direct.echo_center_seconds


@pytest.mark.smoke
def test_esr_cw_sweep_parity_without_fixed_b0() -> None:
    spec = ESRCWSweep(
        microwave_frequency_hz=9.5e9,
        orientations="single",
        num_points=64,
    )
    record = Experiment(sequence=spec, sample=Sample(esr_system=_ESR_SYSTEM)).run()
    direct = simulate_field_sweep(
        _ESR_SYSTEM, 9.5e9, orientations="single", points=64
    )
    assert np.array_equal(record.result.spectrum, direct.spectrum)


@pytest.mark.smoke
def test_esr_deer_parity_and_round_trip(tmp_path) -> None:
    distances = np.linspace(2.0, 4.0, 9)
    weights = np.exp(-0.5 * ((distances - 3.0) / 0.25) ** 2)
    sample = Sample(deer_distribution=DEERDistribution(distances, weights))
    spec = ESRDEER(acquisition_seconds=2e-6, num_points=32, n_theta=101)
    experiment = Experiment(sequence=spec, sample=sample)
    assert Experiment.from_json(experiment.to_json()) == experiment
    record = experiment.run()
    direct = simulate_deer(
        np.linspace(0.0, 2e-6, 32),
        distances,
        weights,
        n_theta=101,
    )
    assert np.array_equal(record.result.form_factor, direct)
    path = tmp_path / "deer.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.unsaved_result_fields == ()
    assert loaded.experiment == experiment


_HYPERFINE = HyperfineCoupling(
    larmor_hz=14.8e6, secular_hz=3.0e6, pseudosecular_hz=1.2e6
)


@pytest.mark.smoke
def test_esr_two_and_three_pulse_eseem_parity() -> None:
    sample = Sample(hyperfine_coupling=_HYPERFINE)
    two = ESRTwoPulseESEEM(
        acquisition_seconds=2e-6, num_points=24, zero_fill=2
    )
    two_record = Experiment(sequence=two, sample=sample).run()
    times = np.linspace(0.0, 2e-6, 24)
    assert two_record.result.model == "analytic"
    assert np.array_equal(
        two_record.result.signal, two_pulse_eseem(times, _HYPERFINE)
    )

    three = ESRThreePulseESEEM(
        acquisition_seconds=2e-6,
        tau_seconds=120e-9,
        num_points=24,
        zero_fill=2,
    )
    three_record = Experiment(sequence=three, sample=sample).run()
    assert np.array_equal(
        three_record.result.signal,
        three_pulse_eseem(times, _HYPERFINE, tau_seconds=120e-9),
    )


@pytest.mark.smoke
def test_esr_hyscore_parity_and_round_trip(tmp_path) -> None:
    spec = ESRHYSCORE(
        evolution1_seconds=1e-6,
        evolution2_seconds=1.2e-6,
        tau_seconds=100e-9,
        num_points1=5,
        num_points2=6,
        zero_fill=1,
    )
    experiment = Experiment(
        sequence=spec, sample=Sample(hyperfine_coupling=_HYPERFINE)
    )
    record = experiment.run()
    t1 = np.linspace(0.0, 1e-6, 5)
    t2 = np.linspace(0.0, 1.2e-6, 6)
    signal = hyscore_signal(t1, t2, _HYPERFINE, tau_seconds=100e-9)
    direct = hyscore_spectrum(t1, t2, signal, zero_fill=1)
    assert np.array_equal(record.result.signal, signal)
    assert np.array_equal(record.result.spectrum, direct.spectrum)
    path = tmp_path / "hyscore.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.unsaved_result_fields == ()
    assert loaded.experiment == experiment
    assert np.array_equal(loaded.result.spectrum, record.result.spectrum)


@pytest.mark.smoke
def test_esr_endor_parity_and_hyperfine_input_guard() -> None:
    sample = Sample(hyperfine_coupling=_HYPERFINE)
    davies = ESRDaviesENDOR(
        num_points=32,
        linewidth_hz=0.1e6,
        frequency_min_hz=10e6,
        frequency_max_hz=20e6,
    )
    record = Experiment(sequence=davies, sample=sample).run()
    axis = np.linspace(10e6, 20e6, 32)
    direct = davies_endor_spectrum(axis, _HYPERFINE, linewidth_hz=0.1e6)
    assert np.array_equal(record.result.spectrum, direct.spectrum)

    mims = ESRMimsENDOR(
        tau_seconds=200e-9,
        num_points=32,
        linewidth_hz=0.1e6,
        frequency_min_hz=10e6,
        frequency_max_hz=20e6,
    )
    mims_record = Experiment(sequence=mims, sample=sample).run()
    mims_direct = mims_endor_spectrum(
        axis, _HYPERFINE, tau_seconds=200e-9, linewidth_hz=0.1e6
    )
    assert np.array_equal(mims_record.result.spectrum, mims_direct.spectrum)
    plan = Experiment(sequence=davies).plan()
    assert not plan.ok
    assert any("hyperfine_coupling" in error for error in plan.errors)


@pytest.mark.smoke
def test_esr_requires_system_and_field() -> None:
    plan = Experiment(sequence=_ESR_FID, hardware=_ESR_HW).plan()
    assert not plan.ok
    assert any("esr_system" in e for e in plan.errors)

    plan2 = Experiment(sequence=_ESR_FID, sample=Sample(esr_system=_ESR_SYSTEM)).plan()
    assert not plan2.ok
    assert any("field_tesla" in e for e in plan2.errors)


@pytest.mark.smoke
def test_esr_spin_system_equality_is_array_safe() -> None:
    assert ESRSpinSystem() == ESRSpinSystem()
    assert ESRSpinSystem(g_tensor=(2.0, 2.0, 2.2)) == ESRSpinSystem(
        g_tensor=(2.0, 2.0, 2.2)
    )
    assert ESRSpinSystem(g_tensor=2.0) != ESRSpinSystem(g_tensor=2.1)
    assert ESRSpinSystem() != "not a system"


@pytest.mark.smoke
def test_esr_json_and_save_round_trip(tmp_path) -> None:
    experiment = Experiment(
        sequence=_ESR_FID,
        sample=Sample(esr_system=ESRSpinSystem(g_tensor=(2.0, 2.0, 2.2))),
        hardware=_ESR_HW,
    )
    assert Experiment.from_json(experiment.to_json()) == experiment

    record = experiment.run()
    path = tmp_path / "esr.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.experiment == experiment
    assert np.array_equal(loaded.result.signal, record.result.signal)
    rerun = loaded.experiment.run()
    assert np.array_equal(rerun.result.signal, record.result.signal)


def _general_ir(*, hardware_effects=None) -> SequenceIR:
    return SequenceIR(
        blocks=(
            SequenceBlock(
                duration_seconds=1e-3,
                rf=RFPulse([250.0j], dwell_seconds=1e-3),
                label="excitation",
            ),
            SequenceBlock(
                duration_seconds=2e-3,
                adc=ADCEvent(
                    2,
                    dwell_seconds=1e-3,
                    phase_offset_rad=0.2,
                    phase_offsets_rad=[0.0, np.pi / 2.0],
                ),
                label="readout",
            ),
        ),
        hardware_effects=(
            HardwareEffectsPolicy()
            if hardware_effects is None
            else hardware_effects
        ),
    )


def _general_domain() -> SequenceDomain:
    return SequenceDomain(
        axes=(np.array([-1e-3, 1e-3]),),
        density=np.array([0.4, 0.6]),
    )


@pytest.mark.smoke
def test_sequence_ir_facade_matches_direct_motion_backend() -> None:
    ir = _general_ir()
    experiment = Experiment(
        sequence=SequenceIRExecution(ir=ir),
        sample=Sample(sequence_domain=_general_domain()),
    )
    plan = experiment.plan()
    assert plan.ok
    assert plan.workflow == "run_sequence_ir"
    finding = next(item for item in plan.findings if item.rule == "sequence_ir")
    assert finding.details["adc_samples"] == 2

    record = experiment.run()
    compiled = compile_sequence(ir)
    domain = SpatialDomain(_general_domain().axes)
    ensemble = initialize_ensemble_from_domain(
        domain, _general_domain().density
    )
    fields = make_motion_field_maps(domain)
    direct = run_motion_sequence(
        ensemble,
        fields,
        compiled_to_motion_steps(compiled, spatial_dimensions=1),
    )
    expected = direct.signal * np.exp(-1j * compiled.adc.phase_offsets_rad)
    assert np.array_equal(record.result.clean_signal, expected)
    assert np.array_equal(record.result.signal, expected)
    assert np.array_equal(
        record.result.sample_times_seconds, compiled.adc.times_seconds
    )
    assert all(label.startswith("readout:") for label in record.result.sample_labels)


@pytest.mark.smoke
def test_sequence_ir_gradient_channel_mapping_and_adc_frequency_phase() -> None:
    ir = SequenceIR(
        blocks=(
            SequenceBlock(
                duration_seconds=1e-3,
                gradients=(
                    None,
                    None,
                    GradientWaveform([7.0], dwell_seconds=1e-3),
                ),
                adc=ADCEvent(
                    1,
                    dwell_seconds=1e-3,
                    frequency_offset_hz=100.0,
                    phase_offset_rad=0.2,
                ),
            ),
        )
    )
    compiled = compile_sequence(ir)
    steps = compiled_to_motion_steps(
        compiled, spatial_dimensions=2, gradient_axes=(0, 2)
    )
    assert steps[0].gradient == pytest.approx((0.0, 2.0 * np.pi * 7.0))
    assert compiled.adc.phase_offsets_rad[0] == pytest.approx(
        0.2 + 2.0 * np.pi * 100.0 * 0.5e-3
    )


@pytest.mark.smoke
def test_sequence_ir_facade_round_trip_noise_and_reproduction(tmp_path) -> None:
    experiment = Experiment(
        sequence=SequenceIRExecution(ir=_general_ir(), seed=19),
        sample=Sample(sequence_domain=_general_domain()),
        acquisition=Acquisition(noise=NoiseSpec(sigma=0.01, seed=23)),
    )
    assert Experiment.from_json(experiment.to_json()) == experiment
    record = experiment.run()
    assert not np.array_equal(record.result.signal, record.result.clean_signal)
    path = tmp_path / "sequence-ir.npz"
    record.save(str(path))
    loaded = load_run(str(path))
    assert loaded.unsaved_result_fields == ()
    assert loaded.experiment == experiment
    assert loaded.result_matches is True
    assert loaded.verify_reproduction().matches


@pytest.mark.smoke
def test_sequence_ir_facade_executes_pulseq_round_trip() -> None:
    native = SequenceIR(
        blocks=(
            SequenceBlock(
                duration_seconds=1e-3,
                rf=RFPulse([250.0j], dwell_seconds=1e-3),
            ),
            SequenceBlock(
                duration_seconds=2e-3,
                adc=ADCEvent(2, dwell_seconds=1e-3),
            ),
        ),
        definitions={
            "BlockDurationRaster": 1e-3,
            "GradientRasterTime": 1e-3,
            "RadiofrequencyRasterTime": 1e-3,
        },
    )
    pulseq = parse_pulseq(serialize_pulseq(native))
    experiment = Experiment(
        sequence=SequenceIRExecution(ir=pulseq),
        sample=Sample(sequence_domain=_general_domain()),
    )
    plan = experiment.plan()
    finding = next(item for item in plan.findings if item.rule == "sequence_ir")
    assert plan.ok
    assert finding.details["source_format"] == "pulseq"
    assert experiment.run().result.signal.size == 2


@pytest.mark.smoke
def test_sequence_ir_facade_rejects_missing_domain_and_probe_effects() -> None:
    missing = Experiment(sequence=SequenceIRExecution(ir=_general_ir())).plan()
    assert not missing.ok
    assert any("sequence_domain" in error for error in missing.errors)

    policy = HardwareEffectsPolicy(transmit="apply")
    unsupported = Experiment(
        sequence=SequenceIRExecution(ir=_general_ir(hardware_effects=policy)),
        sample=Sample(sequence_domain=_general_domain()),
    ).plan()
    assert not unsupported.ok
    assert any("probe hardware effects" in error for error in unsupported.errors)

    extended = Experiment(
        sequence=SequenceIRExecution(
            ir=SequenceIR(
                blocks=(SequenceBlock(1e-3, extensions=("custom",)),)
            )
        ),
        sample=Sample(sequence_domain=_general_domain()),
    ).plan()
    assert not extended.ok
    assert any("block extensions" in error for error in extended.errors)


@pytest.mark.smoke
def test_registry_entries_point_at_public_workflows() -> None:
    import spin_dynamics.esr as esr
    import spin_dynamics.nqr as nqr

    entries = available_workflows()
    assert len(entries) == 31
    public = set(workflows.STABLE_WORKFLOW_API) | set(workflows.EXTENDED_WORKFLOW_API)
    for entry in entries:
        if getattr(nqr, entry.name, None) is not None:
            assert getattr(nqr, entry.name) is entry.func, entry.name
        elif entry.sequence_type is ESRDEER:
            from spin_dynamics.experiment import esr_adapter

            assert entry.func is esr_adapter.run_deer
        elif entry.sequence_type in (
            ESRTwoPulseESEEM,
            ESRThreePulseESEEM,
            ESRHYSCORE,
        ):
            from spin_dynamics.experiment import esr_multidim_adapter

            assert entry.func.__module__ == esr_multidim_adapter.__name__
        elif entry.sequence_type is SequenceIRExecution:
            from spin_dynamics.experiment import sequence_adapter

            assert entry.func is sequence_adapter.run_sequence_ir
        elif getattr(esr, entry.name, None) is not None:
            assert getattr(esr, entry.name) is entry.func, entry.name
        else:
            assert entry.name in public, entry.name
            assert getattr(workflows, entry.name) is entry.func
