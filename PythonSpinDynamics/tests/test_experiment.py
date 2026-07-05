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
    CPMGImaging,
    CPMGIRTrain,
    CPMGTrain,
    Experiment,
    ExperimentPlanError,
    Hardware,
    ImagingPlane,
    Phantom,
    Sample,
    SerializationError,
    SolenoidCoil,
    TxCoil,
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
def test_registry_entries_point_at_public_workflows() -> None:
    entries = available_workflows()
    assert len(entries) == 15
    public = set(workflows.STABLE_WORKFLOW_API) | set(workflows.EXTENDED_WORKFLOW_API)
    for entry in entries:
        assert entry.name in public, entry.name
        assert getattr(workflows, entry.name) is entry.func
