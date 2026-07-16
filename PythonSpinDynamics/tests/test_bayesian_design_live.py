"""Phase 4 batched acquisition, recovery, latency, and audit tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from spin_dynamics.design import (
    AcquisitionOutcome,
    AdaptiveDesignSession,
    CallableConstraint,
    CallableCost,
    CandidateDesignSpace,
    ExpectedInformationGain,
    GaussianLikelihood,
    GridPosterior,
    LiveDesignSession,
    PredictiveModel,
    load_design_state,
    save_design_state_atomic,
)


@dataclass(frozen=True)
class _Action:
    x: float
    duration_seconds: float = 0.2


def _quantity(parameters):
    return parameters["theta"]


def _dependencies():
    actions = CandidateDesignSpace(
        (_Action(0.25), _Action(0.75), _Action(1.5), _Action(4.0))
    )
    return {
        "model": PredictiveModel(
            lambda parameters, design: parameters["theta"] * design.x,
            GaussianLikelihood(0.08),
        ),
        "design_space": actions,
        "utility": ExpectedInformationGain(samples=32),
        "cost": CallableCost(lambda design: design.duration_seconds),
        "constraints": (
            CallableConstraint(lambda design: design.x < 3.0, "x exceeds hardware"),
        ),
    }


def _session(*, seed: int = 4):
    dependencies = _dependencies()
    return AdaptiveDesignSession(
        posterior=GridPosterior({"theta": np.array([-1.0, 0.0, 1.0])}),
        seed=seed,
        **dependencies,
    )


class _Clock:
    def __init__(self, *values: float):
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class _AcceptingInstrument:
    def acquire(self, requests):
        return [
            AcquisitionOutcome(
                request.request_id,
                accepted=True,
                observation=0.6 * request.design.x,
                physical_seconds=request.expected_physical_seconds,
                metadata={"source": "test"},
            )
            for request in requests
        ]


class _MixedInstrument:
    def acquire(self, requests):
        return [
            AcquisitionOutcome(
                requests[0].request_id,
                accepted=False,
                reason="amplifier interlock",
                physical_seconds=0.01,
            ),
            *[
                AcquisitionOutcome(
                    request.request_id,
                    accepted=True,
                    observation=0.6 * request.design.x,
                    physical_seconds=0.2,
                )
                for request in requests[1:]
            ],
        ]


class _FailingInstrument:
    def acquire(self, requests):
        raise TimeoutError(f"lost acknowledgement for {requests[0].request_id}")


def test_batch_planning_is_unique_timed_and_audited() -> None:
    live = LiveDesignSession(
        _session(),
        _AcceptingInstrument(),
        batch_size=2,
        latency_budget_seconds=0.1,
        audit_quantities={"theta": _quantity},
        clock=_Clock(10.0, 10.25),
    )

    pending = live.plan_batch()

    assert len(pending.requests) == 2
    assert len({request.design_index for request in pending.requests}) == 2
    assert pending.planning_seconds == pytest.approx(0.25)
    assert pending.latency_exceeded
    assert live.total_operational_seconds == pytest.approx(0.25)
    planned = live.audit_records[-1]
    assert planned["posterior_before"]["quantities"]["theta"]["mean"] == pytest.approx(
        [0.0]
    )
    assert any(
        candidate["constraint_messages"] == ["x exceeds hardware"]
        for candidate in planned["candidates"]
    )
    assert sum(candidate["selected"] for candidate in planned["candidates"]) == 2


def test_rejected_acquisition_is_not_told_and_is_excluded_later() -> None:
    live = LiveDesignSession(
        _session(),
        _MixedInstrument(),
        batch_size=2,
        planning_can_overlap=True,
        clock=_Clock(0.0, 0.02, 1.0, 1.03),
    )

    execution = live.run_batch()
    rejected_index = execution.requests[0].design_index

    assert execution.accepted == 1
    assert execution.rejected == 1
    assert len(live.session.history) == 1
    assert live.physical_seconds == pytest.approx(0.21)
    assert live.total_operational_seconds == pytest.approx(0.21)
    assert rejected_index in live.permanently_rejected_design_indices
    next_batch = live.plan_batch()
    assert rejected_index not in {
        request.design_index for request in next_batch.requests
    }


def test_atomic_checkpoint_replaces_complete_json(tmp_path) -> None:
    path = tmp_path / "state.json"
    save_design_state_atomic(path, {"version": 1, "value": np.array([1.0])})
    save_design_state_atomic(path, {"version": 1, "value": np.array([2.0, 3.0])})

    state = load_design_state(path)

    assert np.array_equal(state["value"], np.array([2.0, 3.0]))
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_atomic_checkpoint_failure_retains_previous_state(tmp_path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    save_design_state_atomic(path, {"version": 1, "value": "complete"})

    def fail_replace(source, destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("spin_dynamics.design.io.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace"):
        save_design_state_atomic(path, {"version": 1, "value": "partial"})

    assert load_design_state(path)["value"] == "complete"
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_failed_instrument_call_restores_ambiguous_pending_batch(tmp_path) -> None:
    checkpoint = tmp_path / "live.json"
    audit = tmp_path / "audit.json"
    live = LiveDesignSession(
        _session(seed=8),
        _FailingInstrument(),
        batch_size=2,
        checkpoint_path=checkpoint,
        audit_path=audit,
        clock=_Clock(0.0, 0.01),
    )

    with pytest.raises(TimeoutError, match="lost acknowledgement"):
        live.run_batch()

    stored = load_design_state(checkpoint)
    assert stored["pending_batch"]["attempt_count"] == 1
    assert load_design_state(audit)["records"][-1]["event"] == "instrument_call_failed"

    restored = LiveDesignSession.from_checkpoint(
        checkpoint,
        instrument=_AcceptingInstrument(),
        audit_quantities={"theta": _quantity},
        **_dependencies(),
    )
    with pytest.raises(RuntimeError, match="may already have reached"):
        restored.execute_pending()

    recovered = [
        AcquisitionOutcome(
            request.request_id,
            accepted=True,
            observation=0.6 * request.design.x,
            physical_seconds=request.expected_physical_seconds,
            metadata={"reconciled_from": "instrument archive"},
        )
        for request in restored.pending_batch.requests
    ]
    execution = restored.resolve_pending(recovered)

    assert execution.accepted == 2
    assert restored.pending_batch is None
    assert len(restored.session.history) == 2
    assert load_design_state(checkpoint)["pending_batch"] is None
    assert load_design_state(audit)["records"][-1]["event"] == "batch_completed"


def test_invalid_batch_outcome_does_not_partially_update_posterior() -> None:
    live = LiveDesignSession(
        _session(),
        _AcceptingInstrument(),
        batch_size=2,
        clock=_Clock(0.0, 0.01),
    )
    pending = live.plan_batch()
    outcomes = [
        AcquisitionOutcome(
            pending.requests[0].request_id,
            accepted=True,
            observation=0.2,
        ),
        AcquisitionOutcome(
            pending.requests[1].request_id,
            accepted=True,
            observation=np.nan,
        ),
    ]

    with pytest.raises(ValueError, match="finite"):
        live.resolve_pending(outcomes)

    assert len(live.session.history) == 0
    assert live.pending_batch is pending
