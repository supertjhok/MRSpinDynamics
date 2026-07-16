"""Recoverable batched acquisition around an adaptive design session.

The live layer does not implement hardware control.  It defines a narrow
instrument protocol, records a pending batch before calling that protocol, and
only updates the Bayesian posterior for accepted observations.  This keeps
external I/O separate from inference while making crash recovery and audit
behavior explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from spin_dynamics.design.constraints import DesignConstraint
from spin_dynamics.design.diagnostics import QuantityOfInterest, summarize_quantity
from spin_dynamics.design.io import (
    load_design_state,
    save_design_state_atomic,
)
from spin_dynamics.design.models import PredictiveModel
from spin_dynamics.design.session import (
    AdaptiveDesignSession,
    CandidateScore,
)
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import PhysicalCost, StopRule
from spin_dynamics.design.utilities import DesignUtility


@dataclass(frozen=True, eq=False)
class AcquisitionRequest:
    """One selected action sent to an external instrument adapter."""

    request_id: str
    batch_index: int
    design_index: int
    design: Any
    expected_physical_seconds: float
    selection_reason: str


@dataclass(frozen=True, eq=False)
class AcquisitionOutcome:
    """Accepted observation or explicit instrument rejection for one request."""

    request_id: str
    accepted: bool
    observation: np.ndarray | float | complex | None = None
    reason: str = ""
    retryable: bool = False
    physical_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class DesignInstrument(Protocol):
    """External acquisition boundary consumed by :class:`LiveDesignSession`."""

    def acquire(
        self, requests: Sequence[AcquisitionRequest]
    ) -> Sequence[AcquisitionOutcome]: ...


@dataclass(frozen=True, eq=False)
class PendingBatch:
    """Checkpointed batch whose external acquisition is not yet reconciled."""

    batch_index: int
    requests: tuple[AcquisitionRequest, ...]
    evaluation_seed: int
    planning_seconds: float
    latency_exceeded: bool
    attempt_count: int = 0


@dataclass(frozen=True, eq=False)
class BatchExecution:
    """Completed batch with outcomes and operational timing."""

    batch_index: int
    requests: tuple[AcquisitionRequest, ...]
    outcomes: tuple[AcquisitionOutcome, ...]
    planning_seconds: float
    physical_seconds: float
    accepted: int
    rejected: int
    latency_exceeded: bool
    stopped: bool


class LiveDesignSession:
    """Run small, recoverable acquisition batches through an instrument protocol.

    Candidate utilities are ranked once per batch. The leading distinct,
    feasible, non-excluded actions are acquired without posterior adaptation
    inside that batch; accepted observations are then applied in request order.
    This is a pragmatic small-batch policy, not a non-myopic joint optimizer.

    A pending batch is atomically checkpointed before the instrument is called.
    If the call raises, automatic replay is refused because the hardware may
    already have acquired data. Use :meth:`resolve_pending` with externally
    recovered outcomes, or pass ``allow_retry=True`` explicitly.
    """

    def __init__(
        self,
        session: AdaptiveDesignSession,
        instrument: DesignInstrument,
        *,
        batch_size: int = 1,
        latency_budget_seconds: float | None = None,
        planning_can_overlap: bool = False,
        audit_quantities: Mapping[str, QuantityOfInterest] | None = None,
        audit_probability: float = 0.95,
        checkpoint_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if latency_budget_seconds is not None and (
            not np.isfinite(latency_budget_seconds) or latency_budget_seconds < 0.0
        ):
            raise ValueError("latency_budget_seconds must be finite and non-negative")
        if not 0.0 < audit_probability < 1.0:
            raise ValueError("audit_probability must lie strictly between zero and one")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.session = session
        self.instrument = instrument
        self.batch_size = int(batch_size)
        self.latency_budget_seconds = (
            None if latency_budget_seconds is None else float(latency_budget_seconds)
        )
        self.planning_can_overlap = bool(planning_can_overlap)
        self.audit_quantities = dict(audit_quantities or {})
        self.audit_probability = float(audit_probability)
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.clock = clock
        self.pending_batch: PendingBatch | None = None
        self.batch_count = 0
        self.planning_seconds = 0.0
        self.physical_seconds = 0.0
        self.permanently_rejected_design_indices: set[int] = set()
        self.audit_records: list[dict[str, Any]] = []

    @property
    def total_operational_seconds(self) -> float:
        """Physical time plus planning time that cannot overlap acquisition."""

        planning = 0.0 if self.planning_can_overlap else self.planning_seconds
        return float(self.physical_seconds + planning)

    @property
    def latency_exceeded_batches(self) -> int:
        """Number of planned batches whose measured planning exceeded budget."""

        return sum(
            record.get("event") == "batch_planned"
            and bool(record.get("latency_exceeded"))
            for record in self.audit_records
        )

    def _posterior_summary(self) -> dict[str, Any]:
        posterior = self.session.posterior
        result: dict[str, Any] = {
            "particle_count": posterior.particle_count,
            "effective_sample_size": posterior.effective_sample_size,
            "observations": len(self.session.history),
            "ask_count": self.session.ask_count,
            "quantities": {},
        }
        quantities: dict[str, Any] = {}
        for name, quantity in self.audit_quantities.items():
            summary = summarize_quantity(
                posterior, quantity, probability=self.audit_probability
            )
            quantities[name] = {
                "mean": summary.mean.copy(),
                "standard_deviation": summary.standard_deviation.copy(),
                "lower": summary.lower.copy(),
                "upper": summary.upper.copy(),
                "probability": summary.probability,
            }
        result["quantities"] = quantities
        return result

    @staticmethod
    def _score_record(
        score: CandidateScore,
        *,
        selected_indices: set[int],
        live_excluded_indices: set[int],
    ) -> dict[str, Any]:
        utility = None
        if score.utility is not None:
            utility = {
                "value": score.utility.value,
                "standard_error": score.utility.standard_error,
                "samples": int(np.asarray(score.utility.samples).size),
                "units": score.utility.units,
            }
        return {
            "design_index": score.design_index,
            "design": repr(score.design),
            "feasible": score.feasible,
            "constraint_messages": list(score.messages),
            "live_excluded": score.design_index in live_excluded_indices,
            "selected": score.design_index in selected_indices,
            "cost_seconds": score.cost_seconds,
            "utility_rate": score.utility_rate,
            "utility": utility,
        }

    def _append_audit(self, event: str, **payload: Any) -> None:
        self.audit_records.append(
            {
                "sequence": len(self.audit_records) + 1,
                "event": event,
                **payload,
            }
        )

    def _persist(self) -> None:
        if self.checkpoint_path is not None:
            save_design_state_atomic(self.checkpoint_path, self.state_dict())
        if self.audit_path is not None:
            save_design_state_atomic(
                self.audit_path,
                {"version": 1, "records": self.audit_records},
            )

    def plan_batch(self) -> PendingBatch:
        """Rank candidates once and atomically record a pending open-loop batch."""

        if self.pending_batch is not None:
            raise RuntimeError("resolve the pending batch before planning another")
        if self.session.should_stop():
            raise RuntimeError("the posterior stopping rule has already been reached")
        started = float(self.clock())
        recommendation = self.session.ask()
        available = [
            score
            for score in recommendation.scores
            if score.feasible
            and score.design_index not in self.permanently_rejected_design_indices
        ]
        if not available:
            raise RuntimeError("no feasible non-rejected acquisition actions remain")
        selected = available[: self.batch_size]
        elapsed = float(self.clock()) - started
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("clock produced a non-finite or negative planning duration")
        latency_exceeded = (
            self.latency_budget_seconds is not None
            and elapsed > self.latency_budget_seconds
        )
        batch_index = self.batch_count + 1
        requests = tuple(
            AcquisitionRequest(
                request_id=f"batch-{batch_index:06d}-item-{rank:03d}",
                batch_index=batch_index,
                design_index=score.design_index,
                design=score.design,
                expected_physical_seconds=float(score.cost_seconds),
                selection_reason=(
                    f"rank {rank} by expected utility per physical second in "
                    "a distinct open-loop batch"
                ),
            )
            for rank, score in enumerate(selected, start=1)
        )
        self.pending_batch = PendingBatch(
            batch_index=batch_index,
            requests=requests,
            evaluation_seed=recommendation.evaluation_seed,
            planning_seconds=elapsed,
            latency_exceeded=latency_exceeded,
        )
        self.planning_seconds += elapsed
        selected_indices = {request.design_index for request in requests}
        self._append_audit(
            "batch_planned",
            batch_index=batch_index,
            evaluation_seed=recommendation.evaluation_seed,
            planning_seconds=elapsed,
            latency_budget_seconds=self.latency_budget_seconds,
            latency_exceeded=latency_exceeded,
            selection_policy="ranked unique actions; no within-batch adaptation",
            posterior_before=self._posterior_summary(),
            candidates=[
                self._score_record(
                    score,
                    selected_indices=selected_indices,
                    live_excluded_indices=self.permanently_rejected_design_indices,
                )
                for score in recommendation.scores
            ],
            requests=[self._request_record(request) for request in requests],
        )
        self._persist()
        return self.pending_batch

    @staticmethod
    def _request_record(request: AcquisitionRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "batch_index": request.batch_index,
            "design_index": request.design_index,
            "design": repr(request.design),
            "expected_physical_seconds": request.expected_physical_seconds,
            "selection_reason": request.selection_reason,
        }

    @staticmethod
    def _outcome_record(
        outcome: AcquisitionOutcome, request: AcquisitionRequest, charged_seconds: float
    ) -> dict[str, Any]:
        observation = None
        if outcome.observation is not None:
            observation = np.asarray(outcome.observation).copy()
        return {
            "request_id": outcome.request_id,
            "design_index": request.design_index,
            "accepted": outcome.accepted,
            "observation": observation,
            "reason": outcome.reason,
            "retryable": outcome.retryable,
            "physical_seconds": charged_seconds,
            "metadata": dict(outcome.metadata),
        }

    def execute_pending(self, *, allow_retry: bool = False) -> BatchExecution:
        """Call the instrument for the pending batch and reconcile its outcomes."""

        pending = self.pending_batch
        if pending is None:
            raise RuntimeError("there is no pending batch to execute")
        if pending.attempt_count > 0 and not allow_retry:
            raise RuntimeError(
                "the pending batch may already have reached the instrument; "
                "resolve it explicitly or pass allow_retry=True"
            )
        self.pending_batch = replace(pending, attempt_count=pending.attempt_count + 1)
        attempt_count = self.pending_batch.attempt_count
        self._append_audit(
            "instrument_call_started",
            batch_index=pending.batch_index,
            attempt_count=attempt_count,
            explicit_retry=bool(pending.attempt_count > 0),
        )
        self._persist()
        try:
            outcomes = tuple(self.instrument.acquire(pending.requests))
        except Exception as exc:
            self._append_audit(
                "instrument_call_failed",
                batch_index=pending.batch_index,
                attempt_count=attempt_count,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._persist()
            raise
        try:
            return self.resolve_pending(outcomes)
        except Exception as exc:
            self._append_audit(
                "instrument_outcomes_invalid",
                batch_index=pending.batch_index,
                attempt_count=attempt_count,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._persist()
            raise

    def _ordered_validated_outcomes(
        self, outcomes: Sequence[AcquisitionOutcome]
    ) -> tuple[tuple[AcquisitionOutcome, AcquisitionRequest, float], ...]:
        pending = self.pending_batch
        if pending is None:
            raise RuntimeError("there is no pending batch to resolve")
        by_id: dict[str, AcquisitionOutcome] = {}
        for outcome in outcomes:
            if not isinstance(outcome, AcquisitionOutcome):
                raise TypeError("instrument outcomes must be AcquisitionOutcome objects")
            if outcome.request_id in by_id:
                raise ValueError(f"duplicate outcome for {outcome.request_id!r}")
            by_id[outcome.request_id] = outcome
        expected_ids = {request.request_id for request in pending.requests}
        if set(by_id) != expected_ids:
            missing = sorted(expected_ids - set(by_id))
            extra = sorted(set(by_id) - expected_ids)
            raise ValueError(f"instrument outcome IDs differ; missing={missing}, extra={extra}")
        validated = []
        for request in pending.requests:
            outcome = by_id[request.request_id]
            if outcome.accepted:
                if outcome.observation is None:
                    raise ValueError("accepted outcomes require an observation")
                observed = np.asarray(outcome.observation)
                if np.any(~np.isfinite(observed)):
                    raise ValueError("accepted observations must be finite")
            elif not outcome.reason:
                raise ValueError("rejected outcomes require a reason")
            if outcome.retryable and outcome.accepted:
                raise ValueError("accepted outcomes cannot be marked retryable")
            if outcome.physical_seconds is None:
                charged = request.expected_physical_seconds if outcome.accepted else 0.0
            else:
                charged = float(outcome.physical_seconds)
            if not np.isfinite(charged) or charged < 0.0:
                raise ValueError("outcome physical_seconds must be finite and non-negative")
            validated.append((outcome, request, charged))
        return tuple(validated)

    def _restore_session(self, state: Mapping[str, Any]) -> None:
        current = self.session
        self.session = AdaptiveDesignSession.from_state(
            state,
            model=current.model,
            design_space=current.design_space,
            utility=current.utility,
            cost=current.cost,
            constraints=current.constraints,
            stopping_rule=current.stopping_rule,
        )

    def resolve_pending(
        self, outcomes: Sequence[AcquisitionOutcome]
    ) -> BatchExecution:
        """Reconcile known outcomes without calling the instrument again."""

        pending = self.pending_batch
        if pending is None:
            raise RuntimeError("there is no pending batch to resolve")
        validated = self._ordered_validated_outcomes(outcomes)
        session_state = self.session.state_dict()
        try:
            for outcome, request, _ in validated:
                if outcome.accepted:
                    self.session.tell(request.design, outcome.observation)
        except Exception:
            self._restore_session(session_state)
            raise
        charged_seconds = float(sum(item[2] for item in validated))
        rejected_indices = {
            request.design_index
            for outcome, request, _ in validated
            if not outcome.accepted and not outcome.retryable
        }
        self.physical_seconds += charged_seconds
        self.permanently_rejected_design_indices.update(rejected_indices)
        accepted = sum(outcome.accepted for outcome, _, _ in validated)
        rejected = len(validated) - accepted
        self.batch_count += 1
        self.pending_batch = None
        stopped = self.session.should_stop()
        execution = BatchExecution(
            batch_index=pending.batch_index,
            requests=pending.requests,
            outcomes=tuple(item[0] for item in validated),
            planning_seconds=pending.planning_seconds,
            physical_seconds=charged_seconds,
            accepted=accepted,
            rejected=rejected,
            latency_exceeded=pending.latency_exceeded,
            stopped=stopped,
        )
        self._append_audit(
            "batch_completed",
            batch_index=pending.batch_index,
            planning_seconds=pending.planning_seconds,
            physical_seconds=charged_seconds,
            total_physical_seconds=self.physical_seconds,
            total_planning_seconds=self.planning_seconds,
            total_operational_seconds=self.total_operational_seconds,
            accepted=accepted,
            rejected=rejected,
            permanently_rejected_design_indices=sorted(
                self.permanently_rejected_design_indices
            ),
            stopped=stopped,
            posterior_after=self._posterior_summary(),
            outcomes=[
                self._outcome_record(outcome, request, charged)
                for outcome, request, charged in validated
            ],
        )
        self._persist()
        return execution

    def run_batch(self) -> BatchExecution:
        """Plan and execute one batch, refusing ambiguous automatic retries."""

        if self.pending_batch is None:
            self.plan_batch()
        return self.execute_pending()

    def run(self, *, maximum_batches: int) -> tuple[BatchExecution, ...]:
        """Run until the stopping rule is reached or the batch limit is exhausted."""

        if (
            isinstance(maximum_batches, bool)
            or int(maximum_batches) != maximum_batches
            or maximum_batches <= 0
        ):
            raise ValueError("maximum_batches must be a positive integer")
        executions: list[BatchExecution] = []
        for _ in range(int(maximum_batches)):
            if self.session.should_stop():
                break
            executions.append(self.run_batch())
        return tuple(executions)

    def state_dict(self) -> dict[str, Any]:
        """Return checkpoint state including any unresolved external batch."""

        pending = None
        if self.pending_batch is not None:
            pending = {
                "batch_index": self.pending_batch.batch_index,
                "requests": [
                    {
                        "request_id": request.request_id,
                        "batch_index": request.batch_index,
                        "design_index": request.design_index,
                        "expected_physical_seconds": request.expected_physical_seconds,
                        "selection_reason": request.selection_reason,
                    }
                    for request in self.pending_batch.requests
                ],
                "evaluation_seed": self.pending_batch.evaluation_seed,
                "planning_seconds": self.pending_batch.planning_seconds,
                "latency_exceeded": self.pending_batch.latency_exceeded,
                "attempt_count": self.pending_batch.attempt_count,
            }
        return {
            "version": 1,
            "session": self.session.state_dict(),
            "config": {
                "batch_size": self.batch_size,
                "latency_budget_seconds": self.latency_budget_seconds,
                "planning_can_overlap": self.planning_can_overlap,
                "audit_probability": self.audit_probability,
                "audit_path": None if self.audit_path is None else str(self.audit_path),
            },
            "batch_count": self.batch_count,
            "planning_seconds": self.planning_seconds,
            "physical_seconds": self.physical_seconds,
            "permanently_rejected_design_indices": sorted(
                self.permanently_rejected_design_indices
            ),
            "pending_batch": pending,
            "audit_records": list(self.audit_records),
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        instrument: DesignInstrument,
        model: PredictiveModel,
        design_space: CandidateDesignSpace,
        utility: DesignUtility,
        cost: PhysicalCost,
        constraints: Sequence[DesignConstraint] = (),
        stopping_rule: StopRule | None = None,
        audit_quantities: Mapping[str, QuantityOfInterest] | None = None,
        checkpoint_path: str | Path | None = None,
        audit_path: str | Path | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> "LiveDesignSession":
        """Restore live state while re-supplying executable dependencies."""

        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported live-design checkpoint version")
        base = AdaptiveDesignSession.from_state(
            state["session"],
            model=model,
            design_space=design_space,
            utility=utility,
            cost=cost,
            constraints=constraints,
            stopping_rule=stopping_rule,
        )
        config = state["config"]
        if audit_path is None and config.get("audit_path") is not None:
            audit_path = Path(config["audit_path"])
        live = cls(
            base,
            instrument,
            batch_size=int(config["batch_size"]),
            latency_budget_seconds=config.get("latency_budget_seconds"),
            planning_can_overlap=bool(config.get("planning_can_overlap", False)),
            audit_quantities=audit_quantities,
            audit_probability=float(config.get("audit_probability", 0.95)),
            checkpoint_path=checkpoint_path,
            audit_path=audit_path,
            clock=clock,
        )
        live.batch_count = int(state["batch_count"])
        live.planning_seconds = float(state["planning_seconds"])
        live.physical_seconds = float(state["physical_seconds"])
        live.permanently_rejected_design_indices = {
            int(index) for index in state["permanently_rejected_design_indices"]
        }
        if any(
            index < 0 or index >= len(design_space.actions)
            for index in live.permanently_rejected_design_indices
        ):
            raise ValueError("checkpoint contains an invalid rejected design index")
        live.audit_records = [dict(record) for record in state["audit_records"]]
        pending = state.get("pending_batch")
        if pending is not None:
            requests = []
            for item in pending["requests"]:
                design_index = int(item["design_index"])
                if design_index < 0 or design_index >= len(design_space.actions):
                    raise ValueError("checkpoint pending batch has invalid design index")
                requests.append(
                    AcquisitionRequest(
                        request_id=str(item["request_id"]),
                        batch_index=int(item["batch_index"]),
                        design_index=design_index,
                        design=design_space.actions[design_index],
                        expected_physical_seconds=float(
                            item["expected_physical_seconds"]
                        ),
                        selection_reason=str(item["selection_reason"]),
                    )
                )
            live.pending_batch = PendingBatch(
                batch_index=int(pending["batch_index"]),
                requests=tuple(requests),
                evaluation_seed=int(pending["evaluation_seed"]),
                planning_seconds=float(pending["planning_seconds"]),
                latency_exceeded=bool(pending["latency_exceeded"]),
                attempt_count=int(pending["attempt_count"]),
            )
        return live

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, **dependencies: Any
    ) -> "LiveDesignSession":
        """Load an atomic live checkpoint and restore executable dependencies."""

        dependencies.setdefault("checkpoint_path", path)
        return cls.from_state(load_design_state(path), **dependencies)
