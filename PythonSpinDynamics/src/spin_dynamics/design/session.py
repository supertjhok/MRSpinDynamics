"""Adaptive ask/tell sessions over finite candidate design spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from spin_dynamics.design.constraints import (
    DesignConstraint,
    evaluate_constraints,
)
from spin_dynamics.design.io import state_from_json, state_to_json
from spin_dynamics.design.models import PredictiveModel
from spin_dynamics.design.posterior import (
    ParticlePosterior,
    posterior_from_state,
)
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import PhysicalCost, StopRule
from spin_dynamics.design.utilities import DesignUtility, UtilityEstimate


@dataclass(frozen=True, eq=False)
class CandidateScore:
    """Utility, physical cost, and feasibility for one action."""

    design: Any
    design_index: int
    feasible: bool
    messages: tuple[str, ...]
    utility: UtilityEstimate | None
    cost_seconds: float | None
    utility_rate: float | None


@dataclass(frozen=True, eq=False)
class DesignRecommendation:
    """Ranked candidate scores from one common-random-number evaluation."""

    best: CandidateScore
    scores: tuple[CandidateScore, ...]
    evaluation_seed: int


@dataclass(frozen=True, eq=False)
class DesignObservation:
    """One observed result identified by its stable candidate index."""

    design_index: int
    observation: np.ndarray


class AdaptiveDesignSession:
    """Replayable finite-space Bayesian adaptive-design loop.

    Candidate utilities in one :meth:`ask` call use identical pseudorandom
    streams. This common-random-number comparison reduces ranking noise while
    the returned standard errors retain the uncertainty of each utility
    estimate. Checkpoints store posterior particles, RNG state, and candidate
    indices; the model, utility, cost, constraints, and candidate objects are
    supplied again when restoring because arbitrary callables are not safely
    serializable.
    """

    def __init__(
        self,
        *,
        model: PredictiveModel,
        posterior: ParticlePosterior,
        design_space: CandidateDesignSpace,
        utility: DesignUtility,
        cost: PhysicalCost,
        constraints: Sequence[DesignConstraint] = (),
        stopping_rule: StopRule | None = None,
        seed: int | None = None,
        resample_fraction: float | None = None,
    ) -> None:
        if resample_fraction is not None and not 0.0 < resample_fraction <= 1.0:
            raise ValueError("resample_fraction must lie in (0, 1]")
        self.model = model
        self.posterior = posterior
        self.design_space = design_space
        self.utility = utility
        self.cost = cost
        self.constraints = tuple(constraints)
        self.stopping_rule = stopping_rule
        self.resample_fraction = resample_fraction
        self.rng = np.random.default_rng(seed)
        self.history: list[DesignObservation] = []
        self.ask_count = 0

    def ask(self) -> DesignRecommendation:
        """Score every feasible action and recommend the best utility rate."""

        evaluation_seed = int(
            self.rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)
        )
        scores: list[CandidateScore] = []
        for index, design in enumerate(self.design_space.actions):
            feasible, messages = evaluate_constraints(design, self.constraints)
            if not feasible:
                scores.append(
                    CandidateScore(
                        design=design,
                        design_index=index,
                        feasible=False,
                        messages=messages,
                        utility=None,
                        cost_seconds=None,
                        utility_rate=None,
                    )
                )
                continue
            local_rng = np.random.default_rng(evaluation_seed)
            estimate = self.utility.estimate(
                self.model, self.posterior, design, local_rng
            )
            seconds = float(self.cost.seconds(design))
            if not np.isfinite(seconds) or seconds <= 0.0:
                raise ValueError("physical acquisition cost must be finite and positive")
            scores.append(
                CandidateScore(
                    design=design,
                    design_index=index,
                    feasible=True,
                    messages=(),
                    utility=estimate,
                    cost_seconds=seconds,
                    utility_rate=estimate.value / seconds,
                )
            )

        feasible_scores = [score for score in scores if score.feasible]
        if not feasible_scores:
            reasons = sorted({message for score in scores for message in score.messages})
            detail = "; ".join(reasons) or "all candidates were rejected"
            raise ValueError(f"no feasible candidate designs: {detail}")
        ranked_feasible = sorted(
            feasible_scores,
            key=lambda score: (
                -float(score.utility_rate),
                -float(score.utility.value),
                float(score.cost_seconds),
                score.design_index,
            ),
        )
        rank = {score.design_index: order for order, score in enumerate(ranked_feasible)}
        ranked = tuple(
            sorted(
                scores,
                key=lambda score: (
                    0 if score.feasible else 1,
                    rank.get(score.design_index, score.design_index),
                ),
            )
        )
        self.ask_count += 1
        return DesignRecommendation(
            best=ranked_feasible[0],
            scores=ranked,
            evaluation_seed=evaluation_seed,
        )

    def tell(
        self,
        design: Any,
        observation: np.ndarray | float | complex,
    ) -> DesignObservation:
        """Ingest an observation and update importance weights."""

        design_index = self.design_space.index(design)
        observed = np.asarray(observation)
        if np.any(~np.isfinite(observed)):
            raise ValueError("observation must be finite")
        log_likelihood = self.model.log_likelihood(
            observed, self.posterior.parameters, design
        )
        self.posterior = self.posterior.updated(log_likelihood)
        if (
            self.resample_fraction is not None
            and self.posterior.effective_sample_size
            < self.resample_fraction * self.posterior.particle_count
        ):
            self.posterior = self.posterior.resampled(self.rng)
        record = DesignObservation(design_index, observed.copy())
        self.history.append(record)
        return record

    def should_stop(self) -> bool:
        """Evaluate the configured posterior stopping rule."""

        if self.stopping_rule is None:
            return False
        return bool(self.stopping_rule.reached(self.posterior))

    @property
    def elapsed_seconds(self) -> float:
        return float(
            sum(
                self.cost.seconds(self.design_space.actions[item.design_index])
                for item in self.history
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "posterior": self.posterior.state_dict(),
            "rng_state": self.rng.bit_generator.state,
            "history": [
                {
                    "design_index": item.design_index,
                    "observation": item.observation.copy(),
                }
                for item in self.history
            ],
            "ask_count": self.ask_count,
            "candidate_count": len(self.design_space.actions),
            "resample_fraction": self.resample_fraction,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return state_to_json(self.state_dict(), indent=indent)

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        model: PredictiveModel,
        design_space: CandidateDesignSpace,
        utility: DesignUtility,
        cost: PhysicalCost,
        constraints: Sequence[DesignConstraint] = (),
        stopping_rule: StopRule | None = None,
    ) -> "AdaptiveDesignSession":
        """Restore state while explicitly re-supplying executable objects."""

        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported adaptive-session checkpoint version")
        if int(state["candidate_count"]) != len(design_space.actions):
            raise ValueError("checkpoint candidate count does not match design_space")
        session = cls(
            model=model,
            posterior=posterior_from_state(state["posterior"]),
            design_space=design_space,
            utility=utility,
            cost=cost,
            constraints=constraints,
            stopping_rule=stopping_rule,
            seed=0,
            resample_fraction=state.get("resample_fraction"),
        )
        session.rng.bit_generator.state = dict(state["rng_state"])
        session.history = [
            DesignObservation(
                int(item["design_index"]), np.asarray(item["observation"]).copy()
            )
            for item in state["history"]
        ]
        if any(
            item.design_index < 0 or item.design_index >= len(design_space.actions)
            for item in session.history
        ):
            raise ValueError("checkpoint contains an invalid design index")
        session.ask_count = int(state["ask_count"])
        return session

    @classmethod
    def from_json(
        cls,
        payload: str,
        **dependencies: Any,
    ) -> "AdaptiveDesignSession":
        return cls.from_state(state_from_json(payload), **dependencies)

