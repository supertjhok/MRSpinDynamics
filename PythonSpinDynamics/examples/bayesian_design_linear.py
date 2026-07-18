"""Run the generic Phase 1 adaptive-design loop on a linear measurement.

The example estimates an uncertain slope and offset from scalar observations.
It demonstrates particle priors, a predictive likelihood, finite candidates,
physical cost, a goal-oriented utility, stopping, and checkpoint output without
requiring optional dependencies or instrument control.
"""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from spin_dynamics.design import (
    AdaptiveDesignSession,
    CallableConstraint,
    CallableCost,
    CandidateDesignSpace,
    ExpectedInformationGain,
    ExpectedVarianceReduction,
    GaussianLikelihood,
    IndependentPrior,
    NormalPrior,
    ParticlePosterior,
    PosteriorStandardDeviationStopping,
    PredictiveModel,
    save_design_state,
    summarize_quantity,
)


@dataclass(frozen=True)
class MeasurementSetting:
    x: float
    duration_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--particles", type=int, default=2048)
    parser.add_argument("--utility-samples", type=int, default=128)
    parser.add_argument("--max-actions", type=int, default=8)
    parser.add_argument(
        "--utility",
        choices=("slope-variance", "full-eig"),
        default="slope-variance",
    )
    parser.add_argument("--checkpoint", help="optional JSON session-state path")
    return parser


# Keep orchestration in one entry point so helper functions remain reusable.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.particles <= 0 or args.utility_samples <= 0 or args.max_actions <= 0:
        raise SystemExit("particle, utility-sample, and action counts must be positive")

    truth_slope = 1.4
    truth_offset = -0.2
    observation_sigma = 0.2

    def quantity(parameters):
        return parameters["slope"]

    prior = IndependentPrior(
        {
            "slope": NormalPrior(0.0, 2.0),
            "offset": NormalPrior(0.0, 1.0),
        }
    )
    posterior = ParticlePosterior.from_prior(
        prior, args.particles, np.random.default_rng(args.seed)
    )

    def predict(parameters, design):
        return parameters["slope"] * design.x + parameters["offset"]

    model = PredictiveModel(predict, GaussianLikelihood(observation_sigma))
    actions = CandidateDesignSpace(
        MeasurementSetting(float(x), 0.5 + abs(float(x)))
        for x in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    )
    if args.utility == "full-eig":
        utility = ExpectedInformationGain(samples=args.utility_samples)
    else:
        utility = ExpectedVarianceReduction(
            quantity, samples=args.utility_samples, scale=1.0
        )
    session = AdaptiveDesignSession(
        model=model,
        posterior=posterior,
        design_space=actions,
        utility=utility,
        cost=CallableCost(lambda design: design.duration_seconds),
        constraints=(
            CallableConstraint(lambda design: abs(design.x) <= 2.0, "|x| exceeds 2"),
        ),
        stopping_rule=PosteriorStandardDeviationStopping(quantity, 0.08),
        seed=args.seed + 1,
        resample_fraction=0.25,
    )
    measurement_rng = np.random.default_rng(args.seed + 2)

    for action_number in range(1, args.max_actions + 1):
        recommendation = session.ask()
        score = recommendation.best
        design = score.design
        observation = measurement_rng.normal(
            truth_slope * design.x + truth_offset,
            observation_sigma,
        )
        session.tell(design, observation)
        summary = summarize_quantity(session.posterior, quantity, probability=0.9)
        print(
            f"{action_number:2d}: x={design.x:+.1f}, y={observation:+.3f}, "
            f"utility={score.utility.value:.4g} +/- {score.utility.standard_error:.2g}, "
            f"rate={score.utility_rate:.4g}/s, "
            f"slope={summary.mean[0]:+.3f} +/- {summary.standard_deviation[0]:.3f}"
        )
        if session.should_stop():
            break

    if args.checkpoint:
        save_design_state(args.checkpoint, session.state_dict())
        print(f"checkpoint: {args.checkpoint}")
    print(f"physical acquisition time: {session.elapsed_seconds:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
