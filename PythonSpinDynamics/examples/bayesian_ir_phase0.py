"""Compare Phase 0 adaptive and fixed inversion-recovery schedules.

This is a synthetic algorithm benchmark. It does not control an instrument or
establish experimental performance.
"""

from __future__ import annotations

import argparse

import numpy as np

from spin_dynamics.design import (
    IRAcquisitionCost,
    IRDesign,
    IRGridPrior,
    IRTruth,
    run_ir_reference_trial,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="number of prior-grid truths to simulate (default: 12)",
    )
    parser.add_argument(
        "--seed", type=int, default=20260716, help="benchmark RNG seed"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")

    prior = IRGridPrior(
        t1_seconds=np.geomspace(0.05, 0.8, 17),
        amplitude=np.array([0.9, 1.0, 1.1]),
        baseline=np.array([-0.03, 0.0, 0.03]),
        sigma=np.array([0.04, 0.06]),
    )
    candidates = tuple(IRDesign(delay) for delay in np.geomspace(0.015, 1.6, 10))
    cost = IRAcquisitionCost(readout_seconds=0.01, recovery_seconds=0.08)
    master = np.random.default_rng(args.seed)
    truth_indices = master.integers(0, prior.t1_seconds.size, size=args.trials)

    adaptive_times: list[float] = []
    fixed_times: list[float] = []
    adaptive_reached = 0
    fixed_reached = 0
    for trial_index, truth_index in enumerate(truth_indices):
        truth = IRTruth(
            t1_seconds=float(prior.t1_seconds[truth_index]),
            sigma=0.05,
        )
        seed = args.seed + trial_index
        adaptive = run_ir_reference_trial(
            prior=prior,
            truth=truth,
            candidate_designs=candidates,
            cost=cost,
            rng=np.random.default_rng(seed),
            policy="adaptive",
            maximum_actions=12,
            maximum_relative_width=0.3,
            quadrature_order=8,
        )
        fixed = run_ir_reference_trial(
            prior=prior,
            truth=truth,
            candidate_designs=candidates,
            fixed_schedule=tuple(reversed(candidates)),
            cost=cost,
            rng=np.random.default_rng(seed),
            policy="fixed",
            maximum_actions=12,
            maximum_relative_width=0.3,
            quadrature_order=8,
        )
        adaptive_times.append(adaptive.elapsed_seconds)
        fixed_times.append(fixed.elapsed_seconds)
        adaptive_reached += int(adaptive.reached_target)
        fixed_reached += int(fixed.reached_target)

    adaptive_median = float(np.median(adaptive_times))
    fixed_median = float(np.median(fixed_times))
    print(f"adaptive target reached: {adaptive_reached}/{args.trials}")
    print(f"fixed target reached:    {fixed_reached}/{args.trials}")
    print(f"adaptive median time:    {adaptive_median:.3f} s")
    print(f"fixed median time:       {fixed_median:.3f} s")
    if fixed_median > 0.0:
        print(f"median time ratio:       {adaptive_median / fixed_median:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

