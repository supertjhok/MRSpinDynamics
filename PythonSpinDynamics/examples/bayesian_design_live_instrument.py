"""Run recoverable batched Bayesian design through an instrument boundary.

The supplied instrument is synthetic but implements the same ``acquire``
protocol expected from a hardware driver. It rejects the first requested action
to demonstrate interlock handling, then returns noisy CPMG-IR echo integrals.
Replace ``SyntheticCPMGInstrument`` with a laboratory adapter while retaining
the live session, atomic checkpoint, and audit workflow.
"""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.design import (  # noqa: E402
    AcquisitionOutcome,
    CPMGIRAdapter,
    CPMGIRDesign,
    CandidateDesignSpace,
    ComplexGaussianLikelihood,
    CredibleIntervalStopping,
    ExpectedVarianceReduction,
    LiveDesignSession,
    ParticlePosterior,
    make_adapter_session,
    summarize_quantity,
)
from spin_dynamics.experiment import (  # noqa: E402
    Acquisition,
    CPMGIRTrain,
    Experiment,
    Sample,
)


def t1_quantity(parameters):
    return parameters["t1_seconds"]


class SyntheticCPMGInstrument:
    """Example external adapter; replace ``acquire`` with hardware I/O."""

    def __init__(self, adapter, likelihood, truth, *, seed: int):
        self.adapter = adapter
        self.likelihood = likelihood
        self.truth = truth
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def acquire(self, requests):
        self.calls += 1
        outcomes = []
        for index, request in enumerate(requests):
            if self.calls == 1 and index == 0:
                outcomes.append(
                    AcquisitionOutcome(
                        request.request_id,
                        accepted=False,
                        reason="synthetic RF-amplifier interlock",
                        physical_seconds=1e-3,
                        metadata={"interlock": "rf_amplifier"},
                    )
                )
                continue
            clean = self.adapter.simulate(self.truth, request.design)
            observed = self.likelihood.sample(clean, self.rng)
            outcomes.append(
                AcquisitionOutcome(
                    request.request_id,
                    accepted=True,
                    observation=observed,
                    physical_seconds=request.expected_physical_seconds,
                    metadata={"instrument": "synthetic_cpmg"},
                )
            )
        return outcomes


def _build(seed: int, checkpoint: Path, audit: Path, batch_size: int, latency: float):
    template = Experiment(
        CPMGIRTrain(num_echoes=2, echo_spacing_seconds=1e-3),
        Sample(t1_seconds=0.15, t2_seconds=0.06),
        acquisition=Acquisition(numpts=21, maxoffs=5.0, rephase_action="ignore"),
    )
    adapter = CPMGIRAdapter(
        template,
        {"t1_seconds": 0.15, "t2_seconds": 0.06},
        echo_index=0,
        recovery_seconds=20e-3,
        fixed_overhead_seconds=2e-3,
    )
    t1 = np.geomspace(0.05, 0.45, 17)
    posterior = ParticlePosterior(
        {
            "t1_seconds": t1,
            "t2_seconds": np.full(t1.size, 0.06),
        }
    )
    actions = CandidateDesignSpace(
        CPMGIRDesign(float(delay)) for delay in np.geomspace(5e-3, 0.6, 8)
    )
    likelihood = ComplexGaussianLikelihood(0.5)
    base = make_adapter_session(
        adapter=adapter,
        likelihood=likelihood,
        posterior=posterior,
        design_space=actions,
        utility=ExpectedVarianceReduction(t1_quantity, samples=32, scale=0.1),
        stopping_rule=CredibleIntervalStopping(
            t1_quantity, maximum_width=0.05, probability=0.9
        ),
        seed=seed,
    )
    instrument = SyntheticCPMGInstrument(
        adapter,
        likelihood,
        {"t1_seconds": 0.18, "t2_seconds": 0.06},
        seed=seed + 1,
    )
    live = LiveDesignSession(
        base,
        instrument,
        batch_size=batch_size,
        latency_budget_seconds=latency,
        audit_quantities={"T1_seconds": t1_quantity},
        audit_probability=0.9,
        checkpoint_path=checkpoint,
        audit_path=audit,
    )
    return live


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latency-budget-ms", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("results/bayesian-live-state.json")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("results/bayesian-live-audit.json")
    )
    args = parser.parse_args()
    if args.batches <= 0 or args.batch_size <= 0 or args.latency_budget_ms < 0.0:
        parser.error("batches and batch-size must be positive; latency must be non-negative")

    live = _build(
        args.seed,
        args.checkpoint,
        args.audit,
        args.batch_size,
        args.latency_budget_ms * 1e-3,
    )
    for execution in live.run(maximum_batches=args.batches):
        delays = ", ".join(
            f"{1e3 * request.design.delay_seconds:.1f} ms"
            for request in execution.requests
        )
        print(
            f"batch={execution.batch_index} delays=[{delays}] "
            f"accepted={execution.accepted} rejected={execution.rejected} "
            f"lab={execution.physical_seconds:.4g} s "
            f"planning={execution.planning_seconds:.4g} s "
            f"latency_exceeded={execution.latency_exceeded}"
        )
    summary = summarize_quantity(live.session.posterior, t1_quantity, probability=0.9)
    print(
        f"posterior T1={summary.mean[0]:.4g} s "
        f"90% interval=[{summary.lower[0]:.4g}, {summary.upper[0]:.4g}] s"
    )
    print(
        f"physical={live.physical_seconds:.4g} s "
        f"planning={live.planning_seconds:.4g} s "
        f"operational={live.total_operational_seconds:.4g} s"
    )
    print(f"checkpoint: {args.checkpoint}")
    print(f"audit: {args.audit}")


if __name__ == "__main__":
    main()
