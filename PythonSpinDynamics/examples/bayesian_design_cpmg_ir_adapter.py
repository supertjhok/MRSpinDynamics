"""Choose CPMG inversion delays through the Experiment facade.

The script is a synthetic closed-loop demonstration: ``ask()`` recommends a
planner-validated acquisition, the existing CPMG-IR workflow generates a noisy
observation at a hidden truth, and ``tell()`` updates the particle posterior.
Replace only the synthetic acquisition block when connecting real data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spin_dynamics.design import (  # noqa: E402
    CPMGIRAdapter,
    CPMGIRDesign,
    CandidateDesignSpace,
    ComplexGaussianLikelihood,
    ExpectedVarianceReduction,
    ParticlePosterior,
    make_adapter_session,
)
from spin_dynamics.experiment import (  # noqa: E402
    Acquisition,
    CPMGIRTrain,
    Experiment,
    Sample,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    template = Experiment(
        sequence=CPMGIRTrain(num_echoes=2, echo_spacing_seconds=1e-3),
        sample=Sample(t1_seconds=0.15, t2_seconds=0.06),
        acquisition=Acquisition(numpts=21, maxoffs=5.0, rephase_action="ignore"),
    )
    adapter = CPMGIRAdapter(
        template=template,
        nominal_parameters={"t1_seconds": 0.15, "t2_seconds": 0.06},
        echo_index=0,
        recovery_seconds=20e-3,
        fixed_overhead_seconds=2e-3,
    )
    t1_particles = np.geomspace(0.05, 0.45, 15)
    posterior = ParticlePosterior(
        {
            "t1_seconds": t1_particles,
            "t2_seconds": np.full(t1_particles.size, 0.06),
        }
    )
    actions = CandidateDesignSpace(
        CPMGIRDesign(float(delay))
        for delay in np.geomspace(5e-3, 0.6, 8)
    )
    likelihood = ComplexGaussianLikelihood(sigma=0.5)
    session = make_adapter_session(
        adapter=adapter,
        likelihood=likelihood,
        posterior=posterior,
        design_space=actions,
        utility=ExpectedVarianceReduction(
            lambda parameters: parameters["t1_seconds"],
            samples=32,
            scale=0.1,
        ),
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed + 1)
    truth = {"t1_seconds": 0.18, "t2_seconds": 0.06}
    for step in range(args.steps):
        recommendation = session.ask()
        design = recommendation.best.design

        # Synthetic acquisition. In a laboratory, execute ``design`` and pass
        # the measured echo integral to ``session.tell`` instead.
        clean = adapter.simulate(truth, design)
        observed = likelihood.sample(clean, rng)
        session.tell(design, observed)

        weights = session.posterior.weights
        mean_t1 = float(np.sum(weights * session.posterior.parameters["t1_seconds"]))
        print(
            f"step={step + 1} delay={design.delay_seconds:.4g} s "
            f"cost={recommendation.best.cost_seconds:.4g} s "
            f"posterior_mean_T1={mean_t1:.4g} s"
        )


if __name__ == "__main__":
    main()
