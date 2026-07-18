"""Adapt a Qdyne reference frequency to an uncertain coherent signal."""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spin_dynamics.design import (  # noqa: E402
    CandidateDesignSpace,
    ExpectedVarianceReduction,
    GaussianLikelihood,
    NanoMRQdyneAdapter,
    NanoMRQdyneDesign,
    ParticlePosterior,
    make_adapter_session,
)
from spin_dynamics.experiment import (  # noqa: E402
    Experiment,
    Hardware,
    NanoMRQdyne,
    NanoMRSensor,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()

    template = Experiment(
        sequence=NanoMRQdyne(
            signal_frequency_hz=2.0e6,
            field_amplitude_tesla=20.0e-9,
            shot_count=128,
            sensing_duration_seconds=2.0e-6,
            repetition_interval_seconds=20.0e-6,
            reference_frequency_hz=2.0e6,
            seed=3,
        ),
        hardware=Hardware(nano_mr_sensor=NanoMRSensor()),
    )
    adapter = NanoMRQdyneAdapter(
        template=template,
        nominal_parameters={
            "signal_frequency_hz": 2.0e6,
            "field_amplitude_tesla": 20.0e-9,
        },
        sample_index=64,
        fixed_overhead_seconds=5.0e-6,
    )
    frequencies = np.linspace(1.992e6, 2.008e6, 17)
    posterior = ParticlePosterior(
        {
            "signal_frequency_hz": frequencies,
            "field_amplitude_tesla": np.full(frequencies.size, 20.0e-9),
        }
    )
    actions = CandidateDesignSpace(
        NanoMRQdyneDesign(float(reference))
        for reference in np.linspace(1.994e6, 2.006e6, 7)
    )
    likelihood = GaussianLikelihood(sigma=2.0e-3)
    session = make_adapter_session(
        adapter=adapter,
        likelihood=likelihood,
        posterior=posterior,
        design_space=actions,
        utility=ExpectedVarianceReduction(
            lambda parameters: parameters["signal_frequency_hz"],
            samples=24,
            scale=10.0e3,
        ),
        seed=args.seed,
    )

    truth = {
        "signal_frequency_hz": 2.003e6,
        "field_amplitude_tesla": 20.0e-9,
    }
    rng = np.random.default_rng(args.seed + 1)
    for step in range(args.steps):
        recommendation = session.ask()
        design = recommendation.best.design
        clean = adapter.simulate(truth, design)
        observed = likelihood.sample(clean, rng)
        session.tell(design, observed)
        mean = float(
            np.sum(
                session.posterior.weights
                * session.posterior.parameters["signal_frequency_hz"]
            )
        )
        print(
            f"step={step + 1} reference={design.reference_frequency_hz:.0f} Hz "
            f"cost={recommendation.best.cost_seconds:.6g} s "
            f"posterior_mean={mean:.3f} Hz"
        )


if __name__ == "__main__":
    main()
