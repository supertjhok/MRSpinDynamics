"""Run a config-ready Qdyne experiment through the unified facade."""
# Follow the example from physical inputs through simulation to reported observables.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spin_dynamics.experiment import (  # noqa: E402
    Experiment,
    Hardware,
    NanoMROpticalReadout,
    NanoMRQdyne,
    NanoMRSensor,
    load_config,
    save_config,
)


def default_experiment() -> Experiment:
    return Experiment(
        sequence=NanoMRQdyne(
            signal_frequency_hz=2.0e6,
            field_amplitude_tesla=20.0e-9,
            shot_count=1024,
            sensing_duration_seconds=2.0e-6,
            repetition_interval_seconds=20.0e-6,
            reference_frequency_hz=1.999e6,
            sensor_coherence_seconds=50.0e-6,
            sample_coherence_seconds=20.0e-3,
            seed=11,
        ),
        hardware=Hardware(
            nano_mr_sensor=NanoMRSensor(depth_nm=5.0),
            nano_mr_optical_readout=NanoMROpticalReadout(
                bright_count_rate_hz=2.0e6,
                readout_contrast=0.25,
            ),
        ),
    )


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--write-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    experiment = load_config(args.config) if args.config else default_experiment()
    if args.write_config:
        save_config(experiment, args.write_config)

    plan = experiment.plan()
    print(plan.report())
    record = experiment.run()
    result = record.result
    print(f"expected beat:  {result.expected_beat_frequency_hz:.6g} Hz")
    print(f"estimated beat: {result.estimated_beat_frequency_hz:.6g} Hz")
    print(f"sensor contrast: {result.sensor_contrast:.6g}")
    if result.optical_readout is not None:
        counts = result.optical_readout.sampled_counts
        print(f"detected photons: {int(counts.sum())}")
    if args.output:
        record.save(str(args.output))


if __name__ == "__main__":
    main()
