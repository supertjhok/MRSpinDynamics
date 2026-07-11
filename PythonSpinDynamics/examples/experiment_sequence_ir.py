"""Plan, execute, visualize, and save a native or Pulseq SequenceIR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib
from plot_sequence_timeline import make_demo_spin_echo

add_src_to_path()

from spin_dynamics.experiment import (
    Experiment,
    Sample,
    SequenceDomain,
    SequenceIRExecution,
)
from spin_dynamics.sequences import read_pulseq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        nargs="?",
        type=Path,
        help="Optional Pulseq .seq file; omit for the built-in spin echo.",
    )
    parser.add_argument("--system-frequency-hz", type=float, default=None)
    parser.add_argument("--points", type=int, default=17, help="Points per x/z axis.")
    parser.add_argument("--extent-m", type=float, default=0.02)
    parser.add_argument("--diffusion-m2-s", type=float, default=0.0)
    parser.add_argument("--walkers-per-cell", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--timeline-output",
        type=Path,
        default=None,
        help="Optional sequence timeline image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sequence_ir_run.npz"),
        help="Provenance-bearing result archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.points < 2:
        raise ValueError("--points must be at least 2")
    sequence = read_pulseq(args.sequence) if args.sequence else make_demo_spin_echo()
    axis = np.linspace(-args.extent_m / 2.0, args.extent_m / 2.0, args.points)
    experiment = Experiment(
        sequence=SequenceIRExecution(
            ir=sequence,
            system_frequency_hz=args.system_frequency_hz,
            walkers_per_cell=args.walkers_per_cell,
            seed=args.seed,
        ),
        sample=Sample(
            t1_seconds=1.0,
            t2_seconds=0.1,
            diffusion_coefficient=args.diffusion_m2_s,
            sequence_domain=SequenceDomain(
                axes=(axis, axis),
                density=np.ones((args.points, args.points)),
                gradient_channels=("x", "z"),
            ),
        ),
    )
    plan = experiment.plan()
    print(plan.report())
    record = experiment.run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record.save(str(args.output))
    print(f"saved: {args.output}")
    print(f"ADC samples: {record.result.signal.size}")

    if args.timeline_output is not None:
        load_matplotlib()
        figure, _axes = sequence.plot(
            system_frequency_hz=args.system_frequency_hz,
            show_blocks=True,
        )
        args.timeline_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.timeline_output, dpi=150)
        print(f"saved: {args.timeline_output}")


if __name__ == "__main__":
    main()
