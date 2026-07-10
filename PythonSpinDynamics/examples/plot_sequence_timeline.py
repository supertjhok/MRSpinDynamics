"""Visualize a Pulseq file or a built-in spin-echo sequence timeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.sequences import (
    ADCEvent,
    GradientWaveform,
    RFPulse,
    SequenceBlock,
    SequenceIR,
    read_pulseq,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        nargs="?",
        type=Path,
        help="Optional Pulseq .seq file; omit for the built-in spin echo.",
    )
    parser.add_argument(
        "--system-frequency-hz",
        type=float,
        default=None,
        help="Larmor frequency needed when the file uses PPM offsets.",
    )
    parser.add_argument(
        "--time-unit",
        choices=("auto", "s", "ms", "us", "ns"),
        default="auto",
        help="Time-axis unit.",
    )
    parser.add_argument(
        "--hide-blocks",
        action="store_true",
        help="Hide block boundaries and labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path.",
    )
    return parser.parse_args()


def make_demo_spin_echo() -> SequenceIR:
    """Return a compact slice-selective spin-echo timeline."""

    rf_dwell = 20e-6
    gradient_dwell = 20e-6
    excitation = np.hanning(100)
    refocusing = np.hanning(160)
    slice_gradient = np.full(excitation.size, 12_000.0)
    readout_gradient = np.full(200, 8_000.0)
    return SequenceIR(
        blocks=(
            SequenceBlock(
                duration_seconds=2e-3,
                rf=RFPulse(
                    220.0 * excitation,
                    dwell_seconds=rf_dwell,
                    use="excitation",
                ),
                gradients=(
                    None,
                    None,
                    GradientWaveform(slice_gradient, gradient_dwell),
                ),
                label="excitation",
            ),
            SequenceBlock(duration_seconds=8e-3, label="TE/2"),
            SequenceBlock(
                duration_seconds=3.2e-3,
                rf=RFPulse(
                    420.0j * refocusing,
                    dwell_seconds=rf_dwell,
                    use="refocusing",
                ),
                label="refocusing",
            ),
            SequenceBlock(duration_seconds=8e-3, label="TE/2"),
            SequenceBlock(
                duration_seconds=4e-3,
                gradients=(
                    GradientWaveform(readout_gradient, gradient_dwell),
                    None,
                    None,
                ),
                adc=ADCEvent(num_samples=100, dwell_seconds=40e-6),
                label="readout",
            ),
        ),
        definitions={"Name": "demo_spin_echo"},
    )


def main() -> None:
    args = _parse_args()
    load_matplotlib()
    sequence = read_pulseq(args.sequence) if args.sequence else make_demo_spin_echo()
    figure, _axes = sequence.plot(
        system_frequency_hz=args.system_frequency_hz,
        time_unit=args.time_unit,
        show_blocks=not args.hide_blocks,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
