"""Plot absolute-phase-resolved refocusing pulse shapes.

This diagnostic solves the probe waveform for one nominal refocusing pulse at
several absolute RF phases.  The plotted rotating-frame shapes are the same
piecewise pulse segments used by the finite CPMG builders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from _mandal2015_absolute_phase import solve_refocusing_pulse_shape  # noqa: E402


def _phase_label(cycles: float) -> str:
    return f"{cycles:g} cycles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "untuned", "matched"], default="tuned")
    parser.add_argument("--numpts", type=int, default=17)
    parser.add_argument("--maxoffs", type=float, default=10.0)
    parser.add_argument(
        "--absolute-phases",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75],
        help="Absolute RF phases at pulse start, in cycles.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = load_matplotlib(headless=args.output is not None)
    shapes = [
        solve_refocusing_pulse_shape(
            probe=args.probe,
            absolute_phase_rad=2.0 * np.pi * phase_cycles,
            numpts=args.numpts,
            maxoffs=args.maxoffs,
        )
        for phase_cycles in args.absolute_phases
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for shape in shapes:
        time_us = 1.0e6 * shape.time_seconds
        drive = shape.drive
        label = _phase_label(shape.absolute_phase_cycles)
        phase = np.unwrap(np.angle(drive))
        threshold = 0.02 * float(np.max(np.abs(drive)))
        phase = np.where(np.abs(drive) > threshold, phase, np.nan)
        axes[0, 0].plot(time_us, np.real(drive), label=label)
        axes[0, 1].plot(time_us, np.imag(drive), label=label)
        axes[1, 0].plot(time_us, np.abs(drive), label=label)
        axes[1, 1].plot(time_us, phase, label=label)

    axes[0, 0].set_title("In-Phase Component")
    axes[0, 0].set_xlabel("Time (us)")
    axes[0, 0].set_ylabel("Re(B1)")
    axes[0, 0].legend(title="Abs. phase")

    axes[0, 1].set_title("Quadrature Component")
    axes[0, 1].set_xlabel("Time (us)")
    axes[0, 1].set_ylabel("Im(B1)")
    axes[0, 1].legend(title="Abs. phase")

    axes[1, 0].set_title("Amplitude")
    axes[1, 0].set_xlabel("Time (us)")
    axes[1, 0].set_ylabel("|B1|")
    axes[1, 0].legend(title="Abs. phase")

    axes[1, 1].set_title("Unwrapped Phase Above 2% Amplitude")
    axes[1, 1].set_xlabel("Time (us)")
    axes[1, 1].set_ylabel("Phase (rad)")
    axes[1, 1].legend(title="Abs. phase")

    fig.suptitle(
        "Absolute-phase-resolved refocusing pulse shapes "
        f"({args.probe} probe)"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
