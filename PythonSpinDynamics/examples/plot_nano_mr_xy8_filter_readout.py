"""Plot an XY8 nano-MR filter and its photon-count readout.

Run ``python examples/plot_nano_mr_xy8_filter_readout.py --help`` for
adjustable sequence, pulse, and output options.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    OpticalCycleModel,
    OpticalReadoutModel,
    SPADDetectorModel,
    dephasing_filter_function,
    sample_optical_readout,
    xy_sequence,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-us", type=float, default=80.0)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--pulse-width-ns", type=float, default=100.0)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument(
        "--rate-readout",
        action="store_true",
        help="overlay the five-level optical-cycle plus SPAD mean",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    duration = args.duration_us * 1.0e-6
    pulse_width = args.pulse_width_ns * 1.0e-9
    ideal = xy_sequence(8, args.repetitions, duration)
    finite = xy_sequence(
        8,
        args.repetitions,
        duration,
        pulse_duration_seconds=pulse_width,
    )

    center_hz = len(ideal.electron_pulses) / (2.0 * duration)
    frequencies_hz = np.linspace(0.15, 1.85, 600) * center_hz
    angular_frequencies = 2.0 * np.pi * frequencies_hz
    ideal_filter = dephasing_filter_function(ideal, angular_frequencies)
    finite_filter = dephasing_filter_function(
        finite,
        angular_frequencies,
        pulse_model="finite",
    )

    phase = np.linspace(-np.pi, np.pi, 81)
    bright_probability = 0.5 * (1.0 + np.cos(phase))
    readout = OpticalReadoutModel(
        initialization_fidelity=0.95,
        initialization_seconds=2.0e-6,
        bright_count_rate_hz=250.0e3,
        readout_contrast=0.25,
        readout_seconds=400.0e-9,
        background_count_rate_hz=10.0e3,
        dead_time_seconds=1.0e-6,
    )
    counts = sample_optical_readout(
        readout,
        bright_probability,
        repetitions=args.shots,
        sensing_seconds=duration,
        seed=args.seed,
    )
    rate_model_counts = None
    if args.rate_readout:
        optical_cycle = OpticalCycleModel()
        bright_emitted = optical_cycle.expected_emitted_photons(
            1.0,
            readout_seconds=readout.readout_seconds,
        )
        efficiency = (
            readout.bright_count_rate_hz
            * readout.readout_seconds
            / bright_emitted
        )
        spad = SPADDetectorModel(
            detection_efficiency=efficiency,
            background_count_rate_hz=readout.background_count_rate_hz,
            dark_count_rate_hz=0.0,
            dead_time_seconds=35.0e-9,
            afterpulse_probability=0.01,
            afterpulse_time_seconds=120.0e-9,
        )
        dark_emitted = optical_cycle.expected_emitted_photons(
            0.0,
            readout_seconds=readout.readout_seconds,
        )
        emitted = (
            bright_probability * bright_emitted
            + (1.0 - bright_probability) * dark_emitted
        )
        rate_model_counts = args.shots * np.array(
            [
                spad.approximate_expected_counts(
                    value,
                    readout_seconds=readout.readout_seconds,
                )
                for value in emitted
            ]
        )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].plot(
        frequencies_hz * 1.0e-3,
        ideal_filter,
        label="instantaneous pulses",
    )
    axes[0].plot(
        frequencies_hz * 1.0e-3,
        finite_filter,
        "--",
        label=f"{args.pulse_width_ns:g} ns pulses",
    )
    axes[0].axvline(center_hz * 1.0e-3, color="0.5", lw=1.0, ls=":")
    axes[0].set(
        xlabel="AC-field frequency (kHz)",
        ylabel=r"Filter $\omega^2 |Y(\omega)|^2$",
        title=f"XY8-{args.repetitions} sensing filter",
    )
    axes[0].legend()

    axes[1].plot(phase, counts.expected_counts, label="Poisson mean")
    axes[1].scatter(
        phase,
        counts.sampled_counts,
        s=12,
        alpha=0.65,
        label=f"sampled ({args.shots:,} shots)",
    )
    if rate_model_counts is not None:
        axes[1].plot(
            phase,
            rate_model_counts,
            "--",
            label="five-level optical cycle + SPAD mean",
        )
    axes[1].set(
        xlabel="Accumulated sensor phase (rad)",
        ylabel="Detected photons",
        title="Contrast-limited optical readout",
    )
    axes[1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
