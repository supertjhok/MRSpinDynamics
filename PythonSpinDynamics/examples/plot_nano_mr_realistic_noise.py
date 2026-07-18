"""Plot correlated spin noise and time-resolved NV/SPAD optical readout.

The example contrasts exponential surface noise with a long-tailed diffusing
target, displays the resulting scan covariance, and samples a five-level NV
optical cycle followed by SPAD efficiency, background, dead time, afterpulsing,
and timing jitter. Run
``python examples/plot_nano_mr_realistic_noise.py --help``.
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
    CorrelatedFieldNoiseModel,
    FieldNoiseComponent,
    OpticalCycleModel,
    SPADDetectorModel,
    effective_sample_size,
    optical_contrast_trace,
    sample_time_resolved_optical_readout,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=8000)
    parser.add_argument("--readout-ns", type=float, default=400.0)
    parser.add_argument("--detection-efficiency", type=float, default=0.12)
    parser.add_argument("--afterpulse-probability", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.shots < 100:
        parser.error("--shots must be at least 100")
    if args.readout_ns <= 0.0:
        parser.error("--readout-ns must be positive")
    if not 0.0 <= args.detection_efficiency <= 1.0:
        parser.error("--detection-efficiency must lie in [0, 1]")
    if not 0.0 <= args.afterpulse_probability <= 1.0:
        parser.error("--afterpulse-probability must lie in [0, 1]")

    surface = FieldNoiseComponent(
        "surface electron bath",
        rms_field_tesla=35.0e-9,
        correlation_time_seconds=5.0e-6,
        envelope="exponential",
    )
    target = FieldNoiseComponent(
        "diffusing nuclear target",
        rms_field_tesla=12.0e-9,
        correlation_time_seconds=25.0e-6,
        envelope="power_law",
        power_law_exponent=1.5,
        spatial_correlation_length_nm=5.0,
    )
    lags = np.geomspace(20.0e-9, 2.0e-3, 500)

    x = np.tile(np.linspace(-8.0, 8.0, 10), 8)
    y = np.repeat(np.linspace(-6.0, 6.0, 8), 10)
    positions = np.column_stack((x, y, np.full(x.size, -8.0)))
    times = np.arange(x.size) * 2.0e-6
    noise = CorrelatedFieldNoiseModel((surface, target))
    covariance = noise.covariance(times, positions_lab_nm=positions)
    standard = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(standard, standard)

    optical = OpticalCycleModel(
        bright_ionization_rate_hz=0.3e6,
        dark_ionization_rate_hz=0.6e6,
        recombination_to_bright_rate_hz=0.2e6,
        recombination_to_dark_rate_hz=0.05e6,
    )
    detector = SPADDetectorModel(
        detection_efficiency=args.detection_efficiency,
        background_count_rate_hz=25.0e3,
        dark_count_rate_hz=100.0,
        dead_time_seconds=35.0e-9,
        afterpulse_probability=args.afterpulse_probability,
        afterpulse_time_seconds=120.0e-9,
        timing_jitter_seconds=0.35e-9,
    )
    readout_seconds = args.readout_ns * 1.0e-9
    trace_times = np.linspace(0.0, readout_seconds, 300)
    bright_rate, dark_rate, contrast = optical_contrast_trace(
        optical,
        trace_times,
    )
    bright_counts = sample_time_resolved_optical_readout(
        optical,
        detector,
        1.0,
        repetitions=args.shots,
        readout_seconds=readout_seconds,
        seed=args.seed,
    )
    dark_counts = sample_time_resolved_optical_readout(
        optical,
        detector,
        0.0,
        repetitions=args.shots,
        readout_seconds=readout_seconds,
        seed=args.seed + 1,
    )
    mixed_counts = sample_time_resolved_optical_readout(
        optical,
        detector,
        0.5,
        repetitions=args.shots,
        readout_seconds=readout_seconds,
        seed=args.seed + 2,
    )

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), constrained_layout=True)

    axes[0, 0].loglog(
        lags * 1.0e6,
        np.abs(surface.temporal_correlation(lags)),
        label="surface: exponential",
    )
    axes[0, 0].loglog(
        lags * 1.0e6,
        np.abs(target.temporal_correlation(lags)),
        label=r"target: $(1+t/\tau_D)^{-3/2}$",
    )
    axes[0, 0].set(
        xlabel=r"Lag $\tau$ ($\mu$s)",
        ylabel=r"$|C(\tau)/C(0)|$",
        title="Distinct sensor and target correlations",
        ylim=(1.0e-5, 1.05),
    )
    axes[0, 0].legend()

    image = axes[0, 1].imshow(
        correlation,
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )
    axes[0, 1].set(
        xlabel="scan sample",
        ylabel="scan sample",
        title=(
            "Temporal/spatial scan correlation\n"
            f"effective samples = {effective_sample_size(covariance):.1f}"
            f" of {times.size}"
        ),
    )
    fig.colorbar(image, ax=axes[0, 1], label="correlation")

    axes[1, 0].plot(
        trace_times * 1.0e9,
        bright_rate * detector.detection_efficiency * 1.0e-6,
        label=r"initial $m_s=0$",
    )
    axes[1, 0].plot(
        trace_times * 1.0e9,
        dark_rate * detector.detection_efficiency * 1.0e-6,
        label=r"initial $m_s=\pm1$",
    )
    axes[1, 0].set(
        xlabel="readout time (ns)",
        ylabel="detected signal rate (Mcps)",
        title=f"Rate-equation optical cycling; peak contrast {np.max(contrast):.2f}",
    )
    axes[1, 0].legend()

    maximum = int(
        max(
            np.max(bright_counts.detected_counts),
            np.max(dark_counts.detected_counts),
            np.max(mixed_counts.detected_counts),
        )
    )
    bins = np.arange(-0.5, maximum + 1.5)
    axes[1, 1].hist(
        bright_counts.detected_counts,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        label=rf"bright, $F={bright_counts.fano_factor:.2f}$",
    )
    axes[1, 1].hist(
        dark_counts.detected_counts,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        label=rf"dark, $F={dark_counts.fano_factor:.2f}$",
    )
    axes[1, 1].hist(
        mixed_counts.detected_counts,
        bins=bins,
        density=True,
        histtype="step",
        lw=1.8,
        label=rf"50/50 mixture, $F={mixed_counts.fano_factor:.2f}$",
    )
    axes[1, 1].set(
        xlabel="detected photons per shot",
        ylabel="probability",
        title=(
            f"NV optical path + SPAD transfer ({args.shots:,} shots)\n"
            f"dead time 35 ns, afterpulse {args.afterpulse_probability:.1%}"
        ),
        xlim=(-0.5, maximum + 0.5),
    )
    axes[1, 1].set_xticks(np.arange(maximum + 1))
    axes[1, 1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
