"""Plot Qdyne, synchronized readout, and sensor-memory correlation.

Run ``python examples/plot_nano_mr_qdyne.py --help`` for clock, coherence,
signal, shot-count, and output options.
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
    ClockModel,
    HighResolutionBudget,
    QdyneProtocol,
    sensor_memory_correlation,
    simulate_qdyne,
    simulate_synchronized_readout,
    xy_sequence,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-khz", type=float, default=200.0)
    parser.add_argument("--beat-hz", type=float, default=7.0)
    parser.add_argument("--field-nt", type=float, default=8.0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--repetition-ms", type=float, default=1.0)
    parser.add_argument("--sensor-t2-us", type=float, default=100.0)
    parser.add_argument("--sample-t2-s", type=float, default=1.2)
    parser.add_argument("--diffusion-time-s", type=float, default=2.0)
    parser.add_argument("--memory-t2-s", type=float, default=3.0)
    parser.add_argument("--clock-instability-ppb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    for name in (
        "signal_khz",
        "field_nt",
        "repetition_ms",
        "sensor_t2_us",
        "sample_t2_s",
        "diffusion_time_s",
        "memory_t2_s",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.shots < 64:
        parser.error("--shots must be at least 64")
    if args.clock_instability_ppb < 0.0:
        parser.error("--clock-instability-ppb must be non-negative")

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    signal_frequency_hz = args.signal_khz * 1.0e3
    total_sensing_seconds = 8.0 / (2.0 * signal_frequency_hz)
    sequence = xy_sequence(8, 1, total_sensing_seconds)
    budget = HighResolutionBudget(
        sensor_coherence_seconds=args.sensor_t2_us * 1.0e-6,
        sensor_stretch_exponent=2.0,
        sample_coherence_seconds=args.sample_t2_s,
        diffusion_correlation_seconds=args.diffusion_time_s,
        memory_coherence_seconds=args.memory_t2_s,
    )
    protocol = QdyneProtocol(
        sequence,
        repetition_interval_seconds=args.repetition_ms * 1.0e-3,
        reference_frequency_hz=signal_frequency_hz - args.beat_hz,
        analysis_contrast=0.8,
        budget=budget,
        clock=ClockModel(
            interval_fractional_frequency_instability=(
                args.clock_instability_ppb * 1.0e-9
            ),
        ),
    )
    common = dict(
        signal_frequency_hz=signal_frequency_hz,
        field_amplitude_tesla=args.field_nt * 1.0e-9,
        sensor_gamma_rad_s_t=2.0 * np.pi * 28.0e9,
        shot_count=args.shots,
        signal_phase_rad=0.35,
        seed=args.seed,
    )
    qdyne = simulate_qdyne(protocol, **common)
    synchronized = simulate_synchronized_readout(protocol, **common)

    correlation_times = np.linspace(0.0, min(2.0, args.memory_t2_s * 1.5), 1200)
    memory = sensor_memory_correlation(
        correlation_times,
        frequency_hz=6.0,
        sensing_seconds=total_sensing_seconds,
        budget=budget,
        contrast=0.8,
        transfer_fidelity=0.92,
        retrieval_fidelity=0.90,
        phase_rad=0.35,
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)
    display = qdyne.nominal_times_seconds <= min(
        0.6,
        qdyne.nominal_times_seconds[-1],
    )
    axes[0, 0].plot(
        qdyne.nominal_times_seconds[display],
        qdyne.normalized_signal[display],
        label="Qdyne projection",
    )
    axes[0, 0].plot(
        synchronized.nominal_times_seconds[display],
        synchronized.complex_signal.real[display],
        "--",
        label="synchronized I",
    )
    axes[0, 0].plot(
        synchronized.nominal_times_seconds[display],
        synchronized.complex_signal.imag[display],
        ":",
        label="synchronized Q",
    )
    axes[0, 0].set(
        xlabel="Clocked acquisition time (s)",
        ylabel="Sensor response",
        title=f"{args.beat_hz:g} Hz beat sampled beyond one sensor block",
    )
    axes[0, 0].legend()

    q_mask = qdyne.baseband_frequencies_hz <= 20.0
    s_mask = np.abs(synchronized.baseband_frequencies_hz) <= 20.0
    q_scale = max(float(np.max(qdyne.spectrum[q_mask])), np.finfo(float).tiny)
    s_scale = max(
        float(np.max(synchronized.spectrum[s_mask])),
        np.finfo(float).tiny,
    )
    axes[0, 1].plot(
        qdyne.baseband_frequencies_hz[q_mask],
        qdyne.spectrum[q_mask] / q_scale,
        label="Qdyne: unsigned",
    )
    axes[0, 1].plot(
        synchronized.baseband_frequencies_hz[s_mask],
        synchronized.spectrum[s_mask] / s_scale,
        label="synchronized: signed",
    )
    axes[0, 1].axvline(args.beat_hz, color="0.45", ls=":", lw=1.0)
    axes[0, 1].set(
        xlabel="Baseband frequency (Hz)",
        ylabel="Normalized amplitude",
        title=(
            f"Fourier bin {1.0 / (args.shots * args.repetition_ms * 1e-3):.3f} Hz"
        ),
    )
    axes[0, 1].legend()

    envelope_times = np.linspace(0.0, min(4.0, qdyne.nominal_times_seconds[-1]), 500)
    sample_envelope = np.exp(-envelope_times / args.sample_t2_s)
    diffusion_envelope = np.exp(-envelope_times / args.diffusion_time_s)
    memory_envelope = np.exp(-envelope_times / args.memory_t2_s)
    axes[1, 0].plot(envelope_times, sample_envelope, label=r"sample $T_2$")
    axes[1, 0].plot(envelope_times, diffusion_envelope, label="diffusion")
    axes[1, 0].plot(envelope_times, memory_envelope, label=r"memory $T_2$")
    axes[1, 0].plot(
        envelope_times,
        sample_envelope * diffusion_envelope,
        color="black",
        lw=1.7,
        label="Qdyne intershot product",
    )
    axes[1, 0].set(
        xlabel="Evolution time (s)",
        ylabel="Envelope",
        ylim=(-0.03, 1.03),
        title=(
            "Independent resolution limits\n"
            f"within-block sensor contrast {qdyne.sensor_contrast:.3f}"
        ),
    )
    axes[1, 0].legend()

    axes[1, 1].plot(
        memory.correlation_times_seconds,
        memory.signal,
        label="memory-assisted correlation",
    )
    axes[1, 1].plot(
        memory.correlation_times_seconds,
        0.8
        * memory.transfer_retrieval_fidelity
        * memory.sensor_contrast
        * memory.sample_envelope
        * memory.diffusion_envelope
        * memory.memory_envelope,
        "--",
        label="reported total envelope",
    )
    axes[1, 1].set(
        xlabel="Memory storage / correlation time (s)",
        ylabel="Correlation contrast",
        title="Sensor-memory correlation",
    )
    axes[1, 1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
