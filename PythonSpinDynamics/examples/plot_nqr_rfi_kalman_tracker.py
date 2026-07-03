"""Plot reference-free Kalman tracking of a drifting AM carrier.

This Phase-5 example targets ``kalman_harmonic_canceller``. Unlike the reference
cancellers, it uses *no* reference channel: it exploits the narrowband structure
of an amplitude-modulated broadcast carrier, tracking its slowly drifting complex
amplitude with a Kalman filter and subtracting it. Measurement updates are frozen
during the SLSE echo windows, so the tracker coasts through the NQR response
instead of absorbing it.

The carrier sits close to the NQR line (the in-band case), yet because the
tracker models only the carrier frequency it removes the interferer while leaving
the echo train. The panels show the time record, the tracked envelope against the
true amplitude modulation, the spectrum before and after, and the recovered-NQR
quality.

Run with ``--output nqr_rfi_kalman_tracker.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    kalman_harmonic_canceller,
    rfi_suppression_db,
    signal_bias,
)
from spin_dynamics.nqr import slse_acquisition_mask, slse_sequence  # noqa: E402


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _synthetic_echo_train(
    times: np.ndarray, echo_spacing: float, num_echoes: int, detection_hz: float
) -> np.ndarray:
    signal = np.zeros(times.size, dtype=np.float64)
    width = 0.14 * echo_spacing
    for idx in range(num_echoes):
        center = (idx + 0.72) * echo_spacing
        envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
        decay = np.exp(-(idx + 1) / max(1.0, 0.5 * num_echoes))
        signal += decay * envelope * np.cos(2.0 * np.pi * detection_hz * (times - center))
    return signal


def _build_record(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    fs = args.sample_rate_khz * 1e3
    echo_spacing = args.echo_spacing_ms * 1e-3
    sequence = slse_sequence(
        "x",
        pulse_duration_seconds=args.pulse_us * 1e-6,
        nutation_hz=10e3,
        echo_spacing_seconds=echo_spacing,
        num_echoes=args.num_echoes,
    )
    mask = slse_acquisition_mask(
        sequence,
        fs,
        ringdown_seconds=args.ringdown_us * 1e-6,
        pre_baseline_seconds=args.pre_baseline_ms * 1e-3,
        post_baseline_seconds=args.post_baseline_ms * 1e-3,
    )
    times = mask.times
    n = mask.num_samples
    receive = mask.receive_mask

    sequence_times = times - args.pre_baseline_ms * 1e-3
    clean = _synthetic_echo_train(
        sequence_times, echo_spacing, args.num_echoes, args.nqr_detect_khz * 1e3
    )
    clean[~receive] = 0.0
    peak = float(np.max(np.abs(clean))) or 1.0
    echo_mask = receive & (np.abs(clean) > args.echo_threshold * peak)
    clean = clean * (args.carrier_rms / args.rfi_to_signal) / (_rms(clean[echo_mask]) or 1.0)

    # Drifting amplitude-modulated carrier (a broadcast-AM model), reference-free.
    drift = np.cumsum(rng.normal(scale=args.drift_std, size=n))
    drift = np.convolve(drift, np.ones(args.drift_width) / args.drift_width, mode="same")
    envelope = 1.0 + args.modulation_depth * np.sin(2.0 * np.pi * args.modulation_hz * times)
    envelope = envelope + args.drift_depth * drift
    carrier = envelope * np.cos(2.0 * np.pi * args.carrier_khz * 1e3 * times + args.carrier_phase)
    carrier = carrier * (args.carrier_rms / (_rms(carrier) or 1.0))

    primary = clean + carrier + args.primary_noise * rng.normal(size=n)
    # Track only where no NQR echo is expected; coast through the echoes.
    update = receive & ~echo_mask

    return {
        "times": times,
        "sample_rate_hz": fs,
        "clean": clean,
        "carrier": carrier,
        "true_envelope": np.abs(envelope) * (args.carrier_rms / (_rms(carrier) or 1.0)),
        "primary": primary,
        "receive_mask": receive,
        "echo_mask": echo_mask,
        "update_mask": update,
    }


def _run(record: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    result = kalman_harmonic_canceller(
        record["primary"],
        [args.carrier_khz * 1e3],
        float(record["sample_rate_hz"]),
        update_mask=record["update_mask"],
        process_std=args.process_std,
        measurement_std=args.measurement_std,
    )
    receive = record["receive_mask"]
    echo = record["echo_mask"]
    clean = record["clean"]
    primary = record["primary"]
    suppression = rfi_suppression_db(
        primary, result.cleaned, receive & ~echo, clean_signal=clean
    ).suppression_db
    bias = signal_bias(clean, result.cleaned, echo).amplitude_bias
    raw_bias = signal_bias(clean, primary, echo).amplitude_bias
    return {
        "result": result,
        "suppression": suppression,
        "bias": bias,
        "raw_bias": raw_bias,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate-khz", type=float, default=50.0)
    parser.add_argument("--num-echoes", type=int, default=14)
    parser.add_argument("--echo-spacing-ms", type=float, default=2.0)
    parser.add_argument("--pulse-us", type=float, default=180.0)
    parser.add_argument("--ringdown-us", type=float, default=120.0)
    parser.add_argument("--pre-baseline-ms", type=float, default=3.0)
    parser.add_argument("--post-baseline-ms", type=float, default=3.0)
    parser.add_argument("--echo-threshold", type=float, default=0.06)
    parser.add_argument("--nqr-detect-khz", type=float, default=6.5)
    parser.add_argument(
        "--carrier-khz",
        type=float,
        default=7.0,
        help="AM carrier frequency (in-band, near the NQR line).",
    )
    parser.add_argument("--carrier-phase", type=float, default=0.6)
    parser.add_argument("--carrier-rms", type=float, default=1.0)
    parser.add_argument("--rfi-to-signal", type=float, default=12.0)
    parser.add_argument("--modulation-hz", type=float, default=60.0)
    parser.add_argument("--modulation-depth", type=float, default=0.4)
    parser.add_argument("--drift-std", type=float, default=0.02)
    parser.add_argument("--drift-width", type=int, default=151)
    parser.add_argument("--drift-depth", type=float, default=0.6)
    parser.add_argument("--process-std", type=float, default=6e-3)
    parser.add_argument("--measurement-std", type=float, default=0.2)
    parser.add_argument("--primary-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    record = _build_record(args)
    results = _run(record, args)
    _plot(plt, record, results, args)
    _print_summary(results)


def _plot(plt, record: dict[str, object], results: dict[str, object], args: argparse.Namespace) -> None:
    times_ms = record["times"] * 1e3
    clean = record["clean"]
    primary = record["primary"]
    receive = record["receive_mask"]
    result = results["result"]
    cleaned = np.real(result.cleaned)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)

    # (0,0) Time record around an echo.
    center = int(np.argmax(np.abs(clean)))
    fs = float(record["sample_rate_hz"])
    half = max(1, int(1.5e-3 * fs))
    sl = slice(max(0, center - half), min(times_ms.size, center + half))
    axes[0, 0].plot(times_ms[sl], primary[sl], color="0.7", lw=0.6, label="contaminated")
    axes[0, 0].plot(times_ms[sl], cleaned[sl], color="C0", lw=0.9, label="Kalman cleaned")
    axes[0, 0].plot(times_ms[sl], clean[sl], color="C2", lw=1.1, label="clean NQR")
    axes[0, 0].set_title("Time Record (reference-free)")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("primary channel")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    # (0,1) Tracked amplitude vs the true modulation envelope.
    tracked = np.abs(result.coefficient_history[:, 0])
    axes[0, 1].plot(times_ms, record["true_envelope"], color="0.4", lw=1.2, label="true envelope")
    axes[0, 1].plot(times_ms, tracked, color="C0", lw=1.0, ls="--", label="tracked |a(t)|")
    axes[0, 1].set_title("Kalman-Tracked Carrier Amplitude")
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].set_ylabel("amplitude")
    axes[0, 1].legend(fontsize=8)

    # (1,0) Spectrum before and after (receive samples).
    gated_primary = np.where(receive, primary, 0.0)
    gated_cleaned = np.where(receive, cleaned, 0.0)
    freqs = np.fft.rfftfreq(primary.size, d=1.0 / fs) / 1e3
    spec_before = np.abs(np.fft.rfft(gated_primary))
    spec_after = np.abs(np.fft.rfft(gated_cleaned))
    axes[1, 0].semilogy(freqs, spec_before + 1e-9, color="0.6", label="before")
    axes[1, 0].semilogy(freqs, spec_after + 1e-9, color="C0", label="after")
    axes[1, 0].axvline(args.carrier_khz, color="C3", ls=":", lw=0.9, label="carrier")
    axes[1, 0].axvline(args.nqr_detect_khz, color="C2", ls=":", lw=0.9, label="NQR line")
    axes[1, 0].set_xlim(args.nqr_detect_khz - 3.0, args.carrier_khz + 3.0)
    axes[1, 0].set_title("Spectrum Before/After")
    axes[1, 0].set_xlabel("frequency (kHz)")
    axes[1, 0].set_ylabel("FFT amplitude")
    axes[1, 0].legend(fontsize=8)

    # (1,1) Recovered-NQR quality.
    axes[1, 1].bar(
        ["raw", "Kalman"],
        [abs(results["raw_bias"]), abs(results["bias"])],
        color=["0.6", "C0"],
        width=0.55,
    )
    axes[1, 1].set_ylabel("|NQR amplitude bias|")
    axes[1, 1].set_title(f"NQR Bias (suppression {results['suppression']:.0f} dB)")

    fig.suptitle("Reference-Free Kalman Tracking of a Drifting AM Carrier")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(results: dict[str, object]) -> None:
    print("Reference-free Kalman AM-carrier tracking")
    print(f"coherent suppression: {results['suppression']:.1f} dB")
    print(f"NQR amplitude bias: raw={results['raw_bias']:+.3f} -> Kalman={results['bias']:+.3f}")


if __name__ == "__main__":
    main()
