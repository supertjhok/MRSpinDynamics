"""Plot active feedforward RFI cancellation vs. digital-cancellation-too-late.

This Phase-5 example demonstrates the compensation-coil model in
``spin_dynamics.interference.active``. A strong in-band interferer drives the
primary NQR receiver past its ADC full scale. Two mitigations are compared:

* **digital (too late)** -- the ADC clips the raw primary first, then a fitted
  reference model is subtracted. The clipped peaks are unrecoverable, so residual
  RFI and NQR distortion remain.
* **active feedforward** -- a calibrated linear model commands a compensation
  coil that subtracts the RFI *before* the ADC, so the digitiser only ever sees
  the small residual and never clips.

The last two panels show the physical limits of feedforward: a causal actuator
latency bounds the cancellation bandwidth, and a finite compensation-coil drive
range bounds how much of a large interferer can be removed.

Run with ``--output nqr_rfi_active_feedforward.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    CompensationActuator,
    ReferenceCoil,
    UniformPlaneWaveSource,
    am_carrier_waveform,
    coil_voltage,
    feedforward_cancel,
    fit_gated_ridge_fir,
    reference_matrix,
    rfi_suppression_db,
    saturation_diagnostics,
    signal_bias,
    tone_waveform,
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


def _coherent_sources(n: int, sample_rate_hz: float):
    waveforms = [
        am_carrier_waveform(
            n, sample_rate_hz, carrier_hz=7000.0, modulation_hz=45.0, modulation_depth=0.6
        ),
        tone_waveform(n, sample_rate_hz, frequency_hz=11000.0),
        am_carrier_waveform(
            n, sample_rate_hz, carrier_hz=5000.0, modulation_hz=80.0, modulation_depth=0.4
        ),
    ]
    polarizations = [(0.80, 0.50, 0.30), (0.20, 0.90, 0.40), (0.50, 0.30, 0.90)]
    return [
        UniformPlaneWaveSource(polarization=pol, waveform=wave)
        for pol, wave in zip(polarizations, waveforms)
    ]


def _build_record(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    sample_rate_hz = args.sample_rate_khz * 1e3
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
        sample_rate_hz,
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

    sources = _coherent_sources(n, sample_rate_hz)
    coils = [
        ReferenceCoil(pickup_vector=(1.0, 0.0, 0.0), label="ref-x"),
        ReferenceCoil(pickup_vector=(0.0, 1.0, 0.0), label="ref-y"),
        ReferenceCoil(pickup_vector=(0.0, 0.0, 1.0), label="ref-z"),
    ]
    primary_coil = ReferenceCoil(pickup_vector=(0.60, 0.50, 0.62), label="primary")

    primary_coh = coil_voltage(primary_coil, sources)
    refs_coh = reference_matrix(coils, sources)
    scale = args.rfi_rms / (_rms(primary_coh[receive]) or 1.0)
    primary_coh = primary_coh * scale
    refs_coh = refs_coh * scale
    clean = clean * (args.rfi_rms / args.rfi_to_signal) / (_rms(clean[echo_mask]) or 1.0)

    references = refs_coh + args.reference_noise * rng.normal(size=refs_coh.shape)
    primary = clean + primary_coh + args.primary_noise * rng.normal(size=n)

    return {
        "times": times,
        "sample_rate_hz": sample_rate_hz,
        "clean": clean,
        "primary": primary,
        "primary_coh": primary_coh,
        "refs_coh": refs_coh,
        "references": references,
        "receive_mask": receive,
        "echo_mask": echo_mask,
    }


def _run(record: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    primary = record["primary"]
    references = record["references"]
    receive = record["receive_mask"]
    fs = float(record["sample_rate_hz"])
    sat = args.adc_saturation

    # The feedforward transfer is calibrated: fit on the noise-free coherent
    # system so the commanded compensation reproduces the primary-coil RFI.
    model = fit_gated_ridge_fir(
        record["primary_coh"], record["refs_coh"], receive, taps=args.taps, ridge=args.ridge
    )

    # Digital cancellation applied too late: the ADC clips the raw primary first.
    clipped = np.clip(primary, -sat, sat)
    digital = model.apply(clipped, references)

    latency = int(round(args.latency_us * 1e-6 * fs))
    actuator = CompensationActuator(
        latency_samples=latency,
        gain=args.actuator_gain,
        max_field=args.max_field if args.max_field > 0 else None,
        noise_sigma=args.actuator_noise,
    )
    active = feedforward_cancel(
        primary, references, model, actuator, fs, adc_saturation=sat, seed=args.seed + 3
    )

    # Latency sweep over integer sample delays (the actuator resolves whole
    # samples), converted to microseconds for display.
    max_latency_samples = max(1, int(round(args.max_latency_us * 1e-6 * fs)))
    sample_counts = np.unique(
        np.round(np.linspace(0, max_latency_samples, 24)).astype(int)
    )
    latencies_us = sample_counts / fs * 1e6
    latency_supp = []
    for count in sample_counts:
        act = CompensationActuator(latency_samples=int(count))
        res = feedforward_cancel(primary, references, model, act, fs)
        latency_supp.append(
            rfi_suppression_db(primary, res.analog_residual, receive).suppression_db
        )

    # Drive-range sweep: coherent suppression vs. compensation-coil headroom.
    command_peak = float(np.max(np.abs(model.predict(references))))
    drive_ratios = np.linspace(0.3, 1.6, 24)
    drive_supp = []
    for ratio in drive_ratios:
        act = CompensationActuator(max_field=ratio * command_peak)
        res = feedforward_cancel(primary, references, model, act, fs)
        drive_supp.append(
            rfi_suppression_db(primary, res.analog_residual, receive).suppression_db
        )

    return {
        "model": model,
        "digital": digital,
        "active": active,
        "clipped": clipped,
        "adc_saturation": sat,
        "latencies_us": latencies_us,
        "latency_supp": np.asarray(latency_supp),
        "drive_ratios": drive_ratios,
        "drive_supp": np.asarray(drive_supp),
        "raw_saturation": saturation_diagnostics(primary, sat),
        "active_saturation": saturation_diagnostics(active.analog_residual, sat),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate-khz", type=float, default=250.0)
    parser.add_argument("--num-echoes", type=int, default=14)
    parser.add_argument("--echo-spacing-ms", type=float, default=2.0)
    parser.add_argument("--pulse-us", type=float, default=180.0)
    parser.add_argument("--ringdown-us", type=float, default=120.0)
    parser.add_argument("--pre-baseline-ms", type=float, default=3.0)
    parser.add_argument("--post-baseline-ms", type=float, default=3.0)
    parser.add_argument("--echo-threshold", type=float, default=0.06)
    parser.add_argument("--nqr-detect-khz", type=float, default=6.5)
    parser.add_argument("--taps", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--rfi-rms", type=float, default=1.0)
    parser.add_argument("--rfi-to-signal", type=float, default=15.0)
    parser.add_argument(
        "--adc-saturation",
        type=float,
        default=0.9,
        help="ADC full-scale; the RFI peak is several times this, so it clips.",
    )
    parser.add_argument("--latency-us", type=float, default=1.0)
    parser.add_argument("--max-latency-us", type=float, default=80.0)
    parser.add_argument("--actuator-gain", type=float, default=1.0)
    parser.add_argument("--actuator-noise", type=float, default=0.003)
    parser.add_argument(
        "--max-field",
        type=float,
        default=0.0,
        help="Compensation-coil drive limit (0 = unlimited).",
    )
    parser.add_argument("--reference-noise", type=float, default=0.01)
    parser.add_argument("--primary-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    record = _build_record(args)
    results = _run(record, args)
    _plot(plt, record, results, args)
    _print_summary(record, results)


def _plot(plt, record: dict[str, object], results: dict[str, object], args: argparse.Namespace) -> None:
    times_ms = record["times"] * 1e3
    clean = record["clean"]
    primary = record["primary"]
    echo = record["echo_mask"]
    sat = results["adc_saturation"]
    digital = np.real(results["digital"].cleaned)
    active = results["active"]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)

    # (0,0) Analog picture: raw RFI runs off the ADC scale; active residual fits.
    axes[0, 0].plot(times_ms, primary, color="0.7", lw=0.6, label="raw primary")
    axes[0, 0].plot(times_ms, active.analog_residual, color="C0", lw=0.7, label="active residual")
    axes[0, 0].axhline(sat, color="C3", ls="--", lw=0.8, label="ADC full scale")
    axes[0, 0].axhline(-sat, color="C3", ls="--", lw=0.8)
    axes[0, 0].set_title("Analog Stage: Compensation Before the ADC")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("primary channel")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    # Zoom around the strongest echo for the two digitised outputs.
    center = int(np.argmax(np.abs(clean)))
    fs = float(record["sample_rate_hz"])
    half = max(1, int(1.2e-3 * fs))
    sl = slice(max(0, center - half), min(times_ms.size, center + half))
    baseline_scale = 6.0 * (_rms(clean[echo]) or 1.0)

    # (0,1) Digital-too-late output.
    axes[0, 1].plot(times_ms[sl], digital[sl], color="C1", lw=1.0, label="digital cleaned")
    axes[0, 1].plot(times_ms[sl], clean[sl], color="C2", lw=1.1, label="clean NQR")
    axes[0, 1].set_ylim(-baseline_scale, baseline_scale)
    axes[0, 1].set_title("Digital Cancellation (too late)")
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].set_ylabel("cleaned channel")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    # (0,2) Active feedforward output.
    axes[0, 2].plot(times_ms[sl], active.digitized[sl], color="C0", lw=1.0, label="active digitised")
    axes[0, 2].plot(times_ms[sl], clean[sl], color="C2", lw=1.1, label="clean NQR")
    axes[0, 2].set_ylim(-baseline_scale, baseline_scale)
    axes[0, 2].set_title("Active Feedforward")
    axes[0, 2].set_xlabel("time (ms)")
    axes[0, 2].set_ylabel("cleaned channel")
    axes[0, 2].legend(loc="upper right", fontsize=8)

    # (1,0) NQR recovery error in echo windows.
    digital_err = _rms(digital[echo] - clean[echo])
    active_err = _rms(active.digitized[echo] - clean[echo])
    axes[1, 0].bar(
        ["digital\n(too late)", "active\nfeedforward"],
        [digital_err, active_err],
        color=["C1", "C0"],
        width=0.55,
    )
    axes[1, 0].set_ylabel("NQR recovery RMS error")
    axes[1, 0].set_title("Echo-Window Recovery Error")

    # (1,1) Latency limits the cancellation bandwidth.
    axes[1, 1].plot(results["latencies_us"], results["latency_supp"], "o-", color="C0", ms=4)
    axes[1, 1].axvline(args.latency_us, color="0.5", ls=":", label="operating latency")
    axes[1, 1].set_xlabel("actuator latency (us)")
    axes[1, 1].set_ylabel("coherent suppression (dB)")
    axes[1, 1].set_title("Latency Bounds the Bandwidth")
    axes[1, 1].legend(fontsize=8)

    # (1,2) Finite drive range limits large-interferer cancellation.
    axes[1, 2].plot(results["drive_ratios"], results["drive_supp"], "s-", color="C0")
    axes[1, 2].axvline(1.0, color="0.5", ls=":", label="command peak")
    axes[1, 2].set_xlabel("drive limit / command peak")
    axes[1, 2].set_ylabel("coherent suppression (dB)")
    axes[1, 2].set_title("Drive Range Bounds Deep Nulls")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Active Feedforward RFI Cancellation With a Compensation Coil")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(record: dict[str, object], results: dict[str, object]) -> None:
    clean = record["clean"]
    echo = record["echo_mask"]
    digital = np.real(results["digital"].cleaned)
    active = results["active"]
    raw_sat = results["raw_saturation"]
    active_sat = results["active_saturation"]
    print("Active feedforward RFI cancellation")
    print(
        f"raw primary clipped samples: {100.0 * raw_sat.saturated_fraction:.1f}% "
        f"(peak {raw_sat.max_abs:.2f} vs full scale {results['adc_saturation']:.2f})"
    )
    print(f"active analog-residual clipped samples: {100.0 * active_sat.saturated_fraction:.1f}%")
    digital_bias = signal_bias(clean, digital, echo)
    active_bias = signal_bias(clean, active.digitized, echo)
    print(
        f"digital (too late): recovery RMS={_rms(digital[echo] - clean[echo]):.4f}, "
        f"amp bias={digital_bias.amplitude_bias:+.3f}"
    )
    print(
        f"active feedforward : recovery RMS={_rms(active.digitized[echo] - clean[echo]):.4f}, "
        f"amp bias={active_bias.amplitude_bias:+.3f}"
    )


if __name__ == "__main__":
    main()
