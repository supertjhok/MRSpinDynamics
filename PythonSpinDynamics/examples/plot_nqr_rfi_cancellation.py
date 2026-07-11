"""Plot pulsed-NQR RFI cancellation cases with diagnostics.

This Phase-5 example builds a synthetic SLSE-style receive record with a weak
NQR echo train and coherent AM-like RFI. Reference channels run continuously on
the same shot clock. The panels compare:

* rank-deficient one-reference subtraction,
* multi-reference scalar cancellation,
* offline windowed ridge cancellation for randomly modulated coupling,
* reference NQR leakage, and
* primary-channel clipping, which offline cleanup cannot undo.

Run with ``--output nqr_rfi_cancellation.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    matched_filter_snr_improvement,
    reference_design_diagnostics,
    residual_spectral_lines,
    rfi_suppression_db,
    saturation_diagnostics,
    scalar_canceller,
    signal_bias,
    windowed_ridge_fir_canceller,
)
from spin_dynamics.nqr import slse_acquisition_mask, slse_sequence  # noqa: E402


def _smooth_random(rng: np.random.Generator, size: int, width: int) -> np.ndarray:
    values = rng.normal(size=size)
    kernel = np.ones(max(1, int(width)), dtype=np.float64)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def _synthetic_echo_train(times: np.ndarray, echo_spacing: float, num_echoes: int) -> np.ndarray:
    signal = np.zeros(times.size, dtype=np.float64)
    width = 0.14 * echo_spacing
    for idx in range(num_echoes):
        center = (idx + 0.72) * echo_spacing
        envelope = np.exp(-0.5 * ((times - center) / width) ** 2)
        decay = np.exp(-(idx + 1) / max(1.0, 0.45 * num_echoes))
        signal += 0.08 * decay * envelope * np.cos(2.0 * np.pi * 650.0 * (times - center))
    return signal


def _build_record(args: argparse.Namespace) -> dict[str, np.ndarray | float]:
    rng = np.random.default_rng(args.seed)
    sample_rate_hz = args.sample_rate_khz * 1e3
    echo_spacing = args.echo_spacing_ms * 1e-3
    pulse_duration = args.pulse_us * 1e-6
    sequence = slse_sequence(
        "x",
        pulse_duration_seconds=pulse_duration,
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
    sequence_times = times - args.pre_baseline_ms * 1e-3
    clean = _synthetic_echo_train(sequence_times, echo_spacing, args.num_echoes)
    clean[~mask.receive_mask] = 0.0
    echo_mask = mask.receive_mask & (np.abs(clean) > args.echo_threshold * np.max(np.abs(clean)))
    fit_mask = mask.receive_mask & ~echo_mask

    carrier = 2.0 * np.pi * args.carrier_khz * 1e3 * times
    modulation = 1.0 + 0.22 * np.sin(2.0 * np.pi * args.modulation_hz * times)
    modulation += args.random_modulation * _smooth_random(rng, times.size, args.random_width)
    modulation = np.clip(modulation, 0.25, None)
    phase_wander = 0.2 * _smooth_random(rng, times.size, args.random_width)
    bx = modulation * np.cos(carrier + phase_wander)
    by = 0.65 * modulation * np.sin(carrier + 0.55 * phase_wander)
    bz = 0.28 * modulation * np.cos(carrier + 0.35 * np.sin(2.0 * np.pi * 3.0 * times))
    references = np.vstack([bx, by, bz])
    references += args.reference_noise * rng.normal(size=references.shape)

    gain_x = 0.55 + 0.38 * _smooth_random(rng, times.size, 3 * args.random_width)
    gain_y = -0.36 + 0.28 * _smooth_random(rng, times.size, 2 * args.random_width)
    gain_z = 0.24 + 0.18 * np.sin(2.0 * np.pi * 2.1 * times)
    rfi = gain_x * bx + gain_y * by + gain_z * bz
    rfi += 0.025 * np.cos(2.0 * np.pi * (args.carrier_khz * 1e3 + 170.0) * times)
    contaminated = clean + args.rfi_scale * rfi + args.primary_noise * rng.normal(size=times.size)
    clipped = np.clip(contaminated, -args.clip_threshold, args.clip_threshold)
    leaky_references = references + args.reference_leakage * clean[np.newaxis, :]

    return {
        "times": times,
        "clean": clean,
        "contaminated": contaminated,
        "clipped": clipped,
        "references": references,
        "leaky_references": leaky_references,
        "fit_mask": fit_mask,
        "echo_mask": echo_mask,
        "receive_mask": mask.receive_mask,
        "transmit_mask": mask.transmit_mask,
        "ringdown_mask": mask.ringdown_mask,
        "true_gains": np.vstack([gain_x, gain_y, gain_z]),
        "sample_rate_hz": sample_rate_hz,
    }


def _run_cancellers(record: dict[str, np.ndarray | float], args: argparse.Namespace) -> dict[str, object]:
    y = record["contaminated"]
    clipped = record["clipped"]
    refs = record["references"]
    leaky_refs = record["leaky_references"]
    fit = record["fit_mask"]
    sample_rate_hz = float(record["sample_rate_hz"])
    window_samples = max(32, int(round(args.window_ms * 1e-3 * sample_rate_hz)))

    one_axis = scalar_canceller(y, refs[:1], fit, ridge=args.ridge)
    scalar = scalar_canceller(y, refs, fit, ridge=args.ridge)
    windowed = windowed_ridge_fir_canceller(
        y,
        refs,
        fit,
        window_samples=window_samples,
        ridge=args.ridge,
        smoothness=args.smoothness,
    )
    leaky = windowed_ridge_fir_canceller(
        y,
        leaky_refs,
        fit,
        window_samples=window_samples,
        ridge=args.ridge,
        smoothness=args.smoothness,
    )
    clipped_case = windowed_ridge_fir_canceller(
        clipped,
        refs,
        fit,
        window_samples=window_samples,
        ridge=args.ridge,
        smoothness=args.smoothness,
    )

    return {
        "one_axis": one_axis,
        "scalar": scalar,
        "windowed": windowed,
        "leaky": leaky,
        "clipped": clipped_case,
        "window_samples": window_samples,
        "diagnostics": _diagnostics(record, {
            "one_axis": one_axis.cleaned,
            "scalar": scalar.cleaned,
            "windowed": windowed.cleaned,
            "leaky": leaky.cleaned,
            "clipped": clipped_case.cleaned,
        }, args.clip_threshold),
    }


def _diagnostics(
    record: dict[str, np.ndarray | float],
    cleaned_cases: dict[str, np.ndarray],
    clip_threshold: float,
) -> dict[str, object]:
    clean = record["clean"]
    contaminated = record["contaminated"]
    echo_mask = record["echo_mask"]
    receive_mask = record["receive_mask"]
    refs = record["references"]
    fs = float(record["sample_rate_hz"])
    diag: dict[str, object] = {
        "design": reference_design_diagnostics(refs, receive_mask),
        "one_axis_design": reference_design_diagnostics(refs[:1], receive_mask),
        "saturation": saturation_diagnostics(record["clipped"], clip_threshold),
        "before_lines": residual_spectral_lines(contaminated - clean, fs, receive_mask, top_n=3),
    }
    for name, cleaned in cleaned_cases.items():
        diag[f"{name}_suppression"] = rfi_suppression_db(
            contaminated,
            cleaned,
            receive_mask,
            clean_signal=clean,
        )
        diag[f"{name}_bias"] = signal_bias(clean, cleaned, echo_mask)
        diag[f"{name}_snr"] = matched_filter_snr_improvement(
            clean[echo_mask],
            (contaminated[echo_mask])[np.newaxis, :],
            (cleaned[echo_mask])[np.newaxis, :],
        )
    diag["after_lines"] = residual_spectral_lines(
        cleaned_cases["windowed"] - clean,
        fs,
        receive_mask,
        top_n=3,
    )
    return diag


# Keep CLI choices together so scientific defaults are easy to find and override.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate-khz", type=float, default=100.0)
    parser.add_argument("--carrier-khz", type=float, default=9.0)
    parser.add_argument("--modulation-hz", type=float, default=37.0)
    parser.add_argument("--random-modulation", type=float, default=0.16)
    parser.add_argument("--random-width", type=int, default=181)
    parser.add_argument("--num-echoes", type=int, default=28)
    parser.add_argument("--echo-spacing-ms", type=float, default=4.0)
    parser.add_argument("--pulse-us", type=float, default=220.0)
    parser.add_argument("--ringdown-us", type=float, default=160.0)
    parser.add_argument("--pre-baseline-ms", type=float, default=4.0)
    parser.add_argument("--post-baseline-ms", type=float, default=4.0)
    parser.add_argument("--echo-threshold", type=float, default=0.06)
    parser.add_argument("--rfi-scale", type=float, default=0.45)
    parser.add_argument("--primary-noise", type=float, default=0.01)
    parser.add_argument("--reference-noise", type=float, default=0.015)
    parser.add_argument("--reference-leakage", type=float, default=0.35)
    parser.add_argument("--clip-threshold", type=float, default=0.36)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--smoothness", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    record = _build_record(args)
    results = _run_cancellers(record, args)
    diag = results["diagnostics"]
    _plot(plt, record, results, diag, args)
    _print_summary(diag)


# Keep visualization separate from simulation for headless runs and reuse.
def _plot(plt, record: dict[str, np.ndarray | float], results: dict[str, object], diag: dict[str, object], args: argparse.Namespace) -> None:
    times_ms = record["times"] * 1e3
    clean = record["clean"]
    contaminated = record["contaminated"]
    windowed = np.real(results["windowed"].cleaned)
    leaky = np.real(results["leaky"].cleaned)
    clipped = np.real(results["clipped"].cleaned)
    receive = record["receive_mask"]
    echo = record["echo_mask"]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)
    plot_slice = slice(0, min(times_ms.size, int(28e-3 * float(record["sample_rate_hz"]))))
    axes[0, 0].plot(times_ms[plot_slice], contaminated[plot_slice], color="0.65", lw=0.9, label="contaminated")
    axes[0, 0].plot(times_ms[plot_slice], clean[plot_slice], color="C2", lw=1.1, label="clean NQR")
    axes[0, 0].plot(times_ms[plot_slice], windowed[plot_slice], color="C0", lw=1.0, label="windowed ridge")
    axes[0, 0].set_title("SLSE-Gated Time Record")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("primary channel")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    before_lines = diag["before_lines"]
    after_lines = diag["after_lines"]
    positive_before = before_lines.frequencies_hz >= 0
    positive_after = after_lines.frequencies_hz >= 0
    axes[0, 1].semilogy(
        before_lines.frequencies_hz[positive_before] / 1e3,
        before_lines.amplitudes[positive_before] + 1e-12,
        color="0.5",
        label="before",
    )
    axes[0, 1].semilogy(
        after_lines.frequencies_hz[positive_after] / 1e3,
        after_lines.amplitudes[positive_after] + 1e-12,
        color="C0",
        label="after windowed",
    )
    axes[0, 1].set_xlim(0.0, 2.0 * args.carrier_khz)
    axes[0, 1].set_title("Residual Spectral Lines")
    axes[0, 1].set_xlabel("frequency (kHz)")
    axes[0, 1].set_ylabel("FFT amplitude")
    axes[0, 1].legend(fontsize=8)

    labels = ["1-axis", "3-axis", "windowed", "leaky ref", "clipped"]
    keys = ["one_axis", "scalar", "windowed", "leaky", "clipped"]
    suppression = [diag[f"{key}_suppression"].suppression_db for key in keys]
    axes[0, 2].bar(labels, suppression, color=["C3", "C1", "C0", "C4", "C5"])
    axes[0, 2].set_title("Receive-Gap RFI Suppression")
    axes[0, 2].set_ylabel("suppression (dB)")
    axes[0, 2].tick_params(axis="x", rotation=25)

    true_gains = args.rfi_scale * record["true_gains"]
    coeff = results["windowed"].coefficient_history[:, :, 0]
    axes[1, 0].plot(times_ms[receive], true_gains[0, receive], "--", color="C0", label="true x")
    axes[1, 0].plot(times_ms[receive], np.real(coeff[receive, 0]), color="C0", label="fit x")
    axes[1, 0].plot(times_ms[receive], true_gains[1, receive], "--", color="C1", label="true y")
    axes[1, 0].plot(times_ms[receive], np.real(coeff[receive, 1]), color="C1", label="fit y")
    axes[1, 0].set_title("Offline Coefficient Tracking")
    axes[1, 0].set_xlabel("time (ms)")
    axes[1, 0].set_ylabel("coefficient")
    axes[1, 0].legend(fontsize=8, ncol=2)

    bias_values = [diag[f"{key}_bias"].amplitude_bias for key in keys]
    phase_values = [diag[f"{key}_bias"].phase_bias_rad for key in keys]
    xloc = np.arange(len(labels))
    axes[1, 1].bar(xloc - 0.18, bias_values, width=0.35, label="amplitude")
    axes[1, 1].bar(xloc + 0.18, phase_values, width=0.35, label="phase rad")
    axes[1, 1].set_xticks(xloc, labels, rotation=25)
    axes[1, 1].set_title("NQR Signal Bias in Echo Windows")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].plot(times_ms, contaminated, color="0.85", lw=0.7, label="contaminated")
    axes[1, 2].plot(times_ms, clipped, color="C5", lw=0.8, label="clipped cleanup")
    axes[1, 2].plot(times_ms, leaky, color="C4", lw=0.8, label="leaky-ref cleanup")
    axes[1, 2].fill_between(
        times_ms,
        -0.12,
        0.12,
        where=~receive,
        color="0.8",
        alpha=0.4,
        transform=axes[1, 2].get_xaxis_transform(),
        label="blanked",
    )
    axes[1, 2].fill_between(
        times_ms,
        -0.12,
        0.12,
        where=echo,
        color="C2",
        alpha=0.15,
        transform=axes[1, 2].get_xaxis_transform(),
        label="echo mask",
    )
    axes[1, 2].set_title("Failure Modes")
    axes[1, 2].set_xlabel("time (ms)")
    axes[1, 2].set_ylabel("cleaned channel")
    axes[1, 2].legend(fontsize=8, loc="upper right")

    fig.suptitle("Pulsed-NQR RFI Cancellation: References, Masks, and Diagnostics")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(diag: dict[str, object]) -> None:
    design = diag["design"]
    one_axis = diag["one_axis_design"]
    sat = diag["saturation"]
    print("Pulsed-NQR RFI cancellation example")
    print(f"reference design rank: {design.rank}/{design.column_count}")
    print(f"one-axis design rank: {one_axis.rank}/{one_axis.column_count}")
    print(f"clipped samples: {sat.saturated_count} ({100.0 * sat.saturated_fraction:.2f}%)")
    for key, label in [
        ("one_axis", "one-axis"),
        ("scalar", "3-axis scalar"),
        ("windowed", "windowed ridge"),
        ("leaky", "leaky reference"),
        ("clipped", "clipped primary"),
    ]:
        suppression = diag[f"{key}_suppression"]
        bias = diag[f"{key}_bias"]
        snr = diag[f"{key}_snr"]
        print(
            f"{label:15s}: suppression={suppression.suppression_db:6.1f} dB, "
            f"SNR gain={snr.improvement_db:5.1f} dB, "
            f"amp bias={bias.amplitude_bias:+.3f}, phase={bias.phase_bias_rad:+.3f} rad"
        )


if __name__ == "__main__":
    main()
