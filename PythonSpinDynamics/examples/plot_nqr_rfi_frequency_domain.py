"""Plot frequency-domain (per-bin Wiener) RFI cancellation of resonant coupling.

This Phase-5 example targets ``frequency_domain_canceller``. The interference
reaches the primary through a *resonant* coupling path -- a decaying, ringing
impulse response many samples long, as a tuned probe or cable resonance would
impose. A zero-lag scalar canceller and even a modest FIR cannot match that
frequency-dependent transfer, so residual RFI survives around the resonance. The
frequency-domain canceller estimates a complex transfer in every DFT bin from
averaged cross-spectra and removes the coupling across the band.

The panels show the recovered transfer against the true coupling response, the
multiple-coherence spectrum (which frequencies the reference can explain), and a
suppression comparison against scalar and FIR cancellers.

Run with ``--output nqr_rfi_frequency_domain.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    frequency_domain_canceller,
    gated_ridge_fir_canceller,
    rfi_suppression_db,
    scalar_canceller,
    signal_bias,
)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _resonant_kernel(sample_rate_hz: float, resonance_hz: float, decay_us: float) -> np.ndarray:
    """A causal decaying-oscillation impulse response (a resonant coupling)."""

    decay = decay_us * 1e-6
    length = int(round(6.0 * decay * sample_rate_hz))
    t = np.arange(length, dtype=np.float64) / sample_rate_hz
    kernel = np.exp(-t / decay) * np.cos(2.0 * np.pi * resonance_hz * t)
    return kernel / (np.max(np.abs(kernel)) or 1.0)


def _build_record(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    fs = args.sample_rate_khz * 1e3
    n = args.num_samples
    t = np.arange(n, dtype=np.float64) / fs

    reference = rng.normal(size=n)
    kernel = _resonant_kernel(fs, args.resonance_khz * 1e3, args.decay_us)
    rfi_raw = np.convolve(reference, kernel)[:n]
    kernel_scale = args.rfi_rms / (_rms(rfi_raw) or 1.0)
    rfi = rfi_raw * kernel_scale

    # A single Gaussian-enveloped NQR echo near the resonance (the in-band case).
    center = 0.5 * t[-1]
    width = 0.06 * t[-1]
    envelope = np.exp(-0.5 * ((t - center) / width) ** 2)
    signal = envelope * np.cos(2.0 * np.pi * args.nqr_detect_khz * 1e3 * t)
    signal = signal * (args.rfi_rms / args.rfi_to_signal) / (_rms(signal) or 1.0)
    signal_region = envelope > 0.05

    references = reference.reshape(1, -1) + args.reference_noise * rng.normal(size=(1, n))
    primary = rfi + signal + args.primary_noise * rng.normal(size=n)

    return {
        "times": t,
        "sample_rate_hz": fs,
        "kernel": kernel,
        "kernel_scale": kernel_scale,
        "clean": signal,
        "primary": primary,
        "references": references,
        "signal_region": signal_region,
    }


def _run(record: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    primary = record["primary"]
    references = record["references"]
    fs = float(record["sample_rate_hz"])
    n = primary.size
    fit = np.ones(n, dtype=bool)  # signal is uncorrelated with the reference

    scalar = scalar_canceller(primary, references, fit)
    fir = gated_ridge_fir_canceller(primary, references, fit, taps=args.fir_taps, ridge=1e-6)
    freq = frequency_domain_canceller(
        primary, references, segment_length=args.segment_length, sample_rate_hz=fs
    )

    # Score suppression on the interior, away from the signal burst and the WOLA
    # edge taper, so it measures coherent-RFI removal.
    edge = args.segment_length
    score = np.zeros(n, dtype=bool)
    score[edge : n - edge] = True
    score &= ~record["signal_region"]

    results = {"scalar": scalar, "fir": fir, "freq": freq, "score_mask": score}
    for name in ("scalar", "fir", "freq"):
        cleaned = results[name].cleaned
        results[f"{name}_supp"] = rfi_suppression_db(
            primary, cleaned, score, clean_signal=record["clean"]
        ).suppression_db
        results[f"{name}_bias"] = signal_bias(
            record["clean"], cleaned, record["signal_region"]
        ).amplitude_bias
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate-khz", type=float, default=20.0)
    parser.add_argument("--num-samples", type=int, default=8192)
    parser.add_argument("--resonance-khz", type=float, default=0.9)
    parser.add_argument("--decay-us", type=float, default=700.0)
    parser.add_argument("--nqr-detect-khz", type=float, default=0.9)
    parser.add_argument("--rfi-rms", type=float, default=1.0)
    parser.add_argument("--rfi-to-signal", type=float, default=10.0)
    parser.add_argument("--fir-taps", type=int, default=16)
    parser.add_argument("--segment-length", type=int, default=256)
    parser.add_argument("--reference-noise", type=float, default=0.01)
    parser.add_argument("--primary-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    record = _build_record(args)
    results = _run(record, args)
    _plot(plt, record, results, args)
    _print_summary(record, results, args)


def _plot(plt, record: dict[str, object], results: dict[str, object], args: argparse.Namespace) -> None:
    times_ms = record["times"] * 1e3
    clean = record["clean"]
    primary = record["primary"]
    fs = float(record["sample_rate_hz"])
    freq = results["freq"]

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), constrained_layout=True)

    # (0,0) Time record around the echo burst.
    center = int(np.argmax(np.abs(clean)))
    half = max(1, int(0.12 * times_ms.size))
    sl = slice(max(0, center - half), min(times_ms.size, center + half))
    axes[0, 0].plot(times_ms[sl], primary[sl], color="0.7", lw=0.6, label="contaminated")
    axes[0, 0].plot(times_ms[sl], clean[sl], color="C2", lw=1.1, label="clean NQR")
    axes[0, 0].plot(times_ms[sl], np.real(freq.cleaned)[sl], color="C0", lw=0.9, label="freq-domain cleaned")
    axes[0, 0].set_title("Time Record (resonant coupling)")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("primary channel")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    # (0,1) Recovered transfer vs. the true coupling response. The applied
    # coupling is kernel_scale * kernel, so its response is scaled to match.
    length = args.segment_length
    true_response = record["kernel_scale"] * np.abs(np.fft.fft(record["kernel"], length))
    recovered = np.abs(freq.transfer_function[0])
    order = np.argsort(freq.frequencies)
    fpos = freq.frequencies[order] / 1e3
    axes[0, 1].plot(fpos, true_response[order], color="0.4", lw=1.4, label="|true coupling|")
    axes[0, 1].plot(fpos, recovered[order], color="C0", lw=1.0, ls="--", label="|recovered W(f)|")
    axes[0, 1].set_xlim(0.0, fs / 2 / 1e3)
    axes[0, 1].set_title("Recovered Transfer Function")
    axes[0, 1].set_xlabel("frequency (kHz)")
    axes[0, 1].set_ylabel("|transfer|")
    axes[0, 1].legend(fontsize=8)

    # (1,0) Multiple-coherence spectrum.
    axes[1, 0].plot(fpos, freq.coherence[order], color="C0")
    axes[1, 0].axvline(args.resonance_khz, color="C3", ls=":", label="resonance")
    axes[1, 0].set_xlim(0.0, fs / 2 / 1e3)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Reference-Primary Coherence")
    axes[1, 0].set_xlabel("frequency (kHz)")
    axes[1, 0].set_ylabel(r"$\gamma^2(f)$")
    axes[1, 0].legend(fontsize=8)

    # (1,1) Suppression comparison.
    labels = ["scalar", f"FIR ({args.fir_taps} taps)", "freq-domain"]
    supp = [results["scalar_supp"], results["fir_supp"], results["freq_supp"]]
    axes[1, 1].bar(labels, supp, color=["C3", "C1", "C0"], width=0.6)
    axes[1, 1].set_ylabel("coherent suppression (dB)")
    axes[1, 1].set_title("Suppression vs. Canceller")
    axes[1, 1].tick_params(axis="x", rotation=12)

    fig.suptitle("Frequency-Domain RFI Cancellation of Resonant Coupling")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(record: dict[str, object], results: dict[str, object], args: argparse.Namespace) -> None:
    print("Frequency-domain RFI cancellation of resonant coupling")
    print(f"resonance {args.resonance_khz:.2f} kHz, kernel length {record['kernel'].size} samples")
    for key, label in [("scalar", "scalar"), ("fir", f"FIR({args.fir_taps})"), ("freq", "freq-domain")]:
        print(
            f"{label:14s}: coherent suppression={results[f'{key}_supp']:6.1f} dB, "
            f"NQR amp bias={results[f'{key}_bias']:+.3f}"
        )


if __name__ == "__main__":
    main()
