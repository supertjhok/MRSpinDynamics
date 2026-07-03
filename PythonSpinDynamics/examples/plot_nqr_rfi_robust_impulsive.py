"""Plot robust (Huber-IRLS) RFI cancellation under impulsive pickup.

This Phase-5 example targets the two features added on top of the base RFI
workflow: the outlier-robust FIR canceller (``robust_fir_canceller``) and the
reference-noise-injection diagnostic (``reference_noise_injection``).

The scene is a pulsed-NQR SLSE record contaminated by two physically distinct
interferers:

* **coherent broadcast-band RFI** -- three independent far-field carriers sensed
  by three continuously-running reference coils. Because the references span the
  carriers, a multi-reference scalar/FIR canceller removes this coherent RFI
  almost perfectly *when its transfer coefficients are estimated cleanly*.
* **impulsive switching transients** -- conducted/cable pickup that couples
  directly into the primary receiver (a switching power converter near the
  preamp). The remote reference coils barely see it, so these bursts are
  *outliers* to the reference model rather than something the references can
  cancel.

Ordinary gated least squares minimises squared error, so the rare, large
impulse samples bias the fitted coherent-RFI transfer and leave residual RFI and
NQR-amplitude bias. The Huber-IRLS canceller down-weights the impulsive tail and
recovers the clean transfer. The final panel uses ``reference_noise_injection``
to show the other side of the trade-off: a fitted canceller passes reference
noise into the cleaned output (``var = sum_k sigma_k^2 sum_l |h[k,l]|^2``), so
cancellation is only a net win while that injected noise stays below the RFI it
removes.

Run with ``--output nqr_rfi_robust_impulsive.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    ReferenceCoil,
    UniformPlaneWaveSource,
    am_carrier_waveform,
    coil_voltage,
    fit_gated_ridge_fir,
    fit_robust_fir,
    impulsive_waveform,
    matched_filter_snr_improvement,
    reference_design_diagnostics,
    reference_matrix,
    reference_noise_injection,
    rfi_suppression_db,
    robust_fir_canceller,
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
    """Three independent far-field carriers with spanning polarizations."""

    waveforms = [
        am_carrier_waveform(
            n, sample_rate_hz, carrier_hz=7000.0, modulation_hz=45.0, modulation_depth=0.6
        ),
        tone_waveform(n, sample_rate_hz, frequency_hz=11000.0),
        am_carrier_waveform(
            n, sample_rate_hz, carrier_hz=5000.0, modulation_hz=80.0, modulation_depth=0.4
        ),
    ]
    polarizations = [
        (0.80, 0.50, 0.30),
        (0.20, 0.90, 0.40),
        (0.50, 0.30, 0.90),
    ]
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
    fit_mask = receive & ~echo_mask

    # Coherent, reference-cancellable RFI through the source/coil model.
    sources = _coherent_sources(n, sample_rate_hz)
    coils = [
        ReferenceCoil(pickup_vector=(1.0, 0.0, 0.0), label="ref-x"),
        ReferenceCoil(pickup_vector=(0.0, 1.0, 0.0), label="ref-y"),
        ReferenceCoil(pickup_vector=(0.0, 0.0, 1.0), label="ref-z"),
    ]
    primary_coil = ReferenceCoil(pickup_vector=(0.60, 0.50, 0.62), label="primary")

    primary_coh = coil_voltage(primary_coil, sources)
    refs_coh = reference_matrix(coils, sources)  # (3, n), noise-free
    scale = args.rfi_rms / (_rms(primary_coh[receive]) or 1.0)
    primary_coh = primary_coh * scale
    refs_coh = refs_coh * scale

    # Scale the NQR echo train relative to the coherent RFI level.
    clean = clean * (args.rfi_rms / args.rfi_to_signal) / (_rms(clean[echo_mask]) or 1.0)

    # Impulsive switching transients: conducted pickup local to the primary.
    burst = impulsive_waveform(
        n,
        sample_rate_hz,
        event_rate_hz=args.impulse_rate_hz,
        amplitude=1.0,
        decay_seconds=args.impulse_decay_us * 1e-6,
        ring_hz=args.impulse_ring_khz * 1e3,
        seed=args.seed + 7,
    ).samples
    burst = burst / (float(np.max(np.abs(burst))) or 1.0)
    primary_impulse = args.impulse_ratio * args.rfi_rms * burst
    impulse_region = np.abs(primary_impulse) > 0.05 * args.impulse_ratio * args.rfi_rms

    # References see only a tiny fraction of the local impulsive pickup.
    references = refs_coh + args.impulse_leak * primary_impulse[np.newaxis, :]
    references = references + args.reference_noise * rng.normal(size=references.shape)
    primary = (
        clean
        + primary_coh
        + primary_impulse
        + args.primary_noise * rng.normal(size=n)
    )

    return {
        "times": times,
        "sample_rate_hz": sample_rate_hz,
        "clean": clean,
        "primary": primary,
        "primary_coh": primary_coh,
        "primary_impulse": primary_impulse,
        "references": references,
        "refs_coh": refs_coh,
        "receive_mask": receive,
        "echo_mask": echo_mask,
        "fit_mask": fit_mask,
        "impulse_region": impulse_region,
    }


def _run_cancellers(record: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    primary = record["primary"]
    references = record["references"]
    fit = record["fit_mask"]
    receive = record["receive_mask"]

    # Ground-truth coherent transfer: fit on the clean, impulse-free system.
    truth_model = fit_gated_ridge_fir(
        record["primary_coh"], record["refs_coh"], receive, taps=args.taps, ridge=args.ridge
    )
    l2_model = fit_gated_ridge_fir(primary, references, fit, taps=args.taps, ridge=args.ridge)
    robust_model = fit_robust_fir(
        primary,
        references,
        fit,
        taps=args.taps,
        ridge=args.ridge,
        huber_delta=args.huber_delta,
    )
    # The convenience form re-runs the same IRLS fit but also returns the final
    # per-sample weights and iteration count for the down-weighting panel.
    robust = robust_fir_canceller(
        primary,
        references,
        fit,
        taps=args.taps,
        ridge=args.ridge,
        huber_delta=args.huber_delta,
    )
    return {
        "truth_model": truth_model,
        "l2_model": l2_model,
        "robust_model": robust_model,
        "l2": l2_model.apply(primary, references),
        "robust": robust,
    }


def _diagnostics(
    record: dict[str, object],
    results: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    clean = record["clean"]
    primary = record["primary"]
    primary_coh = record["primary_coh"]
    refs_coh = record["refs_coh"]
    receive = record["receive_mask"]
    echo = record["echo_mask"]
    # The impulse bursts are detected (the IRLS weights flag them) and excised
    # before NQR estimation, so the NQR endpoints are scored on impulse-free echo
    # samples, where the *residual coherent RFI* is what remains to bias them.
    echo_eval = echo & ~record["impulse_region"]

    h_true = np.asarray(results["truth_model"].coefficients)
    diag: dict[str, object] = {
        "design": reference_design_diagnostics(record["references"], receive, taps=args.taps),
        "echo_eval": echo_eval,
    }
    for name in ("l2", "robust"):
        cleaned = results[name].cleaned
        coeff = np.asarray(results[f"{name}_model"].coefficients)
        diag[f"{name}_coeff_error"] = float(np.linalg.norm(coeff - h_true))
        # Coherent suppression is the residual left when the fitted transfer is
        # applied to the *impulse- and noise-free* coherent system, so it
        # reflects transfer quality without being capped by the uncancellable
        # impulse tails that survive in every linear-canceller output.
        coherent_residual = primary_coh - results[f"{name}_model"].predict(refs_coh)
        diag[f"{name}_suppression"] = rfi_suppression_db(
            primary_coh, coherent_residual, receive
        )
        diag[f"{name}_bias"] = signal_bias(clean, cleaned, echo_eval)
        diag[f"{name}_snr"] = matched_filter_snr_improvement(
            clean[echo_eval],
            (primary[echo_eval])[np.newaxis, :],
            (cleaned[echo_eval])[np.newaxis, :],
        )
    diag["h_true_norm"] = float(np.linalg.norm(h_true))

    # Reference-noise injection for the robust model as the noise floor scales.
    robust_coeff = np.asarray(results["robust_model"].coefficients)
    sigma_vec = np.full(robust_coeff.shape[0], args.reference_noise)
    injection = reference_noise_injection(robust_coeff, sigma_vec)
    removed = _rms(np.real(results["robust_model"].predict(refs_coh))[receive])
    diag["injection"] = injection
    diag["removed_rfi_rms"] = removed
    diag["break_even_sigma"] = (
        removed / injection.noise_gain if injection.noise_gain > 0 else np.inf
    )
    return diag


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
    parser.add_argument(
        "--nqr-detect-khz",
        type=float,
        default=6.5,
        help="NQR detection frequency; the 7 kHz carrier is deliberately in-band.",
    )
    parser.add_argument("--taps", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument("--rfi-rms", type=float, default=1.0)
    parser.add_argument("--rfi-to-signal", type=float, default=12.0)
    parser.add_argument("--impulse-ratio", type=float, default=18.0)
    parser.add_argument("--impulse-rate-hz", type=float, default=500.0)
    parser.add_argument("--impulse-decay-us", type=float, default=60.0)
    parser.add_argument("--impulse-ring-khz", type=float, default=4.0)
    parser.add_argument("--impulse-leak", type=float, default=0.01)
    parser.add_argument("--reference-noise", type=float, default=0.01)
    parser.add_argument("--primary-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    record = _build_record(args)
    results = _run_cancellers(record, args)
    diag = _diagnostics(record, results, args)
    _plot(plt, record, results, diag, args)
    _print_summary(diag, args)


def _plot(
    plt,
    record: dict[str, object],
    results: dict[str, object],
    diag: dict[str, object],
    args: argparse.Namespace,
) -> None:
    times_ms = record["times"] * 1e3
    clean = record["clean"]
    primary = record["primary"]
    receive = record["receive_mask"]
    l2_cleaned = np.real(results["l2"].cleaned)
    robust_cleaned = np.real(results["robust"].cleaned)
    weights = results["robust"].fit_weights

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)

    # (0,0) Full record overview.
    axes[0, 0].plot(times_ms, primary, color="0.7", lw=0.6, label="contaminated")
    axes[0, 0].plot(times_ms, clean, color="C2", lw=1.0, label="clean NQR")
    axes[0, 0].plot(times_ms, robust_cleaned, color="C0", lw=0.8, label="robust cleaned")
    axes[0, 0].set_title("SLSE Record with Impulsive Pickup")
    axes[0, 0].set_xlabel("time (ms)")
    axes[0, 0].set_ylabel("primary channel")
    axes[0, 0].legend(loc="upper right", fontsize=8)

    # (0,1) Zoom around the largest impulse to contrast robust vs L2 baseline.
    impulse = np.abs(record["primary_impulse"])
    center = int(np.argmax(impulse * receive))
    fs = float(record["sample_rate_hz"])
    half = max(1, int(1.5e-3 * fs))
    sl = slice(max(0, center - half), min(times_ms.size, center + half))
    axes[0, 1].plot(times_ms[sl], l2_cleaned[sl], color="C1", lw=1.0, label="L2 cleaned")
    axes[0, 1].plot(times_ms[sl], robust_cleaned[sl], color="C0", lw=1.0, label="robust cleaned")
    axes[0, 1].plot(times_ms[sl], clean[sl], color="C2", lw=1.1, label="clean NQR")
    # The impulse spike itself is uncancellable and runs off-scale; clip the y
    # range to the baseline so the coherent-residual difference is visible: L2
    # leaves a structured coherent ripple, robust leaves a flat baseline.
    baseline_scale = 5.0 * (_rms(clean[record["echo_mask"]]) or 1.0)
    axes[0, 1].set_ylim(-baseline_scale, baseline_scale)
    axes[0, 1].set_title("Zoom Near an Impulse (baseline detail)")
    axes[0, 1].set_xlabel("time (ms)")
    axes[0, 1].set_ylabel("cleaned channel")
    axes[0, 1].legend(loc="upper right", fontsize=8)

    # (0,2) Coherent-transfer coefficient error and suppression.
    labels = ["L2\n(gated LS)", "robust\n(Huber IRLS)"]
    coeff_err = [diag["l2_coeff_error"], diag["robust_coeff_error"]]
    supp = [diag["l2_suppression"].suppression_db, diag["robust_suppression"].suppression_db]
    ax_err = axes[0, 2]
    xloc = np.arange(2)
    ax_err.bar(xloc, coeff_err, color=["C1", "C0"], width=0.55)
    ax_err.set_ylabel(r"transfer error $\|\hat h - h_{\mathrm{true}}\|$")
    ax_err.set_xticks(xloc, labels)
    ax_err.set_title("Coherent-Transfer Bias & Suppression")
    ax_supp = ax_err.twinx()
    ax_supp.plot(xloc, supp, "D", color="0.2", ms=9, label="suppression")
    for x, value in zip(xloc, supp):
        ax_supp.annotate(
            f"{value:.0f} dB",
            (x, value),
            textcoords="offset points",
            xytext=(8, 0),
            fontsize=8,
            va="center",
        )
    ax_supp.set_ylabel("coherent suppression (dB)")
    ax_supp.set_ylim(0.0, 1.25 * max(supp))
    ax_supp.legend(loc="center right", fontsize=8)

    # (1,0) IRLS weights flag the impulsive samples.
    fit_mask = record["fit_mask"]
    resid = np.abs(primary - np.real(results["robust"].predicted))
    ax_w = axes[1, 0]
    ax_w.plot(times_ms[fit_mask], resid[fit_mask], color="0.7", lw=0.6, label="|fit residual|")
    ax_w.set_xlabel("time (ms)")
    ax_w.set_ylabel("|fit residual|")
    ax_w.set_title("Huber IRLS Down-Weighting")
    ax_wt = ax_w.twinx()
    ax_wt.plot(times_ms[fit_mask], weights[fit_mask], ".", color="C3", ms=2.5, label="IRLS weight")
    ax_wt.set_ylabel("IRLS weight")
    ax_wt.set_ylim(-0.05, 1.08)
    lines = ax_w.get_lines() + ax_wt.get_lines()
    ax_w.legend(lines, [ln.get_label() for ln in lines], loc="upper right", fontsize=8)

    # (1,1) NQR signal bias in echo windows.
    amp_bias = [diag["l2_bias"].amplitude_bias, diag["robust_bias"].amplitude_bias]
    phase_bias = [diag["l2_bias"].phase_bias_rad, diag["robust_bias"].phase_bias_rad]
    ax_b = axes[1, 1]
    ax_b.bar(xloc - 0.18, amp_bias, width=0.35, color=["C1", "C0"], label="amplitude bias")
    ax_b.bar(
        xloc + 0.18,
        phase_bias,
        width=0.35,
        color=["C1", "C0"],
        alpha=0.5,
        label="phase bias (rad)",
    )
    ax_b.axhline(0.0, color="0.5", lw=0.8)
    ax_b.set_xticks(xloc, labels)
    ax_b.set_title("Recovered-NQR Bias (impulse-excised echoes)")
    ax_b.legend(fontsize=8)

    # (1,2) Reference-noise injection vs. RFI removed (break-even).
    injection = diag["injection"]
    removed = diag["removed_rfi_rms"]
    break_even = diag["break_even_sigma"]
    operating = args.reference_noise
    upper = 2.0 * break_even if np.isfinite(break_even) else 100.0 * operating
    sigma_axis = np.geomspace(0.1 * operating, upper, 120)
    injected = injection.noise_gain * sigma_axis
    ax_n = axes[1, 2]
    ax_n.semilogx(sigma_axis, injected, color="C0", label="injected noise RMS")
    ax_n.axhline(removed, color="C3", ls="--", label="coherent RFI removed")
    ax_n.axvline(operating, color="0.5", ls=":", label="operating point")
    if np.isfinite(break_even) and sigma_axis[0] <= break_even <= sigma_axis[-1]:
        ax_n.plot([break_even], [removed], "o", color="k")
        ax_n.annotate(
            f"break-even\n$\\sigma$={break_even:.2g}",
            (break_even, removed),
            textcoords="offset points",
            xytext=(-52, -6),
            fontsize=8,
        )
    headroom_db = 20.0 * np.log10(removed / (injection.noise_gain * operating)) if operating > 0 else np.inf
    ax_n.annotate(
        f"{headroom_db:.0f} dB headroom",
        (operating, injection.noise_gain * operating),
        textcoords="offset points",
        xytext=(8, 20),
        fontsize=8,
        color="0.3",
    )
    ax_n.set_xlabel(r"reference noise $\sigma$")
    ax_n.set_ylabel("RMS")
    ax_n.set_title("Reference-Noise Injection Trade-off")
    ax_n.legend(fontsize=8, loc="upper left")

    fig.suptitle("Robust RFI Cancellation Under Impulsive Pickup")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(diag: dict[str, object], args: argparse.Namespace) -> None:
    design = diag["design"]
    injection = diag["injection"]
    print("Robust RFI cancellation under impulsive pickup")
    print(f"reference design rank: {design.rank}/{design.column_count}")
    print(f"true coherent transfer norm: {diag['h_true_norm']:.4f}")
    for key, label in [("l2", "L2 gated LS"), ("robust", "robust IRLS")]:
        supp = diag[f"{key}_suppression"]
        bias = diag[f"{key}_bias"]
        snr = diag[f"{key}_snr"]
        print(
            f"{label:13s}: transfer err={diag[f'{key}_coeff_error']:.4f}, "
            f"coherent suppression={supp.suppression_db:6.1f} dB, "
            f"SNR gain={snr.improvement_db:5.1f} dB, "
            f"amp bias={bias.amplitude_bias:+.3f}"
        )
    print(
        f"reference-noise injection: noise_gain={injection.noise_gain:.3f}, "
        f"injected RMS={injection.injected_rms:.4f} at sigma={args.reference_noise:g}"
    )
    print(
        f"coherent RFI removed RMS={diag['removed_rfi_rms']:.4f}, "
        f"break-even sigma={diag['break_even_sigma']:.4f}"
    )


if __name__ == "__main__":
    main()
