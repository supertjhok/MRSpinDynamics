"""Monte Carlo study of RFI cancellation noise robustness and reference count.

The study addresses two experimental questions:

1. How robust are reference-cancellation methods, measured by NQR parameter
   accuracy, when the primary and reference channels contain both coherent RFI
   and receiver noise?
2. How much does cancellation improve when a near-field interferer is observed
   by multiple vector reference stations?

Each trial synthesizes several independent near-field magnetic-dipole RFI
sources, projects them onto a primary receive coil and tri-axial reference
stations, adds receiver noise at a requested initial-echo SNR, and evaluates
masked cancellers on an SLSE-style shot clock. The ``echo-aware joint`` method
fits windowed reference-derived RFI and an SLSE echo-train basis in one offline
ridge problem; its plotted endpoint is the fitted NQR signal component. The
default ``in-band`` mode puts RFI near the NQR echo-to-echo phase frequency so
it overlaps the estimator; use ``--rfi-spectral-mode out-of-band`` for the
useful null/control case where RFI mostly averages out of the echo estimator.
Each cleaned or fitted record is reduced to estimated resonant frequency,
initial amplitude, and echo-decay time constant T2e. Curves show Monte Carlo
means with approximate 95% confidence intervals of the mean.

Run with ``--output nqr_rfi_statistical_study.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.interference import (  # noqa: E402
    MagneticDipoleSource,
    RFIWaveform,
    ReferenceCoil,
    adaptive_lms_canceller,
    reference_design_diagnostics,
    rfi_suppression_db,
    scalar_canceller,
    windowed_joint_signal_reference_canceller,
    windowed_ridge_fir_canceller,
)
from spin_dynamics.interference._signal import spectral_derivative  # noqa: E402
from spin_dynamics.nqr import slse_acquisition_mask, slse_sequence  # noqa: E402


METHODS = (
    ("one_station", "1 station scalar", "C3"),
    ("all_scalar", "all stations scalar", "C1"),
    ("windowed", "windowed ridge", "C0"),
    ("adaptive", "NLMS adaptive", "C2"),
    ("joint", "echo-aware joint", "C4"),
)


@dataclass(frozen=True)
class NQRParameters:
    """Synthetic NQR parameters estimated from the echo train."""

    frequency_hz: float
    initial_amplitude: float
    t2e_seconds: float


@dataclass(frozen=True)
class TrialBase:
    """One noise-free near-field RFI realization and clean NQR record."""

    clean: np.ndarray
    echo_templates: np.ndarray
    echo_times: np.ndarray
    truth: NQRParameters
    primary_rfi: np.ndarray
    references: np.ndarray
    receive_mask: np.ndarray
    fit_mask: np.ndarray
    echo_mask: np.ndarray
    sample_rate_hz: float


def _smooth_random(rng: np.random.Generator, size: int, width: int) -> np.ndarray:
    values = rng.normal(size=size)
    kernel = np.ones(max(1, int(width)), dtype=np.float64)
    kernel /= np.sum(kernel)
    return np.convolve(values, kernel, mode="same")


def _echo_centers(echo_spacing: float, num_echoes: int) -> np.ndarray:
    return (np.arange(num_echoes, dtype=np.float64) + 0.70) * echo_spacing


def _echo_templates(
    times: np.ndarray,
    echo_spacing: float,
    num_echoes: int,
) -> tuple[np.ndarray, np.ndarray]:
    centers = _echo_centers(echo_spacing, num_echoes)
    width = 0.13 * echo_spacing
    templates = np.exp(-0.5 * ((times[np.newaxis, :] - centers[:, np.newaxis]) / width) ** 2)
    return centers, templates


def _synthetic_echo_train(
    times: np.ndarray,
    echo_spacing: float,
    num_echoes: int,
    truth: NQRParameters,
    *,
    phase_rad: float = 0.35,
) -> np.ndarray:
    centers, templates = _echo_templates(times, echo_spacing, num_echoes)
    amplitudes = truth.initial_amplitude * np.exp(-centers / truth.t2e_seconds)
    phases = phase_rad + 2.0 * np.pi * truth.frequency_hz * centers
    signal = np.sum(
        amplitudes[:, np.newaxis] * np.exp(1j * phases[:, np.newaxis]) * templates,
        axis=0,
    )
    return signal


def _source_positions() -> tuple[tuple[float, float, float], ...]:
    return (
        (0.06, -0.04, 0.03),
        (-0.05, 0.05, -0.02),
        (0.02, 0.08, 0.04),
        (-0.08, -0.02, 0.01),
        (0.04, 0.00, -0.07),
    )


def _station_positions(count: int, radius_m: float) -> list[np.ndarray]:
    count = int(count)
    if count <= 0:
        raise ValueError("station count must be positive")
    out: list[np.ndarray] = []
    for idx in range(count):
        angle = 2.0 * np.pi * idx / count
        out.append(
            np.array(
                [
                    radius_m * np.cos(angle),
                    radius_m * np.sin(angle),
                    0.025 * ((idx % 2) - 0.5),
                ],
                dtype=np.float64,
            )
        )
    return out


def _reference_coils(station_count: int, radius_m: float) -> list[ReferenceCoil]:
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    coils: list[ReferenceCoil] = []
    for station_idx, position in enumerate(_station_positions(station_count, radius_m)):
        for axis_idx, axis in enumerate(axes):
            coils.append(
                ReferenceCoil(
                    pickup_vector=axis,
                    position_m=position,
                    label=f"s{station_idx}_{axis_idx}",
                )
            )
    return coils


def _rfi_sources(
    times: np.ndarray,
    sample_rate_hz: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> list[MagneticDipoleSource]:
    positions = _source_positions()
    sources: list[MagneticDipoleSource] = []
    for idx, position in enumerate(positions):
        carrier = _rfi_carrier_hz(rng, args)
        modulation = args.modulation_hz * (1.0 + 0.25 * rng.normal())
        envelope = 1.0 + 0.22 * np.sin(2.0 * np.pi * modulation * times + rng.uniform(0, 2 * np.pi))
        envelope += args.random_modulation * _smooth_random(rng, times.size, args.random_width)
        envelope = np.clip(envelope, 0.2, None)
        phase_wander = 0.18 * _smooth_random(rng, times.size, args.random_width)
        waveform = envelope * np.cos(2.0 * np.pi * carrier * times + phase_wander + rng.uniform(0, 2 * np.pi))
        direction = rng.normal(size=3)
        sources.append(
            MagneticDipoleSource(
                position_m=position,
                moment_direction=direction,
                waveform=RFIWaveform(samples=waveform, sample_rate_hz=sample_rate_hz),
                amplitude=1.0,
            )
        )
    return sources


def _rfi_carrier_hz(rng: np.random.Generator, args: argparse.Namespace) -> float:
    if args.rfi_spectral_mode == "out-of-band":
        return float(args.carrier_khz * 1e3 + rng.normal(scale=args.carrier_spread_hz))
    if args.rfi_spectral_mode == "in-band":
        return float(args.nqr_frequency_hz + rng.normal(scale=args.overlap_spread_hz))
    raise ValueError("rfi_spectral_mode must be 'in-band' or 'out-of-band'")


def _rfi_channels(
    sources: list[MagneticDipoleSource],
    primary: ReferenceCoil,
    references: list[ReferenceCoil],
) -> tuple[np.ndarray, np.ndarray]:
    coils = [primary, *references]
    derivatives = np.vstack(
        [
            spectral_derivative(source.waveform.samples, source.waveform.sample_rate_hz)
            for source in sources
        ]
    )
    coupling = np.zeros((len(coils), len(sources)), dtype=np.float64)
    for src_idx, source in enumerate(sources):
        positions = np.vstack([coil.position_m for coil in coils])
        pattern = source.spatial_pattern(positions)
        for coil_idx, coil in enumerate(coils):
            coupling[coil_idx, src_idx] = float(coil.pickup_vector @ pattern[coil_idx])
    channels = coupling @ derivatives
    return channels[0], channels[1:]


def _trial_base(
    rng: np.random.Generator,
    args: argparse.Namespace,
    *,
    station_count: int,
) -> TrialBase:
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
    sequence_times = mask.times - args.pre_baseline_ms * 1e-3
    truth = NQRParameters(
        frequency_hz=args.nqr_frequency_hz,
        initial_amplitude=args.nqr_amplitude,
        t2e_seconds=args.t2e_ms * 1e-3,
    )
    echo_times, templates = _echo_templates(sequence_times, echo_spacing, args.num_echoes)
    clean = _synthetic_echo_train(sequence_times, echo_spacing, args.num_echoes, truth)
    clean[~mask.receive_mask] = 0.0
    echo_threshold = args.echo_threshold * float(np.max(np.abs(clean)))
    echo_mask = mask.receive_mask & (np.abs(clean) > echo_threshold)
    fit_mask = mask.receive_mask & ~echo_mask

    sources = _rfi_sources(mask.times, sample_rate_hz, rng, args)
    primary = ReferenceCoil(pickup_vector=[0.0, 0.0, 1.0], position_m=[0.0, 0.0, 0.0])
    refs = _reference_coils(station_count, args.sensor_radius_cm * 1e-2)
    primary_rfi, references = _rfi_channels(sources, primary, refs)
    scale = args.rfi_rms / _rms(primary_rfi[mask.receive_mask])
    return TrialBase(
        clean=clean,
        echo_templates=templates,
        echo_times=echo_times,
        truth=truth,
        primary_rfi=scale * primary_rfi,
        references=scale * references,
        receive_mask=mask.receive_mask,
        fit_mask=fit_mask,
        echo_mask=echo_mask,
        sample_rate_hz=sample_rate_hz,
    )


def _add_noise(
    base: TrialBase,
    initial_echo_snr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    primary_sigma, ref_sigma = _noise_sigmas_for_initial_echo_snr(
        base,
        initial_echo_snr,
        reference_noise_ratio=args.reference_noise_ratio,
    )
    primary = (
        base.clean
        + base.primary_rfi
        + rng.normal(scale=primary_sigma, size=base.clean.size)
        + 1j * rng.normal(scale=primary_sigma, size=base.clean.size)
    )
    refs = base.references + rng.normal(scale=ref_sigma, size=base.references.shape)
    return primary, refs


def _noise_sigmas_for_initial_echo_snr(
    base: TrialBase,
    initial_echo_snr: float,
    *,
    reference_noise_ratio: float,
) -> tuple[float, float]:
    snr = float(initial_echo_snr)
    if not np.isfinite(snr) or snr <= 0.0:
        raise ValueError("initial echo SNR must be positive and finite")
    response = _echo_responses(base.clean, base.echo_templates)[0]
    template_power = float(np.sum(np.abs(base.echo_templates[0]) ** 2))
    primary_sigma = abs(response) * np.sqrt(template_power) / snr
    rfi_fraction = primary_sigma / _rms(base.primary_rfi[base.receive_mask])
    ref_sigma = reference_noise_ratio * rfi_fraction * _rms(base.references[:, base.receive_mask])
    return float(primary_sigma), float(ref_sigma)


def _echo_responses(signal: np.ndarray, templates: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.complex128).reshape(-1)
    weights = np.asarray(templates, dtype=np.float64)
    denom = np.sum(weights**2, axis=1)
    if np.any(denom <= 0.0):
        raise ValueError("echo templates must have non-zero energy")
    return (weights @ values) / denom


def _estimate_nqr_parameters(signal: np.ndarray, base: TrialBase) -> NQRParameters:
    responses = _echo_responses(signal, base.echo_templates)
    magnitudes = np.abs(responses)
    valid = magnitudes > max(1e-12, 1e-4 * float(np.max(magnitudes)))
    if np.sum(valid) < 2:
        return NQRParameters(frequency_hz=np.nan, initial_amplitude=np.nan, t2e_seconds=np.nan)
    times = base.echo_times[valid]
    log_amp = np.log(magnitudes[valid])
    amp_slope, amp_intercept = np.polyfit(times, log_amp, 1)
    t2e = -1.0 / amp_slope if amp_slope < 0.0 else np.inf
    phase = np.unwrap(np.angle(responses[valid]))
    phase_slope, _ = np.polyfit(times, phase, 1)
    return NQRParameters(
        frequency_hz=float(phase_slope / (2.0 * np.pi)),
        initial_amplitude=float(np.exp(amp_intercept)),
        t2e_seconds=float(t2e),
    )


def _parameter_errors(signal: np.ndarray, base: TrialBase) -> tuple[float, float, float]:
    estimate = _estimate_nqr_parameters(signal, base)
    truth = base.truth
    if not (
        np.isfinite(estimate.frequency_hz)
        and np.isfinite(estimate.initial_amplitude)
        and np.isfinite(estimate.t2e_seconds)
    ):
        return np.nan, np.nan, np.nan
    return (
        abs(estimate.frequency_hz - truth.frequency_hz),
        abs(estimate.initial_amplitude - truth.initial_amplitude) / truth.initial_amplitude,
        abs(estimate.t2e_seconds - truth.t2e_seconds) / truth.t2e_seconds,
    )


def _echo_aware_basis(base: TrialBase, args: argparse.Namespace) -> np.ndarray:
    points = int(args.echo_aware_frequency_points)
    if points <= 0:
        raise ValueError("--echo-aware-frequency-points must be positive")
    span = float(args.echo_aware_frequency_span_hz)
    if not np.isfinite(span) or span < 0.0:
        raise ValueError("--echo-aware-frequency-span-hz must be non-negative")
    if points == 1:
        frequencies = np.array([args.nqr_frequency_hz], dtype=np.float64)
    else:
        half_span = 0.5 * span
        frequencies = np.linspace(
            args.nqr_frequency_hz - half_span,
            args.nqr_frequency_hz + half_span,
            points,
        )
    t2e_seconds = np.array(args.echo_aware_t2e_grid_ms, dtype=np.float64) * 1e-3
    if t2e_seconds.size == 0 or np.any(t2e_seconds <= 0.0):
        raise ValueError("--echo-aware-t2e-grid-ms values must be positive")

    columns = []
    for t2e in t2e_seconds:
        decay = np.exp(-base.echo_times / t2e)
        for frequency in frequencies:
            phase = np.exp(1j * 2.0 * np.pi * frequency * base.echo_times)
            column = np.sum(
                decay[:, np.newaxis] * phase[:, np.newaxis] * base.echo_templates,
                axis=0,
            )
            column = np.asarray(column, dtype=np.complex128)
            column[~base.receive_mask] = 0.0
            norm = np.sqrt(np.sum(np.abs(column[base.receive_mask]) ** 2))
            if norm > 0.0:
                columns.append(column / norm)
    if not columns:
        raise ValueError("echo-aware basis is empty")
    return np.vstack(columns)


def _apply_methods(
    primary: np.ndarray,
    references: np.ndarray,
    base: TrialBase,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    window_samples = max(32, int(round(args.window_ms * 1e-3 * base.sample_rate_hz)))
    one_station_refs = references[:3]
    echo_basis = _echo_aware_basis(base, args)
    joint = windowed_joint_signal_reference_canceller(
        primary,
        references,
        base.receive_mask,
        echo_basis,
        window_samples=window_samples,
        reference_ridge=args.ridge,
        smoothness=args.smoothness,
        signal_ridge=args.signal_ridge,
    )
    results = {
        "one_station": scalar_canceller(primary, one_station_refs, base.fit_mask, ridge=args.ridge).cleaned,
        "all_scalar": scalar_canceller(primary, references, base.fit_mask, ridge=args.ridge).cleaned,
        "windowed": windowed_ridge_fir_canceller(
            primary,
            references,
            base.fit_mask,
            window_samples=window_samples,
            ridge=args.ridge,
            smoothness=args.smoothness,
        ).cleaned,
        "adaptive": adaptive_lms_canceller(
            primary,
            references,
            base.fit_mask,
            step=args.nlms_step,
            normalized=True,
            leak=args.nlms_leak,
        ).cleaned,
        "joint": joint.signal_estimate,
    }
    return results


def _metrics(primary: np.ndarray, cleaned: np.ndarray, base: TrialBase) -> tuple[float, float, float, float]:
    suppression = rfi_suppression_db(primary, cleaned, base.receive_mask, clean_signal=base.clean)
    frequency_error, amplitude_error, t2e_error = _parameter_errors(cleaned, base)
    return suppression.suppression_db, frequency_error, amplitude_error, t2e_error


def _run_noise_study(args: argparse.Namespace) -> dict[str, np.ndarray]:
    initial_echo_snrs = np.array(args.initial_echo_snrs, dtype=np.float64)
    method_count = len(METHODS)
    suppression = np.zeros((args.trials, initial_echo_snrs.size, method_count))
    frequency_error = np.zeros_like(suppression)
    amplitude_error = np.zeros_like(suppression)
    t2e_error = np.zeros_like(suppression)
    rng = np.random.default_rng(args.seed)
    for trial in range(args.trials):
        base = _trial_base(rng, args, station_count=args.noise_stations)
        for snr_idx, initial_echo_snr in enumerate(initial_echo_snrs):
            primary, refs = _add_noise(base, initial_echo_snr, rng, args)
            cleaned = _apply_methods(primary, refs, base, args)
            for method_idx, (key, _, _) in enumerate(METHODS):
                (
                    suppression[trial, snr_idx, method_idx],
                    frequency_error[trial, snr_idx, method_idx],
                    amplitude_error[trial, snr_idx, method_idx],
                    t2e_error[trial, snr_idx, method_idx],
                ) = _metrics(primary, cleaned[key], base)
    return {
        "initial_echo_snrs": initial_echo_snrs,
        "suppression": suppression,
        "frequency_error_hz": frequency_error,
        "amplitude_relative_error": amplitude_error,
        "t2e_relative_error": t2e_error,
    }


def _run_sensor_study(args: argparse.Namespace) -> dict[str, np.ndarray]:
    station_counts = np.array(args.station_counts, dtype=int)
    max_stations = int(np.max(station_counts))
    method_keys = ("all_scalar", "windowed", "adaptive", "joint")
    suppression = np.zeros((args.trials, station_counts.size, len(method_keys)))
    frequency_error = np.zeros_like(suppression)
    amplitude_error = np.zeros_like(suppression)
    t2e_error = np.zeros_like(suppression)
    condition = np.zeros((args.trials, station_counts.size))
    rank = np.zeros((args.trials, station_counts.size))
    rng = np.random.default_rng(args.seed + 10_000)
    for trial in range(args.trials):
        base = _trial_base(rng, args, station_count=max_stations)
        primary_sigma, _ = _noise_sigmas_for_initial_echo_snr(
            base,
            args.sensor_initial_echo_snr,
            reference_noise_ratio=args.reference_noise_ratio,
        )
        rfi_fraction = primary_sigma / _rms(base.primary_rfi[base.receive_mask])
        primary_noise = rng.normal(scale=primary_sigma, size=base.clean.size) + 1j * rng.normal(
            scale=primary_sigma,
            size=base.clean.size,
        )
        primary = base.clean + base.primary_rfi + primary_noise
        for count_idx, station_count in enumerate(station_counts):
            channel_count = 3 * int(station_count)
            refs_clean = base.references[:channel_count]
            ref_sigma = (
                args.reference_noise_ratio
                * rfi_fraction
                * _rms(refs_clean[:, base.receive_mask])
            )
            refs = refs_clean + rng.normal(scale=ref_sigma, size=refs_clean.shape)
            cleaned = _apply_methods(primary, refs, base, args)
            for method_idx, key in enumerate(method_keys):
                (
                    suppression[trial, count_idx, method_idx],
                    frequency_error[trial, count_idx, method_idx],
                    amplitude_error[trial, count_idx, method_idx],
                    t2e_error[trial, count_idx, method_idx],
                ) = _metrics(primary, cleaned[key], base)
            design = reference_design_diagnostics(refs, base.fit_mask)
            condition[trial, count_idx] = design.condition_number
            rank[trial, count_idx] = design.rank
    return {
        "station_counts": station_counts,
        "method_keys": np.array(method_keys),
        "suppression": suppression,
        "frequency_error_hz": frequency_error,
        "amplitude_relative_error": amplitude_error,
        "t2e_relative_error": t2e_error,
        "condition": condition,
        "rank": rank,
    }


def _mean_ci(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(arr, axis=axis)
    n = np.sum(np.isfinite(arr), axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = 1.96 * np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(n)
    ci = np.where(n <= 1, 0.0, ci)
    return mean, ci


# Keep visualization separate from simulation for headless runs and reuse.
def _plot(
    plt,
    noise_study: dict[str, np.ndarray],
    sensor_study: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), constrained_layout=True)
    snrs = noise_study["initial_echo_snrs"]
    method_lookup = {key: (label, color) for key, label, color in METHODS}

    for method_idx, (key, label, color) in enumerate(METHODS):
        mean, ci = _mean_ci(noise_study["frequency_error_hz"][:, :, method_idx])
        axes[0, 0].errorbar(snrs, mean, yerr=ci, marker="o", color=color, label=label)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("initial echo SNR")
    axes[0, 0].set_ylabel("|frequency error| (Hz)")
    axes[0, 0].set_title("Frequency Accuracy vs SNR")
    axes[0, 0].legend(fontsize=8)

    for method_idx, (key, label, color) in enumerate(METHODS):
        mean, ci = _mean_ci(noise_study["amplitude_relative_error"][:, :, method_idx])
        axes[0, 1].errorbar(snrs, mean, yerr=ci, marker="o", color=color, label=label)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("initial echo SNR")
    axes[0, 1].set_ylabel("relative A0 error")
    axes[0, 1].set_title("Initial-Amplitude Accuracy")

    for method_idx, (key, label, color) in enumerate(METHODS):
        mean, ci = _mean_ci(noise_study["t2e_relative_error"][:, :, method_idx])
        axes[0, 2].errorbar(snrs, mean, yerr=ci, marker="o", color=color, label=label)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_xlabel("initial echo SNR")
    axes[0, 2].set_ylabel("relative T2e error")
    axes[0, 2].set_title("Decay-Time Accuracy")

    counts = sensor_study["station_counts"]
    for method_idx, key in enumerate(sensor_study["method_keys"]):
        label, color = method_lookup[str(key)]
        mean, ci = _mean_ci(sensor_study["frequency_error_hz"][:, :, method_idx])
        axes[1, 0].errorbar(counts, mean, yerr=ci, marker="o", color=color, label=label)
    axes[1, 0].set_xlabel("tri-axial reference stations")
    axes[1, 0].set_ylabel("|frequency error| (Hz)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Near-Field Frequency Scaling")
    axes[1, 0].legend(fontsize=8)

    for method_idx, key in enumerate(sensor_study["method_keys"]):
        label, color = method_lookup[str(key)]
        mean, ci = _mean_ci(sensor_study["t2e_relative_error"][:, :, method_idx])
        axes[1, 1].errorbar(counts, mean, yerr=ci, marker="o", color=color, label=label)
    axes[1, 1].set_xlabel("tri-axial reference stations")
    axes[1, 1].set_ylabel("relative T2e error")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Near-Field T2e Scaling")

    for method_idx, key in enumerate(sensor_study["method_keys"]):
        label, color = method_lookup[str(key)]
        mean, ci = _mean_ci(sensor_study["suppression"][:, :, method_idx])
        axes[1, 2].errorbar(counts, mean, yerr=ci, marker="o", color=color, label=label)
    axes[1, 2].set_xlabel("tri-axial reference stations")
    axes[1, 2].set_ylabel("RFI suppression (dB)")
    axes[1, 2].set_title("Suppression Diagnostic")

    fig.suptitle(
        f"Statistical RFI Cancellation Study ({args.trials} Monte Carlo trials, "
        f"{args.rfi_spectral_mode} RFI, 95% CI of mean)"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def _print_summary(
    noise_study: dict[str, np.ndarray],
    sensor_study: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    print("NQR RFI statistical study")
    print(f"RFI spectral mode: {args.rfi_spectral_mode}")
    print(f"trials: {args.trials}")
    low_snr_idx = 0
    high_snr_idx = -1
    for method_idx, (_, label, _) in enumerate(METHODS):
        low_freq = np.nanmean(noise_study["frequency_error_hz"][:, low_snr_idx, method_idx])
        high_freq = np.nanmean(noise_study["frequency_error_hz"][:, high_snr_idx, method_idx])
        low_t2e = np.nanmean(noise_study["t2e_relative_error"][:, low_snr_idx, method_idx])
        high_t2e = np.nanmean(noise_study["t2e_relative_error"][:, high_snr_idx, method_idx])
        print(
            f"{label:18s}: freq error {low_freq:7.2f} -> {high_freq:7.2f} Hz, "
            f"T2e rel error {low_t2e:.3f} -> {high_t2e:.3f} "
            f"from SNR={noise_study['initial_echo_snrs'][low_snr_idx]:.3g} "
            f"to {noise_study['initial_echo_snrs'][high_snr_idx]:.3g}"
        )
    best_station_freq = np.nanmean(sensor_study["frequency_error_hz"][:, -1, 1])
    first_station_freq = np.nanmean(sensor_study["frequency_error_hz"][:, 0, 1])
    best_station_t2e = np.nanmean(sensor_study["t2e_relative_error"][:, -1, 1])
    first_station_t2e = np.nanmean(sensor_study["t2e_relative_error"][:, 0, 1])
    print(
        "windowed ridge near-field scaling: frequency error "
        f"{first_station_freq:.2f} -> {best_station_freq:.2f} Hz, "
        f"T2e error {first_station_t2e:.3f} -> {best_station_t2e:.3f} from "
        f"{sensor_study['station_counts'][0]} to {sensor_study['station_counts'][-1]} stations"
    )


# Keep CLI choices together so scientific defaults are easy to find and override.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--initial-echo-snrs", type=float, nargs="+", default=[4.0, 7.0, 12.0, 20.0, 35.0])
    parser.add_argument("--station-counts", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--noise-stations", type=int, default=4)
    parser.add_argument("--sensor-initial-echo-snr", type=float, default=12.0)
    parser.add_argument("--reference-noise-ratio", type=float, default=0.5)
    parser.add_argument("--sample-rate-khz", type=float, default=80.0)
    parser.add_argument(
        "--rfi-spectral-mode",
        choices=["in-band", "out-of-band"],
        default="in-band",
        help="Use in-band RFI for a challenging estimator-overlap case or out-of-band as a null control.",
    )
    parser.add_argument("--carrier-khz", type=float, default=8.0)
    parser.add_argument("--carrier-spread-hz", type=float, default=260.0)
    parser.add_argument("--overlap-spread-hz", type=float, default=18.0)
    parser.add_argument("--modulation-hz", type=float, default=31.0)
    parser.add_argument("--random-modulation", type=float, default=0.18)
    parser.add_argument("--random-width", type=int, default=161)
    parser.add_argument("--num-echoes", type=int, default=20)
    parser.add_argument("--echo-spacing-ms", type=float, default=3.5)
    parser.add_argument("--pulse-us", type=float, default=180.0)
    parser.add_argument("--ringdown-us", type=float, default=140.0)
    parser.add_argument("--pre-baseline-ms", type=float, default=3.5)
    parser.add_argument("--post-baseline-ms", type=float, default=3.5)
    parser.add_argument("--echo-threshold", type=float, default=0.06)
    parser.add_argument("--nqr-frequency-hz", type=float, default=47.0)
    parser.add_argument("--nqr-amplitude", type=float, default=0.12)
    parser.add_argument("--t2e-ms", type=float, default=35.0)
    parser.add_argument("--sensor-radius-cm", type=float, default=11.0)
    parser.add_argument("--rfi-rms", type=float, default=0.18)
    parser.add_argument("--window-ms", type=float, default=10.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--smoothness", type=float, default=0.2)
    parser.add_argument("--signal-ridge", type=float, default=1e-3)
    parser.add_argument("--echo-aware-frequency-span-hz", type=float, default=96.0)
    parser.add_argument("--echo-aware-frequency-points", type=int, default=9)
    parser.add_argument("--echo-aware-t2e-grid-ms", type=float, nargs="+", default=[20.0, 35.0, 65.0])
    parser.add_argument("--nlms-step", type=float, default=0.22)
    parser.add_argument("--nlms-leak", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    args = _parse_args()
    if args.trials <= 0:
        raise SystemExit("--trials must be positive")
    if any(value <= 0.0 for value in args.initial_echo_snrs):
        raise SystemExit("--initial-echo-snrs values must be positive")
    if args.sensor_initial_echo_snr <= 0.0:
        raise SystemExit("--sensor-initial-echo-snr must be positive")
    plt = load_matplotlib(headless=args.output is not None)
    # The first sweep varies detector SNR at fixed reference quality; the second
    # varies reference-sensor quality at fixed detector SNR. Keeping them separate
    # reveals whether performance is noise-limited or reference-limited.
    noise_study = _run_noise_study(args)
    sensor_study = _run_sensor_study(args)
    # Plots show trial-to-trial distributions; the compact text summary reports
    # the median improvements most useful for experiment planning.
    _plot(plt, noise_study, sensor_study, args)
    _print_summary(noise_study, sensor_study, args)


if __name__ == "__main__":
    main()
