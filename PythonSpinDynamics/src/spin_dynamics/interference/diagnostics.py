"""Mask-aware diagnostics for RFI cancellation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.interference.cancellers import _tapped_design
from spin_dynamics.noise import MatchedFilterSNRResult, estimate_matched_filter_snr


@dataclass(frozen=True)
class RFISuppressionResult:
    """RMS RFI level before and after cancellation on a selected mask."""

    before_rms: float
    after_rms: float
    suppression_db: float
    sample_count: int


@dataclass(frozen=True)
class MatchedFilterImprovementResult:
    """Matched-filter SNR before and after cancellation."""

    before: MatchedFilterSNRResult
    after: MatchedFilterSNRResult
    improvement_db: float


@dataclass(frozen=True)
class SignalBiasResult:
    """Complex gain, amplitude bias, phase bias, and residual error."""

    complex_gain: complex
    amplitude_bias: float
    phase_bias_rad: float
    error_rms: float
    sample_count: int


@dataclass(frozen=True)
class ResidualLine:
    """One prominent residual spectral line."""

    frequency_hz: float
    amplitude: float


@dataclass(frozen=True)
class ResidualSpectrumResult:
    """FFT spectrum and the largest residual spectral lines."""

    frequencies_hz: np.ndarray
    amplitudes: np.ndarray
    top_lines: tuple[ResidualLine, ...]


@dataclass(frozen=True)
class DesignMatrixDiagnostics:
    """Rank and conditioning diagnostics for a tapped reference matrix."""

    rank: int
    condition_number: float
    singular_values: np.ndarray
    sample_count: int
    column_count: int


@dataclass(frozen=True)
class SaturationDiagnostics:
    """Samples that exceed a symmetric front-end threshold."""

    threshold: float
    saturated_mask: np.ndarray
    saturated_fraction: float
    max_abs: float
    saturated_count: int


@dataclass(frozen=True)
class ReferenceNoiseInjectionResult:
    """White noise a canceller injects into the cleaned output via its references.

    ``injected_rms`` is the RMS of the noise term the canceller adds by
    subtracting a noisy reference model; ``per_channel_rms`` breaks it down by
    reference channel; ``noise_gain`` is the coefficient two-norm (the injected
    RMS per unit reference noise, independent of the assumed noise floor).
    """

    injected_rms: float
    per_channel_rms: np.ndarray
    noise_gain: float
    reference_noise_sigma: np.ndarray


def rfi_suppression_db(
    before: np.ndarray,
    after: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    clean_signal: np.ndarray | None = None,
) -> RFISuppressionResult:
    """Return RMS suppression in dB on ``mask``.

    If ``clean_signal`` is supplied, suppression is measured on residual RFI
    ``before - clean_signal`` and ``after - clean_signal``. Otherwise it compares
    the raw RMS level of ``before`` and ``after`` on the selected samples.
    """

    before_vec = _vector(before, "before")
    after_vec = _vector(after, "after")
    if after_vec.size != before_vec.size:
        raise ValueError("before and after must have the same length")
    if clean_signal is not None:
        clean = _vector(clean_signal, "clean_signal")
        if clean.size != before_vec.size:
            raise ValueError("clean_signal must match before and after")
        before_vec = before_vec - clean
        after_vec = after_vec - clean
    selected = _mask(mask, before_vec.size)
    if not np.any(selected):
        raise ValueError("mask selects no samples")
    before_rms = _rms(before_vec[selected])
    after_rms = _rms(after_vec[selected])
    return RFISuppressionResult(
        before_rms=before_rms,
        after_rms=after_rms,
        suppression_db=_ratio_db(before_rms, after_rms),
        sample_count=int(np.sum(selected)),
    )


def matched_filter_snr_improvement(
    clean_signal: np.ndarray,
    before_signals: np.ndarray,
    after_signals: np.ndarray,
    *,
    pnoise: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    noise_scale: float = 1.0,
    matched_filter: np.ndarray | None = None,
) -> MatchedFilterImprovementResult:
    """Estimate matched-filter SNR improvement from cancellation."""

    before = estimate_matched_filter_snr(
        clean_signal,
        before_signals,
        pnoise=pnoise,
        frequencies=frequencies,
        offsets=offsets,
        noise_scale=noise_scale,
        matched_filter=matched_filter,
    )
    after = estimate_matched_filter_snr(
        clean_signal,
        after_signals,
        pnoise=pnoise,
        frequencies=frequencies,
        offsets=offsets,
        noise_scale=noise_scale,
        matched_filter=matched_filter,
    )
    return MatchedFilterImprovementResult(
        before=before,
        after=after,
        improvement_db=_ratio_db(after.measured_snr, before.measured_snr),
    )


def signal_bias(
    clean_signal: np.ndarray,
    cleaned_signal: np.ndarray,
    mask: np.ndarray | None = None,
) -> SignalBiasResult:
    """Estimate amplitude and phase bias of ``cleaned_signal`` vs ``clean_signal``."""

    clean = _vector(clean_signal, "clean_signal")
    cleaned = _vector(cleaned_signal, "cleaned_signal")
    if cleaned.size != clean.size:
        raise ValueError("cleaned_signal must match clean_signal")
    selected = _mask(mask, clean.size)
    if not np.any(selected):
        raise ValueError("mask selects no samples")
    reference = clean[selected]
    observed = cleaned[selected]
    denom = np.vdot(reference, reference)
    if abs(denom) == 0.0:
        raise ValueError("clean_signal has zero energy on mask")
    gain = complex(np.vdot(reference, observed) / denom)
    error = observed - gain * reference
    return SignalBiasResult(
        complex_gain=gain,
        amplitude_bias=float(abs(gain) - 1.0),
        phase_bias_rad=float(np.angle(gain)),
        error_rms=_rms(error),
        sample_count=int(np.sum(selected)),
    )


def residual_spectral_lines(
    residual: np.ndarray,
    sample_rate_hz: float,
    mask: np.ndarray | None = None,
    *,
    top_n: int = 5,
) -> ResidualSpectrumResult:
    """Return FFT amplitudes and the largest residual spectral lines."""

    values = _vector(residual, "residual")
    sample_rate_hz = float(sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive and finite")
    selected = _mask(mask, values.size)
    if not np.any(selected):
        raise ValueError("mask selects no samples")
    gated = np.zeros(values.size, dtype=np.complex128)
    gated[selected] = values[selected] - np.mean(values[selected])
    amplitudes = np.abs(np.fft.fft(gated)) / max(1, int(np.sum(selected)))
    frequencies = np.fft.fftfreq(values.size, d=1.0 / sample_rate_hz)
    top_n = int(top_n)
    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    order = np.argsort(amplitudes)[::-1][:top_n]
    lines = tuple(
        ResidualLine(frequency_hz=float(frequencies[idx]), amplitude=float(amplitudes[idx]))
        for idx in order
    )
    return ResidualSpectrumResult(
        frequencies_hz=frequencies,
        amplitudes=amplitudes,
        top_lines=lines,
    )


def reference_design_diagnostics(
    references: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    taps: int = 1,
    rtol: float | None = None,
) -> DesignMatrixDiagnostics:
    """Return rank and condition number of the tapped reference design matrix."""

    design = _tapped_design(references, taps)
    selected = _mask(mask, design.shape[0])
    if taps > 1:
        selected[: taps - 1] = False
    if not np.any(selected):
        raise ValueError("mask selects no samples with enough FIR history")
    matrix = design[selected]
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        rank = 0
        condition = np.inf
    else:
        tol = (
            np.max(matrix.shape) * np.finfo(float).eps * float(singular[0])
            if rtol is None
            else float(rtol) * float(singular[0])
        )
        rank = int(np.sum(singular > tol))
        condition = (
            np.inf
            if singular[-1] == 0.0
            else float(singular[0] / singular[-1])
        )
    return DesignMatrixDiagnostics(
        rank=rank,
        condition_number=condition,
        singular_values=singular,
        sample_count=int(np.sum(selected)),
        column_count=matrix.shape[1],
    )


def saturation_diagnostics(
    signal: np.ndarray,
    threshold: float,
) -> SaturationDiagnostics:
    """Flag samples whose magnitude reaches or exceeds ``threshold``."""

    values = _vector(signal, "signal")
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be positive and finite")
    saturated = np.abs(values) >= threshold
    return SaturationDiagnostics(
        threshold=threshold,
        saturated_mask=saturated,
        saturated_fraction=float(np.mean(saturated)),
        max_abs=float(np.max(np.abs(values))),
        saturated_count=int(np.sum(saturated)),
    )


def reference_noise_injection(
    coefficients: np.ndarray,
    reference_noise_sigma: np.ndarray | float,
) -> ReferenceNoiseInjectionResult:
    """Estimate the reference-channel noise a canceller injects into its output.

    A reference canceller subtracts ``predicted[n] = sum_{k,l} h[k, l] X[k, n-l]``.
    Independent white measurement noise on reference channel ``k`` (standard
    deviation ``sigma_k``) therefore passes through the fitted coefficients into
    ``cleaned = primary - predicted``, adding a zero-mean noise term of variance

        var = sum_k sigma_k**2 * sum_l |h[k, l]|**2

    (samples are independent in time and across channels). This is the intrinsic
    cost of cancellation: an ill-conditioned or over-fitted reference model can
    suppress RFI while *adding* white noise. Read alongside ``rfi_suppression_db``
    and ``reference_design_diagnostics`` it says whether cancellation is a net
    win -- suppression only helps if the injected noise stays below the RFI it
    removes.

    ``coefficients`` accepts the ``(K, taps)`` array returned by the fixed
    cancellers (or a flat ``(K,)`` / ``(K*taps,)`` vector). For adaptive
    cancellers pass the final ``coefficients`` field. ``reference_noise_sigma`` is
    the per-channel noise floor (e.g. ``[coil.noise_sigma for coil in coils]``)
    or a scalar applied to every channel.
    """

    coeff = np.asarray(coefficients, dtype=np.complex128)
    if coeff.ndim == 1:
        coeff = coeff.reshape(-1, 1)
    if coeff.ndim != 2:
        raise ValueError("coefficients must have shape (K, taps), (K,), or (K*taps,)")
    if coeff.size == 0:
        raise ValueError("coefficients must contain at least one entry")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("coefficients must be finite")
    channels = coeff.shape[0]

    sigma = np.asarray(reference_noise_sigma, dtype=np.float64).reshape(-1)
    if sigma.size == 1:
        sigma = np.full(channels, float(sigma[0]))
    if sigma.size != channels:
        raise ValueError("reference_noise_sigma must be scalar or length K")
    if np.any(sigma < 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("reference_noise_sigma must be non-negative and finite")

    channel_energy = np.sum(np.abs(coeff) ** 2, axis=1)
    per_channel_rms = sigma * np.sqrt(channel_energy)
    injected_rms = float(np.sqrt(np.sum(per_channel_rms**2)))
    noise_gain = float(np.sqrt(np.sum(channel_energy)))
    return ReferenceNoiseInjectionResult(
        injected_rms=injected_rms,
        per_channel_rms=per_channel_rms,
        noise_gain=noise_gain,
        reference_noise_sigma=sigma,
    )


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.complex128).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite samples")
    return out


def _mask(mask: np.ndarray | None, size: int) -> np.ndarray:
    if mask is None:
        return np.ones(size, dtype=bool)
    out = np.asarray(mask, dtype=bool).reshape(-1)
    if out.size != size:
        raise ValueError("mask length must match the sample count")
    return out.copy()


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(values) ** 2)))


def _ratio_db(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if numerator == 0.0 and denominator == 0.0:
        return 0.0
    if denominator == 0.0:
        return np.inf
    if numerator == 0.0:
        return -np.inf
    return float(20.0 * np.log10(abs(numerator) / abs(denominator)))
