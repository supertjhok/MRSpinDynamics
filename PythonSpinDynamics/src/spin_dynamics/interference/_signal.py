"""Shared small DSP helpers for the interference package.

These operate on the *last* axis (the time/sample axis) and use FFT-based
(spectral) operators so that a single implementation serves band-limited tones,
AM carriers, chirps, and colored noise alike. Spectral differentiation and
all-pass phase shifting are exact for band-limited periodic signals; near the
record ends they carry the usual circular-convolution wrap-around, which is
acceptable for the long records the RFI cancellers operate on and is documented
where it matters.
"""

from __future__ import annotations

import numpy as np


def sample_times(num_samples: int, sample_rate_hz: float) -> np.ndarray:
    """Return the sample time grid ``[0, 1/fs, 2/fs, ...]`` in seconds."""

    num_samples = int(num_samples)
    sample_rate_hz = float(sample_rate_hz)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    return np.arange(num_samples, dtype=np.float64) / sample_rate_hz


def _angular_frequencies(num_samples: int, sample_rate_hz: float) -> np.ndarray:
    return 2.0 * np.pi * np.fft.fftfreq(num_samples, d=1.0 / sample_rate_hz)


def spectral_derivative(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Return the time derivative ``d/dt`` of ``signal`` along the last axis.

    The derivative is evaluated by multiplying the spectrum by ``i * omega``.
    A real input returns a real derivative; a complex input stays complex.
    """

    arr = np.asarray(signal)
    sample_rate_hz = float(sample_rate_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    n = arr.shape[-1]
    if n < 2:
        return np.zeros_like(arr, dtype=np.result_type(arr, np.float64))
    omega = _angular_frequencies(n, sample_rate_hz)
    spectrum = np.fft.fft(arr, axis=-1)
    derivative = np.fft.ifft(spectrum * (1j * omega), axis=-1)
    if np.isrealobj(arr):
        return np.real(derivative)
    return derivative


def spectral_phase_shift(signal: np.ndarray, phase_rad: float) -> np.ndarray:
    """Return a constant (all-pass) phase shift of a real signal.

    Positive-frequency components are rotated by ``-phase_rad`` and negative
    ones by ``+phase_rad`` (Hilbert-transform convention), so the operator is
    unit-magnitude at every frequency and reduces to identity at
    ``phase_rad = 0``.
    """

    phase_rad = float(phase_rad)
    if not np.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    arr = np.asarray(signal, dtype=np.float64)
    n = arr.shape[-1]
    if n < 2 or phase_rad == 0.0:
        return arr.copy()
    freqs = np.fft.fftfreq(n)
    rotation = np.exp(-1j * phase_rad * np.sign(freqs))
    shifted = np.fft.ifft(np.fft.fft(arr, axis=-1) * rotation, axis=-1)
    return np.real(shifted)


def spectral_lowpass(
    signal: np.ndarray,
    sample_rate_hz: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Return a brick-wall low-pass of a real signal at ``cutoff_hz``.

    A simple ideal filter used to model finite reference-coil / preamplifier
    bandwidth. Frequencies with ``|f| > cutoff_hz`` are zeroed.
    """

    sample_rate_hz = float(sample_rate_hz)
    cutoff_hz = float(cutoff_hz)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    if not np.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive and finite")
    arr = np.asarray(signal, dtype=np.float64)
    n = arr.shape[-1]
    if n < 2:
        return arr.copy()
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate_hz)
    mask = np.abs(freqs) <= cutoff_hz
    filtered = np.fft.ifft(np.fft.fft(arr, axis=-1) * mask, axis=-1)
    return np.real(filtered)
