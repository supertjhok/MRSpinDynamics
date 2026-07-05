"""Field-referred detector abstraction for the signal-detection chain.

Every receiver model in the package assumes an inductive (Faraday) coil, whose
observable is ``dPhi/dt`` and whose noise is the coil's Johnson noise. SQUIDs and
OPMs (atomic magnetometers) instead respond to the *field* itself, with a
frequency-flat noise floor. To describe all three on one axis, this module works
in **magnetic field at the sensor** (T, T/sqrt(Hz)) rather than volts.

A :class:`Detector` is then fully specified by two frequency functions:

* ``transfer(f)`` -- a dimensionless response shape ``H(f)`` (bandpass for a tuned
  coil, ~flat for a SQUID, a Lorentzian for an RF-OPM), and
* ``field_noise_psd(f)`` -- the detector noise referred back through ``H`` to an
  equivalent field at the pickup, ``N^2(f)`` in ``T^2/Hz``.

The matched-filter detected SNR of a field-at-sensor spectrum ``S(f)`` is then
detector-agnostic::

    SNR^2 = integral |S(f)|^2 / N^2(f) df

The field the precessing sample produces at the pickup, ``S(f)``, is the same for
every detector; all detector differences live in ``N^2``. Faraday's ``B0^2``
signal collapse is *not* a special case here -- a coil's field-referred noise
``N_coil = sqrt(4kTR)/|H_coil|`` rises toward low frequency, so ``integral
|S|^2/N^2`` reproduces the quadratic scaling automatically, while a magnetometer
simply has a flat ``N^2``. See ``docs/non_inductive_detection.md``.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


class Detector(abc.ABC):
    """A frequency-domain detector referred to magnetic field at the sensor.

    Concrete detectors implement :meth:`transfer` (the dimensionless response
    shape ``H(f)``) and :meth:`field_noise_psd` (the field-referred noise PSD
    ``N^2(f)`` in ``T^2/Hz``). Both accept a frequency grid in Hz and an array
    module ``xp`` (NumPy by default; pass ``jax.numpy`` inside JAX objectives).
    """

    name: str = "detector"

    @abc.abstractmethod
    def transfer(self, freqs, *, xp=np):
        """Return the dimensionless response ``H(f)`` on the ``freqs`` grid."""

    @abc.abstractmethod
    def field_noise_psd(self, freqs, *, xp=np):
        """Return the field-referred noise PSD ``N^2(f)`` (``T^2/Hz``)."""

    def field_noise_amplitude(self, freqs, *, xp=np):
        """Field-referred noise amplitude ``N(f) = sqrt(PSD)`` (``T/sqrt(Hz)``)."""

        return xp.sqrt(self.field_noise_psd(freqs, xp=xp))


@dataclass(frozen=True)
class DetectedFieldSNR:
    """Result of a field-referred matched-filter SNR estimate."""

    snr: float
    signal_response: float
    noise_rms: float
    matched_filter: np.ndarray
    field_noise_psd: np.ndarray


def field_referred_from_output(output_psd, transfer):
    """Refer an output-referred (volts^2/Hz) noise PSD to field via ``H``.

    ``N_field^2(f) = N_out^2(f) / |H(f)|^2``. This is the bridge that lets an
    inductive probe's Johnson-noise density (computed in volts) enter the
    field-referred framework; the ``1/|H|^2`` factor is exactly what makes a
    tuned coil's field sensitivity degrade away from resonance.
    """

    out = np.asarray(output_psd, dtype=np.float64)
    h = np.asarray(transfer)
    h2 = np.abs(h) ** 2
    if np.any(~np.isfinite(h2)) or np.any(h2 <= 0):
        raise ValueError("transfer must be finite and non-zero to refer noise to field")
    return out / h2


def detected_field_snr(field_spectrum, freqs, detector, *, df=None) -> DetectedFieldSNR:
    """Matched-filter detected SNR of a field-at-sensor spectrum.

    ``field_spectrum`` is ``S(f)`` (the sample field at the pickup, detector
    independent) sampled on ``freqs`` (Hz). The optimal filter in colored noise
    is ``S*(f)/N^2(f)``; the resulting SNR is ``sqrt(integral |S|^2/N^2 df)``,
    computed here with the same normalized-matched-filter convention as
    :func:`spin_dynamics.noise.estimate_matched_filter_snr`, so an
    :class:`~spin_dynamics.detection.inductive.InductiveCoilDetector` reproduces
    that routine's ``predicted_snr`` bit-for-bit.

    ``df=None`` integrates with the trapezoid rule over ``freqs``; passing a
    scalar ``df`` uses a rectangular sum ``sum(.) * df`` instead (useful for a
    uniform bin grid or a degenerate single-frequency line).
    """

    s = np.asarray(field_spectrum, dtype=np.complex128).reshape(-1)
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    if f.size != s.size:
        raise ValueError("freqs must match field_spectrum")
    if s.size == 0:
        raise ValueError("field_spectrum must be non-empty")
    if np.any(~np.isfinite(f)):
        raise ValueError("freqs must be finite")

    n2 = np.asarray(detector.field_noise_psd(f), dtype=np.float64).reshape(-1)
    if n2.size != s.size:
        raise ValueError("detector.field_noise_psd must match field_spectrum")
    if np.any(~np.isfinite(n2)) or np.any(n2 <= 0):
        raise ValueError("field_noise_psd must be finite and positive")

    mf_unnorm = np.conj(s) / n2
    norm = np.sqrt(_integrate(np.abs(mf_unnorm) ** 2, f, df))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("matched-filter norm must be finite and positive")
    mf = mf_unnorm / norm

    signal_response = float(np.real(_integrate(s * mf, f, df)))
    noise_rms = float(np.sqrt(np.real(_integrate(n2 * np.abs(mf) ** 2, f, df))))
    snr = np.inf if noise_rms == 0 else signal_response / noise_rms
    return DetectedFieldSNR(
        snr=float(snr),
        signal_response=signal_response,
        noise_rms=noise_rms,
        matched_filter=mf,
        field_noise_psd=n2,
    )


def _integrate(values: np.ndarray, freqs: np.ndarray, df: float | None):
    """Integrate ``values`` over ``freqs`` (trapezoid) or as ``sum * df``.

    Complex-safe: preserves the dtype of ``values`` so a complex integrand (the
    matched-filter signal response) is not silently truncated to its real part.
    """

    vals = np.asarray(values)
    if df is not None:
        step = float(df)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("df must be finite and positive")
        return np.sum(vals) * step
    if vals.size == 1:
        return vals.reshape(-1)[0]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapz(vals, freqs)
