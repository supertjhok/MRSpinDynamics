"""End-to-end spatial detected SNR through a pickup geometry.

Ties a frequency-domain :class:`~spin_dynamics.detection.base.Detector` (its
noise floor ``N^2(f)``) to a :class:`~spin_dynamics.detection.gradiometer.Gradiometer`
pickup (its spatial coupling), so a distributed sample and distant ambient
sources are handled together:

* **signal** -- the sample field at the sensor is the pickup-weighted sum of the
  source fields, ``S(f) = sum_k g(r_k) m_k(f)``, where ``g`` is the pickup's
  normalized (dimensionless) reciprocal sensitivity and ``m_k(f)`` is the field
  source ``k`` would present at the reference point;
* **noise** -- the sensor's own floor ``N^2_sensor(f)`` plus any ambient field
  sources coupling through the *same* pickup, ``sum_j g(r_j)^2 P_j(f)``. A
  gradiometer's ``g(r_j) << 1`` for distant ``r_j`` is exactly how it rejects
  ambient noise (Clarke's ~1000x common-mode rejection) while a sample at the
  bottom loop couples with ``g ~ 1``.

The detected SNR is then the matched-filter value against the augmented floor.
Couplings are normalized to the pickup's bottom-loop reference (see
:meth:`Gradiometer.normalized_sensitivity`), so signal and noise share field
units and different pickups are compared fairly at equal sample coupling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import DetectedFieldSNR, snr_from_field_noise_psd


@dataclass(frozen=True)
class AmbientFieldSource:
    """A stochastic ambient field source with a field-noise PSD.

    ``psd`` is the field-noise PSD (``T^2/Hz``) the source would couple *at the
    pickup reference point*, on the signal frequency grid. ``position_m`` is a
    3-vector (metres) for a localized source -- it couples through
    ``g(position)^2 * psd`` via :meth:`Gradiometer.normalized_sensitivity`. Leave
    ``position_m=None`` for a spatially **uniform** ambient field, which couples
    through ``uniform_coupling()^2 * psd`` (``~1`` for a magnetometer, ``~0`` for
    a balanced gradiometer -- the common-mode rejection Clarke's system exploits).
    """

    psd: np.ndarray
    position_m: tuple[float, float, float] | None = None

    def coupling(self, pickup, *, reference=None) -> float:
        if self.position_m is None:
            return pickup.uniform_coupling()
        pos = np.asarray(self.position_m, dtype=np.float64).reshape(1, 3)
        return float(pickup.normalized_sensitivity(pos, reference=reference)[0])


def _require_pickup(detector, pickup):
    resolved = pickup if pickup is not None else getattr(detector, "pickup", None)
    if resolved is None:
        raise ValueError("a pickup Gradiometer is required (pass pickup= or set detector.pickup)")
    return resolved


def pickup_signal_spectrum(pickup, positions, moment_spectra, *, reference=None) -> np.ndarray:
    """Pickup-weighted field-at-sensor spectrum ``S(f) = sum_k g(r_k) m_k(f)``.

    ``positions`` is ``(K, 3)`` in metres; ``moment_spectra`` is ``(K, F)`` (or
    ``(F,)`` for a single source) of complex per-source field spectra referred to
    the pickup reference point. Returns the ``(F,)`` net spectrum.
    """

    pts = np.atleast_2d(np.asarray(positions, dtype=np.float64))
    if pts.shape[-1] != 3:
        raise ValueError("positions must have trailing dimension 3")
    m = np.atleast_2d(np.asarray(moment_spectra, dtype=np.complex128))
    if m.shape[0] != pts.shape[0]:
        raise ValueError("moment_spectra must have one row per position")
    g = pickup.normalized_sensitivity(pts, reference=reference)  # (K,)
    return g.astype(np.complex128) @ m


def spatial_field_noise_psd(
    detector, freqs, ambient_sources, *, pickup=None, reference=None
) -> np.ndarray:
    """Sensor floor augmented by ambient sources coupling through the pickup.

    ``N^2(f) = detector.field_noise_psd(f) + sum_j g(r_j)^2 P_j(f)``.
    """

    resolved = _require_pickup(detector, pickup)
    f = np.asarray(freqs, dtype=np.float64).reshape(-1)
    n2 = np.asarray(detector.field_noise_psd(f), dtype=np.float64).reshape(-1).copy()
    for source in ambient_sources or ():
        g = source.coupling(resolved, reference=reference)
        psd = np.asarray(source.psd, dtype=np.float64).reshape(-1)
        if psd.size != f.size:
            raise ValueError("ambient source psd must match freqs")
        n2 = n2 + (g * g) * psd
    return n2


def detected_field_snr_spatial(
    detector,
    positions,
    moment_spectra,
    freqs,
    *,
    ambient_sources=None,
    pickup=None,
    reference=None,
    df=None,
) -> DetectedFieldSNR:
    """Detected SNR of a distributed sample through ``detector`` + its pickup.

    Computes the pickup-weighted signal spectrum and the sensor floor augmented
    by any ``ambient_sources``, then returns the matched-filter SNR. The pickup
    is taken from ``detector.pickup`` unless ``pickup=`` is given.
    """

    resolved = _require_pickup(detector, pickup)
    signal = pickup_signal_spectrum(resolved, positions, moment_spectra, reference=reference)
    if ambient_sources:
        n2 = spatial_field_noise_psd(
            detector, freqs, ambient_sources, pickup=resolved, reference=reference
        )
    else:
        f = np.asarray(freqs, dtype=np.float64).reshape(-1)
        n2 = np.asarray(detector.field_noise_psd(f), dtype=np.float64).reshape(-1)
    return snr_from_field_noise_psd(signal, freqs, n2, df=df)
