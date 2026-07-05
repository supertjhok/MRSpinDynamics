"""Inductive (Faraday) coil as a field-referred :class:`Detector`.

This adapter expresses the package's existing tuned/untuned/matched probe noise
model in the field-referred framework of :mod:`spin_dynamics.detection.base`. Its
purpose in this first increment is a **regression guard**: fed the same
output-referred noise density that the probe SNR routines use, it reproduces
:func:`spin_dynamics.noise.estimate_matched_filter_snr`'s ``predicted_snr``
exactly, proving the new detector layer changes no numbers.

An inductive coil's field-referred noise is ``N_field^2 = N_out^2 / |H|^2``,
where ``N_out^2`` is the coil Johnson + amplifier density (volts^2/Hz) and ``H``
is the receiver transfer (volts per field). With the default flat ``H = 1`` the
field- and output-referred SNR coincide; pass ``rx_transfer`` to model the tuned
transfer whose ``1/|H|^2`` is what makes coil field-sensitivity degrade off
resonance (the mechanism behind the SQUID/OPM crossover in later increments).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spin_dynamics.noise import (
    matched_probe_output_noise_density,
    tuned_probe_output_noise_density,
    untuned_probe_output_noise_density,
)

from .base import Detector


@dataclass(frozen=True)
class InductiveCoilDetector(Detector):
    """Field-referred adapter over an inductive probe's output-noise density.

    ``freqs_hz`` and ``output_noise_psd`` are the sampled receiver noise density
    (volts^2/Hz), as returned by the ``*_probe_output_noise_density`` helpers in
    :mod:`spin_dynamics.noise`. ``rx_transfer`` is the optional complex receiver
    transfer ``H(f)`` on the same grid (defaults to flat ``1``). Queries at
    arbitrary frequencies interpolate (real/imag parts) onto the requested grid.
    """

    freqs_hz: np.ndarray
    output_noise_psd: np.ndarray
    rx_transfer: np.ndarray | None = None
    name: str = "inductive coil"
    _f: np.ndarray = field(default=None, repr=False)
    _psd: np.ndarray = field(default=None, repr=False)
    _h: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        f = np.asarray(self.freqs_hz, dtype=np.float64).reshape(-1)
        psd = np.asarray(self.output_noise_psd, dtype=np.float64).reshape(-1)
        if f.size < 2 or f.size != psd.size:
            raise ValueError("freqs_hz and output_noise_psd must match and have >= 2 points")
        if np.any(~np.isfinite(f)):
            raise ValueError("freqs_hz must be finite")
        if np.any(~np.isfinite(psd)) or np.any(psd <= 0):
            raise ValueError("output_noise_psd must be finite and positive")
        if self.rx_transfer is None:
            h = np.ones(f.size, dtype=np.complex128)
        else:
            h = np.asarray(self.rx_transfer, dtype=np.complex128).reshape(-1)
            if h.size != f.size:
                raise ValueError("rx_transfer must match freqs_hz")
            if np.any(~np.isfinite(h)) or np.any(h == 0):
                raise ValueError("rx_transfer must be finite and non-zero")
        order = np.argsort(f)
        object.__setattr__(self, "_f", f[order])
        object.__setattr__(self, "_psd", psd[order])
        object.__setattr__(self, "_h", h[order])
        object.__setattr__(self, "name", str(self.name))

    def transfer(self, freqs, *, xp=np):
        f = np.asarray(freqs, dtype=np.float64).reshape(-1)
        re = np.interp(f, self._f, self._h.real)
        im = np.interp(f, self._f, self._h.imag)
        h = re + 1j * im
        return xp.asarray(h) if xp is not np else h

    def field_noise_psd(self, freqs, *, xp=np):
        f = np.asarray(freqs, dtype=np.float64).reshape(-1)
        psd = np.interp(f, self._f, self._psd)
        h = self.transfer(f, xp=np)
        n2 = psd / (np.abs(h) ** 2)
        return xp.asarray(n2) if xp is not np else n2

    @classmethod
    def from_output_density(
        cls,
        freqs_hz,
        output_noise_psd,
        *,
        rx_transfer=None,
        name: str = "inductive coil",
    ) -> "InductiveCoilDetector":
        """Build directly from a sampled ``(freqs, density[, H])``."""

        return cls(
            freqs_hz=np.asarray(freqs_hz),
            output_noise_psd=np.asarray(output_noise_psd),
            rx_transfer=None if rx_transfer is None else np.asarray(rx_transfer),
            name=name,
        )

    @classmethod
    def tuned(
        cls,
        sp: Mapping[str, Any] | Any,
        pp: Mapping[str, Any] | Any,
        *,
        rx_transfer=None,
    ) -> "InductiveCoilDetector":
        """Adapter over :func:`noise.tuned_probe_output_noise_density`."""

        psd, f = tuned_probe_output_noise_density(sp, pp)
        return cls.from_output_density(f, psd, rx_transfer=rx_transfer, name="tuned coil")

    @classmethod
    def untuned(
        cls,
        sp: Mapping[str, Any] | Any,
        pp: Mapping[str, Any] | Any,
        *,
        rx_transfer=None,
    ) -> "InductiveCoilDetector":
        """Adapter over :func:`noise.untuned_probe_output_noise_density`."""

        psd, f = untuned_probe_output_noise_density(sp, pp)
        return cls.from_output_density(f, psd, rx_transfer=rx_transfer, name="untuned coil")

    @classmethod
    def matched(
        cls,
        sp: Mapping[str, Any] | Any,
        pp: Mapping[str, Any] | Any,
        *,
        tf1=None,
        rx_transfer=None,
    ) -> "InductiveCoilDetector":
        """Adapter over :func:`noise.matched_probe_output_noise_density`."""

        psd, f = matched_probe_output_noise_density(sp, pp, tf1=tf1)
        return cls.from_output_density(f, psd, rx_transfer=rx_transfer, name="matched coil")


@dataclass(frozen=True)
class IdealFaradayCoil(Detector):
    """Idealized inductive coil with ``1/f`` field-referred noise.

    A Faraday coil converts field to voltage with a responsivity proportional to
    frequency (``EMF = -dPhi/dt``, so ``H(f) = f / f_ref``), while its intrinsic
    voltage noise is, to leading order, frequency-flat across the tuned band. The
    field-referred noise therefore rises toward low frequency:

        ``N(f) = N_ref * (f_ref / |f|)``   (T/sqrt(Hz)),

    i.e. ``field_noise_psd = N_ref^2 * (f_ref/|f|)^2``. This is the analytic
    baseline for the SQUID/OPM crossover: with a prepolarized sample (signal set
    by ``Bp``, not ``B0``) it reproduces Faraday SNR ``~ B0`` and, with a thermal
    sample, SNR ``~ B0^2`` -- exactly the scalings in Clarke 2007, Summary Pt. 2.

    This is the leading-order (``EMF ~ omega``) model; a fuller coil accounts for
    skin-effect resistance ``R ~ sqrt(f)`` and the sample-noise-dominated regime.
    For a measured circuit use :class:`InductiveCoilDetector` instead.
    """

    field_noise_T_per_rtHz_ref: float
    f_ref_hz: float
    name: str = "ideal Faraday coil"

    def __post_init__(self) -> None:
        ref = float(self.field_noise_T_per_rtHz_ref)
        f_ref = float(self.f_ref_hz)
        if not np.isfinite(ref) or ref <= 0:
            raise ValueError("field_noise_T_per_rtHz_ref must be positive and finite")
        if not np.isfinite(f_ref) or f_ref <= 0:
            raise ValueError("f_ref_hz must be positive and finite")
        object.__setattr__(self, "field_noise_T_per_rtHz_ref", ref)
        object.__setattr__(self, "f_ref_hz", f_ref)
        object.__setattr__(self, "name", str(self.name))

    def transfer(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        return (f / self.f_ref_hz).astype(xp.asarray(1.0 + 0j).dtype)

    def field_noise_psd(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        af = xp.abs(f)
        positive = af[af > 0]
        floor = positive.min() if positive.size else xp.asarray(1.0)
        af = xp.where(af > 0, af, floor)
        ref = self.field_noise_T_per_rtHz_ref
        return (ref * ref) * (self.f_ref_hz / af) ** 2

