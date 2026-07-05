"""Optically pumped (atomic) magnetometer as a field-referred :class:`Detector`.

Like a SQUID, an OPM responds to the field itself and reaches sub-fT/sqrt(Hz)
sensitivity without cryogenics. Unlike a SQUID, its response is band-limited by
the atomic magnetic resonance: the vapour-cell spins have a finite relaxation
rate, so the sensor acts as a Lorentzian filter of half-width ``bandwidth_hz``.
That roll-off is the defining feature -- it is what makes OPM performance depend
on where the signal sits relative to the sensor's operating point, and omitting
it would badly overstate OPM SNR at MHz NQR frequencies.

Two operating modes:

* ``"serf"`` -- spin-exchange-relaxation-free, operated near zero field. The
  response is a **low-pass** centred at DC (``H = 1/(1 + i f/df)``); sensitivity
  is highest (0.16-0.5 fT/sqrt(Hz)) but the bandwidth is low (<~ kHz), so SERF
  OPMs suit zero/ultra-low-field NMR, not MHz NQR.
* ``"rf"`` -- a radio-frequency (Mx) magnetometer whose bias field tunes the
  atomic Larmor frequency onto a ``carrier_hz`` (the NQR/precession line). The
  response is a **band-pass** centred there (``H = 1/(1 + i (f - fc)/df)``);
  ~0.24 fT/sqrt(Hz) has been demonstrated at 423 kHz detecting the 14N NQR of
  ammonium nitrate (Savukov et al.; US 7,521,928).

Field-referred noise follows the same rule as every detector,
``N^2 = S0^2 / |H|^2``: flat at the floor ``S0`` on resonance, rising as the
atomic response rolls off,

    ``N^2(f) = S0^2 * (1 + ((f - f_center)/df)^2)``   (T^2/Hz),

with ``f_center = 0`` (SERF) or ``carrier_hz`` (RF).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Detector
from .gradiometer import Gradiometer


@dataclass(frozen=True)
class OPMMagnetometer(Detector):
    """Atomic magnetometer with a Lorentzian atomic-response bandwidth.

    ``field_noise_T_per_rtHz`` is the on-resonance (in-band) field-noise floor
    ``S0``. ``bandwidth_hz`` is the atomic-response half-width ``df`` (the
    relaxation-limited sensor bandwidth). ``mode`` is ``"rf"`` (band-pass at
    ``carrier_hz``) or ``"serf"`` (low-pass at DC). ``cell_temperature_k`` is
    carried as context only.
    """

    field_noise_T_per_rtHz: float = 0.24e-15
    bandwidth_hz: float = 300.0
    mode: str = "rf"
    carrier_hz: float = 0.0
    name: str = "OPM magnetometer"
    cell_temperature_k: float = 450.0
    pickup: Gradiometer | None = None

    def __post_init__(self) -> None:
        s0 = float(self.field_noise_T_per_rtHz)
        bw = float(self.bandwidth_hz)
        mode = str(self.mode).lower()
        carrier = float(self.carrier_hz)
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("field_noise_T_per_rtHz must be positive and finite")
        if not np.isfinite(bw) or bw <= 0:
            raise ValueError("bandwidth_hz must be positive and finite")
        if mode not in {"rf", "serf"}:
            raise ValueError("mode must be 'rf' or 'serf'")
        if not np.isfinite(carrier) or carrier < 0:
            raise ValueError("carrier_hz must be non-negative and finite")
        if mode == "rf" and carrier <= 0:
            raise ValueError("rf mode requires a positive carrier_hz")
        if self.pickup is not None and not isinstance(self.pickup, Gradiometer):
            raise ValueError("pickup must be a Gradiometer or None")
        object.__setattr__(self, "field_noise_T_per_rtHz", s0)
        object.__setattr__(self, "bandwidth_hz", bw)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "carrier_hz", carrier)
        object.__setattr__(self, "name", str(self.name))

    @property
    def center_hz(self) -> float:
        """Response centre: ``carrier_hz`` for RF mode, ``0`` for SERF."""

        return self.carrier_hz if self.mode == "rf" else 0.0

    def transfer(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        detune = (f - self.center_hz) / self.bandwidth_hz
        return 1.0 / (1.0 + 1j * detune)

    def field_noise_psd(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        detune = (f - self.center_hz) / self.bandwidth_hz
        s0 = self.field_noise_T_per_rtHz
        return (s0 * s0) * (1.0 + detune * detune)

    @classmethod
    def serf(
        cls,
        *,
        field_noise_T_per_rtHz: float = 0.16e-15,
        bandwidth_hz: float = 200.0,
        **kwargs,
    ) -> "OPMMagnetometer":
        """Zero-field SERF OPM (sub-fT floor, low bandwidth)."""

        return cls(
            field_noise_T_per_rtHz=field_noise_T_per_rtHz,
            bandwidth_hz=bandwidth_hz,
            mode="serf",
            name="OPM (SERF)",
            **kwargs,
        )

    @classmethod
    def rf(
        cls,
        carrier_hz: float,
        *,
        field_noise_T_per_rtHz: float = 0.24e-15,
        bandwidth_hz: float = 300.0,
        **kwargs,
    ) -> "OPMMagnetometer":
        """RF/Mx OPM tuned to ``carrier_hz`` (Savukov 14N-NQR regime)."""

        return cls(
            field_noise_T_per_rtHz=field_noise_T_per_rtHz,
            bandwidth_hz=bandwidth_hz,
            mode="rf",
            carrier_hz=carrier_hz,
            name="OPM (RF)",
            **kwargs,
        )
