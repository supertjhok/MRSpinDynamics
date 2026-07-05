"""SQUID magnetometer as a field-referred :class:`Detector`.

A SQUID coupled to an untuned superconducting flux transformer is a broadband
*field* sensor: its response is independent of frequency and its noise floor is
flat (plus a ``1/f`` upturn at low frequency). This is the physics that makes
ultra-low-field NMR/MRI practical -- unlike a Faraday coil, whose field-referred
noise rises as ``1/f`` toward low frequency (see
:class:`~spin_dynamics.detection.inductive.IdealFaradayCoil`), the SQUID floor
does not, so the two cross at a few MHz and the SQUID wins below it.

Reference: Clarke, Hatridge & Moessle, *Annu. Rev. Biomed. Eng.* **9**, 389
(2007). An optimized flux transformer reaches ~1 fT/sqrt(Hz), with projected
upgrades to 0.4 and 0.2 fT/sqrt(Hz).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import Detector
from .gradiometer import Gradiometer


@dataclass(frozen=True)
class SQUIDMagnetometer(Detector):
    """Untuned SQUID magnetometer: flat field-noise floor with a ``1/f`` knee.

    ``field_noise_T_per_rtHz`` is the white (high-frequency) field-noise floor
    ``S0`` in T/sqrt(Hz). ``one_over_f_knee_hz`` is the corner ``f_k`` of the
    low-frequency upturn, giving the power spectral density

        ``N^2(f) = S0^2 * (1 + f_k / |f|)``   (T^2/Hz),

    the standard SQUID ``1/f`` form; set ``f_k = 0`` for a purely white floor.
    The transfer is flat (``H = 1``): the untuned flux transformer responds to
    field equally at all frequencies of interest. ``bath_temperature_k`` and
    ``flux_transformer_turns`` are carried as context only (not dynamically
    modelled here).
    """

    field_noise_T_per_rtHz: float = 1.0e-15
    one_over_f_knee_hz: float = 0.0
    name: str = "SQUID magnetometer"
    bath_temperature_k: float = 4.2
    flux_transformer_turns: int | None = None
    pickup: Gradiometer | None = None

    def __post_init__(self) -> None:
        s0 = float(self.field_noise_T_per_rtHz)
        knee = float(self.one_over_f_knee_hz)
        if not np.isfinite(s0) or s0 <= 0:
            raise ValueError("field_noise_T_per_rtHz must be positive and finite")
        if not np.isfinite(knee) or knee < 0:
            raise ValueError("one_over_f_knee_hz must be non-negative and finite")
        if self.pickup is not None and not isinstance(self.pickup, Gradiometer):
            raise ValueError("pickup must be a Gradiometer or None")
        object.__setattr__(self, "field_noise_T_per_rtHz", s0)
        object.__setattr__(self, "one_over_f_knee_hz", knee)
        object.__setattr__(self, "name", str(self.name))

    def transfer(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        return xp.ones(f.shape, dtype=xp.asarray(1.0 + 0j).dtype)

    def field_noise_psd(self, freqs, *, xp=np):
        f = xp.asarray(freqs)
        s0 = self.field_noise_T_per_rtHz
        white = s0 * s0
        if self.one_over_f_knee_hz == 0.0:
            return white * xp.ones(f.shape)
        # Guard f = 0 (the 1/f term diverges at DC); clamp to the smallest
        # positive |f| on the grid so the PSD stays finite and positive.
        af = xp.abs(f)
        positive = af[af > 0]
        floor = positive.min() if positive.size else xp.asarray(1.0)
        af = xp.where(af > 0, af, floor)
        return white * (1.0 + self.one_over_f_knee_hz / af)

    @classmethod
    def berkeley_2007(cls, **kwargs) -> "SQUIDMagnetometer":
        """1 fT/sqrt(Hz) untuned flux transformer (Clarke 2007 system)."""

        return cls(field_noise_T_per_rtHz=1.0e-15, name="SQUID (1 fT/rtHz)", **kwargs)

    @classmethod
    def projected(cls, **kwargs) -> "SQUIDMagnetometer":
        """0.2 fT/sqrt(Hz) projected upgrade (Clarke 2007 Future Outlook)."""

        return cls(field_noise_T_per_rtHz=0.2e-15, name="SQUID (0.2 fT/rtHz)", **kwargs)
