"""Reference and primary pickup coils for RFI cancellation.

A small pickup loop with effective pickup vector ``c`` and area produces a
voltage proportional to the flux rate ``d/dt (integral B.dA) ~ c.dB/dt`` (Faraday
induction). For a spatially varying field the coil samples the field at its
location, so reference coils at different positions see different projections of
a near-field source -- exactly the degrees of freedom multi-reference
cancellation exploits.

``ReferenceCoil`` bundles the pickup geometry with the non-idealities the note
requires: gain and phase error, a noise floor, an optional bandwidth limit, a
saturation (clip) level, and optional leakage of the NQR signal into the
reference channel. :func:`reference_matrix` assembles the stacked reference
observation ``X`` (shape ``(K, N)``); :func:`coupling_matrix` returns
``C = [c_1, ..., c_K]`` whose rank/conditioning tells you whether the references
can span the interference the primary coil sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.interference._signal import (
    spectral_derivative,
    spectral_lowpass,
    spectral_phase_shift,
)
from spin_dynamics.interference.sources import RFISource, total_field


def _unit_vector(vector: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError("pickup vectors must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0:
        raise ValueError("pickup vectors must be non-zero")
    return out / norm


@dataclass(frozen=True)
class ReferenceCoil:
    """A pickup coil that senses one projection of the RFI field.

    ``pickup_vector`` is the (unit) sensitivity direction ``c``; ``position_m`` is
    the coil location used to evaluate a spatially varying field. ``gain`` and
    ``phase_rad`` model transfer-function errors, ``noise_sigma`` an additive
    white noise floor, ``bandwidth_hz`` an optional brick-wall low-pass,
    ``saturation`` a symmetric clip level, and ``nqr_leakage`` a scalar coupling
    of the NQR signal into this channel (zero for a clean reference).
    """

    pickup_vector: np.ndarray
    position_m: np.ndarray = None  # type: ignore[assignment]
    gain: float = 1.0
    phase_rad: float = 0.0
    noise_sigma: float = 0.0
    bandwidth_hz: float | None = None
    saturation: float | None = None
    nqr_leakage: float = 0.0
    label: str = "ref"

    def __post_init__(self) -> None:
        pickup = _unit_vector(self.pickup_vector)
        position = (
            np.zeros(3, dtype=np.float64)
            if self.position_m is None
            else np.asarray(self.position_m, dtype=np.float64).reshape(3)
        )
        if not np.all(np.isfinite(position)):
            raise ValueError("position_m must be finite")
        gain = float(self.gain)
        phase_rad = float(self.phase_rad)
        noise_sigma = float(self.noise_sigma)
        nqr_leakage = float(self.nqr_leakage)
        if not np.isfinite(gain):
            raise ValueError("gain must be finite")
        if not np.isfinite(phase_rad):
            raise ValueError("phase_rad must be finite")
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative and finite")
        if not np.isfinite(nqr_leakage):
            raise ValueError("nqr_leakage must be finite")
        bandwidth_hz = self.bandwidth_hz
        if bandwidth_hz is not None:
            bandwidth_hz = float(bandwidth_hz)
            if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
                raise ValueError("bandwidth_hz must be positive and finite")
        saturation = self.saturation
        if saturation is not None:
            saturation = float(saturation)
            if not np.isfinite(saturation) or saturation <= 0:
                raise ValueError("saturation must be positive and finite")
        object.__setattr__(self, "pickup_vector", pickup)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "noise_sigma", noise_sigma)
        object.__setattr__(self, "bandwidth_hz", bandwidth_hz)
        object.__setattr__(self, "saturation", saturation)
        object.__setattr__(self, "nqr_leakage", nqr_leakage)
        object.__setattr__(self, "label", str(self.label))


def _induced_voltage(
    coil: ReferenceCoil,
    field_at_coil: np.ndarray,
    sample_rate_hz: float,
) -> np.ndarray:
    """Faraday voltage ``gain * c.dB/dt`` with optional bandwidth and phase."""

    flux = coil.pickup_vector @ field_at_coil  # (N,), project field onto pickup
    voltage = coil.gain * spectral_derivative(flux, sample_rate_hz)
    if coil.bandwidth_hz is not None:
        voltage = spectral_lowpass(voltage, sample_rate_hz, coil.bandwidth_hz)
    if coil.phase_rad != 0.0:
        voltage = spectral_phase_shift(voltage, coil.phase_rad)
    return voltage


def _apply_saturation(voltage: np.ndarray, saturation: float | None) -> np.ndarray:
    if saturation is None:
        return voltage
    return np.clip(voltage, -saturation, saturation)


def coil_voltage(
    coil: ReferenceCoil,
    sources: list[RFISource] | tuple[RFISource, ...],
    *,
    nqr_signal: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return the measured voltage of one coil under the given RFI sources.

    The voltage is the Faraday-induced RFI pickup plus optional NQR leakage,
    additive white noise, and clipping, in that physical order.
    """

    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one source is required")
    rate = sources[0].waveform.sample_rate_hz
    n = sources[0].waveform.num_samples
    field_at_coil = total_field(sources, coil.position_m.reshape(1, 3))[0]  # (3, N)
    voltage = _induced_voltage(coil, field_at_coil, rate)

    if coil.nqr_leakage != 0.0 and nqr_signal is not None:
        leak = np.real(np.asarray(nqr_signal, dtype=np.complex128)).reshape(-1)
        if leak.size != n:
            raise ValueError("nqr_signal length must match the source sample clock")
        voltage = voltage + coil.nqr_leakage * leak

    if coil.noise_sigma > 0.0:
        generator = rng if rng is not None else np.random.default_rng(seed)
        voltage = voltage + generator.normal(scale=coil.noise_sigma, size=n)

    return _apply_saturation(voltage, coil.saturation)


def reference_matrix(
    coils: list[ReferenceCoil] | tuple[ReferenceCoil, ...],
    sources: list[RFISource] | tuple[RFISource, ...],
    *,
    nqr_signal: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return the stacked reference observation ``X`` with shape ``(K, N)``.

    Each coil draws independent noise; a single ``seed``/``rng`` seeds the whole
    stack reproducibly.
    """

    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    coils = tuple(coils)
    if not coils:
        raise ValueError("at least one coil is required")
    generator = rng if rng is not None else np.random.default_rng(seed)
    rows = [
        coil_voltage(coil, sources, nqr_signal=nqr_signal, rng=generator)
        for coil in coils
    ]
    return np.vstack(rows)


def coupling_matrix(
    coils: list[ReferenceCoil] | tuple[ReferenceCoil, ...],
) -> np.ndarray:
    """Return ``C = [c_1, ..., c_K]`` (shape ``(3, K)``) of coil pickup vectors.

    In the spatially uniform limit the reference channels span the primary-coil
    interference iff ``C`` spans the primary pickup direction; the rank and
    condition number of ``C`` are the first-order cancellability diagnostics.
    """

    coils = tuple(coils)
    if not coils:
        raise ValueError("at least one coil is required")
    return np.column_stack([coil.pickup_vector for coil in coils])
