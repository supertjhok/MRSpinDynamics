"""Synthetic radio-frequency-interference (RFI) sources as vector fields.

RFI is modelled as a magnetic field ``B_RFI(r, t)`` so that reference coils and
the primary receive coil see *projections* of the same physical field. Two
spatial regimes are provided, matching the two experimentally distinct cases:

* :class:`UniformPlaneWaveSource` -- distant emitters (AM broadcast stations,
  wavelength hundreds of metres) whose field is effectively uniform over the
  probe. A small set of near-orthogonal reference axes can then span it.
* :class:`MagneticDipoleSource` -- local, near-field sources (switching power
  converters, cable pickup, re-radiating metal) with a genuine ``1/r**3`` field
  pattern and spatial gradients. A single reference station is then generally
  rank-deficient, motivating multiple vector stations.

Each source factors as a fixed spatial pattern times a scalar temporal waveform
``B(r, t) = pattern(r) * amplitude * w(t)``. The temporal waveform is a real,
band-limited signal built by the ``*_waveform`` helpers (tone, AM carrier,
chirp, colored noise, impulsive). The Faraday time derivative that turns field
into coil voltage is applied downstream in :mod:`spin_dynamics.interference.coils`.

Amplitudes are in tesla for the uniform source; for the dipole source the field
follows ``mu0/(4*pi) * [3 rhat (m.rhat) - m] / r**3`` with a phenomenological
moment scale, so magnitudes are physically shaped but not calibrated to a
specific emitter. Relative studies (suppression in dB, rank/conditioning) do not
depend on the absolute scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from spin_dynamics.interference._signal import sample_times

MU0 = 4.0e-7 * np.pi


def _unit_vector(vector: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError("direction vectors must be finite")
    norm = float(np.linalg.norm(out))
    if norm <= 0:
        raise ValueError("direction vectors must be non-zero")
    return out / norm


@dataclass(frozen=True)
class RFIWaveform:
    """A real, scalar interference waveform sampled on a shot clock."""

    samples: np.ndarray
    sample_rate_hz: float

    def __post_init__(self) -> None:
        sample_rate_hz = float(self.sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")
        samples = np.asarray(self.samples, dtype=np.float64).reshape(-1)
        if samples.size == 0:
            raise ValueError("waveform must contain at least one sample")
        if not np.all(np.isfinite(samples)):
            raise ValueError("waveform samples must be finite")
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)
        object.__setattr__(self, "samples", samples)

    @property
    def num_samples(self) -> int:
        """Number of samples in the waveform."""

        return int(self.samples.size)

    @property
    def times(self) -> np.ndarray:
        """Sample time grid in seconds."""

        return sample_times(self.num_samples, self.sample_rate_hz)


def tone_waveform(
    num_samples: int,
    sample_rate_hz: float,
    *,
    frequency_hz: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> RFIWaveform:
    """Return a single narrowband tone ``A cos(2*pi*f*t + phi)``."""

    times = sample_times(num_samples, sample_rate_hz)
    values = amplitude * np.cos(2.0 * np.pi * float(frequency_hz) * times + float(phase_rad))
    return RFIWaveform(samples=values, sample_rate_hz=sample_rate_hz)


def am_carrier_waveform(
    num_samples: int,
    sample_rate_hz: float,
    *,
    carrier_hz: float,
    modulation_hz: float,
    modulation_depth: float = 0.5,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> RFIWaveform:
    """Return an amplitude-modulated carrier (broadcast-AM model).

    ``A * (1 + m cos(2*pi*f_m*t)) * cos(2*pi*f_c*t + phi)`` with modulation
    depth ``m`` in ``[0, 1]``.
    """

    depth = float(modulation_depth)
    if not np.isfinite(depth) or depth < 0.0 or depth > 1.0:
        raise ValueError("modulation_depth must be in [0, 1]")
    times = sample_times(num_samples, sample_rate_hz)
    envelope = 1.0 + depth * np.cos(2.0 * np.pi * float(modulation_hz) * times)
    carrier = np.cos(2.0 * np.pi * float(carrier_hz) * times + float(phase_rad))
    return RFIWaveform(samples=amplitude * envelope * carrier, sample_rate_hz=sample_rate_hz)


def chirp_waveform(
    num_samples: int,
    sample_rate_hz: float,
    *,
    start_hz: float,
    stop_hz: float,
    amplitude: float = 1.0,
    phase_rad: float = 0.0,
) -> RFIWaveform:
    """Return a linear-frequency chirp sweeping ``start_hz`` -> ``stop_hz``."""

    times = sample_times(num_samples, sample_rate_hz)
    duration = times[-1] if times.size > 1 else 1.0 / float(sample_rate_hz)
    rate = (float(stop_hz) - float(start_hz)) / duration if duration > 0 else 0.0
    instantaneous_phase = 2.0 * np.pi * (float(start_hz) * times + 0.5 * rate * times**2)
    values = amplitude * np.cos(instantaneous_phase + float(phase_rad))
    return RFIWaveform(samples=values, sample_rate_hz=sample_rate_hz)


def colored_noise_waveform(
    num_samples: int,
    sample_rate_hz: float,
    *,
    amplitude: float = 1.0,
    exponent: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> RFIWaveform:
    """Return broadband noise with a power-law spectrum ``S(f) ~ f**(-exponent)``.

    ``exponent = 0`` is white noise; ``exponent = 2`` is a red/Brownian slope.
    ``amplitude`` is the RMS of the returned real waveform.
    """

    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    n = int(num_samples)
    if n <= 0:
        raise ValueError("num_samples must be positive")
    white = generator.normal(size=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate_hz))
    shaping = np.ones_like(freqs)
    nonzero = freqs > 0
    shaping[nonzero] = freqs[nonzero] ** (-0.5 * float(exponent))
    shaping[~nonzero] = 0.0  # drop DC so the field has zero mean
    shaped = np.fft.irfft(np.fft.rfft(white) * shaping, n=n)
    current_rms = float(np.sqrt(np.mean(shaped**2)))
    if current_rms > 0:
        shaped = shaped * (float(amplitude) / current_rms)
    return RFIWaveform(samples=shaped, sample_rate_hz=sample_rate_hz)


def impulsive_waveform(
    num_samples: int,
    sample_rate_hz: float,
    *,
    event_rate_hz: float,
    amplitude: float = 1.0,
    decay_seconds: float = 1.0e-5,
    ring_hz: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> RFIWaveform:
    """Return impulsive bursts (a switching-transient model).

    Poisson-timed decaying (optionally ringing) transients at mean rate
    ``event_rate_hz``, a compact model for the spiky, broadband emissions of
    switching power converters.
    """

    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(seed)
    times = sample_times(num_samples, sample_rate_hz)
    duration = times[-1] + 1.0 / float(sample_rate_hz) if times.size else 0.0
    rate = float(event_rate_hz)
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("event_rate_hz must be non-negative and finite")
    decay = float(decay_seconds)
    if not np.isfinite(decay) or decay <= 0:
        raise ValueError("decay_seconds must be positive and finite")
    expected = max(1, int(np.ceil(rate * duration)) + 1)
    n_events = int(generator.poisson(rate * duration)) if rate > 0 else 0
    values = np.zeros_like(times)
    for _ in range(min(n_events, 100 * expected)):
        t0 = float(generator.uniform(0.0, duration)) if duration > 0 else 0.0
        sign = 1.0 if generator.uniform() < 0.5 else -1.0
        dt = times - t0
        active = dt >= 0.0
        burst = np.zeros_like(times)
        burst[active] = sign * amplitude * np.exp(-dt[active] / decay)
        if ring_hz > 0.0:
            burst[active] *= np.cos(2.0 * np.pi * float(ring_hz) * dt[active])
        values = values + burst
    return RFIWaveform(samples=values, sample_rate_hz=sample_rate_hz)


@runtime_checkable
class RFISource(Protocol):
    """A vector-field RFI source evaluated at lab-frame positions."""

    @property
    def waveform(self) -> RFIWaveform: ...

    def field(self, positions: np.ndarray) -> np.ndarray:
        """Return ``B_RFI`` (T) with shape ``(P, 3, N)`` at ``P`` positions."""


@dataclass(frozen=True)
class UniformPlaneWaveSource:
    """A spatially uniform (far-field) RFI source.

    The field is ``polarization * waveform(t)`` at every position -- the
    plane-wave / distant-transmitter limit where the wavelength is large
    compared with the probe.
    """

    polarization: np.ndarray
    waveform: RFIWaveform

    def __post_init__(self) -> None:
        object.__setattr__(self, "polarization", _unit_vector(self.polarization))

    def field(self, positions: np.ndarray) -> np.ndarray:
        pts = np.asarray(positions, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[-1] != 3:
            raise ValueError("positions must have shape (P, 3)")
        w = self.waveform.samples
        pattern = self.polarization  # (3,), position-independent
        return np.broadcast_to(
            pattern[np.newaxis, :, np.newaxis] * w[np.newaxis, np.newaxis, :],
            (pts.shape[0], 3, w.size),
        ).copy()


@dataclass(frozen=True)
class MagneticDipoleSource:
    """A near-field magnetic-dipole RFI source with a ``1/r**3`` pattern.

    The moment points along ``moment_direction`` and is modulated in time by the
    waveform (scaled by ``amplitude``). Its field is spatially structured, so
    reference coils at different locations see genuinely different projections --
    the case where a single reference station fails.
    """

    position_m: np.ndarray
    moment_direction: np.ndarray
    waveform: RFIWaveform
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(position)):
            raise ValueError("position_m must be finite")
        amplitude = float(self.amplitude)
        if not np.isfinite(amplitude):
            raise ValueError("amplitude must be finite")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "moment_direction", _unit_vector(self.moment_direction))
        object.__setattr__(self, "amplitude", amplitude)

    def spatial_pattern(self, positions: np.ndarray) -> np.ndarray:
        """Return the static dipole field pattern (T) per unit moment at ``positions``."""

        pts = np.asarray(positions, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        if pts.shape[-1] != 3:
            raise ValueError("positions must have shape (P, 3)")
        r = pts - self.position_m[np.newaxis, :]
        r2 = np.sum(r**2, axis=-1)
        m = self.amplitude * self.moment_direction
        mdotr = r @ m
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_r3 = np.where(r2 > 0.0, r2**-1.5, 0.0)
            inv_r5 = np.where(r2 > 0.0, inv_r3 / r2, 0.0)
        prefactor = MU0 / (4.0 * np.pi)
        pattern = prefactor * (
            3.0 * r * mdotr[:, np.newaxis] * inv_r5[:, np.newaxis]
            - m[np.newaxis, :] * inv_r3[:, np.newaxis]
        )
        return pattern

    def field(self, positions: np.ndarray) -> np.ndarray:
        pattern = self.spatial_pattern(positions)  # (P, 3)
        w = self.waveform.samples
        return pattern[:, :, np.newaxis] * w[np.newaxis, np.newaxis, :]


def total_field(
    sources: list[RFISource] | tuple[RFISource, ...],
    positions: np.ndarray,
) -> np.ndarray:
    """Return the summed field ``(P, 3, N)`` of several sources at ``positions``.

    All sources must share the same sample clock (rate and length).
    """

    sources = tuple(sources)
    if not sources:
        raise ValueError("at least one source is required")
    rate = sources[0].waveform.sample_rate_hz
    n = sources[0].waveform.num_samples
    for src in sources[1:]:
        if src.waveform.sample_rate_hz != rate or src.waveform.num_samples != n:
            raise ValueError("all sources must share the same sample clock")
    fields = [src.field(positions) for src in sources]
    return np.sum(fields, axis=0)
