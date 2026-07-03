"""Active feedforward RFI cancellation with a compensation coil.

The digital cancellers in :mod:`~spin_dynamics.interference.cancellers` subtract an
RFI estimate *after* digitisation. That is powerless once the interference
saturates the low-noise amplifier or ADC: a clipped waveform has already lost the
information a linear model would need to reconstruct it. Active cancellation
instead drives a **compensation coil** that produces a field opposing the RFI at
the primary pickup, so the receiver sees the residual -- small enough not to clip
-- and digitises cleanly.

This module models that analog path. A fitted linear model (any
:class:`~spin_dynamics.interference.cancellers.LinearCancellerModel`, from the
gated-ridge or robust fit) supplies the *commanded* compensation from the
continuously-running reference channels. The :class:`CompensationActuator` turns
that command into the field actually delivered at the primary, subject to the
non-idealities that bound real feedforward:

* **latency** -- the drive can only use past reference samples, so a causal delay
  limits how much of the high-frequency RFI is cancelled (the residual of a tone
  at angular frequency ``w`` scales like ``|1 - g e^{-i w L}|``),
* **finite drive range** -- the coil/amplifier can only produce ``+/- max_field``;
  RFI beyond that is only partially cancelled,
* **gain/phase mismatch and bandwidth** of the analog chain, and
* **drive noise** injected by the compensation path itself.

The pipeline is: command = ``model.predict(references)``; delivered =
``actuator.realize(command)``; analog residual = ``primary - delivered``; then the
ADC clips the residual. Comparing this with the digital-only path -- clip first,
then subtract -- is the "digital cancellation is too late" story made concrete.
Because the digitised active residual is again linear, a digital canceller can be
run on it afterwards for the last few dB.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.interference._signal import spectral_lowpass, spectral_phase_shift


@dataclass(frozen=True)
class CompensationActuator:
    """Analog feedforward path from a commanded field to the field at the primary.

    ``gain`` and ``phase_rad`` are the amplitude and constant-phase mismatch of
    the realized field relative to the command (``gain = 1``, ``phase_rad = 0`` is
    ideal). ``latency_samples`` is the causal delay of the drive electronics,
    ``bandwidth_hz`` an optional finite actuator bandwidth, ``max_field`` the
    largest field the coil/amplifier can produce (a symmetric clip on the
    command), and ``noise_sigma`` an additive white noise the compensation path
    injects at the primary.
    """

    gain: float = 1.0
    phase_rad: float = 0.0
    latency_samples: int = 0
    bandwidth_hz: float | None = None
    max_field: float | None = None
    noise_sigma: float = 0.0
    label: str = "comp"

    def __post_init__(self) -> None:
        gain = float(self.gain)
        phase_rad = float(self.phase_rad)
        latency = int(self.latency_samples)
        noise_sigma = float(self.noise_sigma)
        if not np.isfinite(gain):
            raise ValueError("gain must be finite")
        if not np.isfinite(phase_rad):
            raise ValueError("phase_rad must be finite")
        if latency < 0:
            raise ValueError("latency_samples must be non-negative")
        if not np.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative and finite")
        bandwidth_hz = self.bandwidth_hz
        if bandwidth_hz is not None:
            bandwidth_hz = float(bandwidth_hz)
            if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0:
                raise ValueError("bandwidth_hz must be positive and finite")
        max_field = self.max_field
        if max_field is not None:
            max_field = float(max_field)
            if not np.isfinite(max_field) or max_field <= 0:
                raise ValueError("max_field must be positive and finite")
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "phase_rad", phase_rad)
        object.__setattr__(self, "latency_samples", latency)
        object.__setattr__(self, "bandwidth_hz", bandwidth_hz)
        object.__setattr__(self, "max_field", max_field)
        object.__setattr__(self, "noise_sigma", noise_sigma)
        object.__setattr__(self, "label", str(self.label))

    def realize(
        self,
        command: np.ndarray,
        sample_rate_hz: float,
        *,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Return the field actually delivered at the primary for a ``command``.

        The command is clipped to ``max_field``, delayed by ``latency_samples``,
        scaled by ``gain``, band-limited, phase-shifted, and has drive noise
        added -- in that physical order.
        """

        if rng is not None and seed is not None:
            raise ValueError("provide either rng or seed, not both")
        drive = np.real(np.asarray(command, dtype=np.complex128)).astype(np.float64)
        drive = drive.reshape(-1)
        if drive.size == 0:
            raise ValueError("command must contain at least one sample")
        if not np.all(np.isfinite(drive)):
            raise ValueError("command samples must be finite")
        sample_rate_hz = float(sample_rate_hz)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive and finite")

        if self.max_field is not None:
            drive = np.clip(drive, -self.max_field, self.max_field)
        if self.latency_samples > 0:
            delayed = np.zeros_like(drive)
            if self.latency_samples < drive.size:
                delayed[self.latency_samples :] = drive[: -self.latency_samples]
            drive = delayed
        field = self.gain * drive
        if self.bandwidth_hz is not None:
            field = spectral_lowpass(field, sample_rate_hz, self.bandwidth_hz)
        if self.phase_rad != 0.0:
            field = spectral_phase_shift(field, self.phase_rad)
        if self.noise_sigma > 0.0:
            generator = rng if rng is not None else np.random.default_rng(seed)
            field = field + generator.normal(scale=self.noise_sigma, size=field.size)
        return field


@dataclass(frozen=True)
class ActiveCancellationResult:
    """Outcome of an analog feedforward cancellation before and after the ADC."""

    command: np.ndarray
    compensation: np.ndarray
    analog_residual: np.ndarray
    digitized: np.ndarray
    adc_saturation: float | None
    clipped_fraction: float


def feedforward_cancel(
    primary: np.ndarray,
    references: np.ndarray,
    model,
    actuator: CompensationActuator,
    sample_rate_hz: float,
    *,
    adc_saturation: float | None = None,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> ActiveCancellationResult:
    """Cancel RFI at the primary with a compensation coil before digitisation.

    ``model`` is a fitted
    :class:`~spin_dynamics.interference.cancellers.LinearCancellerModel` (from
    ``fit_gated_ridge_fir`` or ``fit_robust_fir``) that maps the reference
    channels to the RFI seen by the primary; its prediction is the *commanded*
    compensation field. ``actuator`` realizes that command imperfectly, the
    delivered field is subtracted from ``primary`` in the analog domain, and the
    resulting residual is clipped by the ADC at ``adc_saturation`` (``None`` for
    an ideal digitiser).

    Contrast with the digital-only path ``model.apply(np.clip(primary, -s, s),
    references)``, which subtracts *after* the same clip and therefore cannot
    recover samples the RFI already saturated.
    """

    primary_arr = np.real(np.asarray(primary, dtype=np.complex128)).astype(np.float64)
    primary_arr = primary_arr.reshape(-1)
    if primary_arr.size == 0:
        raise ValueError("primary must contain at least one sample")
    if not np.all(np.isfinite(primary_arr)):
        raise ValueError("primary samples must be finite")

    command = np.real(model.predict(references)).astype(np.float64).reshape(-1)
    if command.size != primary_arr.size:
        raise ValueError("model prediction and primary lengths must match")
    compensation = actuator.realize(command, sample_rate_hz, rng=rng, seed=seed)
    analog_residual = primary_arr - compensation

    if adc_saturation is not None:
        adc_saturation = float(adc_saturation)
        if not np.isfinite(adc_saturation) or adc_saturation <= 0:
            raise ValueError("adc_saturation must be positive and finite")
        digitized = np.clip(analog_residual, -adc_saturation, adc_saturation)
        clipped_fraction = float(np.mean(np.abs(analog_residual) >= adc_saturation))
    else:
        digitized = analog_residual.copy()
        clipped_fraction = 0.0

    return ActiveCancellationResult(
        command=command,
        compensation=compensation,
        analog_residual=analog_residual,
        digitized=digitized,
        adc_saturation=adc_saturation,
        clipped_fraction=clipped_fraction,
    )
