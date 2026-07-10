"""Duty-cycled heat sources for the thermal solvers.

Converts the package's electromagnetic quantities into average thermal powers:

- coil Joule heating from RF/gradient drive currents and the coil resistance
  (:mod:`spin_dynamics.fields.coil_properties` /
  :mod:`spin_dynamics.fields.coil_peec` supply ``R``, including ``R(T)``);
- sample SAR from the first-order eddy solver
  (:mod:`spin_dynamics.fields.quasistatic`), either as a volume-integrated
  power or as a per-cell density map for the conduction solvers;
- pulse-sequence duty cycles from the ``pp`` parameter sets so peak RF powers
  become average thermal loads.

Sinusoidal-drive convention: ``peak`` currents are amplitudes, so the power
during a pulse is ``I_peak^2 R / 2``; ``rms`` values use ``I_rms^2 R``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ConstantSource",
    "DutyCycledSource",
    "coil_joule_power",
    "transmit_coil_current",
    "duty_cycle_from_pulse_params",
    "average_coil_power",
    "gradient_waveform_power",
    "sar_source_from_eddy",
    "sar_power_from_loading",
]


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _check_positive(**values: float) -> None:
    for label, value in values.items():
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")


def _check_non_negative(**values: float) -> None:
    for label, value in values.items():
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and non-negative")


@dataclass(frozen=True)
class ConstantSource:
    """A steady heat source of ``power`` watts attached to a named node."""

    name: str
    power: float

    def __post_init__(self) -> None:
        _check_non_negative(power=self.power)

    @property
    def average_power(self) -> float:
        return self.power


@dataclass(frozen=True)
class DutyCycledSource:
    """A pulsed source: ``peak_power`` watts active for ``duty_cycle`` of the time."""

    name: str
    peak_power: float
    duty_cycle: float

    def __post_init__(self) -> None:
        _check_non_negative(peak_power=self.peak_power)
        if not (0.0 <= self.duty_cycle <= 1.0):
            raise ValueError("duty_cycle must be in [0, 1]")

    @property
    def average_power(self) -> float:
        return self.peak_power * self.duty_cycle


def coil_joule_power(current: float, resistance: float, *, rms: bool = False) -> float:
    """Instantaneous coil dissipation ``I^2 R / 2`` (amplitude) or ``I_rms^2 R``.

    ``current`` is the sinusoidal drive amplitude unless ``rms=True``.
    """

    _check_non_negative(current=current)
    _check_positive(resistance=resistance)
    scale = 1.0 if rms else 0.5
    return scale * current**2 * resistance


def transmit_coil_current(b1_tesla: float, b1_per_current: float) -> float:
    """Peak coil current (A) to produce a rotating-frame ``B1`` amplitude.

    ``b1_per_current`` is the coil field per unit current ``B1_hat`` (T/A) at
    the sample (linear coil: the *rotating* component is half the linear
    amplitude, so a rotating-frame ``B1`` needs a linear field ``2*B1``).
    """

    _check_positive(b1_tesla=b1_tesla, b1_per_current=b1_per_current)
    return 2.0 * b1_tesla / b1_per_current


def duty_cycle_from_pulse_params(pp: Mapping[str, Any] | Any) -> float:
    """RF duty cycle of a refocusing-train cycle from a ``pp`` parameter set.

    Uses the ``tref`` (segment durations) and ``aref`` (segment amplitudes)
    arrays shared by the tuned/untuned/matched pulse parameter sets: the duty
    cycle is the fraction of the repeating cycle with nonzero RF amplitude,
    ``sum(tref[aref != 0]) / sum(tref)``. For a standard CPMG cycle
    ``[preDelay, T_180, postDelay]`` with ``aref = [0, 1, 0]`` this is
    ``T_180 / (preDelay + T_180 + postDelay)``.
    """

    tref = np.asarray(_field(pp, "tref"), dtype=np.float64).reshape(-1)
    aref = np.asarray(_field(pp, "aref"), dtype=np.float64).reshape(-1)
    if tref.size != aref.size:
        raise ValueError("tref and aref must have the same length")
    if tref.size == 0 or np.any(tref < 0) or not np.all(np.isfinite(tref)):
        raise ValueError("tref must be non-empty, finite, and non-negative")
    total = float(np.sum(tref))
    if total <= 0:
        raise ValueError("tref must sum to a positive cycle time")
    active = float(np.sum(tref[aref != 0.0]))
    return active / total


def average_coil_power(
    current: float,
    resistance: float,
    duty_cycle: float,
    *,
    rms: bool = False,
    name: str = "coil",
) -> DutyCycledSource:
    """Duty-cycled coil Joule source: ``I^2 R / 2`` active for ``duty_cycle``."""

    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("duty_cycle must be in [0, 1]")
    peak = coil_joule_power(current, resistance, rms=rms)
    return DutyCycledSource(name=name, peak_power=peak, duty_cycle=duty_cycle)


def gradient_waveform_power(
    current_waveform: np.ndarray,
    times: np.ndarray,
    resistance: float,
    *,
    repetition_time: float | None = None,
    name: str = "gradient",
) -> ConstantSource:
    """Average ``i(t)^2 R`` of a gradient waveform (audio-frequency: no 1/2).

    ``current_waveform`` (A) is sampled at ``times`` (s). The mean square is
    trapezoid-integrated over the waveform and, if ``repetition_time`` is
    given (>= waveform duration), diluted by ``duration / repetition_time``.
    """

    i = np.asarray(current_waveform, dtype=np.float64).reshape(-1)
    t = np.asarray(times, dtype=np.float64).reshape(-1)
    if i.size != t.size or i.size < 2:
        raise ValueError("current_waveform and times must have equal length >= 2")
    if not np.all(np.diff(t) > 0):
        raise ValueError("times must be strictly increasing")
    _check_positive(resistance=resistance)
    duration = float(t[-1] - t[0])
    integral = float(np.trapezoid(i**2, t) if hasattr(np, "trapezoid") else np.trapz(i**2, t))
    mean_square = integral / duration
    if repetition_time is not None:
        if repetition_time < duration or not np.isfinite(repetition_time):
            raise ValueError("repetition_time must be finite and >= waveform duration")
        mean_square *= duration / repetition_time
    return ConstantSource(name=name, power=mean_square * resistance)


def sar_source_from_eddy(
    eddy_result: Any,
    *,
    duty_cycle: float = 1.0,
    conductivity: float | np.ndarray | None = None,
    name: str = "sample SAR",
) -> tuple[DutyCycledSource, np.ndarray]:
    """Sample SAR source and volumetric density map from an ``EddyResult``.

    ``eddy_result`` is :class:`spin_dynamics.fields.quasistatic.EddyResult`
    computed at the *peak* drive (``dI/dt`` amplitude ``omega * I_peak`` for a
    sinusoidal current). Returns the duty-cycled total power source and the
    per-cell time-averaged deposition density ``q_v = (1/2) sigma |E|^2 *
    duty`` in W/m^3 (for the conduction solvers). ``conductivity`` defaults to
    ``|J|^2 / (sigma^2 |E|^2)``-free reconstruction via ``J . E``:
    ``q_v = (1/2) Re(J) . Re(E)``-equivalent ``(1/2) J . E`` for the real
    first-order fields, so it can be omitted whenever ``eddy_result`` carries
    both ``current_density`` and ``e_field``.
    """

    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("duty_cycle must be in [0, 1]")
    e = np.asarray(eddy_result.e_field, dtype=np.float64)
    if conductivity is None:
        j = np.asarray(eddy_result.current_density, dtype=np.float64)
        density = 0.5 * np.sum(j * e, axis=-1)
    else:
        sigma = np.asarray(conductivity, dtype=np.float64)
        density = 0.5 * sigma * np.sum(e * e, axis=-1)
    if np.any(density < -1e-30):
        raise ValueError("negative deposition density; check the eddy inputs")
    density = np.clip(density, 0.0, None) * duty_cycle
    source = DutyCycledSource(
        name=name, peak_power=float(eddy_result.power), duty_cycle=duty_cycle
    )
    return source, density


def sar_power_from_loading(
    current: float,
    reflected_resistance: float,
    *,
    duty_cycle: float = 1.0,
    name: str = "sample SAR",
) -> DutyCycledSource:
    """Sample deposition from the circuit side: ``P = I_peak^2 R_reflected / 2``.

    ``reflected_resistance`` is the sample-loading resistance referred to the
    coil (:func:`spin_dynamics.fields.quasistatic.reflected_resistance` or a
    measured ``Q`` drop). Must equal the volume-integrated eddy power for a
    consistent model -- that identity is a Phase 0 validation test.
    """

    _check_non_negative(current=current, reflected_resistance=reflected_resistance)
    if not (0.0 <= duty_cycle <= 1.0):
        raise ValueError("duty_cycle must be in [0, 1]")
    return DutyCycledSource(
        name=name,
        peak_power=0.5 * current**2 * reflected_resistance,
        duty_cycle=duty_cycle,
    )
