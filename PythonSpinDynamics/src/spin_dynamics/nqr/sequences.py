"""Pulse-sequence dataclasses for pulsed NQR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nqr.pulses import SelectivePulse


@dataclass(frozen=True)
class SLSESequence:
    """Spin-lock spin-echo detection sequence."""

    detection: SelectivePulse
    echo_spacing_seconds: float
    num_echoes: int

    def __post_init__(self) -> None:
        echo_spacing_seconds = float(self.echo_spacing_seconds)
        num_echoes = int(self.num_echoes)
        if not np.isfinite(echo_spacing_seconds) or echo_spacing_seconds < 0:
            raise ValueError("echo_spacing_seconds must be non-negative and finite")
        if num_echoes <= 0:
            raise ValueError("num_echoes must be positive")
        object.__setattr__(self, "echo_spacing_seconds", echo_spacing_seconds)
        object.__setattr__(self, "num_echoes", num_echoes)


def slse_sequence(
    transition_label: str,
    *,
    pulse_duration_seconds: float,
    nutation_hz: float,
    echo_spacing_seconds: float,
    num_echoes: int,
    phase: float = 0.0,
    rf_frequency_hz: float | None = None,
) -> SLSESequence:
    """Build a rectangular-pulse SLSE sequence."""

    return SLSESequence(
        detection=SelectivePulse(
            transition_label=transition_label,
            duration_seconds=pulse_duration_seconds,
            nutation_hz=nutation_hz,
            phase=phase,
            rf_frequency_hz=rf_frequency_hz,
        ),
        echo_spacing_seconds=echo_spacing_seconds,
        num_echoes=num_echoes,
    )
