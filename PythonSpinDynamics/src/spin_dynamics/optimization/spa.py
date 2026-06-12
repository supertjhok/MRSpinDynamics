"""SPA pulse catalog and performance-metric helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/SPA_pulse_list.m
    SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/SPA_optimization_*.m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SPAPulse:
    """Fixed-amplitude SPA refocusing pulse phase program."""

    index: int
    phases: np.ndarray
    segment_fraction: float = 0.1

    @property
    def pulse_length_t180(self) -> float:
        return self.segment_fraction * self.phases.size

    @property
    def amplitudes(self) -> np.ndarray:
        return np.ones_like(self.phases, dtype=np.float64)

    @property
    def segment_lengths_t180(self) -> np.ndarray:
        return self.segment_fraction * np.ones_like(self.phases, dtype=np.float64)


@dataclass(frozen=True)
class SPAMetrics:
    """Normalized SPA/rectangular pulse performance metrics."""

    pulse_length_t180: np.ndarray
    echo_spacing_t180: np.ndarray
    snr: np.ndarray
    fom_time: np.ndarray
    fom_energy: np.ndarray
    labels: tuple[str, ...]


_SPA_PHASE_BITS = (
    (1, 1, 0, 1, 0, 1, 0, 1, 1),
    (1, 1, 0, 0, 0, 0, 0, 0, 1, 1),
    (1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1),
    (1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1),
    (1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1),
    (0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0),
    (1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1),
    (0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0),
    (0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)


def spa_pulse_list(segment_fraction: float = 0.1) -> tuple[SPAPulse, ...]:
    """Return the fixed broadband SPA refocusing pulses from Mandal et al."""

    return tuple(
        SPAPulse(
            index=idx,
            phases=np.pi * np.asarray(bits, dtype=np.float64),
            segment_fraction=float(segment_fraction),
        )
        for idx, bits in enumerate(_SPA_PHASE_BITS, start=1)
    )


def rectangular_refocusing_lengths() -> np.ndarray:
    """Return the rectangular reference pulse lengths used by MATLAB SPA scripts."""

    return np.array([0.6, 0.8, 1.0], dtype=np.float64)


def evaluate_spa_metrics(
    spa_snr: np.ndarray | list[float],
    rectangular_snr: np.ndarray | list[float],
    *,
    free_precession_t180: float = 3.0,
    segment_fraction: float = 0.1,
) -> SPAMetrics:
    """Normalize SPA and rectangular performance metrics like MATLAB.

    The MATLAB `SPA_optimization_*` scripts use the 1.0 x T180 rectangular
    pulse as the reference, include the shorter 0.6 and 0.8 rectangular pulses
    in the returned arrays, then append the fixed SPA pulse catalog.
    """

    spa = np.asarray(spa_snr, dtype=np.float64).reshape(-1)
    rect = np.asarray(rectangular_snr, dtype=np.float64).reshape(-1)
    pulses = spa_pulse_list(segment_fraction=segment_fraction)
    rect_lengths = rectangular_refocusing_lengths()

    if spa.size != len(pulses):
        raise ValueError(f"spa_snr must contain {len(pulses)} values")
    if rect.size != rect_lengths.size:
        raise ValueError(f"rectangular_snr must contain {rect_lengths.size} values")
    if not np.all(np.isfinite(spa)) or not np.all(np.isfinite(rect)):
        raise ValueError("SNR values must be finite")
    if np.any(spa <= 0) or np.any(rect <= 0):
        raise ValueError("SNR values must be positive")

    spa_lengths = np.array([pulse.pulse_length_t180 for pulse in pulses], dtype=np.float64)
    spa_echo_spacing = 2 * float(free_precession_t180) + spa_lengths
    rect_echo_spacing = 2 * float(free_precession_t180) + rect_lengths

    spa_fom_time = spa_echo_spacing / spa**2
    spa_fom_energy = spa_echo_spacing * spa_lengths / spa**2
    rect_fom_time = rect_echo_spacing / rect**2
    rect_fom_energy = rect_echo_spacing * rect_lengths / rect**2

    reference_snr = rect[-1]
    reference_fom_time = rect_fom_time[-1]
    reference_fom_energy = rect_fom_energy[-1]

    lengths = np.concatenate([rect_lengths[:-1], spa_lengths])
    echo_spacing = np.concatenate([rect_echo_spacing[:-1], spa_echo_spacing])
    snr = np.concatenate([rect[:-1], spa]) / reference_snr
    fom_time = np.concatenate([rect_fom_time[:-1], spa_fom_time]) / reference_fom_time
    fom_energy = np.concatenate([rect_fom_energy[:-1], spa_fom_energy]) / reference_fom_energy
    labels = tuple([f"rect{length:g}" for length in rect_lengths[:-1]]) + tuple(
        f"spa{pulse.index}" for pulse in pulses
    )

    return SPAMetrics(
        pulse_length_t180=lengths,
        echo_spacing_t180=echo_spacing,
        snr=snr,
        fom_time=fom_time,
        fom_energy=fom_energy,
        labels=labels,
    )
