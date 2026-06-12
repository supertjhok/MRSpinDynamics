"""SPA pulse catalog and performance-metric helpers.

MATLAB references:
    SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/SPA_pulse_list.m
    SpinDynamicsUpdated/Version_2/code/OCT_Pulse_Examples/SPA_optimization_*.m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.parameters import (
    set_params_matched_spa,
    set_params_tuned_spa,
    set_params_untuned_spa,
)
from spin_dynamics.probes.matched import calc_masy_matched_probe_orig
from spin_dynamics.probes.tuned import calc_masy_tuned_probe_lp_orig
from spin_dynamics.probes.untuned import calc_masy_untuned_probe_lp


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


@dataclass(frozen=True)
class TunedRefocusingEvaluation:
    """Non-plotting tuned-probe arbitrary-refocusing-pulse evaluation."""

    del_w: np.ndarray
    mrx: np.ndarray
    masy: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float
    pulse_length_t180: float
    phases: np.ndarray


@dataclass(frozen=True)
class UntunedRefocusingEvaluation:
    """Non-plotting untuned-probe arbitrary-refocusing-pulse evaluation."""

    del_w: np.ndarray
    mrx: np.ndarray
    masy: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float
    pulse_length_t180: float
    phases: np.ndarray


@dataclass(frozen=True)
class MatchedRefocusingEvaluation:
    """Non-plotting matched-probe arbitrary-refocusing-pulse evaluation."""

    del_w: np.ndarray
    mrx: np.ndarray
    masy: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float
    pulse_length_t180: float
    phases: np.ndarray


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


def evaluate_tuned_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    excitation_amplitude: float = 6.0,
) -> TunedRefocusingEvaluation:
    """Evaluate a fixed-amplitude tuned-probe refocusing phase program.

    This ports the non-plotting core of MATLAB
    `opt_pulse/plot_masy_arbref_tuned.m` using the SPA parameter defaults.
    The excitation pulse is shortened by `excitation_amplitude`, matching the
    broadband-excitation setup in `SPA_optimization_tuned.m`.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")
    if excitation_amplitude <= 0:
        raise ValueError("excitation_amplitude must be positive")

    params, sp, pp = set_params_tuned_spa(numpts=numpts)
    texc = pp.T_90 / float(excitation_amplitude)
    params = params.__class__(
        **{
            **params.__dict__,
            "aexc": np.array([float(excitation_amplitude)], dtype=np.float64),
            "texc": np.array([texc], dtype=np.float64),
            "pref": phase_arr,
            "aref": np.ones(phase_arr.size, dtype=np.float64),
            "tref": pp.T_180 * float(segment_fraction) * np.ones(
                phase_arr.size,
                dtype=np.float64,
            ),
        }
    )
    pp = pp.__class__(**{**pp.__dict__, "tcorr": -(2 / np.pi) * texc})
    sp = sp.__class__(**{**sp.__dict__, "plt_axis": 0, "plt_tx": 0, "plt_rx": 0})

    mrx, masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, sp.del_w)
    return TunedRefocusingEvaluation(
        del_w=sp.del_w,
        mrx=mrx,
        masy=masy,
        echo=echo,
        tvect=tvect,
        snr=snr,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
    )


def evaluate_untuned_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    excitation_amplitude: float = 6.0,
) -> UntunedRefocusingEvaluation:
    """Evaluate a fixed-amplitude untuned-probe refocusing phase program.

    This ports the non-plotting core of MATLAB
    `opt_pulse/plot_masy_arbref_untuned.m` using the SPA parameter defaults.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")
    if excitation_amplitude <= 0:
        raise ValueError("excitation_amplitude must be positive")

    params, sp, pp = set_params_untuned_spa(numpts=numpts)
    texc = pp.T_90 / float(excitation_amplitude)
    params = params.__class__(
        **{
            **params.__dict__,
            "aexc": np.array([float(excitation_amplitude)], dtype=np.float64),
            "texc": np.array([texc], dtype=np.float64),
            "pref": phase_arr,
            "aref": np.ones(phase_arr.size, dtype=np.float64),
            "tref": pp.T_180 * float(segment_fraction) * np.ones(
                phase_arr.size,
                dtype=np.float64,
            ),
        }
    )
    pp = pp.__class__(**{**pp.__dict__, "tcorr": -(2 / np.pi) * texc})
    sp = sp.__class__(**{**sp.__dict__, "plt_axis": 0, "plt_tx": 0, "plt_rx": 0})

    mrx, masy, snr = calc_masy_untuned_probe_lp(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, sp.del_w)
    return UntunedRefocusingEvaluation(
        del_w=sp.del_w,
        mrx=mrx,
        masy=masy,
        echo=echo,
        tvect=tvect,
        snr=snr,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
    )


def evaluate_matched_refocusing_pulse(
    phases: np.ndarray | list[float],
    *,
    segment_fraction: float = 0.1,
    numpts: int = 101,
    excitation_amplitude: float = 6.0,
) -> MatchedRefocusingEvaluation:
    """Evaluate a fixed-amplitude matched-probe refocusing phase program.

    This ports the non-plotting core of MATLAB
    `opt_pulse/plot_masy_arbref_matched.m` using the SPA parameter defaults.
    """

    phase_arr = np.asarray(phases, dtype=np.float64).reshape(-1)
    if phase_arr.size == 0:
        raise ValueError("phases must not be empty")
    if segment_fraction <= 0:
        raise ValueError("segment_fraction must be positive")
    if excitation_amplitude <= 0:
        raise ValueError("excitation_amplitude must be positive")

    sp, pp = set_params_matched_spa(numpts=numpts)
    texc = pp.T_90 / float(excitation_amplitude)
    pp = pp.__class__(
        **{
            **pp.__dict__,
            "aexc": np.array([float(excitation_amplitude)], dtype=np.float64),
            "texc": np.array([texc], dtype=np.float64),
            "tcorr": -(2 / np.pi) * texc,
            "pref": np.concatenate([[0.0], phase_arr, [0.0]]),
            "aref": np.concatenate(
                [[0.0], np.ones(phase_arr.size, dtype=np.float64), [0.0]]
            ),
            "tref": np.concatenate(
                [
                    [pp.preDelay],
                    pp.T_180 * float(segment_fraction) * np.ones(
                        phase_arr.size,
                        dtype=np.float64,
                    ),
                    [pp.postDelay],
                ]
            ),
        }
    )
    sp = sp.__class__(**{**sp.__dict__, "plt_axis": 0, "plt_tx": 0, "plt_rx": 0})

    mrx, masy, snr = calc_masy_matched_probe_orig(sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, sp.del_w)
    return MatchedRefocusingEvaluation(
        del_w=sp.del_w,
        mrx=mrx,
        masy=masy,
        echo=echo,
        tvect=tvect,
        snr=snr,
        pulse_length_t180=float(segment_fraction) * phase_arr.size,
        phases=phase_arr,
    )


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
