"""Facade adapters for ESEEM, HYSCORE, and ENDOR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.esr import (
    EndorSpectrum,
    HyperfineCoupling,
    cross_peak_positions,
    davies_endor_spectrum,
    endor_frequencies,
    eseem_spectrum,
    hyscore_signal,
    hyscore_spectrum,
    mims_endor_spectrum,
    three_pulse_eseem,
    three_pulse_eseem_quantum,
    two_pulse_eseem,
    two_pulse_eseem_quantum,
)


@dataclass(frozen=True)
class ESRESEEMResult:
    """Time-domain ESEEM trace and its one-sided magnitude spectrum."""

    times_seconds: np.ndarray
    signal: np.ndarray
    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    sequence: str
    model: str


@dataclass(frozen=True)
class ESRHYSCOREResult:
    """HYSCORE time plane, 2-D spectrum, and predicted cross-peaks."""

    t1_seconds: np.ndarray
    t2_seconds: np.ndarray
    signal: np.ndarray
    frequencies1_hz: np.ndarray
    frequencies2_hz: np.ndarray
    spectrum: np.ndarray
    cross_peaks_hz: np.ndarray


def require_coupling(experiment: Any) -> HyperfineCoupling:
    coupling = experiment.sample.hyperfine_coupling
    if coupling is None:
        raise ValueError(
            "ESEEM, HYSCORE, and ENDOR require sample.hyperfine_coupling"
        )
    if not isinstance(coupling, HyperfineCoupling):
        raise ValueError(
            "sample.hyperfine_coupling must be a spin_dynamics.esr.HyperfineCoupling"
        )
    return coupling


def resolved_eseem_model(
    requested: str,
    coupling: HyperfineCoupling,
    *,
    electron_offset_hz: float = 0.0,
) -> str:
    if requested == "auto":
        return (
            "analytic"
            if coupling.is_spin_half and electron_offset_hz == 0.0
            else "quantum"
        )
    if requested == "analytic" and not coupling.is_spin_half:
        raise ValueError("analytic ESEEM supports spin-1/2 nuclei only")
    if requested == "analytic" and electron_offset_hz != 0.0:
        raise ValueError("analytic ESEEM does not model electron_offset_hz")
    return requested


def eseem_kwargs(experiment: Any) -> dict[str, Any]:
    sequence = experiment.sequence
    return {
        "times_seconds": np.linspace(
            0.0, sequence.acquisition_seconds, sequence.num_points
        ),
        "coupling": require_coupling(experiment),
        "model": sequence.model,
        "zero_fill": sequence.zero_fill,
    }


def two_pulse_kwargs(experiment: Any) -> dict[str, Any]:
    kwargs = eseem_kwargs(experiment)
    kwargs["electron_offset_hz"] = experiment.sequence.electron_offset_hz
    return kwargs


def three_pulse_kwargs(experiment: Any) -> dict[str, Any]:
    kwargs = eseem_kwargs(experiment)
    kwargs["tau_seconds"] = experiment.sequence.tau_seconds
    return kwargs


def run_two_pulse_eseem(
    times_seconds: np.ndarray,
    coupling: HyperfineCoupling,
    *,
    model: str,
    electron_offset_hz: float,
    zero_fill: int,
) -> ESRESEEMResult:
    resolved = resolved_eseem_model(
        model, coupling, electron_offset_hz=electron_offset_hz
    )
    if resolved == "analytic":
        signal = two_pulse_eseem(times_seconds, coupling)
    else:
        signal = two_pulse_eseem_quantum(
            times_seconds, coupling, electron_offset_hz=electron_offset_hz
        )
    frequencies, spectrum = eseem_spectrum(
        times_seconds, signal, zero_fill=zero_fill
    )
    return ESRESEEMResult(
        times_seconds=times_seconds,
        signal=signal,
        frequencies_hz=frequencies,
        spectrum=spectrum,
        sequence="two_pulse",
        model=resolved,
    )


def run_three_pulse_eseem(
    times_seconds: np.ndarray,
    coupling: HyperfineCoupling,
    *,
    model: str,
    tau_seconds: float,
    zero_fill: int,
) -> ESRESEEMResult:
    resolved = resolved_eseem_model(model, coupling)
    func = three_pulse_eseem if resolved == "analytic" else three_pulse_eseem_quantum
    signal = func(times_seconds, coupling, tau_seconds=tau_seconds)
    frequencies, spectrum = eseem_spectrum(
        times_seconds, signal, zero_fill=zero_fill
    )
    return ESRESEEMResult(
        times_seconds=times_seconds,
        signal=signal,
        frequencies_hz=frequencies,
        spectrum=spectrum,
        sequence="three_pulse",
        model=resolved,
    )


def hyscore_kwargs(experiment: Any) -> dict[str, Any]:
    sequence = experiment.sequence
    return {
        "t1_seconds": np.linspace(
            0.0, sequence.evolution1_seconds, sequence.num_points1
        ),
        "t2_seconds": np.linspace(
            0.0, sequence.evolution2_seconds, sequence.num_points2
        ),
        "coupling": require_coupling(experiment),
        "tau_seconds": sequence.tau_seconds,
        "zero_fill": sequence.zero_fill,
    }


def run_hyscore(
    t1_seconds: np.ndarray,
    t2_seconds: np.ndarray,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
    zero_fill: int,
) -> ESRHYSCOREResult:
    signal = hyscore_signal(
        t1_seconds, t2_seconds, coupling, tau_seconds=tau_seconds
    )
    transformed = hyscore_spectrum(
        t1_seconds, t2_seconds, signal, zero_fill=zero_fill
    )
    return ESRHYSCOREResult(
        t1_seconds=t1_seconds,
        t2_seconds=t2_seconds,
        signal=signal,
        frequencies1_hz=transformed.frequencies1_hz,
        frequencies2_hz=transformed.frequencies2_hz,
        spectrum=transformed.spectrum,
        cross_peaks_hz=np.asarray(cross_peak_positions(coupling)),
    )


def endor_kwargs(experiment: Any) -> dict[str, Any]:
    sequence = experiment.sequence
    coupling = require_coupling(experiment)
    lines = endor_frequencies(coupling)
    lower = sequence.frequency_min_hz
    upper = sequence.frequency_max_hz
    margin = 5.0 * sequence.linewidth_hz
    if lower is None:
        lower = max(0.0, float(np.min(lines)) - margin)
    if upper is None:
        upper = float(np.max(lines)) + margin
    if upper <= lower:
        raise ValueError("resolved ENDOR frequency axis is empty")
    return {
        "frequencies_hz": np.linspace(lower, upper, sequence.num_points),
        "coupling": coupling,
        "linewidth_hz": sequence.linewidth_hz,
    }


def mims_endor_kwargs(experiment: Any) -> dict[str, Any]:
    kwargs = endor_kwargs(experiment)
    kwargs["tau_seconds"] = experiment.sequence.tau_seconds
    return kwargs


__all__ = [
    "EndorSpectrum",
    "ESRESEEMResult",
    "ESRHYSCOREResult",
    "davies_endor_spectrum",
    "endor_kwargs",
    "eseem_kwargs",
    "hyscore_kwargs",
    "mims_endor_kwargs",
    "mims_endor_spectrum",
    "require_coupling",
    "resolved_eseem_model",
    "run_hyscore",
    "run_three_pulse_eseem",
    "run_two_pulse_eseem",
    "three_pulse_kwargs",
    "two_pulse_kwargs",
]
