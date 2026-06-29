"""ENDOR (electron-nuclear double resonance) spectra for an S=1/2, I=1/2 pair.

ENDOR detects the nuclear transition frequencies of nuclei coupled to an electron
spin by sweeping a radiofrequency field and reading the change it induces in the
electron signal. The line positions are the same nuclear frequencies that govern
ESEEM,

    nu_alpha = sqrt((omega_I + A/2)^2 + (B/2)^2),
    nu_beta  = sqrt((omega_I - A/2)^2 + (B/2)^2),

so ENDOR resolves them directly as peaks rather than as a modulation pattern.

The two common pulsed variants differ in their response function:

* **Davies ENDOR** gives both lines with comparable intensity and no blind
  spots, but discriminates against small hyperfine couplings.
* **Mims ENDOR** is sensitive to small couplings but has ``tau``-dependent blind
  spots: the response of a line at frequency ``nu`` carries the factor
  ``sin^2(pi nu tau)``, which vanishes whenever ``nu tau`` is an integer.

This module returns the line positions and the analytic Davies and Mims ENDOR
spectra (peaks broadened to a chosen linewidth), built on the same
:class:`~spin_dynamics.esr.eseem.HyperfineCoupling` model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.esr.eseem import HyperfineCoupling, nuclear_frequencies


@dataclass(frozen=True)
class EndorSpectrum:
    """An ENDOR spectrum sampled on a radiofrequency axis."""

    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    line_positions_hz: tuple[float, float]
    line_intensities: tuple[float, float]
    method: str


def endor_frequencies(coupling: HyperfineCoupling) -> tuple[float, float]:
    """Return the ENDOR line positions ``(nu_alpha, nu_beta)`` in Hz."""

    return nuclear_frequencies(coupling)


def mims_blind_spot_factor(frequency_hz: float, tau_seconds: float) -> float:
    """Return the Mims ENDOR response factor ``sin^2(pi nu tau)`` for one line.

    The factor is zero at the blind spots ``nu tau = 0, 1, 2, ...`` and one at
    ``nu tau = 1/2, 3/2, ...``.
    """

    return float(np.sin(np.pi * float(frequency_hz) * float(tau_seconds)) ** 2)


def _lineshape(axis: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((axis - center) / width) ** 2)


def _frequency_axis(frequencies_hz) -> np.ndarray:
    axis = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    if axis.size == 0:
        raise ValueError("frequencies_hz must not be empty")
    if not np.all(np.isfinite(axis)):
        raise ValueError("frequencies_hz must be finite")
    return axis


def davies_endor_spectrum(
    frequencies_hz,
    coupling: HyperfineCoupling,
    *,
    linewidth_hz: float = 1.0e5,
) -> EndorSpectrum:
    """Return a Davies ENDOR spectrum (both lines, no blind spots)."""

    axis = _frequency_axis(frequencies_hz)
    width = float(linewidth_hz)
    if width <= 0 or not np.isfinite(width):
        raise ValueError("linewidth_hz must be positive and finite")
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    intensities = (1.0, 1.0)
    spectrum = intensities[0] * _lineshape(axis, nu_alpha, width)
    spectrum = spectrum + intensities[1] * _lineshape(axis, nu_beta, width)
    return EndorSpectrum(
        frequencies_hz=axis,
        spectrum=spectrum,
        line_positions_hz=(nu_alpha, nu_beta),
        line_intensities=intensities,
        method="davies",
    )


def mims_endor_spectrum(
    frequencies_hz,
    coupling: HyperfineCoupling,
    *,
    tau_seconds: float,
    linewidth_hz: float = 1.0e5,
) -> EndorSpectrum:
    """Return a Mims ENDOR spectrum with ``tau``-dependent blind-spot weighting."""

    axis = _frequency_axis(frequencies_hz)
    width = float(linewidth_hz)
    if width <= 0 or not np.isfinite(width):
        raise ValueError("linewidth_hz must be positive and finite")
    tau = float(tau_seconds)
    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau_seconds must be positive and finite")
    nu_alpha, nu_beta = nuclear_frequencies(coupling)
    intensities = (
        mims_blind_spot_factor(nu_alpha, tau),
        mims_blind_spot_factor(nu_beta, tau),
    )
    spectrum = intensities[0] * _lineshape(axis, nu_alpha, width)
    spectrum = spectrum + intensities[1] * _lineshape(axis, nu_beta, width)
    return EndorSpectrum(
        frequencies_hz=axis,
        spectrum=spectrum,
        line_positions_hz=(nu_alpha, nu_beta),
        line_intensities=intensities,
        method="mims",
    )
