"""Run ab initio quadrupolar parameters through the spin-dynamics simulator.

The core scientific check here is *self-consistency between two independent
implementations* of the zero-field quadrupole Hamiltonian:

- ``quadrupolar_dft.nqr_frequencies_hz`` builds the Hamiltonian from ``C_Q``
  and ``eta`` directly and returns its eigenvalue differences;
- ``spin_dynamics.nqr.diagonalize_site`` builds it from a ``QuadrupolarSite``
  (parameterized by ``nu_Q``) and returns labelled transitions with dipole
  strengths.

If the :mod:`mr_integration.conversions` mapping is correct, the two line lists
must agree.  :func:`predicted_lines` returns both and their discrepancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadrupolar_dft import nqr_frequencies_hz
from spin_dynamics.nqr import diagonalize_site

from .conversions import nu_q_from_cq_hz, quadrupolar_site_from_cq

#: Lines closer than this are treated as one (collapses Kramers-degenerate
#: half-integer-spin transitions, which the simulator lists individually but the
#: DFT module reports as a single unique frequency).
DEFAULT_DEDUP_TOL_HZ = 1.0


def _unique_within(values: np.ndarray, tol_hz: float) -> np.ndarray:
    """Collapse values within ``tol_hz`` of an earlier one (ascending out)."""

    ordered = np.sort(np.asarray(values, dtype=float))
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > tol_hz:
            unique.append(float(value))
    return np.asarray(unique, dtype=float)


@dataclass(frozen=True)
class PredictedLines:
    """Zero-field NQR lines predicted from one ``(C_Q, eta, spin)`` triple."""

    cq_hz: float
    eta: float
    spin: float
    nu_q_hz: float
    #: Lines from the simulator's diagonalized site (Hz, ascending).
    simulator_hz: np.ndarray
    #: Lines from the DFT module's direct Hamiltonian (Hz, ascending).
    dft_hz: np.ndarray

    @property
    def max_abs_discrepancy_hz(self) -> float:
        """Largest absolute line difference between the two implementations."""

        if self.simulator_hz.size != self.dft_hz.size:
            return float("inf")
        if self.simulator_hz.size == 0:
            return 0.0
        return float(np.max(np.abs(self.simulator_hz - self.dft_hz)))

    def self_consistent(self, *, atol_hz: float = 1.0) -> bool:
        """True when both implementations agree within ``atol_hz``."""

        return (
            self.simulator_hz.size == self.dft_hz.size
            and self.max_abs_discrepancy_hz <= atol_hz
        )


def predicted_lines(
    *,
    cq_hz: float,
    eta: float,
    spin: float,
    isotope: str = "14N",
    dedup_tol_hz: float = DEFAULT_DEDUP_TOL_HZ,
) -> PredictedLines:
    """Predict zero-field NQR lines two independent ways and compare them.

    Both line lists are collapsed to unique frequencies within ``dedup_tol_hz``
    so degenerate transitions (e.g. Kramers doublets for half-integer spin,
    which the simulator enumerates individually) compare like-for-like against
    the DFT module's unique-frequency output.
    """

    site = quadrupolar_site_from_cq(
        cq_hz=cq_hz, eta=eta, spin=spin, isotope=isotope
    )
    eigensystem = diagonalize_site(site)
    simulator = _unique_within(
        np.asarray([t.frequency_hz for t in eigensystem.transitions], dtype=float),
        dedup_tol_hz,
    )
    dft = _unique_within(
        np.asarray(nqr_frequencies_hz(spin=spin, cq_hz=cq_hz, eta=eta)),
        dedup_tol_hz,
    )
    return PredictedLines(
        cq_hz=float(cq_hz),
        eta=float(eta),
        spin=float(spin),
        nu_q_hz=nu_q_from_cq_hz(cq_hz, spin),
        simulator_hz=simulator,
        dft_hz=dft,
    )


def match_lines(
    predicted_hz: np.ndarray,
    measured_hz: np.ndarray,
) -> list[tuple[float, float, float]]:
    """Greedily pair each measured line with its nearest predicted line.

    Returns ``(measured, predicted, signed_difference)`` triples in ascending
    measured order.  Unequal counts are allowed: every measured line is paired
    with its closest predicted line (predictions may be reused or unused).
    """

    predicted = np.sort(np.asarray(predicted_hz, dtype=float))
    measured = np.sort(np.asarray(measured_hz, dtype=float))
    pairs: list[tuple[float, float, float]] = []
    for value in measured:
        if predicted.size == 0:
            pairs.append((float(value), float("nan"), float("nan")))
            continue
        nearest = predicted[int(np.argmin(np.abs(predicted - value)))]
        pairs.append((float(value), float(nearest), float(value - nearest)))
    return pairs
