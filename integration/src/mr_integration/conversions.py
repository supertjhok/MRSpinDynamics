"""Convert ab initio quadrupolar parameters into simulator site objects.

`QuadrupolarDFT` reports the quadrupolar coupling constant

    C_Q = e Q V_zz / h        (Hz)

while `PythonSpinDynamics` parameterizes a :class:`QuadrupolarSite` by its
``quadrupole_frequency_hz`` (``nu_Q``), defined as the ``eta = 0`` transition
frequency.  The two are linked by the prefactor of the quadrupole Hamiltonian.

The standard zero-field quadrupole Hamiltonian, in frequency units, is

    H/h = C_Q / (4 I (2I-1)) * [3 Iz^2 - I(I+1) + eta (Ix^2 - Iy^2)]

The simulator writes the same operator with prefactor ``nu_Q / d`` where
``d = 3`` for spin-1 and ``d = 6`` for spin-3/2 (see
``spin_dynamics.nqr.hamiltonians.quadrupole_frequency_scale_hz``).  Matching the
two prefactors gives

    nu_Q = C_Q * d / (4 I (2I-1))
         = (3/4) C_Q   for spin-1
         = (1/2) C_Q   for spin-3/2

This module owns that mapping in one place and builds the simulator site.
"""

from __future__ import annotations

import numpy as np

from spin_dynamics.nqr import QuadrupolarSite

# Hamiltonian-scale denominators used by the simulator for each supported spin.
# Mirrors spin_dynamics.nqr.hamiltonians.quadrupole_frequency_scale_hz so the
# two codebases stay in sync; tests assert the resulting lines agree.
_SCALE_DENOMINATORS = {1.0: 3.0, 1.5: 6.0}


def _conversion_factor(spin: float) -> float:
    """Return ``nu_Q / C_Q`` for a supported spin."""

    spin = float(spin)
    for supported, denominator in _SCALE_DENOMINATORS.items():
        if np.isclose(spin, supported):
            return denominator / (4.0 * supported * (2.0 * supported - 1.0))
    raise ValueError(
        "C_Q <-> nu_Q conversion is calibrated for spin=1 and spin=3/2 only; "
        f"got spin={spin!r}"
    )


def nu_q_from_cq_hz(cq_hz: float, spin: float) -> float:
    """Return the simulator ``quadrupole_frequency_hz`` (nu_Q) for a C_Q in Hz.

    ``nu_Q`` is taken as a positive frequency; the sign of ``C_Q`` (an ABINIT
    convention that depends on the supplied quadrupole moment) does not change
    the zero-field transition frequencies.
    """

    cq_hz = float(cq_hz)
    if not np.isfinite(cq_hz):
        raise ValueError("cq_hz must be finite")
    return abs(cq_hz) * _conversion_factor(spin)


def cq_hz_from_nu_q(nu_q_hz: float, spin: float) -> float:
    """Inverse of :func:`nu_q_from_cq_hz` (returns a non-negative C_Q)."""

    nu_q_hz = float(nu_q_hz)
    if not np.isfinite(nu_q_hz):
        raise ValueError("nu_q_hz must be finite")
    return nu_q_hz / _conversion_factor(spin)


def quadrupolar_site_from_cq(
    *,
    cq_hz: float,
    eta: float,
    spin: float,
    isotope: str = "14N",
    gamma_hz_per_t: float = 0.0,
    label: str = "site",
) -> QuadrupolarSite:
    """Build a :class:`QuadrupolarSite` from an ab initio ``(C_Q, eta)`` pair."""

    return QuadrupolarSite(
        spin=float(spin),
        quadrupole_frequency_hz=nu_q_from_cq_hz(cq_hz, spin),
        eta=float(eta),
        gamma_hz_per_t=float(gamma_hz_per_t),
        isotope=str(isotope),
        label=str(label),
    )


# Spin quantum number for the isotopes the simulator currently supports.  Used
# to translate a record (which knows C_Q but not the nuclear spin) into a
# simulator site.  Limited to spin-1 and spin-3/2, the regimes for which the
# simulator's quadrupole-frequency scale is calibrated.
ISOTOPE_SPINS = {
    "14N": 1.0,
    "11B": 1.5,
    "23Na": 1.5,
    "35Cl": 1.5,
    "37Cl": 1.5,
    "39K": 1.5,
    "63Cu": 1.5,
    "65Cu": 1.5,
    "75As": 1.5,
    "79Br": 1.5,
    "81Br": 1.5,
}


def spin1_parameters_from_lines(
    lines_hz,
) -> tuple[float, float]:
    """Back out ``(C_Q, eta)`` from three spin-1 zero-field NQR lines.

    The spin-1 lines obey ``nu_plus = (3/4) C_Q (1 + eta/3)``,
    ``nu_minus = (3/4) C_Q (1 - eta/3)`` and ``nu_0 = nu_plus - nu_minus``.
    Inverting the two larger lines gives

        C_Q = (2/3) (nu_plus + nu_minus)
        eta = 3 (nu_plus - nu_minus) / (nu_plus + nu_minus)

    This is the diagnostic inverse used to localize database inconsistencies:
    compare the implied parameters against the stored ``(qcc, eta)``.
    """

    values = np.sort(np.asarray(list(lines_hz), dtype=float))
    if values.size != 3:
        raise ValueError("spin-1 parameter inversion needs exactly three lines")
    nu_minus, nu_plus = float(values[1]), float(values[2])
    total = nu_plus + nu_minus
    if total <= 0:
        raise ValueError("spin-1 lines must be positive")
    cq_hz = (2.0 / 3.0) * total
    eta = 3.0 * (nu_plus - nu_minus) / total
    return cq_hz, eta


def quadrupolar_site_from_efg_record(
    record,
    *,
    isotope: str,
    spin: float | None = None,
    gamma_hz_per_t: float = 0.0,
    label: str | None = None,
) -> QuadrupolarSite:
    """Build a site from a ``quadrupolar_dft.AbinitEFGRecord``.

    ``record`` only needs ``cq_mhz`` and ``eta`` attributes, so any duck-typed
    object works.  ``spin`` defaults to the known value for ``isotope``.
    """

    if spin is None:
        try:
            spin = ISOTOPE_SPINS[isotope]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"unknown spin for isotope {isotope!r}; pass spin= explicitly"
            ) from exc
    return quadrupolar_site_from_cq(
        cq_hz=float(record.cq_mhz) * 1.0e6,
        eta=float(record.eta),
        spin=spin,
        isotope=isotope,
        gamma_hz_per_t=gamma_hz_per_t,
        label=label if label is not None else f"{isotope}@atom{record.atom_index}",
    )
