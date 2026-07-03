"""Presets and unit conversions for common quadrupolar nuclei.

Building a :class:`~spin_dynamics.nqr.systems.QuadrupolarSite` for a real nucleus
usually starts from either the quadrupole coupling constant ``C_Q = e Q V_zz / h``
(what ab initio EFG codes report) or the fundamental NQR line ``nu_Q``. This module
owns the ``C_Q <-> nu_Q`` mapping for arbitrary spin and a small registry of the
nuclear spin and gyromagnetic ratio for the quadrupolar nuclei people most often
simulate, so a site is one call away:

    site = quadrupolar_site("27Al", cq_hz=3.0e6, eta=0.1)   # spin-5/2, gamma filled in

Convention (matches ``mr_integration.conversions`` and
``nqr.hamiltonians.quadrupole_frequency_scale_hz``):

    H/h = C_Q / (4 I (2I-1)) * [3 Iz^2 - I(I+1) + eta (Ix^2 - Iy^2)]
    nu_Q = C_Q * d / (4 I (2I-1))      # d = fundamental_operator_gap(spin) = 3 or 6

``nu_Q`` is the ``eta = 0`` fundamental (lowest) transition; for spin > 3/2 the
higher axial lines follow at ``2 nu_Q``, ``3 nu_Q``, ...

Gyromagnetic ratios are magnitudes ``|gamma| / 2pi`` in Hz/T. Zero-field NQR (this
module's scope) does not depend on the sign; it only enters the optional weak
Zeeman perturbation, where a user who needs the sign can pass ``gamma_hz_per_t``.
"""

from __future__ import annotations

from dataclasses import dataclass

from spin_dynamics.nqr.operators import fundamental_operator_gap, validate_spin
from spin_dynamics.nqr.systems import QuadrupolarSite


@dataclass(frozen=True)
class NuclearIsotope:
    """Nuclear spin and gyromagnetic ratio for one isotope."""

    isotope: str
    spin: float
    gamma_hz_per_t: float  # |gamma| / 2pi, Hz/T


def _mhz_per_t(value: float) -> float:
    return float(value) * 1.0e6


# Curated registry of common quadrupolar nuclei (spin > 1/2). Gyromagnetic ratios
# are |gamma|/2pi in MHz/T (Harris/IUPAC recommended values), stored as Hz/T.
QUADRUPOLAR_ISOTOPES: dict[str, NuclearIsotope] = {
    iso.isotope: iso
    for iso in (
        # spin-1
        NuclearIsotope("2H", 1.0, _mhz_per_t(6.536)),
        NuclearIsotope("6Li", 1.0, _mhz_per_t(6.266)),
        NuclearIsotope("14N", 1.0, _mhz_per_t(3.0766)),
        # spin-3/2
        NuclearIsotope("7Li", 1.5, _mhz_per_t(16.546)),
        NuclearIsotope("9Be", 1.5, _mhz_per_t(5.983)),
        NuclearIsotope("11B", 1.5, _mhz_per_t(13.663)),
        NuclearIsotope("23Na", 1.5, _mhz_per_t(11.262)),
        NuclearIsotope("35Cl", 1.5, _mhz_per_t(4.1765)),
        NuclearIsotope("37Cl", 1.5, _mhz_per_t(3.4765)),
        NuclearIsotope("39K", 1.5, _mhz_per_t(1.9893)),
        NuclearIsotope("63Cu", 1.5, _mhz_per_t(11.319)),
        NuclearIsotope("65Cu", 1.5, _mhz_per_t(12.128)),
        NuclearIsotope("69Ga", 1.5, _mhz_per_t(10.248)),
        NuclearIsotope("71Ga", 1.5, _mhz_per_t(13.021)),
        NuclearIsotope("75As", 1.5, _mhz_per_t(7.315)),
        NuclearIsotope("79Br", 1.5, _mhz_per_t(10.704)),
        NuclearIsotope("81Br", 1.5, _mhz_per_t(11.538)),
        NuclearIsotope("87Rb", 1.5, _mhz_per_t(13.984)),
        # spin-5/2
        NuclearIsotope("17O", 2.5, _mhz_per_t(5.7742)),
        NuclearIsotope("25Mg", 2.5, _mhz_per_t(2.6083)),
        NuclearIsotope("27Al", 2.5, _mhz_per_t(11.1031)),
        NuclearIsotope("55Mn", 2.5, _mhz_per_t(10.5761)),
        NuclearIsotope("67Zn", 2.5, _mhz_per_t(2.6694)),
        NuclearIsotope("121Sb", 2.5, _mhz_per_t(10.2551)),
        NuclearIsotope("127I", 2.5, _mhz_per_t(8.5778)),
        # spin-7/2
        NuclearIsotope("45Sc", 3.5, _mhz_per_t(10.3591)),
        NuclearIsotope("51V", 3.5, _mhz_per_t(11.2133)),
        NuclearIsotope("59Co", 3.5, _mhz_per_t(10.077)),
        NuclearIsotope("123Sb", 3.5, _mhz_per_t(5.5532)),
        NuclearIsotope("133Cs", 3.5, _mhz_per_t(5.6234)),
        NuclearIsotope("139La", 3.5, _mhz_per_t(6.0612)),
        # spin-9/2
        NuclearIsotope("87Sr", 4.5, _mhz_per_t(1.8525)),
        NuclearIsotope("93Nb", 4.5, _mhz_per_t(10.4523)),
        NuclearIsotope("115In", 4.5, _mhz_per_t(9.3856)),
        NuclearIsotope("209Bi", 4.5, _mhz_per_t(6.9628)),
    )
}


def nu_q_over_cq(spin: float) -> float:
    """Return the ratio ``nu_Q / C_Q = d / (4 I (2I-1))`` for a given spin."""

    spin = validate_spin(spin)
    return fundamental_operator_gap(spin) / (4.0 * spin * (2.0 * spin - 1.0))


def nu_q_from_cq_hz(cq_hz: float, spin: float) -> float:
    """Return the fundamental line ``nu_Q`` (Hz) for a coupling constant ``C_Q``."""

    return abs(float(cq_hz)) * nu_q_over_cq(spin)


def cq_hz_from_nu_q(nu_q_hz: float, spin: float) -> float:
    """Return ``C_Q`` (Hz) for a fundamental line ``nu_Q`` (inverse of above)."""

    return float(nu_q_hz) / nu_q_over_cq(spin)


def quadrupolar_isotope(isotope: str) -> NuclearIsotope:
    """Return the registry entry for ``isotope`` (e.g. ``"27Al"``)."""

    try:
        return QUADRUPOLAR_ISOTOPES[str(isotope)]
    except KeyError as exc:
        known = ", ".join(sorted(QUADRUPOLAR_ISOTOPES))
        raise KeyError(
            f"unknown quadrupolar isotope {isotope!r}; known isotopes: {known}. "
            "Pass spin= and gamma_hz_per_t= explicitly to use another nucleus."
        ) from exc


def quadrupolar_site(
    isotope: str,
    *,
    cq_hz: float | None = None,
    nu_q_hz: float | None = None,
    eta: float = 0.0,
    spin: float | None = None,
    gamma_hz_per_t: float | None = None,
    label: str | None = None,
) -> QuadrupolarSite:
    """Build a :class:`QuadrupolarSite` for a named nucleus.

    Provide exactly one of ``cq_hz`` (quadrupole coupling constant) or
    ``nu_q_hz`` (fundamental NQR line). ``spin`` and ``gamma_hz_per_t`` default to
    the registry values for ``isotope`` and may be overridden (required if the
    isotope is not in the registry).
    """

    if (cq_hz is None) == (nu_q_hz is None):
        raise ValueError("provide exactly one of cq_hz or nu_q_hz")

    if spin is None or gamma_hz_per_t is None:
        entry = quadrupolar_isotope(isotope)
        spin = entry.spin if spin is None else spin
        gamma_hz_per_t = entry.gamma_hz_per_t if gamma_hz_per_t is None else gamma_hz_per_t

    frequency_hz = (
        nu_q_from_cq_hz(cq_hz, spin) if cq_hz is not None else float(nu_q_hz)
    )
    return QuadrupolarSite(
        spin=float(spin),
        quadrupole_frequency_hz=frequency_hz,
        eta=float(eta),
        gamma_hz_per_t=float(gamma_hz_per_t),
        isotope=str(isotope),
        label=label if label is not None else str(isotope),
    )
