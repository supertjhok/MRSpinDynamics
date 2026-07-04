"""Nuclear isotope registry for multinuclear low-field J-coupling simulations.

The zero/ultra-low-field (ZULF) and Earth's-field regime is intrinsically
multinuclear: at 50 uT every nucleus sits within a ~2 kHz band, so a simulation
needs the true Larmor frequency ``nu_i = gamma_i B0`` of each spin. This module
extends the quadrupolar registry in :mod:`spin_dynamics.nqr.isotopes` (which
already owns spin-1 ``14N``, ``2H``, ... and higher-spin nuclei) with the common
spin-1/2 nuclei so both are available through one lookup.

Gyromagnetic ratios are magnitudes ``|gamma| / 2pi`` in Hz/T (Harris/IUPAC
recommended values). A few nuclei (``15N``, ``29Si``) have negative ``gamma``;
the stored magnitude fixes the resonance position, and code that needs the sign
can pass ``gamma_hz_per_t`` explicitly.
"""

from __future__ import annotations

from spin_dynamics.nqr.isotopes import QUADRUPOLAR_ISOTOPES, NuclearIsotope


def _mhz_per_t(value: float) -> float:
    return float(value) * 1.0e6


# Common spin-1/2 nuclei, |gamma|/2pi stored as Hz/T.
SPIN_HALF_ISOTOPES: dict[str, NuclearIsotope] = {
    iso.isotope: iso
    for iso in (
        NuclearIsotope("1H", 0.5, _mhz_per_t(42.5774806)),
        NuclearIsotope("19F", 0.5, _mhz_per_t(40.0775017)),
        NuclearIsotope("13C", 0.5, _mhz_per_t(10.7083974)),
        NuclearIsotope("15N", 0.5, _mhz_per_t(4.3172655)),
        NuclearIsotope("31P", 0.5, _mhz_per_t(17.2514599)),
        NuclearIsotope("29Si", 0.5, _mhz_per_t(8.4654734)),
    )
}


# Merged view: spin-1/2 nuclei plus the quadrupolar registry (spin > 1/2).
NUCLEAR_ISOTOPES: dict[str, NuclearIsotope] = {
    **SPIN_HALF_ISOTOPES,
    **QUADRUPOLAR_ISOTOPES,
}


def nuclear_isotope(isotope: str) -> NuclearIsotope:
    """Return the registry entry (spin and ``|gamma|/2pi``) for ``isotope``."""

    try:
        return NUCLEAR_ISOTOPES[str(isotope)]
    except KeyError as exc:
        known = ", ".join(sorted(NUCLEAR_ISOTOPES))
        raise KeyError(
            f"unknown isotope {isotope!r}; known isotopes: {known}. "
            "Pass spin= and gamma_hz_per_t= explicitly to use another nucleus."
        ) from exc


def larmor_frequency_hz(
    isotope: str,
    b0_tesla: float,
    *,
    gamma_hz_per_t: float | None = None,
) -> float:
    """Return the Larmor frequency ``gamma * B0`` in hertz for ``isotope``.

    The magnitude comes from the registry unless ``gamma_hz_per_t`` is supplied
    (use a signed value if the precession sense matters for your system).
    """

    gamma = (
        nuclear_isotope(isotope).gamma_hz_per_t
        if gamma_hz_per_t is None
        else float(gamma_hz_per_t)
    )
    return float(gamma) * float(b0_tesla)
