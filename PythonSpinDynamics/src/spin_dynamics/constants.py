"""Shared CODATA 2018 physical constants (SI units).

One source of truth for constants that the ESR, DNP, nano-MR and nonresonant
packages previously defined separately, from mixed CODATA vintages. Values are
the CODATA 2018 recommended values; ``h``, ``k_B`` and ``N_A`` are exact in the
2019 SI. Gyromagnetic ratios are magnitudes ``|gamma|/(2*pi)`` in Hz/T.
"""

import math

PLANCK_J_S = 6.62607015e-34
"""Planck constant ``h`` in J s (exact)."""

HBAR_J_S = PLANCK_J_S / (2.0 * math.pi)
"""Reduced Planck constant ``h/(2*pi)`` in J s."""

BOLTZMANN_J_PER_K = 1.380649e-23
"""Boltzmann constant ``k_B`` in J/K (exact)."""

AVOGADRO_PER_MOL = 6.02214076e23
"""Avogadro constant ``N_A`` in 1/mol (exact)."""

MU0_OVER_4PI = 1.00000000055e-7
"""Vacuum permeability over ``4*pi`` in T m/A (CODATA 2018 ``mu_0``)."""

BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
"""Bohr magneton ``mu_B`` in J/T."""

BOHR_MAGNETON_HZ_PER_T = 1.39962449361e10
"""Bohr magneton over Planck constant ``mu_B/h`` in Hz/T."""

FREE_ELECTRON_G = 2.00231930436256
"""Free-electron ``g`` factor magnitude ``|g_e|``."""

ELECTRON_GAMMA_HZ_PER_T = 28.0249514242e9
"""Free-electron ``|gamma_e|/(2*pi)`` in Hz/T (equals ``g_e * mu_B/h``)."""

PROTON_GAMMA_HZ_PER_T = 42.577478518e6
"""Proton (unshielded) ``gamma_p/(2*pi)`` in Hz/T."""

__all__ = [
    "AVOGADRO_PER_MOL",
    "BOHR_MAGNETON_HZ_PER_T",
    "BOHR_MAGNETON_J_PER_T",
    "BOLTZMANN_J_PER_K",
    "ELECTRON_GAMMA_HZ_PER_T",
    "FREE_ELECTRON_G",
    "HBAR_J_S",
    "MU0_OVER_4PI",
    "PLANCK_J_S",
    "PROTON_GAMMA_HZ_PER_T",
]
