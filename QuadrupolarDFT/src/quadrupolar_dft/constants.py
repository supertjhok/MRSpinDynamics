"""Physical constants and unit conversions used by QuadrupolarDFT."""

import math

ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_CONSTANT_J_S = 6.62607015e-34
REDUCED_PLANCK_CONSTANT_J_S = PLANCK_CONSTANT_J_S / (2.0 * math.pi)
BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
SPEED_OF_LIGHT_M_PER_S = 299792458.0
BOHR_RADIUS_M = 5.29177210544e-11
HARTREE_J = 4.3597447222060e-18

BARN_M2 = 1e-28
EFG_AU_TO_SI = HARTREE_J / (ELEMENTARY_CHARGE_C * BOHR_RADIUS_M**2)

ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
ANGSTROM_M = 1e-10
BOHR_TO_ANGSTROM = BOHR_RADIUS_M / ANGSTROM_M

# Angular frequency (rad/s) per spectroscopic wavenumber (cm^-1):
# omega = 2 pi c nu_tilde, with c expressed in cm/s.
ANGULAR_FREQUENCY_PER_WAVENUMBER_CM = (
    2.0 * math.pi * SPEED_OF_LIGHT_M_PER_S * 100.0
)

# Nuclear electric quadrupole moment of nitrogen-14 (barn). Used to convert an
# EFG V_zz into a coupling constant for the spin-1 NQR test cases.
NITROGEN_14_QUADRUPOLE_MOMENT_BARN = 0.02044
