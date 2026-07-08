"""Field solvers and dimension-agnostic spatial field maps.

The unified home for the package's field solvers. Three families:

* **Magnetostatics (B0/B1)** -- ``magnetostatics``: analytic permanent-magnet
  B0 and filament Biot-Savart coil B1.
* **Quasistatic (E/eddy)** -- ``quasistatic`` (induced E-field, eddy currents
  and power, first-order/Born) and ``eddy_modes`` (eddy-current L/R eigenmodes
  feeding the M5 gradient-driver pre-emphasis); ``coils`` supplies the geometry.
* **Grid + sampling** -- ``domain``/``maps``/``interpolate``/``positions``: the
  reusable :class:`SpatialDomain` voxel grid and :class:`SpatialFieldMaps`
  physics bundle shared by imaging (Eulerian) and diffusion (Lagrangian).

The spin-dynamics kernels only ever see a flat list of per-isochromat scalars;
spatial structure is factored here so imaging and diffusion share one
representation and one gradient-coupling rule.
"""

from spin_dynamics.fields.coils import (
    conducting_ring,
    cylindrical_shield,
    maxwell_pair,
    planar_spiral,
    solenoid,
)
from spin_dynamics.fields.coil_properties import (
    ALUMINIUM,
    ANNEALED_COPPER,
    HARD_DRAWN_COPPER,
    SILVER,
    CoilProperties,
    ConductorMaterial,
    medhurst_proximity_factor,
    sheath_helix_dispersion,
    solenoid_field_inductance,
    solenoid_properties,
)
from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.eddy_modes import EddyModes, EddyModeSpectrum
from spin_dynamics.fields.interpolate import dlinear_sample
from spin_dynamics.fields.magnetostatics import (
    GAMMA_PROTON,
    BarMagnet,
    FiniteMagnetRod,
    HalbachDipoleFieldMaps,
    MagnetFieldMaps,
    bar_array_b0,
    biot_savart,
    circular_loop,
    finite_magnet_array_b0,
    halbach_dipole_magnets,
    nmr_mouse_magnets,
    sample_halbach_dipole_field,
    sample_magnet_field,
)
from spin_dynamics.fields.maps import SpatialFieldMaps
from spin_dynamics.fields.nonlinear_magnetostatics import (
    AxisymmetricMagnetostatics,
    AxisymmetricSolution,
    BrauerBH,
    MagneticMaterial,
    PlanarMagnetostatics,
    PlanarSolution,
    air,
    linear_material,
    ndfeb,
    rf_ferrite,
    soft_iron,
)
from spin_dynamics.fields.positions import (
    gradient_offset,
    positions_nd,
    velocity_array,
)
from spin_dynamics.fields.scalar_potential_3d import (
    ReducedScalarPotential3D,
    ScalarPotentialSolution,
)
from spin_dynamics.fields.quasistatic import (
    CoilLoading,
    EddyResult,
    coil_inductance,
    coil_loading,
    eddy_currents,
    eddy_power,
    geometric_loss_integral,
    induced_efield,
    mutual_inductance,
    reflected_resistance,
    self_inductance_circular,
    vector_potential,
)

__all__ = [
    "SpatialDomain",
    "SpatialFieldMaps",
    "dlinear_sample",
    "gradient_offset",
    "positions_nd",
    "velocity_array",
    "GAMMA_PROTON",
    "BarMagnet",
    "FiniteMagnetRod",
    "HalbachDipoleFieldMaps",
    "MagnetFieldMaps",
    "bar_array_b0",
    "biot_savart",
    "circular_loop",
    "finite_magnet_array_b0",
    "halbach_dipole_magnets",
    "nmr_mouse_magnets",
    "sample_halbach_dipole_field",
    "sample_magnet_field",
    # coils
    "solenoid",
    "planar_spiral",
    "maxwell_pair",
    "conducting_ring",
    "cylindrical_shield",
    # coil properties (single-layer solenoid RF extraction)
    "ConductorMaterial",
    "ANNEALED_COPPER",
    "HARD_DRAWN_COPPER",
    "SILVER",
    "ALUMINIUM",
    "CoilProperties",
    "medhurst_proximity_factor",
    "sheath_helix_dispersion",
    "solenoid_properties",
    "solenoid_field_inductance",
    # quasistatic E / eddy
    "vector_potential",
    "induced_efield",
    "mutual_inductance",
    "self_inductance_circular",
    "eddy_power",
    "eddy_currents",
    "EddyResult",
    "geometric_loss_integral",
    "reflected_resistance",
    "coil_inductance",
    "coil_loading",
    "CoilLoading",
    "EddyModes",
    "EddyModeSpectrum",
    # nonlinear magnetostatics (ferrite / saturable iron / permanent magnets)
    "BrauerBH",
    "MagneticMaterial",
    "air",
    "linear_material",
    "rf_ferrite",
    "soft_iron",
    "ndfeb",
    "PlanarMagnetostatics",
    "PlanarSolution",
    "AxisymmetricMagnetostatics",
    "AxisymmetricSolution",
    # reduced scalar potential (3D magnetostatics)
    "ReducedScalarPotential3D",
    "ScalarPotentialSolution",
]
