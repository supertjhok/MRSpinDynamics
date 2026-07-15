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
from spin_dynamics.fields.coil_peec import (
    Conductor,
    GroundedBox,
    GroundPlane,
    PEECCoilProperties,
    PEECImpedance,
    coil_properties_peec,
    conductor_from_segments,
    capacitance_to_ground,
    current_distribution,
    extract_impedance,
    extract_impedance_surface,
    helical_solenoid,
    radiation_resistance,
    self_capacitance,
    self_resonant_frequency,
)
from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.eddy_modes import EddyModes, EddyModeSpectrum
from spin_dynamics.fields.gradient_coils import (
    CylindricalGradientSystem,
    CylindricalWindingSurface,
    GradientCoilDesignResult,
    GradientCoilRegularizationPath,
    build_cylindrical_gradient_system,
    design_cylindrical_gradient_coil,
    linear_gradient_target,
    solve_gradient_coil,
    solve_regularization_path,
    spherical_target_points,
)
from spin_dynamics.fields.gradient_engineering import (
    GradientCoilEngineeringMetrics,
    GradientElectricalMetrics,
    GradientFieldMetrics,
    GradientImagingFieldMap,
    GradientMechanicalMetrics,
    estimate_gradient_electrical_metrics,
    gradient_coil_engineering_metrics,
    winding_field,
    winding_force_torque,
    winding_imaging_field_map,
    winding_peec_conductors,
    winding_to_gradient_driver,
)
from spin_dynamics.fields.gradient_shielding import (
    ActivelyShieldedGradientDesignResult,
    ActivelyShieldedGradientSystem,
    ActivelyShieldedRegularizationPath,
    build_actively_shielded_gradient_system,
    cylindrical_shield_points,
    design_actively_shielded_gradient_coil,
    solve_actively_shielded_gradient_coil,
    solve_actively_shielded_regularization_path,
    spherical_shell_points,
)
from spin_dynamics.fields.gradient_windings import (
    ActivelyShieldedWinding,
    GradientWinding,
    WindingContour,
    extract_actively_shielded_winding,
    extract_winding_contours,
    stream_function_contours,
    winding_from_design,
    winding_contour_levels,
    winding_segments,
)
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
    segment_field_sensitivity,
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
    "segment_field_sensitivity",
    # coils
    "solenoid",
    "planar_spiral",
    "maxwell_pair",
    "conducting_ring",
    "cylindrical_shield",
    # stream-function gradient-coil design
    "CylindricalWindingSurface",
    "CylindricalGradientSystem",
    "GradientCoilDesignResult",
    "GradientCoilRegularizationPath",
    "ActivelyShieldedGradientSystem",
    "ActivelyShieldedGradientDesignResult",
    "ActivelyShieldedRegularizationPath",
    "WindingContour",
    "GradientWinding",
    "ActivelyShieldedWinding",
    "spherical_target_points",
    "linear_gradient_target",
    "build_cylindrical_gradient_system",
    "solve_gradient_coil",
    "solve_regularization_path",
    "design_cylindrical_gradient_coil",
    "spherical_shell_points",
    "cylindrical_shield_points",
    "build_actively_shielded_gradient_system",
    "solve_actively_shielded_gradient_coil",
    "solve_actively_shielded_regularization_path",
    "design_actively_shielded_gradient_coil",
    "winding_contour_levels",
    "extract_winding_contours",
    "winding_from_design",
    "extract_actively_shielded_winding",
    "stream_function_contours",
    "winding_segments",
    # realized gradient-coil engineering and workflow adapters
    "GradientFieldMetrics",
    "GradientElectricalMetrics",
    "GradientMechanicalMetrics",
    "GradientCoilEngineeringMetrics",
    "GradientImagingFieldMap",
    "winding_field",
    "winding_force_torque",
    "estimate_gradient_electrical_metrics",
    "gradient_coil_engineering_metrics",
    "winding_peec_conductors",
    "winding_to_gradient_driver",
    "winding_imaging_field_map",
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
    # coil PEEC solver (arbitrary-geometry L/R/C/Q/f_res)
    "Conductor",
    "conductor_from_segments",
    "helical_solenoid",
    "extract_impedance",
    "extract_impedance_surface",
    "current_distribution",
    "self_capacitance",
    "capacitance_to_ground",
    "GroundedBox",
    "GroundPlane",
    "self_resonant_frequency",
    "radiation_resistance",
    "coil_properties_peec",
    "PEECImpedance",
    "PEECCoilProperties",
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
