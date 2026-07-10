"""Coupled thermal modeling: materials, heat sources, and solvers.

See ``docs/thermal_modeling.md`` for the physics options and the phased work
plan. Phase 0 provides thermal material properties and duty-cycled heat
sources built on the existing field solvers (coil resistance from
:mod:`spin_dynamics.fields.coil_properties`, sample SAR from
:mod:`spin_dynamics.fields.quasistatic`).
"""

from spin_dynamics.thermal.materials import (
    AIR,
    ALUMINA,
    COPPER_THERMAL,
    FILM_COEFFICIENTS,
    GLASS_BOROSILICATE,
    MUSCLE_TISSUE,
    PTFE,
    SAPPHIRE,
    ThermalMaterial,
    WATER_THERMAL,
)
from spin_dynamics.thermal.coupling import (
    CoupledCoilDrive,
    CoupledSAR,
    ThermalCoupling,
    ThermalCouplingResult,
    resistance_at_temperature,
)
from spin_dynamics.thermal.network import (
    STEFAN_BOLTZMANN,
    ThermalLink,
    ThermalNetwork,
    ThermalNode,
    ThermalTransientResult,
    conduction_conductance,
    convection_conductance,
    cylindrical_shell_conductance,
    radiation_link,
)
from spin_dynamics.thermal.sources import (
    ConstantSource,
    DutyCycledSource,
    average_coil_power,
    coil_joule_power,
    duty_cycle_from_pulse_params,
    gradient_waveform_power,
    sar_power_from_loading,
    sar_source_from_eddy,
    transmit_coil_current,
)

__all__ = [
    "AIR",
    "ALUMINA",
    "COPPER_THERMAL",
    "FILM_COEFFICIENTS",
    "GLASS_BOROSILICATE",
    "MUSCLE_TISSUE",
    "PTFE",
    "SAPPHIRE",
    "ThermalMaterial",
    "WATER_THERMAL",
    "CoupledCoilDrive",
    "CoupledSAR",
    "ThermalCoupling",
    "ThermalCouplingResult",
    "resistance_at_temperature",
    "STEFAN_BOLTZMANN",
    "ThermalLink",
    "ThermalNetwork",
    "ThermalNode",
    "ThermalTransientResult",
    "conduction_conductance",
    "convection_conductance",
    "cylindrical_shell_conductance",
    "radiation_link",
    "ConstantSource",
    "DutyCycledSource",
    "average_coil_power",
    "coil_joule_power",
    "duty_cycle_from_pulse_params",
    "gradient_waveform_power",
    "sar_power_from_loading",
    "sar_source_from_eddy",
    "transmit_coil_current",
]
