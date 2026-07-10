"""Thermal material properties and convection film coefficients.

Complements :class:`spin_dynamics.fields.coil_properties.ConductorMaterial`
(which owns *electrical* resistivity and its temperature model): this module
owns the *thermal* constants -- conductivity ``k`` (W/m/K), mass density
(kg/m^3), specific heat ``c_p`` (J/kg/K), and surface emissivity -- used by the
lumped-network and conduction solvers.

Preset values are room-temperature (~293-300 K) handbook numbers, adequate for
the package's quasi-static coupling loop; strongly cryogenic problems should
supply their own temperature-resolved values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ThermalMaterial",
    "COPPER_THERMAL",
    "SAPPHIRE",
    "ALUMINA",
    "PTFE",
    "GLASS_BOROSILICATE",
    "WATER_THERMAL",
    "MUSCLE_TISSUE",
    "AIR",
    "FILM_COEFFICIENTS",
]


@dataclass(frozen=True)
class ThermalMaterial:
    """Thermal constants of a solid, liquid, or gas.

    ``conductivity`` in W/m/K, ``density`` in kg/m^3, ``specific_heat`` in
    J/kg/K, ``emissivity`` dimensionless (0 disables radiation from surfaces
    of this material unless a link overrides it).
    """

    name: str
    conductivity: float
    density: float
    specific_heat: float
    emissivity: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("conductivity", self.conductivity),
            ("density", self.density),
            ("specific_heat", self.specific_heat),
        ):
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{label} must be finite and positive")
        if not (0.0 <= self.emissivity <= 1.0):
            raise ValueError("emissivity must be in [0, 1]")

    @property
    def volumetric_heat_capacity(self) -> float:
        """``rho * c_p`` in J/(m^3 K)."""

        return self.density * self.specific_heat

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity ``alpha = k / (rho c_p)`` in m^2/s."""

        return self.conductivity / self.volumetric_heat_capacity

    def heat_capacity(self, volume: float) -> float:
        """Lumped heat capacity ``C = rho c_p V`` (J/K) of a given volume (m^3)."""

        if volume <= 0 or not np.isfinite(volume):
            raise ValueError("volume must be finite and positive")
        return self.volumetric_heat_capacity * volume


COPPER_THERMAL = ThermalMaterial("copper", 401.0, 8960.0, 385.0, emissivity=0.03)
SAPPHIRE = ThermalMaterial("sapphire", 35.0, 3980.0, 760.0, emissivity=0.5)
ALUMINA = ThermalMaterial("alumina", 30.0, 3900.0, 880.0, emissivity=0.75)
PTFE = ThermalMaterial("ptfe", 0.25, 2200.0, 1000.0, emissivity=0.85)
GLASS_BOROSILICATE = ThermalMaterial("borosilicate glass", 1.14, 2230.0, 830.0, emissivity=0.9)
WATER_THERMAL = ThermalMaterial("water", 0.6, 997.0, 4184.0, emissivity=0.96)
# Skeletal muscle at body temperature (IT'IS tissue database rounded values).
MUSCLE_TISSUE = ThermalMaterial("muscle tissue", 0.49, 1090.0, 3421.0, emissivity=0.97)
AIR = ThermalMaterial("air", 0.026, 1.18, 1005.0)

# Representative convection film coefficients h in W/(m^2 K). Order-of-
# magnitude engineering values: pick per geometry or fit from a measured
# step response; these are inputs, not predictions (no flow solving).
FILM_COEFFICIENTS: dict[str, float] = {
    "still_air": 5.0,
    "natural_convection_air": 10.0,
    "forced_air": 50.0,
    "still_water": 100.0,
    "flowing_water": 500.0,
}
