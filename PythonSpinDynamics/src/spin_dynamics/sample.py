"""Sample descriptions: geometry, spin density, temperature, relaxation.

This module owns the *sample-side* physical properties and keeps them distinct
from detector/coil properties, which may sit at a different temperature. The
probe parameter sets (`sp`) keep describing the circuit -- ``sp.T`` is the
coil/circuit temperature -- while the sample temperature, spin count, and
thermal polarization live here.

The derived quantities follow the spin-noise references summarized in
``docs/spin_noise.md``: the Curie-law magnetization density, the thermal
polarization ``p``, and the transverse fluctuation scales (total moment
``sqrt(N) * gamma * hbar / 2`` and its normalized counterpart
``1 / (sqrt(N) * p)``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.radiation_damping import (
    AVOGADRO,
    HBAR,
    KB,
    PROTON_GAMMA,
    RadiationDampingSample,
)

__all__ = [
    "SampleGeometry",
    "Sample",
    "cylinder_geometry",
    "sphere_geometry",
    "cuboid_geometry",
    "water_sample",
]


@dataclass(frozen=True)
class SampleGeometry:
    """Sample shape and dimensions in meters.

    ``kind`` is one of ``"cylinder"`` (radius, length), ``"sphere"`` (radius),
    ``"cuboid"`` (x, y, z edge lengths), or ``"explicit"`` (volume supplied
    directly). Dimensions are stored in ``dimensions`` in the order listed.
    """

    kind: str
    dimensions: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = {"cylinder": 2, "sphere": 1, "cuboid": 3, "explicit": 1}
        if self.kind not in expected:
            raise ValueError(
                "geometry kind must be 'cylinder', 'sphere', 'cuboid', or 'explicit'"
            )
        dims = tuple(float(d) for d in self.dimensions)
        if len(dims) != expected[self.kind]:
            raise ValueError(
                f"{self.kind} geometry needs {expected[self.kind]} dimension(s)"
            )
        if any(not np.isfinite(d) or d <= 0 for d in dims):
            raise ValueError("geometry dimensions must be finite and positive")
        object.__setattr__(self, "dimensions", dims)

    @property
    def volume(self) -> float:
        """Sample volume in m^3."""

        if self.kind == "cylinder":
            radius, length = self.dimensions
            return float(np.pi * radius**2 * length)
        if self.kind == "sphere":
            (radius,) = self.dimensions
            return float(4.0 / 3.0 * np.pi * radius**3)
        if self.kind == "cuboid":
            x, y, z = self.dimensions
            return float(x * y * z)
        (volume,) = self.dimensions
        return float(volume)


def cylinder_geometry(radius: float, length: float) -> SampleGeometry:
    """Cylindrical sample of the given radius and length (meters)."""

    return SampleGeometry(kind="cylinder", dimensions=(radius, length))


def sphere_geometry(radius: float) -> SampleGeometry:
    """Spherical sample of the given radius (meters)."""

    return SampleGeometry(kind="sphere", dimensions=(radius,))


def cuboid_geometry(x: float, y: float, z: float) -> SampleGeometry:
    """Rectangular-cuboid sample with the given edge lengths (meters)."""

    return SampleGeometry(kind="cuboid", dimensions=(x, y, z))


@dataclass(frozen=True)
class Sample:
    """Physical sample description, distinct from the detector/coil.

    ``spin_density`` is the number of resonant spins per m^3, ``temperature``
    the sample (spin bath) temperature in kelvin -- the coil temperature stays
    on the probe parameters. ``gamma`` is the gyromagnetic ratio in rad/s/T and
    ``spin`` the spin quantum number ``I``. ``t1``/``t2`` are relaxation times
    in seconds (``t2`` sets the spin-noise correlation time; ``t1`` is inert at
    equilibrium but kept for workflow reuse).
    """

    name: str
    geometry: SampleGeometry
    spin_density: float
    temperature: float
    gamma: float = PROTON_GAMMA
    spin: float = 0.5
    t1: float = np.inf
    t2: float = np.inf
    polarization_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.spin_density <= 0 or not np.isfinite(self.spin_density):
            raise ValueError("spin_density must be finite and positive")
        if self.temperature <= 0 or not np.isfinite(self.temperature):
            raise ValueError("temperature must be finite and positive")
        if self.gamma <= 0 or not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite and positive")
        twice_spin = 2.0 * float(self.spin)
        if self.spin <= 0 or abs(twice_spin - round(twice_spin)) > 1e-9:
            raise ValueError("spin must be a positive half-integer")
        for label, value in (("t1", self.t1), ("t2", self.t2)):
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        if self.polarization_scale <= 0 or not np.isfinite(self.polarization_scale):
            raise ValueError("polarization_scale must be finite and positive")

    @property
    def volume(self) -> float:
        """Sample volume in m^3."""

        return self.geometry.volume

    @property
    def number_of_spins(self) -> float:
        """Total number of resonant spins ``N`` in the sample."""

        return self.spin_density * self.volume

    def polarization(self, field_tesla: float) -> float:
        """High-temperature thermal polarization ``p`` at ``field_tesla``.

        For spin 1/2 this is ``tanh(gamma*hbar*B0 / (2*k*T))`` in the linear
        regime; the general-``I`` Curie form is used so that
        ``M0 = N/V * gamma * hbar * I * p_eff`` reproduces
        :func:`magnetization_density`. ``polarization_scale`` models
        hyperpolarization or saturation.
        """

        if field_tesla <= 0:
            raise ValueError("field_tesla must be positive")
        spin_factor = (float(self.spin) + 1.0) / 3.0
        p = (
            self.gamma
            * HBAR
            * spin_factor
            * float(field_tesla)
            / (KB * self.temperature)
        )
        return float(p * self.polarization_scale)

    def magnetization_density(self, field_tesla: float) -> float:
        """Curie-law equilibrium magnetization density ``M0`` in A/m.

        ``M0 = n * gamma^2 * hbar^2 * I(I+1) * B0 / (3 * k * T)``, scaled by
        ``polarization_scale``. Matches
        :func:`spin_dynamics.radiation_damping.proton_thermal_magnetization_density`
        for water protons.
        """

        if field_tesla <= 0:
            raise ValueError("field_tesla must be positive")
        spin_factor = float(self.spin) * (float(self.spin) + 1.0)
        m0 = (
            self.spin_density
            * self.gamma**2
            * HBAR**2
            * spin_factor
            * float(field_tesla)
            / (3.0 * KB * self.temperature)
        )
        return float(m0 * self.polarization_scale)

    def transverse_fluctuation_moment(self) -> float:
        """RMS transverse fluctuation magnetic moment in A*m^2.

        Per spin the transverse variance is ``(gamma*hbar)^2 * I(I+1)/3``, so
        the ensemble RMS moment is ``gamma*hbar*sqrt(N * I(I+1)/3)``. For spin
        1/2 this is Hoult & Ginsberg Eq. 21, ``sqrt(N) * gamma*hbar/2`` (the
        spin-noise EMF being ``w0 * B1_hat * (gamma*hbar/2) * sqrt(N)``).
        """

        spin_factor = float(self.spin) * (float(self.spin) + 1.0) / 3.0
        return float(
            self.gamma * HBAR * np.sqrt(self.number_of_spins * spin_factor)
        )

    def normalized_transverse_fluctuation(self, field_tesla: float) -> float:
        """Stationary RMS transverse fluctuation relative to ``M0 * V_s``.

        For spin 1/2, ``M0*V_s = N * (gamma*hbar/2) * p`` gives the compact
        form ``1 / (sqrt(N) * p)``. This is the ``m_rms`` used by the
        stochastic spin-noise model, in the normalized units of
        :mod:`spin_dynamics.radiation_damping` where ``mth = 1``.
        """

        m0_vs = self.magnetization_density(field_tesla) * self.volume
        return float(self.transverse_fluctuation_moment() / m0_vs)

    def to_radiation_damping_sample(self, field_tesla: float) -> RadiationDampingSample:
        """Adapter to the RD convenience preset used by existing workflows."""

        concentration = self.spin_density / (1000.0 * AVOGADRO)
        return RadiationDampingSample(
            name=self.name,
            equilibrium_magnetization=self.magnetization_density(field_tesla),
            field_tesla=float(field_tesla),
            temperature_kelvin=self.temperature,
            proton_concentration_mol_per_liter=concentration,
            polarization_scale=self.polarization_scale,
        )

    def with_temperature(self, temperature: float) -> "Sample":
        """Copy of this sample at a different temperature."""

        return replace(self, temperature=float(temperature))


def water_sample(
    geometry: SampleGeometry,
    *,
    temperature: float = 300.0,
    t1: float = 2.0,
    t2: float = 2.0,
    polarization_scale: float = 1.0,
) -> Sample:
    """Liquid-water proton sample (~111 mol/L of protons) with the given shape."""

    spin_density = 111.0 * 1000.0 * AVOGADRO
    return Sample(
        name="water protons",
        geometry=geometry,
        spin_density=spin_density,
        temperature=float(temperature),
        gamma=PROTON_GAMMA,
        spin=0.5,
        t1=float(t1),
        t2=float(t2),
        polarization_scale=float(polarization_scale),
    )
