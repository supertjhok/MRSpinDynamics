"""Nuclear-bath polarization and spatial-density models for statistical nano-NMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from spin_dynamics.nano_mr.geometry import (
    ISOTOPE_GAMMA_HZ_PER_T,
    SurfaceGeometry,
)
from spin_dynamics.nqr.operators import validate_spin


PLANCK_CONSTANT_J_S = 6.62607015e-34
"""Planck constant in joule seconds."""

BOLTZMANN_CONSTANT_J_PER_K = 1.380649e-23
"""Boltzmann constant in joules per kelvin."""

ISOTOPE_SPIN = {
    "1H": 0.5,
    "13C": 0.5,
    "19F": 0.5,
    "29Si": 0.5,
    "31P": 0.5,
}
"""Spin quantum numbers for the built-in nano-MR isotope presets."""

PolarizationMode = Literal["statistical", "thermal", "fixed"]
_POLARIZATION_MODES = ("statistical", "thermal", "fixed")


@dataclass(frozen=True)
class PolarizationScalingResult:
    """Coherent and statistical spin-projection scaling for ``N`` nuclei."""

    spin_count: float
    mean_projection_per_spin: float
    projection_variance_per_spin: float
    coherent_mean_projection: float
    statistical_rms_projection: float


@dataclass(frozen=True)
class NuclearBathSpecies:
    """One isotope and its polarization/correlation model.

    ``statistical`` uses an unpolarized spin distribution. ``thermal`` uses
    Boltzmann populations at ``temperature_kelvin`` and the supplied static
    field. ``fixed`` uses the maximum-entropy spin distribution whose mean
    projection is ``polarization_fraction * spin``.

    ``correlation_time_seconds`` gives the exponential correlation time of the
    rotating transverse field and therefore the Lorentzian half-width
    ``1 / (2*pi*correlation_time_seconds)`` in hertz.
    """

    isotope: str
    gamma_hz_per_t: float
    spin: float = 0.5
    polarization_mode: PolarizationMode = "statistical"
    temperature_kelvin: float = 300.0
    polarization_fraction: float = 0.0
    correlation_time_seconds: float = 100.0e-6
    label: str = ""

    def __post_init__(self) -> None:
        gamma = float(self.gamma_hz_per_t)
        spin = validate_spin(self.spin)
        temperature = float(self.temperature_kelvin)
        polarization = float(self.polarization_fraction)
        correlation = float(self.correlation_time_seconds)
        if gamma == 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma_hz_per_t must be finite and non-zero")
        if self.polarization_mode not in _POLARIZATION_MODES:
            raise ValueError(
                f"polarization_mode must be one of {_POLARIZATION_MODES}"
            )
        if temperature <= 0.0 or not np.isfinite(temperature):
            raise ValueError("temperature_kelvin must be positive and finite")
        if not -1.0 <= polarization <= 1.0 or not np.isfinite(polarization):
            raise ValueError("polarization_fraction must lie in [-1, 1]")
        if correlation <= 0.0 or not np.isfinite(correlation):
            raise ValueError(
                "correlation_time_seconds must be positive and finite"
            )
        object.__setattr__(self, "isotope", str(self.isotope))
        object.__setattr__(self, "gamma_hz_per_t", gamma)
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "polarization_fraction", polarization)
        object.__setattr__(self, "correlation_time_seconds", correlation)
        object.__setattr__(
            self,
            "label",
            str(self.label) if self.label else str(self.isotope),
        )

    @classmethod
    def from_isotope(
        cls,
        isotope: str,
        **kwargs,
    ) -> "NuclearBathSpecies":
        """Construct a species from built-in gamma and spin values."""

        key = str(isotope)
        try:
            gamma = ISOTOPE_GAMMA_HZ_PER_T[key]
            spin = ISOTOPE_SPIN[key]
        except KeyError as exc:
            raise ValueError(f"unknown isotope preset: {isotope!r}") from exc
        return cls(
            isotope=key,
            gamma_hz_per_t=gamma,
            spin=spin,
            **kwargs,
        )

    def level_probabilities(self, field_tesla: float) -> np.ndarray:
        """Return populations ordered from ``m=-I`` through ``m=+I``."""

        field = float(field_tesla)
        if field < 0.0 or not np.isfinite(field):
            raise ValueError("field_tesla must be finite and non-negative")
        levels = _spin_projections(self.spin)
        if self.polarization_mode == "statistical":
            return np.full(levels.size, 1.0 / levels.size)
        if self.polarization_mode == "thermal":
            exponent = (
                PLANCK_CONSTANT_J_S
                * self.gamma_hz_per_t
                * field
                / (BOLTZMANN_CONSTANT_J_PER_K * self.temperature_kelvin)
            )
        else:
            target = self.polarization_fraction * self.spin
            exponent = _maximum_entropy_exponent(levels, target)
        weights = np.exp(exponent * levels - np.max(exponent * levels))
        return weights / np.sum(weights)

    def mean_spin_projection(self, field_tesla: float) -> float:
        """Return ``<I_z>`` along the static-field direction."""

        levels = _spin_projections(self.spin)
        return float(levels @ self.level_probabilities(field_tesla))

    def transverse_spin_second_moment(self, field_tesla: float) -> float:
        """Return ``<I_x^2> = <I_y^2>`` for the axial population state."""

        levels = _spin_projections(self.spin)
        probabilities = self.level_probabilities(field_tesla)
        mean_square_z = float((levels**2) @ probabilities)
        return 0.5 * (self.spin * (self.spin + 1.0) - mean_square_z)

    def polarization_scaling(
        self,
        field_tesla: float,
        spin_count: float,
    ) -> PolarizationScalingResult:
        """Return ``N`` versus ``sqrt(N)`` polarization scaling.

        The coherent projection is ``N*<I_z>``. Independent statistical
        fluctuations have RMS projection
        ``sqrt(N*(<I_z^2>-<I_z>^2))``.
        """

        count = float(spin_count)
        if count < 0.0 or not np.isfinite(count):
            raise ValueError("spin_count must be finite and non-negative")
        levels = _spin_projections(self.spin)
        probabilities = self.level_probabilities(field_tesla)
        mean = float(levels @ probabilities)
        variance = float((levels**2) @ probabilities - mean**2)
        return PolarizationScalingResult(
            spin_count=count,
            mean_projection_per_spin=mean,
            projection_variance_per_spin=variance,
            coherent_mean_projection=count * mean,
            statistical_rms_projection=float(np.sqrt(count * variance)),
        )


@dataclass(frozen=True)
class UniformBathComponent:
    """One uniformly distributed nuclear species."""

    species: NuclearBathSpecies
    number_density_m3: float

    def __post_init__(self) -> None:
        density = float(self.number_density_m3)
        if density < 0.0 or not np.isfinite(density):
            raise ValueError("number_density_m3 must be finite and non-negative")
        if not isinstance(self.species, NuclearBathSpecies):
            raise TypeError("species must be a NuclearBathSpecies")
        object.__setattr__(self, "number_density_m3", density)


@dataclass(frozen=True)
class UniformNuclearLayer:
    """Uniform layer or half-space above a planar sensor surface.

    ``bottom_offset_nm`` is a non-negative gap or coating thickness measured
    from the host surface into the sample. ``thickness_nm=None`` denotes a
    half-space.
    """

    surface: SurfaceGeometry
    components: tuple[UniformBathComponent, ...]
    thickness_nm: float | None = None
    bottom_offset_nm: float = 0.0
    label: str = "uniform nuclear layer"

    def __post_init__(self) -> None:
        if not isinstance(self.surface, SurfaceGeometry):
            raise TypeError("surface must be a SurfaceGeometry")
        components = tuple(self.components)
        if not components:
            raise ValueError("components must not be empty")
        if not all(isinstance(item, UniformBathComponent) for item in components):
            raise TypeError("components must contain UniformBathComponent values")
        offset = float(self.bottom_offset_nm)
        if offset < 0.0 or not np.isfinite(offset):
            raise ValueError("bottom_offset_nm must be finite and non-negative")
        if self.thickness_nm is None:
            thickness = None
        else:
            thickness = float(self.thickness_nm)
            if thickness <= 0.0 or not np.isfinite(thickness):
                raise ValueError("thickness_nm must be positive and finite")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "bottom_offset_nm", offset)
        object.__setattr__(self, "thickness_nm", thickness)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True, eq=False)
class VoxelBathComponent:
    """One isotope with scalar or per-voxel number density."""

    species: NuclearBathSpecies
    number_density_m3: float | np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.species, NuclearBathSpecies):
            raise TypeError("species must be a NuclearBathSpecies")
        density = np.asarray(self.number_density_m3, dtype=np.float64)
        if density.ndim > 1 or np.any(density < 0.0) or not np.all(np.isfinite(density)):
            raise ValueError(
                "number_density_m3 must be a finite non-negative scalar or vector"
            )
        object.__setattr__(self, "number_density_m3", density.copy())


@dataclass(frozen=True, eq=False)
class VoxelNuclearSample:
    """Arbitrary voxel centers, volumes, and isotope densities."""

    positions_lab_nm: np.ndarray
    voxel_volumes_nm3: np.ndarray
    components: tuple[VoxelBathComponent, ...]
    label: str = "voxel nuclear sample"

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_lab_nm, dtype=np.float64)
        volumes = np.asarray(self.voxel_volumes_nm3, dtype=np.float64)
        components = tuple(self.components)
        if (
            positions.ndim != 2
            or positions.shape[1] != 3
            or positions.shape[0] == 0
            or not np.all(np.isfinite(positions))
        ):
            raise ValueError("positions_lab_nm must be a finite non-empty Nx3 array")
        if (
            volumes.shape != (positions.shape[0],)
            or np.any(volumes <= 0.0)
            or not np.all(np.isfinite(volumes))
        ):
            raise ValueError("voxel_volumes_nm3 must be a positive length-N vector")
        if not components:
            raise ValueError("components must not be empty")
        for component in components:
            if not isinstance(component, VoxelBathComponent):
                raise TypeError("components must contain VoxelBathComponent values")
            density = component.number_density_m3
            if density.ndim == 1 and density.shape != (positions.shape[0],):
                raise ValueError("per-voxel number density must have length N")
        object.__setattr__(self, "positions_lab_nm", positions.copy())
        object.__setattr__(self, "voxel_volumes_nm3", volumes.copy())
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "label", str(self.label))

    def density_vector_m3(self, component: VoxelBathComponent) -> np.ndarray:
        """Return one component's number density as a length-N vector."""

        density = component.number_density_m3
        if density.ndim == 0:
            return np.full(self.positions_lab_nm.shape[0], float(density))
        return density.copy()


NuclearBathSample: TypeAlias = UniformNuclearLayer | VoxelNuclearSample


def _spin_projections(spin: float) -> np.ndarray:
    return np.linspace(-spin, spin, int(round(2.0 * spin)) + 1)


def _maximum_entropy_exponent(levels: np.ndarray, target_mean: float) -> float:
    if np.isclose(target_mean, levels[0], atol=1.0e-14):
        return -100.0
    if np.isclose(target_mean, levels[-1], atol=1.0e-14):
        return 100.0
    low = -100.0
    high = 100.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        weights = np.exp(middle * levels - np.max(middle * levels))
        mean = float(levels @ weights / np.sum(weights))
        if mean < target_mean:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


__all__ = [
    "BOLTZMANN_CONSTANT_J_PER_K",
    "ISOTOPE_SPIN",
    "NuclearBathSample",
    "NuclearBathSpecies",
    "PLANCK_CONSTANT_J_S",
    "PolarizationMode",
    "PolarizationScalingResult",
    "UniformBathComponent",
    "UniformNuclearLayer",
    "VoxelBathComponent",
    "VoxelNuclearSample",
]
