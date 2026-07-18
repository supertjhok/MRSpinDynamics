"""Surface, target-spin, and point-dipole geometry for nano-MR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nano_mr.frames import unit_vector
from spin_dynamics.nano_mr.sensors import DefectSpinSensor
from spin_dynamics.nqr.operators import validate_spin


MU0_OVER_4PI = 1.0e-7
"""Vacuum permeability divided by ``4*pi`` in SI units."""

BOHR_MAGNETON_J_PER_T = 9.2740100783e-24
"""Bohr magneton in joules per tesla."""

ISOTOPE_GAMMA_HZ_PER_T = {
    "1H": 42.57747892e6,
    "13C": 10.7084e6,
    "19F": 40.052e6,
    "29Si": -8.465e6,
    "31P": 17.235e6,
}
"""Convenience gyromagnetic ratios for common nano-MR target nuclei."""


def _point_nm(values, *, name: str) -> np.ndarray:
    point = np.asarray(values, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite")
    return point


@dataclass(frozen=True, eq=False)
class SurfaceGeometry:
    """Planar sample interface.

    ``normal_lab`` points from the host material into the sample.  A defect at
    positive depth is therefore located at ``point_lab_nm - depth*normal_lab``.
    """

    point_lab_nm: np.ndarray
    normal_lab: np.ndarray
    label: str = "surface"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "point_lab_nm", _point_nm(self.point_lab_nm, name="point_lab_nm")
        )
        object.__setattr__(
            self,
            "normal_lab",
            unit_vector(self.normal_lab, name="normal_lab"),
        )
        object.__setattr__(self, "label", str(self.label))

    def sensor_position_lab_nm(self, sensor: DefectSpinSensor) -> np.ndarray:
        """Return the defect location implied by its depth below this surface."""

        return self.point_lab_nm - sensor.depth_nm * self.normal_lab

    def signed_distance_nm(self, positions_lab_nm) -> np.ndarray:
        """Return signed distances from the plane, positive into the sample."""

        positions = np.asarray(positions_lab_nm, dtype=np.float64)
        if positions.shape[-1] != 3 or not np.all(np.isfinite(positions)):
            raise ValueError("positions_lab_nm must have final dimension 3 and be finite")
        return (positions - self.point_lab_nm) @ self.normal_lab


@dataclass(frozen=True, eq=False)
class NuclearSpin:
    """One target nucleus at a laboratory-frame position."""

    isotope: str
    position_lab_nm: np.ndarray
    gamma_hz_per_t: float
    spin: float = 0.5
    label: str = ""

    def __post_init__(self) -> None:
        gamma = float(self.gamma_hz_per_t)
        if not np.isfinite(gamma) or gamma == 0.0:
            raise ValueError("gamma_hz_per_t must be finite and non-zero")
        object.__setattr__(self, "spin", validate_spin(self.spin))
        object.__setattr__(
            self,
            "position_lab_nm",
            _point_nm(self.position_lab_nm, name="position_lab_nm"),
        )
        object.__setattr__(self, "gamma_hz_per_t", gamma)
        object.__setattr__(self, "isotope", str(self.isotope))
        object.__setattr__(
            self,
            "label",
            str(self.label) if self.label else str(self.isotope),
        )

    @classmethod
    def from_isotope(
        cls,
        isotope: str,
        position_lab_nm,
        *,
        spin: float = 0.5,
        label: str = "",
    ) -> "NuclearSpin":
        """Construct a target using a built-in gyromagnetic-ratio value."""

        try:
            gamma = ISOTOPE_GAMMA_HZ_PER_T[str(isotope)]
        except KeyError as exc:
            raise ValueError(f"unknown isotope preset: {isotope!r}") from exc
        return cls(
            isotope=str(isotope),
            position_lab_nm=position_lab_nm,
            gamma_hz_per_t=gamma,
            spin=spin,
            label=label,
        )


def dipole_spatial_tensor_inverse_m3(displacement_lab_nm) -> np.ndarray:
    """Return ``(I - 3 rhat rhat^T) / r^3`` in inverse cubic metres."""

    displacement_nm = _point_nm(
        displacement_lab_nm, name="displacement_lab_nm"
    )
    distance_nm = float(np.linalg.norm(displacement_nm))
    if distance_nm <= 0.0:
        raise ValueError("dipole displacement must be non-zero")
    direction = displacement_nm / distance_nm
    distance_m = distance_nm * 1.0e-9
    return (np.eye(3) - 3.0 * np.outer(direction, direction)) / distance_m**3


def point_dipolar_hyperfine_tensor_hz(
    sensor: DefectSpinSensor,
    target: NuclearSpin,
    *,
    sensor_position_lab_nm,
) -> np.ndarray:
    """Return the point-dipole electron-nuclear tensor in hertz.

    The tensor ``A`` is defined by ``H/h = S_local^T A I_lab``.  Its rows use
    defect-local electron-spin components and its columns use laboratory-frame
    nuclear-spin components.  The nuclear gyromagnetic-ratio sign is retained.
    Contact hyperfine coupling is not included.
    """

    sensor_position = _point_nm(
        sensor_position_lab_nm, name="sensor_position_lab_nm"
    )
    displacement = target.position_lab_nm - sensor_position
    spatial = dipole_spatial_tensor_inverse_m3(displacement)
    rotation = sensor.frame.rotation_lab_from_local
    prefactor = -MU0_OVER_4PI * BOHR_MAGNETON_J_PER_T * target.gamma_hz_per_t
    return prefactor * sensor.g_tensor.T @ rotation.T @ spatial


__all__ = [
    "BOHR_MAGNETON_J_PER_T",
    "ISOTOPE_GAMMA_HZ_PER_T",
    "MU0_OVER_4PI",
    "NuclearSpin",
    "SurfaceGeometry",
    "dipole_spatial_tensor_inverse_m3",
    "point_dipolar_hyperfine_tensor_hz",
]
