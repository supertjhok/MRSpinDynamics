"""Susceptibility-induced internal field gradients for porous-media NMR.

In porous media the dominant source of static field inhomogeneity is usually not
the applied gradient but the *internal* gradient set up by the magnetic-
susceptibility contrast between the solid matrix and the pore fluid. These
internal gradients accelerate CPMG decay (diffusion in internal gradients) and
bias diffusion measurements, and they are what background-gradient-suppressing
sequences are designed to cancel.

This module generates the internal off-resonance field for the canonical
analytically tractable geometry that fits the package's two-dimensional motion
field maps: infinitely long cylinders whose axes are perpendicular to the map
plane, magnetized by a uniform applied field lying in the plane. Outside such a
cylinder the field perturbation along the applied field is the classic 2D dipole

    dB_parallel(rho, phi) = B0 * (delta_chi / 2) * (a / rho)**2 * cos(2 phi),

where ``a`` is the cylinder radius, ``rho`` the distance from the axis, and
``phi`` the angle of the position vector relative to the in-plane applied-field
direction. Contributions from several cylinders superpose in the dilute
(``|delta_chi| << 1``) limit. The resulting off-resonance map is returned in the
angular-frequency (rad/s) convention used by ``spin_dynamics.motion`` so it
drops straight into the walker pipeline, and a companion helper extracts the
internal-gradient field and its pore-space distribution in tesla per metre.

The model assumes a static, linear, dilute susceptibility perturbation and does
not include higher-order (``chi``-squared) corrections, conductive samples, or
demagnetizing-field self-consistency beyond the leading dipole term.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from spin_dynamics.motion import MotionFieldMaps2D, make_motion_field_maps_2d


__all__ = [
    "CylindricalInclusion",
    "SusceptibilityField",
    "InternalGradientDistribution",
    "susceptibility_offresonance_map",
    "make_susceptibility_field_maps",
    "internal_gradient_maps",
    "internal_gradient_distribution",
]

# Proton gyromagnetic ratio (rad/s/T); matches the value used by the examples.
PROTON_GAMMA = 2.675222005e8


@dataclass(frozen=True)
class CylindricalInclusion:
    """One infinitely long cylindrical inclusion perpendicular to the map plane.

    ``center_x`` and ``center_z`` locate the cylinder axis in the spatial map
    plane and ``radius`` is its radius (metres). ``susceptibility_difference``
    overrides the system-wide ``delta_chi`` for this inclusion when given; it is
    the inclusion susceptibility minus the surrounding pore-fluid susceptibility
    (dimensionless SI volume susceptibility).
    """

    center_x: float
    center_z: float
    radius: float
    susceptibility_difference: float | None = None

    def __post_init__(self) -> None:
        radius = float(self.radius)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if not (np.isfinite(self.center_x) and np.isfinite(self.center_z)):
            raise ValueError("inclusion center must be finite")
        object.__setattr__(self, "center_x", float(self.center_x))
        object.__setattr__(self, "center_z", float(self.center_z))
        object.__setattr__(self, "radius", radius)
        if self.susceptibility_difference is not None:
            chi = float(self.susceptibility_difference)
            if not np.isfinite(chi):
                raise ValueError("susceptibility_difference must be finite")
            object.__setattr__(self, "susceptibility_difference", chi)


@dataclass(frozen=True)
class SusceptibilityField:
    """Internal off-resonance field from a susceptibility-contrast geometry.

    ``offresonance_hz`` is the linear-frequency off-resonance map (Hz) and
    ``offresonance_rad`` the angular-frequency map (rad/s) consumed directly by
    the motion helpers. ``inclusion_mask`` is True on grid points that fall
    inside a solid inclusion (typically excluded from the mobile pore fluid).
    """

    x_axis: np.ndarray
    z_axis: np.ndarray
    offresonance_rad: np.ndarray
    inclusion_mask: np.ndarray
    b0_tesla: float
    gamma: float

    @property
    def offresonance_hz(self) -> np.ndarray:
        return self.offresonance_rad / (2.0 * np.pi)

    @property
    def pore_mask(self) -> np.ndarray:
        return ~self.inclusion_mask


@dataclass(frozen=True)
class InternalGradientDistribution:
    """Pore-space distribution of the internal-gradient magnitude (T/m)."""

    bin_edges: np.ndarray
    histogram: np.ndarray
    mean: float
    rms: float
    maximum: float
    gradient_magnitude: np.ndarray


def _axis(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 2:
        raise ValueError(f"{name} must have at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return array


def susceptibility_offresonance_map(
    x_axis: Iterable[float] | np.ndarray,
    z_axis: Iterable[float] | np.ndarray,
    inclusions: Iterable[CylindricalInclusion],
    *,
    b0_tesla: float,
    susceptibility_difference: float = 0.0,
    gamma: float = PROTON_GAMMA,
    b0_in_plane_angle: float = 0.0,
    interior_fill: str = "uniform",
) -> SusceptibilityField:
    """Return the internal off-resonance field for cylindrical inclusions.

    The applied field lies in the map plane at ``b0_in_plane_angle`` radians from
    the ``x`` axis. Outside each cylinder the parallel field perturbation is the
    2D dipole ``B0 (delta_chi / 2) (a / rho)**2 cos(2 phi)``; contributions from
    multiple cylinders superpose. ``interior_fill`` controls the value written
    inside an inclusion: ``"uniform"`` uses the leading uniform interior shift
    ``-B0 delta_chi / 2`` (transverse-cylinder demagnetizing factor 1/2),
    ``"zero"`` writes zero, and ``"nan"`` marks the interior as undefined. The
    interior rarely matters because solid grains usually carry no mobile signal;
    use ``inclusion_mask`` to exclude them.
    """

    x = _axis(x_axis, "x_axis")
    z = _axis(z_axis, "z_axis")
    inclusion_list = list(inclusions)
    b0 = float(b0_tesla)
    if not np.isfinite(b0):
        raise ValueError("b0_tesla must be finite")
    gamma = float(gamma)
    if not np.isfinite(gamma) or gamma == 0.0:
        raise ValueError("gamma must be finite and non-zero")
    if interior_fill not in {"uniform", "zero", "nan"}:
        raise ValueError("interior_fill must be 'uniform', 'zero', or 'nan'")

    xx, zz = np.meshgrid(x, z, indexing="ij")
    delta_b = np.zeros(xx.shape, dtype=np.float64)
    mask = np.zeros(xx.shape, dtype=bool)
    cos_a = float(np.cos(b0_in_plane_angle))
    sin_a = float(np.sin(b0_in_plane_angle))

    for inclusion in inclusion_list:
        chi = inclusion.susceptibility_difference
        if chi is None:
            chi = float(susceptibility_difference)
        dx = xx - inclusion.center_x
        dz = zz - inclusion.center_z
        rho2 = dx * dx + dz * dz
        inside = rho2 <= inclusion.radius**2
        mask |= inside
        outside = ~inside
        safe_rho2 = np.where(outside, rho2, 1.0)
        # angle of the position vector measured from the in-plane B0 direction
        proj_par = dx * cos_a + dz * sin_a
        proj_perp = -dx * sin_a + dz * cos_a
        cos_2phi = (proj_par**2 - proj_perp**2) / safe_rho2
        contribution = b0 * 0.5 * chi * (inclusion.radius**2 / safe_rho2) * cos_2phi
        delta_b += np.where(outside, contribution, 0.0)

    # interior fill, applied after superposition so overlaps stay masked
    for inclusion in inclusion_list:
        chi = inclusion.susceptibility_difference
        if chi is None:
            chi = float(susceptibility_difference)
        dx = xx - inclusion.center_x
        dz = zz - inclusion.center_z
        inside = (dx * dx + dz * dz) <= inclusion.radius**2
        if interior_fill == "uniform":
            delta_b = np.where(inside, -b0 * 0.5 * chi, delta_b)
        elif interior_fill == "zero":
            delta_b = np.where(inside, 0.0, delta_b)
        else:
            delta_b = np.where(inside, np.nan, delta_b)

    offresonance_rad = gamma * delta_b
    return SusceptibilityField(
        x_axis=x,
        z_axis=z,
        offresonance_rad=offresonance_rad,
        inclusion_mask=mask,
        b0_tesla=b0,
        gamma=gamma,
    )


def make_susceptibility_field_maps(
    field: SusceptibilityField,
    *,
    b1_tx_map: Iterable[float] | np.ndarray | None = None,
    b1_rx_map: Iterable[float] | np.ndarray | None = None,
) -> MotionFieldMaps2D:
    """Wrap a ``SusceptibilityField`` as motion field maps.

    The angular off-resonance map becomes the ``b0_map`` consumed by
    ``spin_dynamics.motion`` and ``spin_dynamics.sequences.motion``. Inclusion
    interiors filled with NaN are not allowed here because the motion maps must
    be finite; fill with ``"uniform"`` or ``"zero"`` instead.
    """

    if not np.all(np.isfinite(field.offresonance_rad)):
        raise ValueError(
            "offresonance map contains non-finite values; rebuild with "
            "interior_fill='uniform' or 'zero' before making motion maps"
        )
    return make_motion_field_maps_2d(
        field.x_axis,
        field.z_axis,
        b0_map=field.offresonance_rad,
        b1_tx_map=b1_tx_map,
        b1_rx_map=b1_rx_map,
    )


def internal_gradient_maps(
    field: SusceptibilityField,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(g_x, g_z, g_magnitude)`` internal-gradient maps in tesla/metre.

    The gradient is the spatial derivative of the parallel field perturbation
    ``dB = offresonance_rad / gamma`` taken over the map axes. Inclusion
    interiors filled with NaN propagate as NaN.
    """

    delta_b = field.offresonance_rad / field.gamma
    g_x, g_z = np.gradient(delta_b, field.x_axis, field.z_axis, edge_order=1)
    g_mag = np.sqrt(g_x**2 + g_z**2)
    return g_x, g_z, g_mag


def internal_gradient_distribution(
    field: SusceptibilityField,
    *,
    weights: Iterable[float] | np.ndarray | None = None,
    restrict_to_pore_space: bool = True,
    bins: int = 64,
    range_max: float | None = None,
) -> InternalGradientDistribution:
    """Summarize the pore-space internal-gradient magnitude (T/m).

    By default only pore-fluid grid points (outside inclusions) contribute, and
    points adjacent to an inclusion boundary are dropped because the discrete
    gradient straddles the susceptibility jump there. ``weights`` optionally
    supplies a spin-density weighting on the full grid.
    """

    _, _, g_mag = internal_gradient_maps(field)
    valid = np.isfinite(g_mag)
    if restrict_to_pore_space:
        pore = field.pore_mask
        # exclude pore points neighbouring an inclusion: the discrete gradient
        # across the boundary is not the physical pore-fluid gradient
        boundary_adjacent = _dilate(field.inclusion_mask)
        valid &= pore & ~boundary_adjacent
    if weights is None:
        weight_map = np.ones_like(g_mag, dtype=np.float64)
    else:
        weight_map = np.asarray(weights, dtype=np.float64)
        if weight_map.shape != g_mag.shape:
            raise ValueError("weights must match the field map shape")
        if np.any(weight_map < 0.0):
            raise ValueError("weights must be non-negative")
    values = g_mag[valid]
    sample_weights = weight_map[valid]
    total = float(sample_weights.sum())
    if values.size == 0 or total <= 0.0:
        raise ValueError("no valid pore-space gradient samples to summarize")

    mean = float(np.average(values, weights=sample_weights))
    rms = float(np.sqrt(np.average(values**2, weights=sample_weights)))
    maximum = float(values.max())
    upper = float(range_max) if range_max is not None else maximum
    if upper <= 0.0:
        upper = max(maximum, np.finfo(float).eps)
    histogram, bin_edges = np.histogram(
        values, bins=int(bins), range=(0.0, upper), weights=sample_weights
    )
    return InternalGradientDistribution(
        bin_edges=bin_edges,
        histogram=histogram,
        mean=mean,
        rms=rms,
        maximum=maximum,
        gradient_magnitude=g_mag,
    )


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Return a one-cell 4-neighbour dilation of a boolean mask."""

    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    return out
