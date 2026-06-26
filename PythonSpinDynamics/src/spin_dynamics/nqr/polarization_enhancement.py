"""Adiabatic polarization-transfer model for pre-polarized NQR.

This module implements the instrument-level model used for automated
polarization-enhanced NQR: protons are polarized in a permanent magnet, the
sample is moved through a falling fringe field, and cross-polarization to
quadrupolar nuclei can occur when ``gamma_H * B0`` crosses an NQR transition
frequency. The transfer is treated by an adjustable adiabatic-efficiency factor,
not by microscopic propagation through each avoided crossing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields.magnetostatics import (
    FiniteMagnetRod,
    finite_magnet_array_b0,
    halbach_dipole_magnets,
)

PROTON_GAMMA_HZ_PER_T = 42.57747892e6


@dataclass(frozen=True)
class PolarizationEnhancedNQRSample:
    """Sample and spin parameters for polarization-enhanced NQR."""

    line_frequencies_hz: tuple[float, ...]
    line_labels: tuple[str, ...] = ("+", "-", "0")
    protons_per_molecule: float = 1.0
    nitrogens_per_molecule: float = 1.0
    proton_t1_seconds: float = np.inf
    nitrogen_t1_seconds: float = np.inf
    proton_linewidth_hz: float = 25.0e3
    proton_nitrogen_coupling_hz: float = 300.0
    name: str = "sample"

    def __post_init__(self) -> None:
        freqs = tuple(float(item) for item in self.line_frequencies_hz)
        labels = tuple(str(item) for item in self.line_labels)
        if not freqs:
            raise ValueError("line_frequencies_hz must not be empty")
        if len(labels) != len(freqs):
            raise ValueError("line_labels must match line_frequencies_hz")
        if any((not np.isfinite(freq)) or freq <= 0.0 for freq in freqs):
            raise ValueError("line frequencies must be positive and finite")
        for name in ("protons_per_molecule", "nitrogens_per_molecule"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in (
            "proton_t1_seconds",
            "nitrogen_t1_seconds",
            "proton_linewidth_hz",
            "proton_nitrogen_coupling_hz",
        ):
            value = float(getattr(self, name))
            if value <= 0.0 or np.isnan(value):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "line_frequencies_hz", freqs)
        object.__setattr__(self, "line_labels", labels)
        object.__setattr__(self, "name", str(self.name))


@dataclass(frozen=True)
class CylindricalSampleGeometry:
    """Cylindrical sample volume moved through the magnet."""

    length: float = 20.0e-3
    diameter: float = 8.0e-3
    axial_points: int = 9
    radial_rings: int = 1
    azimuthal_points: int = 8

    def __post_init__(self) -> None:
        if self.length <= 0.0 or not np.isfinite(self.length):
            raise ValueError("length must be positive and finite")
        if self.diameter <= 0.0 or not np.isfinite(self.diameter):
            raise ValueError("diameter must be positive and finite")
        if int(self.axial_points) < 1:
            raise ValueError("axial_points must be at least 1")
        if int(self.radial_rings) < 0:
            raise ValueError("radial_rings must be non-negative")
        if int(self.azimuthal_points) < 1:
            raise ValueError("azimuthal_points must be at least 1")
        object.__setattr__(self, "axial_points", int(self.axial_points))
        object.__setattr__(self, "radial_rings", int(self.radial_rings))
        object.__setattr__(self, "azimuthal_points", int(self.azimuthal_points))

    def sample_offsets_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """Return lab-frame offsets and volume weights for the sample."""

        z_offsets = (
            (np.arange(self.axial_points, dtype=np.float64) + 0.5)
            / self.axial_points
            - 0.5
        ) * self.length
        offsets: list[tuple[float, float, float]] = []
        weights: list[float] = []
        radius = 0.5 * self.diameter

        # Ring samples represent equal-radius annuli. The innermost disk is a
        # single center point; outer annuli are split uniformly in azimuth.
        for z in z_offsets:
            if self.radial_rings == 0:
                offsets.append((0.0, 0.0, float(z)))
                weights.append(1.0)
                continue
            for ring in range(self.radial_rings):
                r_inner = radius * ring / self.radial_rings
                r_outer = radius * (ring + 1) / self.radial_rings
                r = np.sqrt(0.5 * (r_inner**2 + r_outer**2))
                annulus_weight = max(r_outer**2 - r_inner**2, 0.0)
                if ring == 0:
                    offsets.append((0.0, 0.0, float(z)))
                    weights.append(float(annulus_weight))
                    continue
                for phi in np.linspace(0.0, 2.0 * np.pi, self.azimuthal_points,
                                       endpoint=False):
                    offsets.append(
                        (float(r * np.cos(phi)), float(r * np.sin(phi)), float(z))
                    )
                    weights.append(float(annulus_weight / self.azimuthal_points))

        weights_array = np.asarray(weights, dtype=np.float64)
        weights_array = weights_array / np.sum(weights_array)
        return np.asarray(offsets, dtype=np.float64), weights_array


@dataclass(frozen=True)
class LinearTransportMotion:
    """Constant-velocity sample motion along one lab axis."""

    start_position: float
    stop_position: float
    velocity: float = 0.1667
    axis: Literal["x", "y", "z"] = "z"
    transverse_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if self.start_position == self.stop_position:
            raise ValueError("start_position and stop_position must differ")
        if self.velocity <= 0.0 or not np.isfinite(self.velocity):
            raise ValueError("velocity must be positive and finite")
        if self.axis not in ("x", "y", "z"):
            raise ValueError("axis must be 'x', 'y', or 'z'")
        center = tuple(float(item) for item in self.transverse_center)
        if len(center) != 3 or any(not np.isfinite(item) for item in center):
            raise ValueError("transverse_center must contain three finite values")
        object.__setattr__(self, "transverse_center", center)

    @property
    def direction(self) -> np.ndarray:
        out = np.zeros(3, dtype=np.float64)
        out[{"x": 0, "y": 1, "z": 2}[self.axis]] = 1.0
        return out

    @property
    def travel_distance(self) -> float:
        return abs(float(self.stop_position) - float(self.start_position))

    @property
    def travel_time(self) -> float:
        return self.travel_distance / float(self.velocity)

    def path_points(self, coordinate: np.ndarray, offset: np.ndarray) -> np.ndarray:
        center = np.asarray(self.transverse_center, dtype=np.float64)
        return center + coordinate[:, np.newaxis] * self.direction + offset


@dataclass(frozen=True)
class HalbachPrepolarizationMagnet:
    """Finite four-rod Halbach magnet for transport simulations."""

    center_radius: float = 25.4e-3
    length: float = 101.6e-3
    remanence: float = 1.40
    rod_shape: Literal["cylinder", "square"] = "square"
    rod_radius: float | None = 12.7e-3
    rod_width: float | None = 25.4e-3
    field_angle: float = 0.0
    n_cross: int = 5
    n_length: int = 31

    def rods(self) -> tuple[FiniteMagnetRod, ...]:
        return halbach_dipole_magnets(
            center_radius=self.center_radius,
            length=self.length,
            remanence=self.remanence,
            rod_shape=self.rod_shape,
            rod_radius=self.rod_radius,
            rod_width=self.rod_width,
            field_angle=self.field_angle,
        )

    def b0_vector(self, points: np.ndarray) -> np.ndarray:
        return finite_magnet_array_b0(
            points,
            self.rods(),
            n_cross=self.n_cross,
            n_length=self.n_length,
        )

    def b0_magnitude(self, points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.b0_vector(points), axis=-1)


@dataclass(frozen=True)
class PolarizationTransferResult:
    """Result of a polarization-enhanced NQR transport simulation."""

    line_labels: tuple[str, ...]
    line_frequencies_hz: np.ndarray
    ideal_enhancement: np.ndarray
    practical_enhancement: np.ndarray
    crossing_positions: np.ndarray
    crossing_fields_tesla: np.ndarray
    gradients_t_per_m: np.ndarray
    crossing_times: np.ndarray
    adiabatic_ratios: np.ndarray
    transfer_efficiency: np.ndarray
    proton_polarization_factor: float
    travel_time_seconds: float
    b0_profile_positions: np.ndarray
    b0_profile_tesla: np.ndarray
    b0_profile_gradient_t_per_m: np.ndarray


def ideal_spin1_enhancement_factors(
    line_frequencies_hz: Sequence[float],
    *,
    max_b0_tesla: float,
    protons_per_molecule: float,
    nitrogens_per_molecule: float,
    gamma_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
) -> np.ndarray:
    """Return ideal spin-1 enhancement factors from the Glickstein model.

    For three lines, the input order is ``(nu_plus, nu_minus, nu_zero)``. For
    other line counts, each line is treated as an independent ideal
    cross-polarization event using the single-crossing expression.
    """

    freqs = np.asarray(line_frequencies_hz, dtype=np.float64)
    if freqs.ndim != 1 or freqs.size == 0:
        raise ValueError("line_frequencies_hz must be a non-empty vector")
    if max_b0_tesla < 0.0 or not np.isfinite(max_b0_tesla):
        raise ValueError("max_b0_tesla must be non-negative and finite")
    nu_h = float(gamma_hz_per_t) * float(max_b0_tesla)
    rho = (2.0 * nitrogens_per_molecule / 3.0) / (
        protons_per_molecule + 2.0 * nitrogens_per_molecule / 3.0
    )
    if freqs.size == 3:
        nu_p, nu_m, nu_0 = freqs
        alpha = (nu_h - nu_p) * (1.0 - rho)
        beta = 0.5 * (nu_h - nu_p) * (1.0 - rho) ** 2 + (1.0 - rho) * nu_0
        gamma = (
            0.75 * (nu_h - nu_p) * (1.0 - rho) ** 3
            + (1.0 - rho) * (nu_m - nu_0)
            + 1.5 * (1.0 - rho) ** 2 * nu_0
        )
        out = np.array(
            [
                1.0 + (alpha + 0.5 * beta + 0.5 * gamma) / nu_p,
                1.0 + (0.5 * alpha + beta - 0.5 * gamma) / nu_m,
                1.0 + (0.5 * alpha - 0.5 * beta + gamma) / nu_0,
            ],
            dtype=np.float64,
        )
    else:
        out = (1.0 - rho) * nu_h / freqs + rho
    return np.where(nu_h >= freqs, np.maximum(out, 1.0), 1.0)


def simulate_adiabatic_polarization_transfer(
    magnet: HalbachPrepolarizationMagnet | Callable[[np.ndarray], np.ndarray],
    sample: PolarizationEnhancedNQRSample,
    sample_geometry: CylindricalSampleGeometry,
    motion: LinearTransportMotion,
    *,
    prepolarization_time_seconds: float = np.inf,
    path_points: int = 501,
    gamma_hz_per_t: float = PROTON_GAMMA_HZ_PER_T,
    adiabatic_scale: float = 1.0,
) -> PolarizationTransferResult:
    """Simulate polarization transfer while a sample moves through a gradient.

    ``magnet`` may be a :class:`HalbachPrepolarizationMagnet` or any callable
    accepting ``(..., 3)`` points and returning either B0 vectors or B0
    magnitudes. ``practical_enhancement`` is the ideal enhancement reduced by
    proton build-up, crossing efficiency, finite sample size, and nitrogen T1
    decay between each crossing and the end of the transport path.
    """

    if int(path_points) < 3:
        raise ValueError("path_points must be at least 3")
    if adiabatic_scale <= 0.0 or not np.isfinite(adiabatic_scale):
        raise ValueError("adiabatic_scale must be positive and finite")

    if isinstance(magnet, HalbachPrepolarizationMagnet):
        b0_sampler = magnet.b0_magnitude
    else:
        b0_sampler = _as_b0_magnitude_sampler(magnet)

    coordinate = np.linspace(
        motion.start_position,
        motion.stop_position,
        int(path_points),
    )
    offsets, weights = sample_geometry.sample_offsets_and_weights()
    freqs = np.asarray(sample.line_frequencies_hz, dtype=np.float64)
    n_lines = freqs.size
    n_offsets = offsets.shape[0]

    crossing_positions = np.full((n_offsets, n_lines), np.nan, dtype=np.float64)
    gradients = np.full_like(crossing_positions, np.nan)
    crossing_times = np.full_like(crossing_positions, np.nan)
    ratios = np.full_like(crossing_positions, np.inf)
    efficiencies = np.zeros_like(crossing_positions)

    b0_profiles = []
    gradient_profiles = []
    for offset_index, offset in enumerate(offsets):
        points = motion.path_points(coordinate, offset)
        b0 = np.asarray(b0_sampler(points), dtype=np.float64).reshape(coordinate.shape)
        grad = np.gradient(b0, coordinate)
        b0_profiles.append(b0)
        gradient_profiles.append(grad)
        for line_index, frequency in enumerate(freqs):
            target_b0 = frequency / gamma_hz_per_t
            crossing = _first_crossing(coordinate, b0, target_b0)
            if crossing is None:
                continue
            crossing_positions[offset_index, line_index] = crossing
            local_grad = abs(float(np.interp(crossing, coordinate, grad)))
            gradients[offset_index, line_index] = local_grad
            distance = abs(crossing - motion.start_position)
            crossing_times[offset_index, line_index] = distance / motion.velocity
            if local_grad > 0.0:
                ratio = (
                    gamma_hz_per_t
                    * motion.velocity
                    * local_grad
                    / (sample.proton_linewidth_hz * sample.proton_nitrogen_coupling_hz)
                )
                ratios[offset_index, line_index] = ratio
                efficiencies[offset_index, line_index] = 1.0 - np.exp(
                    -adiabatic_scale / ratio
                )

    weighted_b0 = np.average(np.asarray(b0_profiles), axis=0, weights=weights)
    weighted_grad = np.average(np.asarray(gradient_profiles), axis=0, weights=weights)
    max_b0 = float(np.max(weighted_b0))
    ideal = ideal_spin1_enhancement_factors(
        freqs,
        max_b0_tesla=max_b0,
        protons_per_molecule=sample.protons_per_molecule,
        nitrogens_per_molecule=sample.nitrogens_per_molecule,
        gamma_hz_per_t=gamma_hz_per_t,
    )

    proton_factor = _relaxation_build_up(prepolarization_time_seconds,
                                         sample.proton_t1_seconds)
    retention = np.exp(
        -np.maximum(motion.travel_time - crossing_times, 0.0)
        / sample.nitrogen_t1_seconds
    )
    transfer_weight = np.nan_to_num(efficiencies * retention, nan=0.0)
    line_transfer = np.average(transfer_weight, axis=0, weights=weights)
    practical = 1.0 + (ideal - 1.0) * proton_factor * line_transfer

    return PolarizationTransferResult(
        line_labels=sample.line_labels,
        line_frequencies_hz=freqs,
        ideal_enhancement=ideal,
        practical_enhancement=practical,
        crossing_positions=np.array(
            [
                _weighted_nanmean(crossing_positions[:, i], weights)
                for i in range(n_lines)
            ]
        ),
        crossing_fields_tesla=freqs / gamma_hz_per_t,
        gradients_t_per_m=np.array(
            [_weighted_nanmean(gradients[:, i], weights) for i in range(n_lines)]
        ),
        crossing_times=np.array(
            [_weighted_nanmean(crossing_times[:, i], weights) for i in range(n_lines)]
        ),
        adiabatic_ratios=np.array(
            [_weighted_nanmean(ratios[:, i], weights) for i in range(n_lines)]
        ),
        transfer_efficiency=np.array(
            [_weighted_nanmean(efficiencies[:, i], weights) for i in range(n_lines)]
        ),
        proton_polarization_factor=proton_factor,
        travel_time_seconds=motion.travel_time,
        b0_profile_positions=coordinate,
        b0_profile_tesla=weighted_b0,
        b0_profile_gradient_t_per_m=weighted_grad,
    )


def _as_b0_magnitude_sampler(
    sampler: Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:
    def wrapped(points: np.ndarray) -> np.ndarray:
        values = np.asarray(sampler(points), dtype=np.float64)
        if values.shape[-1:] == (3,):
            return np.linalg.norm(values, axis=-1)
        return values

    return wrapped


def _first_crossing(
    coordinate: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float | None:
    delta = values - target
    finite = np.isfinite(delta)
    for i in range(delta.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if delta[i] == 0.0:
            return float(coordinate[i])
        if delta[i] * delta[i + 1] <= 0.0:
            span = delta[i + 1] - delta[i]
            if span == 0.0:
                return float(coordinate[i])
            frac = -delta[i] / span
            return float(coordinate[i] + frac * (coordinate[i + 1] - coordinate[i]))
    return None


def _relaxation_build_up(duration: float, t1: float) -> float:
    if np.isinf(duration):
        return 1.0
    if duration <= 0.0:
        return 0.0
    if np.isinf(t1):
        return 0.0
    return float(1.0 - np.exp(-duration / t1))


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return float("nan")
    return float(np.average(values[finite], weights=weights[finite]))
