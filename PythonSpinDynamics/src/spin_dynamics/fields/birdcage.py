"""Prescribed-current reference model for cylindrical birdcage coils.

The reference model keeps every rung and end-ring section as a separate
branch.  It supplies the ideal sinusoidal current modes used to validate the
later circuit solver, including the degenerate linear modes and their complex
quadrature combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from spin_dynamics.fields.magnetostatics import biot_savart
from spin_dynamics.fields.validity import (
    QuasistaticAssessment,
    QuasistaticRegion,
    ValidityPolicy,
    apply_validity_policy,
    assess_quasistatic_validity,
)

Segment = tuple[np.ndarray, np.ndarray]

_AXIS_FRAMES = {
    "x": (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ),
    "y": (
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    ),
    "z": (
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
}


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BirdcageGeometry:
    """Cylindrical rungs and segmented end rings.

    Rung ``n`` lies at azimuth ``2*pi*n/n_rungs`` and is directed from the
    negative to the positive end of ``axis``.  Every end-ring section is
    directed toward increasing azimuth.  These orientations define the signs
    used by :class:`BirdcageCurrentMode`.
    """

    radius: float
    length: float
    n_rungs: int = 16
    center: Sequence[float] = (0.0, 0.0, 0.0)
    axis: str = "z"
    ring_segments_per_section: int = 4

    def __post_init__(self) -> None:
        radius = float(self.radius)
        length = float(self.length)
        n_rungs = int(self.n_rungs)
        ring_segments = int(self.ring_segments_per_section)
        center = np.array(self.center, dtype=np.float64, copy=True)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be finite and positive")
        if not np.isfinite(length) or length <= 0.0:
            raise ValueError("length must be finite and positive")
        if n_rungs < 4:
            raise ValueError("n_rungs must be at least 4")
        if ring_segments < 1:
            raise ValueError("ring_segments_per_section must be at least 1")
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite 3-vector")
        if self.axis not in _AXIS_FRAMES:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        center.setflags(write=False)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "n_rungs", n_rungs)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "ring_segments_per_section", ring_segments)

    @property
    def azimuth_rad(self) -> np.ndarray:
        """Rung azimuths in the transverse right-handed basis."""

        return 2.0 * np.pi * np.arange(self.n_rungs) / self.n_rungs

    @property
    def axis_vector(self) -> np.ndarray:
        """Unit vector along the cage axis."""

        return _AXIS_FRAMES[self.axis][0].copy()

    @property
    def transverse_basis(self) -> tuple[np.ndarray, np.ndarray]:
        """Right-handed transverse basis ``(e1, e2)`` with ``e1 x e2=axis``."""

        return (
            _AXIS_FRAMES[self.axis][1].copy(),
            _AXIS_FRAMES[self.axis][2].copy(),
        )

    def rung_segments(self) -> tuple[Segment, ...]:
        """Return one oriented straight segment per rung."""

        axis, e1, e2 = _AXIS_FRAMES[self.axis]
        result: list[Segment] = []
        for phi in self.azimuth_rad:
            radial = self.radius * (np.cos(phi) * e1 + np.sin(phi) * e2)
            start = self.center + radial - 0.5 * self.length * axis
            end = self.center + radial + 0.5 * self.length * axis
            result.append((start, end))
        return tuple(result)

    def end_ring_sections(self, side: int) -> tuple[tuple[Segment, ...], ...]:
        """Return oriented arc sections for the positive or negative end ring."""

        if side not in (-1, 1):
            raise ValueError("side must be -1 or +1")
        axis, e1, e2 = _AXIS_FRAMES[self.axis]
        axial = 0.5 * side * self.length * axis
        delta = 2.0 * np.pi / self.n_rungs
        result: list[tuple[Segment, ...]] = []
        for phi in self.azimuth_rad:
            angles = np.linspace(
                phi,
                phi + delta,
                self.ring_segments_per_section + 1,
            )
            points = (
                self.center[np.newaxis, :]
                + axial[np.newaxis, :]
                + self.radius
                * (
                    np.cos(angles)[:, np.newaxis] * e1[np.newaxis, :]
                    + np.sin(angles)[:, np.newaxis] * e2[np.newaxis, :]
                )
            )
            result.append(
                tuple(
                    (points[index], points[index + 1])
                    for index in range(points.shape[0] - 1)
                )
            )
        return tuple(result)

    @property
    def positive_end_ring_sections(self) -> tuple[tuple[Segment, ...], ...]:
        """Sections on the positive-axis end ring."""

        return self.end_ring_sections(1)

    @property
    def negative_end_ring_sections(self) -> tuple[tuple[Segment, ...], ...]:
        """Sections on the negative-axis end ring."""

        return self.end_ring_sections(-1)


@dataclass(frozen=True)
class BirdcageCurrentMode:
    """Complex branch-current phasors consistent with the geometry orientation."""

    rung_currents_a: np.ndarray
    positive_end_ring_currents_a: np.ndarray
    negative_end_ring_currents_a: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        rung = np.array(self.rung_currents_a, dtype=np.complex128, copy=True)
        positive = np.array(
            self.positive_end_ring_currents_a,
            dtype=np.complex128,
            copy=True,
        )
        negative = np.array(
            self.negative_end_ring_currents_a,
            dtype=np.complex128,
            copy=True,
        )
        if rung.ndim != 1 or rung.size < 4:
            raise ValueError("rung_currents_a must contain at least four values")
        if positive.shape != rung.shape or negative.shape != rung.shape:
            raise ValueError("end-ring current arrays must match rung_currents_a")
        if not (
            np.all(np.isfinite(rung))
            and np.all(np.isfinite(positive))
            and np.all(np.isfinite(negative))
        ):
            raise ValueError("branch currents must be finite")
        positive_residual = positive - np.roll(positive, 1) - rung
        negative_residual = negative - np.roll(negative, 1) + rung
        scale = max(
            float(np.max(np.abs(rung))),
            float(np.max(np.abs(positive))),
            float(np.max(np.abs(negative))),
            np.finfo(np.float64).tiny,
        )
        if max(
            float(np.max(np.abs(positive_residual))),
            float(np.max(np.abs(negative_residual))),
        ) > 1.0e-10 * scale:
            raise ValueError("branch currents must satisfy end-ring KCL")
        rung.setflags(write=False)
        positive.setflags(write=False)
        negative.setflags(write=False)
        object.__setattr__(self, "rung_currents_a", rung)
        object.__setattr__(self, "positive_end_ring_currents_a", positive)
        object.__setattr__(self, "negative_end_ring_currents_a", negative)
        object.__setattr__(self, "label", str(self.label))

    @property
    def n_rungs(self) -> int:
        """Number of rung and end-ring branches."""

        return int(self.rung_currents_a.size)

    @property
    def kcl_residual_a(self) -> np.ndarray:
        """KCL residuals at the positive then negative end-ring nodes."""

        positive = (
            self.positive_end_ring_currents_a
            - np.roll(self.positive_end_ring_currents_a, 1)
            - self.rung_currents_a
        )
        negative = (
            self.negative_end_ring_currents_a
            - np.roll(self.negative_end_ring_currents_a, 1)
            + self.rung_currents_a
        )
        return np.stack((positive, negative), axis=0)

    @property
    def max_kcl_residual_a(self) -> float:
        """Maximum absolute node-current residual."""

        return float(np.max(np.abs(self.kcl_residual_a)))


def birdcage_current_mode(
    rung_currents_a: Iterable[complex] | np.ndarray,
    *,
    label: str = "prescribed",
) -> BirdcageCurrentMode:
    """Complete zero-sum rung currents with minimum-norm end-ring currents."""

    rung = np.asarray(tuple(rung_currents_a), dtype=np.complex128)
    if rung.ndim != 1 or rung.size < 4 or not np.all(np.isfinite(rung)):
        raise ValueError(
            "rung_currents_a must be a finite one-dimensional array "
            "with at least four values"
        )
    scale = max(
        float(np.max(np.abs(rung))),
        np.finfo(np.float64).tiny,
    )
    if abs(np.sum(rung)) > 1.0e-12 * scale * rung.size:
        raise ValueError("rung_currents_a must sum to zero")
    positive = np.cumsum(rung)
    positive -= np.mean(positive)
    return BirdcageCurrentMode(
        rung_currents_a=rung,
        positive_end_ring_currents_a=positive,
        negative_end_ring_currents_a=-positive,
        label=label,
    )


def birdcage_linear_mode(
    geometry: BirdcageGeometry,
    *,
    mode_index: int = 1,
    azimuthal_phase_rad: float = 0.0,
    current_amplitude_a: complex = 1.0,
    label: str = "",
) -> BirdcageCurrentMode:
    """Return one real sinusoidal cage mode.

    The rung pattern is ``I*cos(mode_index*phi-azimuthal_phase_rad)``.
    Fundamental cosine and sine modes use phase zero and ``pi/2``.
    """

    mode = int(mode_index)
    phase = float(azimuthal_phase_rad)
    amplitude = _finite_complex(current_amplitude_a, "current_amplitude_a")
    if mode < 1 or mode >= geometry.n_rungs / 2:
        raise ValueError("mode_index must satisfy 1 <= mode_index < n_rungs/2")
    if not np.isfinite(phase):
        raise ValueError("azimuthal_phase_rad must be finite")
    rung = amplitude * np.cos(mode * geometry.azimuth_rad - phase)
    mode_label = label or f"linear m={mode}, phase={phase:.6g}"
    return birdcage_current_mode(rung, label=mode_label)


def birdcage_quadrature_mode(
    geometry: BirdcageGeometry,
    *,
    mode_index: int = 1,
    handedness: int = 1,
    current_amplitude_a: complex = 1.0,
    label: str = "",
) -> BirdcageCurrentMode:
    """Return equal-amplitude cosine/sine modes in temporal quadrature.

    ``handedness=+1`` uses rung phasors ``I*exp(-1j*m*phi)`` and produces the
    package's dominant B1+ component for a cage axis and B0 along the same
    positive direction.  ``handedness=-1`` produces the opposite rotation.
    """

    if handedness not in (-1, 1):
        raise ValueError("handedness must be +1 or -1")
    mode = int(mode_index)
    amplitude = _finite_complex(current_amplitude_a, "current_amplitude_a")
    if mode < 1 or mode >= geometry.n_rungs / 2:
        raise ValueError("mode_index must satisfy 1 <= mode_index < n_rungs/2")
    phase = mode * geometry.azimuth_rad
    rung = amplitude * (
        np.cos(phase) - 1.0j * handedness * np.sin(phase)
    )
    mode_label = label or f"quadrature m={mode}, handedness={handedness:+d}"
    return birdcage_current_mode(rung, label=mode_label)


def _branch_field(
    points: np.ndarray,
    segments: Sequence[Segment],
    current_a: complex,
) -> np.ndarray:
    if current_a == 0.0:
        return np.zeros(points.shape, dtype=np.complex128)
    return current_a * biot_savart(points, segments, current=1.0)


@dataclass(frozen=True)
class BirdcageFieldSolution:
    """Complex field and circular components for one prescribed current mode."""

    points_m: np.ndarray
    field_t: np.ndarray
    b1_plus_t: np.ndarray
    b1_minus_t: np.ndarray
    mode: BirdcageCurrentMode
    validity: QuasistaticAssessment | None = None


def solve_birdcage_field(
    geometry: BirdcageGeometry,
    mode: BirdcageCurrentMode,
    points_m: np.ndarray,
    *,
    b0_direction: Sequence[float] | np.ndarray | None = None,
    frequency_hz: float | None = None,
    validity_regions: Sequence[QuasistaticRegion] | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> BirdcageFieldSolution:
    """Evaluate one complex birdcage mode and its B1+/B1- components.

    The spatial field is quasistatic even when the mode contains complex RF
    current phasors. Supply the frequency and relevant material spans to screen
    propagation and loading assumptions. With the default warning policy,
    missing frequency or material context is reported instead of silently
    treating the field as valid. Pass an empty region sequence to explicitly
    assess an unloaded air/vacuum calculation.
    """

    if mode.n_rungs != geometry.n_rungs:
        raise ValueError("mode and geometry must have the same number of rungs")
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points_m must have shape (..., 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_m must be finite")
    validity = assess_quasistatic_validity(
        frequency_hz,
        coil_extent_m=max(2.0 * geometry.radius, geometry.length),
        regions=validity_regions,
    )
    apply_validity_policy(
        validity,
        solver_name="solve_birdcage_field",
        policy=validity_policy,
        stacklevel=3,
    )
    field = np.zeros(points.shape, dtype=np.complex128)
    for segments, current in zip(
        geometry.rung_segments(),
        mode.rung_currents_a,
    ):
        field += _branch_field(points, (segments,), current)
    for segments, current in zip(
        geometry.positive_end_ring_sections,
        mode.positive_end_ring_currents_a,
    ):
        field += _branch_field(points, segments, current)
    for segments, current in zip(
        geometry.negative_end_ring_sections,
        mode.negative_end_ring_currents_a,
    ):
        field += _branch_field(points, segments, current)

    direction = (
        geometry.axis_vector
        if b0_direction is None
        else np.asarray(b0_direction, dtype=np.float64)
    )
    if direction.shape == (3,):
        b0 = np.broadcast_to(direction, points.shape)
    elif direction.shape == points.shape:
        b0 = direction
    else:
        raise ValueError("b0_direction must be a 3-vector or match points_m")
    from spin_dynamics.motion import circular_b1_component

    if points.ndim == 1:
        b1_plus = circular_b1_component(
            b0[np.newaxis, :],
            field[np.newaxis, :],
            handedness=1,
        )[0]
        b1_minus = circular_b1_component(
            b0[np.newaxis, :],
            field[np.newaxis, :],
            handedness=-1,
        )[0]
    else:
        b1_plus = circular_b1_component(b0, field, handedness=1)
        b1_minus = circular_b1_component(b0, field, handedness=-1)
    return BirdcageFieldSolution(
        points_m=points,
        field_t=field,
        b1_plus_t=b1_plus,
        b1_minus_t=b1_minus,
        mode=mode,
        validity=validity,
    )


@dataclass(frozen=True)
class BirdcageFieldMetrics:
    """ROI field-uniformity and polarization diagnostics."""

    target_handedness: int
    mean_b1_t: float
    coefficient_of_variation: float
    peak_to_peak_nonuniformity: float
    circularity: float
    counter_rotating_ratio: float
    circularity_db: float
    transverse_fraction: float


def birdcage_field_metrics(
    solution: BirdcageFieldSolution,
    *,
    roi_mask: np.ndarray | None = None,
    target_handedness: int | None = None,
) -> BirdcageFieldMetrics:
    """Summarize B1 uniformity, circularity, and transverse field in an ROI."""

    spatial_shape = solution.field_t.shape[:-1]
    if roi_mask is None:
        mask = np.ones(spatial_shape, dtype=bool)
    else:
        mask = np.asarray(roi_mask, dtype=bool)
        if mask.shape != spatial_shape:
            raise ValueError("roi_mask must match the field spatial shape")
    if not np.any(mask):
        raise ValueError("roi_mask must select at least one point")
    plus = solution.b1_plus_t[mask]
    minus = solution.b1_minus_t[mask]
    plus_power = float(np.mean(np.abs(plus) ** 2))
    minus_power = float(np.mean(np.abs(minus) ** 2))
    if target_handedness is None:
        handedness = 1 if plus_power >= minus_power else -1
    else:
        handedness = int(target_handedness)
        if handedness not in (-1, 1):
            raise ValueError("target_handedness must be +1 or -1")
    desired = plus if handedness == 1 else minus
    counter = minus if handedness == 1 else plus
    desired_power = float(np.mean(np.abs(desired) ** 2))
    counter_power = float(np.mean(np.abs(counter) ** 2))
    total_circular_power = desired_power + counter_power
    if desired_power <= 0.0 or total_circular_power <= 0.0:
        raise ValueError("the selected ROI has no transverse birdcage field")
    magnitude = np.abs(desired)
    mean_b1 = float(np.mean(magnitude))
    coefficient = float(np.std(magnitude) / mean_b1)
    peak_to_peak = float((np.max(magnitude) - np.min(magnitude)) / mean_b1)
    circularity = float(
        (desired_power - counter_power) / total_circular_power
    )
    counter_ratio = float(np.sqrt(counter_power / desired_power))
    circularity_db = (
        float("inf")
        if counter_ratio == 0.0
        else float(-20.0 * np.log10(counter_ratio))
    )
    field = solution.field_t[mask]
    # In an orthonormal transverse basis,
    # 2*(|B1+|^2+|B1-|^2)=|B_e1|^2+|B_e2|^2.
    transverse_rms = np.sqrt(total_circular_power * 2.0)
    total_rms = float(np.sqrt(np.mean(np.sum(np.abs(field) ** 2, axis=-1))))
    transverse_fraction = (
        min(1.0, float(transverse_rms / total_rms))
        if total_rms > 0.0
        else 0.0
    )
    return BirdcageFieldMetrics(
        target_handedness=handedness,
        mean_b1_t=mean_b1,
        coefficient_of_variation=coefficient,
        peak_to_peak_nonuniformity=peak_to_peak,
        circularity=circularity,
        counter_rotating_ratio=counter_ratio,
        circularity_db=circularity_db,
        transverse_fraction=transverse_fraction,
    )


__all__ = [
    "BirdcageGeometry",
    "BirdcageCurrentMode",
    "BirdcageFieldSolution",
    "BirdcageFieldMetrics",
    "birdcage_current_mode",
    "birdcage_linear_mode",
    "birdcage_quadrature_mode",
    "solve_birdcage_field",
    "birdcage_field_metrics",
]
