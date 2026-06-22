"""Static disorder and strain helpers for ESR spectra."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from spin_dynamics.esr.hamiltonians import (
    resonance_field_tesla,
    resonance_frequency_hz,
)
from spin_dynamics.esr.lineshapes import spectrum_from_lines
from spin_dynamics.esr.spectra import (
    ESRLine,
    OrientationInput,
    simulate_field_sweep,
    simulate_frequency_spectrum,
)
from spin_dynamics.esr.systems import ESRSpinSystem


@dataclass(frozen=True)
class ESRDistributionSample:
    """One weighted ESR static-disorder sample."""

    system: ESRSpinSystem
    weight: float = 1.0
    field_offset_tesla: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        weight = float(self.weight)
        field_offset = float(self.field_offset_tesla)
        if weight < 0 or not np.isfinite(weight):
            raise ValueError("weight must be non-negative and finite")
        if not np.isfinite(field_offset):
            raise ValueError("field_offset_tesla must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "field_offset_tesla", field_offset)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class ESRFieldDistributionResult:
    """Field-swept ESR spectrum averaged over static disorder samples."""

    fields_tesla: np.ndarray
    spectrum: np.ndarray
    lines: tuple[ESRLine, ...]
    samples: tuple[ESRDistributionSample, ...]
    microwave_frequency_hz: float
    broadening_tesla: float
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"


@dataclass(frozen=True)
class ESRFrequencyDistributionResult:
    """Frequency-swept ESR spectrum averaged over static disorder samples."""

    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    lines: tuple[ESRLine, ...]
    samples: tuple[ESRDistributionSample, ...]
    b0_tesla: float
    broadening_hz: float
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"


def normalize_distribution(
    samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...],
) -> tuple[ESRDistributionSample, ...]:
    """Return static-disorder samples with weights normalized to unity."""

    samples = tuple(samples)
    if not samples:
        raise ValueError("at least one distribution sample is required")
    total = sum(sample.weight for sample in samples)
    if total <= 0:
        raise ValueError("distribution weights must have positive sum")
    return tuple(
        ESRDistributionSample(
            system=sample.system,
            weight=sample.weight / total,
            field_offset_tesla=sample.field_offset_tesla,
            label=sample.label,
        )
        for sample in samples
    )


def static_disorder_grid(
    system: ESRSpinSystem,
    *,
    g_std: float | np.ndarray | list[float] | tuple[float, ...] = 0.0,
    field_std_tesla: float = 0.0,
    g_points: int = 3,
    field_points: int = 5,
    n_sigma: float = 2.0,
) -> tuple[ESRDistributionSample, ...]:
    """Return weighted samples for diagonal ``g`` strain and field offsets."""

    component_grids = _component_grids(g_std, g_points, n_sigma)
    g_offsets = tuple(item[0] for item in component_grids)
    g_weights = tuple(item[1] for item in component_grids)
    field_offsets, field_weights = _scalar_grid(field_std_tesla, field_points, n_sigma)
    base_g = np.asarray(system.g_tensor, dtype=np.float64)

    samples: list[ESRDistributionSample] = []
    for gx_idx, gy_idx, gz_idx in product(*(range(len(axis)) for axis in g_offsets)):
        g_delta = np.array(
            [
                g_offsets[0][gx_idx],
                g_offsets[1][gy_idx],
                g_offsets[2][gz_idx],
            ],
            dtype=np.float64,
        )
        g_weight = (
            g_weights[0][gx_idx]
            * g_weights[1][gy_idx]
            * g_weights[2][gz_idx]
        )
        g_tensor = base_g + np.diag(g_delta)
        for field_offset, field_weight in zip(field_offsets, field_weights):
            samples.append(
                ESRDistributionSample(
                    system=ESRSpinSystem(
                        g_tensor=g_tensor,
                        spin=system.spin,
                        label=system.label,
                    ),
                    weight=float(g_weight * field_weight),
                    field_offset_tesla=float(field_offset),
                    label="static_disorder",
                )
            )
    return normalize_distribution(samples)


def simulate_field_sweep_distribution(
    samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...],
    microwave_frequency_hz: float,
    *,
    orientations: OrientationInput = "single",
    broadening_tesla: float = 1.0e-4,
    points: int = 1024,
    span_tesla: float | None = None,
    fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
) -> ESRFieldDistributionResult:
    """Return a field-swept ESR spectrum averaged over static disorder."""

    samples = normalize_distribution(samples)
    frequency = float(microwave_frequency_hz)
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError("microwave_frequency_hz must be positive and finite")
    broadening = _positive_finite(broadening_tesla, "broadening_tesla")

    lines: list[ESRLine] = []
    for sample in samples:
        result = simulate_field_sweep(
            sample.system,
            frequency,
            orientations=orientations,
            broadening_tesla=broadening,
            points=2,
            lineshape=lineshape,
            detection_mode=detection_mode,
        )
        for line in result.lines:
            applied_field = line.field_tesla - sample.field_offset_tesla
            if applied_field < 0:
                continue
            lines.append(
                ESRLine(
                    orientation_index=line.orientation_index,
                    orientation_weight=line.orientation_weight * sample.weight,
                    b0_direction_g=line.b0_direction_g,
                    b1_direction_g=line.b1_direction_g,
                    field_tesla=applied_field,
                    frequency_hz=line.frequency_hz,
                    intensity=line.intensity * sample.weight,
                    transition_strength=line.transition_strength,
                )
            )

    if fields_tesla is None:
        axis = _field_axis(
            [line.field_tesla for line in lines],
            fallback=resonance_field_tesla(samples[0].system, frequency),
            width=broadening,
            points=points,
            span_tesla=span_tesla,
        )
    else:
        axis = _provided_axis(fields_tesla, "fields_tesla")

    spectrum = spectrum_from_lines(
        axis,
        [line.field_tesla for line in lines],
        [line.intensity for line in lines],
        width=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )
    return ESRFieldDistributionResult(
        fields_tesla=axis,
        spectrum=spectrum,
        lines=tuple(lines),
        samples=samples,
        microwave_frequency_hz=frequency,
        broadening_tesla=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )


def simulate_frequency_spectrum_distribution(
    samples: list[ESRDistributionSample] | tuple[ESRDistributionSample, ...],
    b0_tesla: float,
    *,
    orientations: OrientationInput = "single",
    broadening_hz: float = 1.0e6,
    points: int = 1024,
    span_hz: float | None = None,
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
) -> ESRFrequencyDistributionResult:
    """Return a frequency-swept ESR spectrum averaged over static disorder."""

    samples = normalize_distribution(samples)
    b0 = float(b0_tesla)
    if b0 < 0 or not np.isfinite(b0):
        raise ValueError("b0_tesla must be finite and non-negative")
    broadening = _positive_finite(broadening_hz, "broadening_hz")

    lines: list[ESRLine] = []
    for sample in samples:
        local_b0 = b0 + sample.field_offset_tesla
        if local_b0 < 0:
            continue
        result = simulate_frequency_spectrum(
            sample.system,
            local_b0,
            orientations=orientations,
            broadening_hz=broadening,
            points=2,
            lineshape=lineshape,
            detection_mode=detection_mode,
        )
        for line in result.lines:
            lines.append(
                ESRLine(
                    orientation_index=line.orientation_index,
                    orientation_weight=line.orientation_weight * sample.weight,
                    b0_direction_g=line.b0_direction_g,
                    b1_direction_g=line.b1_direction_g,
                    field_tesla=b0,
                    frequency_hz=line.frequency_hz,
                    intensity=line.intensity * sample.weight,
                    transition_strength=line.transition_strength,
                )
            )

    if frequencies_hz is None:
        axis = _frequency_axis(
            [line.frequency_hz for line in lines],
            fallback=resonance_frequency_hz(samples[0].system, b0),
            width=broadening,
            points=points,
            span_hz=span_hz,
        )
    else:
        axis = _provided_axis(frequencies_hz, "frequencies_hz")

    spectrum = spectrum_from_lines(
        axis,
        [line.frequency_hz for line in lines],
        [line.intensity for line in lines],
        width=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )
    return ESRFrequencyDistributionResult(
        frequencies_hz=axis,
        spectrum=spectrum,
        lines=tuple(lines),
        samples=samples,
        b0_tesla=b0,
        broadening_hz=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )


def _component_grids(g_std, points: int, n_sigma: float):
    values = np.asarray(g_std, dtype=np.float64)
    if values.shape == ():
        values = np.full(3, float(values), dtype=np.float64)
    else:
        values = values.reshape(3)
    return tuple(_scalar_grid(std, points, n_sigma) for std in values)


def _scalar_grid(std: float, points: int, n_sigma: float):
    std = float(std)
    points = int(points)
    n_sigma = float(n_sigma)
    if std < 0 or not np.isfinite(std):
        raise ValueError("distribution standard deviations must be non-negative")
    if points <= 0:
        raise ValueError("distribution point counts must be positive")
    if n_sigma <= 0 or not np.isfinite(n_sigma):
        raise ValueError("n_sigma must be positive and finite")
    if std == 0 or points == 1:
        return np.array([0.0]), np.array([1.0])
    offsets = np.linspace(-n_sigma * std, n_sigma * std, points)
    weights = np.exp(-0.5 * (offsets / std) ** 2)
    weights = weights / np.sum(weights)
    return offsets, weights


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if value <= 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be positive and finite")
    return value


def _provided_axis(values, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be finite")
    return axis


def _field_axis(
    centers: list[float],
    *,
    fallback: float,
    width: float,
    points: int,
    span_tesla: float | None,
) -> np.ndarray:
    return _auto_axis(
        centers,
        fallback=fallback,
        width=width,
        points=points,
        span=span_tesla,
    )


def _frequency_axis(
    centers: list[float],
    *,
    fallback: float,
    width: float,
    points: int,
    span_hz: float | None,
) -> np.ndarray:
    return _auto_axis(
        centers,
        fallback=fallback,
        width=width,
        points=points,
        span=span_hz,
    )


def _auto_axis(
    centers: list[float],
    *,
    fallback: float,
    width: float,
    points: int,
    span: float | None,
) -> np.ndarray:
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least two")
    centers = centers or [float(fallback)]
    if span is None:
        half_span = max(
            0.5 * (max(centers) - min(centers)) + 5.0 * width,
            5.0 * width,
        )
    else:
        half_span = 0.5 * float(span)
        if half_span <= 0 or not np.isfinite(half_span):
            raise ValueError("span must be positive and finite")
    center = 0.5 * (max(centers) + min(centers))
    return np.linspace(center - half_span, center + half_span, points)
