"""Frequency- and field-swept ESR spectrum helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.esr.hamiltonians import (
    diagonalize_system,
    resonance_field_tesla,
    resonance_frequency_hz,
)
from spin_dynamics.esr.lineshapes import spectrum_from_lines
from spin_dynamics.esr.orientations import (
    ESROrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.esr.systems import ESRSpinSystem


OrientationInput = str | tuple[ESROrientationSample, ...] | list[ESROrientationSample]


@dataclass(frozen=True)
class ESRLine:
    """One orientation-resolved ESR transition line."""

    orientation_index: int
    orientation_weight: float
    b0_direction_g: np.ndarray
    b1_direction_g: np.ndarray
    field_tesla: float
    frequency_hz: float
    intensity: float
    transition_strength: float


@dataclass(frozen=True)
class ESRFrequencySpectrumResult:
    """Fixed-field ESR spectrum on a frequency axis."""

    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    lines: tuple[ESRLine, ...]
    b0_tesla: float
    broadening_hz: float
    system: ESRSpinSystem
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"


@dataclass(frozen=True)
class ESRFieldSweepResult:
    """Fixed-frequency ESR spectrum on a static-field axis."""

    fields_tesla: np.ndarray
    spectrum: np.ndarray
    lines: tuple[ESRLine, ...]
    microwave_frequency_hz: float
    broadening_tesla: float
    system: ESRSpinSystem
    lineshape: str = "gaussian"
    detection_mode: str = "absorption"


def _perpendicular_direction(direction: np.ndarray) -> np.ndarray:
    reference = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(direction[2])) < 0.9
        else np.array([1.0, 0.0, 0.0])
    )
    out = np.cross(direction, reference)
    return out / np.linalg.norm(out)


def _as_orientations(
    orientations: OrientationInput,
) -> tuple[ESROrientationSample, ...]:
    if isinstance(orientations, str):
        if orientations == "single":
            return normalize_orientations(
                [
                    ESROrientationSample(
                        b0_direction_g=(0.0, 0.0, 1.0),
                        b1_direction_g=(1.0, 0.0, 0.0),
                    )
                ]
            )
        if orientations == "powder":
            return powder_average_grid()
        raise ValueError("orientations string must be 'single' or 'powder'")
    return normalize_orientations(tuple(orientations))


def _line_intensity(
    system: ESRSpinSystem,
    field_tesla: float,
    sample: ESROrientationSample,
) -> tuple[float, float, np.ndarray]:
    b1_direction = (
        _perpendicular_direction(sample.b0_direction_g)
        if sample.b1_direction_g is None
        else sample.b1_direction_g
    )
    eigensystem = diagonalize_system(system, field_tesla * sample.b0_direction_g)
    if not eigensystem.transitions:
        return 0.0, 0.0, b1_direction
    transition = eigensystem.transitions[0]
    rf_amplitude = np.vdot(b1_direction, transition.dipole_vector)
    intensity = float(sample.weight * abs(rf_amplitude) ** 2)
    return intensity, transition.strength, b1_direction


def simulate_frequency_spectrum(
    system: ESRSpinSystem,
    b0_tesla: float,
    *,
    orientations: OrientationInput = "single",
    broadening_hz: float = 1.0e6,
    points: int = 1024,
    span_hz: float | None = None,
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
) -> ESRFrequencySpectrumResult:
    """Return a broadened fixed-field ESR spectrum on a frequency axis."""

    b0 = float(b0_tesla)
    if b0 < 0 or not np.isfinite(b0):
        raise ValueError("b0_tesla must be finite and non-negative")
    broadening = float(broadening_hz)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_hz must be positive and finite")
    samples = _as_orientations(orientations)

    lines: list[ESRLine] = []
    for orientation_index, sample in enumerate(samples):
        frequency = resonance_frequency_hz(system, b0 * sample.b0_direction_g)
        intensity, strength, b1_direction = _line_intensity(system, b0, sample)
        if intensity <= 0:
            continue
        lines.append(
            ESRLine(
                orientation_index=orientation_index,
                orientation_weight=sample.weight,
                b0_direction_g=sample.b0_direction_g,
                b1_direction_g=b1_direction,
                field_tesla=b0,
                frequency_hz=frequency,
                intensity=intensity,
                transition_strength=strength,
            )
        )

    if frequencies_hz is None:
        points = int(points)
        if points < 2:
            raise ValueError("points must be at least two")
        centers = [line.frequency_hz for line in lines] or [
            resonance_frequency_hz(system, b0)
        ]
        if span_hz is None:
            half_span = max(
                0.5 * (max(centers) - min(centers)) + 5.0 * broadening,
                5.0 * broadening,
            )
        else:
            half_span = 0.5 * float(span_hz)
            if half_span <= 0 or not np.isfinite(half_span):
                raise ValueError("span_hz must be positive and finite")
        center = 0.5 * (max(centers) + min(centers))
        axis = np.linspace(center - half_span, center + half_span, points)
    else:
        axis = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
        if axis.size < 2:
            raise ValueError("frequencies_hz must contain at least two points")
        if not np.all(np.isfinite(axis)):
            raise ValueError("frequencies_hz must be finite")

    spectrum = spectrum_from_lines(
        axis,
        [line.frequency_hz for line in lines],
        [line.intensity for line in lines],
        width=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )

    return ESRFrequencySpectrumResult(
        frequencies_hz=axis,
        spectrum=spectrum,
        lines=tuple(lines),
        b0_tesla=b0,
        broadening_hz=broadening,
        system=system,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )


def simulate_field_sweep(
    system: ESRSpinSystem,
    microwave_frequency_hz: float,
    *,
    orientations: OrientationInput = "single",
    broadening_tesla: float = 1.0e-4,
    points: int = 1024,
    span_tesla: float | None = None,
    fields_tesla: np.ndarray | list[float] | tuple[float, ...] | None = None,
    lineshape: str = "gaussian",
    detection_mode: str = "absorption",
) -> ESRFieldSweepResult:
    """Return a broadened fixed-frequency ESR spectrum on a field axis."""

    frequency = float(microwave_frequency_hz)
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError("microwave_frequency_hz must be positive and finite")
    broadening = float(broadening_tesla)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_tesla must be positive and finite")
    samples = _as_orientations(orientations)

    lines: list[ESRLine] = []
    for orientation_index, sample in enumerate(samples):
        field = resonance_field_tesla(system, frequency, sample.b0_direction_g)
        intensity, strength, b1_direction = _line_intensity(system, field, sample)
        if intensity <= 0:
            continue
        lines.append(
            ESRLine(
                orientation_index=orientation_index,
                orientation_weight=sample.weight,
                b0_direction_g=sample.b0_direction_g,
                b1_direction_g=b1_direction,
                field_tesla=field,
                frequency_hz=frequency,
                intensity=intensity,
                transition_strength=strength,
            )
        )

    if fields_tesla is None:
        points = int(points)
        if points < 2:
            raise ValueError("points must be at least two")
        centers = [line.field_tesla for line in lines] or [
            resonance_field_tesla(system, frequency)
        ]
        if span_tesla is None:
            half_span = max(
                0.5 * (max(centers) - min(centers)) + 5.0 * broadening,
                5.0 * broadening,
            )
        else:
            half_span = 0.5 * float(span_tesla)
            if half_span <= 0 or not np.isfinite(half_span):
                raise ValueError("span_tesla must be positive and finite")
        center = 0.5 * (max(centers) + min(centers))
        axis = np.linspace(center - half_span, center + half_span, points)
    else:
        axis = np.asarray(fields_tesla, dtype=np.float64).reshape(-1)
        if axis.size < 2:
            raise ValueError("fields_tesla must contain at least two points")
        if not np.all(np.isfinite(axis)):
            raise ValueError("fields_tesla must be finite")

    spectrum = spectrum_from_lines(
        axis,
        [line.field_tesla for line in lines],
        [line.intensity for line in lines],
        width=broadening,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )

    return ESRFieldSweepResult(
        fields_tesla=axis,
        spectrum=spectrum,
        lines=tuple(lines),
        microwave_frequency_hz=frequency,
        broadening_tesla=broadening,
        system=system,
        lineshape=lineshape,
        detection_mode=detection_mode,
    )
