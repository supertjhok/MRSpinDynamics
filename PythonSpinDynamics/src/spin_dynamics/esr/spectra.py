"""Frequency- and field-swept ESR spectrum helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.constants import FREE_ELECTRON_G
from spin_dynamics.esr.hamiltonians import (
    diagonalize_system,
    effective_g_value,
    resonance_field_tesla,
    resonance_frequency_hz,
)
from spin_dynamics.esr.lineshapes import spectrum_from_lines
from spin_dynamics.esr.orientations import (
    ESROrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.esr.systems import BOHR_MAGNETON_HZ_PER_T, ESRSpinSystem


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


# Default powder grid (L-64): Gauss-Legendre polar knots per linewidth of
# resonance spread, with twice as many azimuths. Convergence is very fast:
# 1.4 per linewidth leaves 0.1-0.5% error in absorption but 2-3% ripple in
# derivative spectra (L-65); 2.1 brings both below 0.05%. The
# cap keeps a sweep under 4 x 10^5 lines (about 2 s); samples of a static
# disorder distribution share that budget.
_POWDER_KNOTS_PER_LINEWIDTH = 2.1
_MIN_POWDER_KNOTS = 16
_MAX_POWDER_KNOTS = 256


def _powder_grid(
    spread: float, width: float, *, shared_by: int = 1
) -> tuple[ESROrientationSample, ...]:
    """Return a powder grid fine enough for ``spread`` (same units as ``width``).

    Three microwave-field directions per orientation average ``|B1 . moment|^2``
    over the plane perpendicular to ``B0`` exactly, because it is quadratic.
    ``shared_by`` sweeps divide the line budget, so the knot cap shrinks
    as its square root.
    """

    cap = max(_MIN_POWDER_KNOTS, int(_MAX_POWDER_KNOTS / np.sqrt(max(shared_by, 1))))
    knots = int(
        np.clip(np.ceil(_POWDER_KNOTS_PER_LINEWIDTH * spread / width), _MIN_POWDER_KNOTS, cap)
    )
    return powder_average_grid(knots, 2 * knots, 3)


def _g_extremes(system: ESRSpinSystem) -> tuple[float, float]:
    """Smallest and largest effective g over all field directions."""

    singular = np.linalg.svd(np.asarray(system.g_tensor, dtype=np.float64), compute_uv=False)
    return float(singular.min()), float(singular.max())


def _field_sweep_spread(system: ESRSpinSystem, microwave_frequency_hz: float) -> float:
    """Range of resonance fields over all orientations, in tesla."""

    g_min, g_max = _g_extremes(system)
    return microwave_frequency_hz / BOHR_MAGNETON_HZ_PER_T * (1.0 / g_min - 1.0 / g_max)


def _frequency_spread(system: ESRSpinSystem, b0_tesla: float) -> float:
    """Range of resonance frequencies over all orientations, in hertz."""

    g_min, g_max = _g_extremes(system)
    return BOHR_MAGNETON_HZ_PER_T * b0_tesla * (g_max - g_min)


def _as_orientations(
    orientations: OrientationInput,
    *,
    spread: float | None = None,
    width: float | None = None,
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
            if spread is None or width is None:
                return powder_average_grid()
            return _powder_grid(spread, width)
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


def _spin_half_lines(
    system: ESRSpinSystem,
    samples: tuple[ESROrientationSample, ...],
    *,
    field_tesla: float | None = None,
    microwave_frequency_hz: float | None = None,
) -> list[ESRLine]:
    """Orientation lines of a Zeeman-only spin 1/2, vectorized over samples.

    These equal diagonalizing each orientation: with quantization axis
    ``m = g^T n / |g^T n|`` and drive ``e = g^T b1 / g_e``, the transition
    probability is ``|e x m|^2 / 4`` and the moment norm is
    ``sqrt(tr(G^T G) - m^T G^T G m) / 2`` with ``G = g / g_e``. Give
    ``field_tesla`` for a frequency spectrum or ``microwave_frequency_hz`` for a
    field sweep, which also carries the ``g_e / g_eff`` factor.
    """

    g = np.asarray(system.g_tensor, dtype=np.float64)
    b0 = np.array([sample.b0_direction_g for sample in samples], dtype=np.float64)
    b1 = np.array(
        [
            _perpendicular_direction(sample.b0_direction_g)
            if sample.b1_direction_g is None
            else sample.b1_direction_g
            for sample in samples
        ],
        dtype=np.float64,
    )
    weights = np.array([sample.weight for sample in samples], dtype=np.float64)
    g_vectors = b0 @ g
    g_eff = np.linalg.norm(g_vectors, axis=1)
    if np.any(g_eff <= 0):
        raise ValueError("effective g value must be positive")
    axis = g_vectors / g_eff[:, None]
    drive = (b1 @ g) / FREE_ELECTRON_G
    probability = np.sum(np.cross(drive, axis) ** 2, axis=1) / 4.0
    metric = (g.T @ g) / FREE_ELECTRON_G**2
    strength = 0.5 * np.sqrt(
        np.maximum(np.trace(metric) - np.einsum("ni,ij,nj->n", axis, metric, axis), 0.0)
    )
    if microwave_frequency_hz is not None:
        fields = microwave_frequency_hz / (BOHR_MAGNETON_HZ_PER_T * g_eff)
        frequencies = np.full(len(samples), float(microwave_frequency_hz))
        intensities = weights * probability * FREE_ELECTRON_G / g_eff
    else:
        assert field_tesla is not None
        fields = np.full(len(samples), float(field_tesla))
        frequencies = BOHR_MAGNETON_HZ_PER_T * field_tesla * g_eff
        intensities = weights * probability
    return [
        ESRLine(
            orientation_index=index,
            orientation_weight=float(weights[index]),
            b0_direction_g=samples[index].b0_direction_g,
            b1_direction_g=b1[index],
            field_tesla=float(fields[index]),
            frequency_hz=float(frequencies[index]),
            intensity=float(intensities[index]),
            transition_strength=float(strength[index]),
        )
        for index in np.nonzero(intensities > 0)[0]
    ]


def _frequency_lines_by_diagonalization(
    system: ESRSpinSystem, b0: float, samples: tuple[ESROrientationSample, ...]
) -> list[ESRLine]:
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
    return lines


def _field_lines_by_diagonalization(
    system: ESRSpinSystem, frequency: float, samples: tuple[ESROrientationSample, ...]
) -> list[ESRLine]:
    lines: list[ESRLine] = []
    for orientation_index, sample in enumerate(samples):
        field = resonance_field_tesla(system, frequency, sample.b0_direction_g)
        intensity, strength, b1_direction = _line_intensity(system, field, sample)
        # Field-swept (Aasa-Vanngard) factor: d(nu)/dB is g_eff mu_B.
        intensity *= FREE_ELECTRON_G / effective_g_value(system, sample.b0_direction_g)
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
    return lines


def _broadened(
    axis: np.ndarray,
    centers: list[float],
    intensities: list[float],
    **kwargs,
) -> np.ndarray:
    """Broaden lines, summing first those at the same position.

    The microwave-field directions of one powder orientation share its
    resonance, so this evaluates one profile per orientation.
    """

    unique, inverse = np.unique(np.asarray(centers, dtype=np.float64), return_inverse=True)
    summed = np.bincount(inverse.reshape(-1), weights=np.asarray(intensities), minlength=unique.size)
    return spectrum_from_lines(axis, unique, summed, **kwargs)


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
    """Return a broadened fixed-field ESR spectrum on a frequency axis.

    ``broadening_hz`` is ``sigma`` for Gaussian and HWHM for Lorentzian lines
    (see :func:`spin_dynamics.esr.lineshapes.width_from_fwhm`).
    ``orientations="powder"`` sizes its grid from the g anisotropy relative to
    the linewidth (16 to 256 polar knots).
    """

    b0 = float(b0_tesla)
    if b0 < 0 or not np.isfinite(b0):
        raise ValueError("b0_tesla must be finite and non-negative")
    broadening = float(broadening_hz)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_hz must be positive and finite")
    samples = _as_orientations(
        orientations, spread=_frequency_spread(system, b0), width=broadening
    )
    if system.spin == 0.5:
        lines = _spin_half_lines(system, samples, field_tesla=b0)
    else:
        lines = _frequency_lines_by_diagonalization(system, b0, samples)

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

    spectrum = _broadened(
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
    """Return a broadened fixed-frequency ESR spectrum on a field axis.

    ``broadening_tesla`` is ``sigma`` for Gaussian and HWHM for Lorentzian
    lines (see :func:`spin_dynamics.esr.lineshapes.width_from_fwhm`).
    Each line carries the transition probability times the field-swept
    Aasa-Vanngard factor ``g_e / g_eff``: a line of unit frequency-domain
    weight spans ``1 / (g_eff mu_B)`` of field, so anisotropic powder
    patterns weight high-``g`` orientations less (as EasySpin's default).
    ``orientations="powder"`` sizes its grid from the g anisotropy relative to
    the linewidth (16 to 256 polar knots).
    """

    frequency = float(microwave_frequency_hz)
    if frequency <= 0 or not np.isfinite(frequency):
        raise ValueError("microwave_frequency_hz must be positive and finite")
    broadening = float(broadening_tesla)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_tesla must be positive and finite")
    samples = _as_orientations(
        orientations, spread=_field_sweep_spread(system, frequency), width=broadening
    )
    if system.spin == 0.5:
        lines = _spin_half_lines(system, samples, microwave_frequency_hz=frequency)
    else:
        lines = _field_lines_by_diagonalization(system, frequency, samples)

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

    spectrum = _broadened(
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
