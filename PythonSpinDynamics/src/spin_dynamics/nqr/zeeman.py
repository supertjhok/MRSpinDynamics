"""Weak-static-field Zeeman perturbation helpers for NQR transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import warnings

import numpy as np

from spin_dynamics.nqr.hamiltonians import (
    diagonalize_site,
    diagonalize_sites_over_b0,
)
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    b0_b1_powder_average_grid,
    normalize_orientations,
)
from spin_dynamics.nqr.systems import QuadrupolarSite


OrientationInput = str | tuple[OrientationSample, ...] | list[OrientationSample]


@dataclass(frozen=True)
class WeakB0Transition:
    """One transition from an exact weak-field NQR diagonalization."""

    orientation_index: int
    orientation_weight: float
    b0_vector_tesla_pas: np.ndarray
    b0_magnitude_tesla: float
    zeeman_frequency_hz: float
    perturbation_ratio: float
    label: str
    lower: int
    upper: int
    frequency_hz: float
    shift_hz: float
    strength: float
    intensity: float


@dataclass(frozen=True)
class WeakB0SpectrumResult:
    """Powder/single-crystal weak-B0 transition spectrum."""

    offsets_hz: np.ndarray
    spectrum: np.ndarray
    transitions: tuple[WeakB0Transition, ...]
    reference_frequency_hz: float
    selected_zero_field_frequencies_hz: np.ndarray
    max_perturbation_ratio: float
    broadening_hz: float
    site: QuadrupolarSite


def zeeman_frequency_hz(
    site: QuadrupolarSite,
    b0_tesla: float | np.ndarray | Sequence[float],
) -> float:
    """Return ``|gamma B0|`` in Hz for a scalar or vector static field."""

    b0 = np.asarray(b0_tesla, dtype=np.float64)
    if b0.shape == ():
        magnitude = abs(float(b0))
    else:
        magnitude = float(np.linalg.norm(b0.reshape(3)))
    if not np.isfinite(magnitude):
        raise ValueError("b0_tesla must be finite")
    return abs(site.gamma_hz_per_t) * magnitude


def weak_field_ratio(
    site: QuadrupolarSite,
    b0_tesla: float | np.ndarray | Sequence[float],
    *,
    reference_frequency_hz: float | None = None,
) -> float:
    """Return ``|gamma B0| / nu_ref`` for weak-field validity checks."""

    if reference_frequency_hz is None:
        eigensystem = diagonalize_site(site)
        if not eigensystem.transitions:
            raise ValueError("site has no non-zero NQR transitions")
        reference_frequency_hz = min(
            item.frequency_hz for item in eigensystem.transitions
        )
    reference = float(reference_frequency_hz)
    if reference <= 0 or not np.isfinite(reference):
        raise ValueError("reference_frequency_hz must be positive and finite")
    return zeeman_frequency_hz(site, b0_tesla) / reference


def simulate_weak_b0_spectrum(
    site: QuadrupolarSite,
    b0_tesla: float,
    *,
    orientations: OrientationInput = "single",
    transition_label: str | None = None,
    broadening_hz: float = 100.0,
    points: int = 1024,
    span_hz: float | None = None,
    selection_window_hz: float | None = None,
    intensity_tolerance: float = 1e-14,
    weak_ratio_action: str = "warn",
    weak_ratio_threshold: float = 0.05,
    backend: str = "numpy",
) -> WeakB0SpectrumResult:
    """Return a broadened transition spectrum for ``H_Q + H_Z`` in weak B0.

    ``backend`` selects the diagonalizer for the orientation scan: ``"numpy"``
    (the reference) or ``"jax"`` (one batched GPU eigensolve over all
    orientations, requires the optional ``jax`` extra). Results are identical.
    """

    b0 = float(b0_tesla)
    if not np.isfinite(b0) or b0 < 0:
        raise ValueError("b0_tesla must be finite and non-negative")
    broadening = float(broadening_hz)
    if broadening <= 0 or not np.isfinite(broadening):
        raise ValueError("broadening_hz must be positive and finite")
    points = int(points)
    if points < 2:
        raise ValueError("points must be at least two")
    threshold = float(weak_ratio_threshold)
    if threshold <= 0 or not np.isfinite(threshold):
        raise ValueError("weak_ratio_threshold must be positive and finite")
    intensity_tolerance = float(intensity_tolerance)
    if intensity_tolerance < 0 or not np.isfinite(intensity_tolerance):
        raise ValueError("intensity_tolerance must be finite and non-negative")

    zero_field = diagonalize_site(site)
    if transition_label is None:
        selected_zero = np.array(
            [item.frequency_hz for item in zero_field.transitions],
            dtype=np.float64,
        )
    else:
        selected_zero = np.array(
            [zero_field.transition(transition_label).frequency_hz],
            dtype=np.float64,
        )
    if selected_zero.size == 0:
        raise ValueError("site has no selected zero-field transitions")
    reference = float(np.mean(selected_zero))
    max_ratio = weak_field_ratio(site, b0, reference_frequency_hz=reference)
    _handle_weak_ratio(max_ratio, threshold, weak_ratio_action)

    if selection_window_hz is None:
        selection_window = max(
            10.0 * zeeman_frequency_hz(site, b0),
            5.0 * broadening,
            1e-9,
        )
    else:
        selection_window = float(selection_window_hz)
        if selection_window <= 0 or not np.isfinite(selection_window):
            raise ValueError("selection_window_hz must be positive and finite")

    samples = _as_zeeman_orientations(orientations)
    b0_vectors = np.array(
        [
            b0
            * (
                sample.b0_direction_pas
                if sample.b0_direction_pas is not None
                else sample.b1_direction_pas
            )
            for sample in samples
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    # One batched eigensolve over every orientation instead of a per-sample loop.
    eigensystems = diagonalize_sites_over_b0(site, b0_vectors, backend=backend)

    transitions: list[WeakB0Transition] = []
    for orientation_index, sample in enumerate(samples):
        b0_vector = b0_vectors[orientation_index]
        eigensystem = eigensystems[orientation_index]
        ratio = weak_field_ratio(site, b0_vector, reference_frequency_hz=reference)
        for transition in eigensystem.transitions:
            nearest = float(
                selected_zero[
                    np.argmin(np.abs(selected_zero - transition.frequency_hz))
                ]
            )
            if abs(transition.frequency_hz - nearest) > selection_window:
                continue
            rf_amplitude = np.vdot(sample.b1_direction_pas, transition.dipole_vector)
            intensity = float(sample.weight * abs(rf_amplitude) ** 2)
            if intensity <= intensity_tolerance:
                continue
            transitions.append(
                WeakB0Transition(
                    orientation_index=orientation_index,
                    orientation_weight=sample.weight,
                    b0_vector_tesla_pas=b0_vector,
                    b0_magnitude_tesla=b0,
                    zeeman_frequency_hz=zeeman_frequency_hz(site, b0_vector),
                    perturbation_ratio=ratio,
                    label=transition.label,
                    lower=transition.lower,
                    upper=transition.upper,
                    frequency_hz=transition.frequency_hz,
                    shift_hz=transition.frequency_hz - nearest,
                    strength=transition.strength,
                    intensity=intensity,
                )
            )

    if span_hz is None:
        max_shift = max(
            [abs(item.frequency_hz - reference) for item in transitions] + [broadening]
        )
        half_span = max(max_shift + 5.0 * broadening, 5.0 * broadening)
    else:
        half_span = 0.5 * float(span_hz)
        if half_span <= 0 or not np.isfinite(half_span):
            raise ValueError("span_hz must be positive and finite")

    offsets = np.linspace(-half_span, half_span, points)
    spectrum = np.zeros(points, dtype=np.float64)
    for transition in transitions:
        center = transition.frequency_hz - reference
        spectrum += transition.intensity * np.exp(
            -0.5 * ((offsets - center) / broadening) ** 2
        )

    max_ratio = max([item.perturbation_ratio for item in transitions] + [max_ratio])
    return WeakB0SpectrumResult(
        offsets_hz=offsets,
        spectrum=spectrum,
        transitions=tuple(transitions),
        reference_frequency_hz=reference,
        selected_zero_field_frequencies_hz=selected_zero,
        max_perturbation_ratio=float(max_ratio),
        broadening_hz=broadening,
        site=site,
    )


def _as_zeeman_orientations(
    orientations: OrientationInput,
) -> tuple[OrientationSample, ...]:
    if isinstance(orientations, str):
        if orientations == "single":
            return normalize_orientations(
                [
                    OrientationSample(
                        (1.0, 0.0, 0.0),
                        b0_direction_pas=(0.0, 0.0, 1.0),
                    )
                ]
            )
        if orientations == "powder":
            return b0_b1_powder_average_grid()
        raise ValueError("orientations string must be 'single' or 'powder'")
    return normalize_orientations(tuple(orientations))


def _handle_weak_ratio(ratio: float, threshold: float, action: str) -> None:
    if ratio <= threshold or action == "ignore":
        return
    message = (
        "Zeeman perturbation is not small compared with the selected NQR "
        f"frequency: gamma_B0_over_nu={ratio:.6g}, threshold={threshold:.6g}. "
        "Reduce b0_tesla or set weak_ratio_action='ignore' after checking the "
        "intended regime."
    )
    if action == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    elif action == "raise":
        raise RuntimeError(message)
    else:
        raise ValueError("weak_ratio_action must be 'ignore', 'warn', or 'raise'")
