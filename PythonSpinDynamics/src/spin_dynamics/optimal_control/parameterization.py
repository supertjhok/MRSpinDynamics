"""Control-vector bounds and seeding for GRAPE.

Layout convention (matches ``objectives.make_grape_objective``): phase-only
control is a flat ``(n_segments,)`` phase array; joint amplitude+phase control
is ``concat([amplitude (n_segments,), phase (n_segments,)])``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlBounds:
    """Per-parameter box constraints matching a flattened GRAPE control vector."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64).reshape(-1)
        upper = np.asarray(self.upper, dtype=np.float64).reshape(-1)
        if lower.shape != upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if lower.size == 0:
            raise ValueError("bounds must not be empty")
        if np.any(lower >= upper):
            raise ValueError("lower must be strictly less than upper elementwise")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def as_pairs(self) -> list[tuple[float, float]]:
        """Return SciPy-style ``[(lower, upper), ...]`` bound pairs."""

        return list(zip(self.lower.tolist(), self.upper.tolist()))


def phase_only_bounds(n_segments: int, *, phase_bound_rad: float = 4 * np.pi) -> ControlBounds:
    """Loose symmetric phase bounds.

    Phase is periodic, so a hard box is not physically required for
    correctness (unlike amplitude, which has a genuine peak-B1 limit) -- this
    bound only keeps L-BFGS-B from wandering arbitrarily far from the
    principal range while it searches.
    """

    n_segments = int(n_segments)
    bound = float(phase_bound_rad)
    if bound <= 0:
        raise ValueError("phase_bound_rad must be positive")
    return ControlBounds(
        lower=np.full(n_segments, -bound),
        upper=np.full(n_segments, bound),
    )


def amplitude_phase_bounds(
    n_segments: int,
    *,
    amplitude_max_hz: float,
    phase_bound_rad: float = 4 * np.pi,
) -> ControlBounds:
    """Box bounds for joint amplitude+phase control.

    Amplitude is bounded to ``[0, amplitude_max_hz]`` -- the real peak-B1
    hardware limit, exact as a box constraint in this polar parameterization
    (unlike a Cartesian I/Q parameterization, where independently
    box-bounding both quadratures would incorrectly permit an effective
    amplitude up to ``amplitude_max_hz * sqrt(2)`` at the box corners). Phase
    is loosely bounded as in :func:`phase_only_bounds`. Layout matches
    ``make_grape_objective(optimize_amplitude=True)``'s control vector.
    """

    n_segments = int(n_segments)
    amp_max = float(amplitude_max_hz)
    if amp_max <= 0:
        raise ValueError("amplitude_max_hz must be positive")
    bound = float(phase_bound_rad)
    if bound <= 0:
        raise ValueError("phase_bound_rad must be positive")
    lower = np.concatenate([np.zeros(n_segments), np.full(n_segments, -bound)])
    upper = np.concatenate([np.full(n_segments, amp_max), np.full(n_segments, bound)])
    return ControlBounds(lower=lower, upper=upper)


def rectangular_seed_phase(n_segments: int, *, phase_rad: float = 0.0) -> np.ndarray:
    """Constant-phase baseline waveform.

    The phase-only analogue of a rectangular pulse: seeding the optimizer
    from this baseline guarantees the optimized result is no worse, the same
    trick used by ``examples/refocusing_gradient_optimization.py``.
    """

    return np.full(int(n_segments), float(phase_rad), dtype=np.float64)


def random_phase_starts(
    num_starts: int,
    n_segments: int,
    *,
    bounds: tuple[float, float] = (-np.pi, np.pi),
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Reproducible random phase starts.

    Mirrors ``optimization.drivers.random_phase_starts``'s signature and
    output shape ``(num_starts, n_segments)``.
    """

    if num_starts <= 0:
        raise ValueError("num_starts must be positive")
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")
    if seed is not None and rng is not None:
        raise ValueError("provide either seed or rng, not both")
    lower, upper = float(bounds[0]), float(bounds[1])
    if lower >= upper:
        raise ValueError("bounds must be ordered as (lower, upper)")
    generator = np.random.default_rng(seed) if rng is None else rng
    return generator.uniform(lower, upper, size=(int(num_starts), int(n_segments)))


def export_to_pulse_shape(
    *,
    duration_seconds: np.ndarray | float,
    amplitude_hz: np.ndarray | float,
    phase_rad: np.ndarray,
):
    """Export an optimized GRAPE waveform as an ``absolute_phase.PulseShape``.

    Amplitude is stored in hertz (``PulseShape.amplitude`` supports either
    normalized or hertz units); callers round-tripping through code that
    expects angular-frequency amplitude (e.g. ``core.rotations.rf_matrix_elements``)
    must multiply by ``2 * pi`` themselves.
    """

    from spin_dynamics.absolute_phase import PulseShape

    phase = np.asarray(phase_rad, dtype=np.float64).reshape(-1)
    n_segments = phase.size
    duration = np.broadcast_to(
        np.asarray(duration_seconds, dtype=np.float64), (n_segments,)
    )
    amplitude = np.broadcast_to(np.asarray(amplitude_hz, dtype=np.float64), (n_segments,))
    return PulseShape(duration=duration.copy(), phase=phase.copy(), amplitude=amplitude.copy())
