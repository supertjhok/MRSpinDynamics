"""Nonresonant field-reversal spin echoes (Brill et al., Science 297, 369, 2002).

Nonresonant NMR manipulates spins *without RF*: nuclear magnetization is rotated by
**suddenly switching** and **adiabatically rotating** applied magnetic fields, with
an efficiency independent of the Larmor frequency (useful for ex-situ sensors in
grossly inhomogeneous fields). Because the manipulation is a field reversal rather
than a resonant pulse, a non-reversible background field -- typically Earth's field,
which cannot be shielded outside a magnet -- survives the reversal and dephases the
echoes within a few milliseconds. This module models that dynamics classically: a
spin-1/2 with no J-coupling is a Bloch 3-vector precessing about the instantaneous
local field, exactly the picture the paper's own "effective rotation axis" analysis
uses.

The engine is an ensemble of isochromats (spin packets) with position-dependent coil
field magnitudes and directions plus a shared non-reversible background. Its key
efficiency trick: a constant-field segment ("hold", including free precession) is a
*single exact rotation* by ``gamma |B| tau`` about the field direction -- no
Larmor-timescale time stepping. Only adiabatic rotations and finite-duration field
reversals are finely sub-stepped, and only finely enough to resolve the (slow) field
trajectory, since a rotation about a fixed axis is exact for any angle.

See :mod:`spin_dynamics.nonresonant.sequences` for the basic and CSAR sequence
builders and ``examples/plot_brill2002_field_reversal_echoes.py`` for a reproduction
of the paper's figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

# |gamma|/2pi for the proton, in hertz per tesla (matches the value used elsewhere
# in the package, e.g. nqr.polarization_enhancement).
PROTON_GAMMA_HZ_PER_T = 42.57747892e6
# A representative Earth's-field magnitude (the non-reversible background).
EARTH_FIELD_TESLA = 50e-6


def _unit(vector) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vec))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("direction must be a finite non-zero vector")
    return vec / norm


@dataclass(frozen=True)
class NonresonantFieldModel:
    """Two orthogonal switching coils plus a non-reversible background field.

    ``coil_a``/``coil_b`` are the two mostly-orthogonal Helmholtz coils of the CSAR
    apparatus (Fig. 1A of Brill 2002); ``*_tesla`` is each coil's nominal field
    magnitude at full current. ``background_tesla`` along ``background_direction`` is
    the non-reversible field (Earth's field, ~50 uT, taken along the coil-B axis as
    in the paper). ``gyromagnetic_hz_per_t`` is ``|gamma|/2pi`` (proton by default).
    """

    coil_a_tesla: float
    coil_b_tesla: float
    coil_a_direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    coil_b_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    background_tesla: float = EARTH_FIELD_TESLA
    background_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    gyromagnetic_hz_per_t: float = PROTON_GAMMA_HZ_PER_T

    @property
    def gamma_rad(self) -> float:
        """Angular gyromagnetic ratio ``gamma`` in rad/s/T."""
        return 2.0 * np.pi * float(self.gyromagnetic_hz_per_t)

    @property
    def background_vector(self) -> np.ndarray:
        """The non-reversible background field as a tesla vector."""
        return float(self.background_tesla) * _unit(self.background_direction)


@dataclass(frozen=True)
class IsochromatEnsemble:
    """Per-isochromat coil directions, field scales, and weights.

    ``coil_a_dir``/``coil_b_dir`` are ``(N, 3)`` unit vectors (each isochromat's coil
    directions, tilted by the coil inhomogeneity); ``coil_a_scale``/``coil_b_scale``
    are ``(N,)`` multipliers on the nominal coil fields (the field-magnitude
    inhomogeneity that field reversal refocuses into echoes). ``background`` is the
    shared ``(3,)`` background tesla vector; ``gamma_rad`` the angular gyromagnetic
    ratio; ``weight`` the ``(N,)`` normalized quadrature weights.
    """

    coil_a_dir: np.ndarray
    coil_b_dir: np.ndarray
    coil_a_scale: np.ndarray
    coil_b_scale: np.ndarray
    background: np.ndarray
    gamma_rad: float
    weight: np.ndarray

    @property
    def size(self) -> int:
        return int(self.coil_a_scale.shape[0])

    def total_field(self, a_tesla: float, b_tesla: float) -> np.ndarray:
        """Return the ``(N, 3)`` total field at nominal coil fields ``a``/``b``."""
        return (
            (self.coil_a_scale * a_tesla)[:, None] * self.coil_a_dir
            + (self.coil_b_scale * b_tesla)[:, None] * self.coil_b_dir
            + self.background[None, :]
        )


def _tilted_directions(nominal, tilt_rad, rng, n) -> np.ndarray:
    """Return ``n`` unit vectors near ``nominal``, each tilted by a random angle."""

    axis = _unit(nominal)
    # an orthonormal pair spanning the plane perpendicular to the nominal axis
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = tilt_rad * np.sqrt(rng.uniform(0.0, 1.0, n))  # area-weighted small cone
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    perp = np.cos(phi)[:, None] * e1[None, :] + np.sin(phi)[:, None] * e2[None, :]
    return np.cos(theta)[:, None] * axis[None, :] + np.sin(theta)[:, None] * perp


def sample_isochromats(
    model: NonresonantFieldModel,
    n: int,
    *,
    b_inhomogeneity: float = 0.25,
    a_inhomogeneity: float = 0.0,
    direction_tilt_deg: float = 15.0,
    seed: int = 0,
) -> IsochromatEnsemble:
    """Sample an isochromat ensemble with coil field and direction inhomogeneity.

    Two distinct inhomogeneities drive the physics: the coil-field **magnitude**
    spread (``b_inhomogeneity``, the ~25% RMS ``B_B`` inhomogeneity of the paper's
    apparatus) sets the free-precession dephasing that a field reversal *refocuses*
    into echoes, while the coil **direction** tilt (``direction_tilt_deg``, giving the
    ~15-degree spread of the angle ``alpha`` between the local field and the
    background) sets the residual, non-refocusable phase from the background field
    (``2 gamma B_e t cos alpha`` per reversal) that decays the basic sequence. All
    isochromats share the same background field.
    """

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    tilt = np.deg2rad(0.5 * float(direction_tilt_deg))  # half-angle of the cone
    coil_a_dir = _tilted_directions(model.coil_a_direction, tilt, rng, n)
    coil_b_dir = _tilted_directions(model.coil_b_direction, tilt, rng, n)
    a_scale = np.clip(1.0 + float(a_inhomogeneity) * rng.standard_normal(n), 0.05, None)
    b_scale = np.clip(1.0 + float(b_inhomogeneity) * rng.standard_normal(n), 0.05, None)
    weight = np.full(n, 1.0 / n)
    return IsochromatEnsemble(
        coil_a_dir=coil_a_dir,
        coil_b_dir=coil_b_dir,
        coil_a_scale=a_scale,
        coil_b_scale=b_scale,
        background=model.background_vector,
        gamma_rad=model.gamma_rad,
        weight=weight,
    )


def rodrigues_rotate(vectors: np.ndarray, axis_unit: np.ndarray, angle) -> np.ndarray:
    """Rotate ``(N, 3)`` vectors about per-row unit axes by ``angle`` (Rodrigues).

    ``angle`` is a scalar or ``(N,)`` array. This is the single vectorized rotation
    core: a spin's Bloch vector rotates about the local field direction by
    ``gamma |B| dt`` over a segment where the field is constant.
    """

    vectors = np.asarray(vectors, dtype=np.float64)
    axis_unit = np.asarray(axis_unit, dtype=np.float64)
    angle = np.atleast_1d(np.asarray(angle, dtype=np.float64))
    if angle.size == 1:
        angle = np.full(vectors.shape[0], float(angle[0]))
    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]
    dot = np.sum(vectors * axis_unit, axis=1)[:, None]
    cross = np.cross(axis_unit, vectors)
    return vectors * cos + cross * sin + axis_unit * dot * (1.0 - cos)


@dataclass(frozen=True)
class FieldSegment:
    """One piecewise-constant stretch of a nonresonant sequence.

    ``a_tesla``/``b_tesla`` are the nominal coil-A/coil-B field magnitudes at each
    sub-step and ``dt`` the sub-step durations (all shape ``(n_steps,)``). A constant
    hold (free precession, or a held field) is a single sub-step -- one exact
    rotation. An adiabatic rotation or a finite-duration reversal is many sub-steps
    that resolve the field trajectory. Build these with the helpers in
    :mod:`spin_dynamics.nonresonant.sequences`.
    """

    a_tesla: np.ndarray
    b_tesla: np.ndarray
    dt: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        a = np.atleast_1d(np.asarray(self.a_tesla, dtype=np.float64))
        b = np.atleast_1d(np.asarray(self.b_tesla, dtype=np.float64))
        dt = np.atleast_1d(np.asarray(self.dt, dtype=np.float64))
        n = max(a.size, b.size, dt.size)
        a = np.broadcast_to(a, (n,)).copy()
        b = np.broadcast_to(b, (n,)).copy()
        dt = np.broadcast_to(dt, (n,)).copy()
        object.__setattr__(self, "a_tesla", a)
        object.__setattr__(self, "b_tesla", b)
        object.__setattr__(self, "dt", dt)

    @property
    def duration(self) -> float:
        return float(np.sum(self.dt))


def sequence_waveform(unit, *, repeats: int = 1):
    """Return ``(times, i_a, i_b)`` step arrays of the coil currents over the sequence.

    Concatenates the per-sub-step nominal coil-A / coil-B field values of ``unit``
    (repeated ``repeats`` times) into piecewise-constant step arrays suitable for
    plotting the drive waveform (solid = coil B, dashed = coil A, as in Brill 2002
    Fig. 1D/Fig. 3). Fields are returned in tesla and times in seconds; the segment
    ``dt`` of 0 (an ideal instantaneous reversal) shows as a vertical step.
    """

    items = list(unit)
    if items and not isinstance(items[0], FieldSegment):
        items = [seg for sub in items for seg in sub]  # flatten a supercycle
    times = [0.0]
    i_a = [float(items[0].a_tesla[0])]
    i_b = [float(items[0].b_tesla[0])]
    t = 0.0
    for _ in range(int(repeats)):
        for seg in items:
            for a, b, dt in zip(seg.a_tesla, seg.b_tesla, seg.dt):
                # step to the new level, then advance time holding it
                times.append(t)
                i_a.append(float(a))
                i_b.append(float(b))
                t += float(dt)
                times.append(t)
                i_a.append(float(a))
                i_b.append(float(b))
    return np.array(times), np.array(i_a), np.array(i_b)


def evolve_segment(
    magnetization: np.ndarray, ensemble: IsochromatEnsemble, segment: FieldSegment
) -> np.ndarray:
    """Evolve the ``(N, 3)`` magnetization through one :class:`FieldSegment`."""

    m = magnetization
    gamma = ensemble.gamma_rad
    for a, b, dt in zip(segment.a_tesla, segment.b_tesla, segment.dt):
        field = ensemble.total_field(float(a), float(b))
        mag = np.linalg.norm(field, axis=1)
        safe = mag > 0
        axis = np.zeros_like(field)
        axis[safe] = field[safe] / mag[safe, None]
        m = rodrigues_rotate(m, axis, gamma * mag * float(dt))
    return m


@dataclass(frozen=True)
class FieldReversalResult:
    """Echo train from a nonresonant field-reversal sequence.

    ``echo_times`` and ``echo_amplitudes`` (complex: in-phase + i*quadrature of the
    coherent transverse magnetization, projected on the detection axes) are the
    per-echo train. ``coherent_amplitudes`` is the train *before* the optional
    ``T2`` envelope; ``relaxation_envelope`` is that envelope (ones if ``t2`` is
    infinite). ``magnetization`` optionally holds the final per-isochromat state.
    """

    echo_times: np.ndarray
    echo_amplitudes: np.ndarray
    coherent_amplitudes: np.ndarray
    relaxation_envelope: np.ndarray
    detection_axes: tuple[np.ndarray, np.ndarray]
    magnetization: np.ndarray | None = None

    @property
    def magnitude(self) -> np.ndarray:
        """Echo magnitudes ``|echo_amplitudes|``."""
        return np.abs(self.echo_amplitudes)


def _detection_axes(model: NonresonantFieldModel):
    """Return two orthonormal axes spanning the plane transverse to the background."""

    zhat = _unit(model.background_direction)
    ref = _unit(model.coil_a_direction)
    x = ref - np.dot(ref, zhat) * zhat
    if np.linalg.norm(x) < 1e-9:
        ref = np.array([0.0, 0.0, 1.0]) if abs(zhat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = ref - np.dot(ref, zhat) * zhat
    x = x / np.linalg.norm(x)
    y = np.cross(zhat, x)
    return x, y


def _as_unit_cycle(unit) -> list[list[FieldSegment]]:
    """Normalize ``unit`` to a list of units (a supercycle), one read per unit.

    Accepts either a single unit (a list of :class:`FieldSegment`, repeated every
    echo) or a supercycle (a list of such units, cycled -- one echo read per unit).
    """

    items = list(unit)
    if not items:
        raise ValueError("unit must contain at least one segment")
    if isinstance(items[0], FieldSegment):
        return [items]
    cycle = [list(u) for u in items]
    if any(len(u) == 0 for u in cycle):
        raise ValueError("every unit in a supercycle must be non-empty")
    return cycle


def simulate_field_reversal_echoes(
    model: NonresonantFieldModel,
    ensemble: IsochromatEnsemble,
    unit: Sequence[FieldSegment] | Sequence[Sequence[FieldSegment]],
    *,
    num_echoes: int,
    echo_spacing_seconds: float | None = None,
    t2_seconds: float = np.inf,
    initial_direction=None,
    return_magnetization: bool = False,
) -> FieldReversalResult:
    """Simulate a repeated nonresonant refocusing unit and read the echo train.

    The magnetization starts polarized along ``initial_direction`` (default: the
    coil-A axis, the polarization field). ``unit`` is either a single refocusing
    period (a list of :class:`FieldSegment`, repeated every echo) or a *supercycle*
    (a list of such units, cycled -- one echo read per unit; used for the Fig. 3C
    alternating-sense supercycle). After each period the coherent transverse
    magnetization is read as the complex echo amplitude. The optional ``t2_seconds``
    multiplies the coherent train by ``exp(-t/T2)`` -- kept as a post-hoc envelope so
    the coherent refocusing physics (the odd-even / Bessel modulation) and the
    intrinsic relaxation stay cleanly separable, exactly as the paper decomposes the
    decay.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    cycle = _as_unit_cycle(unit)

    init = model.coil_a_direction if initial_direction is None else initial_direction
    m0 = _unit(init)
    m = np.tile(m0, (ensemble.size, 1)).astype(np.float64)

    period = sum(seg.duration for seg in cycle[0])
    spacing = period if echo_spacing_seconds is None else float(echo_spacing_seconds)

    ax_x, ax_y = _detection_axes(model)
    weight = ensemble.weight
    coherent = np.empty(num_echoes, dtype=np.complex128)
    for echo in range(num_echoes):
        for seg in cycle[echo % len(cycle)]:
            m = evolve_segment(m, ensemble, seg)
        mean = weight @ m  # coherent (weighted) magnetization vector
        coherent[echo] = np.dot(mean, ax_x) + 1j * np.dot(mean, ax_y)

    echo_times = (np.arange(num_echoes, dtype=np.float64) + 1.0) * spacing
    if np.isfinite(t2_seconds) and t2_seconds > 0:
        envelope = np.exp(-echo_times / float(t2_seconds))
    else:
        envelope = np.ones_like(echo_times)

    return FieldReversalResult(
        echo_times=echo_times,
        echo_amplitudes=coherent * envelope,
        coherent_amplitudes=coherent,
        relaxation_envelope=envelope,
        detection_axes=(ax_x, ax_y),
        magnetization=m if return_magnetization else None,
    )
