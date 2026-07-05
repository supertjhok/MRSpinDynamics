"""Nonresonant sequence builders: basic field reversal and CSAR refocusing units.

Each builder returns one *refocusing unit* -- a list of
:class:`~spin_dynamics.nonresonant.field_reversal.FieldSegment` spanning one echo
period -- which :func:`~spin_dynamics.nonresonant.field_reversal.simulate_field_reversal_echoes`
repeats. Fields are the nominal coil-A / coil-B magnitudes (tesla); adiabatic-rotation
segments smoothly interpolate the coil fields between endpoints (rotating the field
direction slowly compared with the Larmor rate), and reversal segments flip a coil's
field -- instantly (ideal sudden switch) or over a finite ``tau_rev`` (the dominant
imperfection in Brill 2002).
"""

from __future__ import annotations

import numpy as np

from spin_dynamics.nonresonant.field_reversal import (
    FieldSegment,
    IsochromatEnsemble,
    NonresonantFieldModel,
    evolve_segment,
)


def _hold(a_tesla: float, b_tesla: float, duration: float) -> FieldSegment:
    return FieldSegment(np.array([a_tesla]), np.array([b_tesla]),
                        np.array([duration]), label="hold")


def _adiabatic(a0, b0, a1, b1, duration, n_steps) -> FieldSegment:
    """Adiabatic rotation: smoothly ramp the coil fields ``(a,b)`` between endpoints.

    A raised-cosine ramp of the coil currents rotates the field direction slowly; the
    magnetization follows adiabatically while accumulating the dynamical and geometric
    phase the CSAR refocusing relies on.
    """

    steps = max(int(n_steps), 1)
    s = (np.arange(steps) + 0.5) / steps
    w = 0.5 * (1.0 - np.cos(np.pi * s))  # smooth 0 -> 1
    a = a0 + (a1 - a0) * w
    b = b0 + (b1 - b0) * w
    dt = np.full(steps, duration / steps)
    return FieldSegment(a, b, dt, label="adiabatic")


def _reverse(a0, b0, a1, b1, tau_rev, n_steps) -> FieldSegment:
    """A sudden field reversal ramping ``(a,b)`` from one point to another.

    ``tau_rev == 0`` is an ideal instantaneous switch (the magnetization does not move
    during it); a finite ``tau_rev`` linearly ramps through the near-zero-field region
    where only the background survives -- the imperfection that seeds the odd-even
    (Bessel-like) echo modulation.
    """

    if tau_rev <= 0:
        return FieldSegment(np.array([a1]), np.array([b1]), np.array([0.0]),
                            label="reverse")
    steps = max(int(n_steps), 1)
    s = (np.arange(steps) + 0.5) / steps
    a = a0 + (a1 - a0) * s
    b = b0 + (b1 - b0) * s
    dt = np.full(steps, tau_rev / steps)
    return FieldSegment(a, b, dt, label="reverse")


def basic_reversal_sequence(
    model: NonresonantFieldModel,
    *,
    echo_spacing_seconds: float,
    tau_rev_seconds: float = 0.0,
    reversal_steps: int = 16,
) -> list[FieldSegment]:
    """The basic nonresonant sequence (Brill 2002 Fig. 1B): periodic ``B_B`` reversal.

    One echo period holds ``+B_B`` for half the period and ``-B_B`` for the other half
    (two reversals, returning to ``+B_B``). The reversal refocuses the coil-field
    magnitude dephasing into an echo, but the background field leaves a residual phase
    ``~2 gamma B_e t cos(alpha)`` each reversal that accumulates into the rapid decay
    the CSAR sequences overcome.
    """

    b0 = model.coil_b_tesla
    half = 0.5 * float(echo_spacing_seconds)
    return [
        _hold(0.0, +b0, half),
        _reverse(0.0, +b0, 0.0, -b0, tau_rev_seconds, reversal_steps),
        _hold(0.0, -b0, half),
        _reverse(0.0, -b0, 0.0, +b0, tau_rev_seconds, reversal_steps),
    ]


def csar_sequence(
    model: NonresonantFieldModel,
    *,
    echo_spacing_seconds: float,
    tau_rev_seconds: float = 0.0,
    free_fraction: float = 0.1,
    reversal_steps: int = 16,
    adiabatic_steps: int = 160,
    sense: int = 1,
) -> list[FieldSegment]:
    """A 90-degree CSAR refocusing unit (Brill 2002 Fig. 1D / Fig. 3A).

    One echo period is: a short free precession under ``B_B``; an adiabatic 90-degree
    rotation of the field to ``B_A``; a sudden reversal ``B_A -> -B_A``
    (``tau_rev_seconds`` = 0 ideal, > 0 the finite-reversal imperfection); an adiabatic
    90-degree rotation back to ``B_B``; and a closing free precession. To first order
    (ideal reversal, fully adiabatic) the echo-to-echo propagator is a ``pi`` rotation
    about an axis in the plane perpendicular to ``B_B`` -- refocusing every *second*
    echo (Fig. 1E). ``sense = -1`` routes the adiabatic rotations through ``-B_A``
    instead of ``+B_A``; alternating the sense builds the Fig. 3C supercycle
    (:func:`csar_supercycle_sequence`).
    """

    b0 = model.coil_b_tesla
    a0 = model.coil_a_tesla * int(np.sign(sense) or 1)
    period = float(echo_spacing_seconds)
    t_free = 0.5 * float(free_fraction) * period
    t_rot = 0.5 * (period - 2.0 * t_free - float(tau_rev_seconds))
    if t_rot <= 0:
        raise ValueError("echo_spacing too short for the requested free_fraction/tau_rev")
    return [
        _hold(0.0, b0, t_free),
        _adiabatic(0.0, b0, a0, 0.0, t_rot, adiabatic_steps),           # B_B -> B_A
        _reverse(a0, 0.0, -a0, 0.0, tau_rev_seconds, reversal_steps),   # reverse coil A
        _adiabatic(-a0, 0.0, 0.0, b0, t_rot, adiabatic_steps),          # -B_A -> B_B
        _hold(0.0, b0, t_free),
    ]


def csar_double_reversal_sequence(
    model: NonresonantFieldModel,
    *,
    echo_spacing_seconds: float,
    tau_rev_seconds: float = 0.0,
    free_fraction: float = 0.1,
    reversal_steps: int = 16,
    adiabatic_steps: int = 80,
) -> list[FieldSegment]:
    """A 2-pi CSAR refocusing unit (Brill 2002 Fig. 3B): two reversals per echo period.

    One echo period contains *two* sudden reversals (two back-to-back 90-degree CSAR
    sub-cycles at half the echo spacing), so the echo-to-echo propagator is (to first
    order) a ``2 pi`` rotation rather than ``pi``. A ``2 pi`` rotation is the identity,
    so *both* even and odd echoes refocus -- unlike the ``pi`` unit
    (:func:`csar_sequence`) which refocuses only even echoes. Because the observed
    magnetization then lies mainly perpendicular to the (near-identity) net-rotation
    axis, finite-reversal deviations from the perfect ``2 pi`` accumulate rapidly and
    the later echoes show an enhanced decay, exactly as the paper reports.

    Note: this reproduces the *effect* of Brill 2002 Fig. 3B (a ``2 pi`` echo-to-echo
    propagator refocusing both parities) using two 90-degree CSAR sub-cycles; the
    paper's specific implementation instead uses 180-degree adiabatic rotations, so the
    coil-current waveform differs while the refocusing behaviour matches.
    """

    half = csar_sequence(
        model, echo_spacing_seconds=0.5 * float(echo_spacing_seconds),
        tau_rev_seconds=tau_rev_seconds, free_fraction=free_fraction,
        reversal_steps=reversal_steps, adiabatic_steps=adiabatic_steps, sense=1)
    return [*half, *half]


def csar_supercycle_sequence(
    model: NonresonantFieldModel,
    *,
    echo_spacing_seconds: float,
    tau_rev_seconds: float = 0.0,
    senses: tuple[int, ...] = (1, 1, -1, -1),
    **kwargs,
) -> list[list[FieldSegment]]:
    """The Fig. 3C supercycle: 90-degree CSAR units with alternating rotation senses.

    Returns a *list of units* (a supercycle) to hand to
    :func:`~spin_dynamics.nonresonant.field_reversal.simulate_field_reversal_echoes`,
    which reads one echo per unit and cycles through them. Alternating the *sense* of
    the adiabatic rotations (which coil the field rotates through) over the four-unit
    ``+ + - -`` supercycle progressively compensates the finite-reversal imperfection --
    the nonresonant analogue of composite-pulse phase supercycles in resonant NMR --
    so both echo parities refocus with a much smaller initial decay rate than either
    the single ``pi`` (:func:`csar_sequence`) or the ``2 pi``
    (:func:`csar_double_reversal_sequence`) sequence -- the imperfection decay is
    suppressed until the residual is limited by the intrinsic T2.

    Note: this reproduces the *effect* of Brill 2002 Fig. 3C (a supercycle that
    dramatically slows the imperfection-driven decay) via a ``+ + - -`` cycle of the
    adiabatic-rotation sense; the paper's particular four-unit supercycle differs in
    detail, so the sustained-signal behaviour matches while the exact waveform does not.
    """

    return [
        csar_sequence(model, echo_spacing_seconds=echo_spacing_seconds,
                      tau_rev_seconds=tau_rev_seconds, sense=s, **kwargs)
        for s in senses
    ]


def effective_rotation(
    ensemble: IsochromatEnsemble, unit, isochromat_index: int = 0
) -> tuple[np.ndarray, float]:
    """Return the ``(axis, angle)`` of one isochromat's echo-to-echo net rotation.

    The paper's analysis tool: propagate the three Cartesian basis vectors of a single
    isochromat through one refocusing ``unit`` to build its net ``3x3`` rotation, then
    extract the rotation axis ``n`` and angle. The magnetization component along ``n``
    is invariant echo-to-echo (perfectly refocused, decaying only with T2); the
    powder-averaged fraction ``<(n.x)^2> ~ 1/2`` sets the refocused signal.
    """

    idx = int(isochromat_index)
    single = IsochromatEnsemble(
        coil_a_dir=ensemble.coil_a_dir[idx:idx + 1],
        coil_b_dir=ensemble.coil_b_dir[idx:idx + 1],
        coil_a_scale=ensemble.coil_a_scale[idx:idx + 1],
        coil_b_scale=ensemble.coil_b_scale[idx:idx + 1],
        background=ensemble.background,
        gamma_rad=ensemble.gamma_rad,
        weight=np.array([1.0]),
    )
    cols = []
    for basis in np.eye(3):
        m = basis.reshape(1, 3).copy()
        for seg in unit:
            m = evolve_segment(m, single, seg)
        cols.append(m.reshape(3))
    rot = np.column_stack(cols)
    angle = float(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0)))
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    norm = float(np.linalg.norm(axis))
    axis = np.array([0.0, 0.0, 1.0]) if norm < 1e-12 else axis / norm
    return axis, angle
