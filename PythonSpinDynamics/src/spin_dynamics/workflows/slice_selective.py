"""Slice-selective excitation for 3D imaging.

A slice-selective pulse plays a shaped (windowed-sinc) RF field while a constant
gradient is applied along the slice axis, so the on-resonance condition --- and
therefore the excited band --- is localized to a plane. This is the building
block 3D imaging needs and the physically faithful counterpart to
``imaging_frequency.imaging_slice_sensitivity``, which only approximated the
sensitive region with a hard (rectangular) pulse and no gradient during RF.

The pulse is expressed on the moving-isochromat engine
(``spin_dynamics.sequences.motion``): the shaped RF becomes a train of short
constant-amplitude :class:`MotionSequenceStep` substeps, each carrying the slice
gradient, followed by a refocusing (rephasing) gradient lobe that undoes the
through-slice dephasing accrued during the second half of the pulse.

Because the engine couples an applied gradient to a spin as ``positions @
gradient``, the slice axis is simply whichever spatial axis the gradient is
placed on; the same builder serves a 2-axis ``(x, z)`` plane today and a 3-axis
volume once the engine is widened.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.motion import MotionFieldMaps2D, ParticleEnsemble
from spin_dynamics.sequences.motion import MotionSequenceStep, run_motion_sequence

Window = str  # "hamming" | "hanning" | "none"


def _windowed_sinc(num_samples: int, time_bandwidth: float, window: Window) -> np.ndarray:
    """Return a symmetric windowed-sinc envelope of unit peak.

    ``time_bandwidth`` (the time-bandwidth product) sets how many sinc lobes fall
    inside the pulse: the normalized argument spans ``[-time_bandwidth,
    time_bandwidth]`` so the envelope has ``2 * time_bandwidth`` zero crossings.
    """

    n = int(num_samples)
    if n < 1:
        raise ValueError("num_substeps must be positive")
    if time_bandwidth <= 0.0:
        raise ValueError("time_bandwidth must be positive")
    if n == 1:
        unit = np.array([0.0])
    else:
        unit = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)  # in [-1, 1]
    envelope = np.sinc(float(time_bandwidth) * unit)  # np.sinc(x) = sin(pi x)/(pi x)
    if window == "hamming":
        envelope = envelope * (0.54 + 0.46 * np.cos(np.pi * unit))
    elif window == "hanning":
        envelope = envelope * (0.5 + 0.5 * np.cos(np.pi * unit))
    elif window != "none":
        raise ValueError("window must be 'hamming', 'hanning', or 'none'")
    return envelope


def make_slice_selective_excitation(
    *,
    duration: float,
    slice_gradient: float,
    flip_angle: float = np.pi / 2,
    slice_axis: int = 0,
    ndim: int = 2,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    phase: float = 0.0,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
) -> tuple[MotionSequenceStep, ...]:
    """Build a slice-selective excitation as a tuple of motion-sequence steps.

    The shaped RF is sampled into ``num_substeps`` constant-amplitude intervals
    of width ``duration / num_substeps``, each applied with the slice gradient on
    ``slice_axis``. The amplitude envelope is a windowed sinc scaled so the
    on-resonance flip equals ``flip_angle``. When ``rephase`` is set, a trailing
    gradient lobe of duration ``rephase_fraction * duration`` and reversed
    polarity rephases the slice (half-area refocusing for a symmetric pulse).

    ``slice_gradient`` is the gradient amplitude in engine units (rad/s per unit
    position along ``slice_axis``); the excited slice is centered where
    ``slice_gradient * position == 0``.
    """

    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if not np.isfinite(slice_gradient) or slice_gradient == 0.0:
        raise ValueError("slice_gradient must be a nonzero finite value")
    if not 0 <= int(slice_axis) < int(ndim):
        raise ValueError("slice_axis must be in range(ndim)")
    if rephase_fraction < 0.0:
        raise ValueError("rephase_fraction must be non-negative")

    envelope = _windowed_sinc(num_substeps, time_bandwidth, window)
    dt = float(duration) / int(num_substeps)
    envelope_sum = float(np.sum(envelope))
    if envelope_sum == 0.0:
        raise ValueError("degenerate RF envelope (zero area)")
    # On resonance every substep rotates about the same axis, so the net flip is
    # the sum of per-substep angles: scale the envelope to hit ``flip_angle``.
    scale = float(flip_angle) / (envelope_sum * dt)

    def _grad(value: float) -> tuple[float, ...]:
        vector = [0.0] * int(ndim)
        vector[int(slice_axis)] = float(value)
        return tuple(vector)

    grad_on = _grad(slice_gradient)
    steps: list[MotionSequenceStep] = [
        MotionSequenceStep(
            duration=dt,
            gradient=grad_on,
            rf_amplitude=scale * float(env),
            rf_phase=float(phase),
            substeps=1,
            label=f"slice_rf_{i}",
        )
        for i, env in enumerate(envelope)
    ]
    if rephase and rephase_fraction > 0.0:
        steps.append(
            MotionSequenceStep(
                duration=rephase_fraction * float(duration),
                gradient=_grad(-slice_gradient),
                rf_amplitude=0.0,
                substeps=max(1, int(num_substeps * rephase_fraction)),
                label="slice_rephase",
            )
        )
    return tuple(steps)


def _excite_line(
    positions_axis: np.ndarray,
    *,
    duration: float,
    slice_gradient: float,
    flip_angle: float,
    time_bandwidth: float,
    num_substeps: int,
    window: Window,
    rephase: bool,
    rephase_fraction: float,
) -> ParticleEnsemble:
    """Excite a line of static, on-resonance spins and return the final ensemble.

    Spins are placed along the slice axis at ``positions_axis`` (which must be
    strictly increasing). The slice is centered at position ``0``; to center it
    elsewhere, pass positions shifted by ``-center``.
    """

    positions_axis = np.asarray(positions_axis, dtype=np.float64)
    span = float(np.max(np.abs(positions_axis))) or 1.0
    in_plane = np.array([-span, span], dtype=np.float64)
    shape = (positions_axis.size, in_plane.size)
    fields = MotionFieldMaps2D(
        x_axis=positions_axis,
        z_axis=in_plane,
        b0_map=np.zeros(shape),
        b1_tx_map=np.ones(shape),
        b1_rx_map=np.ones(shape),
    )
    magnetization = np.zeros((3, positions_axis.size), dtype=np.complex128)
    magnetization[0, :] = 1.0
    ensemble = ParticleEnsemble(
        positions=np.column_stack([positions_axis, np.zeros_like(positions_axis)]),
        magnetization=magnetization,
        weights=np.ones(positions_axis.size),
        diffusion_coefficient=np.zeros(positions_axis.size),
    )
    steps = make_slice_selective_excitation(
        duration=duration,
        slice_gradient=slice_gradient,
        flip_angle=flip_angle,
        slice_axis=0,
        ndim=2,
        time_bandwidth=time_bandwidth,
        num_substeps=num_substeps,
        window=window,
        rephase=rephase,
        rephase_fraction=rephase_fraction,
    )
    return run_motion_sequence(ensemble, fields, steps, boundary="clip").final_ensemble


def slice_excitation_weights(
    positions: np.ndarray,
    *,
    duration: float,
    slice_gradient: float,
    center: float = 0.0,
    flip_angle: float = np.pi / 2,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
) -> np.ndarray:
    """Return the complex transverse weight a slice pulse imprints at ``positions``.

    The slice is centered at ``center``; the returned array is the transverse
    magnetization (``M_-``) excited at each through-plane position, i.e. the
    weight that position contributes to a slice image. ``positions`` need not be
    sorted; the result preserves the input ordering.
    """

    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    shifted = positions - float(center)
    order = np.argsort(shifted, kind="stable")
    sorted_axis = shifted[order]
    # The field-map axis must be strictly increasing; nudge any exact ties.
    if np.any(np.diff(sorted_axis) <= 0.0):
        eps = 1e-9 * (float(np.ptp(sorted_axis)) or 1.0)
        sorted_axis = sorted_axis + eps * np.arange(sorted_axis.size)
    final = _excite_line(
        sorted_axis,
        duration=duration,
        slice_gradient=slice_gradient,
        flip_angle=flip_angle,
        time_bandwidth=time_bandwidth,
        num_substeps=num_substeps,
        window=window,
        rephase=rephase,
        rephase_fraction=rephase_fraction,
    )
    transverse_sorted = final.magnetization[1, :]
    weights = np.empty(positions.size, dtype=np.complex128)
    weights[order] = transverse_sorted
    return weights


def slice_profile_table(
    *,
    slice_gradient: float,
    off_resonance_max: float,
    duration: float,
    flip_angle: float = np.pi / 2,
    num: int = 1201,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate the excited transverse magnetization versus off-resonance.

    Returns ``(off_resonance, transverse)`` where ``off_resonance`` spans
    ``[-off_resonance_max, off_resonance_max]`` (rad/s) and ``transverse`` is the
    complex ``M_-`` a spin at that total off-resonance receives from the slice
    pulse. Evaluating this table at a voxel's *local* off-resonance
    ``slice_gradient * (y - center) + b0`` gives the slice weight there, so a
    nonuniform B0 simply shifts and bends the excited band -- the table is built
    once and reused across all voxels and slices.
    """

    if off_resonance_max <= 0.0:
        raise ValueError("off_resonance_max must be positive")
    if num < 2:
        raise ValueError("num must be at least 2")
    off = np.linspace(-float(off_resonance_max), float(off_resonance_max), int(num))
    positions = off / abs(float(slice_gradient))
    final = _excite_line(
        positions,
        duration=duration,
        slice_gradient=abs(float(slice_gradient)),
        flip_angle=flip_angle,
        time_bandwidth=time_bandwidth,
        num_substeps=num_substeps,
        window=window,
        rephase=rephase,
        rephase_fraction=rephase_fraction,
    )
    return off, final.magnetization[1, :]


@dataclass(frozen=True)
class SliceProfileResult:
    """Through-slice magnetization profile of a slice-selective pulse."""

    slice_positions: np.ndarray  # positions along the slice axis
    transverse: np.ndarray  # complex transverse magnetization (M-) per position
    profile: np.ndarray  # |transverse|, the excited slice profile
    longitudinal: np.ndarray  # residual Mz per position
    slice_gradient: float
    flip_angle: float


def simulate_slice_profile(
    *,
    duration: float,
    slice_gradient: float,
    flip_angle: float = np.pi / 2,
    time_bandwidth: float = 4.0,
    num_substeps: int = 48,
    window: Window = "hamming",
    rephase: bool = True,
    rephase_fraction: float = 0.5,
    extent: float | None = None,
    num_positions: int = 201,
) -> SliceProfileResult:
    """Excite a uniform line of spins and return the through-slice profile.

    A row of static, on-resonance spins is laid along the slice axis and the
    slice-selective pulse of :func:`make_slice_selective_excitation` is applied.
    The returned ``profile`` is ``|M_xy|`` versus position: a localized band for
    a selective pulse, sharply bounded compared with a hard pulse.
    """

    if num_positions < 2:
        raise ValueError("num_positions must be at least 2")
    if extent is None:
        # Default to a few slice widths: bandwidth ~ time_bandwidth / duration,
        # half-width in position ~ pi * bandwidth / |gradient|.
        bandwidth = float(time_bandwidth) / float(duration)
        half_width = np.pi * bandwidth / abs(float(slice_gradient))
        extent = 6.0 * half_width
    positions_axis = np.linspace(-float(extent), float(extent), int(num_positions))
    final = _excite_line(
        positions_axis,
        duration=duration,
        slice_gradient=slice_gradient,
        flip_angle=flip_angle,
        time_bandwidth=time_bandwidth,
        num_substeps=num_substeps,
        window=window,
        rephase=rephase,
        rephase_fraction=rephase_fraction,
    )
    transverse = final.magnetization[1, :]
    longitudinal = np.real(final.magnetization[0, :])
    return SliceProfileResult(
        slice_positions=positions_axis,
        transverse=transverse,
        profile=np.abs(transverse),
        longitudinal=longitudinal,
        slice_gradient=float(slice_gradient),
        flip_angle=float(flip_angle),
    )
