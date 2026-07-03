"""Balanced SSFP (bSSFP / TrueFISP / FIESTA) steady-state imaging.

Balanced SSFP is the workhorse steady-state gradient-echo sequence: a train of
small-flip RF pulses at a short, fixed ``TR`` with *fully balanced* gradients
(every gradient moment -- readout and phase encode -- is rewound to zero each
``TR``). Its appeal is SNR: on resonance the steady-state signal is high,
``~ M0/2 * sqrt(T2/T1)`` at the optimal flip, so it is a favourite at low field
where SNR is scarce. Its liability is the flip side of "balanced": because only
the *gradient* phase is refocused each ``TR`` and the *B0* phase is not, the
steady state depends on the per-``TR`` off-resonance precession, producing an
off-resonance response with periodic nulls (the bSSFP "dark bands") spaced
``1/TR`` in frequency. In an inhomogeneous B0 those bands fall across the object
as signal voids, and B1 (flip-angle) inhomogeneity shades and reshapes the
response -- exactly the practical concern this module is built to explore.

Everything here runs on the same moving-isochromat engine and field maps as the
spin-echo imagers in :mod:`spin_dynamics.workflows.imaging_frequency`
(``run_spin_warp_imaging`` / ``run_rare_imaging``): one static isochromat per
voxel, rectangular RF pulses (with local B1 transmit), gradient waveforms folded
into the off-resonance, T1/T2 relaxation, and the same centred-k-space
``reconstruct_image_from_kspace``. bSSFP differs only in the *schedule*: a
continuous steady-state train of balanced TRs with an alternating (phase-cycled)
RF phase, one phase-encode line acquired per TR.

* ``run_bssfp_imaging`` -- image a phantom with a balanced SSFP readout.
* ``bssfp_steady_state_signal`` / ``bssfp_transient`` -- the gradient-free
  off-resonance response and its approach to steady state, using the same engine
  primitives, for characterising banding without imaging.
* ``bssfp_optimal_flip_deg`` / ``bssfp_band_spacing_hz`` -- the standard
  design formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.motion import (
    ParticleEnsemble,
    apply_free_precession,
    apply_rf_rotation,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import MotionSequenceStep, run_motion_sequence
from spin_dynamics.workflows.imaging import reconstruct_image_from_kspace
from spin_dynamics.workflows.imaging_frequency import _resolve_maps


@dataclass(frozen=True)
class BSSFPImagingResult:
    """Result of a balanced SSFP imaging simulation.

    The array interface matches the spin-echo imagers: ``image`` and
    ``magnitude`` are ``(px, pz, 1)`` and ``reconstruct`` returns the complex
    ``(px, pz)`` image, so the same plotting helpers apply.
    """

    kspace: np.ndarray  # (px, pz, 1) centered k-space
    image: np.ndarray  # (px, pz, 1) complex reconstruction
    magnitude: np.ndarray  # (px, pz, 1)
    rho: np.ndarray
    fov: tuple[float, float]
    tr: float
    te: float
    flip_angle_deg: float
    phase_increment_deg: float
    num_dummy_tr: int
    band_spacing_hz: float
    num_offsets: int
    offset_spread: float
    method: str = "bssfp"

    def reconstruct(self) -> np.ndarray:
        """Return the complex image (alias for the stored reconstruction)."""

        return self.image[:, :, 0]


def bssfp_optimal_flip_deg(t1: float, t2: float) -> float:
    """Return the flip angle that maximizes on-resonance bSSFP signal.

    ``cos(alpha*) = (T1/T2 - 1) / (T1/T2 + 1)``; at this flip the band-centre
    signal is ``~ M0/2 * sqrt(T2/T1)``.
    """

    if t1 <= 0.0 or t2 <= 0.0:
        raise ValueError("t1 and t2 must be positive")
    ratio = float(t1) / float(t2)
    return float(np.degrees(np.arccos((ratio - 1.0) / (ratio + 1.0))))


def bssfp_band_spacing_hz(tr: float) -> float:
    """Return the off-resonance spacing ``1/TR`` between bSSFP dark bands."""

    if tr <= 0.0:
        raise ValueError("tr must be positive")
    return 1.0 / float(tr)


def _tr_free_split(tr: float, excitation_duration: float, te: float) -> tuple[float, float]:
    """Split a TR into (pre-echo, post-echo) free-precession durations."""

    free_pre = te - 0.5 * excitation_duration
    free_post = tr - excitation_duration - free_pre
    if free_pre < 0.0 or free_post < 0.0:
        raise ValueError("te and excitation_duration must fit inside tr")
    return free_pre, free_post


def bssfp_transient(
    off_resonance_hz: float | np.ndarray,
    *,
    flip_angle_deg: float,
    tr: float,
    t1: float,
    t2: float,
    num_tr: int,
    te: float | None = None,
    excitation_duration: float = 0.5e-3,
    phase_increment_deg: float = 180.0,
    b1_scale: float = 1.0,
    catalyze: bool = True,
    m0: float = 1.0,
) -> np.ndarray:
    """Return the transverse magnetization at TE after each of ``num_tr`` TRs.

    Uses the engine primitives ``apply_rf_rotation`` / ``apply_free_precession``
    (no gradients), so it is the gradient-free counterpart of the imaging path.
    ``off_resonance_hz`` may be an array; the returned array has shape
    ``(num_tr, ...)`` with the trailing axes matching the input, tracing the
    approach to steady state (the smooth T1 build-up plus the oscillatory
    transient that the ``catalyze`` half-angle preparation suppresses).
    """

    if num_tr <= 0:
        raise ValueError("num_tr must be positive")
    off = np.atleast_1d(np.asarray(off_resonance_hz, dtype=np.float64))
    shape = off.shape
    omega = 2.0 * np.pi * off.reshape(-1)
    n = omega.size
    te_val = 0.5 * tr if te is None else float(te)
    free_pre, free_post = _tr_free_split(tr, excitation_duration, te_val)
    alpha = np.deg2rad(flip_angle_deg) * float(b1_scale)
    dphi = np.deg2rad(phase_increment_deg)
    w1 = alpha / excitation_duration

    ensemble = ParticleEnsemble(
        positions=np.zeros((n, 2), dtype=np.float64),
        magnetization=np.array(
            [np.full(n, m0), np.zeros(n), np.zeros(n)], dtype=np.complex128
        ),
        weights=np.ones(n, dtype=np.float64),
        diffusion_coefficient=np.zeros(n, dtype=np.float64),
    )
    if catalyze:
        # Deimling/Heid alpha/2 -- TR/2 preparation: start near the steady-state
        # cone so the oscillatory transient is suppressed.
        ensemble = apply_rf_rotation(ensemble, excitation_duration, 0.0, 0.5 * w1, omega)
        ensemble = apply_free_precession(ensemble, 0.5 * tr, omega, t1=t1, t2=t2, mth=m0)

    out = np.empty((int(num_tr), n), dtype=np.complex128)
    for k in range(int(num_tr)):
        ensemble = apply_rf_rotation(ensemble, excitation_duration, float(k * dphi), w1, omega)
        ensemble = apply_free_precession(ensemble, free_pre, omega, t1=t1, t2=t2, mth=m0)
        out[k, :] = ensemble.magnetization[1, :]
        ensemble = apply_free_precession(ensemble, free_post, omega, t1=t1, t2=t2, mth=m0)
    return out.reshape((int(num_tr), *shape))


def bssfp_steady_state_signal(
    off_resonance_hz: float | np.ndarray,
    *,
    flip_angle_deg: float,
    tr: float,
    t1: float,
    t2: float,
    te: float | None = None,
    excitation_duration: float = 0.5e-3,
    phase_increment_deg: float = 180.0,
    b1_scale: float = 1.0,
    num_tr: int | None = None,
    catalyze: bool = True,
    m0: float = 1.0,
) -> np.ndarray:
    """Return the steady-state transverse magnetization vs off-resonance.

    Runs the gradient-free bSSFP TR train (via :func:`bssfp_transient`) until the
    steady state and returns the last-TR transverse magnetization -- the bSSFP
    off-resonance response, whose ``1/TR``-spaced nulls are the dark bands.
    ``num_tr`` defaults to a few T1/TR so the steady state is reached.
    """

    if num_tr is None:
        # bSSFP approaches steady state on a ~T1 time constant, but the slowest
        # mode (near T1 ~ T2) needs many T1/TR to settle, so use a generous
        # multiple; the gradient-free train is cheap even at thousands of TRs.
        num_tr = int(np.clip(15.0 * t1 / tr, 600, 10000))
    trace = bssfp_transient(
        off_resonance_hz,
        flip_angle_deg=flip_angle_deg,
        tr=tr,
        t1=t1,
        t2=t2,
        num_tr=num_tr,
        te=te,
        excitation_duration=excitation_duration,
        phase_increment_deg=phase_increment_deg,
        b1_scale=b1_scale,
        catalyze=catalyze,
        m0=m0,
    )
    return trace[-1]


def _bssfp_tr_steps(
    *,
    rf_phase: float,
    flip_amplitude: float,
    excitation_duration: float,
    phase_time: float,
    readout_time: float,
    moment_readout: float,
    moment_predephase: float,
    moment_rewind: float,
    moment_pe: float,
    px: int,
    acquire: bool,
    substeps: int,
    line: int,
) -> list[MotionSequenceStep]:
    """Build the four balanced steps of one bSSFP TR.

    RF pulse, then a combined x pre-dephase + z phase-encode, the readout (a
    frequency-encode line), and a combined x rewind + z phase-encode rewind that
    returns every gradient moment to zero -- the "balanced" condition.
    """

    return [
        MotionSequenceStep(
            duration=excitation_duration,
            rf_amplitude=flip_amplitude,
            rf_phase=rf_phase,
            substeps=max(1, substeps),
            label="rf",
        ),
        MotionSequenceStep(
            duration=phase_time,
            gradient=(moment_predephase, moment_pe),
            substeps=substeps,
            label="dephase_encode",
        ),
        MotionSequenceStep(
            duration=readout_time,
            gradient=(moment_readout, 0.0),
            acquire=acquire,
            num_samples=px if acquire else 0,
            substeps=substeps,
            label=(f"readout_{line}" if acquire else "readout_dummy"),
        ),
        MotionSequenceStep(
            duration=phase_time,
            gradient=(moment_rewind, -moment_pe),
            substeps=substeps,
            label="rewind",
        ),
    ]


def run_bssfp_imaging(
    rho,
    *,
    t1_map=None,
    t2_map=None,
    b0_map=None,
    b1_tx_map=None,
    b1_rx_map=None,
    fov: tuple[float, float] = (0.02, 0.02),
    flip_angle_deg: float = 40.0,
    excitation_duration: float = 0.5e-3,
    readout_time: float = 2.0e-3,
    phase_time: float = 0.6e-3,
    phase_increment_deg: float = 180.0,
    num_dummy_tr: int = 60,
    catalyze: bool = True,
    num_offsets: int = 1,
    offset_spread: float = 0.0,
    substeps_per_interval: int = 1,
) -> BSSFPImagingResult:
    """Simulate a balanced SSFP image of a phantom in given B0/B1 fields.

    ``rho`` may be a 2D spin-density array (with optional map keywords) or an
    ``ImagingFieldMaps`` container shared with the other imagers. Readout is
    along x (frequency encode) and the phase encode is along z. The sequence is a
    single continuous steady-state train: ``num_dummy_tr`` balanced dummy TRs (an
    optional ``alpha/2`` catalyzation first) drive the magnetization to steady
    state, then one balanced TR per phase-encode line acquires a k-space row
    while the steady state is maintained. The RF phase advances by
    ``phase_increment_deg`` each TR (phase cycling); ``180`` centres the passband
    on resonance with nulls at ``+/- 1/(2 TR)``.

    ``b0_map`` is an absolute off-resonance map in rad/s (the same convention as
    ``run_spin_warp_imaging``); it is what produces banding. ``b1_tx_map`` scales
    the local flip angle (a shading/reshaping of the response) and ``b1_rx_map``
    the receive weight. ``num_offsets`` (> 1) averages that many isochromats over
    ``+/- offset_spread`` (rad/s) per voxel to model an unresolved sub-voxel B0
    spread, as in the spin-echo imagers.
    """

    rho_arr, t1_arr, t2_arr, b0_arr, b1_tx_arr, b1_rx_arr = _resolve_maps(
        rho, t1_map, t2_map, b0_map, b1_tx_map, b1_rx_map
    )
    px, pz = rho_arr.shape
    if min(px, pz) < 2:
        raise ValueError("rho must have at least 2x2 voxels")
    fov_x, fov_z = float(fov[0]), float(fov[1])
    if fov_x <= 0.0 or fov_z <= 0.0:
        raise ValueError("fov entries must be positive")
    if readout_time <= 0.0 or phase_time <= 0.0 or excitation_duration <= 0.0:
        raise ValueError("readout_time, phase_time, excitation_duration must be positive")
    if num_dummy_tr < 0:
        raise ValueError("num_dummy_tr must be non-negative")
    if num_offsets < 1:
        raise ValueError("num_offsets must be at least 1")
    if offset_spread < 0.0:
        raise ValueError("offset_spread must be non-negative")

    tr = excitation_duration + 2.0 * phase_time + readout_time
    te = 0.5 * tr
    flip_amplitude = np.deg2rad(flip_angle_deg) / excitation_duration
    dphi = np.deg2rad(phase_increment_deg)
    sub = int(substeps_per_interval)

    # Centred-k-space geometry, matching run_rare_imaging / reconstruct_*.
    x_axis = (np.arange(px) - px // 2) * (fov_x / px)
    z_axis = (np.arange(pz) - pz // 2) * (fov_z / pz)
    dk_x = 2.0 * np.pi / fov_x
    dk_z = 2.0 * np.pi / fov_z
    moment_readout = px * dk_x / readout_time
    moment_predephase = -(px // 2 + 1) * dk_x / phase_time
    moment_rewind = -(px - (px // 2 + 1)) * dk_x / phase_time

    ensemble = initialize_ensemble_from_density(
        rho_arr, x_axis, z_axis, walkers_per_cell=1, diffusion_coefficient=0.0
    )
    t1_particles = t1_arr.reshape(-1)
    t2_particles = t2_arr.reshape(-1)

    lines = list(range(pz))
    # One continuous steady-state train: catalyzation, dummy TRs, then one TR per
    # phase-encode line. The RF phase index advances every TR (including dummies
    # and the half-angle prep) so phase cycling is continuous.
    pulse_index = 0
    steps: list[MotionSequenceStep] = []
    if catalyze:
        steps.append(
            MotionSequenceStep(
                duration=excitation_duration,
                rf_amplitude=0.5 * flip_amplitude,
                rf_phase=0.0,
                substeps=max(1, sub),
                label="catalyze_half",
            )
        )
        steps.append(
            MotionSequenceStep(duration=0.5 * tr, substeps=sub, label="catalyze_delay")
        )
    for _ in range(int(num_dummy_tr)):
        steps.extend(
            _bssfp_tr_steps(
                rf_phase=float(pulse_index * dphi),
                flip_amplitude=flip_amplitude,
                excitation_duration=excitation_duration,
                phase_time=phase_time,
                readout_time=readout_time,
                moment_readout=moment_readout,
                moment_predephase=moment_predephase,
                moment_rewind=moment_rewind,
                moment_pe=0.0,
                px=px,
                acquire=False,
                substeps=sub,
                line=-1,
            )
        )
        pulse_index += 1
    acquire_order: list[int] = []
    acquire_phase: list[float] = []
    for line in lines:
        moment_pe = (line - pz // 2) * dk_z / phase_time
        rf_phase = float(pulse_index * dphi)
        steps.extend(
            _bssfp_tr_steps(
                rf_phase=rf_phase,
                flip_amplitude=flip_amplitude,
                excitation_duration=excitation_duration,
                phase_time=phase_time,
                readout_time=readout_time,
                moment_readout=moment_readout,
                moment_predephase=moment_predephase,
                moment_rewind=moment_rewind,
                moment_pe=moment_pe,
                px=px,
                acquire=True,
                substeps=sub,
                line=line,
            )
        )
        acquire_order.append(line)
        acquire_phase.append(rf_phase)
        pulse_index += 1

    offsets = (
        np.array([0.0])
        if num_offsets == 1
        else np.linspace(-offset_spread, offset_spread, num_offsets)
    )
    kspace = np.zeros((px, pz), dtype=np.complex128)
    for offset in offsets:
        fields = make_motion_field_maps_2d(
            x_axis,
            z_axis,
            b0_map=b0_arr + float(offset),
            b1_tx_map=b1_tx_arr,
            b1_rx_map=b1_rx_arr,
        )
        result = run_motion_sequence(
            ensemble,
            fields,
            steps,
            t1=t1_particles,
            t2=t2_particles,
            mth=1.0,
            default_substeps=max(1, sub),
        )
        signal = result.signal
        for acquired_index, (line, phase) in enumerate(zip(acquire_order, acquire_phase)):
            # Demodulate by the transmit phase of this TR: the receiver phase must
            # follow the RF phase cycling, otherwise the alternating pulse phase
            # imprints a (-1)^line modulation on k-space that shifts the image by
            # half the FOV along the phase-encode axis.
            samples = signal[acquired_index * px : (acquired_index + 1) * px]
            kspace[:, line] += np.exp(-1j * phase) * samples
    kspace /= float(offsets.size)

    kspace3 = kspace[:, :, np.newaxis]
    image = reconstruct_image_from_kspace(kspace3, 0)[:, :, np.newaxis]
    return BSSFPImagingResult(
        kspace=kspace3,
        image=image,
        magnitude=np.abs(image),
        rho=rho_arr,
        fov=(fov_x, fov_z),
        tr=tr,
        te=te,
        flip_angle_deg=float(flip_angle_deg),
        phase_increment_deg=float(phase_increment_deg),
        num_dummy_tr=int(num_dummy_tr),
        band_spacing_hz=bssfp_band_spacing_hz(tr),
        num_offsets=int(num_offsets),
        offset_spread=float(offset_spread),
    )
