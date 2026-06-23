"""Diffusion-aware matched-probe CPMG workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from spin_dynamics.absolute_phase import (
    AbsolutePhaseMetadata,
    AbsolutePhaseSpec,
    PulseShape,
    PulseShapeLibrary,
    TAU,
    apply_absolute_phase_model,
    as_absolute_phase_spec,
    phase_bin_indices,
    quantize_phase_to_bins,
    wrap_phase,
)
from spin_dynamics.core.isochromats import check_rephasing
from spin_dynamics.core.kernels import (
    sim_spin_dynamics_arb10_diffusion,
    sim_spin_dynamics_arb10_diffusion_chunked,
)
from spin_dynamics.core.rotations import calc_rotation_matrix
from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.probes.matched import matching_network_design2
from spin_dynamics.probes.tuned import tuned_probe_rx_tf
from spin_dynamics.workflows.acquisition import _apply_receiver, _as_vector, _field
from spin_dynamics.workflows.cpmg import (
    _apply_longitudinal_phase_kick,
    _calc_matched_pulse_shape,
    _calc_tuned_pulse_shape,
    _echo_train_from_spectra,
    _maybe_refine_numpts,
)


VALIDATED_MATCHED_DIFFUSION_Q_MAX = 2000.0


@dataclass(frozen=True)
class MatchedDiffusionCPMGResult:
    """Matched-probe diffusion-aware finite CPMG result."""

    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    q_value: float
    diffusion_coefficient: float
    diffusion_time: float
    gradient: float
    dz: float
    probe: str
    absolute_phase: AbsolutePhaseMetadata | None = None


@dataclass(frozen=True)
class TunedDiffusionCPMGResult:
    """Tuned-probe diffusion-aware finite CPMG result."""

    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    q_value: float
    diffusion_coefficient: float
    diffusion_time: float
    gradient: float
    dz: float
    probe: str
    absolute_phase: AbsolutePhaseMetadata | None = None


@dataclass(frozen=True)
class MatchedDiffusionQSweepResult:
    """Q sweep result for matched-probe diffusion-aware CPMG."""

    values: np.ndarray
    value_label: str
    del_w: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    diffusion_coefficient: float
    diffusion_time: float
    gradient: float
    dz: float
    probe: str
    sweep: str
    absolute_phase: tuple[AbsolutePhaseMetadata | None, ...] = ()


def check_matched_diffusion_q_stability(
    q_value: float,
    *,
    action: str = "warn",
) -> bool:
    """Check the compact matched-diffusion Q validation boundary.

    Benchmarks of the current NumPy fixed-step matched transient solver remain
    finite through Q=2000 and become unstable at Q=2500 in the compact
    validation case. This helper exposes that solver-validation boundary without
    treating it as a physical limit.
    """

    if q_value <= VALIDATED_MATCHED_DIFFUSION_Q_MAX:
        return True
    message = (
        "matched diffusion CPMG is only solver-validated through "
        f"Q={VALIDATED_MATCHED_DIFFUSION_Q_MAX:g}; higher-Q cases may become "
        "non-finite with the current fixed-step matched transient solver"
    )
    if action == "ignore":
        return False
    if action == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return False
    if action == "raise":
        raise RuntimeError(message)
    raise ValueError("action must be 'ignore', 'warn', or 'raise'")


def calc_macq_matched_probe_relax_diffusion(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    *,
    apply_receiver: bool = True,
    num_workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffusion-aware matched-probe finite acquisition.

    This is an `arb10`-style Python analogue of MATLAB
    `calc_macq_diff/calc_macq_matched_probe_relax_diff*.m`. It expects
    precomputed pulse matrices in `pp.Rtot` and returns acquired spectra
    without the older acquisition-window convolution.
    """

    t_90 = float(_field(pp, "T_90"))
    params = {
        "tp": _field(pp, "tp"),
        "pul": _field(pp, "pul"),
        "amp": _field(pp, "amp"),
        "acq": _field(pp, "acq"),
        "grad": _field(pp, "grad"),
        "Rtot": _field(pp, "Rtot"),
        "del_w": _field(sp, "del_w"),
        "del_wg": _field(sp, "del_wg"),
        "w_1": _field(sp, "w_1"),
        "T1n": (np.pi / 2) * _as_vector(_field(sp, "T1"), np.float64) / t_90,
        "T2n": (np.pi / 2) * _as_vector(_field(sp, "T2"), np.float64) / t_90,
        "m0": _field(sp, "m0"),
        "mth": _field(sp, "mth"),
        "gamma": _field(sp, "gamma"),
        "gradient": _field(sp, "grad_physical"),
        "diffusion_coefficient": _field(sp, "D"),
        # Seconds per unit of normalized pulse time; lets the diffusion kernel
        # attenuate each free-precession interval on its own duration rather
        # than a single global encoding time.
        "time_scale": 2.0 * t_90 / np.pi,
    }
    if num_workers is None or int(num_workers) > 1:
        macq = sim_spin_dynamics_arb10_diffusion_chunked(params, num_workers=num_workers)
    else:
        macq = sim_spin_dynamics_arb10_diffusion(params)
    if not apply_receiver:
        return macq, macq
    return macq, _apply_receiver(macq, _field(sp, "tf2"), _field(sp, "w_1r"))


def calc_macq_tuned_probe_relax_diffusion(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    *,
    apply_receiver: bool = True,
    num_workers: int | None = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate diffusion-aware tuned-probe finite acquisition."""

    t_90 = float(_field(pp, "T_90"))
    params = {
        "tp": _field(pp, "tp"),
        "pul": _field(pp, "pul"),
        "amp": _field(pp, "amp"),
        "acq": _field(pp, "acq"),
        "grad": _field(pp, "grad"),
        "Rtot": _field(pp, "Rtot"),
        "del_w": _field(sp, "del_w"),
        "del_wg": _field(sp, "del_wg"),
        "w_1": _field(sp, "w_1"),
        "T1n": (np.pi / 2) * _as_vector(_field(sp, "T1"), np.float64) / t_90,
        "T2n": (np.pi / 2) * _as_vector(_field(sp, "T2"), np.float64) / t_90,
        "m0": _field(sp, "m0"),
        "mth": _field(sp, "mth"),
        "gamma": _field(sp, "gamma"),
        "gradient": _field(sp, "grad_physical"),
        "diffusion_coefficient": _field(sp, "D"),
        "time_scale": 2.0 * t_90 / np.pi,
    }
    if num_workers is None or int(num_workers) > 1:
        macq = sim_spin_dynamics_arb10_diffusion_chunked(params, num_workers=num_workers)
    else:
        macq = sim_spin_dynamics_arb10_diffusion(params)
    if not apply_receiver:
        return macq, macq
    return macq, _apply_receiver(macq, _field(sp, "tf"), _field(sp, "w_1r"))


def _pulse_shape(raw_shape: tuple[np.ndarray, np.ndarray, np.ndarray]) -> PulseShape:
    return PulseShape(
        duration=np.asarray(raw_shape[0], dtype=np.float64).reshape(-1),
        phase=np.asarray(raw_shape[1], dtype=np.float64).reshape(-1),
        amplitude=np.asarray(raw_shape[2], dtype=np.float64).reshape(-1),
    )


def _shape_tuple(shape: PulseShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return shape.duration, shape.phase, shape.amplitude


def _diffusion_absolute_phase_times(
    *,
    t90_seconds: float,
    t180_seconds: float,
    encoding_gap: float,
    tfp: float,
    echo_spacing_seconds: float,
    num_echoes: int,
) -> tuple[float, np.ndarray]:
    time_scale = 2.0 * float(t90_seconds) / np.pi
    correction_delay = -1.0 * time_scale
    encoding_start = (
        float(t90_seconds)
        + correction_delay
        + float(encoding_gap) * time_scale
    )
    first_refocus = (
        encoding_start
        + float(t180_seconds)
        + float(encoding_gap) * time_scale
        + float(tfp) * time_scale
    )
    train_starts = first_refocus + float(echo_spacing_seconds) * np.arange(
        int(num_echoes),
        dtype=np.float64,
    )
    return float(encoding_start), train_starts


def _matched_pi_shape(
    sp: Mapping[str, Any],
    pp: Mapping[str, Any] | Any,
    *,
    phase_rad: float,
    delay_seconds: float,
    segment_fraction: float = 1.0,
) -> PulseShape:
    return _pulse_shape(
        _calc_matched_pulse_shape(
            sp,
            pp,
            float(_field(pp, "T_180")),
            0.0,
            1.0,
            float(delay_seconds),
            psi=float(phase_rad),
            segment_fraction=float(segment_fraction),
        )[:3]
    )


def _tuned_pi_shape(
    sp: Mapping[str, Any],
    pp: Mapping[str, Any] | Any,
    *,
    phase_rad: float,
    delay_seconds: float,
) -> PulseShape:
    return _pulse_shape(
        _calc_tuned_pulse_shape(
            sp,
            pp,
            float(_field(pp, "T_180")),
            0.0,
            1.0,
            float(delay_seconds),
            psi=float(phase_rad),
        )
    )


def _build_tuned_diffusion_absolute_phase_matrices(
    *,
    del_w: np.ndarray,
    sp: Mapping[str, Any],
    pp: Mapping[str, Any] | Any,
    spec: AbsolutePhaseSpec | None,
    num_echoes: int,
    echo_spacing_seconds: float,
    encoding_gap: float,
    tfp: float,
    delay_seconds: float,
) -> tuple[
    list[Any],
    np.ndarray,
    np.ndarray,
    AbsolutePhaseMetadata | None,
]:
    t90 = float(_field(pp, "T_90"))
    t180 = float(_field(pp, "T_180"))
    if spec is None:
        exc_y = _calc_tuned_pulse_shape(
            sp,
            pp,
            t90,
            np.pi / 2,
            1.0,
            delay_seconds,
        )
        exc_minus_y = _calc_tuned_pulse_shape(
            sp,
            pp,
            t90,
            3 * np.pi / 2,
            1.0,
            delay_seconds,
        )
        ref_x = _calc_tuned_pulse_shape(
            sp,
            pp,
            t180,
            0.0,
            1.0,
            delay_seconds,
        )
        rtot = [
            calc_rotation_matrix(del_w, _field(sp, "w_1"), *exc_y),
            calc_rotation_matrix(del_w, _field(sp, "w_1"), *exc_minus_y),
            calc_rotation_matrix(del_w, _field(sp, "w_1"), *ref_x),
        ]
        penc = np.array([0, 3, 0], dtype=np.int64)
        pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
        return rtot, penc, pref, None

    encoding_start, refocus_starts = _diffusion_absolute_phase_times(
        t90_seconds=t90,
        t180_seconds=t180,
        encoding_gap=encoding_gap,
        tfp=tfp,
        echo_spacing_seconds=echo_spacing_seconds,
        num_echoes=int(num_echoes),
    )
    excitation_rotating = np.array([np.pi / 2, 3 * np.pi / 2], dtype=np.float64)
    excitation_abs = np.array(
        [spec.pulse_phase(0.0, float(phase)) for phase in excitation_rotating],
        dtype=np.float64,
    )
    excitation_carrier = excitation_abs - excitation_rotating
    exc_y = apply_absolute_phase_model(
        _pulse_shape(
            _calc_tuned_pulse_shape(
                sp,
                pp,
                t90,
                np.pi / 2,
                1.0,
                delay_seconds,
                psi=float(excitation_carrier[0]),
            )
        ),
        spec,
        float(excitation_abs[0]),
        "excitation",
    )
    exc_minus_y = apply_absolute_phase_model(
        _pulse_shape(
            _calc_tuned_pulse_shape(
                sp,
                pp,
                t90,
                3 * np.pi / 2,
                1.0,
                delay_seconds,
                psi=float(excitation_carrier[1]),
            )
        ),
        spec,
        float(excitation_abs[1]),
        "excitation",
    )
    rtot = [
        calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(exc_y)),
        calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(exc_minus_y)),
    ]

    encoding_abs = np.array([spec.pulse_phase(encoding_start, 0.0)], dtype=np.float64)
    refocus_abs = np.array(
        [spec.pulse_phase(float(start), 0.0) for start in refocus_starts],
        dtype=np.float64,
    )
    pi_abs = np.concatenate([encoding_abs, refocus_abs])
    pi_matrix_phases = np.asarray(
        quantize_phase_to_bins(pi_abs, spec.phase_bins),
        dtype=np.float64,
    )
    pi_bins = (
        np.asarray(phase_bin_indices(pi_abs, spec.phase_bins), dtype=np.int64)
        if spec.phase_bins is not None
        else None
    )
    matrix_by_phase: dict[int | float, int] = {}
    pi_matrix_indices: list[int] = []
    unique_phases: list[float] = []
    unique_shapes: list[PulseShape] = []
    for idx, matrix_phase in enumerate(pi_matrix_phases):
        if spec.phase_bins is not None:
            key: int | float = int(pi_bins[idx])
        else:
            key = round(float(matrix_phase) / TAU, 12)
        matrix_index = matrix_by_phase.get(key)
        if matrix_index is None:
            shape = apply_absolute_phase_model(
                _tuned_pi_shape(
                    sp,
                    pp,
                    phase_rad=float(matrix_phase),
                    delay_seconds=delay_seconds,
                ),
                spec,
                float(matrix_phase),
                "refocusing",
            )
            matrix = calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(shape))
            matrix = _apply_longitudinal_phase_kick(
                matrix,
                del_w=del_w,
                spec=spec,
                pp=pp,
                pulse_start_phase_rad=float(matrix_phase),
                pulse_kind="refocusing",
            )
            rtot.append(matrix)
            matrix_index = len(rtot)
            matrix_by_phase[key] = matrix_index
            unique_phases.append(float(matrix_phase))
            unique_shapes.append(shape)
        pi_matrix_indices.append(matrix_index)

    penc = np.array([0, pi_matrix_indices[0], 0], dtype=np.int64)
    pref = np.asarray(
        [value for matrix_index in pi_matrix_indices[1:] for value in (0, matrix_index, 0)],
        dtype=np.int64,
    )
    pulse_start = np.concatenate(
        [
            np.array([0.0, 0.0, encoding_start], dtype=np.float64),
            refocus_starts,
        ]
    )
    pulse_kind = ("excitation", "excitation", "diffusion_refocusing") + (
        "refocusing",
    ) * int(num_echoes)
    pulse_rotating = np.concatenate(
        [excitation_rotating, np.zeros(int(num_echoes) + 1, dtype=np.float64)]
    )
    pulse_absolute = np.concatenate([excitation_abs, encoding_abs, refocus_abs])
    pulse_matrix_phase = np.concatenate(
        [excitation_abs, np.asarray(pi_matrix_phases, dtype=np.float64)]
    )
    pulse_phase_bin = (
        np.asarray(phase_bin_indices(pulse_absolute, spec.phase_bins), dtype=np.int64)
        if spec.phase_bins is not None
        else None
    )
    library = PulseShapeLibrary(
        absolute_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        shapes={"refocusing": tuple(unique_shapes)},
    )
    metadata = AbsolutePhaseMetadata(
        rf_frequency_hz=float(spec.rf_frequency_hz),
        refocus_start_seconds=refocus_starts,
        refocus_absolute_phase_rad=refocus_abs,
        excitation_absolute_phase_rad=excitation_abs,
        echo_spacing_seconds=float(echo_spacing_seconds),
        delta_refocus_phase_rad=float(
            wrap_phase(spec.rf_angular_frequency_rad_s * float(echo_spacing_seconds))
        ),
        receiver_phase_rad=float(spec.receiver_phase_rad),
        transient_model=(
            type(spec.transient_model).__name__ if spec.transient_model else None
        ),
        pulse_matrix_count=len(rtot),
        pulse_start_seconds=pulse_start,
        pulse_kind=pulse_kind,
        pulse_rotating_phase_rad=pulse_rotating,
        pulse_absolute_phase_rad=pulse_absolute,
        pulse_phase_bin=pulse_phase_bin,
        pulse_matrix_phase_rad=pulse_matrix_phase,
        excitation_matrix_indices=np.array([1, 0], dtype=np.int64),
        encoding_absolute_phase_rad=encoding_abs,
        encoding_matrix_indices=penc.copy(),
        encoding_phase_bin=None if pi_bins is None else pi_bins[:1].copy(),
        encoding_matrix_phase_rad=pi_matrix_phases[:1].copy(),
        refocus_matrix_indices=pref.copy(),
        phase_bins=spec.phase_bins,
        refocus_phase_bin=None if pi_bins is None else pi_bins[1:].copy(),
        refocus_matrix_phase_rad=pi_matrix_phases[1:].copy(),
        unique_refocus_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        refocus_pulse_library=library,
    )
    return rtot, penc, pref, metadata


def _build_matched_diffusion_absolute_phase_matrices(
    *,
    del_w: np.ndarray,
    sp: Mapping[str, Any],
    pp: Mapping[str, Any] | Any,
    spec: AbsolutePhaseSpec | None,
    num_echoes: int,
    echo_spacing_seconds: float,
    encoding_gap: float,
    tfp: float,
    delay_seconds: float,
) -> tuple[
    list[Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    AbsolutePhaseMetadata | None,
    np.ndarray,
]:
    t90 = float(_field(pp, "T_90"))
    t180 = float(_field(pp, "T_180"))
    if spec is None:
        exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
            sp,
            pp,
            t90,
            np.pi / 2,
            1.0,
            delay_seconds,
        )
        exc_minus_y = _calc_matched_pulse_shape(
            sp,
            pp,
            t90,
            3 * np.pi / 2,
            1.0,
            delay_seconds,
        )[:3]
        ref_x = _calc_matched_pulse_shape(
            sp,
            pp,
            t180,
            0.0,
            1.0,
            delay_seconds,
        )[:3]
        rtot = [
            calc_rotation_matrix(del_w, _field(sp, "w_1"), exc_y_tp, exc_y_phi, exc_y_amp),
            calc_rotation_matrix(del_w, _field(sp, "w_1"), *exc_minus_y),
            calc_rotation_matrix(del_w, _field(sp, "w_1"), *ref_x),
        ]
        penc = np.array([0, 3, 0], dtype=np.int64)
        pref = np.tile(np.array([0, 3, 0], dtype=np.int64), int(num_echoes))
        return rtot, penc, pref, tf1, tf2, None, np.array([], dtype=np.float64)

    encoding_start, refocus_starts = _diffusion_absolute_phase_times(
        t90_seconds=t90,
        t180_seconds=t180,
        encoding_gap=encoding_gap,
        tfp=tfp,
        echo_spacing_seconds=echo_spacing_seconds,
        num_echoes=int(num_echoes),
    )
    excitation_rotating = np.array([np.pi / 2, 3 * np.pi / 2], dtype=np.float64)
    excitation_abs = np.array(
        [spec.pulse_phase(0.0, float(phase)) for phase in excitation_rotating],
        dtype=np.float64,
    )
    excitation_carrier = excitation_abs - excitation_rotating
    exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
        sp,
        pp,
        t90,
        np.pi / 2,
        1.0,
        delay_seconds,
        psi=float(excitation_carrier[0]),
    )
    exc_y = apply_absolute_phase_model(
        PulseShape(exc_y_tp, exc_y_phi, exc_y_amp),
        spec,
        float(excitation_abs[0]),
        "excitation",
    )
    exc_minus_y = apply_absolute_phase_model(
        _pulse_shape(
            _calc_matched_pulse_shape(
                sp,
                pp,
                t90,
                3 * np.pi / 2,
                1.0,
                delay_seconds,
                psi=float(excitation_carrier[1]),
            )[:3]
        ),
        spec,
        float(excitation_abs[1]),
        "excitation",
    )
    rtot = [
        calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(exc_y)),
        calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(exc_minus_y)),
    ]

    encoding_abs = np.array([spec.pulse_phase(encoding_start, 0.0)], dtype=np.float64)
    refocus_abs = np.array(
        [spec.pulse_phase(float(start), 0.0) for start in refocus_starts],
        dtype=np.float64,
    )
    pi_abs = np.concatenate([encoding_abs, refocus_abs])
    pi_matrix_phases = np.asarray(
        quantize_phase_to_bins(pi_abs, spec.phase_bins),
        dtype=np.float64,
    )
    pi_bins = (
        np.asarray(phase_bin_indices(pi_abs, spec.phase_bins), dtype=np.int64)
        if spec.phase_bins is not None
        else None
    )
    matrix_by_phase: dict[int | float, int] = {}
    pi_matrix_indices: list[int] = []
    unique_phases: list[float] = []
    unique_shapes: list[PulseShape] = []
    for idx, matrix_phase in enumerate(pi_matrix_phases):
        if spec.phase_bins is not None:
            key: int | float = int(pi_bins[idx])
        else:
            key = round(float(matrix_phase) / TAU, 12)
        matrix_index = matrix_by_phase.get(key)
        if matrix_index is None:
            shape = apply_absolute_phase_model(
                _matched_pi_shape(
                    sp,
                    pp,
                    phase_rad=float(matrix_phase),
                    delay_seconds=delay_seconds,
                    segment_fraction=0.5,
                ),
                spec,
                float(matrix_phase),
                "refocusing",
            )
            matrix = calc_rotation_matrix(del_w, _field(sp, "w_1"), *_shape_tuple(shape))
            matrix = _apply_longitudinal_phase_kick(
                matrix,
                del_w=del_w,
                spec=spec,
                pp=pp,
                pulse_start_phase_rad=float(matrix_phase),
                pulse_kind="refocusing",
            )
            rtot.append(matrix)
            matrix_index = len(rtot)
            matrix_by_phase[key] = matrix_index
            unique_phases.append(float(matrix_phase))
            unique_shapes.append(shape)
        pi_matrix_indices.append(matrix_index)

    penc = np.array([0, pi_matrix_indices[0], 0], dtype=np.int64)
    pref = np.asarray(
        [value for matrix_index in pi_matrix_indices[1:] for value in (0, matrix_index, 0)],
        dtype=np.int64,
    )
    pulse_start = np.concatenate(
        [
            np.array([0.0, 0.0, encoding_start], dtype=np.float64),
            refocus_starts,
        ]
    )
    pulse_kind = ("excitation", "excitation", "diffusion_refocusing") + (
        "refocusing",
    ) * int(num_echoes)
    pulse_rotating = np.concatenate(
        [
            excitation_rotating,
            np.zeros(int(num_echoes) + 1, dtype=np.float64),
        ]
    )
    pulse_absolute = np.concatenate([excitation_abs, encoding_abs, refocus_abs])
    pulse_matrix_phase = np.concatenate(
        [excitation_abs, np.asarray(pi_matrix_phases, dtype=np.float64)]
    )
    pulse_phase_bin = (
        np.asarray(
            phase_bin_indices(pulse_absolute, spec.phase_bins),
            dtype=np.int64,
        )
        if spec.phase_bins is not None
        else None
    )
    library = PulseShapeLibrary(
        absolute_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        shapes={"refocusing": tuple(unique_shapes)},
    )
    metadata = AbsolutePhaseMetadata(
        rf_frequency_hz=float(spec.rf_frequency_hz),
        refocus_start_seconds=refocus_starts,
        refocus_absolute_phase_rad=refocus_abs,
        excitation_absolute_phase_rad=excitation_abs,
        echo_spacing_seconds=float(echo_spacing_seconds),
        delta_refocus_phase_rad=float(
            wrap_phase(spec.rf_angular_frequency_rad_s * float(echo_spacing_seconds))
        ),
        receiver_phase_rad=float(spec.receiver_phase_rad),
        transient_model=(
            type(spec.transient_model).__name__ if spec.transient_model else None
        ),
        pulse_matrix_count=len(rtot),
        pulse_start_seconds=pulse_start,
        pulse_kind=pulse_kind,
        pulse_rotating_phase_rad=pulse_rotating,
        pulse_absolute_phase_rad=pulse_absolute,
        pulse_phase_bin=pulse_phase_bin,
        pulse_matrix_phase_rad=pulse_matrix_phase,
        excitation_matrix_indices=np.array([1, 0], dtype=np.int64),
        encoding_absolute_phase_rad=encoding_abs,
        encoding_matrix_indices=penc.copy(),
        encoding_phase_bin=None if pi_bins is None else pi_bins[:1].copy(),
        encoding_matrix_phase_rad=pi_matrix_phases[:1].copy(),
        refocus_matrix_indices=pref.copy(),
        phase_bins=spec.phase_bins,
        refocus_phase_bin=None if pi_bins is None else pi_bins[1:].copy(),
        refocus_matrix_phase_rad=pi_matrix_phases[1:].copy(),
        unique_refocus_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        refocus_pulse_library=library,
    )
    return rtot, penc, pref, tf1, tf2, metadata, refocus_abs


def run_matched_diffusion_cpmg(
    num_echoes: int = 5,
    echo_spacing_seconds: float = 1000e-6,
    t1_seconds: float = 100e-3,
    t2_seconds: float = 100e-3,
    dz: float = 0.001,
    diffusion_time: float = 1000e-6,
    diffusion_coefficient: float | None = None,
    t90_seconds: float = 100e-6,
    q_value: float = 50.0,
    *,
    numpts: int = 101,
    apply_receiver: bool = False,
    num_workers: int | None = 1,
    q_stability_action: str = "warn",
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> MatchedDiffusionCPMGResult:
    """Run a compact matched-probe diffusion-aware CPMG train.

    Mirrors the useful core of MATLAB
    `Sim_Diffusion/sim_dif_matched_CPMG_noRx.m`, with precomputed RF matrices
    and an `arb10`-style diffusion kernel.

    Models *free* diffusion in a *constant* background gradient: each
    free-precession interval is attenuated by ``exp(-(1/3) gamma**2 G**2 D
    t**3)`` on its own duration, reproducing the Carr-Purcell train law
    ``exp(-(1/12) gamma**2 G**2 D t_E**3 N)``. ``diffusion_time`` sets the
    encoding-block geometry (it is not a narrow-pulse PGSE ``Delta``); the
    background gradient enters only through the diffusion attenuation, not the
    spin precession.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing_seconds <= 2 * t90_seconds:
        raise ValueError("echo_spacing_seconds must be longer than T_180")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")
    if dz <= 0 or diffusion_time < 0 or t90_seconds <= 0:
        raise ValueError("dz and t90_seconds must be positive; diffusion_time must be non-negative")
    if diffusion_coefficient is not None and diffusion_coefficient < 0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if q_value <= 0:
        raise ValueError("q_value must be positive")
    check_matched_diffusion_q_stability(q_value, action=q_stability_action)

    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    pp0 = pp0.__class__(
        **{
            **pp0.__dict__,
            "T_90": float(t90_seconds),
            "T_180": 2 * float(t90_seconds),
            "tacq": np.array([echo_spacing_seconds / 2], dtype=np.float64),
        }
    )
    gradient = float(sp0.grad)
    w1 = np.pi / (2 * t90_seconds)
    maxoffs = sp0.gamma * gradient * dz / w1
    t_180 = 2 * t90_seconds
    encoding_gap = (np.pi / 2) * (
        diffusion_time - 0.5 * t90_seconds - 0.5 * t_180
    ) / t90_seconds
    if encoding_gap < 0:
        raise ValueError("diffusion_time is too short for the encoding pulse block")
    tfp = (np.pi / 2) * (echo_spacing_seconds - t_180) / (2 * t90_seconds)
    max_time = float(
        np.pi / 2
        + 2 * encoding_gap
        + np.pi
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = np.linspace(-maxoffs, maxoffs, int(numpts))
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time=max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    c1, c2 = matching_network_design2(sp0.L, q_value, sp0.f0, sp0.Rs)
    diffusion_value = sp0.D if diffusion_coefficient is None else float(diffusion_coefficient)
    sp = {
        **sp0.__dict__,
        "C1": c1,
        "C2": c2,
        "Q": float(q_value),
        "R": 2 * np.pi * sp0.f0 * sp0.L / float(q_value),
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "D": diffusion_value,
        "Delta": float(diffusion_time),
        "grad_physical": gradient,
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    ap_spec = as_absolute_phase_spec(absolute_phase)
    rtot, penc, pref, tf1, tf2, ap_metadata, _refocus_abs = (
        _build_matched_diffusion_absolute_phase_matrices(
            del_w=del_w,
            sp=sp,
            pp=pp0,
            spec=ap_spec,
            num_echoes=int(num_echoes),
            echo_spacing_seconds=float(echo_spacing_seconds),
            encoding_gap=encoding_gap,
            tfp=tfp,
            delay_seconds=2 * pp0.T_90,
        )
    )
    sp["tf1"] = tf1
    sp["tf2"] = tf2

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tenc = np.array(
        [encoding_gap, np.pi, encoding_gap],
        dtype=np.float64,
    )
    aenc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0, 0], dtype=np.int64)
    grad_enc = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t90_seconds,
        "tp": np.concatenate([texc, tenc, tref]),
        "amp": np.concatenate([aexc, aenc, aref]),
        "acq": np.concatenate([acq_exc, acq_enc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_enc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, penc, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, penc, pref])}
    _macq1, mrx1 = calc_macq_matched_probe_relax_diffusion(
        sp,
        pp1,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    _macq2, mrx2 = calc_macq_matched_probe_relax_diffusion(
        sp,
        pp2,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    mrx = mrx1 - mrx2

    tacq = float((np.pi / 2) * (echo_spacing_seconds / 2) / t90_seconds)
    tdw = float((np.pi / 2) * pp0.tdw / t90_seconds)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1)
    return MatchedDiffusionCPMGResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        q_value=float(q_value),
        diffusion_coefficient=float(diffusion_value),
        diffusion_time=float(diffusion_time),
        gradient=gradient,
        dz=float(dz),
        probe="matched",
        absolute_phase=ap_metadata,
    )


def run_tuned_diffusion_cpmg(
    num_echoes: int = 5,
    echo_spacing_seconds: float = 1000e-6,
    t1_seconds: float = 100e-3,
    t2_seconds: float = 100e-3,
    dz: float = 0.001,
    diffusion_time: float = 1000e-6,
    diffusion_coefficient: float | None = None,
    gradient: float = 1.0,
    t90_seconds: float = 100e-6,
    q_value: float | None = None,
    *,
    numpts: int = 101,
    apply_receiver: bool = True,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> TunedDiffusionCPMGResult:
    """Run a compact tuned-probe diffusion-aware CPMG train."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if echo_spacing_seconds <= 2 * t90_seconds:
        raise ValueError("echo_spacing_seconds must be longer than T_180")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")
    if dz <= 0 or diffusion_time < 0 or t90_seconds <= 0:
        raise ValueError("dz and t90_seconds must be positive; diffusion_time must be non-negative")
    if diffusion_coefficient is not None and diffusion_coefficient < 0:
        raise ValueError("diffusion_coefficient must be non-negative")
    if gradient <= 0:
        raise ValueError("gradient must be positive")
    if q_value is not None and q_value <= 0:
        raise ValueError("q_value must be positive")

    _params, sp0, pp0 = set_params_tuned_orig(numpts=numpts)
    if q_value is not None:
        sp0 = sp0.__class__(**{**sp0.__dict__, "Q": float(q_value)})
    sp0 = sp0.__class__(
        **{
            **sp0.__dict__,
            "R": 2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
            "C": 1 / ((2 * np.pi * sp0.f0) ** 2 * sp0.L),
        }
    )
    pp0 = pp0.__class__(
        **{
            **pp0.__dict__,
            "T_90": float(t90_seconds),
            "T_180": 2 * float(t90_seconds),
            "tacq": np.array([echo_spacing_seconds / 2], dtype=np.float64),
        }
    )
    gradient = float(gradient)
    w1 = np.pi / (2 * t90_seconds)
    maxoffs = sp0.gamma * gradient * dz / w1
    t_180 = 2 * t90_seconds
    encoding_gap = (np.pi / 2) * (
        diffusion_time - 0.5 * t90_seconds - 0.5 * t_180
    ) / t90_seconds
    if encoding_gap < 0:
        raise ValueError("diffusion_time is too short for the encoding pulse block")
    tfp = (np.pi / 2) * (echo_spacing_seconds - t_180) / (2 * t90_seconds)
    max_time = float(
        np.pi / 2
        + 2 * encoding_gap
        + np.pi
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = np.linspace(-maxoffs, maxoffs, int(numpts))
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time=max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )

    diffusion_value = 2.0e-12 if diffusion_coefficient is None else float(diffusion_coefficient)
    sp = {
        **sp0.__dict__,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "D": diffusion_value,
        "Delta": float(diffusion_time),
        "grad_physical": gradient,
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    ap_spec = as_absolute_phase_spec(absolute_phase)
    rtot, penc, pref, ap_metadata = _build_tuned_diffusion_absolute_phase_matrices(
        del_w=del_w,
        sp=sp,
        pp=pp0,
        spec=ap_spec,
        num_echoes=int(num_echoes),
        echo_spacing_seconds=float(echo_spacing_seconds),
        encoding_gap=encoding_gap,
        tfp=tfp,
        delay_seconds=2 * pp0.T_90,
    )
    sp["tf"] = tuned_probe_rx_tf(sp, pp0)

    texc = np.array([np.pi / 2, -1.0], dtype=np.float64)
    pexc1 = np.array([1, 0], dtype=np.int64)
    pexc2 = np.array([2, 0], dtype=np.int64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tenc = np.array([encoding_gap, np.pi, encoding_gap], dtype=np.float64)
    aenc = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    acq_enc = np.array([0, 0, 0], dtype=np.int64)
    grad_enc = np.zeros(3, dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": t90_seconds,
        "tp": np.concatenate([texc, tenc, tref]),
        "amp": np.concatenate([aexc, aenc, aref]),
        "acq": np.concatenate([acq_exc, acq_enc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_enc, grad_ref]),
        "Rtot": rtot,
    }
    pp1 = {**pp_common, "pul": np.concatenate([pexc1, penc, pref])}
    pp2 = {**pp_common, "pul": np.concatenate([pexc2, penc, pref])}
    _macq1, mrx1 = calc_macq_tuned_probe_relax_diffusion(
        sp,
        pp1,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    _macq2, mrx2 = calc_macq_tuned_probe_relax_diffusion(
        sp,
        pp2,
        apply_receiver=apply_receiver,
        num_workers=num_workers,
    )
    mrx = (mrx1 - mrx2) / 2

    tacq = float((np.pi / 2) * (echo_spacing_seconds / 2) / t90_seconds)
    tdw = float((np.pi / 2) * pp0.tdw / t90_seconds)
    echo, tvect, echo_integrals = _echo_train_from_spectra(mrx, del_w, tacq, tdw)
    sequence_time = echo_spacing_seconds * (np.arange(int(num_echoes), dtype=np.float64) + 1)
    return TunedDiffusionCPMGResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        q_value=float(sp0.Q),
        diffusion_coefficient=float(diffusion_value),
        diffusion_time=float(diffusion_time),
        gradient=gradient,
        dz=float(dz),
        probe="tuned",
        absolute_phase=ap_metadata,
    )


def run_matched_diffusion_q_sweep(
    q_values: Iterable[float] | np.ndarray | None = None,
    *,
    num_echoes: int = 5,
    echo_spacing_seconds: float = 1000e-6,
    diffusion_coefficient: float | None = None,
    diffusion_time: float = 1000e-6,
    dz: float = 0.001,
    t90_seconds: float = 100e-6,
    numpts: int = 101,
    num_workers: int | None = 1,
    sweep_workers: int | None = 1,
    q_stability_action: str = "warn",
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> MatchedDiffusionQSweepResult:
    """Sweep matched-probe Q for the compact diffusion CPMG workflow."""

    values = np.asarray([20, 50, 80] if q_values is None else q_values)
    values = values.astype(np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("q_values must not be empty")

    def case_runner(q_value: float) -> MatchedDiffusionCPMGResult:
        return run_matched_diffusion_cpmg(
            num_echoes=num_echoes,
            echo_spacing_seconds=echo_spacing_seconds,
            diffusion_time=diffusion_time,
            diffusion_coefficient=diffusion_coefficient,
            dz=dz,
            t90_seconds=t90_seconds,
            q_value=q_value,
            numpts=numpts,
            apply_receiver=False,
            num_workers=num_workers,
            q_stability_action=q_stability_action,
            auto_refine_grid=auto_refine_grid,
            rephase_safety_factor=rephase_safety_factor,
            rephase_action=rephase_action,
            absolute_phase=absolute_phase,
        )

    workers = 1 if sweep_workers is None else int(sweep_workers)
    if workers <= 1:
        rows = [case_runner(float(value)) for value in values]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(case_runner, [float(value) for value in values]))

    return MatchedDiffusionQSweepResult(
        values=values,
        value_label="coil Q",
        del_w=rows[0].del_w,
        echo=np.stack([row.echo for row in rows], axis=0),
        tvect=rows[0].tvect,
        echo_integrals=np.stack([row.echo_integrals for row in rows], axis=0),
        sequence_time=rows[0].sequence_time,
        diffusion_coefficient=rows[0].diffusion_coefficient,
        diffusion_time=rows[0].diffusion_time,
        gradient=rows[0].gradient,
        dz=rows[0].dz,
        probe="matched",
        sweep="diffusion_q",
        absolute_phase=tuple(row.absolute_phase for row in rows),
    )
