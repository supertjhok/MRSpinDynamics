"""CPMG workflow entry points.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/CPMG_Asymp_Examples
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from spin_dynamics.absolute_phase import (
    AbsolutePhaseMetadata,
    AbsolutePhaseSpec,
    FiniteCPMGPhaseSchedule,
    FiniteCPMGPulsePlan,
    LongitudinalPhaseKick,
    PulseShape,
    PulseShapeLibrary,
    apply_absolute_phase_model,
    as_absolute_phase_spec,
    build_cpmg_absolute_phase_metadata,
    build_finite_cpmg_phase_schedule,
    build_finite_cpmg_pulse_plan,
    phase_bin_indices,
    quantize_phase_to_bins,
)
from spin_dynamics.core.echo import calc_time_domain_echo
from spin_dynamics.core.isochromats import (
    check_rephasing,
    recommended_numpts_for_rephasing,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.core.rotations import (
    MatrixElements,
    calc_rot_axis_arba3,
    calc_rotation_matrix,
    free_precession_matrix_elements,
    sim_spin_dynamics_asymp_mag3,
)
from spin_dynamics.noise import (
    NoiseMetadata,
    NoiseSpec,
    add_received_noise,
    as_noise_spec,
    matched_probe_output_noise_density,
    tuned_probe_output_noise_density,
    untuned_probe_output_noise_density,
)
from spin_dynamics.parameters import (
    set_params_ideal,
    set_params_matched_orig,
    set_params_tuned_orig,
    set_params_untuned_orig,
)
from spin_dynamics.phase_cycling import PhaseCycle, cpmg_two_step_phase_cycle
from spin_dynamics.probes.matched import calc_masy_matched_probe_orig
from spin_dynamics.probes.matched import find_coil_current, matching_network_design2
from spin_dynamics.probes.tuned import (
    calc_masy_tuned_probe_lp_orig,
    tuned_probe_lp,
    tuned_probe_rx_tf,
)
from spin_dynamics.probes.untuned import (
    calc_masy_untuned_probe_lp,
    untuned_probe_lp,
    untuned_probe_rx_tf,
)
from spin_dynamics.radiation_damping import (
    RadiationDampingSpec,
    radiation_damping_probe_from_matched,
    radiation_damping_probe_from_tuned,
)
from spin_dynamics.workflows.acquisition import (
    calc_macq_ideal_probe_relax4,
    calc_macq_matched_probe_relax4,
    calc_macq_tuned_probe_relax4,
    calc_macq_untuned_probe_relax4,
)


@dataclass(frozen=True)
class CPMGResult:
    """Common result object for ideal and probe-aware CPMG workflows."""

    del_w: np.ndarray
    masy: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    snr: float | None
    probe: str
    mrx_noisy: np.ndarray | None = None
    echo_noisy: np.ndarray | None = None
    noise: NoiseMetadata | None = None
    phase_cycle: PhaseCycle | None = None


@dataclass(frozen=True)
class CPMGTrainResult:
    """Finite ideal CPMG acquisition result."""

    del_w: np.ndarray
    mrx: np.ndarray
    echo: np.ndarray
    tvect: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str
    mrx_noisy: np.ndarray | None = None
    echo_noisy: np.ndarray | None = None
    echo_integrals_noisy: np.ndarray | None = None
    noise: NoiseMetadata | None = None
    radiation_damping: RadiationDampingSpec | None = None
    absolute_phase: AbsolutePhaseMetadata | None = None
    phase_cycle: PhaseCycle | None = None


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _with_fields(obj: Mapping[str, Any] | Any, **fields: Any) -> SimpleNamespace:
    base = dict(obj) if isinstance(obj, Mapping) else dict(obj.__dict__)
    base.update(fields)
    return SimpleNamespace(**base)


def _as_radiation_damping_spec(
    radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None,
    *,
    probe: str,
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> RadiationDampingSpec | None:
    if radiation_damping is None:
        return None
    if isinstance(radiation_damping, RadiationDampingSpec):
        return radiation_damping
    if not isinstance(radiation_damping, Mapping):
        raise TypeError(
            "radiation_damping must be a RadiationDampingSpec, mapping, or None"
        )
    fill_factor = float(radiation_damping["fill_factor"])
    equilibrium_magnetization = radiation_damping.get("equilibrium_magnetization")
    phase = float(radiation_damping.get("phase", 0.0))
    detuning = float(radiation_damping.get("detuning", 0.0))
    if probe == "tuned":
        rd_probe = radiation_damping_probe_from_tuned(
            sp,
            fill_factor=fill_factor,
            equilibrium_magnetization=equilibrium_magnetization,
            phase=phase,
            detuning=detuning,
        )
    elif probe == "matched":
        rd_probe = radiation_damping_probe_from_matched(
            sp,
            fill_factor=fill_factor,
            equilibrium_magnetization=equilibrium_magnetization,
            phase=phase,
            detuning=detuning,
        )
    else:
        raise ValueError("radiation damping is currently wired for tuned/matched probes")
    return RadiationDampingSpec(
        probe=rd_probe,
        time_scale=(np.pi / 2) / float(_field(pp, "T_90")),
        weights=radiation_damping.get("weights"),
        model=str(radiation_damping.get("model", "instant")),
        max_step=radiation_damping.get("max_step"),
        apply_during_pulses=bool(radiation_damping.get("apply_during_pulses", False)),
        initial_feedback=complex(radiation_damping.get("initial_feedback", 0.0 + 0.0j)),
    )


def _matched_excitation_tf1(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[Any, np.ndarray]:
    c1, c2 = matching_network_design2(
        float(_field(sp, "L")),
        float(_field(sp, "Q")),
        float(_field(sp, "f0")),
        float(_field(sp, "Rs")),
    )
    sp_match = _with_fields(sp, C1=c1, C2=c2)
    pp_curr = _with_fields(
        pp,
        tp=np.concatenate([
            np.asarray(_field(pp, "texc"), dtype=np.float64).reshape(-1),
            [float(_field(pp, "trd"))],
        ]),
        phi=np.concatenate([
            np.asarray(_field(pp, "pexc"), dtype=np.float64).reshape(-1),
            [0.0],
        ]),
        amp=np.concatenate([
            np.asarray(_field(pp, "aexc"), dtype=np.float64).reshape(-1),
            [0.0],
        ]),
    )
    _tvect, _icr, tf1, _tf2 = find_coil_current(sp_match, pp_curr)
    return sp_match, tf1


def _probe_noise_density(
    probe: str,
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    if probe == "tuned":
        return tuned_probe_output_noise_density(sp, pp)
    if probe == "untuned":
        return untuned_probe_output_noise_density(sp, pp)
    if probe == "matched":
        return matched_probe_output_noise_density(sp, pp)
    raise ValueError("probe noise requires tuned, untuned, or matched probe")


def _add_optional_spectrum_noise(
    mrx: np.ndarray,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None,
    *,
    probe: str,
    del_w: np.ndarray | None = None,
    sp: Mapping[str, Any] | Any | None = None,
    pp: Mapping[str, Any] | Any | None = None,
) -> tuple[np.ndarray | None, NoiseMetadata | None]:
    spec = as_noise_spec(noise)
    if spec is None:
        return None, None
    if spec.domain == "time":
        return None, None
    if spec.model == "probe":
        if sp is None or pp is None:
            raise ValueError("probe noise requires spin and pulse parameters")
        pnoise, frequencies = _probe_noise_density(probe, sp, pp)
        return add_received_noise(
            mrx,
            spec,
            pnoise=pnoise,
            frequencies=frequencies,
            sample_axis=del_w,
        )
    return add_received_noise(mrx, spec)


def _add_optional_time_noise(
    echo: np.ndarray,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None,
) -> tuple[np.ndarray | None, NoiseMetadata | None]:
    spec = as_noise_spec(noise)
    if spec is None or spec.domain == "spectrum":
        return None, None
    if spec.model != "white":
        raise ValueError("time-domain noise currently supports only white noise")
    return add_received_noise(echo, spec)


def _default_cpmg_phase_cycle() -> PhaseCycle:
    return cpmg_two_step_phase_cycle()


def _cpmg_excitation_phases(phase_cycle: PhaseCycle) -> np.ndarray:
    phases = phase_cycle.pulse_phases("excitation")
    if phases.size != 2:
        raise ValueError("current CPMG workflows expect a two-step excitation cycle")
    return phases


def _cpmg_excitation_offsets(phase_cycle: PhaseCycle) -> np.ndarray:
    phases = _cpmg_excitation_phases(phase_cycle)
    return phases - phases[0]


def _cpmg_branch_prefixes(phase_cycle: PhaseCycle) -> tuple[np.ndarray, ...]:
    _cpmg_excitation_phases(phase_cycle)
    return tuple(
        np.array([branch + 1, 0], dtype=np.int64)
        for branch in range(phase_cycle.num_steps)
    )


def _combine_cpmg_phase_cycle(
    phase_cycle: PhaseCycle,
    pp_common: Mapping[str, Any],
    refocus_cycle: np.ndarray,
    branch_runner: Callable[[Mapping[str, Any]], np.ndarray],
) -> np.ndarray:
    branch_signals = []
    for prefix in _cpmg_branch_prefixes(phase_cycle):
        pp_branch = {
            **pp_common,
            "pul": np.concatenate([prefix, refocus_cycle]),
        }
        branch_signals.append(branch_runner(pp_branch))
    return phase_cycle.combine(branch_signals)


def calc_masy_ideal(sp: Mapping[str, Any] | Any, pp: Mapping[str, Any] | Any) -> np.ndarray:
    """Calculate ideal CPMG asymptotic magnetization.

    Mirrors MATLAB `calc_masy/calc_masy_ideal.m`, with plotting removed.
    """

    T_90 = _field(pp, "T_90")
    del_w = np.asarray(_field(sp, "del_w"), dtype=np.float64).reshape(-1)

    tacq = (np.pi / 2) * np.asarray(_field(pp, "tacq"), dtype=np.float64) / T_90
    tacq_scalar = float(np.ravel(tacq)[0])

    tref = (np.pi / 2) * np.asarray(_field(pp, "tref"), dtype=np.float64) / T_90
    pref = np.asarray(_field(pp, "pref"), dtype=np.float64)
    aref = np.asarray(_field(pp, "aref"), dtype=np.float64)
    neff = calc_rot_axis_arba3(tref, pref, aref, del_w)

    texc = (np.pi / 2) * np.asarray(_field(pp, "texc"), dtype=np.float64) / T_90
    texc = np.concatenate([texc.reshape(-1), np.array([-1.0])])
    pexc = np.concatenate([
        np.asarray(_field(pp, "pexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])
    aexc = np.concatenate([
        np.asarray(_field(pp, "aexc"), dtype=np.float64).reshape(-1),
        np.array([0.0]),
    ])

    phase_cycle = _default_cpmg_phase_cycle()
    branches = [
        sim_spin_dynamics_asymp_mag3(
            texc,
            pexc + phase_offset,
            aexc,
            neff,
            del_w,
            tacq_scalar,
        )
        for phase_offset in _cpmg_excitation_offsets(phase_cycle)
    ]
    return phase_cycle.combine(branches)


def _offset_grid(numpts: int, maxoffs: float) -> np.ndarray:
    return np.linspace(-float(maxoffs), float(maxoffs), int(numpts))


def _maybe_refine_numpts(
    numpts: int,
    maxoffs: float,
    max_time: float,
    safety_factor: float,
    auto_refine_grid: bool,
) -> int:
    if not auto_refine_grid:
        return int(numpts)
    recommended = recommended_numpts_for_rephasing(maxoffs, max_time, safety_factor)
    return max(int(numpts), recommended)


def _echo_phase_matrix(
    del_w: np.ndarray,
    tacq: float,
    tdw: float,
) -> tuple[np.ndarray, np.ndarray]:
    nacq = round(tacq / tdw) + 1
    tvect = np.linspace(-tacq / 2, tacq / 2, nacq)
    isoc = np.exp(1j * tvect[:, np.newaxis] * del_w[np.newaxis, :])
    return tvect, isoc


def _echo_train_from_spectra(
    mrx: np.ndarray,
    del_w: np.ndarray,
    tacq: float,
    tdw: float,
    *,
    tvect: np.ndarray | None = None,
    isoc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if tvect is None or isoc is None:
        tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)
    echo = (isoc @ mrx.T).T
    echo_integrals = trapezoid(echo, tvect, axis=1)
    return echo, tvect, echo_integrals


def _shape_tuple(shape: PulseShape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return shape.duration, shape.phase, shape.amplitude


def _pulse_shape(
    duration: np.ndarray,
    phase: np.ndarray,
    amplitude: np.ndarray,
) -> PulseShape:
    return PulseShape(
        duration=np.asarray(duration, dtype=np.float64).reshape(-1),
        phase=np.asarray(phase, dtype=np.float64).reshape(-1),
        amplitude=np.asarray(amplitude, dtype=np.float64).reshape(-1),
    )


def _as_pulse_shape(shape: tuple[np.ndarray, np.ndarray, np.ndarray]) -> PulseShape:
    return _pulse_shape(shape[0], shape[1], shape[2])


def _compose_rotation_matrices(
    left: MatrixElements,
    right: MatrixElements,
) -> MatrixElements:
    """Return matrix product ``left @ right`` in coherence-basis storage."""

    return MatrixElements(
        R_00=left.R_00 * right.R_00 + left.R_0m * right.R_m0 + left.R_0p * right.R_p0,
        R_0m=left.R_00 * right.R_0m + left.R_0m * right.R_mm + left.R_0p * right.R_pm,
        R_0p=left.R_00 * right.R_0p + left.R_0m * right.R_mp + left.R_0p * right.R_pp,
        R_m0=left.R_m0 * right.R_00 + left.R_mm * right.R_m0 + left.R_mp * right.R_p0,
        R_mm=left.R_m0 * right.R_0m + left.R_mm * right.R_mm + left.R_mp * right.R_pm,
        R_mp=left.R_m0 * right.R_0p + left.R_mm * right.R_mp + left.R_mp * right.R_pp,
        R_p0=left.R_p0 * right.R_00 + left.R_pm * right.R_m0 + left.R_pp * right.R_p0,
        R_pm=left.R_p0 * right.R_0m + left.R_pm * right.R_mm + left.R_pp * right.R_pm,
        R_pp=left.R_p0 * right.R_0p + left.R_pm * right.R_mp + left.R_pp * right.R_pp,
    )


def _apply_longitudinal_phase_kick(
    matrix: MatrixElements,
    *,
    del_w: np.ndarray,
    spec: AbsolutePhaseSpec | None,
    pp: Mapping[str, Any] | Any,
    pulse_start_phase_rad: float,
    pulse_kind: str,
) -> MatrixElements:
    model = None if spec is None else spec.transient_model
    if not isinstance(model, LongitudinalPhaseKick):
        return matrix
    end_phase = (
        float(pulse_start_phase_rad)
        + float(spec.rf_angular_frequency_rad_s) * float(_field(pp, "T_180"))
    )
    kick = model.phase_kick(end_phase, pulse_kind)
    if kick == 0.0:
        return matrix
    kick_matrix = free_precession_matrix_elements(
        np.full_like(del_w, kick, dtype=np.float64),
        1.0,
    )
    return _compose_rotation_matrices(kick_matrix, matrix)


def _cpmg_absolute_phase_plan(
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None,
    pp: Mapping[str, Any] | Any,
    *,
    num_echoes: int,
    phase_cycle: PhaseCycle,
    excitation_start_seconds: float = 0.0,
) -> tuple[AbsolutePhaseSpec | None, FiniteCPMGPhaseSchedule | None]:
    spec = as_absolute_phase_spec(absolute_phase)
    if spec is None:
        return None, None
    echo_spacing = float(np.sum(np.asarray(_field(pp, "tref"), dtype=np.float64)))
    schedule = build_finite_cpmg_phase_schedule(
        spec=spec,
        excitation_start_seconds=excitation_start_seconds,
        excitation_duration_seconds=float(np.ravel(_field(pp, "texc"))[0]),
        correction_delay_seconds=float(_field(pp, "tcorr")),
        pre_refocus_delay_seconds=float(np.ravel(_field(pp, "tref"))[0]),
        echo_spacing_seconds=echo_spacing,
        num_echoes=int(num_echoes),
        excitation_phases_rad=_cpmg_excitation_phases(phase_cycle),
    )
    return spec, schedule


def _metadata_for_plan(
    spec: AbsolutePhaseSpec | None,
    pp: Mapping[str, Any] | Any,
    schedule: FiniteCPMGPhaseSchedule | None,
    *,
    num_echoes: int,
    pulse_plan: FiniteCPMGPulsePlan,
    phase_cycle: PhaseCycle,
    excitation_start_seconds: float = 0.0,
    refocus_phase_bin: np.ndarray | None = None,
    refocus_matrix_phase_rad: np.ndarray | None = None,
    unique_refocus_phase_rad: np.ndarray | None = None,
    refocus_pulse_library: PulseShapeLibrary | None = None,
) -> AbsolutePhaseMetadata | None:
    if spec is None or schedule is None:
        return None
    echo_spacing = float(np.sum(np.asarray(_field(pp, "tref"), dtype=np.float64)))
    return build_cpmg_absolute_phase_metadata(
        spec=spec,
        excitation_start_seconds=excitation_start_seconds,
        excitation_phases_rad=_cpmg_excitation_phases(phase_cycle),
        refocus_start_seconds=schedule.refocus_start_seconds[: int(num_echoes)],
        refocus_rotating_phase_rad=0.0,
        echo_spacing_seconds=echo_spacing,
        pulse_matrix_count=pulse_plan.pulse_matrix_count,
        schedule=schedule,
        pulse_plan=pulse_plan,
        refocus_phase_bin=refocus_phase_bin,
        refocus_matrix_phase_rad=refocus_matrix_phase_rad,
        unique_refocus_phase_rad=unique_refocus_phase_rad,
        refocus_pulse_library=refocus_pulse_library,
    )


def _build_absolute_phase_rtot_and_pul(
    *,
    del_w: np.ndarray,
    w_1: np.ndarray | float,
    spec: AbsolutePhaseSpec | None,
    pp: Mapping[str, Any] | Any,
    schedule: FiniteCPMGPhaseSchedule | None,
    num_echoes: int,
    exc_y: PulseShape,
    exc_minus_y: PulseShape,
    ref_shape_factory: Any,
    phase_cycle: PhaseCycle,
    excitation_start_seconds: float = 0.0,
) -> tuple[list[Any], np.ndarray, np.ndarray, AbsolutePhaseMetadata | None]:
    _cpmg_excitation_phases(phase_cycle)
    if spec is None or schedule is None:
        pulse_plan = build_finite_cpmg_pulse_plan(
            int(num_echoes),
            per_echo_refocusing=False,
        )
        rtot = [
            calc_rotation_matrix(del_w, w_1, *_shape_tuple(exc_y)),
            calc_rotation_matrix(del_w, w_1, *_shape_tuple(exc_minus_y)),
            calc_rotation_matrix(del_w, w_1, *_shape_tuple(ref_shape_factory(0.0))),
        ]
        pref = pulse_plan.refocus_cycle
        return rtot, pref, pref.copy(), None

    exc_abs = schedule.excitation_absolute_phase_rad
    exc_y_ap = apply_absolute_phase_model(exc_y, spec, float(exc_abs[0]), "excitation")
    exc_minus_y_ap = apply_absolute_phase_model(
        exc_minus_y,
        spec,
        float(exc_abs[1]),
        "excitation",
    )
    rtot = [
        calc_rotation_matrix(del_w, w_1, *_shape_tuple(exc_y_ap)),
        calc_rotation_matrix(del_w, w_1, *_shape_tuple(exc_minus_y_ap)),
    ]

    refocus_abs = schedule.refocus_absolute_phase_rad[: int(num_echoes)]
    matrix_phases = np.asarray(
        quantize_phase_to_bins(refocus_abs, spec.phase_bins),
        dtype=np.float64,
    ).reshape(-1)
    refocus_bins = (
        np.asarray(phase_bin_indices(refocus_abs, spec.phase_bins), dtype=np.int64)
        if spec.phase_bins is not None
        else None
    )
    refocus_cycle: list[int] = []
    matrix_by_phase: dict[int | float, int] = {}
    unique_phases: list[float] = []
    unique_shapes: list[PulseShape] = []

    for echo_idx, matrix_phase in enumerate(matrix_phases):
        if spec.phase_bins is not None:
            key: int | float = int(refocus_bins[echo_idx])
        else:
            key = round(float(matrix_phase) / (2.0 * np.pi), 12)
        matrix_index = matrix_by_phase.get(key)
        if matrix_index is None:
            absolute_start = float(matrix_phase)
            ref_shape = apply_absolute_phase_model(
                ref_shape_factory(absolute_start),
                spec,
                absolute_start,
                "refocusing",
            )
            ref_matrix = calc_rotation_matrix(del_w, w_1, *_shape_tuple(ref_shape))
            ref_matrix = _apply_longitudinal_phase_kick(
                ref_matrix,
                del_w=del_w,
                spec=spec,
                pp=pp,
                pulse_start_phase_rad=absolute_start,
                pulse_kind="refocusing",
            )
            rtot.append(ref_matrix)
            matrix_index = len(rtot)
            matrix_by_phase[key] = matrix_index
            unique_phases.append(absolute_start)
            unique_shapes.append(ref_shape)
        refocus_cycle.extend([0, matrix_index, 0])

    pulse_plan = FiniteCPMGPulsePlan(
        excitation_cycle_one=np.array([1, 0], dtype=np.int64),
        excitation_cycle_two=np.array([2, 0], dtype=np.int64),
        refocus_cycle=np.asarray(refocus_cycle, dtype=np.int64),
        pulse_matrix_count=len(rtot),
    )
    refocus_pulse_library = PulseShapeLibrary(
        absolute_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        shapes={"refocusing": tuple(unique_shapes)},
    )

    pref = pulse_plan.refocus_cycle
    metadata = _metadata_for_plan(
        spec,
        pp,
        schedule,
        num_echoes=num_echoes,
        pulse_plan=pulse_plan,
        phase_cycle=phase_cycle,
        excitation_start_seconds=excitation_start_seconds,
        refocus_phase_bin=refocus_bins,
        refocus_matrix_phase_rad=matrix_phases,
        unique_refocus_phase_rad=np.asarray(unique_phases, dtype=np.float64),
        refocus_pulse_library=refocus_pulse_library,
    )
    return rtot, pref, pref.copy(), metadata


def run_ideal_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> CPMGTrainResult:
    """Run a finite ideal CPMG echo train with relaxation.

    This assembles the same no-probe PAP phase-cycled acquisition pattern used
    by MATLAB finite CPMG examples around `calc_macq_ideal_probe_relax4`.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_ideal(numpts=numpts)
    w1n = (np.pi / 2) / pp0.T_90
    max_time = float(
        np.pi / 2
        + w1n * pp0.tcorr
        + int(num_echoes) * np.sum(w1n * np.asarray(pp0.tref, dtype=np.float64))
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )

    sp = {
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
    }

    phase_cycle = _default_cpmg_phase_cycle()
    excitation_offsets = _cpmg_excitation_offsets(phase_cycle)
    ap_spec, ap_schedule = _cpmg_absolute_phase_plan(
        absolute_phase,
        pp0,
        num_echoes=int(num_echoes),
        phase_cycle=phase_cycle,
    )
    exc_y = _pulse_shape(w1n * pp0.texc, pp0.pexc + excitation_offsets[0], pp0.aexc)
    exc_minus_y = _pulse_shape(
        w1n * pp0.texc,
        pp0.pexc + excitation_offsets[1],
        pp0.aexc,
    )
    ref_x = _pulse_shape(w1n * pp0.tref[1:-1], pp0.pref[1:-1], pp0.aref[1:-1])
    rtot, pref, _pref2, ap_metadata = _build_absolute_phase_rtot_and_pul(
        del_w=del_w,
        w_1=sp["w_1"],
        spec=ap_spec,
        pp=pp0,
        schedule=ap_schedule,
        num_echoes=int(num_echoes),
        exc_y=exc_y,
        exc_minus_y=exc_minus_y,
        ref_shape_factory=lambda _absolute_start: ref_x,
        phase_cycle=phase_cycle,
    )

    texc = np.array([np.pi / 2, w1n * pp0.tcorr], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(
        w1n * np.array([pp0.tref[0], pp0.tref[1], pp0.tref[2]], dtype=np.float64),
        int(num_echoes),
    )
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    mrx = _combine_cpmg_phase_cycle(
        phase_cycle,
        pp_common,
        pref,
        lambda pp_branch: calc_macq_ideal_probe_relax4(
            sp,
            pp_branch,
            num_workers=num_workers,
        ),
    )

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)
    echo, tvect, echo_integrals = _echo_train_from_spectra(
        mrx,
        del_w,
        tacq,
        tdw,
        tvect=tvect,
        isoc=isoc,
    )
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="ideal",
        del_w=del_w,
    )
    echo_noisy = None
    echo_integrals_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy, echo_integrals_noisy = _echo_train_from_spectra(
            mrx_noisy,
            del_w,
            tacq,
            tdw,
            tvect=tvect,
            isoc=isoc,
        )
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
        if echo_noisy is not None:
            echo_integrals_noisy = trapezoid(echo_noisy, tvect, axis=1)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="ideal",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        echo_integrals_noisy=echo_integrals_noisy,
        noise=noise_metadata,
        absolute_phase=ap_metadata,
        phase_cycle=phase_cycle,
    )


def _calc_tuned_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
    *,
    psi: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    carrier_phase = float(_field(pp, "psi")) if psi is None else float(psi)
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tref": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "pref": np.array([pulse_phase + carrier_phase, 0.0], dtype=np.float64),
        "aref": np.array([pulse_amplitude, 0.0], dtype=np.float64),
    }
    tvect, icr, _tvect_raw, _ic = tuned_probe_lp(sp, pp_curr)
    icr = icr * np.exp(-1j * carrier_phase)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0

    amp_range = float(np.max(amp) - np.min(amp))
    if amp_range > 0:
        amp = (amp - np.min(amp)) / amp_range
    amp[amp < amp_zero] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
    )


def _calc_untuned_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
    *,
    psi: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    carrier_phase = float(_field(pp, "psi")) if psi is None else float(psi)
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tref": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "pref": np.array([pulse_phase + carrier_phase, 0.0], dtype=np.float64),
        "aref": np.array([pulse_amplitude, 0.0], dtype=np.float64),
    }
    tvect, icr, _tvect_raw, _ic = untuned_probe_lp(sp, pp_curr)
    icr = icr * np.exp(-1j * carrier_phase)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    B1max = (np.pi / 2) / (T_90 * float(_field(sp, "gamma")))
    amp = np.abs(icr) * float(_field(sp, "sens")) / B1max
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0
    phi[amp == 0] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
    )


def _calc_matched_pulse_shape(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    pulse_duration_seconds: float,
    pulse_phase: float,
    pulse_amplitude: float,
    delay_seconds: float,
    *,
    psi: float | None = None,
    segment_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    T_90 = float(_field(pp, "T_90"))
    delay_normalized = (np.pi / 2) * delay_seconds / T_90
    pp_fields = pp.__dict__ if hasattr(pp, "__dict__") else dict(pp)
    pp_curr = {
        **pp_fields,
        "tp": np.array([pulse_duration_seconds, delay_seconds], dtype=np.float64),
        "phi": np.array([pulse_phase, 0.0], dtype=np.float64),
        "amp": np.array([pulse_amplitude, 0.0], dtype=np.float64),
        "psi": float(_field(pp, "psi")) if psi is None else float(psi),
        "segment_fraction": float(segment_fraction),
    }
    tvect, icr, tf1, tf2 = find_coil_current(sp, pp_curr)

    delt = (np.pi / 2) * (tvect[1] - tvect[0]) / T_90
    tp = delt * np.ones(tvect.size, dtype=np.float64)
    phi = np.arctan2(np.imag(icr), np.real(icr))
    amp = np.abs(icr)
    amp_zero = float(_field(pp, "amp_zero"))
    amp[amp < amp_zero] = 0

    return (
        np.concatenate([tp, [-delay_normalized]]),
        np.concatenate([phi, [0.0]]),
        np.concatenate([amp, [0.0]]),
        tf1,
        tf2,
    )


def run_tuned_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
    radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None,
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> CPMGTrainResult:
    """Run a finite tuned-probe CPMG echo train with relaxation.

    This is the homogeneous-sample finite-acquisition analogue of the tuned
    Version 2 CPMG imaging assembly, without phase-encoding gradients.
    """

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    _params, sp0, pp0 = set_params_tuned_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(
        sp0,
        R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
        C=1 / ((2 * np.pi * sp0.f0) ** 2 * sp0.L),
    )
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    sp = {
        **sp0.__dict__,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    phase_cycle = _default_cpmg_phase_cycle()
    excitation_phases = _cpmg_excitation_phases(phase_cycle)
    ap_spec, ap_schedule = _cpmg_absolute_phase_plan(
        absolute_phase,
        pp0,
        num_echoes=int(num_echoes),
        phase_cycle=phase_cycle,
    )
    excitation_psi = None
    if ap_schedule is not None:
        excitation_psi = (
            ap_schedule.excitation_absolute_phase_rad
            - ap_schedule.excitation_rotating_phase_rad
        )
    exc_y = _calc_tuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[0]),
        1.0,
        2 * pp0.T_90,
        psi=None if excitation_psi is None else float(excitation_psi[0]),
    )
    exc_minus_y = _calc_tuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[1]),
        1.0,
        2 * pp0.T_90,
        psi=None if excitation_psi is None else float(excitation_psi[1]),
    )
    ref_x = _calc_tuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, 2 * pp0.T_90)

    rtot, pref, _pref2, ap_metadata = _build_absolute_phase_rtot_and_pul(
        del_w=del_w,
        w_1=sp["w_1"],
        spec=ap_spec,
        pp=pp0,
        schedule=ap_schedule,
        num_echoes=int(num_echoes),
        exc_y=_as_pulse_shape(exc_y),
        exc_minus_y=_as_pulse_shape(exc_minus_y),
        ref_shape_factory=lambda carrier_phase: _as_pulse_shape(
            ref_x
            if ap_schedule is None
            else _calc_tuned_pulse_shape(
                sp,
                pp0,
                pp0.T_180,
                0.0,
                1.0,
                2 * pp0.T_90,
                psi=carrier_phase,
            )
        ),
        phase_cycle=phase_cycle,
    )

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf"] = tuned_probe_rx_tf(sp, pp0)
    rd_spec = _as_radiation_damping_spec(
        radiation_damping,
        probe="tuned",
        sp=sp,
        pp=pp0,
    )
    mrx = _combine_cpmg_phase_cycle(
        phase_cycle,
        pp_common,
        pref,
        lambda pp_branch: calc_macq_tuned_probe_relax4(
            sp,
            pp_branch,
            num_workers=num_workers,
            radiation_damping=rd_spec,
        )[1],
    )

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)
    echo, tvect, echo_integrals = _echo_train_from_spectra(
        mrx,
        del_w,
        tacq,
        tdw,
        tvect=tvect,
        isoc=isoc,
    )
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="tuned",
        del_w=del_w,
        sp=sp,
        pp=pp0,
    )
    echo_noisy = None
    echo_integrals_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy, echo_integrals_noisy = _echo_train_from_spectra(
            mrx_noisy,
            del_w,
            tacq,
            tdw,
            tvect=tvect,
            isoc=isoc,
        )
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
        if echo_noisy is not None:
            echo_integrals_noisy = trapezoid(echo_noisy, tvect, axis=1)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="tuned",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        echo_integrals_noisy=echo_integrals_noisy,
        noise=noise_metadata,
        radiation_damping=rd_spec,
        absolute_phase=ap_metadata,
        phase_cycle=phase_cycle,
    )


def run_untuned_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> CPMGTrainResult:
    """Run a finite untuned-probe CPMG echo train with relaxation."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    _params, sp0, pp0 = set_params_untuned_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(
        sp0,
        R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q,
        C=1 / ((2 * np.pi * 10 * sp0.f0) ** 2 * sp0.L),
    )
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    sp = {
        **sp0.__dict__,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    phase_cycle = _default_cpmg_phase_cycle()
    excitation_phases = _cpmg_excitation_phases(phase_cycle)
    ap_spec, ap_schedule = _cpmg_absolute_phase_plan(
        absolute_phase,
        pp0,
        num_echoes=int(num_echoes),
        phase_cycle=phase_cycle,
    )
    excitation_psi = None
    if ap_schedule is not None:
        excitation_psi = (
            ap_schedule.excitation_absolute_phase_rad
            - ap_schedule.excitation_rotating_phase_rad
        )
    exc_y = _calc_untuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[0]),
        1.0,
        pp0.trd,
        psi=None if excitation_psi is None else float(excitation_psi[0]),
    )
    exc_minus_y = _calc_untuned_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[1]),
        1.0,
        pp0.trd,
        psi=None if excitation_psi is None else float(excitation_psi[1]),
    )
    ref_x = _calc_untuned_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)

    rtot, pref, _pref2, ap_metadata = _build_absolute_phase_rtot_and_pul(
        del_w=del_w,
        w_1=sp["w_1"],
        spec=ap_spec,
        pp=pp0,
        schedule=ap_schedule,
        num_echoes=int(num_echoes),
        exc_y=_as_pulse_shape(exc_y),
        exc_minus_y=_as_pulse_shape(exc_minus_y),
        ref_shape_factory=lambda carrier_phase: _as_pulse_shape(
            ref_x
            if ap_schedule is None
            else _calc_untuned_pulse_shape(
                sp,
                pp0,
                pp0.T_180,
                0.0,
                1.0,
                pp0.trd,
                psi=carrier_phase,
            )
        ),
        phase_cycle=phase_cycle,
    )

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf"] = untuned_probe_rx_tf(sp, pp0)
    mrx = _combine_cpmg_phase_cycle(
        phase_cycle,
        pp_common,
        pref,
        lambda pp_branch: calc_macq_untuned_probe_relax4(
            sp,
            pp_branch,
            num_workers=num_workers,
        )[1],
    )

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)
    echo, tvect, echo_integrals = _echo_train_from_spectra(
        mrx,
        del_w,
        tacq,
        tdw,
        tvect=tvect,
        isoc=isoc,
    )
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="untuned",
        del_w=del_w,
        sp=sp,
        pp=pp0,
    )
    echo_noisy = None
    echo_integrals_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy, echo_integrals_noisy = _echo_train_from_spectra(
            mrx_noisy,
            del_w,
            tacq,
            tdw,
            tvect=tvect,
            isoc=isoc,
        )
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
        if echo_noisy is not None:
            echo_integrals_noisy = trapezoid(echo_noisy, tvect, axis=1)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="untuned",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        echo_integrals_noisy=echo_integrals_noisy,
        noise=noise_metadata,
        absolute_phase=ap_metadata,
        phase_cycle=phase_cycle,
    )


def run_matched_cpmg_train(
    numpts: int = 101,
    maxoffs: float = 10.0,
    num_echoes: int = 8,
    t1_seconds: float = 2.0,
    t2_seconds: float = 2.0,
    *,
    q_value: float | None = None,
    mistuning_offset: float | None = None,
    num_workers: int | None = 1,
    auto_refine_grid: bool = False,
    rephase_safety_factor: float = 1.25,
    rephase_action: str = "warn",
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
    radiation_damping: RadiationDampingSpec | Mapping[str, Any] | None = None,
    absolute_phase: AbsolutePhaseSpec | Mapping[str, Any] | None = None,
) -> CPMGTrainResult:
    """Run a finite matched-probe CPMG echo train with relaxation."""

    if num_echoes <= 0:
        raise ValueError("num_echoes must be positive")
    if t1_seconds <= 0 or t2_seconds <= 0:
        raise ValueError("t1_seconds and t2_seconds must be positive")

    sp0, pp0 = set_params_matched_orig(numpts=numpts)
    if q_value is not None:
        if q_value <= 0:
            raise ValueError("q_value must be positive")
        sp0 = replace(sp0, Q=float(q_value))
    if mistuning_offset is not None:
        f0 = sp0.fin + (sp0.fin / sp0.Q) * float(mistuning_offset)
        if f0 <= 0:
            raise ValueError("mistuning_offset produced non-positive f0")
        sp0 = replace(sp0, f0=f0)
    sp0 = replace(sp0, R=2 * np.pi * sp0.f0 * sp0.L / sp0.Q)
    tfp = (np.pi / 2) * (pp0.preDelay + pp0.postDelay) / (2 * pp0.T_90)
    max_time = float(
        np.pi / 2
        + (np.pi / 2) * pp0.tcorr / pp0.T_90
        + int(num_echoes) * (tfp + np.pi + tfp)
    )
    numpts = _maybe_refine_numpts(
        numpts,
        maxoffs,
        max_time,
        rephase_safety_factor,
        auto_refine_grid,
    )
    del_w = _offset_grid(numpts, maxoffs)
    if rephase_action != "ignore":
        check_rephasing(
            del_w,
            max_time,
            safety_factor=rephase_safety_factor,
            action=rephase_action,
        )
    c1, c2 = matching_network_design2(sp0.L, sp0.Q, sp0.f0, sp0.Rs)
    sp = {
        **sp0.__dict__,
        "C1": c1,
        "C2": c2,
        "numpts": int(numpts),
        "maxoffs": float(maxoffs),
        "del_w": del_w,
        "del_wg": np.zeros_like(del_w),
        "w_1": np.ones_like(del_w),
        "w_1r": np.ones_like(del_w),
        "T1": t1_seconds * np.ones_like(del_w),
        "T2": t2_seconds * np.ones_like(del_w),
        "m0": sp0.m0 * np.ones_like(del_w),
        "mth": sp0.mth * np.ones_like(del_w),
        "plt_tx": 0,
        "plt_rx": 0,
        "plt_sequence": 0,
        "plt_axis": 0,
        "plt_mn": 0,
        "plt_echo": 0,
    }

    phase_cycle = _default_cpmg_phase_cycle()
    excitation_phases = _cpmg_excitation_phases(phase_cycle)
    ap_spec, ap_schedule = _cpmg_absolute_phase_plan(
        absolute_phase,
        pp0,
        num_echoes=int(num_echoes),
        phase_cycle=phase_cycle,
    )
    excitation_psi = None
    if ap_schedule is not None:
        excitation_psi = (
            ap_schedule.excitation_absolute_phase_rad
            - ap_schedule.excitation_rotating_phase_rad
        )
    exc_y_tp, exc_y_phi, exc_y_amp, tf1, tf2 = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[0]),
        1.0,
        pp0.trd,
        psi=None if excitation_psi is None else float(excitation_psi[0]),
    )
    exc_minus_y = _calc_matched_pulse_shape(
        sp,
        pp0,
        pp0.T_90,
        float(excitation_phases[1]),
        1.0,
        pp0.trd,
        psi=None if excitation_psi is None else float(excitation_psi[1]),
    )[:3]
    ref_x = _calc_matched_pulse_shape(sp, pp0, pp0.T_180, 0.0, 1.0, pp0.trd)[:3]
    rtot, pref, _pref2, ap_metadata = _build_absolute_phase_rtot_and_pul(
        del_w=del_w,
        w_1=sp["w_1"],
        spec=ap_spec,
        pp=pp0,
        schedule=ap_schedule,
        num_echoes=int(num_echoes),
        exc_y=_pulse_shape(exc_y_tp, exc_y_phi, exc_y_amp),
        exc_minus_y=_as_pulse_shape(exc_minus_y),
        ref_shape_factory=lambda carrier_phase: _as_pulse_shape(
            ref_x
            if ap_schedule is None
            else _calc_matched_pulse_shape(
                sp,
                pp0,
                pp0.T_180,
                0.0,
                1.0,
                pp0.trd,
                psi=carrier_phase,
                segment_fraction=0.5,
            )[:3]
        ),
        phase_cycle=phase_cycle,
    )

    texc = np.array([np.pi / 2, (np.pi / 2) * pp0.tcorr / pp0.T_90], dtype=np.float64)
    aexc = np.array([1.0, 0.0], dtype=np.float64)
    acq_exc = np.array([0, 0], dtype=np.int64)
    grad_exc = np.array([0.0, 0.0], dtype=np.float64)

    tref = np.tile(np.array([tfp, np.pi, tfp], dtype=np.float64), int(num_echoes))
    aref = np.tile(np.array([0.0, 1.0, 0.0], dtype=np.float64), int(num_echoes))
    acq_ref = np.tile(np.array([0, 0, 1], dtype=np.int64), int(num_echoes))
    grad_ref = np.zeros(3 * int(num_echoes), dtype=np.float64)

    pp_common = {
        "T_90": pp0.T_90,
        "tp": np.concatenate([texc, tref]),
        "amp": np.concatenate([aexc, aref]),
        "acq": np.concatenate([acq_exc, acq_ref]),
        "grad": np.concatenate([grad_exc, grad_ref]),
        "Rtot": rtot,
    }

    sp["tf1"] = tf1
    sp["tf2"] = tf2
    rd_spec = _as_radiation_damping_spec(
        radiation_damping,
        probe="matched",
        sp=sp,
        pp=pp0,
    )
    mrx = _combine_cpmg_phase_cycle(
        phase_cycle,
        pp_common,
        pref,
        lambda pp_branch: calc_macq_matched_probe_relax4(
            sp,
            pp_branch,
            num_workers=num_workers,
            radiation_damping=rd_spec,
        )[1],
    )

    tacq = float((np.pi / 2) * np.ravel(pp0.tacq)[0] / pp0.T_90)
    tdw = float((np.pi / 2) * pp0.tdw / pp0.T_90)
    tvect, isoc = _echo_phase_matrix(del_w, tacq, tdw)
    echo, tvect, echo_integrals = _echo_train_from_spectra(
        mrx,
        del_w,
        tacq,
        tdw,
        tvect=tvect,
        isoc=isoc,
    )
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="matched",
        del_w=del_w,
        sp=sp,
        pp=pp0,
    )
    echo_noisy = None
    echo_integrals_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy, echo_integrals_noisy = _echo_train_from_spectra(
            mrx_noisy,
            del_w,
            tacq,
            tdw,
            tvect=tvect,
            isoc=isoc,
        )
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
        if echo_noisy is not None:
            echo_integrals_noisy = trapezoid(echo_noisy, tvect, axis=1)
    sequence_time = np.sum(pp0.tref) * (
        np.arange(int(num_echoes), dtype=np.float64) + 0.5
    )

    return CPMGTrainResult(
        del_w=del_w,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        echo_integrals=echo_integrals,
        sequence_time=sequence_time,
        probe="matched",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        echo_integrals_noisy=echo_integrals_noisy,
        noise=noise_metadata,
        radiation_damping=rd_spec,
        absolute_phase=ap_metadata,
        phase_cycle=phase_cycle,
    )


def run_ideal_cpmg(
    numpts: int = 101,
    maxoffs: float = 10.0,
    *,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
) -> CPMGResult:
    """Run the validated ideal no-probe CPMG workflow."""

    phase_cycle = _default_cpmg_phase_cycle()
    del_w = _offset_grid(numpts, maxoffs)
    sp, pp = set_params_ideal(numpts=numpts)
    sp = replace(sp, maxoffs=float(maxoffs), del_w=del_w)
    masy = calc_masy_ideal(sp, pp)
    echo, tvect = calc_time_domain_echo(masy, del_w)
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        masy,
        noise,
        probe="ideal",
        del_w=del_w,
    )
    echo_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy = calc_time_domain_echo(mrx_noisy, del_w)
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=masy,
        echo=echo,
        tvect=tvect,
        snr=None,
        probe="ideal",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        noise=noise_metadata,
        phase_cycle=phase_cycle,
    )


def run_tuned_cpmg(
    numpts: int = 101,
    maxoffs: float = 10.0,
    *,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
) -> CPMGResult:
    """Run the original/reference tuned-probe CPMG workflow."""

    phase_cycle = _default_cpmg_phase_cycle()
    del_w = _offset_grid(numpts, maxoffs)
    params, sp, pp = set_params_tuned_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
    )
    mrx, masy, snr = calc_masy_tuned_probe_lp_orig(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="tuned",
        del_w=del_w,
        sp=sp,
        pp=pp,
    )
    echo_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy = calc_time_domain_echo(mrx_noisy, del_w)
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="tuned",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        noise=noise_metadata,
        phase_cycle=phase_cycle,
    )


def run_untuned_cpmg(
    numpts: int = 101,
    maxoffs: float = 10.0,
    *,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
) -> CPMGResult:
    """Run the original/reference untuned-probe CPMG workflow."""

    phase_cycle = _default_cpmg_phase_cycle()
    del_w = _offset_grid(numpts, maxoffs)
    params, sp, pp = set_params_untuned_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
        plt_axis=0,
    )
    mrx, masy, snr = calc_masy_untuned_probe_lp(params, sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="untuned",
        del_w=del_w,
        sp=sp,
        pp=pp,
    )
    echo_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy = calc_time_domain_echo(mrx_noisy, del_w)
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="untuned",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        noise=noise_metadata,
        phase_cycle=phase_cycle,
    )


def run_matched_cpmg(
    numpts: int = 101,
    maxoffs: float = 10.0,
    *,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None = None,
) -> CPMGResult:
    """Run the original/reference matched-probe CPMG workflow."""

    phase_cycle = _default_cpmg_phase_cycle()
    del_w = _offset_grid(numpts, maxoffs)
    sp, pp = set_params_matched_orig(numpts=numpts)
    sp = replace(
        sp,
        numpts=int(numpts),
        maxoffs=float(maxoffs),
        del_w=del_w,
        plt_tx=0,
        plt_rx=0,
        plt_echo=0,
        plt_axis=0,
        plt_mn=0,
    )
    mrx, masy, snr = calc_masy_matched_probe_orig(sp, pp)
    echo, tvect = calc_time_domain_echo(mrx, del_w)
    sp_noise = sp
    if as_noise_spec(noise) is not None and as_noise_spec(noise).model == "probe":
        sp_match, tf1 = _matched_excitation_tf1(sp, pp)
        sp_noise = _with_fields(sp_match, tf1=tf1)
    mrx_noisy, noise_metadata = _add_optional_spectrum_noise(
        mrx,
        noise,
        probe="matched",
        del_w=del_w,
        sp=sp_noise,
        pp=pp,
    )
    echo_noisy = None
    if mrx_noisy is not None:
        echo_noisy, _tvect_noisy = calc_time_domain_echo(mrx_noisy, del_w)
    else:
        echo_noisy, noise_metadata = _add_optional_time_noise(echo, noise)
    return CPMGResult(
        del_w=del_w,
        masy=masy,
        mrx=mrx,
        echo=echo,
        tvect=tvect,
        snr=snr,
        probe="matched",
        mrx_noisy=mrx_noisy,
        echo_noisy=echo_noisy,
        noise=noise_metadata,
        phase_cycle=phase_cycle,
    )
