"""FID workflow entry points.

MATLAB reference folder:
    SpinDynamicsUpdated/Version_2/code/FID_Example
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.core.echo import calc_fid_time_domain
from spin_dynamics.core.kernels import sim_spin_dynamics_arb7
from spin_dynamics.parameters import set_params_matched_orig, set_params_tuned_orig
from spin_dynamics.radiation_damping import (
    RadiationDampingProbe,
    analytic_radiation_damping_envelope,
    proton_thermal_magnetization_density,
    radiation_damping_probe_from_matched,
    radiation_damping_probe_from_tuned,
    simulate_radiation_damping_fid,
)


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


@dataclass(frozen=True)
class RadiationDampingFIDResult:
    """Workflow result for an ideal hard-pulse FID with radiation damping."""

    time_seconds: np.ndarray
    normalized_time: np.ndarray
    mxy: np.ndarray
    mz: np.ndarray
    feedback: np.ndarray
    analytic_envelope: np.ndarray
    probe: RadiationDampingProbe
    model: str
    flip_angle: float
    pulse_phase: float

    @property
    def envelope(self) -> np.ndarray:
        return np.abs(self.mxy)


def calc_macq_fid(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    params: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, float]:
    """Calculate acquired ideal FID magnetization.

    Mirrors MATLAB `calc_macq/calc_macq_fid.m`, with plotting removed.
    """

    T_90 = float(_field(pp, "T_90"))
    normalized = {
        "tp": (np.pi / 2) * np.asarray(_field(params, "tp"), dtype=np.float64) / T_90,
        "phi": np.asarray(_field(params, "phi"), dtype=np.float64),
        "amp": np.asarray(_field(params, "amp"), dtype=np.float64),
        "acq": np.asarray(_field(params, "acq"), dtype=bool),
        "grad": np.asarray(_field(params, "grad"), dtype=np.float64),
        "len_acq": (np.pi / 2) * float(_field(pp, "tacq")) / T_90,
        "del_w": np.asarray(_field(params, "del_w"), dtype=np.float64),
        "w_1": np.asarray(_field(params, "w_1"), dtype=np.float64),
        "m0": np.asarray(_field(params, "m0"), dtype=np.complex128)
        * np.ones_like(np.asarray(_field(params, "del_w"), dtype=np.float64)),
        "T1n": (np.pi / 2) * np.asarray(_field(sp, "T1"), dtype=np.float64) / T_90,
        "T2n": (np.pi / 2) * np.asarray(_field(sp, "T2"), dtype=np.float64) / T_90,
        "mth": np.asarray(_field(sp, "mth"), dtype=np.complex128)
        * np.ones_like(np.asarray(_field(params, "del_w"), dtype=np.float64)),
    }
    tacq_normalized = normalized["len_acq"]
    return sim_spin_dynamics_arb7(normalized), float(tacq_normalized)


def sim_fid_ideal(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate the ideal no-probe FID workflow.

    Mirrors MATLAB `Sim_FID/simFID_ideal.m`, returning the acquired spectrum,
    time-domain FID, and normalized acquisition time vector.
    """

    T_90 = float(_field(pp, "T_90"))
    params = {
        "tp": np.array(
            [
                T_90,
                _field(pp, "acqDelay"),
                _field(pp, "tacq"),
                _field(pp, "acqDelay"),
            ],
            dtype=np.float64,
        ),
        "phi": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "amp": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "acq": np.array([0, 0, 1, 0], dtype=bool),
        "grad": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "len_acq": float(_field(pp, "tacq")),
        "del_w": np.asarray(_field(sp, "del_w"), dtype=np.float64),
        "w_1": np.asarray(_field(sp, "w_1"), dtype=np.float64),
        "m0": _field(sp, "m0"),
        "T1n": np.asarray(_field(sp, "T1"), dtype=np.float64),
        "T2n": np.asarray(_field(sp, "T2"), dtype=np.float64),
        "mth": _field(sp, "mth"),
    }
    macq, tacq = calc_macq_fid(sp, pp, params)
    echo, tvect = calc_fid_time_domain(
        macq[0, :],
        np.asarray(_field(sp, "del_w"), dtype=np.float64),
        tacq,
        (np.pi / 2) * float(_field(pp, "tdw")) / T_90,
    )
    return macq, echo, tvect


def run_radiation_damping_fid(
    *,
    probe: str = "matched",
    fill_factor: float = 0.7,
    equilibrium_magnetization: float | None = None,
    field_tesla: float = 1.0,
    proton_concentration_mol_per_liter: float = 111.0,
    temperature_kelvin: float = 300.0,
    polarization_scale: float = 1.0,
    flip_angle: float = np.pi / 2,
    pulse_phase: float = 0.0,
    phase: float = 0.0,
    detuning: float = 0.0,
    duration_seconds: float | None = None,
    num_points: int = 401,
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    model: str = "instant",
) -> RadiationDampingFIDResult:
    """Run an ideal hard-pulse FID with probe-coupled radiation damping.

    This workflow is intentionally separate from `sim_fid_ideal`, which keeps
    its MATLAB-compatible return contract. It is the analytic validation anchor
    for the nonlinear radiation-damping machinery.
    """

    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if polarization_scale <= 0:
        raise ValueError("polarization_scale must be positive")
    if equilibrium_magnetization is None:
        mth = proton_thermal_magnetization_density(
            field_tesla,
            proton_concentration_mol_per_liter=proton_concentration_mol_per_liter,
            temperature_kelvin=temperature_kelvin,
        )
        mth *= float(polarization_scale)
    else:
        mth = float(equilibrium_magnetization)

    if probe == "matched":
        sp, pp = set_params_matched_orig(numpts=21)
        rd_probe = radiation_damping_probe_from_matched(
            sp,
            fill_factor=fill_factor,
            equilibrium_magnetization=mth,
            phase=phase,
            detuning=detuning,
        )
    elif probe == "tuned":
        _params, sp, pp = set_params_tuned_orig(numpts=21)
        rd_probe = radiation_damping_probe_from_tuned(
            sp,
            fill_factor=fill_factor,
            equilibrium_magnetization=mth,
            phase=phase,
            detuning=detuning,
        )
    else:
        raise ValueError("probe must be 'matched' or 'tuned'")

    duration = 5.0 * rd_probe.trd if duration_seconds is None else float(duration_seconds)
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    time = np.linspace(0.0, duration, int(num_points), dtype=np.float64)
    result = simulate_radiation_damping_fid(
        time,
        rd_probe,
        flip_angle=flip_angle,
        pulse_phase=pulse_phase,
        t1=t1_seconds,
        t2=t2_seconds,
        model=model,
    )
    analytic = analytic_radiation_damping_envelope(
        time,
        flip_angle,
        rd_probe.trd,
        t2=t2_seconds,
    )
    normalized_time = (np.pi / 2) * time / float(_field(pp, "T_90"))
    return RadiationDampingFIDResult(
        time_seconds=time,
        normalized_time=normalized_time,
        mxy=result.mxy,
        mz=result.mz,
        feedback=result.feedback,
        analytic_envelope=analytic,
        probe=rd_probe,
        model=model,
        flip_angle=float(flip_angle),
        pulse_phase=float(pulse_phase),
    )
