"""Radiation-damping back-action models coupled to probe parameters.

The model follows the rotating-frame back-action equations in Section 10.2.5
of the local Measurements textbook. Magnetization is represented in normalized
units, so ``mth=1`` corresponds to the equilibrium magnetization density used
to build the probe coupling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


MU0 = 4e-7 * np.pi
KB = 1.380649e-23
HBAR = 1.054571817e-34
AVOGADRO = 6.02214076e23
PROTON_GAMMA = 2 * np.pi * 42.57747892e6


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _field_or_default(obj: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def radiation_damping_time(
    gamma: float,
    fill_factor: float,
    equilibrium_magnetization: float,
    probe_q: float,
) -> float:
    """Return the radiation-damping time constant ``Trd`` in seconds.

    Uses the SI Bloch-Maxwell convention
    ``1 / Trd = (1/2) * mu0 * gamma * M0 * Q * eta``, i.e.
    ``Trd = 2 / (gamma * mu0 * eta * M0 * Q)``, where ``eta`` (``fill_factor``)
    is the coil *magnetic-energy* filling factor and ``M0``
    (``equilibrium_magnetization``) is the thermal magnetization density in A/m.
    Other references fold a factor of 2 or 2*pi into the definition of ``eta``;
    this function assumes the bare magnetic-energy filling factor with the
    explicit 1/2 shown above.
    """

    gamma = float(gamma)
    fill_factor = float(fill_factor)
    equilibrium_magnetization = float(equilibrium_magnetization)
    probe_q = float(probe_q)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if not (0 < fill_factor <= 1):
        raise ValueError("fill_factor must be in the interval (0, 1]")
    if equilibrium_magnetization <= 0:
        raise ValueError("equilibrium_magnetization must be positive")
    if probe_q <= 0:
        raise ValueError("probe_q must be positive")
    return 2.0 / (gamma * MU0 * fill_factor * equilibrium_magnetization * probe_q)


def proton_thermal_magnetization_density(
    field_tesla: float,
    *,
    proton_concentration_mol_per_liter: float = 111.0,
    temperature_kelvin: float = 300.0,
) -> float:
    """Estimate spin-1/2 proton thermal magnetization density in A/m.

    The default concentration is approximately liquid water, including two
    protons per water molecule.
    """

    if field_tesla <= 0:
        raise ValueError("field_tesla must be positive")
    if proton_concentration_mol_per_liter <= 0:
        raise ValueError("proton_concentration_mol_per_liter must be positive")
    if temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive")
    spins_per_m3 = proton_concentration_mol_per_liter * 1000.0 * AVOGADRO
    spin_factor = 0.5 * (0.5 + 1.0)
    return (
        spins_per_m3
        * PROTON_GAMMA**2
        * HBAR**2
        * spin_factor
        * field_tesla
        / (3.0 * KB * temperature_kelvin)
    )


@dataclass(frozen=True)
class RadiationDampingSample:
    """Convenience description of a sample's equilibrium magnetization."""

    name: str
    equilibrium_magnetization: float
    field_tesla: float
    temperature_kelvin: float
    proton_concentration_mol_per_liter: float
    polarization_scale: float = 1.0


def water_proton_sample(
    field_tesla: float,
    *,
    temperature_kelvin: float = 300.0,
    polarization_scale: float = 1.0,
) -> RadiationDampingSample:
    """Return a liquid-water proton sample preset for RD coupling."""

    concentration = 111.0
    magnetization = proton_thermal_magnetization_density(
        field_tesla,
        proton_concentration_mol_per_liter=concentration,
        temperature_kelvin=temperature_kelvin,
    )
    return RadiationDampingSample(
        name="water protons",
        equilibrium_magnetization=magnetization * float(polarization_scale),
        field_tesla=float(field_tesla),
        temperature_kelvin=float(temperature_kelvin),
        proton_concentration_mol_per_liter=concentration,
        polarization_scale=float(polarization_scale),
    )


def hyperpolarized_proton_sample(
    field_tesla: float,
    *,
    proton_concentration_mol_per_liter: float = 111.0,
    temperature_kelvin: float = 300.0,
    polarization_scale: float = 1e4,
) -> RadiationDampingSample:
    """Return a proton sample preset with boosted non-equilibrium polarization."""

    magnetization = proton_thermal_magnetization_density(
        field_tesla,
        proton_concentration_mol_per_liter=proton_concentration_mol_per_liter,
        temperature_kelvin=temperature_kelvin,
    )
    return RadiationDampingSample(
        name="hyperpolarized protons",
        equilibrium_magnetization=magnetization * float(polarization_scale),
        field_tesla=float(field_tesla),
        temperature_kelvin=float(temperature_kelvin),
        proton_concentration_mol_per_liter=float(proton_concentration_mol_per_liter),
        polarization_scale=float(polarization_scale),
    )


def normalized_radiation_damping_weights(
    density: np.ndarray,
    sensitivity: np.ndarray | None = None,
) -> np.ndarray:
    """Return normalized RD ensemble weights from density and coil sensitivity.

    Radiation damping is weighted by the same reciprocal coupling that generates
    the feedback field. For scalar sensitivity maps this uses ``|B1/I|^2``.
    """

    weights = np.asarray(density, dtype=np.float64)
    if sensitivity is not None:
        sens = np.asarray(sensitivity)
        if sens.shape != weights.shape:
            raise ValueError("sensitivity must have the same shape as density")
        weights = weights * np.abs(sens) ** 2
    if np.any(weights < 0.0):
        raise ValueError("density-derived radiation damping weights must be non-negative")
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("radiation damping weights must sum to a positive value")
    return (weights / total).reshape(-1)


@dataclass(frozen=True)
class RadiationDampingProbe:
    """Probe coupling parameters for radiation-damping simulations."""

    gamma: float
    omega0: float
    q: float
    fill_factor: float
    equilibrium_magnetization: float
    phase: float = 0.0
    detuning: float = 0.0
    name: str = "probe"

    @property
    def trd(self) -> float:
        return radiation_damping_time(
            self.gamma,
            self.fill_factor,
            self.equilibrium_magnetization,
            self.q,
        )

    @property
    def krd(self) -> float:
        """Radiation-damping feedback-field coefficient in T per (A/m).

        The dimensional feedback field is ``B_rd = krd * M_xy`` with
        ``krd = mu0 * eta * Q / 2`` (units T*m/A). This is a convenience for
        dimensional estimates; the simulator itself works in normalized units
        and drives the feedback through ``1 / trd`` rather than this property,
        so the two are consistent but ``krd`` is not used in the integrator.
        """

        return MU0 * self.fill_factor * self.q / 2.0

    @property
    def resonator_time_constant(self) -> float:
        """Probe ringdown time ``2Q/omega0`` in seconds."""

        return 2.0 * self.q / self.omega0


@dataclass(frozen=True)
class RadiationDampingResult:
    """Time-domain magnetization and probe feedback from an RD simulation."""

    time: np.ndarray
    mxy: np.ndarray
    mz: np.ndarray
    feedback: np.ndarray
    probe: RadiationDampingProbe
    model: str

    @property
    def envelope(self) -> np.ndarray:
        return np.abs(self.mxy)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.mxy)


@dataclass(frozen=True)
class RadiationDampingSpec:
    """Settings for RD-aware arbitrary-sequence propagation.

    The sequence kernels use normalized time units. ``time_scale`` converts
    physical seconds in ``probe`` to those units; for the current finite CPMG
    workflows this is ``(pi/2) / T_90``.
    """

    probe: RadiationDampingProbe
    time_scale: float
    weights: np.ndarray | None = None
    model: str = "instant"
    max_step: float | None = None
    apply_during_pulses: bool = False
    initial_feedback: complex = 0.0 + 0.0j

    @property
    def trd(self) -> float:
        return self.probe.trd * self.time_scale

    @property
    def resonator_time_constant(self) -> float:
        return self.probe.resonator_time_constant * self.time_scale

    @property
    def detuning(self) -> float:
        return self.probe.detuning / self.time_scale


def radiation_damping_probe_from_parameters(
    sp: Mapping[str, Any] | Any,
    *,
    fill_factor: float,
    equilibrium_magnetization: float | None = None,
    q: float | None = None,
    phase: float = 0.0,
    detuning: float = 0.0,
    name: str = "probe",
) -> RadiationDampingProbe:
    """Build a radiation-damping probe from existing tuned/matched ``sp``."""

    gamma = float(_field_or_default(sp, "gamma", PROTON_GAMMA))
    f0 = float(_field_or_default(sp, "f0", _field_or_default(sp, "fin", 0.0)))
    if f0 <= 0:
        w0 = float(_field_or_default(sp, "w0", 0.0))
    else:
        w0 = 2.0 * np.pi * f0
    if w0 <= 0:
        raise ValueError("probe parameters must supply positive f0, fin, or w0")
    probe_q = float(_field(sp, "Q") if q is None else q)
    if equilibrium_magnetization is None:
        mth_values = np.asarray(_field_or_default(sp, "mth", 1.0), dtype=np.float64)
        mth = float(np.mean(mth_values))
    else:
        mth = float(equilibrium_magnetization)
    return RadiationDampingProbe(
        gamma=gamma,
        omega0=w0,
        q=probe_q,
        fill_factor=fill_factor,
        equilibrium_magnetization=mth,
        phase=float(phase),
        detuning=float(detuning),
        name=name,
    )


def radiation_damping_probe_from_tuned(
    sp: Mapping[str, Any] | Any,
    *,
    fill_factor: float,
    equilibrium_magnetization: float | None = None,
    phase: float = 0.0,
    detuning: float = 0.0,
) -> RadiationDampingProbe:
    """Build an RD coupling object from a tuned-probe parameter set."""

    return radiation_damping_probe_from_parameters(
        sp,
        fill_factor=fill_factor,
        equilibrium_magnetization=equilibrium_magnetization,
        phase=phase,
        detuning=detuning,
        name="tuned",
    )


def radiation_damping_probe_from_matched(
    sp: Mapping[str, Any] | Any,
    *,
    fill_factor: float,
    equilibrium_magnetization: float | None = None,
    phase: float = 0.0,
    detuning: float = 0.0,
) -> RadiationDampingProbe:
    """Build an RD coupling object from a matched-probe parameter set."""

    return radiation_damping_probe_from_parameters(
        sp,
        fill_factor=fill_factor,
        equilibrium_magnetization=equilibrium_magnetization,
        phase=phase,
        detuning=detuning,
        name="matched",
    )


def initial_state_from_flip_angle(
    flip_angle: float,
    *,
    pulse_phase: float = 0.0,
    equilibrium_magnetization: float = 1.0,
) -> tuple[complex, float]:
    """Return the post-pulse normalized state for an ideal hard pulse.

    ``pulse_phase=0`` follows the package convention for an x-pulse, placing
    the transverse magnetization along ``-y`` after a positive flip.
    """

    theta = float(flip_angle)
    scale = float(equilibrium_magnetization)
    return (
        -1j * np.exp(1j * float(pulse_phase)) * scale * np.sin(theta),
        scale * float(np.cos(theta)),
    )


def analytic_radiation_damping_envelope(
    time: np.ndarray,
    flip_angle: float,
    trd: float,
    *,
    equilibrium_magnetization: float = 1.0,
    t2: float = np.inf,
) -> np.ndarray:
    """Analytic FID envelope for an on-resonance hard pulse with no T1 term."""

    time = np.asarray(time, dtype=np.float64)
    theta = float(flip_angle)
    if not (0.0 < theta < np.pi):
        raise ValueError("flip_angle must be between 0 and pi for the analytic form")
    if trd <= 0:
        raise ValueError("trd must be positive")
    arg = time / float(trd) - np.log(np.tan(theta / 2.0))
    envelope = float(equilibrium_magnetization) / np.cosh(arg)
    if np.isfinite(t2):
        if t2 <= 0:
            raise ValueError("t2 must be positive")
        envelope = envelope * np.exp(-time / float(t2))
    return envelope


def _drive_at(
    drive: complex | Callable[[float], complex] | None,
    t: float,
) -> complex:
    if drive is None:
        return 0.0 + 0.0j
    if callable(drive):
        return complex(drive(t))
    return complex(drive)


def _rhs(
    state: np.ndarray,
    t: float,
    probe: RadiationDampingProbe,
    t1: float,
    t2: float,
    equilibrium_mz: float,
    drive: complex | Callable[[float], complex] | None,
    model: str,
) -> np.ndarray:
    mxy = complex(state[0])
    mz = float(np.real(state[1]))
    feedback = complex(state[2])
    bxy = _drive_at(drive, t)

    target_feedback = np.exp(1j * probe.phase) * np.conj(mxy) / probe.trd
    if model == "instant":
        feedback = target_feedback
        dfeedback = 0.0 + 0.0j
    elif model == "circuit":
        tau = probe.resonator_time_constant
        dfeedback = (target_feedback - feedback) / tau - 1j * probe.detuning * feedback
    else:
        raise ValueError("model must be 'instant' or 'circuit'")

    dmxy = 1j * probe.gamma * mz * bxy + mz * feedback
    dmz = (1j * probe.gamma / 2.0) * (mxy * np.conj(bxy) - np.conj(mxy) * bxy)
    dmz += -np.real(np.conj(mxy) * feedback)
    if np.isfinite(t2):
        dmxy -= mxy / t2
    if np.isfinite(t1):
        dmz += (float(equilibrium_mz) - mz) / t1
    return np.array([dmxy, dmz, dfeedback], dtype=np.complex128)


def simulate_radiation_damping(
    time: np.ndarray,
    probe: RadiationDampingProbe,
    *,
    initial_mxy: complex,
    initial_mz: float,
    t1: float = np.inf,
    t2: float = np.inf,
    equilibrium_mz: float = 1.0,
    drive: complex | Callable[[float], complex] | None = None,
    model: str = "instant",
    initial_feedback: complex | None = None,
    max_step: float | None = None,
) -> RadiationDampingResult:
    """Integrate the rotating-frame Bloch equations with RD back-action."""

    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError("time must contain at least two samples")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    if t1 <= 0 and np.isfinite(t1):
        raise ValueError("t1 must be positive")
    if t2 <= 0 and np.isfinite(t2):
        raise ValueError("t2 must be positive")

    feedback0 = (
        np.exp(1j * probe.phase) * np.conj(complex(initial_mxy)) / probe.trd
        if initial_feedback is None and model == "instant"
        else (0.0 + 0.0j if initial_feedback is None else complex(initial_feedback))
    )
    state = np.array([complex(initial_mxy), float(initial_mz), feedback0], dtype=np.complex128)
    history = np.zeros((time.size, 3), dtype=np.complex128)
    history[0, :] = state

    step_limit_candidates = [probe.trd / 50.0]
    if model == "circuit":
        step_limit_candidates.append(probe.resonator_time_constant / 20.0)
    if np.isfinite(t1):
        step_limit_candidates.append(float(t1) / 50.0)
    if np.isfinite(t2):
        step_limit_candidates.append(float(t2) / 50.0)
    if max_step is not None:
        if max_step <= 0:
            raise ValueError("max_step must be positive")
        step_limit_candidates.append(float(max_step))
    step_limit = min(step_limit_candidates)

    for idx in range(time.size - 1):
        t = float(time[idx])
        h_total = float(time[idx + 1] - time[idx])
        steps = max(1, int(np.ceil(h_total / step_limit)))
        h = h_total / steps
        for _ in range(steps):
            k1 = _rhs(state, t, probe, t1, t2, equilibrium_mz, drive, model)
            k2 = _rhs(
                state + h * k1 / 2.0,
                t + h / 2.0,
                probe,
                t1,
                t2,
                equilibrium_mz,
                drive,
                model,
            )
            k3 = _rhs(
                state + h * k2 / 2.0,
                t + h / 2.0,
                probe,
                t1,
                t2,
                equilibrium_mz,
                drive,
                model,
            )
            k4 = _rhs(state + h * k3, t + h, probe, t1, t2, equilibrium_mz, drive, model)
            state = state + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            t += h
        history[idx + 1, :] = state

    return RadiationDampingResult(
        time=time,
        mxy=history[:, 0],
        mz=np.real(history[:, 1]),
        feedback=history[:, 2],
        probe=probe,
        model=model,
    )


def simulate_radiation_damping_fid(
    time: np.ndarray,
    probe: RadiationDampingProbe,
    *,
    flip_angle: float = np.pi / 2,
    pulse_phase: float = 0.0,
    t1: float = np.inf,
    t2: float = np.inf,
    equilibrium_mz: float = 1.0,
    model: str = "instant",
    max_step: float | None = None,
) -> RadiationDampingResult:
    """Simulate an FID after an ideal hard pulse in the RD model."""

    mxy0, mz0 = initial_state_from_flip_angle(flip_angle, pulse_phase=pulse_phase)
    return simulate_radiation_damping(
        time,
        probe,
        initial_mxy=mxy0,
        initial_mz=mz0,
        t1=t1,
        t2=t2,
        equilibrium_mz=equilibrium_mz,
        model=model,
        max_step=max_step,
    )


def simulate_nmr_maser(
    time: np.ndarray,
    probe: RadiationDampingProbe,
    *,
    seed_mxy: complex = -1e-6j,
    initial_mz: float = -1.0,
    pump_mz: float = -1.0,
    t1: float,
    t2: float,
    model: str = "circuit",
    initial_feedback: complex | None = None,
    max_step: float | None = None,
) -> RadiationDampingResult:
    """Simulate an idealized pumped NMR maser in the RD feedback model.

    The pump is represented as longitudinal relaxation toward ``pump_mz``.
    Maser gain requires an inverted pump with roughly
    ``-pump_mz / Trd > 1 / T2`` in the favorable feedback quadrature.
    """

    if t1 <= 0 or not np.isfinite(t1):
        raise ValueError("t1 must be finite and positive for a pumped maser")
    if t2 <= 0 or not np.isfinite(t2):
        raise ValueError("t2 must be finite and positive for a pumped maser")
    if seed_mxy == 0:
        raise ValueError("seed_mxy must be nonzero")
    return simulate_radiation_damping(
        time,
        probe,
        initial_mxy=seed_mxy,
        initial_mz=initial_mz,
        t1=t1,
        t2=t2,
        equilibrium_mz=pump_mz,
        model=model,
        initial_feedback=initial_feedback,
        max_step=max_step,
    )


__all__ = [
    "AVOGADRO",
    "HBAR",
    "KB",
    "MU0",
    "PROTON_GAMMA",
    "RadiationDampingProbe",
    "RadiationDampingResult",
    "RadiationDampingSample",
    "RadiationDampingSpec",
    "analytic_radiation_damping_envelope",
    "hyperpolarized_proton_sample",
    "initial_state_from_flip_angle",
    "normalized_radiation_damping_weights",
    "proton_thermal_magnetization_density",
    "radiation_damping_probe_from_matched",
    "radiation_damping_probe_from_parameters",
    "radiation_damping_probe_from_tuned",
    "radiation_damping_time",
    "simulate_radiation_damping",
    "simulate_radiation_damping_fid",
    "simulate_nmr_maser",
    "water_proton_sample",
]
