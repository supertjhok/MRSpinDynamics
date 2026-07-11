"""High-level quadrupolar-relaxation and field-cycling/NMRD workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from spin_dynamics.nqr.hamiltonians import diagonalize_site, nqr_hamiltonian
from spin_dynamics.nqr.systems import QuadrupolarSite
from spin_dynamics.relaxation import (
    IsotropicLiquidMotionalAveraging,
    RedfieldEFGRelaxationModel,
    VibrationalMotionalAveraging,
    arrhenius_correlation_time,
    bpp_relaxation_rates,
)


@dataclass(frozen=True)
class QuadrupolarRelaxationWorkflowResult:
    """Transition-resolved initial decay rates from an EFG Redfield model.

    Rate arrays have shape ``(n_correlation_times, n_transitions)``. ``R1`` is
    the initial decay rate of the transition population difference and ``R2``
    is the initial decay rate of its energy-basis coherence. These are local
    rates; a strongly coupled multiexponential decay should be inspected with
    the full relaxation superoperator instead of being reduced to one lifetime.
    """

    site: QuadrupolarSite
    correlation_time_seconds: np.ndarray
    transition_labels: tuple[str, ...]
    transition_frequencies_hz: np.ndarray
    r1_per_second: np.ndarray
    r2_per_second: np.ndarray
    t1_seconds: np.ndarray
    t2_seconds: np.ndarray
    temperature_kelvin: np.ndarray | None = None


@dataclass(frozen=True)
class NMRDWorkflowResult:
    """Field-cycling/NMRD grid with temperature rows and frequency columns."""

    larmor_frequency_hz: np.ndarray
    field_tesla: np.ndarray | None
    temperature_kelvin: np.ndarray | None
    correlation_time_seconds: np.ndarray
    r1_per_second: np.ndarray
    r2_per_second: np.ndarray
    t1_seconds: np.ndarray
    t2_seconds: np.ndarray
    j0_seconds: np.ndarray
    jw_seconds: np.ndarray
    j2w_seconds: np.ndarray

    @property
    def relaxivity_shape(self) -> tuple[int, int]:
        """Return ``(number of conditions, number of fields)``."""

        return self.r1_per_second.shape


def _positive_axis(values: float | Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    axis = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)) or np.any(axis <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return axis


def _nonnegative_axis(
    values: float | Iterable[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    axis = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if axis.ndim != 1 or axis.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(axis)) or np.any(axis < 0.0):
        raise ValueError(f"{name} must be non-negative and finite")
    return axis


def _initial_decay_rate(superoperator: np.ndarray, state: np.ndarray) -> float:
    vector = np.asarray(state, dtype=np.complex128).reshape(-1, order="F")
    norm = float(np.vdot(vector, vector).real)
    rate = -float(np.vdot(vector, superoperator @ vector).real / norm)
    if rate < -1.0e-10:
        raise ValueError("relaxation superoperator predicts a growing mode")
    return max(rate, 0.0)


def run_quadrupolar_relaxation(
    site: QuadrupolarSite,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
    *,
    fluctuation_amplitude_hz: float,
    b0_vector_tesla_pas: Iterable[float] | np.ndarray | None = None,
    vibration_frequency_hz: float = 0.0,
    temperature_kelvin: Iterable[float] | np.ndarray | None = None,
) -> QuadrupolarRelaxationWorkflowResult:
    """Calculate transition-resolved quadrupolar relaxation for one NQR site.

    ``fluctuation_amplitude_hz`` is the RMS fluctuating EFG interaction in the
    convention of :class:`~spin_dynamics.relaxation.RedfieldEFGRelaxationModel`.
    Set ``vibration_frequency_hz`` to use a symmetric pair of shifted
    Lorentzians; zero uses ordinary isotropic exponential correlation.
    """

    if not isinstance(site, QuadrupolarSite):
        raise TypeError("site must be a QuadrupolarSite")
    tau = _positive_axis(correlation_time_seconds, "correlation_time_seconds")
    amplitude = float(fluctuation_amplitude_hz)
    vibration = float(vibration_frequency_hz)
    if not np.isfinite(amplitude) or amplitude < 0.0:
        raise ValueError("fluctuation_amplitude_hz must be non-negative and finite")
    if not np.isfinite(vibration) or vibration < 0.0:
        raise ValueError("vibration_frequency_hz must be non-negative and finite")
    temperatures = None
    if temperature_kelvin is not None:
        temperatures = _positive_axis(temperature_kelvin, "temperature_kelvin")
        if temperatures.shape != tau.shape:
            raise ValueError("temperature_kelvin must match correlation_time_seconds")

    eigensystem = diagonalize_site(site, b0_vector_tesla_pas)
    hamiltonian = nqr_hamiltonian(site, b0_vector_tesla_pas)
    n_transitions = len(eigensystem.transitions)
    r1 = np.empty((tau.size, n_transitions), dtype=np.float64)
    r2 = np.empty_like(r1)
    for tau_index, correlation_time in enumerate(tau):
        if vibration == 0.0:
            motion = IsotropicLiquidMotionalAveraging(float(correlation_time))
        else:
            motion = VibrationalMotionalAveraging(
                float(correlation_time),
                2.0 * np.pi * vibration,
            )
        model = RedfieldEFGRelaxationModel(
            spin=site.spin,
            fluctuation_amplitude_hz=amplitude,
            motion=motion,
        )
        relaxation = model.superoperator(hamiltonian)
        for transition_index, transition in enumerate(eigensystem.transitions):
            lower = eigensystem.eigenvectors[:, transition.lower]
            upper = eigensystem.eigenvectors[:, transition.upper]
            population_difference = np.outer(upper, upper.conj()) - np.outer(
                lower,
                lower.conj(),
            )
            coherence = np.outer(upper, lower.conj())
            r1[tau_index, transition_index] = _initial_decay_rate(
                relaxation,
                population_difference,
            )
            r2[tau_index, transition_index] = _initial_decay_rate(
                relaxation,
                coherence,
            )

    return QuadrupolarRelaxationWorkflowResult(
        site=site,
        correlation_time_seconds=tau.copy(),
        transition_labels=tuple(item.label for item in eigensystem.transitions),
        transition_frequencies_hz=np.array(
            [item.frequency_hz for item in eigensystem.transitions]
        ),
        r1_per_second=r1,
        r2_per_second=r2,
        t1_seconds=np.divide(1.0, r1, out=np.full_like(r1, np.inf), where=r1 > 0.0),
        t2_seconds=np.divide(1.0, r2, out=np.full_like(r2, np.inf), where=r2 > 0.0),
        temperature_kelvin=None if temperatures is None else temperatures.copy(),
    )


def run_arrhenius_quadrupolar_relaxation(
    site: QuadrupolarSite,
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    *,
    tau_ref_seconds: float,
    reference_temperature_kelvin: float,
    activation_energy_j_per_mol: float,
    fluctuation_amplitude_hz: float,
    b0_vector_tesla_pas: Iterable[float] | np.ndarray | None = None,
    vibration_frequency_hz: float = 0.0,
) -> QuadrupolarRelaxationWorkflowResult:
    """Run the transition workflow with one Arrhenius correlation-time law."""

    temperature = _positive_axis(temperature_kelvin, "temperature_kelvin")
    tau = arrhenius_correlation_time(
        temperature,
        tau_ref_seconds=tau_ref_seconds,
        reference_temperature_kelvin=reference_temperature_kelvin,
        activation_energy_j_per_mol=activation_energy_j_per_mol,
    )
    return run_quadrupolar_relaxation(
        site,
        tau,
        fluctuation_amplitude_hz=fluctuation_amplitude_hz,
        b0_vector_tesla_pas=b0_vector_tesla_pas,
        vibration_frequency_hz=vibration_frequency_hz,
        temperature_kelvin=temperature,
    )


def run_nmrd(
    larmor_frequency_hz: float | Iterable[float] | np.ndarray,
    correlation_time_seconds: float | Iterable[float] | np.ndarray,
    *,
    coupling_scale_per_second2: float,
    temperature_kelvin: Iterable[float] | np.ndarray | None = None,
    field_tesla: Iterable[float] | np.ndarray | None = None,
    baseline_r1_per_second: float = 0.0,
    baseline_r2_per_second: float = 0.0,
) -> NMRDWorkflowResult:
    """Run a BPP NMRD sweep over frequency for one or more conditions.

    Rows correspond to correlation times (or temperatures); columns correspond
    to Larmor frequencies. The returned spectral densities make it possible to
    see which part of the dispersion produces a fitted rate.
    """

    frequency = _nonnegative_axis(larmor_frequency_hz, "larmor_frequency_hz")
    tau = _positive_axis(correlation_time_seconds, "correlation_time_seconds")
    temperatures = None
    if temperature_kelvin is not None:
        temperatures = _positive_axis(temperature_kelvin, "temperature_kelvin")
        if temperatures.shape != tau.shape:
            raise ValueError("temperature_kelvin must match correlation_time_seconds")
    fields = None
    if field_tesla is not None:
        fields = _nonnegative_axis(field_tesla, "field_tesla")
        if fields.shape != frequency.shape:
            raise ValueError("field_tesla must match larmor_frequency_hz")

    temperature_grid = None if temperatures is None else temperatures[:, None]
    rates = bpp_relaxation_rates(
        angular_frequency_rad_per_s=2.0 * np.pi * frequency[None, :],
        correlation_time_seconds=tau[:, None],
        temperature_kelvin=temperature_grid,
        coupling_scale_per_second2=coupling_scale_per_second2,
        baseline_r1_per_second=baseline_r1_per_second,
        baseline_r2_per_second=baseline_r2_per_second,
    )
    return NMRDWorkflowResult(
        larmor_frequency_hz=frequency.copy(),
        field_tesla=None if fields is None else fields.copy(),
        temperature_kelvin=None if temperatures is None else temperatures.copy(),
        correlation_time_seconds=tau.copy(),
        r1_per_second=rates.r1_per_second,
        r2_per_second=rates.r2_per_second,
        t1_seconds=rates.t1_seconds,
        t2_seconds=rates.t2_seconds,
        j0_seconds=rates.j0_seconds,
        jw_seconds=rates.jw_seconds,
        j2w_seconds=rates.j2w_seconds,
    )


def run_field_cycling_nmrd(
    field_tesla: float | Iterable[float] | np.ndarray,
    temperature_kelvin: float | Iterable[float] | np.ndarray,
    *,
    gamma_hz_per_t: float,
    tau_ref_seconds: float,
    reference_temperature_kelvin: float,
    activation_energy_j_per_mol: float,
    coupling_scale_per_second2: float,
    baseline_r1_per_second: float = 0.0,
    baseline_r2_per_second: float = 0.0,
) -> NMRDWorkflowResult:
    """Run an Arrhenius BPP field-cycling experiment on a field grid."""

    fields = _nonnegative_axis(field_tesla, "field_tesla")
    gamma = float(gamma_hz_per_t)
    if not np.isfinite(gamma) or gamma == 0.0:
        raise ValueError("gamma_hz_per_t must be finite and non-zero")
    temperature = _positive_axis(temperature_kelvin, "temperature_kelvin")
    tau = arrhenius_correlation_time(
        temperature,
        tau_ref_seconds=tau_ref_seconds,
        reference_temperature_kelvin=reference_temperature_kelvin,
        activation_energy_j_per_mol=activation_energy_j_per_mol,
    )
    return run_nmrd(
        np.abs(gamma) * fields,
        tau,
        coupling_scale_per_second2=coupling_scale_per_second2,
        temperature_kelvin=temperature,
        field_tesla=fields,
        baseline_r1_per_second=baseline_r1_per_second,
        baseline_r2_per_second=baseline_r2_per_second,
    )


__all__ = [
    "NMRDWorkflowResult",
    "QuadrupolarRelaxationWorkflowResult",
    "run_arrhenius_quadrupolar_relaxation",
    "run_field_cycling_nmrd",
    "run_nmrd",
    "run_quadrupolar_relaxation",
]
