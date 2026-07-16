"""Adapters from Bayesian design actions to declarative experiments.

An adapter binds one posterior particle and one controllable action to a
simulation-only :class:`~spin_dynamics.experiment.Experiment`, executes the
existing validated workflow, extracts its observable, and reports physical
acquisition time.  :func:`make_adapter_session` always installs
:class:`ExperimentPlanConstraint`, so infeasible actions are rejected by the
same static planner used by ordinary experiments before utility is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from spin_dynamics.design.constraints import ConstraintResult, DesignConstraint
from spin_dynamics.design.models import PredictiveModel
from spin_dynamics.design.posterior import ParticlePosterior
from spin_dynamics.design.session import AdaptiveDesignSession
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import ObservationLikelihood, StopRule
from spin_dynamics.design.utilities import DesignUtility
from spin_dynamics.esr.systems import ESRSpinSystem
from spin_dynamics.experiment import (
    CPMGIRTrain,
    ESRHahnEcho,
    Experiment,
    NQRFID,
    PGSE,
)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _index(values: Any, index: int | None, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        raise ValueError(f"workflow returned an empty {name}")
    if index is None:
        return array.copy()
    try:
        return np.asarray(array[index])
    except IndexError as exc:
        raise ValueError(f"{name} index {index} is out of range") from exc


def _scaled(values: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
    scale = _finite(parameters.get("signal_scale", 1.0), "signal_scale")
    baseline = _finite(parameters.get("baseline", 0.0), "baseline")
    return scale * np.asarray(values) + baseline


def _nominal(mapping: Mapping[str, float]) -> dict[str, float]:
    values = {str(name): _finite(value, str(name)) for name, value in mapping.items()}
    if not values:
        raise ValueError("nominal_parameters must not be empty")
    return values


def _overheads(fixed: float, recovery: float) -> tuple[float, float]:
    return (
        _nonnegative(fixed, "fixed_overhead_seconds"),
        _nonnegative(recovery, "recovery_seconds"),
    )


@dataclass(frozen=True)
class CPMGIRDesign:
    """One inversion delay for a CPMG inversion-recovery acquisition."""

    delay_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "delay_seconds", _nonnegative(self.delay_seconds, "delay_seconds")
        )


@dataclass(frozen=True)
class PGSEDesign:
    """Gradient amplitude and timing for one deterministic PGSE acquisition."""

    gradient_amplitude_t_per_m: float
    gradient_duration_seconds: float
    diffusion_time_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gradient_amplitude_t_per_m",
            _nonnegative(
                self.gradient_amplitude_t_per_m, "gradient_amplitude_t_per_m"
            ),
        )
        duration = _positive(
            self.gradient_duration_seconds, "gradient_duration_seconds"
        )
        diffusion = _positive(self.diffusion_time_seconds, "diffusion_time_seconds")
        if diffusion < duration:
            raise ValueError(
                "diffusion_time_seconds must be at least gradient_duration_seconds"
            )
        object.__setattr__(self, "gradient_duration_seconds", duration)
        object.__setattr__(self, "diffusion_time_seconds", diffusion)


@dataclass(frozen=True)
class NQRFrequencyDesign:
    """NQR carrier selection with optional pulse duration and nutation rate."""

    rf_frequency_hz: float
    pulse_duration_seconds: float | None = None
    nutation_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rf_frequency_hz", _positive(self.rf_frequency_hz, "rf_frequency_hz")
        )
        if self.pulse_duration_seconds is not None:
            object.__setattr__(
                self,
                "pulse_duration_seconds",
                _positive(self.pulse_duration_seconds, "pulse_duration_seconds"),
            )
        if self.nutation_hz is not None:
            object.__setattr__(
                self, "nutation_hz", _positive(self.nutation_hz, "nutation_hz")
            )


@dataclass(frozen=True)
class ESRDelayDesign:
    """Hahn-echo delay with an optional explicitly selected ESR carrier."""

    tau_seconds: float
    rf_frequency_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tau_seconds", _positive(self.tau_seconds, "tau_seconds"))
        if self.rf_frequency_hz is not None:
            object.__setattr__(
                self, "rf_frequency_hz", _positive(self.rf_frequency_hz, "rf_frequency_hz")
            )


class ExperimentDesignAdapter(Protocol):
    """Contract implemented by experiment-facade design adapters."""

    nominal_parameters: Mapping[str, float]

    def build_experiment(
        self, parameters: Mapping[str, float], design: Any
    ) -> Experiment: ...

    def extract_observable(
        self, result: Any, parameters: Mapping[str, float]
    ) -> np.ndarray: ...

    def physical_seconds(self, design: Any) -> float: ...

    def plan(self, design: Any) -> Any: ...


class _AdapterMixin:
    template: Experiment
    nominal_parameters: Mapping[str, float]

    def plan(self, design: Any) -> Any:
        return self.build_experiment(self.nominal_parameters, design).plan(estimate=False)

    def simulate(self, parameters: Mapping[str, float], design: Any) -> np.ndarray:
        experiment = self.build_experiment(parameters, design)
        if experiment.acquisition.noise is not None:
            raise ValueError(
                "adapter templates must disable acquisition.noise; noise belongs in "
                "the Bayesian observation likelihood"
            )
        record = experiment.run()
        observable = np.asarray(self.extract_observable(record.result, parameters))
        if np.any(~np.isfinite(observable)):
            raise ValueError("adapter observable must be finite")
        return observable


@dataclass(frozen=True, eq=False)
class CPMGIRAdapter(_AdapterMixin):
    """Bind ``t1_seconds`` and ``t2_seconds`` to a ``CPMGIRTrain`` template.

    The action selects one inversion delay.  The observable is either the full
    complex echo-integral vector or one echo selected by ``echo_index``.
    Optional particle fields ``signal_scale`` and ``baseline`` apply an affine
    calibration after simulation.
    """

    template: Experiment
    nominal_parameters: Mapping[str, float]
    echo_index: int | None = None
    fixed_overhead_seconds: float = 0.0
    recovery_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.template.sequence, CPMGIRTrain):
            raise TypeError("CPMGIRAdapter requires a CPMGIRTrain template")
        if self.template.acquisition.noise is not None:
            raise ValueError("adapter template acquisition.noise must be None")
        object.__setattr__(self, "nominal_parameters", _nominal(self.nominal_parameters))
        fixed, recovery = _overheads(
            self.fixed_overhead_seconds, self.recovery_seconds
        )
        object.__setattr__(self, "fixed_overhead_seconds", fixed)
        object.__setattr__(self, "recovery_seconds", recovery)

    def build_experiment(
        self, parameters: Mapping[str, float], design: CPMGIRDesign
    ) -> Experiment:
        if not isinstance(design, CPMGIRDesign):
            raise TypeError("CPMGIRAdapter requires CPMGIRDesign actions")
        sample = self.template.sample
        t1 = parameters.get("t1_seconds", sample.t1_seconds)
        t2 = parameters.get("t2_seconds", sample.t2_seconds)
        if t1 is None or t2 is None:
            raise ValueError("t1_seconds and t2_seconds must be bound or set in template")
        return replace(
            self.template,
            sequence=replace(self.template.sequence, tauvect=(design.delay_seconds,)),
            sample=replace(
                sample,
                t1_seconds=_positive(t1, "t1_seconds"),
                t2_seconds=_positive(t2, "t2_seconds"),
            ),
        )

    def extract_observable(
        self, result: Any, parameters: Mapping[str, float]
    ) -> np.ndarray:
        values = np.asarray(result.echo_integrals)[0]
        return _scaled(_index(values, self.echo_index, "echo_integrals"), parameters)

    def physical_seconds(self, design: CPMGIRDesign) -> float:
        sequence = self.build_experiment(self.nominal_parameters, design).sequence
        return float(
            self.fixed_overhead_seconds
            + self.recovery_seconds
            + design.delay_seconds
            + sequence.num_echoes * sequence.echo_spacing_seconds
        )


@dataclass(frozen=True, eq=False)
class PGSEAdapter(_AdapterMixin):
    """Bind diffusion and T2 parameters to a deterministic ``PGSE`` template."""

    template: Experiment
    nominal_parameters: Mapping[str, float]
    echo_index: int | None = None
    fixed_overhead_seconds: float = 0.0
    recovery_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.template.sequence, PGSE):
            raise TypeError("PGSEAdapter requires a PGSE template")
        if self.template.acquisition.noise is not None:
            raise ValueError("adapter template acquisition.noise must be None")
        object.__setattr__(self, "nominal_parameters", _nominal(self.nominal_parameters))
        fixed, recovery = _overheads(
            self.fixed_overhead_seconds, self.recovery_seconds
        )
        object.__setattr__(self, "fixed_overhead_seconds", fixed)
        object.__setattr__(self, "recovery_seconds", recovery)

    def build_experiment(
        self, parameters: Mapping[str, float], design: PGSEDesign
    ) -> Experiment:
        if not isinstance(design, PGSEDesign):
            raise TypeError("PGSEAdapter requires PGSEDesign actions")
        sample = self.template.sample
        diffusion = parameters.get(
            "diffusion_coefficient", sample.diffusion_coefficient
        )
        t2 = parameters.get("t2_seconds", sample.t2_seconds)
        if diffusion is None or t2 is None:
            raise ValueError(
                "diffusion_coefficient and t2_seconds must be bound or set in template"
            )
        return replace(
            self.template,
            sequence=replace(
                self.template.sequence,
                gradient_amplitude=design.gradient_amplitude_t_per_m,
                gradient_duration=design.gradient_duration_seconds,
                diffusion_time=design.diffusion_time_seconds,
            ),
            sample=replace(
                sample,
                diffusion_coefficient=_nonnegative(
                    diffusion, "diffusion_coefficient"
                ),
                t2_seconds=_positive(t2, "t2_seconds"),
            ),
        )

    def extract_observable(
        self, result: Any, parameters: Mapping[str, float]
    ) -> np.ndarray:
        return _scaled(_index(result.signal, self.echo_index, "signal"), parameters)

    def physical_seconds(self, design: PGSEDesign) -> float:
        sequence = self.build_experiment(self.nominal_parameters, design).sequence
        first_echo = sequence.first_echo_time_seconds
        if first_echo is None:
            first_echo = 2.0 * sequence.diffusion_time
        spacing = sequence.echo_spacing_seconds
        if spacing is None:
            spacing = first_echo
        final_echo = first_echo + (sequence.num_echoes - 1) * spacing
        return float(self.fixed_overhead_seconds + self.recovery_seconds + final_echo)


@dataclass(frozen=True, eq=False)
class NQRFIDAdapter(_AdapterMixin):
    """Bind NQR site properties to an ``NQRFID`` carrier/pulse action.

    ``quadrupole_frequency_hz`` and ``eta`` replace the corresponding fields
    of the template's :class:`~spin_dynamics.nqr.QuadrupolarSite`.  The full
    complex FID is the default observable; ``sample_index`` selects one point.
    """

    template: Experiment
    nominal_parameters: Mapping[str, float]
    sample_index: int | None = None
    fixed_overhead_seconds: float = 0.0
    recovery_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.template.sequence, NQRFID):
            raise TypeError("NQRFIDAdapter requires an NQRFID template")
        if self.template.sample.site is None:
            raise ValueError("NQRFIDAdapter template requires sample.site")
        if self.template.acquisition.noise is not None:
            raise ValueError("adapter template acquisition.noise must be None")
        object.__setattr__(self, "nominal_parameters", _nominal(self.nominal_parameters))
        fixed, recovery = _overheads(
            self.fixed_overhead_seconds, self.recovery_seconds
        )
        object.__setattr__(self, "fixed_overhead_seconds", fixed)
        object.__setattr__(self, "recovery_seconds", recovery)

    def build_experiment(
        self, parameters: Mapping[str, float], design: NQRFrequencyDesign
    ) -> Experiment:
        if not isinstance(design, NQRFrequencyDesign):
            raise TypeError("NQRFIDAdapter requires NQRFrequencyDesign actions")
        site = self.template.sample.site
        frequency = parameters.get(
            "quadrupole_frequency_hz", site.quadrupole_frequency_hz
        )
        eta = parameters.get("eta", site.eta)
        sequence = replace(
            self.template.sequence,
            rf_frequency_hz=design.rf_frequency_hz,
            pulse_duration_seconds=(
                design.pulse_duration_seconds
                if design.pulse_duration_seconds is not None
                else self.template.sequence.pulse_duration_seconds
            ),
            nutation_hz=(
                design.nutation_hz
                if design.nutation_hz is not None
                else self.template.sequence.nutation_hz
            ),
        )
        bound_site = replace(
            site,
            quadrupole_frequency_hz=_positive(
                frequency, "quadrupole_frequency_hz"
            ),
            eta=_nonnegative(eta, "eta"),
        )
        return replace(
            self.template,
            sequence=sequence,
            sample=replace(self.template.sample, site=bound_site),
        )

    def extract_observable(
        self, result: Any, parameters: Mapping[str, float]
    ) -> np.ndarray:
        return _scaled(_index(result.signal, self.sample_index, "signal"), parameters)

    def physical_seconds(self, design: NQRFrequencyDesign) -> float:
        sequence = self.build_experiment(self.nominal_parameters, design).sequence
        return float(
            self.fixed_overhead_seconds
            + self.recovery_seconds
            + sequence.pulse_duration_seconds
            + sequence.acquisition_seconds
        )


@dataclass(frozen=True, eq=False)
class ESRHahnAdapter(_AdapterMixin):
    """Bind ESR T2 or an isotropic g factor to a Hahn-echo delay action."""

    template: Experiment
    nominal_parameters: Mapping[str, float]
    sample_index: int | None = None
    fixed_overhead_seconds: float = 0.0
    recovery_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.template.sequence, ESRHahnEcho):
            raise TypeError("ESRHahnAdapter requires an ESRHahnEcho template")
        if self.template.sample.esr_system is None:
            raise ValueError("ESRHahnAdapter template requires sample.esr_system")
        if self.template.acquisition.noise is not None:
            raise ValueError("adapter template acquisition.noise must be None")
        object.__setattr__(self, "nominal_parameters", _nominal(self.nominal_parameters))
        fixed, recovery = _overheads(
            self.fixed_overhead_seconds, self.recovery_seconds
        )
        object.__setattr__(self, "fixed_overhead_seconds", fixed)
        object.__setattr__(self, "recovery_seconds", recovery)

    def build_experiment(
        self, parameters: Mapping[str, float], design: ESRDelayDesign
    ) -> Experiment:
        if not isinstance(design, ESRDelayDesign):
            raise TypeError("ESRHahnAdapter requires ESRDelayDesign actions")
        sequence = self.template.sequence
        t2 = parameters.get("t2_seconds", sequence.t2_seconds)
        if t2 is None:
            raise ValueError("t2_seconds must be bound or set in template")
        bound_sequence = replace(
            sequence,
            tau_seconds=design.tau_seconds,
            rf_frequency_hz=(
                design.rf_frequency_hz
                if design.rf_frequency_hz is not None
                else sequence.rf_frequency_hz
            ),
            t2_seconds=_positive(t2, "t2_seconds"),
        )
        system = self.template.sample.esr_system
        if "g_factor" in parameters:
            system = ESRSpinSystem(
                g_tensor=_positive(parameters["g_factor"], "g_factor"),
                spin=system.spin,
                label=system.label,
            )
        return replace(
            self.template,
            sequence=bound_sequence,
            sample=replace(self.template.sample, esr_system=system),
        )

    def extract_observable(
        self, result: Any, parameters: Mapping[str, float]
    ) -> np.ndarray:
        values = _index(result.signal, self.sample_index, "signal")
        return _scaled(values, parameters)

    def physical_seconds(self, design: ESRDelayDesign) -> float:
        sequence = self.build_experiment(self.nominal_parameters, design).sequence
        refocus = sequence.refocus_duration_seconds
        if refocus is None:
            refocus = 2.0 * sequence.excitation_duration_seconds
        acquisition = sequence.acquisition_seconds
        if acquisition is None:
            acquisition = 2.0 * sequence.tau_seconds
        return float(
            self.fixed_overhead_seconds
            + self.recovery_seconds
            + sequence.excitation_duration_seconds
            + sequence.tau_seconds
            + refocus
            + acquisition
        )


class ExperimentPredictor:
    """Vectorize an experiment adapter over posterior particles.

    Simulator outputs are cached by action and scalar particle values.  This
    matters because nested Monte Carlo utilities and posterior updates reuse
    the same deterministic predictions.
    """

    def __init__(self, adapter: ExperimentDesignAdapter, *, cache: bool = True) -> None:
        self.adapter = adapter
        self.cache = bool(cache)
        self._cache: dict[tuple[str, tuple[tuple[str, float], ...]], np.ndarray] = {}

    def clear_cache(self) -> None:
        """Discard all cached simulator observables."""

        self._cache.clear()

    def __call__(self, parameters: Mapping[str, np.ndarray], design: Any) -> np.ndarray:
        if not parameters:
            raise ValueError("parameters must not be empty")
        arrays = {name: np.asarray(value) for name, value in parameters.items()}
        counts = {value.shape[0] for value in arrays.values() if value.ndim > 0}
        if any(value.ndim == 0 for value in arrays.values()) or len(counts) != 1:
            raise ValueError("parameter arrays must share one leading particle dimension")
        count = counts.pop()
        predictions: list[np.ndarray] = []
        for index in range(count):
            particle: dict[str, float] = {}
            for name, values in arrays.items():
                item = np.asarray(values[index])
                if item.shape != ():
                    raise ValueError(
                        "experiment adapters currently require scalar latent parameters"
                    )
                particle[name] = _finite(item.item(), name)
            key = (repr(design), tuple(sorted(particle.items())))
            if self.cache and key in self._cache:
                observable = self._cache[key]
            else:
                simulate = getattr(self.adapter, "simulate", None)
                if simulate is None:
                    experiment = self.adapter.build_experiment(particle, design)
                    record = experiment.run()
                    observable = np.asarray(
                        self.adapter.extract_observable(record.result, particle)
                    )
                else:
                    observable = np.asarray(simulate(particle, design))
                if self.cache:
                    self._cache[key] = observable.copy()
            predictions.append(observable)
        try:
            return np.stack(predictions, axis=0)
        except ValueError as exc:
            raise ValueError(
                "adapter observables must have the same shape for every particle"
            ) from exc


@dataclass(frozen=True)
class ExperimentPlanConstraint:
    """Reject an action unless its nominal experiment plan has no errors."""

    adapter: ExperimentDesignAdapter

    def evaluate(self, design: Any) -> ConstraintResult:
        try:
            plan = self.adapter.plan(design)
        except (TypeError, ValueError) as exc:
            return ConstraintResult(False, f"experiment adapter rejected action: {exc}")
        if plan.ok:
            return ConstraintResult(True)
        return ConstraintResult(False, "; ".join(plan.errors))


@dataclass(frozen=True)
class ExperimentAdapterCost:
    """Expose an adapter's physical-duration model as a design cost."""

    adapter: ExperimentDesignAdapter

    def seconds(self, design: Any) -> float:
        return _positive(self.adapter.physical_seconds(design), "physical_seconds")


def make_adapter_model(
    adapter: ExperimentDesignAdapter,
    likelihood: ObservationLikelihood,
    *,
    cache: bool = True,
) -> PredictiveModel:
    """Create a likelihood-backed predictive model from an adapter."""

    return PredictiveModel(ExperimentPredictor(adapter, cache=cache), likelihood)


def make_adapter_session(
    *,
    adapter: ExperimentDesignAdapter,
    likelihood: ObservationLikelihood,
    posterior: ParticlePosterior,
    design_space: CandidateDesignSpace,
    utility: DesignUtility,
    constraints: Sequence[DesignConstraint] = (),
    stopping_rule: StopRule | None = None,
    seed: int | None = None,
    resample_fraction: float | None = None,
    cache: bool = True,
) -> AdaptiveDesignSession:
    """Build an adaptive session with mandatory experiment-plan validation."""

    return AdaptiveDesignSession(
        model=make_adapter_model(adapter, likelihood, cache=cache),
        posterior=posterior,
        design_space=design_space,
        utility=utility,
        cost=ExperimentAdapterCost(adapter),
        constraints=(ExperimentPlanConstraint(adapter), *tuple(constraints)),
        stopping_rule=stopping_rule,
        seed=seed,
        resample_fraction=resample_fraction,
    )
