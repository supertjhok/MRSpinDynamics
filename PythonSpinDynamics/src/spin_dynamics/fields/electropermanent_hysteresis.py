"""Return-point hysteresis and neighbor-coupled EPM programming.

The rate-independent model is a weighted bank of scalar play operators.  It
provides exact return-point memory and wiping-out behavior without pretending
that a static AlNiCo B-H table contains pulse-history information.  Model
parameters are therefore calibration objects with explicit evidence.

The coupling layer maps retained remanence to the axial H bias at every EPM
element.  Off-diagonal terms can be calculated from the existing finite-magnet
field solver, while diagonal terms represent user-supplied self-demagnetizing
factors.  Programming one element then uses a fixed-point iteration so its
state-dependent inductance, bias field, circuit waveform, and retained state
agree at convergence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.electropermanent import (
    MU0,
    ElectropermanentRod,
    ElectropermanentSource,
    EvidenceRecord,
    RemanenceBranch,
    RemanenceState,
)
from spin_dynamics.fields.electropermanent_pulses import (
    ProgrammingPulse,
    PulsePowerDriver,
    PulseWaveform,
)


def _push_reversal(stack: tuple[float, ...], value: float) -> tuple[float, ...]:
    """Push one turning point and remove closed inner reversal pairs."""

    points = [*stack, float(value)]
    while len(points) >= 3:
        first, middle, last = points[-3:]
        closes_upper = middle > first and last <= first
        closes_lower = middle < first and last >= first
        if not (closes_upper or closes_lower):
            break
        points[-3:] = [last]
    return tuple(points)


@dataclass(frozen=True)
class ReturnPointState:
    """Internal play-operator state plus the public retained remanence."""

    remanence: RemanenceState
    play_outputs_a_per_m: tuple[float, ...]
    last_field_a_per_m: float = 0.0
    last_direction: int = 0
    reversal_stack_a_per_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        outputs = tuple(float(value) for value in self.play_outputs_a_per_m)
        reversals = tuple(float(value) for value in self.reversal_stack_a_per_m)
        if not outputs or any(not np.isfinite(value) for value in outputs):
            raise ValueError("play_outputs_a_per_m must be a non-empty finite tuple")
        if not np.isfinite(self.last_field_a_per_m):
            raise ValueError("last_field_a_per_m must be finite")
        if self.last_direction not in {-1, 0, 1}:
            raise ValueError("last_direction must be -1, 0, or +1")
        if any(not np.isfinite(value) for value in reversals):
            raise ValueError("reversal_stack_a_per_m must be finite")
        object.__setattr__(self, "play_outputs_a_per_m", outputs)
        object.__setattr__(self, "reversal_stack_a_per_m", reversals)


@dataclass(frozen=True)
class ReturnPointTrace:
    """Applied H history, retained-remanence trajectory, and final memory."""

    fields_a_per_m: np.ndarray
    remanence_t: np.ndarray
    temperatures_k: np.ndarray
    final_state: ReturnPointState


@dataclass(frozen=True)
class PlayHysteresisModel:
    """Weighted scalar play operators with exact return-point memory.

    ``thresholds_a_per_m`` are positive play radii. ``weights`` are normalized
    internally and determine each operator's contribution to the saturated
    remanence.  The optional temperature coefficient scales the saturation
    endpoint; it does not change thresholds.
    """

    thresholds_a_per_m: tuple[float, ...]
    weights: tuple[float, ...]
    saturation_remanence_t: float
    calibration_id: str
    uncertainty_t: float = 0.0
    reference_temperature_k: float = 293.15
    remanence_temperature_coefficient_per_k: float = 0.0
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.thresholds_a_per_m)
        weights = np.asarray(self.weights, dtype=np.float64)
        if not thresholds or len(thresholds) != weights.size:
            raise ValueError("thresholds_a_per_m and weights must have equal nonzero length")
        if any(not np.isfinite(value) or value <= 0.0 for value in thresholds):
            raise ValueError("thresholds_a_per_m must be finite and positive")
        if any(
            thresholds[index + 1] <= thresholds[index]
            for index in range(len(thresholds) - 1)
        ):
            raise ValueError("thresholds_a_per_m must be strictly increasing")
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0) or np.sum(weights) <= 0.0:
            raise ValueError("weights must be finite, non-negative, and not all zero")
        if not np.isfinite(self.saturation_remanence_t) or self.saturation_remanence_t <= 0.0:
            raise ValueError("saturation_remanence_t must be finite and positive")
        if not self.calibration_id.strip():
            raise ValueError("calibration_id must not be empty")
        if not np.isfinite(self.uncertainty_t) or self.uncertainty_t < 0.0:
            raise ValueError("uncertainty_t must be finite and non-negative")
        if not np.isfinite(self.reference_temperature_k) or self.reference_temperature_k <= 0.0:
            raise ValueError("reference_temperature_k must be finite and positive")
        if not np.isfinite(self.remanence_temperature_coefficient_per_k):
            raise ValueError("remanence_temperature_coefficient_per_k must be finite")
        object.__setattr__(self, "thresholds_a_per_m", thresholds)
        object.__setattr__(self, "weights", tuple(float(value) for value in weights / np.sum(weights)))
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def saturation_at_temperature(self, temperature_k: float) -> float:
        """Return the signed-endpoint magnitude at ``temperature_k``."""

        if not np.isfinite(temperature_k) or temperature_k <= 0.0:
            raise ValueError("temperature_k must be finite and positive")
        scale = 1.0 + self.remanence_temperature_coefficient_per_k * (
            temperature_k - self.reference_temperature_k
        )
        if scale <= 0.0:
            raise ValueError("temperature scaling produces non-positive saturation")
        return float(self.saturation_remanence_t * scale)

    def remanence_from_outputs(
        self,
        outputs_a_per_m: Sequence[float],
        temperature_k: float,
    ) -> float:
        """Map play outputs to the bounded retained-remanence coordinate."""

        outputs = np.asarray(outputs_a_per_m, dtype=np.float64)
        thresholds = np.asarray(self.thresholds_a_per_m)
        if outputs.shape != thresholds.shape or not np.all(np.isfinite(outputs)):
            raise ValueError("outputs must be finite and match the model thresholds")
        normalized = np.clip(outputs / thresholds, -1.0, 1.0)
        return float(
            self.saturation_at_temperature(temperature_k)
            * np.dot(np.asarray(self.weights), normalized)
        )

    def initialize(
        self,
        remanence: RemanenceState,
        *,
        field_a_per_m: float = 0.0,
    ) -> ReturnPointState:
        """Construct a compatible internal state from a public remanence record."""

        if not np.isfinite(field_a_per_m):
            raise ValueError("field_a_per_m must be finite")
        saturation = self.saturation_at_temperature(remanence.temperature_k)
        fraction = remanence.remanence_t / saturation
        if abs(fraction) > 1.0 + 1e-12:
            raise ValueError("remanence exceeds this model's saturation endpoint")
        fraction = float(np.clip(fraction, -1.0, 1.0))
        outputs = tuple(fraction * np.asarray(self.thresholds_a_per_m))
        return ReturnPointState(
            remanence=remanence,
            play_outputs_a_per_m=outputs,
            last_field_a_per_m=float(field_a_per_m),
            reversal_stack_a_per_m=remanence.reversal_fields_a_per_m,
        )

    def _step(
        self,
        state: ReturnPointState,
        field_a_per_m: float,
        temperature_k: float,
    ) -> ReturnPointState:
        thresholds = np.asarray(self.thresholds_a_per_m)
        previous = np.asarray(state.play_outputs_a_per_m)
        field = float(field_a_per_m)
        outputs = np.maximum(
            field - thresholds,
            np.minimum(field + thresholds, previous),
        )
        delta = field - state.last_field_a_per_m
        direction = 0 if delta == 0.0 else (1 if delta > 0.0 else -1)
        reversal_stack = state.reversal_stack_a_per_m
        if (
            direction
            and state.last_direction
            and direction != state.last_direction
        ):
            reversal_stack = _push_reversal(
                reversal_stack,
                state.last_field_a_per_m,
            )
        active_direction = state.last_direction if direction == 0 else direction
        retained = self.remanence_from_outputs(outputs, temperature_k)
        saturation = self.saturation_at_temperature(temperature_k)
        if retained >= saturation * (1.0 - 1e-10):
            branch: RemanenceBranch = "positive_saturation"
        elif retained <= -saturation * (1.0 - 1e-10):
            branch = "negative_saturation"
        else:
            branch = "partial"
        public = RemanenceState(
            retained,
            branch=branch,
            reversal_fields_a_per_m=reversal_stack,
            temperature_k=float(temperature_k),
            calibration_id=self.calibration_id,
            uncertainty_t=self.uncertainty_t,
            evidence=self.evidence,
        )
        return ReturnPointState(
            remanence=public,
            play_outputs_a_per_m=tuple(outputs),
            last_field_a_per_m=field,
            last_direction=active_direction,
            reversal_stack_a_per_m=reversal_stack,
        )

    def propagate(
        self,
        state: ReturnPointState | RemanenceState,
        fields_a_per_m: Sequence[float] | np.ndarray,
        *,
        temperatures_k: float | Sequence[float] | np.ndarray | None = None,
    ) -> ReturnPointTrace:
        """Propagate a field history through every play operator."""

        fields = np.asarray(fields_a_per_m, dtype=np.float64)
        if fields.ndim != 1 or not fields.size or not np.all(np.isfinite(fields)):
            raise ValueError("fields_a_per_m must be a non-empty finite 1-D array")
        current = self.initialize(state) if isinstance(state, RemanenceState) else state
        if len(current.play_outputs_a_per_m) != len(self.thresholds_a_per_m):
            raise ValueError("state play outputs do not match this model")
        if temperatures_k is None:
            temperatures = np.full(fields.shape, current.remanence.temperature_k)
        else:
            temperatures = np.asarray(temperatures_k, dtype=np.float64)
            if temperatures.ndim == 0:
                temperatures = np.full(fields.shape, float(temperatures))
            if temperatures.shape != fields.shape or np.any(~np.isfinite(temperatures)):
                raise ValueError("temperatures_k must be scalar or match fields")
            if np.any(temperatures <= 0.0):
                raise ValueError("temperatures_k must be positive")
        retained = np.empty_like(fields)
        for index, (field, temperature) in enumerate(zip(fields, temperatures)):
            current = self._step(current, float(field), float(temperature))
            retained[index] = current.remanence.remanence_t
        return ReturnPointTrace(fields, retained, temperatures, current)

    def apply(
        self,
        state: ReturnPointState | RemanenceState,
        waveform: PulseWaveform,
    ) -> ReturnPointTrace:
        """Apply a realized programming waveform to one return-point state."""

        return self.propagate(
            state,
            waveform.internal_h_a_per_m,
            temperatures_k=waveform.coil_temperature_k,
        )


def illustrative_alnico_return_point_model() -> PlayHysteresisModel:
    """Return a qualitative AlNiCo play model, not a measured calibration."""

    evidence = EvidenceRecord(
        source="PythonSpinDynamics illustrative AlNiCo return-point model",
        classification="inferred",
        detail=(
            "Threshold distribution chosen to demonstrate nested minor loops and "
            "neighbor coupling; raw Weinberg minor-loop data remain unavailable"
        ),
    )
    return PlayHysteresisModel(
        thresholds_a_per_m=(5e3, 10e3, 20e3, 40e3, 80e3, 160e3),
        weights=(0.08, 0.12, 0.18, 0.22, 0.22, 0.18),
        saturation_remanence_t=0.33,
        calibration_id="illustrative-alnico-play-v1",
        uncertainty_t=0.05,
        remanence_temperature_coefficient_per_k=-2.0e-4,
        evidence=(evidence,),
    )


def _source_center(source: ElectropermanentSource) -> np.ndarray:
    return (
        np.asarray(source.center_m, dtype=np.float64)
        if isinstance(source, ElectropermanentRod)
        else np.asarray(source.center_m, dtype=np.float64)
    )


def _source_axis(source: ElectropermanentSource) -> np.ndarray:
    return (
        np.asarray(source.axis, dtype=np.float64)
        if isinstance(source, ElectropermanentRod)
        else np.asarray(source.axis, dtype=np.float64)
    )


def _source_material(source: ElectropermanentSource):
    if isinstance(source, ElectropermanentRod):
        return source.material
    material = source.rods[0].material
    if any(rod.material != material for rod in source.rods):
        raise ValueError("coupled programming requires one material per bundle")
    return material


def _source_state(source: ElectropermanentSource) -> RemanenceState:
    if isinstance(source, ElectropermanentRod):
        return source.state
    state = source.rods[0].state
    if any(rod.state != state for rod in source.rods):
        raise ValueError("coupled programming requires one shared state per bundle")
    return state


def neighbor_coupling_matrix(
    sources: Sequence[ElectropermanentSource],
    *,
    self_demagnetizing_factors: Sequence[float] | None = None,
    n_cross: int = 5,
    n_length: int = 21,
) -> np.ndarray:
    """Return axial H-bias coupling in ``(A/m)/T`` of retained remanence.

    Off-diagonal entry ``K[i, j]`` is the axial H at element ``i`` produced by
    one tesla of retained remanence in element ``j``. Diagonal entries are
    ``-N/MU0`` when self-demagnetizing factors are supplied.
    """

    elements = tuple(sources)
    if not elements:
        raise ValueError("sources must not be empty")
    count = len(elements)
    if self_demagnetizing_factors is None:
        demag = np.zeros(count)
    else:
        demag = np.asarray(self_demagnetizing_factors, dtype=np.float64)
        if demag.shape != (count,) or np.any(~np.isfinite(demag)):
            raise ValueError("self_demagnetizing_factors must match sources")
        if np.any((demag < 0.0) | (demag > 1.0)):
            raise ValueError("self-demagnetizing factors must lie in [0, 1]")
    matrix = np.zeros((count, count), dtype=np.float64)
    matrix[np.diag_indices(count)] = -demag / MU0
    centers = tuple(_source_center(source) for source in elements)
    axes = tuple(_source_axis(source) for source in elements)
    for column, source in enumerate(elements):
        material = _source_material(source)
        reference_remanence = min(1.0, 0.5 * material.remanence_t)
        unit_source = source.with_state(
            RemanenceState(reference_remanence, branch="partial")
        )
        for row in range(count):
            if row == column:
                continue
            field = unit_source.field_vectors(
                centers[row][None, :],
                n_cross=n_cross,
                n_length=n_length,
            )[0]
            matrix[row, column] = float(
                np.dot(field, axes[row]) / (MU0 * reference_remanence)
            )
    return matrix


@dataclass(frozen=True)
class CoupledProgrammingResult:
    """Fixed-point result for one programmed element in an interacting array."""

    element_index: int
    initial_states: tuple[ReturnPointState, ...]
    final_states: tuple[ReturnPointState, ...]
    final_sources: tuple[ElectropermanentSource, ...]
    waveform: PulseWaveform
    neighbor_bias_a_per_m: float
    iterations: int
    residual_t: float
    converged: bool


@dataclass(frozen=True)
class CoupledEPMProgrammer:
    """Self-consistent circuit/hysteresis update for interacting EPM elements."""

    sources: tuple[ElectropermanentSource, ...]
    drivers: tuple[PulsePowerDriver, ...]
    hysteresis_models: tuple[PlayHysteresisModel, ...]
    coupling_a_per_m_per_t: np.ndarray

    def __post_init__(self) -> None:
        sources = tuple(self.sources)
        drivers = tuple(self.drivers)
        models = tuple(self.hysteresis_models)
        if not sources or len(drivers) != len(sources) or len(models) != len(sources):
            raise ValueError("sources, drivers, and hysteresis_models must have equal nonzero length")
        coupling = np.asarray(self.coupling_a_per_m_per_t, dtype=np.float64)
        if coupling.shape != (len(sources), len(sources)) or np.any(~np.isfinite(coupling)):
            raise ValueError("coupling_a_per_m_per_t must be a finite square matrix")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "drivers", drivers)
        object.__setattr__(self, "hysteresis_models", models)
        object.__setattr__(self, "coupling_a_per_m_per_t", coupling)

    def initial_states(self) -> tuple[ReturnPointState, ...]:
        """Initialize every hysteresis model from its source's public state."""

        return tuple(
            model.initialize(_source_state(source))
            for source, model in zip(self.sources, self.hysteresis_models)
        )

    def program(
        self,
        element_index: int,
        times_s: Sequence[float] | np.ndarray,
        pulse: ProgrammingPulse,
        *,
        states: Sequence[ReturnPointState] | None = None,
        tolerance_t: float = 1e-7,
        max_iterations: int = 30,
        relaxation: float = 0.7,
        max_step_s: float | None = None,
        raise_on_nonconvergence: bool = True,
    ) -> CoupledProgrammingResult:
        """Program one element with fixed-point circuit/magnetic coupling."""

        index = int(element_index)
        if index != element_index or not 0 <= index < len(self.sources):
            raise ValueError("element_index is out of range")
        initial = self.initial_states() if states is None else tuple(states)
        if len(initial) != len(self.sources):
            raise ValueError("states must match the number of sources")
        if not np.isfinite(tolerance_t) or tolerance_t <= 0.0:
            raise ValueError("tolerance_t must be finite and positive")
        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0, 1]")

        original = initial[index]
        remanence = np.asarray(
            [state.remanence.remanence_t for state in initial],
            dtype=np.float64,
        )
        guess = float(remanence[index])
        trial = original
        waveform: PulseWaveform | None = None
        bias = 0.0
        residual = np.inf
        converged = False
        material = _source_material(self.sources[index])
        for iteration in range(1, max_iterations + 1):
            remanence[index] = guess
            bias = float(self.coupling_a_per_m_per_t[index] @ remanence)
            guessed_public = RemanenceState(
                guess,
                branch="partial",
                temperature_k=original.remanence.temperature_k,
                calibration_id=original.remanence.calibration_id,
                uncertainty_t=original.remanence.uncertainty_t,
                evidence=original.remanence.evidence,
            )
            waveform = self.drivers[index].simulate(
                np.asarray(times_s, dtype=np.float64),
                pulse,
                state=guessed_public,
                material=material,
                bias_field_a_per_m=bias,
                max_step_s=max_step_s,
            )
            trial = self.hysteresis_models[index].apply(original, waveform).final_state
            updated = trial.remanence.remanence_t
            residual = abs(updated - guess)
            if residual <= tolerance_t:
                converged = True
                break
            guess = (1.0 - relaxation) * guess + relaxation * updated
        if waveform is None:  # pragma: no cover - guarded by max_iterations validation
            raise RuntimeError("programming iteration did not start")
        if not converged and raise_on_nonconvergence:
            raise RuntimeError(
                "coupled EPM programming did not converge: "
                f"residual={residual:.3g} T after {max_iterations} iterations"
            )
        final_states = list(initial)
        final_states[index] = trial
        final_sources = list(self.sources)
        final_sources[index] = self.sources[index].with_state(trial.remanence)
        return CoupledProgrammingResult(
            element_index=index,
            initial_states=initial,
            final_states=tuple(final_states),
            final_sources=tuple(final_sources),
            waveform=waveform,
            neighbor_bias_a_per_m=bias,
            iterations=iteration,
            residual_t=float(residual),
            converged=converged,
        )


__all__ = [
    "ReturnPointState",
    "ReturnPointTrace",
    "PlayHysteresisModel",
    "illustrative_alnico_return_point_model",
    "neighbor_coupling_matrix",
    "CoupledProgrammingResult",
    "CoupledEPMProgrammer",
]
