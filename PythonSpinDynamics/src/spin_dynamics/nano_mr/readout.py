"""Effective optical initialization and photon-count readout models."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal

import numpy as np


def _probability(value, *, name: str) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(probability)):
        raise ValueError(f"{name} must be finite")
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError(f"{name} must lie in [0, 1]")
    return probability


@dataclass(frozen=True)
class OpticalReadoutModel:
    """Effective bright/dark initialization and fluorescence-count model.

    ``bright_count_rate_hz`` is the detected rate for a fully bright state
    before background. A fully dark state has signal rate
    ``bright_count_rate_hz * (1-readout_contrast)``.
    """

    initialization_fidelity: float = 1.0
    initialization_seconds: float = 0.0
    bright_count_rate_hz: float = 100.0e3
    readout_contrast: float = 0.2
    readout_seconds: float = 300.0e-9
    background_count_rate_hz: float = 0.0
    dead_time_seconds: float = 0.0

    def __post_init__(self) -> None:
        _probability(self.initialization_fidelity, name="initialization_fidelity")
        _probability(self.readout_contrast, name="readout_contrast")
        for name in (
            "initialization_seconds",
            "bright_count_rate_hz",
            "readout_seconds",
            "background_count_rate_hz",
            "dead_time_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.readout_seconds <= 0.0:
            raise ValueError("readout_seconds must be positive")

    @property
    def initialized_bright_probability(self) -> float:
        """Bright-state probability immediately after optical initialization."""

        return float(self.initialization_fidelity)

    def normalized_fluorescence(self, bright_probability) -> np.ndarray:
        """Return signal fluorescence relative to the fully bright rate."""

        probability = _probability(
            bright_probability, name="bright_probability"
        )
        return 1.0 - self.readout_contrast * (1.0 - probability)

    def apply_initialization_fidelity(self, bright_probability) -> np.ndarray:
        """Apply imperfect bright-state preparation to an ideal qubit result.

        The effective model assumes unital two-level control: preparation in
        the wrong basis state inverts the ideal population response. A
        fidelity of one leaves the result unchanged, while a fidelity of one
        half removes all spin contrast.
        """

        probability = _probability(
            bright_probability,
            name="bright_probability",
        )
        polarization = 2.0 * float(self.initialization_fidelity) - 1.0
        return 0.5 + polarization * (probability - 0.5)

    def expected_counts(
        self,
        bright_probability,
        *,
        repetitions: int = 1,
    ) -> np.ndarray:
        """Return the Poisson mean photon counts for repeated measurements."""

        repetitions = _positive_integer(repetitions, name="repetitions")
        signal_rate = self.bright_count_rate_hz * self.normalized_fluorescence(
            bright_probability
        )
        total_rate = signal_rate + self.background_count_rate_hz
        return repetitions * self.readout_seconds * total_rate

    def cycle_seconds(self, sensing_seconds: float) -> float:
        """Return initialization+sensing+readout+dead time for one repetition."""

        sensing = float(sensing_seconds)
        if sensing < 0.0 or not np.isfinite(sensing):
            raise ValueError("sensing_seconds must be finite and non-negative")
        return (
            self.initialization_seconds
            + sensing
            + self.readout_seconds
            + self.dead_time_seconds
        )


@dataclass(frozen=True)
class OpticalReadoutResult:
    """Expected and sampled fluorescence counts with acquisition metadata."""

    bright_probability: np.ndarray
    effective_bright_probability: np.ndarray
    expected_counts: np.ndarray
    sampled_counts: np.ndarray
    repetitions: int
    acquisition_seconds: float
    seed: int | None


OpticalState = Literal["g0", "g1", "e0", "e1", "singlet", "nv0"]
DeadTimeMode = Literal["nonparalyzable", "paralyzable"]
OPTICAL_STATES: tuple[OpticalState, ...] = (
    "g0",
    "g1",
    "e0",
    "e1",
    "singlet",
    "nv0",
)


@dataclass(frozen=True)
class OpticalCycleModel:
    """Room-temperature NV optical-cycle continuous-time rate model.

    The first five states form the conventional bright/dark triplet plus
    metastable-singlet model. Nonzero ionization or recombination rates enable
    the sixth, neutral-charge state. Rates are configurable because shallow
    defects and optical setups do not share universal photophysics.
    """

    excitation_rate_hz: float = 60.0e6
    radiative_rate_hz: float = 77.0e6
    bright_isc_rate_hz: float = 5.0e6
    dark_isc_rate_hz: float = 55.0e6
    singlet_to_bright_rate_hz: float = 4.0e6
    singlet_to_dark_rate_hz: float = 1.0e6
    bright_ionization_rate_hz: float = 0.0
    dark_ionization_rate_hz: float = 0.0
    recombination_to_bright_rate_hz: float = 0.0
    recombination_to_dark_rate_hz: float = 0.0
    label: str = "five-level room-temperature NV optical cycle"

    def __post_init__(self) -> None:
        for name in (
            "excitation_rate_hz",
            "radiative_rate_hz",
            "bright_isc_rate_hz",
            "dark_isc_rate_hz",
            "singlet_to_bright_rate_hz",
            "singlet_to_dark_rate_hz",
            "bright_ionization_rate_hz",
            "dark_ionization_rate_hz",
            "recombination_to_bright_rate_hz",
            "recombination_to_dark_rate_hz",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.excitation_rate_hz <= 0.0:
            raise ValueError("excitation_rate_hz must be positive")
        if self.radiative_rate_hz <= 0.0:
            raise ValueError("radiative_rate_hz must be positive")
        object.__setattr__(self, "label", str(self.label))

    def rate_matrix(self) -> np.ndarray:
        """Return the row-generator matrix in :data:`OPTICAL_STATES` order."""

        matrix = np.zeros((6, 6), dtype=np.float64)
        _set_rate(matrix, 0, 2, self.excitation_rate_hz)
        _set_rate(matrix, 1, 3, self.excitation_rate_hz)
        _set_rate(matrix, 2, 0, self.radiative_rate_hz)
        _set_rate(matrix, 3, 1, self.radiative_rate_hz)
        _set_rate(matrix, 2, 4, self.bright_isc_rate_hz)
        _set_rate(matrix, 3, 4, self.dark_isc_rate_hz)
        _set_rate(matrix, 4, 0, self.singlet_to_bright_rate_hz)
        _set_rate(matrix, 4, 1, self.singlet_to_dark_rate_hz)
        _set_rate(matrix, 2, 5, self.bright_ionization_rate_hz)
        _set_rate(matrix, 3, 5, self.dark_ionization_rate_hz)
        _set_rate(matrix, 5, 0, self.recombination_to_bright_rate_hz)
        _set_rate(matrix, 5, 1, self.recombination_to_dark_rate_hz)
        matrix[np.diag_indices_from(matrix)] = -np.sum(matrix, axis=1)
        return matrix

    def population_trace(
        self,
        bright_probability: float,
        times_seconds,
    ) -> np.ndarray:
        """Return optical-state populations at monotonically increasing times."""

        probability = _scalar_probability(
            bright_probability,
            name="bright_probability",
        )
        times = np.asarray(times_seconds, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size == 0
            or not np.all(np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "times_seconds must be finite, non-negative, and increasing"
            )
        population = np.zeros(6, dtype=np.float64)
        population[0] = probability
        population[1] = 1.0 - probability
        trace = np.empty((times.size, 6), dtype=np.float64)
        previous = 0.0
        generator = self.rate_matrix()
        for index, time in enumerate(times):
            population = population @ _ctmc_transition(
                generator,
                float(time - previous),
            )
            trace[index] = population
            previous = float(time)
        return trace

    def expected_emitted_photons(
        self,
        bright_probability: float,
        *,
        readout_seconds: float,
        time_bins: int = 256,
    ) -> float:
        """Integrate the expected radiative photon rate over one readout."""

        duration = _positive_duration(readout_seconds, "readout_seconds")
        bins = _positive_integer(time_bins, name="time_bins")
        if bins < 2:
            raise ValueError("time_bins must be at least two")
        times = np.linspace(0.0, duration, bins)
        trace = self.population_trace(bright_probability, times)
        rate = self.radiative_rate_hz * (trace[:, 2] + trace[:, 3])
        return float(
            np.sum(0.5 * (rate[:-1] + rate[1:]) * np.diff(times))
        )


@dataclass(frozen=True)
class SPADDetectorModel:
    """Single-photon avalanche-detector transfer and nuisance counts."""

    detection_efficiency: float = 0.1
    background_count_rate_hz: float = 0.0
    dark_count_rate_hz: float = 100.0
    dead_time_seconds: float = 0.0
    dead_time_mode: DeadTimeMode = "nonparalyzable"
    afterpulse_probability: float = 0.0
    afterpulse_time_seconds: float = 1.0e-6
    timing_jitter_seconds: float = 0.0
    label: str = "SPAD"

    def __post_init__(self) -> None:
        _scalar_probability(
            self.detection_efficiency,
            name="detection_efficiency",
        )
        _scalar_probability(
            self.afterpulse_probability,
            name="afterpulse_probability",
        )
        for name in (
            "background_count_rate_hz",
            "dark_count_rate_hz",
            "dead_time_seconds",
            "timing_jitter_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        afterpulse_time = float(self.afterpulse_time_seconds)
        if afterpulse_time <= 0.0 or not np.isfinite(afterpulse_time):
            raise ValueError("afterpulse_time_seconds must be positive and finite")
        if self.dead_time_mode not in {"nonparalyzable", "paralyzable"}:
            raise ValueError(
                "dead_time_mode must be 'nonparalyzable' or 'paralyzable'"
            )
        object.__setattr__(self, "afterpulse_time_seconds", afterpulse_time)
        object.__setattr__(self, "label", str(self.label))

    def approximate_expected_counts(
        self,
        emitted_photons: float,
        *,
        readout_seconds: float,
    ) -> float:
        """Return a rate-corrected count expectation for diagnostics."""

        duration = _positive_duration(readout_seconds, "readout_seconds")
        emitted = float(emitted_photons)
        if emitted < 0.0 or not np.isfinite(emitted):
            raise ValueError("emitted_photons must be finite and non-negative")
        raw_rate = (
            self.detection_efficiency * emitted / duration
            + self.background_count_rate_hz
            + self.dark_count_rate_hz
        )
        raw_rate *= 1.0 + self.afterpulse_probability
        if self.dead_time_seconds == 0.0:
            observed_rate = raw_rate
        elif self.dead_time_mode == "nonparalyzable":
            observed_rate = raw_rate / (
                1.0 + raw_rate * self.dead_time_seconds
            )
        else:
            observed_rate = raw_rate * np.exp(
                -raw_rate * self.dead_time_seconds
            )
        return float(duration * observed_rate)

    def detect(
        self,
        emitted_arrival_times_seconds,
        *,
        readout_seconds: float,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Transfer emitted photon times through efficiency and SPAD effects."""

        if seed is not None and rng is not None:
            raise ValueError("provide seed or rng, not both")
        duration = _positive_duration(readout_seconds, "readout_seconds")
        emitted = np.asarray(
            emitted_arrival_times_seconds,
            dtype=np.float64,
        )
        if (
            emitted.ndim != 1
            or not np.all(np.isfinite(emitted))
            or np.any(emitted < 0.0)
            or np.any(emitted >= duration)
        ):
            raise ValueError(
                "emitted arrival times must be finite and inside the readout"
            )
        generator = np.random.default_rng(seed) if rng is None else rng
        return _sample_spad_arrivals(
            self,
            emitted,
            duration,
            generator,
        )


@dataclass(frozen=True, eq=False)
class TimeResolvedOpticalReadoutResult:
    """Shot-resolved optical emission and detector records."""

    bright_probability: np.ndarray
    emitted_counts: np.ndarray
    detected_counts: np.ndarray
    detected_arrival_times_seconds: tuple[np.ndarray, ...]
    expected_emitted_counts_per_shot: np.ndarray
    approximate_expected_detected_counts_per_shot: np.ndarray
    repetitions: int
    readout_seconds: float
    acquisition_seconds: float
    seed: int | None

    @property
    def fano_factor(self) -> float:
        """Return detected-count variance divided by its mean."""

        counts = self.detected_counts.reshape(-1).astype(np.float64)
        mean = float(np.mean(counts))
        if mean <= 0.0:
            return 0.0
        return float(np.var(counts, ddof=1) / mean) if counts.size > 1 else 0.0


def sample_optical_readout(
    model: OpticalReadoutModel,
    bright_probability,
    *,
    repetitions: int = 1,
    sensing_seconds: float = 0.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> OpticalReadoutResult:
    """Draw reproducible Poisson photon counts from an effective readout model."""

    if seed is not None and rng is not None:
        raise ValueError("provide seed or rng, not both")
    probability = _probability(
        bright_probability, name="bright_probability"
    )
    repetitions = _positive_integer(repetitions, name="repetitions")
    effective_probability = model.apply_initialization_fidelity(probability)
    expected = model.expected_counts(
        effective_probability,
        repetitions=repetitions,
    )
    generator = np.random.default_rng(seed) if rng is None else rng
    sampled = generator.poisson(expected)
    return OpticalReadoutResult(
        bright_probability=probability.copy(),
        effective_bright_probability=np.asarray(
            effective_probability,
            dtype=np.float64,
        ),
        expected_counts=np.asarray(expected, dtype=np.float64),
        sampled_counts=np.asarray(sampled, dtype=np.int64),
        repetitions=repetitions,
        acquisition_seconds=repetitions * model.cycle_seconds(sensing_seconds),
        seed=seed,
    )


def sample_time_resolved_optical_readout(
    optical_model: OpticalCycleModel,
    detector_model: SPADDetectorModel,
    bright_probability,
    *,
    repetitions: int = 1,
    readout_seconds: float = 300.0e-9,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> TimeResolvedOpticalReadoutResult:
    """Sample optical-state paths, photon emission, and SPAD transfer.

    The final dimension of ``emitted_counts`` and ``detected_counts`` indexes
    repetitions. All preceding dimensions match ``bright_probability``.
    Arrival-time arrays are flattened in input-major, repetition-minor order.
    """

    if not isinstance(optical_model, OpticalCycleModel):
        raise TypeError("optical_model must be an OpticalCycleModel")
    if not isinstance(detector_model, SPADDetectorModel):
        raise TypeError("detector_model must be a SPADDetectorModel")
    if seed is not None and rng is not None:
        raise ValueError("provide seed or rng, not both")
    probability = _probability(
        bright_probability,
        name="bright_probability",
    )
    repetition_count = _positive_integer(repetitions, name="repetitions")
    duration = _positive_duration(readout_seconds, "readout_seconds")
    generator = np.random.default_rng(seed) if rng is None else rng

    bright_emitted = optical_model.expected_emitted_photons(
        1.0,
        readout_seconds=duration,
    )
    dark_emitted = optical_model.expected_emitted_photons(
        0.0,
        readout_seconds=duration,
    )
    expected_emitted = (
        probability * bright_emitted + (1.0 - probability) * dark_emitted
    )
    expected_detected = np.empty_like(expected_emitted)
    for index, value in np.ndenumerate(expected_emitted):
        expected_detected[index] = detector_model.approximate_expected_counts(
            float(value),
            readout_seconds=duration,
        )

    output_shape = probability.shape + (repetition_count,)
    emitted_counts = np.empty(output_shape, dtype=np.int64)
    detected_counts = np.empty(output_shape, dtype=np.int64)
    arrivals: list[np.ndarray] = []
    generator_matrix = optical_model.rate_matrix()
    for flat_index, item_probability in enumerate(probability.reshape(-1)):
        input_index = np.unravel_index(flat_index, probability.shape)
        for repetition in range(repetition_count):
            initial_state = 0 if generator.random() < item_probability else 1
            emitted = _sample_optical_emission_times(
                generator_matrix,
                initial_state,
                duration,
                generator,
            )
            detected = detector_model.detect(
                emitted,
                readout_seconds=duration,
                rng=generator,
            )
            output_index = input_index + (repetition,)
            emitted_counts[output_index] = emitted.size
            detected_counts[output_index] = detected.size
            arrivals.append(detected)
    return TimeResolvedOpticalReadoutResult(
        bright_probability=probability.copy(),
        emitted_counts=emitted_counts,
        detected_counts=detected_counts,
        detected_arrival_times_seconds=tuple(arrivals),
        expected_emitted_counts_per_shot=np.asarray(
            expected_emitted,
            dtype=np.float64,
        ),
        approximate_expected_detected_counts_per_shot=np.asarray(
            expected_detected,
            dtype=np.float64,
        ),
        repetitions=repetition_count,
        readout_seconds=duration,
        acquisition_seconds=(
            probability.size * repetition_count * duration
        ),
        seed=seed,
    )


def optical_contrast_trace(
    model: OpticalCycleModel,
    times_seconds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bright/dark fluorescence rates and their transient contrast."""

    if not isinstance(model, OpticalCycleModel):
        raise TypeError("model must be an OpticalCycleModel")
    times = np.asarray(times_seconds, dtype=np.float64)
    bright = model.population_trace(1.0, times)
    dark = model.population_trace(0.0, times)
    bright_rate = model.radiative_rate_hz * (bright[:, 2] + bright[:, 3])
    dark_rate = model.radiative_rate_hz * (dark[:, 2] + dark[:, 3])
    denominator = np.maximum(bright_rate, np.finfo(float).tiny)
    contrast = (bright_rate - dark_rate) / denominator
    return bright_rate, dark_rate, contrast


def _sample_optical_emission_times(
    generator_matrix: np.ndarray,
    initial_state: int,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    time = 0.0
    state = initial_state
    emitted: list[float] = []
    while time < duration:
        rates = generator_matrix[state].copy()
        rates[state] = 0.0
        total = float(np.sum(rates))
        if total <= 0.0:
            break
        time += float(rng.exponential(1.0 / total))
        if time >= duration:
            break
        draw = rng.random() * total
        next_state = int(np.searchsorted(np.cumsum(rates), draw, side="right"))
        if (
            (state == 2 and next_state == 0)
            or (state == 3 and next_state == 1)
        ):
            emitted.append(time)
        state = next_state
    return np.asarray(emitted, dtype=np.float64)


def _sample_spad_arrivals(
    model: SPADDetectorModel,
    emitted_times: np.ndarray,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    keep = rng.random(emitted_times.size) < model.detection_efficiency
    candidates = emitted_times[keep].copy()
    nuisance_rate = (
        model.background_count_rate_hz + model.dark_count_rate_hz
    )
    nuisance_count = int(rng.poisson(nuisance_rate * duration))
    if nuisance_count:
        candidates = np.concatenate(
            (candidates, rng.uniform(0.0, duration, nuisance_count))
        )
    if model.timing_jitter_seconds > 0.0 and candidates.size:
        candidates += rng.normal(
            0.0,
            model.timing_jitter_seconds,
            candidates.size,
        )
        candidates = candidates[
            (candidates >= 0.0) & (candidates < duration)
        ]

    queue = [(float(time), True) for time in candidates]
    heapq.heapify(queue)
    accepted: list[float] = []
    blocked_until = -np.inf
    while queue:
        time, may_afterpulse = heapq.heappop(queue)
        if time < blocked_until:
            if model.dead_time_mode == "paralyzable":
                blocked_until = time + model.dead_time_seconds
            continue
        accepted.append(time)
        blocked_until = time + model.dead_time_seconds
        if may_afterpulse and rng.random() < model.afterpulse_probability:
            afterpulse = time + float(
                rng.exponential(model.afterpulse_time_seconds)
            )
            if model.timing_jitter_seconds > 0.0:
                afterpulse += float(
                    rng.normal(0.0, model.timing_jitter_seconds)
                )
            if time <= afterpulse < duration:
                heapq.heappush(queue, (afterpulse, False))
    return np.asarray(accepted, dtype=np.float64)


def _ctmc_transition(generator: np.ndarray, duration: float) -> np.ndarray:
    """Matrix exponential of a CTMC row generator by uniformization."""

    if duration == 0.0:
        return np.eye(generator.shape[0])
    rate = float(np.max(-np.diag(generator)))
    if rate == 0.0:
        return np.eye(generator.shape[0])
    embedded = np.eye(generator.shape[0]) + generator / rate
    scaled = rate * duration
    if scaled > 100.0:
        steps = int(np.ceil(scaled / 50.0))
        short = _ctmc_transition(generator, duration / steps)
        return np.linalg.matrix_power(short, steps)
    weight = float(np.exp(-scaled))
    power = np.eye(generator.shape[0])
    transition = weight * power
    cumulative = weight
    for order in range(1, 10001):
        power = power @ embedded
        weight *= scaled / order
        transition += weight * power
        cumulative += weight
        if order > scaled and 1.0 - cumulative < 1.0e-14:
            break
    else:  # pragma: no cover - protects pathological user rates
        raise RuntimeError("CTMC uniformization did not converge")
    transition = np.maximum(transition, 0.0)
    row_sum = np.sum(transition, axis=1)
    return transition / row_sum[:, np.newaxis]


def _set_rate(
    matrix: np.ndarray,
    source: int,
    destination: int,
    rate_hz: float,
) -> None:
    matrix[source, destination] = rate_hz


def _scalar_probability(value: float, *, name: str) -> float:
    probability = _probability(value, name=name)
    if probability.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return float(probability)


def _positive_duration(value: float, name: str) -> float:
    duration = float(value)
    if duration <= 0.0 or not np.isfinite(duration):
        raise ValueError(f"{name} must be positive and finite")
    return duration


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    integer = int(value)
    if integer != value or integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


__all__ = [
    "DeadTimeMode",
    "OPTICAL_STATES",
    "OpticalCycleModel",
    "OpticalReadoutModel",
    "OpticalReadoutResult",
    "OpticalState",
    "SPADDetectorModel",
    "TimeResolvedOpticalReadoutResult",
    "optical_contrast_trace",
    "sample_optical_readout",
    "sample_time_resolved_optical_readout",
]
