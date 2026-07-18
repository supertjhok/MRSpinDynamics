"""Clocked high-resolution protocols for coherent nano-MR signals.

This module keeps the time scales that limit high-resolution measurements
explicit. Sensor coherence acts within one sensing block; sample coherence,
diffusion, and ancillary-memory coherence act between measurements; clock
errors perturb the acquisition timebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spin_dynamics.nano_mr.filter_functions import PulseModel, toggling_integral
from spin_dynamics.nano_mr.geometry import ISOTOPE_GAMMA_HZ_PER_T
from spin_dynamics.nano_mr.readout import (
    OpticalReadoutModel,
    OpticalReadoutResult,
    sample_optical_readout,
)
from spin_dynamics.nano_mr.sequences import SensingSequence


@dataclass(frozen=True)
class ClockModel:
    """Effective timebase errors for a clocked acquisition.

    ``fractional_frequency_offset`` is a fixed scale error. Independent
    interval-to-interval fractional-frequency errors have the configured RMS,
    and ``trigger_jitter_seconds`` adds independent timestamp jitter.
    """

    fractional_frequency_offset: float = 0.0
    interval_fractional_frequency_instability: float = 0.0
    trigger_jitter_seconds: float = 0.0

    def __post_init__(self) -> None:
        offset = float(self.fractional_frequency_offset)
        instability = float(self.interval_fractional_frequency_instability)
        jitter = float(self.trigger_jitter_seconds)
        if not np.isfinite(offset):
            raise ValueError("fractional_frequency_offset must be finite")
        if instability < 0.0 or not np.isfinite(instability):
            raise ValueError(
                "interval_fractional_frequency_instability must be finite "
                "and non-negative"
            )
        if jitter < 0.0 or not np.isfinite(jitter):
            raise ValueError("trigger_jitter_seconds must be finite and non-negative")
        object.__setattr__(self, "fractional_frequency_offset", offset)
        object.__setattr__(
            self,
            "interval_fractional_frequency_instability",
            instability,
        )
        object.__setattr__(self, "trigger_jitter_seconds", jitter)

    def sample_times(
        self,
        count: int,
        interval_seconds: float,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return nominal and clock-perturbed timestamps."""

        if seed is not None and rng is not None:
            raise ValueError("provide seed or rng, not both")
        count = _positive_integer(count, "count")
        interval = _positive(interval_seconds, "interval_seconds")
        nominal = interval * np.arange(count, dtype=np.float64)
        generator = np.random.default_rng(seed) if rng is None else rng
        if count == 1:
            accumulated = np.zeros(1)
        else:
            interval_errors = (
                interval
                * self.interval_fractional_frequency_instability
                * generator.standard_normal(count - 1)
            )
            accumulated = np.concatenate(([0.0], np.cumsum(interval_errors)))
        jitter = self.trigger_jitter_seconds * generator.standard_normal(count)
        actual = (
            nominal * (1.0 + self.fractional_frequency_offset) + accumulated + jitter
        )
        if np.any(np.diff(actual) <= 0.0):
            raise ValueError(
                "clock perturbations produced non-increasing physical sample times"
            )
        return nominal, actual


@dataclass(frozen=True)
class HighResolutionBudget:
    """Independent coherence limits for high-resolution protocols."""

    sensor_coherence_seconds: float = np.inf
    sensor_stretch_exponent: float = 1.0
    sample_coherence_seconds: float = np.inf
    diffusion_correlation_seconds: float = np.inf
    memory_coherence_seconds: float = np.inf

    def __post_init__(self) -> None:
        for name in (
            "sensor_coherence_seconds",
            "sample_coherence_seconds",
            "diffusion_correlation_seconds",
            "memory_coherence_seconds",
        ):
            object.__setattr__(
                self, name, _positive_or_infinite(getattr(self, name), name)
            )
        exponent = float(self.sensor_stretch_exponent)
        if exponent <= 0.0 or not np.isfinite(exponent):
            raise ValueError("sensor_stretch_exponent must be positive and finite")
        object.__setattr__(self, "sensor_stretch_exponent", exponent)

    def sensor_contrast(self, sensing_seconds: float) -> float:
        """Return the within-block sensor-coherence envelope."""

        sensing = _nonnegative(sensing_seconds, "sensing_seconds")
        if np.isinf(self.sensor_coherence_seconds):
            return 1.0
        return float(
            np.exp(
                -(
                    (sensing / self.sensor_coherence_seconds)
                    ** self.sensor_stretch_exponent
                )
            )
        )

    def intershot_envelope(
        self,
        times_seconds,
        *,
        include_memory: bool = False,
    ) -> np.ndarray:
        """Return sample, diffusion, and optionally memory coherence loss."""

        times = _nonnegative_times(times_seconds, "times_seconds")
        envelope = np.ones_like(times)
        for duration in (
            self.sample_coherence_seconds,
            self.diffusion_correlation_seconds,
        ):
            if np.isfinite(duration):
                envelope *= np.exp(-times / duration)
        if include_memory and np.isfinite(self.memory_coherence_seconds):
            envelope *= np.exp(-times / self.memory_coherence_seconds)
        return envelope


@dataclass(frozen=True)
class QdyneProtocol:
    """One clocked Qdyne or synchronized-readout acquisition."""

    sequence: SensingSequence
    repetition_interval_seconds: float
    reference_frequency_hz: float = 0.0
    baseline_bright_probability: float = 0.5
    analysis_contrast: float = 1.0
    analysis_phase_rad: float = 0.0
    pulse_model: PulseModel = "ideal"
    budget: HighResolutionBudget = field(default_factory=HighResolutionBudget)
    clock: ClockModel = field(default_factory=ClockModel)

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, SensingSequence):
            raise TypeError("sequence must be a SensingSequence")
        interval = _positive(
            self.repetition_interval_seconds,
            "repetition_interval_seconds",
        )
        if interval < self.sequence.total_duration_seconds:
            raise ValueError(
                "repetition_interval_seconds cannot be shorter than the sequence"
            )
        reference = float(self.reference_frequency_hz)
        phase = float(self.analysis_phase_rad)
        if not np.isfinite(reference) or not np.isfinite(phase):
            raise ValueError(
                "reference_frequency_hz and analysis_phase_rad must be finite"
            )
        baseline = _unit_interval(
            self.baseline_bright_probability,
            "baseline_bright_probability",
        )
        contrast = _unit_interval(self.analysis_contrast, "analysis_contrast")
        if self.pulse_model not in {"ideal", "finite"}:
            raise ValueError("pulse_model must be 'ideal' or 'finite'")
        if not isinstance(self.budget, HighResolutionBudget):
            raise TypeError("budget must be a HighResolutionBudget")
        if not isinstance(self.clock, ClockModel):
            raise TypeError("clock must be a ClockModel")
        object.__setattr__(self, "repetition_interval_seconds", interval)
        object.__setattr__(self, "reference_frequency_hz", reference)
        object.__setattr__(self, "baseline_bright_probability", baseline)
        object.__setattr__(self, "analysis_contrast", contrast)
        object.__setattr__(self, "analysis_phase_rad", phase)


@dataclass(frozen=True)
class QdyneResult:
    """Clocked single-quadrature Qdyne record and baseband spectrum.

    ``estimated_beat_frequency_hz`` is ``nan`` when the non-DC spectrum has no
    positive peak.
    """

    nominal_times_seconds: np.ndarray
    actual_times_seconds: np.ndarray
    bright_probability: np.ndarray
    normalized_signal: np.ndarray
    baseband_frequencies_hz: np.ndarray
    spectrum: np.ndarray
    filter_response_seconds: complex
    sensor_contrast: float
    intershot_envelope: np.ndarray
    raw_beat_frequency_hz: float
    expected_beat_frequency_hz: float
    alias_order: int
    estimated_beat_frequency_hz: float
    optical_readout: OpticalReadoutResult | None


@dataclass(frozen=True)
class SynchronizedReadoutResult:
    """Two-quadrature synchronized-readout record.

    ``estimated_beat_frequency_hz`` is ``nan`` when the spectrum is identically
    zero.
    """

    nominal_times_seconds: np.ndarray
    actual_times_seconds: np.ndarray
    in_phase_probability: np.ndarray
    quadrature_probability: np.ndarray
    complex_signal: np.ndarray
    baseband_frequencies_hz: np.ndarray
    spectrum: np.ndarray
    filter_response_seconds: complex
    sensor_contrast: float
    intershot_envelope: np.ndarray
    raw_beat_frequency_hz: float
    expected_beat_frequency_hz: float
    alias_order: int
    estimated_beat_frequency_hz: float


def simulate_qdyne(
    protocol: QdyneProtocol,
    *,
    signal_frequency_hz: float,
    field_amplitude_tesla: float,
    sensor_gamma_rad_s_t: float,
    shot_count: int,
    signal_phase_rad: float = 0.0,
    optical_model: OpticalReadoutModel | None = None,
    seed: int | None = None,
) -> QdyneResult:
    """Simulate a clocked coherent-field Qdyne record.

    The coherent sensor phase is calculated from the control-sequence toggling
    integral. ``optical_model`` optionally samples the existing effective
    Poisson detector without changing the expected spin-projection spectrum.
    """

    _validate_qdyne_inputs(
        protocol,
        signal_frequency_hz,
        field_amplitude_tesla,
        sensor_gamma_rad_s_t,
        shot_count,
        signal_phase_rad,
    )
    shot_count = int(shot_count)
    frequency = float(signal_frequency_hz)
    amplitude = float(field_amplitude_tesla)
    gamma = float(sensor_gamma_rad_s_t)
    signal_phase = float(signal_phase_rad)
    generator = np.random.default_rng(seed)
    nominal, actual = protocol.clock.sample_times(
        shot_count,
        protocol.repetition_interval_seconds,
        rng=generator,
    )
    response = complex(
        toggling_integral(
            protocol.sequence,
            2.0 * np.pi * frequency,
            pulse_model=protocol.pulse_model,
        )
    )
    sensor_contrast = protocol.budget.sensor_contrast(
        protocol.sequence.total_duration_seconds
    )
    envelope = protocol.budget.intershot_envelope(nominal)
    beat_phase = _beat_phase(protocol, frequency, nominal, actual, signal_phase)
    coherent_phase = (
        gamma
        * amplitude
        * abs(response)
        * envelope
        * np.cos(beat_phase + np.angle(response))
    )
    normalized = sensor_contrast * np.sin(coherent_phase + protocol.analysis_phase_rad)
    probability = np.clip(
        protocol.baseline_bright_probability
        + 0.5 * protocol.analysis_contrast * normalized,
        0.0,
        1.0,
    )
    frequency_axis, spectrum = _real_spectrum(
        normalized,
        protocol.repetition_interval_seconds,
    )
    raw_beat = frequency - protocol.reference_frequency_hz
    signed_alias, alias_order = _aliased_frequency(
        raw_beat,
        protocol.repetition_interval_seconds,
    )
    expected_beat = abs(signed_alias)
    estimated = _positive_peak_frequency(frequency_axis, spectrum)
    optical = None
    if optical_model is not None:
        if not isinstance(optical_model, OpticalReadoutModel):
            raise TypeError("optical_model must be an OpticalReadoutModel")
        optical = sample_optical_readout(
            optical_model,
            probability,
            sensing_seconds=protocol.sequence.total_duration_seconds,
            rng=generator,
        )
    return QdyneResult(
        nominal_times_seconds=nominal,
        actual_times_seconds=actual,
        bright_probability=probability,
        normalized_signal=normalized,
        baseband_frequencies_hz=frequency_axis,
        spectrum=spectrum,
        filter_response_seconds=response,
        sensor_contrast=sensor_contrast,
        intershot_envelope=envelope,
        raw_beat_frequency_hz=raw_beat,
        expected_beat_frequency_hz=expected_beat,
        alias_order=alias_order,
        estimated_beat_frequency_hz=estimated,
        optical_readout=optical,
    )


def simulate_synchronized_readout(
    protocol: QdyneProtocol,
    *,
    signal_frequency_hz: float,
    field_amplitude_tesla: float,
    sensor_gamma_rad_s_t: float,
    shot_count: int,
    signal_phase_rad: float = 0.0,
    seed: int | None = None,
) -> SynchronizedReadoutResult:
    """Simulate phase-preserving two-quadrature synchronized readout."""

    _validate_qdyne_inputs(
        protocol,
        signal_frequency_hz,
        field_amplitude_tesla,
        sensor_gamma_rad_s_t,
        shot_count,
        signal_phase_rad,
    )
    shot_count = int(shot_count)
    frequency = float(signal_frequency_hz)
    amplitude = float(field_amplitude_tesla)
    gamma = float(sensor_gamma_rad_s_t)
    generator = np.random.default_rng(seed)
    nominal, actual = protocol.clock.sample_times(
        shot_count,
        protocol.repetition_interval_seconds,
        rng=generator,
    )
    response = complex(
        toggling_integral(
            protocol.sequence,
            2.0 * np.pi * frequency,
            pulse_model=protocol.pulse_model,
        )
    )
    sensor_contrast = protocol.budget.sensor_contrast(
        protocol.sequence.total_duration_seconds
    )
    envelope = protocol.budget.intershot_envelope(nominal)
    phase = (
        _beat_phase(
            protocol,
            frequency,
            nominal,
            actual,
            float(signal_phase_rad),
        )
        + np.angle(response)
        + protocol.analysis_phase_rad
    )
    magnitude = gamma * amplitude * abs(response) * envelope
    in_phase = sensor_contrast * np.sin(magnitude * np.cos(phase))
    quadrature = sensor_contrast * np.sin(magnitude * np.sin(phase))
    p_i = np.clip(
        protocol.baseline_bright_probability
        + 0.5 * protocol.analysis_contrast * in_phase,
        0.0,
        1.0,
    )
    p_q = np.clip(
        protocol.baseline_bright_probability
        + 0.5 * protocol.analysis_contrast * quadrature,
        0.0,
        1.0,
    )
    complex_signal = in_phase + 1j * quadrature
    frequency_axis, spectrum = _complex_spectrum(
        complex_signal,
        protocol.repetition_interval_seconds,
    )
    raw_beat = frequency - protocol.reference_frequency_hz
    expected_beat, alias_order = _aliased_frequency(
        raw_beat,
        protocol.repetition_interval_seconds,
    )
    estimated = _peak_frequency(frequency_axis, spectrum)
    return SynchronizedReadoutResult(
        nominal_times_seconds=nominal,
        actual_times_seconds=actual,
        in_phase_probability=p_i,
        quadrature_probability=p_q,
        complex_signal=complex_signal,
        baseband_frequencies_hz=frequency_axis,
        spectrum=spectrum,
        filter_response_seconds=response,
        sensor_contrast=sensor_contrast,
        intershot_envelope=envelope,
        raw_beat_frequency_hz=raw_beat,
        expected_beat_frequency_hz=expected_beat,
        alias_order=alias_order,
        estimated_beat_frequency_hz=estimated,
    )


@dataclass(frozen=True)
class MemoryCorrelationResult:
    """Effective sensor-memory correlation signal and separate envelopes."""

    correlation_times_seconds: np.ndarray
    signal: np.ndarray
    sensor_contrast: float
    sample_envelope: np.ndarray
    diffusion_envelope: np.ndarray
    memory_envelope: np.ndarray
    transfer_retrieval_fidelity: float


def sensor_memory_correlation(
    correlation_times_seconds,
    *,
    frequency_hz: float,
    sensing_seconds: float,
    budget: HighResolutionBudget,
    contrast: float = 1.0,
    transfer_fidelity: float = 1.0,
    retrieval_fidelity: float = 1.0,
    phase_rad: float = 0.0,
) -> MemoryCorrelationResult:
    """Return an effective two-block sensor-memory correlation signal."""

    if not isinstance(budget, HighResolutionBudget):
        raise TypeError("budget must be a HighResolutionBudget")
    times = _nonnegative_times(
        correlation_times_seconds,
        "correlation_times_seconds",
    )
    frequency = float(frequency_hz)
    phase = float(phase_rad)
    if not np.isfinite(frequency) or not np.isfinite(phase):
        raise ValueError("frequency_hz and phase_rad must be finite")
    contrast = _unit_interval(contrast, "contrast")
    transfer = _unit_interval(transfer_fidelity, "transfer_fidelity")
    retrieval = _unit_interval(retrieval_fidelity, "retrieval_fidelity")
    sensor = budget.sensor_contrast(sensing_seconds) ** 2
    sample = _exponential_envelope(times, budget.sample_coherence_seconds)
    diffusion = _exponential_envelope(
        times,
        budget.diffusion_correlation_seconds,
    )
    memory = _exponential_envelope(times, budget.memory_coherence_seconds)
    fidelity = transfer * retrieval
    signal = (
        contrast
        * fidelity
        * sensor
        * sample
        * diffusion
        * memory
        * np.cos(2.0 * np.pi * frequency * times + phase)
    )
    return MemoryCorrelationResult(
        correlation_times_seconds=times,
        signal=signal,
        sensor_contrast=sensor,
        sample_envelope=sample,
        diffusion_envelope=diffusion,
        memory_envelope=memory,
        transfer_retrieval_fidelity=fidelity,
    )


@dataclass(frozen=True)
class CoherentNMRSite:
    """One first-order coherent nuclear resonance site.

    Repeating a value in ``scalar_couplings_hz`` represents coupling to
    multiple equivalent spin-one-half partners.
    """

    isotope: str
    chemical_shift_ppm: float
    amplitude: complex = 1.0
    transverse_relaxation_seconds: float = np.inf
    scalar_couplings_hz: tuple[float, ...] = ()
    phase_rad: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        isotope = str(self.isotope)
        if isotope not in ISOTOPE_GAMMA_HZ_PER_T:
            raise ValueError(f"unknown isotope preset: {isotope!r}")
        shift = float(self.chemical_shift_ppm)
        amplitude = complex(self.amplitude)
        phase = float(self.phase_rad)
        if not np.isfinite(shift) or not np.isfinite(amplitude.real):
            raise ValueError("chemical_shift_ppm and amplitude must be finite")
        if not np.isfinite(amplitude.imag) or not np.isfinite(phase):
            raise ValueError("amplitude and phase_rad must be finite")
        relaxation = _positive_or_infinite(
            self.transverse_relaxation_seconds,
            "transverse_relaxation_seconds",
        )
        couplings = tuple(float(value) for value in self.scalar_couplings_hz)
        if not np.all(np.isfinite(couplings)):
            raise ValueError("scalar_couplings_hz must be finite")
        object.__setattr__(self, "isotope", isotope)
        object.__setattr__(self, "chemical_shift_ppm", shift)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "transverse_relaxation_seconds", relaxation)
        object.__setattr__(self, "scalar_couplings_hz", couplings)
        object.__setattr__(self, "phase_rad", phase)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class CoherentNMRSpectrumResult:
    """Time-domain coherent signal and chemical-shift/J-resolved spectrum."""

    times_seconds: np.ndarray
    actual_times_seconds: np.ndarray
    fid: np.ndarray
    frequencies_hz: np.ndarray
    spectrum: np.ndarray
    reference_frequency_hz: float
    component_frequencies_hz: np.ndarray
    component_amplitudes: np.ndarray


def simulate_coherent_nmr_spectrum(
    sites: tuple[CoherentNMRSite, ...] | list[CoherentNMRSite],
    b0_tesla: float,
    times_seconds,
    *,
    reference_frequency_hz: float | None = None,
    diffusion_correlation_seconds: float = np.inf,
    clock: ClockModel | None = None,
    seed: int | None = None,
    window: bool = True,
) -> CoherentNMRSpectrumResult:
    """Simulate first-order coherent thermal or polarized NMR spectroscopy."""

    sites = tuple(sites)
    if not sites or not all(isinstance(site, CoherentNMRSite) for site in sites):
        raise ValueError("sites must contain at least one CoherentNMRSite")
    field_tesla = float(b0_tesla)
    if not np.isfinite(field_tesla):
        raise ValueError("b0_tesla must be finite")
    times = _uniform_times(times_seconds, "times_seconds")
    diffusion = _positive_or_infinite(
        diffusion_correlation_seconds,
        "diffusion_correlation_seconds",
    )
    clock = ClockModel() if clock is None else clock
    if not isinstance(clock, ClockModel):
        raise TypeError("clock must be a ClockModel")
    _, actual = clock.sample_times(
        times.size,
        float(times[1] - times[0]),
        seed=seed,
    )
    actual += times[0]
    carriers = np.array(
        [
            abs(ISOTOPE_GAMMA_HZ_PER_T[site.isotope] * field_tesla)
            * (1.0 + 1.0e-6 * site.chemical_shift_ppm)
            for site in sites
        ],
        dtype=np.float64,
    )
    reference = (
        float(carriers[0])
        if reference_frequency_hz is None
        else float(reference_frequency_hz)
    )
    if not np.isfinite(reference):
        raise ValueError("reference_frequency_hz must be finite")
    elapsed_actual = actual - actual[0]
    if np.any(np.diff(actual) <= 0.0):
        raise ValueError(
            "clock perturbations produced non-increasing physical sample times"
        )
    fid = np.zeros(times.size, dtype=np.complex128)
    component_frequencies = []
    component_amplitudes = []
    for site, carrier in zip(sites, carriers):
        envelope = _exponential_envelope(
            elapsed_actual,
            site.transverse_relaxation_seconds,
        )
        if np.isfinite(diffusion):
            envelope *= np.exp(-elapsed_actual / diffusion)
        modulation = np.ones_like(times)
        for coupling in site.scalar_couplings_hz:
            modulation *= np.cos(np.pi * coupling * elapsed_actual)
        complex_amplitude = site.amplitude * np.exp(1j * site.phase_rad)
        fid += (
            complex_amplitude
            * envelope
            * modulation
            * np.exp(2j * np.pi * (carrier * actual - reference * times))
        )
        offsets, weights = _first_order_multiplet(site.scalar_couplings_hz)
        component_frequencies.extend(carrier + offsets)
        component_amplitudes.extend(complex_amplitude * weights)
    transform_values = fid
    if window:
        transform_values = transform_values * np.hanning(times.size)
    transform = np.fft.fftshift(np.fft.fft(transform_values))
    offsets = np.fft.fftshift(np.fft.fftfreq(times.size, d=float(times[1] - times[0])))
    return CoherentNMRSpectrumResult(
        times_seconds=times,
        actual_times_seconds=actual,
        fid=fid,
        frequencies_hz=reference + offsets,
        spectrum=np.abs(transform),
        reference_frequency_hz=reference,
        component_frequencies_hz=np.asarray(
            component_frequencies,
            dtype=np.float64,
        ),
        component_amplitudes=np.asarray(
            component_amplitudes,
            dtype=np.complex128,
        ),
    )


@dataclass(frozen=True)
class DNPModel:
    """Optional effective dynamic-nuclear-polarization preparation."""

    enhancement_factor: float
    buildup_time_seconds: float
    nuclear_t1_seconds: float

    def __post_init__(self) -> None:
        enhancement = float(self.enhancement_factor)
        if not np.isfinite(enhancement):
            raise ValueError("enhancement_factor must be finite")
        object.__setattr__(self, "enhancement_factor", enhancement)
        object.__setattr__(
            self,
            "buildup_time_seconds",
            _positive(self.buildup_time_seconds, "buildup_time_seconds"),
        )
        object.__setattr__(
            self,
            "nuclear_t1_seconds",
            _positive(self.nuclear_t1_seconds, "nuclear_t1_seconds"),
        )


@dataclass(frozen=True)
class DNPPolarizationResult:
    """DNP build-up or relaxation record."""

    times_seconds: np.ndarray
    polarization: np.ndarray
    thermal_polarization: float
    steady_state_polarization: float
    pumping: bool


def dnp_polarization(
    model: DNPModel,
    times_seconds,
    thermal_polarization: float,
    *,
    pumping: bool = True,
    initial_polarization: float | None = None,
) -> DNPPolarizationResult:
    """Return bounded DNP build-up or nuclear-T1 relaxation."""

    if not isinstance(model, DNPModel):
        raise TypeError("model must be a DNPModel")
    times = _nonnegative_times(times_seconds, "times_seconds")
    thermal = float(thermal_polarization)
    if not np.isfinite(thermal) or abs(thermal) > 1.0:
        raise ValueError("thermal_polarization must be finite and lie in [-1, 1]")
    steady = float(np.clip(model.enhancement_factor * thermal, -1.0, 1.0))
    initial = thermal if initial_polarization is None else float(initial_polarization)
    if not np.isfinite(initial) or abs(initial) > 1.0:
        raise ValueError("initial_polarization must be finite and lie in [-1, 1]")
    if pumping:
        polarization = steady + (initial - steady) * np.exp(
            -times / model.buildup_time_seconds
        )
    else:
        polarization = thermal + (initial - thermal) * np.exp(
            -times / model.nuclear_t1_seconds
        )
    return DNPPolarizationResult(
        times_seconds=times,
        polarization=polarization,
        thermal_polarization=thermal,
        steady_state_polarization=steady,
        pumping=bool(pumping),
    )


def _beat_phase(
    protocol: QdyneProtocol,
    signal_frequency_hz: float,
    nominal_times: np.ndarray,
    actual_times: np.ndarray,
    signal_phase_rad: float,
) -> np.ndarray:
    return (
        2.0
        * np.pi
        * (
            signal_frequency_hz * actual_times
            - protocol.reference_frequency_hz * nominal_times
        )
        + signal_phase_rad
    )


def _aliased_frequency(
    frequency_hz: float,
    interval_seconds: float,
) -> tuple[float, int]:
    sample_rate = 1.0 / interval_seconds
    alias = (float(frequency_hz) + 0.5 * sample_rate) % sample_rate - 0.5 * sample_rate
    order = int(np.rint((float(frequency_hz) - alias) / sample_rate))
    return float(alias), order


def _real_spectrum(
    values: np.ndarray,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    window = np.hanning(centered.size)
    frequencies = np.fft.rfftfreq(centered.size, d=interval_seconds)
    return frequencies, np.abs(np.fft.rfft(centered * window))


def _complex_spectrum(
    values: np.ndarray,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=np.complex128) - np.mean(values)
    frequencies = np.fft.fftshift(np.fft.fftfreq(centered.size, d=interval_seconds))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(centered * np.hanning(centered.size))))
    return frequencies, spectrum


def _positive_peak_frequency(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    if spectrum.size <= 1:
        return float("nan")
    positive_spectrum = spectrum[1:]
    if not np.any(positive_spectrum > 0.0):
        return float("nan")
    return float(frequencies[1 + int(np.argmax(positive_spectrum))])


def _peak_frequency(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
) -> float:
    if spectrum.size == 0 or not np.any(spectrum > 0.0):
        return float("nan")
    return float(frequencies[int(np.argmax(spectrum))])


def _first_order_multiplet(
    couplings_hz: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.array([0.0])
    weights = np.array([1.0])
    for coupling in couplings_hz:
        offsets = np.concatenate((offsets - 0.5 * coupling, offsets + 0.5 * coupling))
        weights = np.concatenate((0.5 * weights, 0.5 * weights))
    order = np.argsort(offsets)
    offsets = offsets[order]
    weights = weights[order]
    unique_offsets = []
    unique_weights = []
    for offset, weight in zip(offsets, weights):
        if unique_offsets and np.isclose(
            offset,
            unique_offsets[-1],
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            unique_weights[-1] += weight
        else:
            unique_offsets.append(float(offset))
            unique_weights.append(float(weight))
    return np.asarray(unique_offsets), np.asarray(unique_weights)


def _validate_qdyne_inputs(
    protocol: QdyneProtocol,
    signal_frequency_hz: float,
    field_amplitude_tesla: float,
    sensor_gamma_rad_s_t: float,
    shot_count: int,
    signal_phase_rad: float,
) -> None:
    if not isinstance(protocol, QdyneProtocol):
        raise TypeError("protocol must be a QdyneProtocol")
    for value, name in (
        (signal_frequency_hz, "signal_frequency_hz"),
        (field_amplitude_tesla, "field_amplitude_tesla"),
        (sensor_gamma_rad_s_t, "sensor_gamma_rad_s_t"),
        (signal_phase_rad, "signal_phase_rad"),
    ):
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    _positive_integer(shot_count, "shot_count")


def _uniform_times(values, name: str) -> np.ndarray:
    times = _nonnegative_times(values, name)
    if times.size < 2:
        raise ValueError(f"{name} must contain at least two points")
    spacing = np.diff(times)
    if np.any(spacing <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1.0e-8, atol=1.0e-15):
        raise ValueError(f"{name} must be uniformly sampled")
    return times


def _nonnegative_times(values, name: str) -> np.ndarray:
    times = np.asarray(values, dtype=np.float64).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return times


def _exponential_envelope(times: np.ndarray, duration: float) -> np.ndarray:
    if np.isinf(duration):
        return np.ones_like(times)
    return np.exp(-times / duration)


def _positive(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be positive and finite")
    return out


def _nonnegative(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be finite and non-negative")
    return out


def _positive_or_infinite(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or np.isnan(out):
        raise ValueError(f"{name} must be positive or infinite")
    return out


def _unit_interval(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0 or not np.isfinite(out):
        raise ValueError(f"{name} must lie in [0, 1]")
    return out


def _positive_integer(value: int, name: str) -> int:
    out = int(value)
    if out != value or out <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return out


__all__ = [
    "ClockModel",
    "CoherentNMRSite",
    "CoherentNMRSpectrumResult",
    "DNPModel",
    "DNPPolarizationResult",
    "HighResolutionBudget",
    "MemoryCorrelationResult",
    "QdyneProtocol",
    "QdyneResult",
    "SynchronizedReadoutResult",
    "dnp_polarization",
    "sensor_memory_correlation",
    "simulate_coherent_nmr_spectrum",
    "simulate_qdyne",
    "simulate_synchronized_readout",
]
