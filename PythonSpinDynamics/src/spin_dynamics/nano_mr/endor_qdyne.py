"""Electron-nuclear-double-resonance QDyne for a resolved nuclear spin.

The model follows Meinel et al., Commun. Phys. 6, 302 (2023),
https://doi.org/10.1038/s42005-023-01419-2. Phase-coherent nuclear RF
pulses map transverse nuclear polarization onto ``I_z``. A Ramsey block on
the defect sensor then weakly measures the static longitudinal hyperfine
interaction, avoiding microwave dynamical-decoupling pulses at the nuclear
Larmor frequency.

Hyperfine couplings and RF Rabi frequencies are supplied in spectroscopic
cycles per second (Hz). Internally, rotation angles use radians and therefore
include the corresponding factor of ``2*pi``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from spin_dynamics.nano_mr.high_resolution import ClockModel
from spin_dynamics.nano_mr.readout import (
    OpticalReadoutModel,
    OpticalReadoutResult,
    sample_optical_readout,
)


@dataclass(frozen=True)
class EndorQdyneProtocol:
    """One phase-coherent ENDOR-QDyne acquisition.

    ``free_precession_seconds`` is the part of each cycle during which the
    target accumulates Larmor phase relative to the RF source.
    ``rf_phase_increment_rad`` implements deliberate phase cycling; the
    ``pi/2`` increment used by Meinel et al. moves a resonant signal to one
    quarter of the sampling frequency.

    ``residual_sensing_phase_rad`` is the target phase left after the sensing
    period. It is zero when resonant driving or RF refocusing satisfies the
    paper's ``beta = 2*pi*k`` back-rotation condition.
    """

    free_precession_seconds: float
    sensing_seconds: float
    repetition_interval_seconds: float
    rf_reference_frequency_hz: float
    longitudinal_hyperfine_hz: float
    nuclear_rabi_frequency_hz: float = np.inf
    rf_phase_increment_rad: float = 0.0
    residual_sensing_phase_rad: float = 0.0
    repolarization_seconds: float = 0.0
    optical_readout_seconds: float = 0.0
    wait_seconds: float = 0.0
    sensor_initialization_fidelity: float = 1.0
    sensor_coherence_seconds: float = np.inf
    intrinsic_nuclear_decay_rate_per_second: float = 0.0
    baseline_bright_probability: float = 0.5
    analysis_contrast: float = 1.0
    clock: ClockModel = field(default_factory=ClockModel)

    def __post_init__(self) -> None:
        for name in (
            "free_precession_seconds",
            "sensing_seconds",
            "repetition_interval_seconds",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "nuclear_rabi_frequency_hz",
            _positive_or_infinite(
                self.nuclear_rabi_frequency_hz,
                "nuclear_rabi_frequency_hz",
            ),
        )
        for name in (
            "repolarization_seconds",
            "optical_readout_seconds",
            "wait_seconds",
            "intrinsic_nuclear_decay_rate_per_second",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative(getattr(self, name), name),
            )
        for name in (
            "rf_reference_frequency_hz",
            "longitudinal_hyperfine_hz",
            "rf_phase_increment_rad",
            "residual_sensing_phase_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "sensor_initialization_fidelity",
            _unit_interval(
                self.sensor_initialization_fidelity,
                "sensor_initialization_fidelity",
            ),
        )
        object.__setattr__(
            self,
            "sensor_coherence_seconds",
            _positive_or_infinite(
                self.sensor_coherence_seconds,
                "sensor_coherence_seconds",
            ),
        )
        for name in ("baseline_bright_probability", "analysis_contrast"):
            object.__setattr__(
                self,
                name,
                _unit_interval(getattr(self, name), name),
            )
        if not isinstance(self.clock, ClockModel):
            raise TypeError("clock must be a ClockModel")
        if self.repetition_interval_seconds + 1.0e-15 < self.minimum_cycle_seconds:
            raise ValueError(
                "repetition_interval_seconds is shorter than the modeled "
                "free-precession, RF-control, sensing, and overhead times"
            )

    @property
    def rf_pi_over_two_seconds(self) -> float:
        """Duration of an RF pi/2 pulse, or zero for ideal RF pulses."""

        if np.isinf(self.nuclear_rabi_frequency_hz):
            return 0.0
        return 0.25 / self.nuclear_rabi_frequency_hz

    @property
    def rf_three_pi_over_two_seconds(self) -> float:
        """Duration of an RF 3pi/2 pulse, or zero for ideal RF pulses."""

        return 3.0 * self.rf_pi_over_two_seconds

    @property
    def rf_control_seconds(self) -> float:
        """Total duration of the mapping and back-rotation RF pulses."""

        return self.rf_pi_over_two_seconds + self.rf_three_pi_over_two_seconds

    @property
    def minimum_cycle_seconds(self) -> float:
        """Sum of all explicitly modeled operations in one cycle."""

        return (
            self.free_precession_seconds
            + self.rf_control_seconds
            + self.sensing_seconds
            + self.repolarization_seconds
            + self.optical_readout_seconds
            + self.wait_seconds
        )

    @property
    def measurement_strength_rad(self) -> float:
        """Ramsey measurement strength ``alpha = 2*pi*Azz*tau_sens``."""

        return 2.0 * np.pi * self.longitudinal_hyperfine_hz * self.sensing_seconds

    @property
    def sensor_contrast(self) -> float:
        """Ramsey visibility retained during the sensing interval."""

        if np.isinf(self.sensor_coherence_seconds):
            return 1.0
        return float(np.exp(-self.sensing_seconds / self.sensor_coherence_seconds))


@dataclass(frozen=True)
class EndorQdyneResult:
    """ENDOR-QDyne nuclear record, sensor signal, and baseband spectrum.

    ``estimated_beat_frequency_hz`` is ``nan`` when the non-DC spectrum has no
    positive peak.
    """

    nominal_times_seconds: np.ndarray
    actual_times_seconds: np.ndarray
    nuclear_z_expectation: np.ndarray
    sensor_z_expectation: np.ndarray
    normalized_signal: np.ndarray
    bright_probability: np.ndarray
    baseband_frequencies_hz: np.ndarray
    spectrum: np.ndarray
    coherence_envelope: np.ndarray
    measurement_strength_rad: float
    sensor_contrast: float
    initialization_decay_rate_per_second: float
    leading_initialization_decay_rate_per_second: float
    weak_measurement_backaction_rate_per_second: float
    raw_detuning_hz: float
    raw_cycle_frequency_hz: float
    expected_beat_frequency_hz: float
    alias_order: int
    estimated_beat_frequency_hz: float
    optical_readout: OpticalReadoutResult | None


def initialization_infidelity_decay_rate(
    longitudinal_hyperfine_hz: float,
    interval_seconds: float,
    initialization_fidelity: float,
    *,
    leading_order: bool = False,
) -> float:
    """Return the initialization-infidelity decay rate from Meinel et al.

    The exact expression is Supplementary Eq. 23. ``leading_order=True``
    returns main-text Eq. 3,

    ``Gamma = 2*(1-f)*sin(2*pi*Azz*tau)**2/tau``.

    ``Azz`` is accepted as a cyclic frequency in Hz, so the phase contains
    ``2*pi``. This convention follows the paper's Hamiltonian
    ``H_eff = 2*pi*Azz*Sz*Iz`` and reproduces its approximately 1 kHz numerical
    estimate for ``Azz=6 kHz``, ``tau=105 us``, and ``f=0.9``. The paper's
    nearby prose also labels ``Azz*tau=0.63`` as a phase, which is inconsistent
    with that Hamiltonian and estimate; this implementation resolves the
    ambiguity by making the spectroscopic-Hz convention explicit.
    """

    coupling = float(longitudinal_hyperfine_hz)
    if not np.isfinite(coupling):
        raise ValueError("longitudinal_hyperfine_hz must be finite")
    interval = _positive(interval_seconds, "interval_seconds")
    fidelity = _unit_interval(initialization_fidelity, "initialization_fidelity")
    phase = 2.0 * np.pi * coupling * interval
    if leading_order:
        return float(2.0 * (1.0 - fidelity) * np.sin(phase) ** 2 / interval)
    coherence_squared = (
        np.cos(phase) ** 2 + (2.0 * fidelity - 1.0) ** 2 * np.sin(phase) ** 2
    )
    if coherence_squared <= 0.0:
        return np.inf
    return float(-0.5 * np.log(coherence_squared) / interval)


def meinel_2023_endor_qdyne_protocol(
    *,
    rf_reference_frequency_hz: float = 0.0,
    sensor_initialization_fidelity: float = 0.9,
    intrinsic_nuclear_decay_rate_per_second: float = 0.0,
) -> EndorQdyneProtocol:
    """Return the proof-of-principle parameters reported by Meinel et al.

    The preset uses 10 us free precession, 15 us sensor Ramsey sensing,
    a 66 us nuclear Rabi period, 2.5 us repolarization, 1.9 us optical
    readout, 10 us wait time, a 105.5 us sampling interval, 6 kHz
    longitudinal hyperfine coupling, and a pi/2 RF phase increment.
    """

    return EndorQdyneProtocol(
        free_precession_seconds=10.0e-6,
        sensing_seconds=15.0e-6,
        repetition_interval_seconds=105.5e-6,
        rf_reference_frequency_hz=rf_reference_frequency_hz,
        longitudinal_hyperfine_hz=6.0e3,
        nuclear_rabi_frequency_hz=1.0 / 66.0e-6,
        rf_phase_increment_rad=np.pi / 2.0,
        repolarization_seconds=2.5e-6,
        optical_readout_seconds=1.9e-6,
        wait_seconds=10.0e-6,
        sensor_initialization_fidelity=sensor_initialization_fidelity,
        intrinsic_nuclear_decay_rate_per_second=(
            intrinsic_nuclear_decay_rate_per_second
        ),
    )


def simulate_endor_qdyne(
    protocol: EndorQdyneProtocol,
    *,
    target_frequency_hz: float,
    shot_count: int,
    initial_phase_rad: float = 0.0,
    include_measurement_backaction: bool = True,
    optical_model: OpticalReadoutModel | None = None,
    seed: int | None = None,
) -> EndorQdyneResult:
    """Simulate coherent-basis-mapping ENDOR-QDyne.

    With back-action disabled and unit coherence, the result is exactly the
    paper's Eqs. 1-2:

    ``<Iz>(n) = 0.5*cos((omega_L-omega_i)*n*tau_fid + phase_cycle)``

    and ``<Sz>(n) = -sin(alpha)*<Iz>(n)``.

    With back-action enabled, the transverse nuclear Bloch vector is updated
    using Supplementary Eq. 14. Initialization infidelity uses the exact
    Supplementary Eq. 23 as an independent inter-cycle envelope.
    """

    if not isinstance(protocol, EndorQdyneProtocol):
        raise TypeError("protocol must be an EndorQdyneProtocol")
    target = float(target_frequency_hz)
    initial_phase = float(initial_phase_rad)
    if not np.isfinite(target):
        raise ValueError("target_frequency_hz must be finite")
    if not np.isfinite(initial_phase):
        raise ValueError("initial_phase_rad must be finite")
    if not isinstance(include_measurement_backaction, (bool, np.bool_)):
        raise TypeError("include_measurement_backaction must be boolean")
    count = _integer_at_least(shot_count, "shot_count", 2)
    generator = np.random.default_rng(seed)
    nominal, actual = protocol.clock.sample_times(
        count,
        protocol.repetition_interval_seconds,
        rng=generator,
    )

    # Only the free-precession fraction of the wall-clock record accumulates
    # target/reference detuning. Programmed RF phase cycling remains tied to
    # the nominal clock, as in a phase-coherent AWG implementation.
    duty = protocol.free_precession_seconds / protocol.repetition_interval_seconds
    cycle_index = nominal / protocol.repetition_interval_seconds
    detuning = target - protocol.rf_reference_frequency_hz
    phase = (
        2.0 * np.pi * duty * (detuning * nominal + target * (actual - nominal))
        + cycle_index
        * (protocol.rf_phase_increment_rad + protocol.residual_sensing_phase_rad)
        + initial_phase
    )
    alpha = protocol.measurement_strength_rad
    if include_measurement_backaction:
        nuclear_z = _nuclear_z_with_backaction(phase, alpha)
    else:
        nuclear_z = 0.5 * np.cos(phase)

    exact_initialization_rate = initialization_infidelity_decay_rate(
        protocol.longitudinal_hyperfine_hz,
        protocol.repetition_interval_seconds,
        protocol.sensor_initialization_fidelity,
    )
    leading_initialization_rate = initialization_infidelity_decay_rate(
        protocol.longitudinal_hyperfine_hz,
        protocol.repetition_interval_seconds,
        protocol.sensor_initialization_fidelity,
        leading_order=True,
    )
    total_intercycle_rate = (
        exact_initialization_rate + protocol.intrinsic_nuclear_decay_rate_per_second
    )
    if np.isinf(total_intercycle_rate):
        # Preserve the initialized state at t=0 while representing complete
        # dephasing at every later cycle. Directly evaluating exp(-inf * 0)
        # would otherwise introduce a spurious NaN in the first sample.
        envelope = np.zeros_like(nominal)
        envelope[nominal == 0.0] = 1.0
    else:
        envelope = np.exp(-total_intercycle_rate * nominal)
    nuclear_z = nuclear_z * envelope
    sensor_z = -protocol.sensor_contrast * np.sin(alpha) * nuclear_z
    normalized = 2.0 * sensor_z
    probability = np.clip(
        protocol.baseline_bright_probability + protocol.analysis_contrast * sensor_z,
        0.0,
        1.0,
    )
    frequencies, spectrum = _real_spectrum(
        normalized,
        protocol.repetition_interval_seconds,
    )
    raw_detuning = detuning
    raw_cycle_frequency = raw_detuning * duty + (
        protocol.rf_phase_increment_rad + protocol.residual_sensing_phase_rad
    ) / (2.0 * np.pi * protocol.repetition_interval_seconds)
    signed_alias, alias_order = _aliased_frequency(
        raw_cycle_frequency,
        protocol.repetition_interval_seconds,
    )
    expected = abs(signed_alias)
    estimated = _positive_peak_frequency(frequencies, spectrum)
    optical = None
    if optical_model is not None:
        if not isinstance(optical_model, OpticalReadoutModel):
            raise TypeError("optical_model must be an OpticalReadoutModel")
        optical = sample_optical_readout(
            optical_model,
            probability,
            sensing_seconds=protocol.sensing_seconds,
            rng=generator,
        )
    return EndorQdyneResult(
        nominal_times_seconds=nominal,
        actual_times_seconds=actual,
        nuclear_z_expectation=nuclear_z,
        sensor_z_expectation=sensor_z,
        normalized_signal=normalized,
        bright_probability=probability,
        baseband_frequencies_hz=frequencies,
        spectrum=spectrum,
        coherence_envelope=envelope,
        measurement_strength_rad=alpha,
        sensor_contrast=protocol.sensor_contrast,
        initialization_decay_rate_per_second=exact_initialization_rate,
        leading_initialization_decay_rate_per_second=(leading_initialization_rate),
        weak_measurement_backaction_rate_per_second=(
            alpha**2 / (4.0 * protocol.repetition_interval_seconds)
        ),
        raw_detuning_hz=raw_detuning,
        raw_cycle_frequency_hz=raw_cycle_frequency,
        expected_beat_frequency_hz=expected,
        alias_order=alias_order,
        estimated_beat_frequency_hz=estimated,
        optical_readout=optical,
    )


def _nuclear_z_with_backaction(
    phase_rad: np.ndarray,
    measurement_strength_rad: float,
) -> np.ndarray:
    """Apply the Supplementary Eq. 14 map for a possibly noisy phase record."""

    phase = np.asarray(phase_rad, dtype=np.float64)
    nuclear_z = np.empty_like(phase)
    state = np.array(
        [-0.5 * np.sin(phase[0]), 0.5 * np.cos(phase[0])],
        dtype=np.float64,
    )
    backaction = float(np.cos(measurement_strength_rad))
    for index in range(phase.size):
        nuclear_z[index] = state[1]
        if index + 1 == phase.size:
            break
        state[0] *= backaction
        increment = phase[index + 1] - phase[index]
        cosine = np.cos(increment)
        sine = np.sin(increment)
        state = np.array(
            [
                cosine * state[0] - sine * state[1],
                sine * state[0] + cosine * state[1],
            ]
        )
    return nuclear_z


def _real_spectrum(
    values: np.ndarray,
    interval_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=np.float64) - np.mean(values)
    window = np.hanning(centered.size)
    frequencies = np.fft.rfftfreq(centered.size, d=interval_seconds)
    spectrum = np.abs(np.fft.rfft(centered * window))
    return frequencies, spectrum


def _aliased_frequency(
    frequency_hz: float,
    interval_seconds: float,
) -> tuple[float, int]:
    sample_rate = 1.0 / interval_seconds
    alias = (float(frequency_hz) + 0.5 * sample_rate) % sample_rate - 0.5 * sample_rate
    order = int(np.rint((float(frequency_hz) - alias) / sample_rate))
    return float(alias), order


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


def _positive(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be positive and finite")
    return out


def _positive_or_infinite(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0 or np.isnan(out):
        raise ValueError(f"{name} must be positive or infinite")
    return out


def _nonnegative(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or not np.isfinite(out):
        raise ValueError(f"{name} must be finite and non-negative")
    return out


def _unit_interval(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0 or out > 1.0 or not np.isfinite(out):
        raise ValueError(f"{name} must lie in [0, 1]")
    return out


def _integer_at_least(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    out = int(value)
    if out != value or out < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return out


__all__ = [
    "EndorQdyneProtocol",
    "EndorQdyneResult",
    "initialization_infidelity_decay_rate",
    "meinel_2023_endor_qdyne_protocol",
    "simulate_endor_qdyne",
]
