"""Static EFG-distribution models for NQR inhomogeneous broadening."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import warnings

import numpy as np

from spin_dynamics.noise import NoiseMetadata, NoiseSpec, add_received_noise
from spin_dynamics.nqr.hamiltonians import TAU, diagonalize_site
from spin_dynamics.nqr.orientations import (
    OrientationSample,
    normalize_orientations,
    powder_average_grid,
)
from spin_dynamics.nqr.pulses import SelectivePulse, apply_selective_pulse
from spin_dynamics.nqr.simulation import (
    SLSEResult,
    _require_spin_one_selective_pulse_site,
    equilibrium_density,
    simulate_slse,
    transition_signal,
)
from spin_dynamics.nqr.systems import QuadrupolarSite


OrientationInput = str | tuple[OrientationSample, ...] | list[OrientationSample]


@dataclass(frozen=True)
class EFGIsochromat:
    """One static EFG variant in an inhomogeneous NQR ensemble."""

    site: QuadrupolarSite
    weight: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        weight = float(self.weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("weight must be non-negative and finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "label", str(self.label))


@dataclass(frozen=True)
class EFGDistribution:
    """Normalized collection of static quadrupolar-site variants."""

    isochromats: tuple[EFGIsochromat, ...]

    def __post_init__(self) -> None:
        if not self.isochromats:
            raise ValueError("isochromats must not be empty")
        total = float(sum(item.weight for item in self.isochromats))
        if total <= 0:
            raise ValueError("at least one isochromat must have positive weight")
        normalized = tuple(
            EFGIsochromat(item.site, item.weight / total, item.label)
            for item in self.isochromats
        )
        object.__setattr__(self, "isochromats", normalized)

    @property
    def weights(self) -> np.ndarray:
        """Return normalized isochromat weights."""

        return np.array([item.weight for item in self.isochromats], dtype=np.float64)


@dataclass(frozen=True)
class NQRFIDDistributionResult:
    """Time-domain and spectral NQR signal from an EFG distribution."""

    times: np.ndarray
    signal: np.ndarray
    spectrum_frequencies_hz: np.ndarray
    spectrum: np.ndarray
    carrier_frequency_hz: float
    isochromat_frequencies_hz: np.ndarray
    isochromat_amplitudes: np.ndarray
    distribution: EFGDistribution


@dataclass(frozen=True)
class SLSEDistributionResult:
    """SLSE echo train summed over a static EFG distribution."""

    echo_times: np.ndarray
    echo_amplitudes: np.ndarray
    local_results: tuple[SLSEResult, ...]
    distribution: EFGDistribution


@dataclass(frozen=True)
class WindowDeconvolutionResult:
    """Regularized deconvolution of a finite acquisition-window spectrum."""

    deconvolved_spectrum: np.ndarray
    regularization_strength: float
    residual_norm: float
    solution_norm: float


@dataclass(frozen=True)
class SLSEAcquisitionSpectrumResult:
    """Finite-window SLSE echo acquisition and spectrum."""

    acquisition_times_seconds: np.ndarray
    clean_echo_signal: np.ndarray
    echo_signal: np.ndarray
    spectrum_frequencies_hz: np.ndarray
    clean_spectrum: np.ndarray
    spectrum: np.ndarray
    carrier_frequency_hz: float
    isochromat_frequencies_hz: np.ndarray
    isochromat_amplitudes: np.ndarray
    selected_echo_index: int
    echo_train: SLSEDistributionResult
    noise_metadata: NoiseMetadata | None = None
    deconvolution: WindowDeconvolutionResult | None = None


@dataclass(frozen=True)
class EFGRephasingAnalysis:
    """Discretization estimate for a static EFG frequency grid."""

    max_frequency_gap_hz: float
    rephase_time_seconds: float
    max_time_seconds: float
    safety_factor: float
    required_max_gap_hz: float
    recommended_numpts: int | None
    ok: bool


def _as_orientations(orientations: OrientationInput) -> tuple[OrientationSample, ...]:
    if isinstance(orientations, str):
        if orientations == "powder":
            return powder_average_grid()
        if orientations == "single":
            return normalize_orientations(
                [OrientationSample(b1_direction_pas=(1.0, 0.0, 0.0))]
            )
        raise ValueError("orientations string must be 'powder' or 'single'")
    return normalize_orientations(tuple(orientations))


def _distribution_frequencies(
    distribution: EFGDistribution,
    transition_label: str,
) -> np.ndarray:
    return np.array(
        [
            diagonalize_site(item.site).transition(transition_label).frequency_hz
            for item in distribution.isochromats
        ],
        dtype=np.float64,
    )


def _time_domain_noise(noise: NoiseSpec | Mapping | float | int | None):
    if noise is None or isinstance(noise, NoiseSpec):
        return noise
    if isinstance(noise, Mapping):
        return noise if "domain" in noise else {**noise, "domain": "time"}
    return NoiseSpec(sigma=float(noise), domain="time")


def analyze_efg_rephasing(
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...],
    max_time_seconds: float,
    safety_factor: float = 1.25,
) -> EFGRephasingAnalysis:
    """Estimate whether an EFG isochromat grid may discretely rephase."""

    frequencies = np.unique(
        np.sort(np.asarray(frequencies_hz, dtype=np.float64).reshape(-1))
    )
    max_time_seconds = float(max_time_seconds)
    safety_factor = float(safety_factor)
    if max_time_seconds < 0:
        raise ValueError("max_time_seconds must be non-negative")
    if safety_factor <= 0:
        raise ValueError("safety_factor must be positive")
    if frequencies.size < 2:
        return EFGRephasingAnalysis(
            max_frequency_gap_hz=np.inf,
            rephase_time_seconds=np.inf,
            max_time_seconds=max_time_seconds,
            safety_factor=safety_factor,
            required_max_gap_hz=np.inf if max_time_seconds == 0 else 1.0 / (
                safety_factor * max_time_seconds
            ),
            recommended_numpts=None,
            ok=True,
        )

    gaps = np.diff(frequencies)
    max_gap = float(np.max(gaps))
    rephase_time = np.inf if max_gap <= 0 else 1.0 / max_gap
    required_max_gap = (
        np.inf if max_time_seconds == 0 else 1.0 / (safety_factor * max_time_seconds)
    )
    ok = bool(rephase_time > safety_factor * max_time_seconds)
    recommended = None
    if not ok:
        span = float(np.max(frequencies) - np.min(frequencies))
        recommended = int(np.ceil(span / required_max_gap)) + 1

    return EFGRephasingAnalysis(
        max_frequency_gap_hz=max_gap,
        rephase_time_seconds=float(rephase_time),
        max_time_seconds=max_time_seconds,
        safety_factor=safety_factor,
        required_max_gap_hz=float(required_max_gap),
        recommended_numpts=recommended,
        ok=ok,
    )


def check_efg_rephasing(
    frequencies_hz: np.ndarray | list[float] | tuple[float, ...],
    max_time_seconds: float,
    safety_factor: float = 1.25,
    action: str = "warn",
) -> EFGRephasingAnalysis:
    """Warn or raise when an EFG grid may produce rephasing artifacts."""

    analysis = analyze_efg_rephasing(frequencies_hz, max_time_seconds, safety_factor)
    if analysis.ok or action == "ignore":
        return analysis

    message = (
        "EFG isochromat grid may rephase before the simulated response ends: "
        f"max_gap_hz={analysis.max_frequency_gap_hz:.6g}, "
        f"rephase_time_seconds={analysis.rephase_time_seconds:.6g}, "
        f"max_time_seconds={analysis.max_time_seconds:.6g}, "
        f"safety_factor={analysis.safety_factor:.6g}. "
        f"Use at least samples={analysis.recommended_numpts} across this "
        "frequency span or set rephase_action='ignore' after checking convergence."
    )
    if action == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    elif action == "raise":
        raise RuntimeError(message)
    else:
        raise ValueError("action must be 'ignore', 'warn', or 'raise'")
    return analysis


def gaussian_efg_distribution(
    base_site: QuadrupolarSite,
    *,
    quadrupole_std_hz: float = 0.0,
    eta_std: float = 0.0,
    samples: int = 31,
    sigma_span: float = 3.0,
) -> EFGDistribution:
    """Return a Gaussian static distribution of ``nu_Q`` and optionally ``eta``."""

    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    quadrupole_std_hz = float(quadrupole_std_hz)
    eta_std = float(eta_std)
    sigma_span = float(sigma_span)
    if quadrupole_std_hz < 0 or eta_std < 0:
        raise ValueError("standard deviations must be non-negative")
    if not np.isfinite(sigma_span) or sigma_span <= 0:
        raise ValueError("sigma_span must be positive and finite")

    nu_offsets = (
        np.array([0.0], dtype=np.float64)
        if quadrupole_std_hz == 0 or samples == 1
        else np.linspace(-sigma_span, sigma_span, samples)
    )
    eta_offsets = (
        np.array([0.0], dtype=np.float64)
        if eta_std == 0 or samples == 1
        else np.linspace(-sigma_span, sigma_span, samples)
    )
    nu_axis = base_site.quadrupole_frequency_hz + quadrupole_std_hz * nu_offsets
    eta_axis = base_site.eta + eta_std * eta_offsets

    isochromats: list[EFGIsochromat] = []
    for nu_q in nu_axis:
        nu_weight = 1.0 if quadrupole_std_hz == 0 else np.exp(
            -0.5 * ((nu_q - base_site.quadrupole_frequency_hz) / quadrupole_std_hz) ** 2
        )
        for eta in eta_axis:
            if eta < 0.0 or eta > 1.0:
                continue
            eta_weight = 1.0 if eta_std == 0 else np.exp(
                -0.5 * ((eta - base_site.eta) / eta_std) ** 2
            )
            site = QuadrupolarSite(
                spin=base_site.spin,
                quadrupole_frequency_hz=float(nu_q),
                eta=float(eta),
                gamma_hz_per_t=base_site.gamma_hz_per_t,
                isotope=base_site.isotope,
                label=base_site.label,
            )
            isochromats.append(EFGIsochromat(site, float(nu_weight * eta_weight)))
    return EFGDistribution(tuple(isochromats))


def temperature_efg_distribution(
    base_site: QuadrupolarSite,
    temperatures_kelvin: np.ndarray | list[float] | tuple[float, ...],
    *,
    weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
    reference_temperature_kelvin: float = 293.15,
    quadrupole_slope_hz_per_kelvin: float = 0.0,
    eta_slope_per_kelvin: float = 0.0,
) -> EFGDistribution:
    """Map a static temperature distribution onto ``nu_Q`` and ``eta`` shifts."""

    temperatures = np.asarray(temperatures_kelvin, dtype=np.float64).reshape(-1)
    if temperatures.size == 0:
        raise ValueError("temperatures_kelvin must not be empty")
    if weights is None:
        weights_arr = np.ones(temperatures.size, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights_arr.size != temperatures.size:
            raise ValueError("weights must match temperatures_kelvin")
    isochromats: list[EFGIsochromat] = []
    for temperature, weight in zip(temperatures, weights_arr):
        delta_t = float(temperature - reference_temperature_kelvin)
        eta = base_site.eta + eta_slope_per_kelvin * delta_t
        if eta < 0.0 or eta > 1.0:
            continue
        site = QuadrupolarSite(
            spin=base_site.spin,
            quadrupole_frequency_hz=(
                base_site.quadrupole_frequency_hz
                + quadrupole_slope_hz_per_kelvin * delta_t
            ),
            eta=float(eta),
            gamma_hz_per_t=base_site.gamma_hz_per_t,
            isotope=base_site.isotope,
            label=base_site.label,
        )
        isochromats.append(EFGIsochromat(site, float(weight), f"{temperature:g} K"))
    return EFGDistribution(tuple(isochromats))


def fid_spectrum(
    signal: np.ndarray,
    times: np.ndarray,
    *,
    zero_fill_factor: int = 2,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a centered FFT spectrum for a uniformly sampled complex FID."""

    signal = np.asarray(signal, dtype=np.complex128).reshape(-1)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if signal.size != times.size:
        raise ValueError("signal and times must have the same length")
    if signal.size < 2:
        raise ValueError("at least two time samples are required")
    dt = float(times[1] - times[0])
    if dt <= 0 or not np.allclose(np.diff(times), dt):
        raise ValueError("times must be uniformly spaced and increasing")
    zero_fill_factor = int(zero_fill_factor)
    if zero_fill_factor <= 0:
        raise ValueError("zero_fill_factor must be positive")
    if window == "hann":
        weighted = signal * np.hanning(signal.size)
    elif window in {"none", "rectangular"}:
        weighted = signal
    else:
        raise ValueError("window must be 'hann', 'none', or 'rectangular'")
    nfft = int(2 ** np.ceil(np.log2(signal.size * zero_fill_factor)))
    spectrum = np.fft.fftshift(np.fft.fft(weighted, n=nfft))
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, dt))
    return frequencies, spectrum


def deconvolve_acquisition_window(
    spectrum: np.ndarray,
    frequencies_hz: np.ndarray,
    acquisition_times_seconds: np.ndarray,
    *,
    regularization_strength: float = 1e-2,
) -> WindowDeconvolutionResult:
    """Deconvolve the finite acquisition window with Tikhonov regularization."""

    measured = np.asarray(spectrum, dtype=np.complex128).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64).reshape(-1)
    times = np.asarray(acquisition_times_seconds, dtype=np.float64).reshape(-1)
    if measured.size != frequencies.size:
        raise ValueError("spectrum and frequencies_hz must have the same length")
    if times.size < 2:
        raise ValueError("at least two acquisition samples are required")
    if not np.all(np.diff(times) > 0):
        raise ValueError("acquisition_times_seconds must be increasing")
    strength = float(regularization_strength)
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("regularization_strength must be finite and non-negative")

    response = np.exp(1j * TAU * np.outer(times, frequencies))
    kernel = np.fft.fftshift(np.fft.fft(response, n=measured.size, axis=0), axes=0)
    scale = float(times.size)
    kernel = kernel / scale
    target = measured / scale

    lhs = kernel.conj().T @ kernel
    if strength > 0:
        lhs = lhs + (strength**2) * np.eye(lhs.shape[0], dtype=np.complex128)
    rhs = kernel.conj().T @ target
    deconvolved = np.linalg.solve(lhs, rhs)
    residual = kernel @ deconvolved - target
    return WindowDeconvolutionResult(
        deconvolved_spectrum=deconvolved,
        regularization_strength=strength,
        residual_norm=float(np.linalg.norm(residual)),
        solution_norm=float(np.linalg.norm(deconvolved)),
    )


def efg_line_spectrum(
    distribution: EFGDistribution,
    transition_label: str,
    *,
    carrier_frequency_hz: float | None = None,
    amplitudes: np.ndarray | list[complex] | tuple[complex, ...] | None = None,
    linewidth_hz: float = 100.0,
    points: int = 1024,
    span_hz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smoothed line spectrum for a static EFG distribution."""

    frequencies = _distribution_frequencies(distribution, transition_label)
    if carrier_frequency_hz is None:
        carrier_frequency_hz = float(np.sum(distribution.weights * frequencies))
    offsets = frequencies - float(carrier_frequency_hz)
    if amplitudes is None:
        intensities = distribution.weights
    else:
        amp = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
        if amp.size != offsets.size:
            raise ValueError("amplitudes must match the number of EFG isochromats")
        intensities = np.abs(amp)

    linewidth_hz = float(linewidth_hz)
    points = int(points)
    if linewidth_hz <= 0 or not np.isfinite(linewidth_hz):
        raise ValueError("linewidth_hz must be positive and finite")
    if points < 2:
        raise ValueError("points must be at least two")
    if span_hz is None:
        half_span = float(np.max(np.abs(offsets))) if offsets.size else 0.0
        half_span = max(half_span + 5.0 * linewidth_hz, 5.0 * linewidth_hz)
    else:
        half_span = 0.5 * float(span_hz)
        if half_span <= 0 or not np.isfinite(half_span):
            raise ValueError("span_hz must be positive and finite")

    axis = np.linspace(-half_span, half_span, points)
    spectrum = np.zeros(points, dtype=np.float64)
    for offset, intensity in zip(offsets, intensities):
        spectrum += float(intensity) * np.exp(
            -0.5 * ((axis - float(offset)) / linewidth_hz) ** 2
        )
    return axis, spectrum


def simulate_fid_efg_distribution(
    distribution: EFGDistribution,
    transition_label: str,
    times_seconds: np.ndarray | list[float] | tuple[float, ...],
    *,
    excitation: SelectivePulse,
    carrier_frequency_hz: float | None = None,
    orientations: OrientationInput = "single",
    t2_seconds: float = np.inf,
    zero_fill_factor: int = 2,
    window: str = "hann",
    rephase_action: str = "warn",
    rephase_safety_factor: float = 1.25,
) -> NQRFIDDistributionResult:
    """Simulate a selective-pulse NQR FID from a static EFG distribution."""

    times = np.asarray(times_seconds, dtype=np.float64).reshape(-1)
    if times.size < 2:
        raise ValueError("at least two time samples are required")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times_seconds must be increasing and non-negative")
    if t2_seconds <= 0:
        raise ValueError("t2_seconds must be positive")
    for isochromat in distribution.isochromats:
        _require_spin_one_selective_pulse_site(isochromat.site)
    samples = _as_orientations(orientations)
    check_efg_rephasing(
        _distribution_frequencies(distribution, transition_label),
        max_time_seconds=float(times[-1]),
        safety_factor=rephase_safety_factor,
        action=rephase_action,
    )
    if carrier_frequency_hz is None:
        carrier_frequency_hz = diagonalize_site(
            distribution.isochromats[0].site
        ).transition(transition_label).frequency_hz
    carrier = float(carrier_frequency_hz)

    signal = np.zeros(times.size, dtype=np.complex128)
    frequencies: list[float] = []
    amplitudes: list[complex] = []
    decay = np.exp(-times / t2_seconds) if np.isfinite(t2_seconds) else 1.0

    for isochromat in distribution.isochromats:
        for orientation in samples:
            eigensystem = diagonalize_site(isochromat.site)
            transition = eigensystem.transition(transition_label)
            density = equilibrium_density(eigensystem.levels_hz)
            pulse = SelectivePulse(
                excitation.transition_label,
                duration_seconds=excitation.duration_seconds,
                nutation_hz=excitation.nutation_hz,
                phase=excitation.phase,
                rf_frequency_hz=carrier,
            )
            density = apply_selective_pulse(
                density,
                transition,
                pulse,
                b1_direction_pas=orientation.b1_direction_pas,
            )
            amplitude = transition_signal(
                density,
                transition,
                b1_direction_pas=orientation.b1_direction_pas,
            )
            weight = isochromat.weight * orientation.weight
            offset_hz = transition.frequency_hz - carrier
            contribution = weight * amplitude * np.exp(1j * TAU * offset_hz * times)
            signal = signal + contribution * decay
            frequencies.append(transition.frequency_hz)
            amplitudes.append(weight * amplitude)

    spectrum_frequencies, spectrum = fid_spectrum(
        signal,
        times,
        zero_fill_factor=zero_fill_factor,
        window=window,
    )
    return NQRFIDDistributionResult(
        times=times,
        signal=signal,
        spectrum_frequencies_hz=spectrum_frequencies,
        spectrum=spectrum,
        carrier_frequency_hz=carrier,
        isochromat_frequencies_hz=np.asarray(frequencies, dtype=np.float64),
        isochromat_amplitudes=np.asarray(amplitudes, dtype=np.complex128),
        distribution=distribution,
    )


def simulate_slse_efg_distribution(
    distribution: EFGDistribution,
    sequence,
    *,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    relaxation=None,
    rephase_action: str = "warn",
    rephase_safety_factor: float = 1.25,
) -> SLSEDistributionResult:
    """Simulate an SLSE echo train summed over a static EFG distribution."""

    local_results: list[SLSEResult] = []
    echo_times: np.ndarray | None = None
    echo_amplitudes: np.ndarray | None = None
    check_efg_rephasing(
        _distribution_frequencies(distribution, sequence.detection.transition_label),
        max_time_seconds=(
            sequence.echo_spacing_seconds * max(sequence.num_echoes, 1)
        ),
        safety_factor=rephase_safety_factor,
        action=rephase_action,
    )
    for isochromat in distribution.isochromats:
        result = simulate_slse(
            isochromat.site,
            sequence,
            orientations=orientations,
            b0_tesla=b0_tesla,
            t2e_seconds=t2e_seconds,
            relaxation=relaxation,
        )
        local_results.append(result)
        echo_times = result.echo_times
        weighted = isochromat.weight * result.echo_amplitudes
        echo_amplitudes = (
            weighted if echo_amplitudes is None else echo_amplitudes + weighted
        )
    if echo_times is None or echo_amplitudes is None:
        raise AssertionError("distribution validation should prevent empty samples")
    return SLSEDistributionResult(
        echo_times=echo_times,
        echo_amplitudes=echo_amplitudes,
        local_results=tuple(local_results),
        distribution=distribution,
    )


def simulate_slse_acquisition_spectrum(
    distribution: EFGDistribution,
    sequence,
    *,
    acquisition_duration_seconds: float,
    acquisition_points: int = 256,
    echo_index: int = -1,
    carrier_frequency_hz: float | None = None,
    orientations: OrientationInput = "powder",
    b0_tesla: float = 0.0,
    t2e_seconds: float = np.inf,
    relaxation=None,
    zero_fill_factor: int = 2,
    spectrum_window: str = "none",
    noise: NoiseSpec | Mapping | float | int | None = None,
    deconvolution_strength: float | None = None,
    rephase_action: str = "warn",
    rephase_safety_factor: float = 1.25,
) -> SLSEAcquisitionSpectrumResult:
    """Simulate the spectrum of one finite-window acquired SLSE echo."""

    acquisition_duration = float(acquisition_duration_seconds)
    if acquisition_duration <= 0 or not np.isfinite(acquisition_duration):
        raise ValueError("acquisition_duration_seconds must be positive and finite")
    if acquisition_duration >= sequence.echo_spacing_seconds:
        raise ValueError(
            "acquisition_duration_seconds must be shorter than echo spacing"
        )
    points = int(acquisition_points)
    if points < 2:
        raise ValueError("acquisition_points must be at least two")
    if deconvolution_strength is not None and spectrum_window not in {
        "none",
        "rectangular",
    }:
        raise ValueError("deconvolution requires a rectangular spectrum window")

    selected = int(echo_index)
    if selected < 0:
        selected = sequence.num_echoes + selected
    if selected < 0 or selected >= sequence.num_echoes:
        raise IndexError("echo_index is out of range for sequence.num_echoes")

    frequencies = _distribution_frequencies(
        distribution,
        sequence.detection.transition_label,
    )
    check_efg_rephasing(
        frequencies,
        max_time_seconds=max(
            sequence.echo_spacing_seconds * (selected + 1),
            acquisition_duration,
        ),
        safety_factor=rephase_safety_factor,
        action=rephase_action,
    )
    echo_train = simulate_slse_efg_distribution(
        distribution,
        sequence,
        orientations=orientations,
        b0_tesla=b0_tesla,
        t2e_seconds=t2e_seconds,
        relaxation=relaxation,
        rephase_action="ignore",
        rephase_safety_factor=rephase_safety_factor,
    )

    if carrier_frequency_hz is None:
        if sequence.detection.rf_frequency_hz is not None:
            carrier_frequency_hz = sequence.detection.rf_frequency_hz
        else:
            carrier_frequency_hz = float(np.sum(distribution.weights * frequencies))
    carrier = float(carrier_frequency_hz)

    amplitudes = np.array(
        [
            isochromat.weight * local.echo_amplitudes[selected]
            for isochromat, local in zip(
                distribution.isochromats,
                echo_train.local_results,
            )
        ],
        dtype=np.complex128,
    )
    offsets = frequencies - carrier
    dt = acquisition_duration / points
    acquisition_times = (np.arange(points, dtype=np.float64) - 0.5 * points) * dt
    clean_signal = np.sum(
        amplitudes[:, np.newaxis]
        * np.exp(1j * TAU * offsets[:, np.newaxis] * acquisition_times[np.newaxis, :]),
        axis=0,
    )
    echo_signal, noise_metadata = add_received_noise(
        clean_signal,
        _time_domain_noise(noise),
    )
    spectrum_frequencies, clean_spectrum = fid_spectrum(
        clean_signal,
        acquisition_times,
        zero_fill_factor=zero_fill_factor,
        window=spectrum_window,
    )
    noisy_frequencies, spectrum = fid_spectrum(
        echo_signal,
        acquisition_times,
        zero_fill_factor=zero_fill_factor,
        window=spectrum_window,
    )
    if not np.allclose(noisy_frequencies, spectrum_frequencies):
        raise AssertionError("clean and noisy FFT axes should match")

    deconvolution = None
    if deconvolution_strength is not None:
        deconvolution = deconvolve_acquisition_window(
            spectrum,
            spectrum_frequencies,
            acquisition_times,
            regularization_strength=deconvolution_strength,
        )

    return SLSEAcquisitionSpectrumResult(
        acquisition_times_seconds=acquisition_times,
        clean_echo_signal=clean_signal,
        echo_signal=echo_signal,
        spectrum_frequencies_hz=spectrum_frequencies,
        clean_spectrum=clean_spectrum,
        spectrum=spectrum,
        carrier_frequency_hz=carrier,
        isochromat_frequencies_hz=frequencies,
        isochromat_amplitudes=amplitudes,
        selected_echo_index=selected,
        echo_train=echo_train,
        noise_metadata=noise_metadata,
        deconvolution=deconvolution,
    )
