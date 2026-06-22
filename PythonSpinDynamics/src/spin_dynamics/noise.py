"""Opt-in received-signal noise helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoiseSpec:
    """Configuration for additive received-signal noise.

    `sigma` is the standard deviation of each real quadrature. Complex noise
    therefore has expected power `2 * sigma**2` per sample.
    """

    model: str = "white"
    sigma: float | None = None
    target_snr: float | None = None
    seed: int | None = None
    rng: np.random.Generator | None = None
    complex_noise: bool = True
    scale: float = 1.0
    domain: str = "spectrum"


@dataclass(frozen=True)
class NoiseMetadata:
    """Summary of the generated noise realization."""

    model: str
    sigma: float | None
    target_snr: float | None
    seed: int | None
    complex_noise: bool
    scale: float
    domain: str
    signal_rms: float
    noise_rms: float
    realized_snr: float


@dataclass(frozen=True)
class MatchedFilterSNRResult:
    """Matched-filter SNR estimate from clean and noisy spectra."""

    predicted_snr: float | None
    measured_snr: float
    clean_response: complex
    predicted_noise_rms: float | None
    measured_noise_rms: float
    matched_filter: np.ndarray
    noisy_responses: np.ndarray


def as_noise_spec(noise: NoiseSpec | Mapping[str, Any] | float | int | None) -> NoiseSpec | None:
    """Normalize public noise inputs to a validated `NoiseSpec`."""

    if noise is None:
        return None
    if isinstance(noise, NoiseSpec):
        spec = noise
    elif isinstance(noise, Mapping):
        spec = NoiseSpec(**noise)
    else:
        spec = NoiseSpec(sigma=float(noise))
    _validate_noise_spec(spec)
    return spec


def estimate_matched_filter_snr(
    clean_signal: np.ndarray,
    noisy_signals: np.ndarray,
    *,
    pnoise: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    offsets: np.ndarray | None = None,
    noise_scale: float = 1.0,
    matched_filter: np.ndarray | None = None,
) -> MatchedFilterSNRResult:
    """Estimate matched-filter SNR from repeated noisy spectra.

    The last axis is interpreted as the frequency/offset axis. If `pnoise` is
    supplied and `matched_filter` is omitted, the optimal filter convention
    used by the probe SNR routines is applied: `conj(clean_signal) / pnoise`.
    The filter is normalized over `offsets`, matching the MATLAB-style SNR
    calculations in the probe modules.
    """

    clean = np.asarray(clean_signal, dtype=np.complex128).reshape(-1)
    noisy = np.asarray(noisy_signals, dtype=np.complex128)
    if noisy.shape[-1] != clean.size:
        raise ValueError("noisy_signals last dimension must match clean_signal")
    if noise_scale < 0 or not np.isfinite(noise_scale):
        raise ValueError("noise_scale must be finite and non-negative")

    if matched_filter is None:
        if pnoise is None:
            mf = np.conj(clean)
        else:
            density = np.asarray(pnoise, dtype=np.float64).reshape(-1)
            if density.size != clean.size:
                raise ValueError("pnoise must match clean_signal")
            if np.any(~np.isfinite(density)) or np.any(density <= 0):
                raise ValueError("pnoise must contain finite positive values")
            mf = np.conj(clean) / density
    else:
        mf = np.asarray(matched_filter, dtype=np.complex128).reshape(-1)
        if mf.size != clean.size:
            raise ValueError("matched_filter must match clean_signal")

    offset_axis = _axis_or_index(offsets, clean.size, "offsets")
    norm = np.sqrt(np.real(_integrate(np.abs(mf) ** 2, offset_axis, axis=-1)))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("matched filter norm must be finite and positive")
    mf = mf / norm

    clean_response = complex(_integrate(clean * mf, offset_axis, axis=-1))
    noisy_responses = _integrate(noisy * mf, offset_axis, axis=-1)
    noise_responses = noisy_responses - clean_response
    measured_noise_rms = float(np.sqrt(np.mean(np.abs(noise_responses) ** 2)))
    measured_snr = np.inf if measured_noise_rms == 0 else float(
        np.real(clean_response) / measured_noise_rms
    )

    predicted_noise_rms = None
    predicted_snr = None
    if pnoise is not None and frequencies is not None:
        density = np.asarray(pnoise, dtype=np.float64).reshape(-1)
        if density.size != clean.size:
            raise ValueError("pnoise must match clean_signal")
        freq_axis = _axis_or_index(frequencies, clean.size, "frequencies")
        predicted_noise_rms = float(
            np.sqrt(np.real(_integrate(density * float(noise_scale) * np.abs(mf) ** 2, freq_axis)))
        )
        predicted_snr = (
            np.inf
            if predicted_noise_rms == 0
            else float(np.real(clean_response) / predicted_noise_rms)
        )

    return MatchedFilterSNRResult(
        predicted_snr=predicted_snr,
        measured_snr=measured_snr,
        clean_response=clean_response,
        predicted_noise_rms=predicted_noise_rms,
        measured_noise_rms=measured_noise_rms,
        matched_filter=mf,
        noisy_responses=np.asarray(noisy_responses),
    )


def add_received_noise(
    signal: np.ndarray,
    noise: NoiseSpec | Mapping[str, Any] | float | int | None,
    *,
    pnoise: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    sample_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, NoiseMetadata | None]:
    """Return `signal` with additive noise plus generation metadata."""

    spec = as_noise_spec(noise)
    clean = np.asarray(signal, dtype=np.complex128)
    if spec is None:
        return clean.copy(), None
    if spec.model == "white":
        return _add_white_noise(clean, spec)
    if spec.model == "probe":
        if pnoise is None or frequencies is None:
            raise ValueError("probe noise requires pnoise and frequencies")
        return _add_probe_noise(clean, spec, pnoise, frequencies, sample_axis)
    raise ValueError("noise model must be 'white' or 'probe'")


def ideal_noise_density(
    signal: np.ndarray,
    noise: NoiseSpec | Mapping[str, Any] | float | int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a flat output-referred density matching a white-noise spec."""

    spec = as_noise_spec(noise)
    if spec is None:
        raise ValueError("noise must not be None")
    sigma = _resolve_white_sigma(np.asarray(signal, dtype=np.complex128), spec)
    pnoise = np.full(np.asarray(signal).shape[-1], 2 * sigma**2, dtype=np.float64)
    frequencies = np.arange(pnoise.size, dtype=np.float64)
    return pnoise, frequencies


def tuned_probe_output_noise_density(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tuned-probe output-referred noise density and frequencies."""

    k = float(_field(sp, "k"))
    T = float(_field(sp, "T"))
    L = float(_field(sp, "L"))
    R = float(_field(sp, "R"))
    C = float(_field(sp, "C"))
    Cin = float(_field(sp, "Cin"))
    Rin = float(_field(sp, "Rin"))
    Rd = float(_field(sp, "Rd"))
    vn = float(_field(sp, "vn"))
    inn = float(_field(sp, "in_"))
    w0 = float(_field(sp, "w0"))
    del_w = _as_vector(_field(sp, "del_w"))
    w1_max = (np.pi / 2) / float(_field(pp, "T_90"))
    s = 1j * (w0 + del_w * w1_max)
    f = np.imag(s) / (2 * np.pi)
    Yin = s * Cin + 1 / Rin
    Yp = s * C + 1 / Rd + 1 / (s * L + R)
    tf = 1 / (1 + (s * L + R) * (s * C + 1 / Rd + Yin))
    Zs = 1 / (Yin + Yp)
    vni2 = 4 * k * T * R * np.abs(tf) ** 2
    pnoise = vn**2 + inn**2 * np.abs(Zs) ** 2 + vni2
    return np.asarray(pnoise, dtype=np.float64), f


def untuned_probe_output_noise_density(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return untuned-probe output-referred noise density and frequencies."""

    k = float(_field(sp, "k"))
    T = float(_field(sp, "T"))
    L = float(_field(sp, "L"))
    R = float(_field(sp, "R"))
    C = float(_field(sp, "C"))
    Cin = float(_field(sp, "Cin"))
    Rin = float(_field(sp, "Rin"))
    Rd = float(_field(sp, "Rd"))
    Rdup = float(_field(sp, "Rdup"))
    Nrx = float(_field(sp, "Nrx"))
    krx = float(_field(sp, "krx"))
    L1 = float(_field(sp, "L1"))
    R1 = float(_field(sp, "R1"))
    vn = float(_field(sp, "vn"))
    inn = float(_field(sp, "in_"))
    w0 = float(_field(sp, "w0"))
    del_w = _as_vector(_field(sp, "del_w"))
    w1_max = (np.pi / 2) / float(_field(pp, "T_90"))
    s = 1j * (w0 + del_w * w1_max)
    f = np.imag(s) / (2 * np.pi)
    Yin = s * Cin + 1 / Rin + 1 / Rd
    Yp = s * C + 1 / (s * L + R)
    Zp = 1 / Yp
    Nv = krx * Nrx * L1 / (L + L1)
    Zeff = (Zp + Rdup + R1) * Nv**2 + s * L1 * Nrx**2 * (1 - krx**2) + Nrx * R1
    tf = Nv / (1 + Zeff * Yin)
    Zs = Zeff / (1 + Yin * Zeff)
    vni2 = 4 * k * T * R * np.abs(tf) ** 2
    pnoise = (
        vn**2
        + inn**2 * np.abs(Zs) ** 2
        + 4 * k * T * (Rdup + R1 + Nrx * R1 / Nv**2) * np.abs(tf) ** 2
        + vni2
    )
    return np.asarray(pnoise, dtype=np.float64), f


def matched_probe_output_noise_density(
    sp: Mapping[str, Any] | Any,
    pp: Mapping[str, Any] | Any,
    *,
    tf1: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched-probe output-referred noise density and frequencies."""

    k = float(_field(sp, "k"))
    T = float(_field(sp, "T"))
    L = float(_field(sp, "L"))
    f0 = float(_field(sp, "f0"))
    Q = float(_field(sp, "Q"))
    Rc = (2 * np.pi * f0 * L) / Q
    del_w = _as_vector(_field(sp, "del_w"))
    if tf1 is None:
        tf1 = _as_vector(_field(sp, "tf1"))
    else:
        tf1 = np.asarray(tf1, dtype=np.complex128).reshape(-1)
    w1_max = (np.pi / 2) / float(_field(pp, "T_90"))
    f = (2 * np.pi * f0 + del_w * w1_max) / (2 * np.pi)
    vni2 = 4 * k * T * Rc * np.abs(tf1) ** 2
    Fn = 10 ** (float(_field(sp, "NF")) / 10)
    # Amplifier excess-noise term `k*T*Rin*(F-1)`. The apparent "missing" factor
    # of 4 relative to the coil term's open-circuit `4*k*T*Rc` is correct for a
    # matched probe: `tf1` is the loaded transfer to the matched amplifier input
    # (the source EMF carries the `Vs0 = 2*sqrt(Rc/Rs)` normalization), so the
    # matched load sees half the EMF -- a 1/2 voltage / 1/4 power transfer. That
    # turns the open-circuit coil noise `4*k*T*Rc*|tf1|**2` into the delivered
    # node value `k*T*Rin`, i.e. the available-power basis `k*T`. The amplifier
    # excess noise is written directly on that same available-power basis,
    # `k*T*Rin*(F-1)`, and the signal `mrx` is referred through the same `tf1`,
    # so signal and both noise terms are consistent at the matched input node.
    vn2 = k * T * float(_field(sp, "Rin")) * (Fn - 1) * np.ones(f.size)
    return np.asarray(vni2 + vn2, dtype=np.float64), f


def frequency_bin_width(frequencies: np.ndarray) -> float:
    """Estimate a representative frequency-bin width."""

    freq = np.asarray(frequencies, dtype=np.float64).reshape(-1)
    if freq.size < 2:
        return 1.0
    diffs = np.diff(np.sort(freq))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def _add_white_noise(signal: np.ndarray, spec: NoiseSpec) -> tuple[np.ndarray, NoiseMetadata]:
    sigma = _resolve_white_sigma(signal, spec)
    rng = _rng(spec)
    samples = _normal_samples(rng, signal.shape, sigma, spec.complex_noise)
    return signal + samples, _metadata(spec, signal, samples, sigma=sigma, scale=1.0)


def _add_probe_noise(
    signal: np.ndarray,
    spec: NoiseSpec,
    pnoise: np.ndarray,
    frequencies: np.ndarray,
    sample_axis: np.ndarray | None,
) -> tuple[np.ndarray, NoiseMetadata]:
    density = np.asarray(pnoise, dtype=np.float64)
    if np.any(~np.isfinite(density)) or np.any(density < 0):
        raise ValueError("pnoise must contain finite non-negative values")
    df = frequency_bin_width(frequencies)
    dx = frequency_bin_width(_axis_or_index(sample_axis, density.size, "sample_axis"))
    # density [units**2/Hz] * df [Hz] is the per-bin variance. The `/ dx**2`
    # factor rescales that variance onto a user-supplied `sample_axis` grid; it
    # is a no-op in the default case (sample_axis=None -> dx=1). The factor is a
    # convention for mapping the receiver PSD onto an arbitrary sample spacing
    # and is not independently validated for physical (non-unit) sample axes
    # (see docs/python_api/known_gaps.md); pass sample_axis=None for the
    # validated behavior.
    variance = _broadcast_density(density, signal.shape) * df / dx**2
    scale = float(spec.scale)
    if spec.target_snr is not None:
        if spec.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        signal_rms = _rms(signal)
        noise_rms = float(np.sqrt(np.mean(variance)))
        scale = 0.0 if signal_rms == 0 else (signal_rms / spec.target_snr / noise_rms) ** 2
    if scale < 0 or not np.isfinite(scale):
        raise ValueError("scale must be finite and non-negative")
    sigma = np.sqrt(variance * scale / (2.0 if spec.complex_noise else 1.0))
    rng = _rng(spec)
    if spec.complex_noise:
        samples = rng.normal(scale=sigma) + 1j * rng.normal(scale=sigma)
    else:
        samples = rng.normal(scale=sigma).astype(np.complex128)
    return signal + samples, _metadata(spec, signal, samples, sigma=None, scale=scale)


def _validate_noise_spec(spec: NoiseSpec) -> None:
    if spec.model not in {"white", "probe"}:
        raise ValueError("noise model must be 'white' or 'probe'")
    if spec.domain not in {"spectrum", "time"}:
        raise ValueError("noise domain must be 'spectrum' or 'time'")
    if spec.rng is not None and spec.seed is not None:
        raise ValueError("provide either rng or seed, not both")
    if spec.sigma is not None and (spec.sigma < 0 or not np.isfinite(spec.sigma)):
        raise ValueError("sigma must be finite and non-negative")
    if spec.target_snr is not None and (
        spec.target_snr <= 0 or not np.isfinite(spec.target_snr)
    ):
        raise ValueError("target_snr must be finite and positive")
    if spec.model == "white" and spec.sigma is None and spec.target_snr is None:
        raise ValueError("white noise requires sigma or target_snr")
    if spec.scale < 0 or not np.isfinite(spec.scale):
        raise ValueError("scale must be finite and non-negative")


def _resolve_white_sigma(signal: np.ndarray, spec: NoiseSpec) -> float:
    if spec.sigma is not None:
        return float(spec.sigma)
    if spec.target_snr is None:
        raise ValueError("white noise requires sigma or target_snr")
    return _rms(signal) / float(spec.target_snr)


def _normal_samples(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    sigma: float,
    complex_noise: bool,
) -> np.ndarray:
    if complex_noise:
        return rng.normal(scale=sigma, size=shape) + 1j * rng.normal(scale=sigma, size=shape)
    return rng.normal(scale=sigma, size=shape).astype(np.complex128)


def _rng(spec: NoiseSpec) -> np.random.Generator:
    if spec.rng is not None:
        return spec.rng
    return np.random.default_rng(spec.seed)


def _metadata(
    spec: NoiseSpec,
    signal: np.ndarray,
    samples: np.ndarray,
    *,
    sigma: float | None,
    scale: float,
) -> NoiseMetadata:
    signal_rms = _rms(signal)
    noise_rms = _rms(samples)
    realized_snr = np.inf if noise_rms == 0 else signal_rms / noise_rms
    return NoiseMetadata(
        model=spec.model,
        sigma=sigma,
        target_snr=spec.target_snr,
        seed=spec.seed,
        complex_noise=spec.complex_noise,
        scale=scale,
        domain=spec.domain,
        signal_rms=signal_rms,
        noise_rms=noise_rms,
        realized_snr=float(realized_snr),
    )


def _broadcast_density(density: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if density.shape == shape:
        return density
    if density.ndim == 1 and shape and density.size == shape[-1]:
        return np.broadcast_to(density.reshape((1,) * (len(shape) - 1) + (-1,)), shape)
    return np.broadcast_to(density, shape)


def _axis_or_index(axis: np.ndarray | None, size: int, name: str) -> np.ndarray:
    if axis is None:
        return np.arange(size, dtype=np.float64)
    arr = np.asarray(axis, dtype=np.float64).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must match clean_signal")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _integrate(values: np.ndarray, axis_values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    if values.shape[axis] == 1:
        return np.take(values, 0, axis=axis)
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, axis_values, axis=axis)
    return np.trapz(values, axis_values, axis=axis)


def _rms(signal: np.ndarray) -> float:
    arr = np.asarray(signal, dtype=np.complex128)
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(arr) ** 2)))


def _field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj[name]
    return getattr(obj, name)


def _as_vector(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)
