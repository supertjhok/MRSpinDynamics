"""q-space pore imaging from diffusion-diffraction responses.

In the short-gradient-pulse, long-diffusion-time limit the restricted-diffusion
echo samples the pore form factor. A complex, phase-preserving response can be
inverted directly. A conventional diffusion-diffraction magnitude response
contains only ``|F(q)|`` or ``|F(q)|^2``; its direct inverse is the pore
autocorrelation, while a pore image requires an additional phase-retrieval
constraint such as finite support and non-negativity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.motion import Boundary, MotionFieldMaps2D, Velocity
from spin_dynamics.workflows.pgse import run_pgse_walkers


QSpaceDataKind = Literal["complex", "magnitude", "intensity"]


@dataclass(frozen=True)
class QSpaceReconstructionResult:
    """Image-domain result reconstructed from a centered q-space grid."""

    image: np.ndarray
    x_axis: np.ndarray
    z_axis: np.ndarray
    qx_axis: np.ndarray
    qz_axis: np.ndarray
    data_kind: str
    iterations: int = 0
    residual: float = 0.0

    @property
    def magnitude(self) -> np.ndarray:
        """Return ``abs(image)`` for display-oriented callers."""

        return np.abs(self.image)


@dataclass(frozen=True)
class QSpacePhaseRetrievalResult:
    """Constrained pore-shape estimate from magnitude-only q-space samples."""

    density: np.ndarray
    x_axis: np.ndarray
    z_axis: np.ndarray
    qx_axis: np.ndarray
    qz_axis: np.ndarray
    support: np.ndarray
    iterations: int
    residual_history: np.ndarray
    sample_mask: np.ndarray | None = None

    @property
    def residual(self) -> float:
        """Return the final relative Fourier-magnitude residual."""

        if self.residual_history.size == 0:
            return 0.0
        return float(self.residual_history[-1])

    @property
    def coverage_fraction(self) -> float:
        """Return the fraction of q-space samples constrained by measurement."""

        if self.sample_mask is None:
            return 1.0
        return float(np.mean(self.sample_mask))


@dataclass(frozen=True)
class QSpaceShapeMetrics:
    """Shift/reflection-invariant quality metrics for a reconstructed pore."""

    correlation: float
    intersection_over_union: float
    area_ratio: float
    shift_pixels: tuple[int, int]
    reflected_x: bool
    reflected_z: bool


@dataclass(frozen=True)
class PGSEQSpaceWalkerResult:
    """Finite-pulse PGSE response sampled on a centered two-dimensional q grid."""

    response: np.ndarray
    qx_axis: np.ndarray
    qz_axis: np.ndarray
    zero_q_signal: complex
    gradient_duration: float
    diffusion_time: float
    diffusion_coefficient: float
    gamma: float
    walkers_per_cell: int
    seed: int | None

    @property
    def intensity(self) -> np.ndarray:
        """Return the real, non-negative long-time pore-intensity estimate.

        Restricted PGSE measures the characteristic function of displacement.
        In the long-diffusion-time limit it approaches ``|F(q)|^2``. Finite
        walker sampling can leave small imaginary or negative components, which
        are removed here for magnitude-only reconstruction.
        """

        return np.maximum(np.real(self.response), 0.0)


def acquire_pgse_qspace_walkers(
    rho: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    qx_axis: np.ndarray,
    qz_axis: np.ndarray,
    *,
    gradient_duration: float = 0.5e-3,
    diffusion_time: float = 20.0e-3,
    diffusion_coefficient: float = 2.3e-9,
    gamma: float = 2.675e8,
    walkers_per_cell: int = 32,
    seed: int | None = None,
    jitter: bool = True,
    excitation_duration: float = 100.0e-6,
    refocusing_duration: float = 200.0e-6,
    t1_seconds: float = np.inf,
    t2_seconds: float = np.inf,
    velocity: Velocity = None,
    fields: MotionFieldMaps2D | None = None,
    boundary: Boundary = "reflect",
    substeps_per_interval: int = 8,
) -> PGSEQSpaceWalkerResult:
    """Acquire a finite-pulse restricted-diffusion response on a q-space grid.

    The angular q-vector convention is ``q = gamma * G * delta`` in rad/m.
    Each grid point is run through :func:`run_pgse_walkers`; a shared seed gives
    every point the same initial ensemble and Brownian trajectory, so changes
    across q reflect encoding rather than independent Monte Carlo noise. The
    response is normalized by the explicitly acquired zero-q echo.

    At long diffusion time this response approaches the pore intensity
    ``|F(q)|^2``. For finite pulses and finite diffusion time it retains the
    physically realistic edge averaging and incomplete-mixing blur.
    """

    rho_arr = _density2d(rho)
    x = _uniform_axis(x_axis, "x_axis")
    z = _uniform_axis(z_axis, "z_axis")
    if rho_arr.shape != (x.size, z.size):
        raise ValueError("rho shape must match (len(x_axis), len(z_axis))")
    qx = _uniform_axis(qx_axis, "qx_axis")
    qz = _uniform_axis(qz_axis, "qz_axis")
    delta = float(gradient_duration)
    gamma_value = float(gamma)
    if delta <= 0.0:
        raise ValueError("gradient_duration must be positive")
    if gamma_value <= 0.0:
        raise ValueError("gamma must be positive")

    zero_x = np.flatnonzero(np.isclose(qx, 0.0, rtol=0.0, atol=1e-12))
    zero_z = np.flatnonzero(np.isclose(qz, 0.0, rtol=0.0, atol=1e-12))
    if zero_x.size != 1 or zero_z.size != 1:
        raise ValueError("qx_axis and qz_axis must each contain exactly one zero sample")

    response = np.empty((qx.size, qz.size), dtype=np.complex128)
    common = dict(
        rho=rho_arr,
        x_axis=x,
        z_axis=z,
        gradient_duration=delta,
        diffusion_time=float(diffusion_time),
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=gamma_value,
        walkers_per_cell=int(walkers_per_cell),
        seed=seed,
        jitter=bool(jitter),
        excitation_duration=float(excitation_duration),
        refocusing_duration=float(refocusing_duration),
        t1_seconds=float(t1_seconds),
        t2_seconds=float(t2_seconds),
        velocity=velocity,
        fields=fields,
        boundary=boundary,
        substeps_per_interval=int(substeps_per_interval),
    )
    for ix, qx_value in enumerate(qx):
        for iz, qz_value in enumerate(qz):
            q_vector = np.array([qx_value, qz_value], dtype=np.float64)
            q_norm = float(np.linalg.norm(q_vector))
            if q_norm <= np.finfo(float).eps:
                amplitude = 0.0
                direction: str | tuple[float, float] = "x"
            else:
                amplitude = q_norm / (gamma_value * delta)
                direction = (float(q_vector[0] / q_norm), float(q_vector[1] / q_norm))
            result = run_pgse_walkers(
                gradient_amplitude=amplitude,
                gradient_axis=direction,
                **common,
            )
            response[ix, iz] = result.signal[0]

    zero_signal = complex(response[int(zero_x[0]), int(zero_z[0])])
    if abs(zero_signal) <= np.finfo(float).eps:
        raise ValueError("zero-q PGSE echo is zero and cannot normalize the response")
    response = response / zero_signal
    return PGSEQSpaceWalkerResult(
        response=response,
        qx_axis=qx,
        qz_axis=qz,
        zero_q_signal=zero_signal,
        gradient_duration=delta,
        diffusion_time=float(diffusion_time),
        diffusion_coefficient=float(diffusion_coefficient),
        gamma=gamma_value,
        walkers_per_cell=int(walkers_per_cell),
        seed=seed,
    )


def qspace_axes_from_real_space(
    x_axis: np.ndarray,
    z_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered angular q axes compatible with a real-space grid.

    The returned axes are angular wavevectors in rad/m, i.e. the PGSE
    ``q_ang = gamma G delta`` convention. The input axes must be uniformly
    spaced voxel centers.
    """

    x = _uniform_axis(x_axis, "x_axis")
    z = _uniform_axis(z_axis, "z_axis")
    dx = _spacing(x)
    dz = _spacing(z)
    qx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(x.size, d=dx))
    qz = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(z.size, d=dz))
    return qx.astype(np.float64), qz.astype(np.float64)


def real_space_axes_from_qspace(
    qx_axis: np.ndarray,
    qz_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return centered real-space axes for a uniformly sampled q-space grid."""

    qx = _uniform_axis(qx_axis, "qx_axis")
    qz = _uniform_axis(qz_axis, "qz_axis")
    dx = 2.0 * np.pi / (qx.size * _spacing(qx))
    dz = 2.0 * np.pi / (qz.size * _spacing(qz))
    x = (np.arange(qx.size, dtype=np.float64) - qx.size // 2) * dx
    z = (np.arange(qz.size, dtype=np.float64) - qz.size // 2) * dz
    return x, z


def pore_form_factor_from_density(
    density: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Return the centered complex pore form factor of a 2D density map.

    ``density`` is interpreted on the centered FFT grid used by the imaging
    workflows. With ``normalize=True`` the zero-q sample is one for non-empty
    positive densities, matching normalized q-space echo attenuation.
    """

    rho = _density2d(density)
    form = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rho)))
    if normalize:
        total = np.sum(rho)
        if abs(total) > np.finfo(float).eps:
            form = form / total
    return form.astype(np.complex128, copy=False)


def qspace_sampling_mask(
    qx_axis: np.ndarray,
    qz_axis: np.ndarray,
    *,
    qmax_fraction: float = 1.0,
    missing_fraction: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Build a reproducible radial-window and random-dropout sampling mask.

    qmax_fraction is the retained elliptical radius relative to the largest
    sampled absolute qx and qz values. missing_fraction then removes that
    fraction of otherwise retained samples at random. The zero-q sample is
    always preserved because it supplies the intensity normalization.
    """

    qx = _uniform_axis(qx_axis, "qx_axis")
    qz = _uniform_axis(qz_axis, "qz_axis")
    fraction = float(qmax_fraction)
    missing = float(missing_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("qmax_fraction must be in (0, 1]")
    if not (0.0 <= missing < 1.0):
        raise ValueError("missing_fraction must be in [0, 1)")

    if fraction == 1.0:
        mask = np.ones((qx.size, qz.size), dtype=bool)
    else:
        qx_scale = max(float(np.max(np.abs(qx))), np.finfo(float).eps)
        qz_scale = max(float(np.max(np.abs(qz))), np.finfo(float).eps)
        qxx, qzz = np.meshgrid(qx / qx_scale, qz / qz_scale, indexing="ij")
        mask = qxx**2 + qzz**2 <= fraction**2 + 10.0 * np.finfo(float).eps

    zero_x = np.flatnonzero(np.isclose(qx, 0.0, rtol=0.0, atol=1e-12))
    zero_z = np.flatnonzero(np.isclose(qz, 0.0, rtol=0.0, atol=1e-12))
    if zero_x.size != 1 or zero_z.size != 1:
        raise ValueError("qx_axis and qz_axis must each contain exactly one zero sample")
    zero = (int(zero_x[0]), int(zero_z[0]))

    if missing > 0.0:
        candidates = np.flatnonzero(mask.ravel())
        zero_flat = np.ravel_multi_index(zero, mask.shape)
        candidates = candidates[candidates != zero_flat]
        count = min(int(np.floor(missing * candidates.size)), max(candidates.size - 1, 0))
        if count:
            rng = np.random.default_rng(seed)
            dropped = rng.choice(candidates, size=count, replace=False)
            mask.ravel()[dropped] = False
    mask[zero] = True
    return mask


def add_qspace_intensity_noise(
    intensity: np.ndarray,
    *,
    snr: float,
    seed: int | None = None,
    sample_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Add Gaussian intensity noise and return the clipped data and sigma.

    SNR is peak intensity divided by the Gaussian standard deviation. Noise is
    added only at measured samples when sample_mask is provided; unmeasured
    values are returned as zero and ignored by mask-aware phase retrieval.
    """

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("intensity must be a 2D array with at least 2x2 samples")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("intensity must contain finite non-negative values")
    snr_value = float(snr)
    if np.isnan(snr_value) or snr_value <= 0.0:
        raise ValueError("snr must be positive")
    measured = (
        np.ones(values.shape, dtype=bool)
        if sample_mask is None
        else _support(sample_mask, values.shape, name="sample_mask")
    )
    if np.isinf(snr_value):
        sigma = 0.0
        noisy = values.copy()
    else:
        sigma = float(np.max(values[measured])) / snr_value
        rng = np.random.default_rng(seed)
        noisy = values.copy()
        noisy[measured] += rng.normal(scale=sigma, size=int(np.count_nonzero(measured)))
    noisy = np.maximum(noisy, 0.0)
    noisy[~measured] = 0.0
    return noisy, sigma


def threshold_qspace_intensity(
    intensity: np.ndarray,
    *,
    noise_sigma: float,
    threshold_sigma: float = 2.0,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Suppress a known additive intensity-noise floor before phase retrieval.

    Samples below ``threshold_sigma * noise_sigma`` are set to zero. The gate
    is intentionally explicit: it trades weak high-q diffraction features for
    resistance to the positive bias produced when noisy intensities are clipped
    at zero. Unmeasured samples remain zero when a sampling mask is supplied.
    """

    values = np.asarray(intensity, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("intensity must be a 2D array with at least 2x2 samples")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("intensity must contain finite non-negative values")
    sigma = float(noise_sigma)
    factor = float(threshold_sigma)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("noise_sigma must be finite and non-negative")
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("threshold_sigma must be finite and non-negative")
    measured = (
        np.ones(values.shape, dtype=bool)
        if sample_mask is None
        else _support(sample_mask, values.shape, name="sample_mask")
    )
    gated = np.where(measured & (values >= factor * sigma), values, 0.0)
    return gated


def qspace_shape_metrics(
    estimate: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = 0.2,
) -> QSpaceShapeMetrics:
    """Compare pore shapes modulo translation and axis reflections.

    Both arrays are normalized by their maxima. The best alignment maximizes
    continuous Pearson correlation; intersection-over-union and area ratio use
    the same relative threshold. This explicitly accounts for the unavoidable
    translation and inversion ambiguities of magnitude-only phase retrieval.
    """

    recovered = _density2d(estimate)
    truth = _density2d(reference)
    if recovered.shape != truth.shape:
        raise ValueError("estimate and reference must have the same shape")
    level = float(threshold)
    if not (0.0 < level < 1.0):
        raise ValueError("threshold must be in (0, 1)")
    recovered_scale = float(np.max(recovered))
    truth_scale = float(np.max(truth))
    if recovered_scale <= 0.0 or truth_scale <= 0.0:
        raise ValueError("estimate and reference must each contain a positive value")
    recovered = recovered / recovered_scale
    truth = truth / truth_scale

    best: QSpaceShapeMetrics | None = None
    for reflected_x, reflected_z in ((False, False), (True, False), (False, True), (True, True)):
        candidate = recovered
        if reflected_x:
            candidate = candidate[::-1, :]
        if reflected_z:
            candidate = candidate[:, ::-1]
        cross = np.fft.ifft2(
            np.fft.fft2(candidate) * np.conj(np.fft.fft2(truth))
        ).real
        peak = np.unravel_index(int(np.argmax(cross)), cross.shape)
        shift = tuple(
            -index if index <= size // 2 else size - index
            for index, size in zip(peak, truth.shape, strict=True)
        )
        aligned = np.roll(candidate, shift, axis=(0, 1))
        if np.std(aligned) <= np.finfo(float).eps or np.std(truth) <= np.finfo(float).eps:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(aligned.ravel(), truth.ravel())[0, 1])
        estimate_mask = aligned >= level
        truth_mask = truth >= level
        union = int(np.count_nonzero(estimate_mask | truth_mask))
        intersection = int(np.count_nonzero(estimate_mask & truth_mask))
        iou = float(intersection / union) if union else 0.0
        truth_area = max(int(np.count_nonzero(truth_mask)), 1)
        metrics = QSpaceShapeMetrics(
            correlation=correlation,
            intersection_over_union=iou,
            area_ratio=float(np.count_nonzero(estimate_mask) / truth_area),
            shift_pixels=(int(shift[0]), int(shift[1])),
            reflected_x=reflected_x,
            reflected_z=reflected_z,
        )
        if best is None or (metrics.correlation, metrics.intersection_over_union) > (
            best.correlation,
            best.intersection_over_union,
        ):
            best = metrics
    assert best is not None
    return best


def reconstruct_qspace_image(
    response: np.ndarray,
    qx_axis: np.ndarray,
    qz_axis: np.ndarray,
    *,
    data_kind: QSpaceDataKind = "complex",
    clip_negative: bool = False,
    normalize: bool = True,
) -> QSpaceReconstructionResult:
    """Reconstruct an image or autocorrelation from centered q-space samples.

    ``data_kind="complex"`` treats ``response`` as a phase-preserving form
    factor and returns the direct inverse image. ``"intensity"`` treats it as
    ``|F(q)|^2`` and returns the Patterson/autocorrelation image. ``"magnitude"``
    squares the supplied magnitude first, so it also returns an autocorrelation.
    Magnitude-only data do not determine a unique image without phase retrieval;
    use :func:`phase_retrieve_qspace_magnitude` when a support constraint is
    available.
    """

    data = _qspace2d(response, qx_axis, qz_axis)
    if data_kind == "complex":
        spectrum = np.asarray(data, dtype=np.complex128)
    elif data_kind == "magnitude":
        spectrum = np.asarray(data, dtype=np.float64) ** 2
    elif data_kind == "intensity":
        spectrum = np.asarray(data, dtype=np.float64)
    else:
        raise ValueError("data_kind must be 'complex', 'magnitude', or 'intensity'")

    image = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    if data_kind != "complex":
        image = np.real_if_close(image, tol=1000).real
    if clip_negative:
        image = np.maximum(np.real(image), 0.0)
    if normalize:
        scale = np.max(np.abs(image))
        if scale > np.finfo(float).eps:
            image = image / scale
    x, z = real_space_axes_from_qspace(qx_axis, qz_axis)
    return QSpaceReconstructionResult(
        image=image,
        x_axis=x,
        z_axis=z,
        qx_axis=np.asarray(qx_axis, dtype=np.float64),
        qz_axis=np.asarray(qz_axis, dtype=np.float64),
        data_kind=data_kind,
    )


def phase_retrieve_qspace_magnitude(
    magnitude: np.ndarray,
    qx_axis: np.ndarray,
    qz_axis: np.ndarray,
    *,
    support: np.ndarray | None = None,
    iterations: int = 300,
    beta: float = 0.8,
    seed: int | None = None,
    input_is_intensity: bool = False,
    er_iterations: int = 40,
    sample_mask: np.ndarray | None = None,
) -> QSpacePhaseRetrievalResult:
    """Estimate a non-negative pore image from magnitude-only q-space data.

    This uses the standard hybrid-input-output (HIO) projection followed by a
    short error-reduction cleanup. The result is subject to the usual
    magnitude-only ambiguities: translations, inversion, and support-dependent
    local minima. A loose finite ``support`` mask is therefore strongly
    recommended for pore-shape imaging. If ``input_is_intensity=True``,
    ``magnitude`` is interpreted as ``|F(q)|^2`` and square-rooted first.
    ``sample_mask`` marks acquired q-space points. Fourier amplitudes are
    enforced only there; unmeasured coefficients remain free rather than being
    mistaken for measured zeros.
    """

    amp = _qspace2d(magnitude, qx_axis, qz_axis)
    amp = np.asarray(amp, dtype=np.float64)
    if np.any(amp < 0.0):
        raise ValueError("magnitude/intensity samples must be non-negative")
    target = np.sqrt(amp) if input_is_intensity else amp
    if np.max(target) <= 0.0:
        raise ValueError("q-space magnitude must contain a non-zero sample")
    target = target / np.max(target)

    if iterations < 0 or er_iterations < 0:
        raise ValueError("iterations and er_iterations must be non-negative")
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    support_mask = (
        np.ones(target.shape, dtype=bool)
        if support is None
        else _support(support, target.shape, name="support")
    )
    measured = (
        np.ones(target.shape, dtype=bool)
        if sample_mask is None
        else _support(sample_mask, target.shape, name="sample_mask")
    )
    if np.max(target[measured]) <= 0.0:
        raise ValueError("measured q-space samples must contain a non-zero value")
    target = target / np.max(target[measured])
    target = np.where(measured, target, 0.0)

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=target.shape)
    spectrum = target * np.exp(1j * phase)
    estimate = _positive_real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum))))
    estimate *= support_mask
    history: list[float] = []

    for _ in range(int(iterations)):
        projected = _fourier_project(estimate, target, measured)
        valid = support_mask & (projected >= 0.0)
        next_estimate = np.where(valid, projected, estimate - float(beta) * projected)
        estimate = next_estimate
        history.append(_magnitude_residual(estimate, target, measured))

    for _ in range(int(er_iterations)):
        projected = _fourier_project(estimate, target, measured)
        estimate = np.where(support_mask, np.maximum(projected, 0.0), 0.0)
        history.append(_magnitude_residual(estimate, target, measured))

    total = float(np.sum(estimate))
    if total > np.finfo(float).eps:
        estimate = estimate / total
    x, z = real_space_axes_from_qspace(qx_axis, qz_axis)
    return QSpacePhaseRetrievalResult(
        density=estimate.astype(np.float64, copy=False),
        x_axis=x,
        z_axis=z,
        qx_axis=np.asarray(qx_axis, dtype=np.float64),
        qz_axis=np.asarray(qz_axis, dtype=np.float64),
        support=support_mask,
        sample_mask=measured,
        iterations=int(iterations) + int(er_iterations),
        residual_history=np.asarray(history, dtype=np.float64),
    )


def _fourier_project(
    estimate: np.ndarray,
    target: np.ndarray,
    sample_mask: np.ndarray,
) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(estimate)))
    phase = np.ones_like(spectrum)
    nonzero = np.abs(spectrum) > np.finfo(float).eps
    phase[nonzero] = spectrum[nonzero] / np.abs(spectrum[nonzero])
    constrained = spectrum.copy()
    constrained[sample_mask] = target[sample_mask] * phase[sample_mask]
    projected = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(constrained)))
    return _positive_real(projected, clip=False)


def _magnitude_residual(
    estimate: np.ndarray,
    target: np.ndarray,
    sample_mask: np.ndarray,
) -> float:
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(estimate)))
    scale = max(float(np.linalg.norm(target[sample_mask])), np.finfo(float).eps)
    return float(
        np.linalg.norm(np.abs(spectrum[sample_mask]) - target[sample_mask]) / scale
    )


def _positive_real(values: np.ndarray, *, clip: bool = True) -> np.ndarray:
    real = np.real_if_close(values, tol=1000).real
    return np.maximum(real, 0.0) if clip else real


def _density2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) < 2:
        raise ValueError("density must be a 2D array with at least 2x2 samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError("density must contain finite values")
    return arr


def _qspace2d(values: np.ndarray, qx_axis: np.ndarray, qz_axis: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    qx = _uniform_axis(qx_axis, "qx_axis")
    qz = _uniform_axis(qz_axis, "qz_axis")
    if arr.shape != (qx.size, qz.size):
        raise ValueError("q-space data shape must match (len(qx_axis), len(qz_axis))")
    if not np.all(np.isfinite(arr)):
        raise ValueError("q-space data must contain finite values")
    return arr


def _support(
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    name: str = "support",
) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.shape != shape:
        raise ValueError(f"{name} must have the same shape as q-space data")
    if not np.any(arr):
        raise ValueError(f"{name} must contain at least one true pixel")
    return arr


def _uniform_axis(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    diffs = np.diff(arr)
    if np.any(diffs <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return arr


def _spacing(axis: np.ndarray) -> float:
    return float(axis[1] - axis[0])
