"""Compressed-sensing MRI reconstruction and acquisition stopping rules.

The routines use centered, orthonormal Fourier transforms, matching the
phase-encoded imaging convention in :mod:`spin_dynamics.workflows.imaging`.
The reconstruction is dependency-free: FISTA applies an orthonormal 2-D Haar
wavelet proximal operator, while a small fixed set of acquired k-space samples
is held out to estimate image quality without access to a reference image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveCSResult:
    """Progressive compressed-sensing reconstruction and stopping history."""

    image: np.ndarray
    acquired_mask: np.ndarray
    reconstruction_mask: np.ndarray
    validation_mask: np.ndarray
    sampling_fractions: np.ndarray
    validation_nrmse: np.ndarray
    quality: np.ndarray
    image_change: np.ndarray
    stopped_early: bool
    stop_reason: str
    best_index: int


def centered_fft2(image: np.ndarray) -> np.ndarray:
    """Return the centered orthonormal 2-D Fourier transform."""

    arr = np.asarray(image, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("image must be a 2-D array")
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(arr), norm="ortho"))


def centered_ifft2(kspace: np.ndarray) -> np.ndarray:
    """Return the centered orthonormal inverse 2-D Fourier transform."""

    arr = np.asarray(kspace, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("kspace must be a 2-D array")
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(arr), norm="ortho"))


def _haar_levels(shape: tuple[int, int]) -> int:
    levels = 0
    rows, cols = shape
    while rows % 2 == 0 and cols % 2 == 0 and min(rows, cols) >= 2:
        levels += 1
        rows //= 2
        cols //= 2
    if levels == 0:
        raise ValueError("both image dimensions must be even for Haar reconstruction")
    return levels


def _haar_forward(image: np.ndarray, levels: int) -> np.ndarray:
    coeff = np.asarray(image, dtype=np.complex128).copy()
    rows, cols = coeff.shape
    root2 = np.sqrt(2.0)
    for _ in range(levels):
        block = coeff[:rows, :cols].copy()
        row = np.empty_like(block)
        half_cols = cols // 2
        row[:, :half_cols] = (block[:, 0::2] + block[:, 1::2]) / root2
        row[:, half_cols:] = (block[:, 0::2] - block[:, 1::2]) / root2
        out = np.empty_like(block)
        half_rows = rows // 2
        out[:half_rows, :] = (row[0::2, :] + row[1::2, :]) / root2
        out[half_rows:, :] = (row[0::2, :] - row[1::2, :]) / root2
        coeff[:rows, :cols] = out
        rows, cols = half_rows, half_cols
    return coeff


def _haar_inverse(coefficients: np.ndarray, levels: int) -> np.ndarray:
    image = np.asarray(coefficients, dtype=np.complex128).copy()
    full_rows, full_cols = image.shape
    root2 = np.sqrt(2.0)
    rows = full_rows // (2 ** (levels - 1))
    cols = full_cols // (2 ** (levels - 1))
    for _ in range(levels):
        block = image[:rows, :cols].copy()
        half_rows, half_cols = rows // 2, cols // 2
        row = np.empty_like(block)
        row[0::2, :] = (block[:half_rows, :] + block[half_rows:, :]) / root2
        row[1::2, :] = (block[:half_rows, :] - block[half_rows:, :]) / root2
        out = np.empty_like(block)
        out[:, 0::2] = (row[:, :half_cols] + row[:, half_cols:]) / root2
        out[:, 1::2] = (row[:, :half_cols] - row[:, half_cols:]) / root2
        image[:rows, :cols] = out
        rows = min(full_rows, rows * 2)
        cols = min(full_cols, cols * 2)
    return image


def _complex_soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    magnitude = np.abs(values)
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(magnitude, 1e-30))
    return values * scale


def reconstruct_wavelet_fista(
    kspace: np.ndarray,
    mask: np.ndarray,
    *,
    regularization: float = 5.0e-4,
    iterations: int = 80,
    initial: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct undersampled Cartesian MRI with L1-Haar FISTA.

    Minimizes ``0.5 ||M F x - y||_2^2 + regularization ||H x||_1``.
    The coarsest scaling coefficient is not thresholded, preserving image DC.
    """

    data = np.asarray(kspace, dtype=np.complex128)
    sampling = np.asarray(mask, dtype=bool)
    if data.ndim != 2 or sampling.shape != data.shape:
        raise ValueError("kspace and mask must be matching 2-D arrays")
    if not np.any(sampling):
        raise ValueError("mask must select at least one sample")
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    if int(iterations) < 1:
        raise ValueError("iterations must be positive")
    levels = _haar_levels(data.shape)
    x = centered_ifft2(np.where(sampling, data, 0.0))
    if initial is not None:
        x = np.asarray(initial, dtype=np.complex128).copy()
        if x.shape != data.shape:
            raise ValueError("initial must match the kspace shape")
    z = x.copy()
    momentum = 1.0
    for _ in range(int(iterations)):
        residual = np.where(sampling, centered_fft2(z) - data, 0.0)
        candidate = z - centered_ifft2(residual)
        coeff = _haar_forward(candidate, levels)
        dc = coeff[0, 0]
        coeff = _complex_soft_threshold(coeff, float(regularization))
        coeff[0, 0] = dc
        updated = _haar_inverse(coeff, levels)
        next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
        z = updated + ((momentum - 1.0) / next_momentum) * (updated - x)
        x = updated
        momentum = next_momentum
    return x


def variable_density_order(
    shape: tuple[int, int],
    *,
    seed: int = 0,
    density_power: float = 2.0,
) -> np.ndarray:
    """Return center-out variable-density Cartesian sample coordinates."""

    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("shape must contain two dimensions of at least 2")
    if density_power <= 0 or not np.isfinite(density_power):
        raise ValueError("density_power must be positive and finite")
    yy, xx = np.indices(shape, dtype=np.float64)
    cy, cx = (np.asarray(shape, dtype=np.float64) - 1.0) / 2.0
    radius = np.sqrt(((yy - cy) / max(cy, 1.0)) ** 2 + ((xx - cx) / max(cx, 1.0)) ** 2)
    rng = np.random.default_rng(seed)
    score = radius**density_power * rng.uniform(0.65, 1.35, size=shape)
    flat = np.argsort(score, axis=None, kind="stable")
    return np.column_stack(np.unravel_index(flat, shape)).astype(np.int64)


def adaptive_cs_reconstruction(
    full_kspace: np.ndarray,
    *,
    order: np.ndarray | None = None,
    batch_size: int | None = None,
    validation_fraction: float = 0.04,
    min_sampling_fraction: float = 0.20,
    patience: int = 3,
    min_quality_improvement: float = 2.0e-3,
    regularization: float = 5.0e-4,
    iterations: int = 50,
    seed: int = 0,
) -> AdaptiveCSResult:
    """Acquire k-space progressively and stop when held-out quality plateaus.

    Validation samples are acquired once, distributed through the variable-
    density order, and never supplied to the reconstructor. The reported
    quality ``1 / (1 + validation NRMSE)`` is therefore available during a
    real scan and does not use a ground-truth image.
    """

    data = np.asarray(full_kspace, dtype=np.complex128)
    if data.ndim != 2 or not np.all(np.isfinite(data)):
        raise ValueError("full_kspace must be a finite 2-D array")
    total = data.size
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in (0, 0.5)")
    if not 0 < min_sampling_fraction <= 1:
        raise ValueError("min_sampling_fraction must be in (0, 1]")
    if int(patience) < 1:
        raise ValueError("patience must be positive")
    coords = variable_density_order(data.shape, seed=seed) if order is None else np.asarray(order, dtype=np.int64)
    if coords.shape != (total, 2):
        raise ValueError("order must contain one (row, column) pair per sample")
    flat_order = np.ravel_multi_index((coords[:, 0], coords[:, 1]), data.shape)
    if np.unique(flat_order).size != total:
        raise ValueError("order must visit every k-space sample exactly once")

    n_validation = max(4, int(round(validation_fraction * total)))
    # Keep the calibration/DC core in the reconstruction. Holding out k=0
    # makes the inverse problem artificially ill-conditioned and is unlike a
    # practical autocalibration region.
    validation_start = max(1, int(round(0.05 * total)))
    validation_positions = np.linspace(
        validation_start, total - 1, n_validation, dtype=int
    )
    is_validation = np.zeros(total, dtype=bool)
    is_validation[validation_positions] = True
    validation_flat = flat_order[is_validation]
    training_flat = flat_order[~is_validation]
    validation_mask = np.zeros(total, dtype=bool)
    validation_mask[validation_flat] = True
    validation_mask = validation_mask.reshape(data.shape)
    reconstruction_mask = np.zeros(total, dtype=bool)
    batch = int(batch_size or max(8, round(0.04 * total)))
    if batch < 1:
        raise ValueError("batch_size must be positive")

    fractions: list[float] = []
    errors: list[float] = []
    qualities: list[float] = []
    changes: list[float] = []
    images: list[np.ndarray] = []
    previous: np.ndarray | None = None
    best_quality = -np.inf
    significant_quality = -np.inf
    best_index = 0
    stale = 0
    stopped = False
    stop_reason = "all k-space samples acquired"
    denom = max(float(np.linalg.norm(data[validation_mask])), 1e-30)

    for stop in range(batch, training_flat.size + batch, batch):
        reconstruction_mask[training_flat[: min(stop, training_flat.size)]] = True
        train_mask_2d = reconstruction_mask.reshape(data.shape)
        image = reconstruct_wavelet_fista(
            data,
            train_mask_2d,
            regularization=regularization,
            iterations=iterations,
            initial=previous,
        )
        prediction = centered_fft2(image)
        error = float(np.linalg.norm((prediction - data)[validation_mask]) / denom)
        quality = 1.0 / (1.0 + error)
        change = (
            np.inf
            if previous is None
            else float(np.linalg.norm(image - previous) / max(np.linalg.norm(previous), 1e-30))
        )
        acquired_count = int(np.count_nonzero(train_mask_2d)) + n_validation
        fractions.append(min(acquired_count / total, 1.0))
        errors.append(error)
        qualities.append(quality)
        changes.append(change)
        images.append(image)
        previous = image

        if quality > best_quality:
            best_quality = quality
            best_index = len(images) - 1
        if quality > significant_quality + min_quality_improvement:
            significant_quality = quality
            stale = 0
        else:
            stale += 1
        if fractions[-1] >= min_sampling_fraction and stale >= patience:
            stopped = True
            stop_reason = "held-out k-space quality plateaued"
            break
        if stop >= training_flat.size:
            break

    acquired = reconstruction_mask.reshape(data.shape) | validation_mask
    return AdaptiveCSResult(
        image=images[best_index],
        acquired_mask=acquired,
        reconstruction_mask=reconstruction_mask.reshape(data.shape).copy(),
        validation_mask=validation_mask,
        sampling_fractions=np.asarray(fractions),
        validation_nrmse=np.asarray(errors),
        quality=np.asarray(qualities),
        image_change=np.asarray(changes),
        stopped_early=stopped,
        stop_reason=stop_reason,
        best_index=best_index,
    )


def normalized_root_mean_square_error(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Return magnitude-image NRMSE, useful for synthetic validation only."""

    ref = np.abs(np.asarray(reference))
    est = np.abs(np.asarray(estimate))
    if ref.shape != est.shape:
        raise ValueError("reference and estimate must have matching shapes")
    return float(np.linalg.norm(est - ref) / max(np.linalg.norm(ref), 1e-30))


__all__ = [
    "AdaptiveCSResult",
    "adaptive_cs_reconstruction",
    "centered_fft2",
    "centered_ifft2",
    "normalized_root_mean_square_error",
    "reconstruct_wavelet_fista",
    "variable_density_order",
]
