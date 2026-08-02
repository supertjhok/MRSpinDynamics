"""Receiver-array reconstruction, noise, and coil-combination helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def as_receiver_sensitivities(
    sensitivities: Iterable[complex] | np.ndarray,
    spatial_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return finite complex sensitivity maps with shape ``(channel, x, z)``."""

    maps = np.asarray(sensitivities, dtype=np.complex128)
    if maps.ndim != 3 or maps.shape[0] == 0:
        raise ValueError("receiver_sensitivities must have shape (n_channels, x, z)")
    if spatial_shape is not None and maps.shape[1:] != tuple(spatial_shape):
        raise ValueError("receiver_sensitivities spatial shape must match the phantom")
    if not np.all(np.isfinite(maps)):
        raise ValueError("receiver_sensitivities must be finite")
    return maps


def reconstruct_receiver_images(channel_kspace: np.ndarray) -> np.ndarray:
    """Reconstruct centered Cartesian k-space for every channel and echo."""

    kspace = np.asarray(channel_kspace, dtype=np.complex128)
    if kspace.ndim != 4:
        raise ValueError(
            "channel_kspace must have shape (n_channels, px, pz, num_echoes)"
        )
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(1, 2)), axes=(1, 2)),
        axes=(1, 2),
    )


def centered_kspace_from_images(channel_images: np.ndarray) -> np.ndarray:
    """Transform channel-leading image stacks to centered Cartesian k-space."""

    images = np.asarray(channel_images, dtype=np.complex128)
    if images.ndim not in (3, 4):
        raise ValueError(
            "images must have shape (px, pz, num_echoes) or "
            "(n_channels, px, pz, num_echoes)"
        )
    axes = (0, 1) if images.ndim == 3 else (1, 2)
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(images, axes=axes), axes=axes),
        axes=axes,
    )


def sum_of_squares(channel_images: np.ndarray) -> np.ndarray:
    """Return the conventional root-sum-of-squares magnitude image."""

    images = np.asarray(channel_images, dtype=np.complex128)
    if images.ndim != 4:
        raise ValueError(
            "channel_images must have shape (n_channels, px, pz, num_echoes)"
        )
    return np.sqrt(np.sum(np.abs(images) ** 2, axis=0))


def sensitivity_weighted_combine(
    channel_images: np.ndarray,
    receiver_sensitivities: Iterable[complex] | np.ndarray,
) -> np.ndarray:
    """Combine channels with normalized complex sensitivity weighting.

    For channel data ``y_c = s_c m``, this evaluates
    ``sum(conj(s_c) y_c) / sum(abs(s_c)**2)`` voxel by voxel.
    """

    images = np.asarray(channel_images, dtype=np.complex128)
    if images.ndim != 4:
        raise ValueError(
            "channel_images must have shape (n_channels, px, pz, num_echoes)"
        )
    sensitivities = as_receiver_sensitivities(receiver_sensitivities, images.shape[1:3])
    if sensitivities.shape[0] != images.shape[0]:
        raise ValueError("channel image and sensitivity counts must match")
    numerator = np.einsum("cxz,cxze->xze", sensitivities.conj(), images, optimize=True)
    denominator = np.sum(np.abs(sensitivities) ** 2, axis=0)
    return np.divide(
        numerator,
        denominator[..., np.newaxis],
        out=np.zeros_like(numerator),
        where=denominator[..., np.newaxis] > 0.0,
    )


def validate_noise_covariance(
    covariance: Iterable[complex] | np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Validate a Hermitian positive-semidefinite channel covariance matrix."""

    matrix = np.asarray(covariance, dtype=np.complex128)
    if matrix.shape != (n_channels, n_channels):
        raise ValueError(
            f"noise_covariance must have shape ({n_channels}, {n_channels})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("noise_covariance must be finite")
    scale = max(1.0, float(np.max(np.abs(matrix))))
    tolerance = 1e-12 * scale
    if not np.allclose(matrix, matrix.conj().T, atol=tolerance, rtol=1e-12):
        raise ValueError("noise_covariance must be Hermitian")
    matrix = 0.5 * (matrix + matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(np.min(eigenvalues)) < -tolerance:
        raise ValueError("noise_covariance must be positive semidefinite")
    return matrix


def roemer_combine(
    channel_images: np.ndarray,
    receiver_sensitivities: Iterable[complex] | np.ndarray,
    noise_covariance: Iterable[complex] | np.ndarray,
) -> np.ndarray:
    """Apply the noise-optimal Roemer channel combination voxel by voxel."""

    images = np.asarray(channel_images, dtype=np.complex128)
    if images.ndim != 4:
        raise ValueError(
            "channel_images must have shape (n_channels, px, pz, num_echoes)"
        )
    sensitivities = as_receiver_sensitivities(receiver_sensitivities, images.shape[1:3])
    if sensitivities.shape[0] != images.shape[0]:
        raise ValueError("channel image and sensitivity counts must match")
    covariance = validate_noise_covariance(noise_covariance, images.shape[0])
    inverse = np.linalg.pinv(covariance, hermitian=True)
    weighted_sensitivity = np.einsum(
        "cd,dxz->cxz", inverse, sensitivities, optimize=True
    )
    numerator = np.einsum(
        "cxz,cxze->xze", weighted_sensitivity.conj(), images, optimize=True
    )
    denominator = np.real(
        np.einsum(
            "cxz,cxz->xz",
            sensitivities.conj(),
            weighted_sensitivity,
            optimize=True,
        )
    )
    return np.divide(
        numerator,
        denominator[..., np.newaxis],
        out=np.zeros_like(numerator),
        where=denominator[..., np.newaxis] > 0.0,
    )


def receiver_noise_covariance(
    n_channels: int,
    *,
    noise_std: float = 0.0,
    covariance: Iterable[complex] | np.ndarray | None = None,
) -> np.ndarray | None:
    """Resolve independent noise or a supplied absolute channel covariance."""

    std = float(noise_std)
    if not np.isfinite(std) or std < 0.0:
        raise ValueError("noise_std must be finite and non-negative")
    if covariance is not None and std > 0.0:
        raise ValueError("provide either noise_std or noise_covariance, not both")
    if covariance is not None:
        return validate_noise_covariance(covariance, n_channels)
    if std == 0.0:
        return None
    return (std**2) * np.eye(n_channels, dtype=np.complex128)


def add_receiver_array_noise(
    channel_data: np.ndarray,
    noise_covariance: Iterable[complex] | np.ndarray,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Add circular complex Gaussian noise with covariance ``E[n n^H]``."""

    data = np.asarray(channel_data, dtype=np.complex128)
    if data.ndim < 2:
        raise ValueError("channel_data must be channel-leading")
    covariance = validate_noise_covariance(noise_covariance, data.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    factor = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))[np.newaxis, :]
    rng = np.random.default_rng(seed)
    samples = int(np.prod(data.shape[1:]))
    standard = (
        rng.standard_normal((data.shape[0], samples))
        + 1j * rng.standard_normal((data.shape[0], samples))
    ) / np.sqrt(2.0)
    noise = (factor @ standard).reshape(data.shape)
    return data + noise
