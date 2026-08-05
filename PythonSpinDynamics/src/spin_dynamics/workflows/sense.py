"""Cartesian sensitivity-encoding operators and SENSE reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.workflows.imaging_types import CartesianSENSEResult
from spin_dynamics.workflows.receiver_arrays import (
    as_receiver_sensitivities,
    centered_kspace_from_images,
    reconstruct_receiver_images,
    validate_noise_covariance,
)

SenseAxis = Literal[0, 1, "x", "z"]


def _axis_index(axis: SenseAxis) -> int:
    if axis in (0, "x"):
        return 0
    if axis in (1, "z"):
        return 1
    raise ValueError("axis must be 0/'x' or 1/'z'")


def uniform_cartesian_mask(
    spatial_shape: tuple[int, int],
    acceleration: int,
    *,
    axis: SenseAxis = 0,
    offset: int = 0,
) -> np.ndarray:
    """Return a centered, uniformly undersampled Cartesian line mask.

    ``offset=0`` retains the centered k-space line. The encoded dimension must
    be divisible by ``acceleration`` so that every alias set has the same size.
    """

    if len(spatial_shape) != 2 or any(int(size) <= 0 for size in spatial_shape):
        raise ValueError("spatial_shape must contain two positive dimensions")
    if not isinstance(acceleration, (int, np.integer)) or acceleration < 1:
        raise ValueError("acceleration must be a positive integer")
    axis_index = _axis_index(axis)
    size = int(spatial_shape[axis_index])
    if size % int(acceleration) != 0:
        raise ValueError(
            "the encoded spatial dimension must be divisible by acceleration"
        )
    if not isinstance(offset, (int, np.integer)) or not 0 <= offset < acceleration:
        raise ValueError("offset must satisfy 0 <= offset < acceleration")

    centered_index = np.arange(size) - size // 2
    sampled_lines = np.mod(centered_index - int(offset), int(acceleration)) == 0
    shape = tuple(int(value) for value in spatial_shape)
    if axis_index == 0:
        return np.broadcast_to(sampled_lines[:, np.newaxis], shape).copy()
    return np.broadcast_to(sampled_lines[np.newaxis, :], shape).copy()


def _uniform_mask_parameters(
    mask: np.ndarray,
    axis: SenseAxis,
) -> tuple[np.ndarray, int, int, int]:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or mask_array.size == 0:
        raise ValueError("sampling_mask must be a non-empty 2D array")
    axis_index = _axis_index(axis)
    other_axis = 1 - axis_index
    line_mask = np.all(mask_array, axis=other_axis)
    expanded = (
        line_mask[:, np.newaxis]
        if axis_index == 0
        else line_mask[np.newaxis, :]
    )
    if not np.array_equal(mask_array, np.broadcast_to(expanded, mask_array.shape)):
        raise ValueError("sampling_mask must select complete Cartesian lines")

    size = mask_array.shape[axis_index]
    sampled_count = int(np.count_nonzero(line_mask))
    if sampled_count == 0 or size % sampled_count != 0:
        raise ValueError("sampling_mask must contain uniformly spaced lines")
    acceleration = size // sampled_count
    sampled_indices = np.flatnonzero(line_mask)
    centered_index = sampled_indices - size // 2
    offset = int(np.mod(centered_index[0], acceleration))
    expected = uniform_cartesian_mask(
        mask_array.shape,
        acceleration,
        axis=axis_index,
        offset=offset,
    )
    if not np.array_equal(mask_array, expected):
        raise ValueError("sampling_mask must contain uniformly spaced lines")
    return mask_array, axis_index, acceleration, offset


def noise_whitener(
    noise_covariance: np.ndarray,
    n_channels: int,
) -> np.ndarray:
    """Return the Hermitian inverse square root of a channel covariance."""

    covariance = validate_noise_covariance(noise_covariance, n_channels)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    inverse_root = np.zeros_like(eigenvalues)
    positive = eigenvalues > 1e-12 * scale
    inverse_root[positive] = 1.0 / np.sqrt(eigenvalues[positive])
    return (eigenvectors * inverse_root[np.newaxis, :]) @ eigenvectors.conj().T


def whiten_receiver_channels(
    channel_data: np.ndarray,
    noise_covariance: np.ndarray,
) -> np.ndarray:
    """Whiten channel-leading data with a supplied noise covariance."""

    data = np.asarray(channel_data, dtype=np.complex128)
    if data.ndim < 2 or data.shape[0] == 0:
        raise ValueError("channel_data must be non-empty and channel-leading")
    whitener = noise_whitener(noise_covariance, data.shape[0])
    flat = data.reshape(data.shape[0], -1)
    return (whitener @ flat).reshape(data.shape)


@dataclass(frozen=True)
class CartesianSENSEEncoding:
    """Reusable Cartesian ``P F S`` operator for supplied sensitivity maps."""

    receiver_sensitivities: np.ndarray
    sampling_mask: np.ndarray
    axis: SenseAxis = 0
    noise_covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        sensitivities = as_receiver_sensitivities(self.receiver_sensitivities)
        mask, axis_index, _acceleration, _offset = _uniform_mask_parameters(
            self.sampling_mask,
            self.axis,
        )
        if sensitivities.shape[1:] != mask.shape:
            raise ValueError("sampling_mask shape must match sensitivity maps")
        covariance = (
            np.eye(sensitivities.shape[0], dtype=np.complex128)
            if self.noise_covariance is None
            else validate_noise_covariance(
                self.noise_covariance,
                sensitivities.shape[0],
            )
        )
        object.__setattr__(self, "receiver_sensitivities", sensitivities)
        object.__setattr__(self, "sampling_mask", mask)
        object.__setattr__(self, "axis", axis_index)
        object.__setattr__(self, "noise_covariance", covariance)

    @property
    def acceleration(self) -> int:
        """Uniform line-acceleration factor."""

        return _uniform_mask_parameters(self.sampling_mask, self.axis)[2]

    @property
    def offset(self) -> int:
        """Centered sampling offset in ``[0, acceleration)``."""

        return _uniform_mask_parameters(self.sampling_mask, self.axis)[3]

    def forward(self, image: np.ndarray) -> np.ndarray:
        """Apply sensitivity weighting, centered FFT, and line sampling."""

        image_array = np.asarray(image, dtype=np.complex128)
        if image_array.ndim == 2:
            image_array = image_array[..., np.newaxis]
        if image_array.ndim != 3 or image_array.shape[:2] != self.sampling_mask.shape:
            raise ValueError("image must have shape (x, z) or (x, z, num_echoes)")
        channels = (
            self.receiver_sensitivities[..., np.newaxis]
            * image_array[np.newaxis, ...]
        )
        return centered_kspace_from_images(channels) * self.sampling_mask[
            np.newaxis, ..., np.newaxis
        ]

    def adjoint(self, channel_kspace: np.ndarray) -> np.ndarray:
        """Apply the exact Euclidean adjoint of the unnormalized ``P F S``."""

        kspace = np.asarray(channel_kspace, dtype=np.complex128)
        expected = (
            self.receiver_sensitivities.shape[0],
            *self.sampling_mask.shape,
        )
        if kspace.ndim == 3:
            kspace = kspace[..., np.newaxis]
        if kspace.ndim != 4 or kspace.shape[:3] != expected:
            raise ValueError(
                "channel_kspace must have shape "
                "(n_channels, x, z, num_echoes)"
            )
        sampled = kspace * self.sampling_mask[np.newaxis, ..., np.newaxis]
        channel_images = reconstruct_receiver_images(sampled)
        channel_images *= float(np.prod(self.sampling_mask.shape))
        return np.einsum(
            "cxz,cxze->xze",
            self.receiver_sensitivities.conj(),
            channel_images,
            optimize=True,
        )


def _folding_matrix(line_mask: np.ndarray) -> np.ndarray:
    """Return the centered-FFT zero-fill folding operator for one axis."""

    size = line_mask.size
    basis = np.eye(size, dtype=np.complex128)
    kspace = np.fft.fftshift(
        np.fft.fft(np.fft.ifftshift(basis, axes=0), axis=0),
        axes=0,
    )
    kspace *= line_mask[:, np.newaxis]
    return np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(kspace, axes=0), axis=0),
        axes=0,
    )


def reconstruct_cartesian_sense(
    channel_kspace: np.ndarray,
    receiver_sensitivities: np.ndarray,
    sampling_mask: np.ndarray,
    *,
    axis: SenseAxis = 0,
    noise_covariance: np.ndarray | None = None,
    regularization: float = 0.0,
    rank_tolerance: float | None = None,
    raise_on_rank_deficiency: bool = False,
) -> CartesianSENSEResult:
    """Reconstruct uniformly undersampled Cartesian data by alias-set SENSE.

    The solve at each alias set is
    ``(E^H Psi^-1 E + lambda I) m = E^H Psi^-1 d``.
    Conditioning and g-factor maps describe the unregularized whitened
    encoding. Rank-deficient sets are marked with infinite diagnostics.
    """

    kspace = np.asarray(channel_kspace, dtype=np.complex128)
    if kspace.ndim != 4 or kspace.shape[0] == 0:
        raise ValueError(
            "channel_kspace must have shape "
            "(n_channels, x, z, num_echoes)"
        )
    sensitivities = as_receiver_sensitivities(
        receiver_sensitivities,
        kspace.shape[1:3],
    )
    if sensitivities.shape[0] != kspace.shape[0]:
        raise ValueError("channel and sensitivity counts must match")
    mask, axis_index, acceleration, offset = _uniform_mask_parameters(
        sampling_mask,
        axis,
    )
    if mask.shape != kspace.shape[1:3]:
        raise ValueError("sampling_mask shape must match channel k-space")
    regularization_value = float(regularization)
    if not np.isfinite(regularization_value) or regularization_value < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if rank_tolerance is not None and (
        not np.isfinite(rank_tolerance) or rank_tolerance <= 0.0
    ):
        raise ValueError("rank_tolerance must be finite and positive when set")

    covariance = (
        np.eye(kspace.shape[0], dtype=np.complex128)
        if noise_covariance is None
        else validate_noise_covariance(noise_covariance, kspace.shape[0])
    )
    inverse_covariance = np.linalg.pinv(covariance, hermitian=True)
    whitener = noise_whitener(covariance, kspace.shape[0])
    sampled_kspace = kspace * mask[np.newaxis, ..., np.newaxis]
    zero_filled = reconstruct_receiver_images(sampled_kspace)

    line_mask = (
        np.all(mask, axis=1) if axis_index == 0 else np.all(mask, axis=0)
    )
    folding = _folding_matrix(line_mask)
    encoded_size = mask.shape[axis_index]
    folded_size = encoded_size // acceleration
    other_size = mask.shape[1 - axis_index]
    echoes = kspace.shape[3]
    image = np.zeros((*mask.shape, echoes), dtype=np.complex128)
    condition = np.zeros(mask.shape, dtype=np.float64)
    g_factor = np.zeros(mask.shape, dtype=np.float64)
    rank_map = np.zeros(mask.shape, dtype=np.int64)

    for base in range(folded_size):
        positions = base + folded_size * np.arange(acceleration)
        coefficients = folding[base, positions]
        for other in range(other_size):
            if axis_index == 0:
                local_sensitivity = sensitivities[:, positions, other]
                data = zero_filled[:, base, other, :]
            else:
                local_sensitivity = sensitivities[:, other, positions]
                data = zero_filled[:, other, base, :]
            encoding = local_sensitivity * coefficients[np.newaxis, :]
            whitened_encoding = whitener @ encoding
            singular_values = np.linalg.svd(whitened_encoding, compute_uv=False)
            tolerance = (
                rank_tolerance
                if rank_tolerance is not None
                else (
                    max(whitened_encoding.shape)
                    * np.finfo(np.float64).eps
                    * (float(singular_values[0]) if singular_values.size else 0.0)
                )
            )
            rank = int(np.count_nonzero(singular_values > tolerance))
            rank_deficient = rank < acceleration
            condition_value = (
                np.inf
                if rank_deficient
                else float(singular_values[0] / singular_values[-1])
            )
            gram = encoding.conj().T @ inverse_covariance @ encoding
            inverse_gram = np.linalg.pinv(gram, hermitian=True)
            system = gram + regularization_value * np.eye(
                acceleration,
                dtype=np.complex128,
            )
            rhs = encoding.conj().T @ inverse_covariance @ data
            solution = np.linalg.pinv(system, hermitian=True) @ rhs

            full_information = np.real(
                np.einsum(
                    "ci,cd,di->i",
                    local_sensitivity.conj(),
                    inverse_covariance,
                    local_sensitivity,
                    optimize=True,
                )
            )
            g_values = np.sqrt(
                np.maximum(
                    np.real(np.diag(inverse_gram))
                    * full_information
                    / float(acceleration**2),
                    0.0,
                )
            )
            if rank_deficient:
                g_values[:] = np.inf

            if axis_index == 0:
                image[positions, other, :] = solution
                condition[positions, other] = condition_value
                g_factor[positions, other] = g_values
                rank_map[positions, other] = rank
            else:
                image[other, positions, :] = solution
                condition[other, positions] = condition_value
                g_factor[other, positions] = g_values
                rank_map[other, positions] = rank

    if raise_on_rank_deficiency and np.any(rank_map < acceleration):
        deficient = int(np.count_nonzero(rank_map < acceleration))
        raise np.linalg.LinAlgError(
            f"SENSE encoding is rank deficient at {deficient} voxels"
        )

    return CartesianSENSEResult(
        image=image,
        sampled_kspace=sampled_kspace,
        zero_filled_channel_image=zero_filled,
        sampling_mask=mask,
        acceleration=acceleration,
        axis=axis_index,
        offset=offset,
        regularization=regularization_value,
        condition_number=condition,
        g_factor=g_factor,
        rank=rank_map,
    )
