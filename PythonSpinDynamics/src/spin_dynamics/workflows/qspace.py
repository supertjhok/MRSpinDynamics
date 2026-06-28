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

    @property
    def residual(self) -> float:
        """Return the final relative Fourier-magnitude residual."""

        if self.residual_history.size == 0:
            return 0.0
        return float(self.residual_history[-1])


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
) -> QSpacePhaseRetrievalResult:
    """Estimate a non-negative pore image from magnitude-only q-space data.

    This uses the standard hybrid-input-output (HIO) projection followed by a
    short error-reduction cleanup. The result is subject to the usual
    magnitude-only ambiguities: translations, inversion, and support-dependent
    local minima. A loose finite ``support`` mask is therefore strongly
    recommended for pore-shape imaging. If ``input_is_intensity=True``,
    ``magnitude`` is interpreted as ``|F(q)|^2`` and square-rooted first.
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
        np.ones(target.shape, dtype=bool) if support is None else _support(support, target.shape)
    )

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=target.shape)
    spectrum = target * np.exp(1j * phase)
    estimate = _positive_real(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum))))
    estimate *= support_mask
    history: list[float] = []

    for _ in range(int(iterations)):
        projected = _fourier_project(estimate, target)
        valid = support_mask & (projected >= 0.0)
        next_estimate = np.where(valid, projected, estimate - float(beta) * projected)
        estimate = next_estimate
        history.append(_magnitude_residual(estimate, target))

    for _ in range(int(er_iterations)):
        projected = _fourier_project(estimate, target)
        estimate = np.where(support_mask, np.maximum(projected, 0.0), 0.0)
        history.append(_magnitude_residual(estimate, target))

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
        iterations=int(iterations) + int(er_iterations),
        residual_history=np.asarray(history, dtype=np.float64),
    )


def _fourier_project(estimate: np.ndarray, target: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(estimate)))
    phase = np.ones_like(spectrum)
    nonzero = np.abs(spectrum) > np.finfo(float).eps
    phase[nonzero] = spectrum[nonzero] / np.abs(spectrum[nonzero])
    projected = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(target * phase)))
    return _positive_real(projected, clip=False)


def _magnitude_residual(estimate: np.ndarray, target: np.ndarray) -> float:
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(estimate)))
    scale = max(float(np.linalg.norm(target)), np.finfo(float).eps)
    return float(np.linalg.norm(np.abs(spectrum) - target) / scale)


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


def _support(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=bool)
    if arr.shape != shape:
        raise ValueError("support must have the same shape as q-space data")
    if not np.any(arr):
        raise ValueError("support must contain at least one true pixel")
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
