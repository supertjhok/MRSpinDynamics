"""Result and field-map containers for CPMG imaging workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.noise import NoiseMetadata


@dataclass(frozen=True)
class IdealCPMGImagingResult:
    """Ideal-probe CPMG imaging result."""

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    kspace: np.ndarray
    image: np.ndarray
    magnitude: np.ndarray
    gradx: np.ndarray
    gradz: np.ndarray
    del_w: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str
    kspace_noisy: np.ndarray | None = None
    image_noisy: np.ndarray | None = None
    magnitude_noisy: np.ndarray | None = None
    noise: NoiseMetadata | None = None


@dataclass(frozen=True)
class ProbeCPMGImagingResult:
    """Probe-aware CPMG imaging result."""

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    kspace: np.ndarray
    image: np.ndarray
    magnitude: np.ndarray
    gradx: np.ndarray
    gradz: np.ndarray
    del_w: np.ndarray
    echo_integrals: np.ndarray
    sequence_time: np.ndarray
    probe: str
    kspace_noisy: np.ndarray | None = None
    image_noisy: np.ndarray | None = None
    magnitude_noisy: np.ndarray | None = None
    noise: NoiseMetadata | None = None


@dataclass(frozen=True)
class ImagingEchoFitResult:
    """Voxel-wise mono-exponential fit of reconstructed echo magnitudes."""

    rho_map: np.ndarray
    t2_map: np.ndarray
    fitted_magnitude: np.ndarray
    residual_norm: np.ndarray
    mask: np.ndarray
    echo_times: np.ndarray


@dataclass(frozen=True)
class ImagingNoiseStatistics:
    """Repeated-trial image-domain noise summary."""

    clean_image: np.ndarray
    noisy_mean: np.ndarray
    noise_bias: np.ndarray
    noise_std: np.ndarray
    background_noise_rms: float
    signal_mean: float
    snr: float
    num_trials: int
    mode: str
    echo_index: int


IdealPhaseEncodedCPMGImagingResult = IdealCPMGImagingResult
ProbePhaseEncodedCPMGImagingResult = ProbeCPMGImagingResult


@dataclass(frozen=True)
class ImagingFieldMaps:
    """Spatial sample and field maps for CPMG imaging workflows.

    `b0_map` contains normalized off-resonance offsets added to the generated
    isochromat offset axis. `b1_tx_map` and `b1_rx_map` are relative transmit
    and receive sensitivity maps. All maps are two-dimensional and share the
    same shape as `rho`.
    """

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray
    del_wx: np.ndarray
    del_wz: np.ndarray

    def kernel_maps(
        self,
        ny: int,
        maxoffs: float,
        *,
        density_normalization: Literal["legacy", "preserve"] = "legacy",
    ) -> dict[str, np.ndarray]:
        """Return flattened arrays consumed by the arbitrary-pulse kernels.

        `density_normalization="legacy"` matches the MATLAB-parity imaging
        path by assigning each auxiliary offset sample the full voxel density.
        `density_normalization="preserve"` divides density by the number of
        auxiliary samples so the total represented spin density is unchanged.
        """

        if ny <= 0:
            raise ValueError("ny must be positive")
        if density_normalization not in {"legacy", "preserve"}:
            raise ValueError("density_normalization must be 'legacy' or 'preserve'")
        rho = self.rho
        reps = int(ny)
        del_w0y = np.linspace(-float(maxoffs), float(maxoffs), reps)
        b0 = self.b0_map.reshape(-1)
        density_scale = 1.0 if density_normalization == "legacy" else 1.0 / reps
        density = density_scale * rho.reshape(-1)
        return {
            "del_w": np.concatenate([offset + b0 for offset in del_w0y]),
            "del_wx": np.tile(self.del_wx.reshape(-1), reps),
            "del_wz": np.tile(self.del_wz.reshape(-1), reps),
            "w_1": np.tile(self.b1_tx_map.reshape(-1), reps),
            "w_1r": np.tile(self.b1_rx_map.reshape(-1), reps),
            "m0": np.tile(density, reps),
            "mth": np.tile(density, reps),
            "T1": np.tile(self.t1_map.reshape(-1), reps),
            "T2": np.tile(self.t2_map.reshape(-1), reps),
        }
