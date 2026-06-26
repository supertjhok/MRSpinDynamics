"""Result and field-map containers for CPMG imaging workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from spin_dynamics.fields import SpatialDomain, SpatialFieldMaps
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

        This delegates to the dimension-agnostic `SpatialFieldMaps.flatten`,
        passing the stored `del_wx`/`del_wz` gradient sensitivities through
        verbatim and requesting the legacy axis-key names, so the returned dict
        is identical to the previous in-line implementation.
        """

        domain = SpatialDomain.normalized(self.rho.shape)
        spatial = SpatialFieldMaps(
            domain=domain,
            rho=self.rho,
            t1_map=self.t1_map,
            t2_map=self.t2_map,
            b0_map=self.b0_map,
            b1_tx_map=self.b1_tx_map,
            b1_rx_map=self.b1_rx_map,
            gradient_sensitivity=(self.del_wx, self.del_wz),
        )
        return spatial.flatten(
            ny,
            maxoffs,
            density_normalization,
            axis_names=("del_wx", "del_wz"),
        )
