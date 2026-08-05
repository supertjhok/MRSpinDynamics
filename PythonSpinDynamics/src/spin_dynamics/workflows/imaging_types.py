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
class CartesianSENSEResult:
    """Regularized Cartesian SENSE reconstruction and encoding diagnostics."""

    image: np.ndarray
    sampled_kspace: np.ndarray
    zero_filled_channel_image: np.ndarray
    sampling_mask: np.ndarray
    acceleration: int
    axis: int
    offset: int
    regularization: float
    condition_number: np.ndarray
    g_factor: np.ndarray
    rank: np.ndarray


@dataclass(frozen=True)
class ReceiverArrayCPMGImagingResult:
    """Channel-resolved ideal CPMG imaging result for an Rx array.

    ``channel_kspace`` and ``channel_image`` are channel-leading with shape
    ``(n_channels, px, pz, num_echoes)``. The complex sensitivity maps retain
    receive phase. Compatibility properties expose the noise-optimal Roemer
    combination through the established single-image names.
    """

    rho: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    receiver_sensitivities: np.ndarray
    channel_labels: tuple[str, ...]
    channel_kspace: np.ndarray
    channel_image: np.ndarray
    channel_magnitude: np.ndarray
    rss_magnitude: np.ndarray
    sensitivity_combined_kspace: np.ndarray
    sensitivity_combined_image: np.ndarray
    roemer_combined_kspace: np.ndarray
    roemer_combined_image: np.ndarray
    gradx: np.ndarray
    gradz: np.ndarray
    del_w: np.ndarray
    sequence_time: np.ndarray
    probe: str
    noise_covariance: np.ndarray | None = None
    channel_kspace_noisy: np.ndarray | None = None
    channel_image_noisy: np.ndarray | None = None
    channel_magnitude_noisy: np.ndarray | None = None
    rss_magnitude_noisy: np.ndarray | None = None
    sensitivity_combined_kspace_noisy: np.ndarray | None = None
    sensitivity_combined_image_noisy: np.ndarray | None = None
    roemer_combined_kspace_noisy: np.ndarray | None = None
    roemer_combined_image_noisy: np.ndarray | None = None
    sampling_mask: np.ndarray | None = None
    sense_reference_image: np.ndarray | None = None
    sense_sampled_kspace: np.ndarray | None = None
    sense_sampled_kspace_noisy: np.ndarray | None = None
    sense_zero_filled_channel_image: np.ndarray | None = None
    sense_zero_filled_channel_image_noisy: np.ndarray | None = None
    sense_image: np.ndarray | None = None
    sense_image_noisy: np.ndarray | None = None
    sense_condition_number: np.ndarray | None = None
    sense_g_factor: np.ndarray | None = None
    sense_rank: np.ndarray | None = None
    sense_acceleration: int | None = None
    sense_axis: int | None = None
    sense_offset: int | None = None
    sense_regularization: float | None = None
    geometric_receiver_sensitivities: np.ndarray | None = None
    receiver_transfer_matrix: np.ndarray | None = None
    receiver_source_impedance_ohm: np.ndarray | None = None
    receiver_total_impedance_ohm: np.ndarray | None = None
    receiver_output_impedance_ohm: np.ndarray | None = None
    receiver_network_noise_covariance_v2: np.ndarray | None = None
    receiver_network_noise_correlation: np.ndarray | None = None
    receiver_network_frequency_hz: float | None = None

    @property
    def n_channels(self) -> int:
        """Number of receiver channels."""

        return int(self.channel_kspace.shape[0])

    @property
    def b1_rx_map(self) -> np.ndarray:
        """Legacy-shaped root-sum-of-squares receive sensitivity view."""

        return np.sqrt(np.sum(np.abs(self.receiver_sensitivities) ** 2, axis=0))

    @property
    def kspace(self) -> np.ndarray:
        """Compatibility view: noise-optimal combined k-space."""

        return self.roemer_combined_kspace

    @property
    def image(self) -> np.ndarray:
        """Compatibility view: noise-optimal combined complex image."""

        return self.roemer_combined_image

    @property
    def magnitude(self) -> np.ndarray:
        """Compatibility view: magnitude of the noise-optimal combination."""

        return np.abs(self.roemer_combined_image)

    @property
    def echo_integrals(self) -> np.ndarray:
        """Compatibility view of combined echo integrals."""

        return self.roemer_combined_kspace

    @property
    def kspace_noisy(self) -> np.ndarray | None:
        return self.roemer_combined_kspace_noisy

    @property
    def image_noisy(self) -> np.ndarray | None:
        return self.roemer_combined_image_noisy

    @property
    def magnitude_noisy(self) -> np.ndarray | None:
        if self.roemer_combined_image_noisy is None:
            return None
        return np.abs(self.roemer_combined_image_noisy)

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
