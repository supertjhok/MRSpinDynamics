"""End-to-end low-cost Halbach MRI acquisition model.

The defaults reproduce the operating scale reported for the C8 ferrite system
in *Portable Low-Field MRI Scanners*: a 4.158 MHz proton frequency, 0.88 T/m
intrinsic field gradient, 5 us RF pulses, 100 ms scan spacing, 18.5 W average
RF dissipation, -0.2 %/K ferrite coefficient, and measured SNR=84 as an
independent validation target.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.analysis.compressed_sensing import (
    AdaptiveCSResult,
    adaptive_cs_reconstruction,
    centered_ifft2,
    normalized_root_mean_square_error,
    reconstruct_tv_pocs,
    variable_density_order,
)
from spin_dynamics.core.numerics import trapezoid
from spin_dynamics.fields.coil_peec import coil_properties_peec, helical_solenoid
from spin_dynamics.fields.magnetostatics import (
    MU0,
    FiniteMagnetRod,
    biot_savart,
    circular_loop,
    sample_halbach_dipole_field,
)
from spin_dynamics.optimal_control import TunedProbeResponse
from spin_dynamics.thermal import ThermalLink, ThermalNetwork, ThermalNode


@dataclass(frozen=True)
class PortableHalbachMRIConfig:
    """Book-calibrated system, thermal, receiver, and CS parameters."""

    matrix_size: int = 64
    field_of_view_m: float = 10.0e-3
    nominal_b0_t: float = 0.09765
    resonance_frequency_hz: float = 4.158e6
    intrinsic_gradient_t_per_m: float = 0.88
    rf_pulse_duration_s: float = 5.0e-6
    repetition_time_s: float = 0.100
    scans_per_kspace_point: int = 2
    rf_average_power_w: float = 18.5
    gradient_average_power_w: float = 0.43
    gradient_pulse_duration_s: float = 1.6e-3
    gradient_coil_resistance_ohm: float = 0.20
    gradient_current_limit_a: float = 5.0
    ferrite_temperature_coefficient_per_k: float = -2.0e-3
    receiver_bandwidth_hz: float = 200.0e3
    transmit_probe_bandwidth_hz: float = 200.0e3
    measured_single_scan_snr: float = 84.0
    acquisition_window_s: float = 18.0e-6
    receiver_noise_figure_db: float = 3.0
    sample_temperature_k: float = 293.15
    sample_depth_m: float = 10.0e-3
    proton_number_density_per_m3: float = 6.69e28
    ferrite_eddy_power_w: float = 0.0
    ferrite_imaginary_relative_permeability: float | None = None
    measured_receive_coil_q: float = 128.0
    transmit_coil_inductance_h: float = 3.2e-6
    measured_transmit_coil_q: float = 75.0
    pcmcd_reference_peak_power_w: float = 200.0
    pcmcd_reference_pulse_s: float = 5.0e-6
    pcmcd_short_pulse_efficiency: float = 0.27
    gradient_coil_inductance_h: float = 1.3e-6
    gradient_rise_time_s: float = 100.0e-6
    adc_full_scale_v: float = 2.5
    adc_peak_fraction: float = 0.5
    ambient_temperature_k: float = 293.15
    regularization: float = 1.0e-6
    reconstruction_iterations: int = 120
    tv_regularization: float = 0.08
    tv_iterations: int = 10
    batch_fraction: float = 0.04
    minimum_sampling_fraction: float = 0.32
    stopping_patience: int = 2
    minimum_quality_improvement: float = 2.0e-2
    seed: int = 7


@dataclass(frozen=True)
class PortableHalbachMRIResult:
    """Fields, thermal trajectory, noisy acquisition, and adaptive image."""

    config: PortableHalbachMRIConfig
    phantom: np.ndarray
    spin_density: np.ndarray
    b0_map_t: np.ndarray
    b1_map: np.ndarray
    b1_transmit_map_t_per_a: np.ndarray
    b1_receive_map_t_per_a: np.ndarray
    gx_field_map_t_per_a: np.ndarray
    gz_field_map_t_per_a: np.ndarray
    gx_current_a: np.ndarray
    gz_current_a: np.ndarray
    gradient_coil_average_power_w: float
    receive_coil_inductance_h: float
    receive_coil_resistance_ohm: float
    receive_coil_q_factor: float
    receive_coil_copper_q_factor: float
    ferrite_rf_loss_resistance_ohm: float
    ferrite_imaginary_relative_permeability: float
    predicted_single_scan_snr: float
    measured_reference_snr: float
    water_signal_voltage_v: float
    single_scan_noise_rms_v: float
    ideal_kspace: np.ndarray
    measured_kspace: np.ndarray
    acquisition_order: np.ndarray
    acquisition_times_s: np.ndarray
    coil_temperature_k: np.ndarray
    magnet_temperature_k: np.ndarray
    larmor_drift_hz: np.ndarray
    receiver_gain: np.ndarray
    noise_standard_deviation: float
    adaptive: AdaptiveCSResult
    reference_nrmse: float
    zero_fill_reference_nrmse: float

    @property
    def stopped_time_s(self) -> float:
        """Approximate elapsed scan time at the selected stopping point."""

        fraction = float(self.adaptive.sampling_fractions[-1])
        return fraction * float(self.acquisition_times_s[-1])


@dataclass(frozen=True)
class RFPulseLengthSweep:
    """RF power, sensitivity, and active-volume trade-off versus 90-degree pulse."""

    pulse_lengths_s: np.ndarray
    peak_current_a: np.ndarray
    peak_delivered_coil_current_a: np.ndarray
    peak_forward_power_w: np.ndarray
    peak_dc_input_power_w: np.ndarray
    peak_coil_loss_w: np.ndarray
    predicted_snr: np.ndarray
    active_sample_volume_m3: np.ndarray
    effective_slice_thickness_m: np.ndarray


@dataclass(frozen=True)
class RFCoilDesignMetrics:
    """Electrical and field metrics for the transmit and receive coils."""

    transmit_inductance_h: float
    transmit_ac_resistance_ohm: float
    transmit_q_factor: float
    transmit_loaded_probe_q_factor: float
    transmit_probe_bandwidth_hz: float
    transmit_b1_center_t_per_a: float
    receive_inductance_h: float
    receive_copper_resistance_ohm: float
    receive_ferrite_loss_resistance_ohm: float
    receive_loaded_resistance_ohm: float
    receive_copper_q_factor: float
    receive_loaded_q_factor: float
    receive_loaded_probe_q_factor: float
    receive_probe_bandwidth_hz: float
    receive_b1_center_t_per_a: float


@dataclass(frozen=True)
class GradientCoilDesignMetrics:
    """Gradient efficiency, electrical load, and peak driver requirements."""

    gx_efficiency_t_per_m_per_a: float
    gz_efficiency_t_per_m_per_a: float
    inductance_h: float
    resistance_ohm: float
    peak_current_a: float
    peak_voltage_v: float
    peak_resistive_power_w: float
    average_winding_power_w: float
    rise_time_s: float


@dataclass(frozen=True)
class ReceiverDesignMetrics:
    """Receiver signal/noise levels and required ADC gain."""

    peak_probe_signal_v: float
    single_scan_noise_rms_v: float
    predicted_single_scan_snr: float
    adc_full_scale_v: float
    target_adc_peak_v: float
    required_voltage_gain: float
    required_gain_db: float


@dataclass(frozen=True)
class SystemWeightMetrics:
    """Book Table 10.12 mass budget."""

    magnet_kg: float
    transmitter_kg: float
    other_electronics_kg: float
    batteries_kg: float
    baseplate_kg: float
    total_kg: float
    portable_without_baseplate_kg: float


@dataclass(frozen=True)
class PortableHalbachDesignSummary:
    """Designer-facing capstone metrics derived from one end-to-end run."""

    rf_coils: RFCoilDesignMetrics
    pulse_sweep: RFPulseLengthSweep
    gradients: GradientCoilDesignMetrics
    receiver: ReceiverDesignMetrics
    weight: SystemWeightMetrics


def portable_phantom(matrix_size: int = 64) -> np.ndarray:
    """Return a sparse, asymmetric low-field resolution phantom."""

    if int(matrix_size) < 16:
        raise ValueError("matrix_size must be at least 16")
    axis = np.linspace(-1.0, 1.0, int(matrix_size))
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros_like(xx)
    body = (xx / 0.82) ** 2 + (yy / 0.72) ** 2 <= 1.0
    image[body] = 0.28
    for x0, y0, radius, intensity in (
        (-0.38, -0.30, 0.16, 1.00),
        (0.02, -0.32, 0.11, 0.82),
        (0.36, -0.27, 0.08, 0.67),
        (-0.28, 0.14, 0.10, 0.78),
        (0.06, 0.14, 0.15, 0.92),
        (0.39, 0.20, 0.12, 0.58),
    ):
        image[(xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2] = intensity
    image[(np.abs(xx + 0.02) < 0.055) & (yy > 0.34) & (yy < 0.62)] = 0.88
    image[(np.abs(yy - 0.49) < 0.055) & (xx > -0.18) & (xx < 0.14)] = 0.88
    return image


def _solenoid_segments(
    radius: float, length: float, turns: int, *, points_per_turn: int = 48
) -> list[tuple[np.ndarray, np.ndarray]]:
    segments: list[tuple[np.ndarray, np.ndarray]] = []
    positions = np.linspace(-0.5 * length, 0.5 * length, turns)
    for y_pos in positions:
        segments.extend(
            circular_loop(
                (0.0, float(y_pos), 0.0),
                radius,
                axis="y",
                n_segments=points_per_turn,
            )
        )
    return segments


def _saddle_segments(
    radius: float,
    length: float,
    turns: int,
    *,
    subtended_angle: float = 2.0 * np.pi / 3.0,
    points_per_arc: int = 20,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the book's paired 120-degree saddle transmit winding."""

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for center_index, center in enumerate((0.0, np.pi)):
        angles = np.linspace(
            center - 0.5 * subtended_angle,
            center + 0.5 * subtended_angle,
            points_per_arc + 1,
        )
        if center_index == 1:
            angles = angles[::-1]
        for _ in range(turns):
            first = np.array([radius * np.cos(angles[0]), -0.5 * length, radius * np.sin(angles[0])])
            first_top = first.copy()
            first_top[1] *= -1.0
            segments.append((first, first_top))
            top = np.column_stack(
                [
                    radius * np.cos(angles),
                    np.full_like(angles, 0.5 * length),
                    radius * np.sin(angles),
                ]
            )
            segments.extend((top[i], top[i + 1]) for i in range(points_per_arc))
            last_top = top[-1]
            last = last_top.copy()
            last[1] *= -1.0
            segments.append((last_top, last))
            bottom = np.column_stack(
                [
                    radius * np.cos(angles[::-1]),
                    np.full_like(angles, -0.5 * length),
                    radius * np.sin(angles[::-1]),
                ]
            )
            segments.extend((bottom[i], bottom[i + 1]) for i in range(points_per_arc))
    return segments


def _four_wire_gradient_segments(
    radius: float, length: float, turns: int, angle: float
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the alternating-current four-wire gradient of book Eq. 7.17."""

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(turns):
        for index in range(4):
            phi = angle + index * np.pi / 2.0
            start = np.array(
                [radius * np.cos(phi), -0.5 * length, radius * np.sin(phi)]
            )
            stop = start.copy()
            stop[1] *= -1.0
            segments.append((start, stop) if index % 2 == 0 else (stop, start))
    return segments


def _field_maps(
    config: PortableHalbachMRIConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n = int(config.matrix_size)
    half = 0.5 * float(config.field_of_view_m)
    axis = np.linspace(-half, half, n)
    thin_axis = np.array([-0.1e-3, 0.1e-3])
    rods = []
    for phi in np.arange(8, dtype=np.float64) * np.pi / 4.0:
        rods.append(
            FiniteMagnetRod(
                center=(21.7e-3 * np.cos(phi), 21.7e-3 * np.sin(phi), 0.0),
                length=100.0e-3,
                br=(0.385 * np.cos(2.0 * phi), 0.385 * np.sin(2.0 * phi), 0.0),
                shape="square",
                width=10.0e-3,
            )
        )
    maps = sample_halbach_dipole_field(
        axis,
        axis,
        thin_axis,
        rods=rods,
        n_cross=3,
        n_length=11,
    )
    raw = maps.b0_magnitude[:, :, 0]
    raw = raw * (config.nominal_b0_t / float(np.mean(raw)))
    # The measured 0.88 T/m value is a diffusion-weighting gradient magnitude,
    # not a signed linear imaging gradient across the full bore. Keep it as a
    # sequence/diffusion parameter; the imaging offsets come from the actual
    # eight-block field map so the probe sees the reported few-thousand-ppm span.
    b0 = raw

    points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1)
    points3 = np.zeros((n, n, 3), dtype=np.float64)
    points3[..., 0] = points[..., 0]
    points3[..., 2] = points[..., 1]
    # C8 coil assembly #1 (book Table 4.4): Rx 6.3 mm x 24 mm x 30-turn
    # solenoid; Tx 7.7 mm x 50 mm x 4-turn, 120-degree saddle pair; and
    # two-turn four-wire gradients at radius 10.3 mm and length 83 mm.
    receive = _solenoid_segments(6.3e-3, 24.0e-3, 30)
    transmit = _saddle_segments(7.7e-3, 50.0e-3, 4)
    gx_coil = _four_wire_gradient_segments(10.3e-3, 83.0e-3, 2, 0.0)
    gz_coil = _four_wire_gradient_segments(10.3e-3, 83.0e-3, 2, np.pi / 4.0)
    receive_vector = biot_savart(points3, receive, current=1.0)
    transmit_vector = biot_savart(points3, transmit, current=1.0)
    # Book coordinates place B0 along z, so only x/y RF field is effective.
    b1_receive = np.linalg.norm(receive_vector[..., :2], axis=-1)
    b1_transmit = np.linalg.norm(transmit_vector[..., :2], axis=-1)
    gx_field = biot_savart(points3, gx_coil, current=1.0)[..., 2]
    gz_field = biot_savart(points3, gz_coil, current=1.0)[..., 2]
    b1 = b1_receive * b1_transmit
    b1 /= max(float(np.max(b1)), 1e-30)
    return b0, b1, b1_transmit, b1_receive, gx_field, gz_field


def _ferrite_rf_loss(
    config: PortableHalbachMRIConfig,
    receive_segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    inductance_h: float,
    copper_resistance_ohm: float,
) -> tuple[float, float]:
    """Return ferrite magnetic-loss resistance and the effective mu-r double-prime.

    The RF field is integrated through the eight C8 blocks from Table 3.2. If
    ``ferrite_imaginary_relative_permeability`` is omitted, the value is inferred
    from the independently reported in-magnet receive-coil Q=128. Supplying a
    measured complex-permeability value makes the calculation fully predictive.
    """

    samples = 4
    offsets = (np.arange(samples, dtype=np.float64) + 0.5) / samples - 0.5
    points: list[np.ndarray] = []
    for phi in np.arange(8, dtype=np.float64) * np.pi / 4.0:
        center_x = 21.7e-3 * np.cos(phi)
        center_z = 21.7e-3 * np.sin(phi)
        xx, yy, zz = np.meshgrid(
            center_x + 10.0e-3 * offsets,
            100.0e-3 * offsets,
            center_z + 10.0e-3 * offsets,
            indexing="ij",
        )
        points.append(np.stack([xx, yy, zz], axis=-1).reshape(-1, 3))
    field = biot_savart(np.concatenate(points), receive_segments, current=1.0)
    h_squared = np.sum(field**2, axis=1) / MU0**2
    voxel_volume = (10.0e-3 / samples) ** 2 * (100.0e-3 / samples)
    h_energy_integral = float(np.sum(h_squared) * voxel_volume)
    omega = 2.0 * np.pi * config.resonance_frequency_hz
    resistance_per_mu_double_prime = omega * MU0 * h_energy_integral
    if config.ferrite_imaginary_relative_permeability is None:
        target_resistance = max(
            0.0,
            omega * inductance_h / config.measured_receive_coil_q
            - copper_resistance_ohm,
        )
        mu_double_prime = target_resistance / max(
            resistance_per_mu_double_prime, 1e-30
        )
    else:
        mu_double_prime = float(config.ferrite_imaginary_relative_permeability)
        if mu_double_prime < 0.0 or not np.isfinite(mu_double_prime):
            raise ValueError(
                "ferrite_imaginary_relative_permeability must be finite and non-negative"
            )
    return resistance_per_mu_double_prime * mu_double_prime, mu_double_prime


def _thermal_trajectory(
    config: PortableHalbachMRIConfig,
    times: np.ndarray,
    gradient_coil_power_w: float,
) -> tuple[np.ndarray, np.ndarray]:
    ambient = float(config.ambient_temperature_k)
    network = ThermalNetwork(
        [
            ThermalNode("coil", heat_capacity=45.0, initial_temperature=ambient),
            ThermalNode("magnet", heat_capacity=480.0, initial_temperature=ambient),
            ThermalNode("ambient", heat_capacity=None, initial_temperature=ambient),
        ],
        [
            ThermalLink("coil", "magnet", conductance=1.25),
            ThermalLink("coil", "ambient", conductance=0.38),
            ThermalLink("magnet", "ambient", conductance=0.20),
        ],
        sources={
            "coil": float(config.rf_average_power_w + gradient_coil_power_w),
            # Ferrite is deliberately assigned no gradient eddy-current loss by
            # default; its low conductivity makes that channel negligible.
            "magnet": float(config.ferrite_eddy_power_w),
        },
    )
    result = network.transient(times, max_step=2.0)
    return result.temperatures["coil"], result.temperatures["magnet"]


def simulate_portable_halbach_mri(
    config: PortableHalbachMRIConfig | None = None,
) -> PortableHalbachMRIResult:
    """Simulate thermal drift, noisy acquisition, CS, and automatic stopping."""

    cfg = PortableHalbachMRIConfig() if config is None else config
    n = int(cfg.matrix_size)
    if n < 16 or n % 2:
        raise ValueError("matrix_size must be an even integer of at least 16")
    if cfg.measured_single_scan_snr <= 0:
        raise ValueError("measured_single_scan_snr must be positive")
    phantom = portable_phantom(n)
    b0, b1, b1_tx, b1_rx, gx_field, gz_field = _field_maps(cfg)
    offset_hz = (b0 - cfg.nominal_b0_t) * (
        cfg.resonance_frequency_hz / cfg.nominal_b0_t
    )
    transmit_probe_q = (
        cfg.resonance_frequency_hz / cfg.transmit_probe_bandwidth_hz
    )
    receive_probe_q = cfg.resonance_frequency_hz / cfg.receiver_bandwidth_hz
    transmit_response = TunedProbeResponse.from_quality_factor(
        f0_hz=cfg.resonance_frequency_hz,
        quality_factor=transmit_probe_q,
    )
    receive_response = TunedProbeResponse.from_quality_factor(
        f0_hz=cfg.resonance_frequency_hz,
        quality_factor=receive_probe_q,
    )
    transmit_filter = transmit_response.transfer(offset_hz)
    receive_filter = receive_response.transfer(offset_hz)
    acquisition_window = np.sinc(offset_hz * cfg.acquisition_window_s)
    excitation = (
        np.sinc(offset_hz * cfg.rf_pulse_duration_s) * transmit_filter
    )
    receive_profile = acquisition_window * receive_filter
    spin_density = phantom * b1**2 * excitation * receive_profile
    axis = np.linspace(-0.5 * cfg.field_of_view_m, 0.5 * cfg.field_of_view_m, n)
    center = n // 2
    gx_gradient = np.gradient(gx_field, axis, axis)[0]
    gz_gradient = np.gradient(gz_field, axis, axis)[1]
    gx_per_amp = float(gx_gradient[center, center])
    gz_per_amp = float(gz_gradient[center, center])
    if abs(gx_per_amp) < 1e-12 or abs(gz_per_amp) < 1e-12:
        raise RuntimeError("gradient coil model produced a zero central gradient")
    x_encoding = (gx_field - gx_field[center, center]) / gx_per_amp
    z_encoding = (gz_field - gz_field[center, center]) / gz_per_amp
    k_axis = np.fft.fftshift(np.fft.fftfreq(n, d=cfg.field_of_view_m / n))
    ideal_kspace = np.empty((n, n), dtype=np.complex128)
    normalization = np.sqrt(float(n * n))
    for row, kx in enumerate(k_axis):
        for col, kz in enumerate(k_axis):
            phase = np.exp(-2.0j * np.pi * (kx * x_encoding + kz * z_encoding))
            ideal_kspace[row, col] = np.sum(spin_density * phase) / normalization

    # Predict the receive-coil impedance rather than using the measured Q as an
    # input. AWG-24 has a nominal copper radius of 0.2555 mm. A modest full-PEEC
    # mesh resolves skin and turn-to-turn proximity loss at 4.158 MHz.
    receive_conductor = helical_solenoid(
        diameter=12.6e-3,
        length=24.0e-3,
        turns=30,
        wire_radius=0.2555e-3,
        n_per_turn=4,
        n_radial=2,
        n_angular=4,
        temperature=cfg.sample_temperature_k,
        axis="y",
    )
    receive_properties = coil_properties_peec(
        receive_conductor,
        cfg.resonance_frequency_hz,
        formulation="full",
    )
    copper_q_receive = receive_properties.q_factor
    receive_segments = _solenoid_segments(6.3e-3, 24.0e-3, 30)
    ferrite_resistance, ferrite_mu_double_prime = _ferrite_rf_loss(
        cfg,
        receive_segments,
        inductance_h=receive_properties.inductance,
        copper_resistance_ohm=receive_properties.total_resistance,
    )
    resistance_receive = receive_properties.total_resistance + ferrite_resistance
    q_receive = (
        2.0
        * np.pi
        * cfg.resonance_frequency_hz
        * receive_properties.inductance
        / resistance_receive
    )

    # Reciprocity: V = omega * integral(M_perp * B1/I dV). The water-cylinder
    # reference predicts the same single-echo/scan SNR quoted in the book;
    # importantly, the quoted value is not used to set the simulated noise.
    boltzmann = 1.380649e-23
    hbar = 1.054571817e-34
    gamma_rad = 2.0 * np.pi * cfg.resonance_frequency_hz / cfg.nominal_b0_t
    magnetization = (
        cfg.proton_number_density_per_m3
        * gamma_rad**2
        * hbar**2
        * cfg.nominal_b0_t
        / (4.0 * boltzmann * cfg.sample_temperature_k)
    )
    xx, zz = np.meshgrid(axis, axis, indexing="ij")
    water_mask = xx**2 + zz**2 <= (4.5e-3) ** 2
    transmit_normalized = b1_tx / max(float(np.max(b1_tx)), 1e-30)
    voxel_volume = (cfg.field_of_view_m / n) ** 2 * cfg.sample_depth_m
    omega = 2.0 * np.pi * cfg.resonance_frequency_hz
    water_flux = abs(
        magnetization
        * voxel_volume
        * np.sum(
            water_mask
            * excitation
            * transmit_normalized
            * b1_rx
            * receive_profile
        )
    )
    water_signal_v = omega * water_flux * receive_probe_q
    noise_frequencies = np.linspace(
        -0.5 * cfg.receiver_bandwidth_hz,
        0.5 * cfg.receiver_bandwidth_hz,
        4097,
    )
    noise_filter = receive_response.transfer(noise_frequencies) * np.sinc(
        noise_frequencies * cfg.acquisition_window_s
    )
    noise_bandwidth = 0.5 * float(
        trapezoid(np.abs(noise_filter) ** 2, noise_frequencies)
    )
    noise_factor = 10.0 ** (cfg.receiver_noise_figure_db / 10.0)
    output_noise_density = receive_probe_q * np.sqrt(
        4.0
        * boltzmann
        * cfg.sample_temperature_k
        * resistance_receive
        * noise_factor
    )
    single_scan_noise_rms = float(output_noise_density * np.sqrt(noise_bandwidth))
    predicted_snr = float(water_signal_v / single_scan_noise_rms)

    phantom_flux = abs(
        magnetization
        * voxel_volume
        * np.sum(
            phantom
            * excitation
            * transmit_normalized
            * b1_rx
            * receive_profile
        )
    )
    phantom_signal_v = omega * phantom_flux * receive_probe_q
    dc = ideal_kspace[n // 2, n // 2]
    if abs(dc) > 0.0:
        ideal_kspace *= phantom_signal_v / abs(dc)

    order = variable_density_order((n, n), seed=cfg.seed)
    gamma_hz_per_t = cfg.resonance_frequency_hz / cfg.nominal_b0_t
    gx_current_grid, gz_current_grid = np.meshgrid(
        k_axis / (gamma_hz_per_t * gx_per_amp * cfg.gradient_pulse_duration_s),
        k_axis / (gamma_hz_per_t * gz_per_amp * cfg.gradient_pulse_duration_s),
        indexing="ij",
    )
    peak_current = max(float(np.max(np.abs(gx_current_grid))), float(np.max(np.abs(gz_current_grid))))
    if peak_current > cfg.gradient_current_limit_a * 1.01:
        raise ValueError(
            f"phase encoding requires {peak_current:.2f} A, above the configured "
            f"{cfg.gradient_current_limit_a:.2f} A gradient limit"
        )
    gradient_duty = cfg.gradient_pulse_duration_s / cfg.repetition_time_s
    gradient_coil_power = float(
        cfg.gradient_coil_resistance_ohm
        * np.mean(gx_current_grid**2 + gz_current_grid**2)
        * gradient_duty
    )
    dwell = cfg.repetition_time_s * int(cfg.scans_per_kspace_point)
    acquisition_times = np.arange(n * n, dtype=np.float64) * dwell
    thermal_times = np.concatenate(([0.0], acquisition_times[1:]))
    coil_temperature, magnet_temperature = _thermal_trajectory(
        cfg, thermal_times, gradient_coil_power
    )
    delta_magnet = magnet_temperature - cfg.ambient_temperature_k
    drift = (
        cfg.resonance_frequency_hz
        * cfg.ferrite_temperature_coefficient_per_k
        * delta_magnet
    )
    quality_factor = cfg.resonance_frequency_hz / cfg.receiver_bandwidth_hz
    response = 1.0 / (1.0 + 2.0j * quality_factor * drift / cfg.resonance_frequency_hz)

    measured = np.zeros_like(ideal_kspace)
    # Two quadrature components share the predicted complex RMS noise variance;
    # repeated real/imag scans reduce it by sqrt(scans_per_kspace_point).
    complex_sigma = single_scan_noise_rms / np.sqrt(int(cfg.scans_per_kspace_point))
    rng = np.random.default_rng(cfg.seed + 1)
    noise = complex_sigma / np.sqrt(2.0) * (
        rng.standard_normal((n, n)) + 1.0j * rng.standard_normal((n, n))
    )
    for index, (row, col) in enumerate(order):
        measured[row, col] = ideal_kspace[row, col] * response[index] + noise[row, col]

    adaptive = adaptive_cs_reconstruction(
        measured,
        order=order,
        batch_size=max(8, int(round(cfg.batch_fraction * n * n))),
        min_sampling_fraction=cfg.minimum_sampling_fraction,
        patience=cfg.stopping_patience,
        min_quality_improvement=cfg.minimum_quality_improvement,
        regularization=cfg.regularization
        * max(float(np.max(np.abs(measured))), 1e-30),
        iterations=cfg.reconstruction_iterations,
        seed=cfg.seed,
    )
    adaptive = replace(
        adaptive,
        image=reconstruct_tv_pocs(
            measured,
            adaptive.acquired_mask,
            regularization=cfg.tv_regularization,
            iterations=cfg.tv_iterations,
            initial=adaptive.image,
        ),
    )
    reference_magnitude = np.abs(spin_density)
    reconstructed_magnitude = np.abs(adaptive.image)
    display_scale = float(
        np.sum(reference_magnitude * reconstructed_magnitude)
        / max(np.sum(reconstructed_magnitude**2), 1e-30)
    )
    zero_fill = centered_ifft2(
        np.where(adaptive.acquired_mask, measured, 0.0)
    )
    zero_fill_magnitude = np.abs(zero_fill)
    zero_fill_scale = float(
        np.sum(reference_magnitude * zero_fill_magnitude)
        / max(np.sum(zero_fill_magnitude**2), 1e-30)
    )
    return PortableHalbachMRIResult(
        config=cfg,
        phantom=phantom,
        spin_density=spin_density,
        b0_map_t=b0,
        b1_map=b1,
        b1_transmit_map_t_per_a=b1_tx,
        b1_receive_map_t_per_a=b1_rx,
        gx_field_map_t_per_a=gx_field,
        gz_field_map_t_per_a=gz_field,
        gx_current_a=gx_current_grid,
        gz_current_a=gz_current_grid,
        gradient_coil_average_power_w=gradient_coil_power,
        receive_coil_inductance_h=receive_properties.inductance,
        receive_coil_resistance_ohm=resistance_receive,
        receive_coil_q_factor=q_receive,
        receive_coil_copper_q_factor=copper_q_receive,
        ferrite_rf_loss_resistance_ohm=ferrite_resistance,
        ferrite_imaginary_relative_permeability=ferrite_mu_double_prime,
        predicted_single_scan_snr=predicted_snr,
        measured_reference_snr=cfg.measured_single_scan_snr,
        water_signal_voltage_v=water_signal_v,
        single_scan_noise_rms_v=single_scan_noise_rms,
        ideal_kspace=ideal_kspace,
        measured_kspace=measured,
        acquisition_order=order,
        acquisition_times_s=acquisition_times,
        coil_temperature_k=coil_temperature,
        magnet_temperature_k=magnet_temperature,
        larmor_drift_hz=drift,
        receiver_gain=np.abs(response),
        noise_standard_deviation=complex_sigma,
        adaptive=adaptive,
        reference_nrmse=normalized_root_mean_square_error(
            spin_density, display_scale * adaptive.image
        ),
        zero_fill_reference_nrmse=normalized_root_mean_square_error(
            spin_density, zero_fill_scale * zero_fill
        ),
    )


def summarize_portable_halbach_design(
    result: PortableHalbachMRIResult,
    *,
    pulse_lengths_s: np.ndarray | None = None,
) -> PortableHalbachDesignSummary:
    """Derive RF, gradient, ADC, mass, volume, and slice design metrics.

    Pulse-length sweeps change the drive current so the center of the transmit
    coil remains a 90-degree rotation. ``active_sample_volume_m3`` counts water
    voxels whose combined transmit and off-resonance excitation is at least 50%.
    The effective slice thickness is that active cross-sectional area divided by
    the 9 mm sample diameter; it quantifies static-B0 selection along the
    unencoded direction without pretending that a dedicated slice gradient exists.
    """

    cfg = result.config
    pulses = (
        np.linspace(3.0e-6, 12.0e-6, 19)
        if pulse_lengths_s is None
        else np.asarray(pulse_lengths_s, dtype=np.float64).reshape(-1)
    )
    if pulses.size == 0 or np.any(~np.isfinite(pulses)) or np.any(pulses <= 0.0):
        raise ValueError("pulse_lengths_s must contain positive finite values")

    n = cfg.matrix_size
    center = n // 2
    axis = np.linspace(-0.5 * cfg.field_of_view_m, 0.5 * cfg.field_of_view_m, n)
    xx, zz = np.meshgrid(axis, axis, indexing="ij")
    water_mask = xx**2 + zz**2 <= (4.5e-3) ** 2
    offset_hz = (result.b0_map_t - cfg.nominal_b0_t) * (
        cfg.resonance_frequency_hz / cfg.nominal_b0_t
    )
    b1_tx_center = float(result.b1_transmit_map_t_per_a[center, center])
    b1_rx_center = float(result.b1_receive_map_t_per_a[center, center])
    tx_normalized = result.b1_transmit_map_t_per_a / max(b1_tx_center, 1e-30)
    gamma_rad = 2.0 * np.pi * cfg.resonance_frequency_hz / cfg.nominal_b0_t
    omega = 2.0 * np.pi * cfg.resonance_frequency_hz
    tx_resistance = (
        omega * cfg.transmit_coil_inductance_h / cfg.measured_transmit_coil_q
    )
    transmit_probe_q = (
        cfg.resonance_frequency_hz / cfg.transmit_probe_bandwidth_hz
    )
    receive_probe_q = cfg.resonance_frequency_hz / cfg.receiver_bandwidth_hz
    transmit_response = TunedProbeResponse.from_quality_factor(
        f0_hz=cfg.resonance_frequency_hz,
        quality_factor=transmit_probe_q,
    )
    receive_response = TunedProbeResponse.from_quality_factor(
        f0_hz=cfg.resonance_frequency_hz,
        quality_factor=receive_probe_q,
    )
    receive_profile = receive_response.transfer(offset_hz) * np.sinc(
        offset_hz * cfg.acquisition_window_s
    )

    currents = np.pi / (2.0 * gamma_rad * b1_tx_center * pulses)
    reference_current = np.pi / (
        2.0
        * gamma_rad
        * b1_tx_center
        * cfg.pcmcd_reference_pulse_s
    )
    forward_power = cfg.pcmcd_reference_peak_power_w * (
        currents / reference_current
    ) ** 2
    dc_input_power = forward_power / cfg.pcmcd_short_pulse_efficiency
    delivered_peak_current = currents * (
        1.0 - np.exp(-pulses / transmit_response.tau)
    )
    coil_power = 0.5 * delivered_peak_current**2 * tx_resistance
    signal_proxy = np.empty_like(pulses)
    active_volume = np.empty_like(pulses)
    slice_thickness = np.empty_like(pulses)
    pixel_area = (cfg.field_of_view_m / n) ** 2
    for index, pulse in enumerate(pulses):
        transmit_profile = (
            np.sinc(offset_hz * pulse)
            * transmit_response.transfer(offset_hz)
            * tx_normalized
        )
        signal_proxy[index] = abs(
            np.sum(
                water_mask
                * transmit_profile
                * result.b1_receive_map_t_per_a
                * receive_profile
            )
        )
        active = water_mask & (np.abs(transmit_profile) >= 0.5)
        active_area = float(np.count_nonzero(active) * pixel_area)
        active_volume[index] = active_area * cfg.sample_depth_m
        slice_thickness[index] = active_area / 9.0e-3
    reference_transmit = (
        np.sinc(offset_hz * cfg.rf_pulse_duration_s)
        * transmit_response.transfer(offset_hz)
        * tx_normalized
    )
    reference_proxy = abs(
        np.sum(
            water_mask
            * reference_transmit
            * result.b1_receive_map_t_per_a
            * receive_profile
        )
    )
    snr = result.predicted_single_scan_snr * signal_proxy / max(
        reference_proxy, 1e-30
    )

    spacing = cfg.field_of_view_m / (n - 1)
    gx = np.gradient(result.gx_field_map_t_per_a, spacing, axis=0)
    gz = np.gradient(result.gz_field_map_t_per_a, spacing, axis=1)
    gx_efficiency = float(gx[center, center])
    gz_efficiency = float(gz[center, center])
    peak_current = max(
        float(np.max(np.abs(result.gx_current_a))),
        float(np.max(np.abs(result.gz_current_a))),
    )
    peak_voltage = peak_current * cfg.gradient_coil_resistance_ohm + (
        cfg.gradient_coil_inductance_h
        * peak_current
        / cfg.gradient_rise_time_s
    )
    peak_gradient_power = peak_current**2 * cfg.gradient_coil_resistance_ohm

    target_adc_peak = cfg.adc_full_scale_v * cfg.adc_peak_fraction
    required_gain = target_adc_peak / max(result.water_signal_voltage_v, 1e-30)
    required_gain_db = 20.0 * np.log10(required_gain)

    return PortableHalbachDesignSummary(
        rf_coils=RFCoilDesignMetrics(
            transmit_inductance_h=cfg.transmit_coil_inductance_h,
            transmit_ac_resistance_ohm=float(tx_resistance),
            transmit_q_factor=cfg.measured_transmit_coil_q,
            transmit_loaded_probe_q_factor=transmit_probe_q,
            transmit_probe_bandwidth_hz=cfg.transmit_probe_bandwidth_hz,
            transmit_b1_center_t_per_a=b1_tx_center,
            receive_inductance_h=result.receive_coil_inductance_h,
            receive_copper_resistance_ohm=(
                result.receive_coil_resistance_ohm
                - result.ferrite_rf_loss_resistance_ohm
            ),
            receive_ferrite_loss_resistance_ohm=(
                result.ferrite_rf_loss_resistance_ohm
            ),
            receive_loaded_resistance_ohm=result.receive_coil_resistance_ohm,
            receive_copper_q_factor=result.receive_coil_copper_q_factor,
            receive_loaded_q_factor=result.receive_coil_q_factor,
            receive_loaded_probe_q_factor=receive_probe_q,
            receive_probe_bandwidth_hz=cfg.receiver_bandwidth_hz,
            receive_b1_center_t_per_a=b1_rx_center,
        ),
        pulse_sweep=RFPulseLengthSweep(
            pulse_lengths_s=pulses,
            peak_current_a=currents,
            peak_delivered_coil_current_a=delivered_peak_current,
            peak_forward_power_w=forward_power,
            peak_dc_input_power_w=dc_input_power,
            peak_coil_loss_w=coil_power,
            predicted_snr=snr,
            active_sample_volume_m3=active_volume,
            effective_slice_thickness_m=slice_thickness,
        ),
        gradients=GradientCoilDesignMetrics(
            gx_efficiency_t_per_m_per_a=gx_efficiency,
            gz_efficiency_t_per_m_per_a=gz_efficiency,
            inductance_h=cfg.gradient_coil_inductance_h,
            resistance_ohm=cfg.gradient_coil_resistance_ohm,
            peak_current_a=peak_current,
            peak_voltage_v=float(peak_voltage),
            peak_resistive_power_w=float(peak_gradient_power),
            average_winding_power_w=result.gradient_coil_average_power_w,
            rise_time_s=cfg.gradient_rise_time_s,
        ),
        receiver=ReceiverDesignMetrics(
            peak_probe_signal_v=result.water_signal_voltage_v,
            single_scan_noise_rms_v=result.single_scan_noise_rms_v,
            predicted_single_scan_snr=result.predicted_single_scan_snr,
            adc_full_scale_v=cfg.adc_full_scale_v,
            target_adc_peak_v=target_adc_peak,
            required_voltage_gain=float(required_gain),
            required_gain_db=float(required_gain_db),
        ),
        weight=SystemWeightMetrics(
            magnet_kg=0.6,
            transmitter_kg=0.7,
            other_electronics_kg=0.6,
            batteries_kg=1.7,
            baseplate_kg=1.2,
            total_kg=4.8,
            portable_without_baseplate_kg=3.6,
        ),
    )


__all__ = [
    "GradientCoilDesignMetrics",
    "PortableHalbachMRIConfig",
    "PortableHalbachDesignSummary",
    "PortableHalbachMRIResult",
    "RFCoilDesignMetrics",
    "RFPulseLengthSweep",
    "ReceiverDesignMetrics",
    "SystemWeightMetrics",
    "portable_phantom",
    "simulate_portable_halbach_mri",
    "summarize_portable_halbach_design",
]
