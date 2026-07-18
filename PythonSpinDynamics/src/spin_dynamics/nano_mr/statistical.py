"""Analytic and voxel statistical nano-NMR spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.nano_mr.baths import (
    PLANCK_CONSTANT_J_S,
    NuclearBathSample,
    NuclearBathSpecies,
    UniformNuclearLayer,
    VoxelNuclearSample,
)
from spin_dynamics.nano_mr.filter_functions import PulseModel, toggling_integral
from spin_dynamics.nano_mr.frames import unit_vector
from spin_dynamics.nano_mr.geometry import (
    BOHR_MAGNETON_J_PER_T,
    MU0_OVER_4PI,
)
from spin_dynamics.nano_mr.sensors import DefectSpinSensor
from spin_dynamics.nano_mr.sequences import SensingSequence


@dataclass(frozen=True, eq=False)
class StatisticalSpectrumResult:
    """Two-sided magnetic-field PSD and per-component metadata.

    The PSD uses angular frequency and has units ``tesla^2 second``:
    ``variance = integral S_B(omega) d omega / (2*pi)``.
    """

    angular_frequencies_rad_s: np.ndarray
    component_labels: tuple[str, ...]
    isotopes: tuple[str, ...]
    polarization_modes: tuple[str, ...]
    component_mean_spin_projections: np.ndarray
    larmor_angular_frequencies_rad_s: np.ndarray
    component_field_variances_t2: np.ndarray
    component_psd_t2_s: np.ndarray
    sample_label: str

    @property
    def total_field_variance_t2(self) -> float:
        """Sum of component AC-field variances."""

        return float(np.sum(self.component_field_variances_t2))

    @property
    def rms_field_tesla(self) -> float:
        """Root-mean-square transverse nuclear field."""

        return float(np.sqrt(self.total_field_variance_t2))

    @property
    def total_psd_t2_s(self) -> np.ndarray:
        """Total two-sided magnetic-field PSD."""

        return np.sum(self.component_psd_t2_s, axis=0)


@dataclass(frozen=True)
class GaussianCoherenceResult:
    """Filter-overlap dephasing exponent and remaining coherence."""

    chi: float
    coherence: float
    phase_variance_rad2: float
    sensor_gyromagnetic_ratio_rad_s_t: float


def simulate_statistical_spectrum(
    sensor: DefectSpinSensor,
    sample: NuclearBathSample,
    b0_vector_tesla_lab,
    angular_frequencies_rad_s,
    *,
    sensor_position_lab_nm=None,
) -> StatisticalSpectrumResult:
    """Return the multi-isotope Lorentzian field-noise spectrum.

    Uniform layers use an analytic planar integral. Voxel samples use a
    midpoint sum over voxel volumes and require ``sensor_position_lab_nm``.
    """

    field = np.asarray(b0_vector_tesla_lab, dtype=np.float64).reshape(3)
    frequencies = np.asarray(angular_frequencies_rad_s, dtype=np.float64)
    if not np.all(np.isfinite(field)) or np.linalg.norm(field) <= 0.0:
        raise ValueError("b0_vector_tesla_lab must be finite and non-zero")
    if frequencies.ndim != 1 or frequencies.size < 2:
        raise ValueError("angular_frequencies_rad_s must be one-dimensional")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("angular_frequencies_rad_s must be finite")
    field_magnitude = float(np.linalg.norm(field))
    field_direction = field / field_magnitude

    if isinstance(sample, UniformNuclearLayer):
        components = sample.components
        species = tuple(item.species for item in components)
        variances = np.array(
            [
                _uniform_component_variance(
                    sensor,
                    sample,
                    item.number_density_m3,
                    item.species,
                    field_magnitude,
                    field_direction,
                )
                for item in components
            ],
            dtype=np.float64,
        )
    elif isinstance(sample, VoxelNuclearSample):
        if sensor_position_lab_nm is None:
            raise ValueError(
                "sensor_position_lab_nm is required for a voxel sample"
            )
        species = tuple(item.species for item in sample.components)
        variances = np.array(
            [
                _voxel_component_variance(
                    sensor,
                    sample,
                    item,
                    field_magnitude,
                    field_direction,
                    sensor_position_lab_nm,
                )
                for item in sample.components
            ],
            dtype=np.float64,
        )
    else:
        raise TypeError("sample must be UniformNuclearLayer or VoxelNuclearSample")

    larmor = np.array(
        [
            2.0 * np.pi * abs(item.gamma_hz_per_t) * field_magnitude
            for item in species
        ]
    )
    psd = np.vstack(
        [
            rotating_lorentzian_psd(
                frequencies,
                variance,
                larmor_frequency,
                item.correlation_time_seconds,
            )
            for variance, larmor_frequency, item in zip(
                variances, larmor, species
            )
        ]
    )
    return StatisticalSpectrumResult(
        angular_frequencies_rad_s=frequencies.copy(),
        component_labels=tuple(item.label for item in species),
        isotopes=tuple(item.isotope for item in species),
        polarization_modes=tuple(item.polarization_mode for item in species),
        component_mean_spin_projections=np.array(
            [
                item.mean_spin_projection(field_magnitude)
                for item in species
            ],
            dtype=np.float64,
        ),
        larmor_angular_frequencies_rad_s=larmor,
        component_field_variances_t2=variances,
        component_psd_t2_s=psd,
        sample_label=sample.label,
    )


def rotating_lorentzian_psd(
    angular_frequencies_rad_s,
    field_variance_t2: float,
    larmor_angular_frequency_rad_s: float,
    correlation_time_seconds: float,
) -> np.ndarray:
    """Return a normalized two-sided rotating Lorentzian PSD."""

    frequencies = np.asarray(angular_frequencies_rad_s, dtype=np.float64)
    variance = float(field_variance_t2)
    larmor = float(larmor_angular_frequency_rad_s)
    correlation = float(correlation_time_seconds)
    if variance < 0.0 or not np.isfinite(variance):
        raise ValueError("field_variance_t2 must be finite and non-negative")
    if larmor < 0.0 or not np.isfinite(larmor):
        raise ValueError(
            "larmor_angular_frequency_rad_s must be finite and non-negative"
        )
    if correlation <= 0.0 or not np.isfinite(correlation):
        raise ValueError(
            "correlation_time_seconds must be positive and finite"
        )
    positive = correlation / (
        1.0 + ((frequencies - larmor) * correlation) ** 2
    )
    negative = correlation / (
        1.0 + ((frequencies + larmor) * correlation) ** 2
    )
    return variance * (positive + negative)


def gaussian_filter_coherence(
    sensor: DefectSpinSensor,
    sequence: SensingSequence,
    spectrum: StatisticalSpectrumResult,
    *,
    pulse_model: PulseModel = "ideal",
    samples_per_pulse: int = 64,
) -> GaussianCoherenceResult:
    """Return Gaussian coherence from filter overlap with a two-sided PSD."""

    frequencies = spectrum.angular_frequencies_rad_s
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("spectrum angular frequencies must be strictly increasing")
    symmetry_atol = max(
        1.0e-12,
        1.0e-12 * float(np.max(np.abs(frequencies), initial=0.0)),
    )
    if not np.allclose(
        frequencies,
        -frequencies[::-1],
        rtol=1.0e-10,
        atol=symmetry_atol,
    ):
        raise ValueError(
            "Gaussian coherence requires a two-sided frequency grid "
            "symmetric about zero"
        )
    response = toggling_integral(
        sequence,
        frequencies,
        pulse_model=pulse_model,
        samples_per_pulse=samples_per_pulse,
    )
    gamma_sensor = sensor_gyromagnetic_ratio_rad_s_t(sensor)
    integrand = (
        gamma_sensor**2
        * spectrum.total_psd_t2_s
        * np.abs(response) ** 2
    )
    phase_variance = _trapezoid(integrand, frequencies) / (2.0 * np.pi)
    chi = 0.5 * phase_variance
    return GaussianCoherenceResult(
        chi=float(chi),
        coherence=float(np.exp(-chi)),
        phase_variance_rad2=float(phase_variance),
        sensor_gyromagnetic_ratio_rad_s_t=gamma_sensor,
    )


def sensor_gyromagnetic_ratio_rad_s_t(sensor: DefectSpinSensor) -> float:
    """Return axial addressed-transition Zeeman slope in radians/s/T."""

    local_axis = np.array([0.0, 0.0, 1.0])
    effective_g = float(np.linalg.norm(sensor.g_tensor.T @ local_axis))
    return (
        2.0
        * np.pi
        * BOHR_MAGNETON_J_PER_T
        / PLANCK_CONSTANT_J_S
        * effective_g
    )


def _uniform_component_variance(
    sensor: DefectSpinSensor,
    sample: UniformNuclearLayer,
    number_density_m3: float,
    species: NuclearBathSpecies,
    field_magnitude: float,
    field_direction: np.ndarray,
) -> float:
    lower_m = (sensor.depth_nm + sample.bottom_offset_nm) * 1.0e-9
    upper_m = (
        np.inf
        if sample.thickness_nm is None
        else lower_m + sample.thickness_nm * 1.0e-9
    )
    inverse_cube_range = lower_m**-3
    if np.isfinite(upper_m):
        inverse_cube_range -= upper_m**-3
    geometry = planar_transverse_variance_geometry(
        sensor.axis_lab,
        field_direction,
        sample.surface.normal_lab,
        inverse_cube_range,
    )
    moment_per_spin = (
        MU0_OVER_4PI * PLANCK_CONSTANT_J_S * species.gamma_hz_per_t
    )
    second_moment = species.transverse_spin_second_moment(field_magnitude)
    return float(
        number_density_m3
        * second_moment
        * moment_per_spin**2
        * geometry
    )


def planar_transverse_variance_geometry(
    sensor_axis: np.ndarray,
    field_direction: np.ndarray,
    surface_normal: np.ndarray,
    inverse_cube_range: float,
) -> float:
    """Return the planar transverse dipolar geometry integral in inverse m3.

    ``inverse_cube_range`` is ``r_lower^-3 - r_upper^-3`` for a planar slab.
    This geometry-only helper is shared by statistical spectra and scanning
    depth-profile forward operators.
    """
    n = unit_vector(sensor_axis, name="sensor_axis")
    b = unit_vector(field_direction, name="field_direction")
    k = unit_vector(surface_normal, name="surface_normal")
    bn = float(b @ n)
    bk = float(b @ k)
    nk = float(n @ k)
    j0 = np.pi / 6.0
    j2_nn = np.pi / 36.0 * (1.0 + 3.0 * nk**2)
    j2_bn = np.pi / 36.0 * (bn + 3.0 * bk * nk)
    j4_bbnn = (
        np.pi / 288.0 * (1.0 + 2.0 * bn**2)
        + np.pi
        / 96.0
        * (nk**2 + bk**2 + 4.0 * bn * bk * nk)
        + np.pi / 96.0 * bk**2 * nk**2
    )
    angular_factor = (
        (1.0 - bn**2) * j0
        + 3.0 * j2_nn
        + 6.0 * bn * j2_bn
        - 9.0 * j4_bbnn
    )
    if angular_factor < -1.0e-14:
        raise RuntimeError("analytic planar variance became negative")
    return float(max(0.0, angular_factor) * inverse_cube_range)


def _voxel_component_variance(
    sensor: DefectSpinSensor,
    sample: VoxelNuclearSample,
    component,
    field_magnitude: float,
    field_direction: np.ndarray,
    sensor_position_lab_nm,
) -> float:
    sensor_position = np.asarray(sensor_position_lab_nm, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(sensor_position)):
        raise ValueError("sensor_position_lab_nm must be finite")
    displacement_m = (sample.positions_lab_nm - sensor_position) * 1.0e-9
    distances = np.linalg.norm(displacement_m, axis=1)
    if np.any(distances <= 0.0):
        raise ValueError("voxel centers must not coincide with the sensor")
    directions = displacement_m / distances[:, None]
    n = sensor.axis_lab
    moment_per_spin = (
        MU0_OVER_4PI
        * PLANCK_CONSTANT_J_S
        * component.species.gamma_hz_per_t
    )
    projection = directions @ n
    coupling = (
        moment_per_spin
        * (3.0 * projection[:, None] * directions - n[None, :])
        / distances[:, None] ** 3
    )
    transverse_coupling2 = np.sum(coupling**2, axis=1) - (
        coupling @ field_direction
    ) ** 2
    nuclei = (
        sample.density_vector_m3(component)
        * sample.voxel_volumes_nm3
        * 1.0e-27
    )
    second_moment = component.species.transverse_spin_second_moment(
        field_magnitude
    )
    return float(np.sum(nuclei * second_moment * transverse_coupling2))


def _trapezoid(values: np.ndarray, axis: np.ndarray) -> float:
    return float(
        np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(axis))
    )


__all__ = [
    "GaussianCoherenceResult",
    "StatisticalSpectrumResult",
    "gaussian_filter_coherence",
    "planar_transverse_variance_geometry",
    "rotating_lorentzian_psd",
    "sensor_gyromagnetic_ratio_rad_s_t",
    "simulate_statistical_spectrum",
]
