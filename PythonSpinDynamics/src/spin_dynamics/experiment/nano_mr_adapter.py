"""Workflow adapters from declarative nano-MR specs to native engines."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from spin_dynamics.experiment.specs import (
    Experiment,
    NanoMRBathComponent,
    NanoMRLayer,
    NanoMROpticalReadout,
    NanoMRQdyne,
    NanoMRSensor,
    NanoMRStatisticalSpectrum,
)
from spin_dynamics.nano_mr import (
    ClockModel,
    HighResolutionBudget,
    NuclearBathSpecies,
    OpticalReadoutModel,
    QdyneProtocol,
    SurfaceGeometry,
    UniformBathComponent,
    UniformNuclearLayer,
    diamond_nv_minus,
    sensor_gyromagnetic_ratio_rad_s_t,
    sic_pl6,
    simulate_qdyne,
    simulate_statistical_spectrum,
    xy_sequence,
)


def build_sensor(spec: NanoMRSensor):
    """Build a native defect sensor from a compact facade preset."""

    if not isinstance(spec, NanoMRSensor):
        raise TypeError("spec must be a NanoMRSensor")
    factory = diamond_nv_minus if spec.preset == "diamond_nv_minus" else sic_pl6
    return factory(depth_nm=spec.depth_nm, axis_lab=spec.axis_lab, label=spec.label)


def build_layer(spec: NanoMRLayer) -> UniformNuclearLayer:
    """Build a native uniform nuclear layer from a facade layer spec."""

    if not isinstance(spec, NanoMRLayer):
        raise TypeError("spec must be a NanoMRLayer")
    return UniformNuclearLayer(
        surface=SurfaceGeometry(
            point_lab_nm=np.zeros(3),
            normal_lab=np.asarray(spec.surface_normal_lab),
        ),
        components=tuple(_build_component(item) for item in spec.components),
        thickness_nm=spec.thickness_nm,
        bottom_offset_nm=spec.bottom_offset_nm,
        label=spec.label,
    )


def build_optical_readout(spec: NanoMROpticalReadout | None):
    """Build the native effective optical model, or return ``None``."""

    if spec is None:
        return None
    if not isinstance(spec, NanoMROpticalReadout):
        raise TypeError("spec must be a NanoMROpticalReadout when set")
    return OpticalReadoutModel(**dataclasses.asdict(spec))


def statistical_kwargs(experiment: Experiment) -> dict[str, Any]:
    """Translate a statistical-spectrum experiment into native arguments."""

    sequence = experiment.sequence
    if not isinstance(sequence, NanoMRStatisticalSpectrum):
        raise TypeError("sequence must be NanoMRStatisticalSpectrum")
    return {
        "sensor": build_sensor(experiment.hardware.nano_mr_sensor),
        "sample": build_layer(experiment.sample.nano_mr_layer),
        "b0_vector_tesla_lab": sequence.b0_vector_tesla_lab,
        "angular_frequencies_rad_s": np.linspace(
            sequence.angular_frequency_min_rad_s,
            sequence.angular_frequency_max_rad_s,
            sequence.num_points,
        ),
    }


def qdyne_kwargs(experiment: Experiment) -> dict[str, Any]:
    """Translate a Qdyne experiment into native protocol arguments."""

    sequence = experiment.sequence
    if not isinstance(sequence, NanoMRQdyne):
        raise TypeError("sequence must be NanoMRQdyne")
    sensor = build_sensor(experiment.hardware.nano_mr_sensor)
    sensing = xy_sequence(
        sequence.xy_order,
        sequence.xy_repetitions,
        sequence.sensing_duration_seconds,
        pulse_duration_seconds=sequence.pulse_duration_seconds,
    )
    protocol = QdyneProtocol(
        sequence=sensing,
        repetition_interval_seconds=sequence.repetition_interval_seconds,
        reference_frequency_hz=sequence.reference_frequency_hz,
        baseline_bright_probability=sequence.baseline_bright_probability,
        analysis_contrast=sequence.analysis_contrast,
        analysis_phase_rad=sequence.analysis_phase_rad,
        pulse_model=sequence.pulse_model,
        budget=HighResolutionBudget(
            sensor_coherence_seconds=_duration_or_infinite(
                sequence.sensor_coherence_seconds
            ),
            sensor_stretch_exponent=sequence.sensor_stretch_exponent,
            sample_coherence_seconds=_duration_or_infinite(
                sequence.sample_coherence_seconds
            ),
            diffusion_correlation_seconds=_duration_or_infinite(
                sequence.diffusion_correlation_seconds
            ),
            memory_coherence_seconds=_duration_or_infinite(
                sequence.memory_coherence_seconds
            ),
        ),
        clock=ClockModel(
            fractional_frequency_offset=sequence.fractional_frequency_offset,
            interval_fractional_frequency_instability=(
                sequence.interval_fractional_frequency_instability
            ),
            trigger_jitter_seconds=sequence.trigger_jitter_seconds,
        ),
    )
    return {
        "protocol": protocol,
        "signal_frequency_hz": sequence.signal_frequency_hz,
        "field_amplitude_tesla": sequence.field_amplitude_tesla,
        "sensor_gamma_rad_s_t": sensor_gyromagnetic_ratio_rad_s_t(sensor),
        "shot_count": sequence.shot_count,
        "signal_phase_rad": sequence.signal_phase_rad,
        "optical_model": build_optical_readout(
            experiment.hardware.nano_mr_optical_readout
        ),
        "seed": sequence.seed,
    }


def _build_component(spec: NanoMRBathComponent) -> UniformBathComponent:
    species = NuclearBathSpecies.from_isotope(
        spec.isotope,
        correlation_time_seconds=spec.correlation_time_seconds,
        polarization_mode=spec.polarization_mode,
        temperature_kelvin=spec.temperature_kelvin,
        polarization_fraction=spec.polarization_fraction,
        label=spec.label,
    )
    return UniformBathComponent(
        species=species,
        number_density_m3=spec.number_density_m3,
    )


def _duration_or_infinite(value: float | None) -> float:
    return np.inf if value is None else float(value)


__all__ = [
    "build_layer",
    "build_optical_readout",
    "build_sensor",
    "qdyne_kwargs",
    "statistical_kwargs",
    "simulate_qdyne",
    "simulate_statistical_spectrum",
]
