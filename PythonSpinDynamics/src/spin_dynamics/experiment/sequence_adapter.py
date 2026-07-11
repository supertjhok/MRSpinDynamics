"""Compile and execute general SequenceIR programs through the facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.motion import (
    initialize_ensemble_from_domain,
    make_motion_field_maps,
)
from spin_dynamics.noise import NoiseMetadata, add_received_noise, as_noise_spec
from spin_dynamics.sequences import (
    CompiledSequence,
    compile_sequence,
    compiled_to_motion_steps,
    run_motion_sequence,
)

_GRADIENT_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class SequenceIRResult:
    """General IR execution result with compiled receive metadata."""

    signal: np.ndarray
    clean_signal: np.ndarray
    sample_times_seconds: np.ndarray
    sample_labels: tuple[str, ...]
    adc_frequency_offsets_hz: np.ndarray
    adc_phase_offsets_rad: np.ndarray
    step_end_times_seconds: np.ndarray
    step_labels: tuple[str, ...]
    final_positions_m: np.ndarray
    final_magnetization: np.ndarray
    particle_weights: np.ndarray
    diffusion_coefficient_m2_s: np.ndarray
    noise_metadata: NoiseMetadata | None


def require_domain(experiment: Any):
    domain = experiment.sample.sequence_domain
    if domain is None:
        raise ValueError(
            "SequenceIRExecution requires sample.sequence_domain "
            "(a SequenceDomain)"
        )
    return domain


def compile_for_experiment(experiment: Any) -> CompiledSequence:
    sequence = experiment.sequence
    return compile_sequence(
        sequence.ir, system_frequency_hz=sequence.system_frequency_hz
    )


def prepare_for_experiment(experiment: Any) -> tuple[CompiledSequence, Any]:
    """Validate backend compatibility and return compiled IR motion steps."""

    domain_spec = require_domain(experiment)
    extension_blocks = [
        index
        for index, block in enumerate(experiment.sequence.ir.blocks)
        if block.extensions
    ]
    if extension_blocks:
        raise NotImplementedError(
            "general SequenceIR execution cannot interpret block extensions "
            f"(present in blocks {extension_blocks})"
        )
    compiled = compile_for_experiment(experiment)
    gradient_axes = tuple(
        _GRADIENT_INDEX[channel] for channel in domain_spec.gradient_channels
    )
    steps = compiled_to_motion_steps(
        compiled,
        spatial_dimensions=len(domain_spec.axes),
        gradient_axes=gradient_axes,
    )

    noise_spec = as_noise_spec(experiment.acquisition.noise)
    if noise_spec is not None and noise_spec.model != "white":
        raise ValueError(
            "general SequenceIR execution currently supports white receive "
            "noise only; probe noise requires a probe-aware backend"
        )
    return compiled, steps


def sequence_kwargs(experiment: Any) -> dict[str, Any]:
    execution = experiment.sequence
    sample = experiment.sample
    domain_spec = require_domain(experiment)
    compiled, steps = prepare_for_experiment(experiment)

    if execution.seed is None:
        initialization_seed = None
        rng = np.random.default_rng()
    else:
        initialization_state, motion_state = np.random.SeedSequence(
            execution.seed
        ).spawn(2)
        initialization_seed = int(initialization_state.generate_state(1)[0])
        rng = np.random.default_rng(motion_state)

    spatial_domain = SpatialDomain(domain_spec.axes)
    ensemble = initialize_ensemble_from_domain(
        spatial_domain,
        domain_spec.density,
        walkers_per_cell=execution.walkers_per_cell,
        diffusion_coefficient=(
            0.0
            if sample.diffusion_coefficient is None
            else sample.diffusion_coefficient
        ),
        seed=initialization_seed,
        jitter=execution.jitter,
    )
    fields = make_motion_field_maps(
        spatial_domain,
        b0_map=domain_spec.b0_map_rad_s,
        b1_tx_map=domain_spec.b1_tx_map,
        b1_rx_map=domain_spec.b1_rx_map,
    )
    return {
        "compiled": compiled,
        "ensemble": ensemble,
        "fields": fields,
        "steps": steps,
        "velocity": (
            None
            if domain_spec.velocity_m_per_s is None
            else np.asarray(domain_spec.velocity_m_per_s, dtype=np.float64)
        ),
        "rng": rng,
        "t1": float("inf") if sample.t1_seconds is None else sample.t1_seconds,
        "t2": float("inf") if sample.t2_seconds is None else sample.t2_seconds,
        "boundary": execution.boundary,
        "default_substeps": execution.default_substeps,
        "noise": experiment.acquisition.noise,
    }


def run_sequence_ir(
    *,
    compiled: CompiledSequence,
    ensemble: Any,
    fields: Any,
    steps: Any,
    velocity: Any,
    rng: np.random.Generator,
    t1: float,
    t2: float,
    boundary: str,
    default_substeps: int,
    noise: Any,
) -> SequenceIRResult:
    """Execute compiled IR steps and apply nominal receiver demodulation."""

    motion = run_motion_sequence(
        ensemble,
        fields,
        steps,
        velocity=velocity,
        rng=rng,
        t1=t1,
        t2=t2,
        boundary=boundary,
        default_substeps=default_substeps,
    )
    if motion.signal.size != compiled.adc.times_seconds.size:
        raise RuntimeError("motion backend returned the wrong number of ADC samples")
    clean = motion.signal * np.exp(-1j * compiled.adc.phase_offsets_rad)
    signal, noise_metadata = add_received_noise(
        clean, noise, sample_axis=compiled.adc.times_seconds
    )
    final = motion.final_ensemble
    return SequenceIRResult(
        signal=signal,
        clean_signal=clean,
        sample_times_seconds=motion.sample_times,
        sample_labels=motion.sample_labels,
        adc_frequency_offsets_hz=compiled.adc.frequency_offsets_hz,
        adc_phase_offsets_rad=compiled.adc.phase_offsets_rad,
        step_end_times_seconds=motion.step_end_times,
        step_labels=motion.step_labels,
        final_positions_m=final.positions,
        final_magnetization=final.magnetization,
        particle_weights=final.weights,
        diffusion_coefficient_m2_s=final.diffusion_coefficient,
        noise_metadata=noise_metadata,
    )


__all__ = [
    "SequenceIRResult",
    "compile_for_experiment",
    "prepare_for_experiment",
    "require_domain",
    "run_sequence_ir",
    "sequence_kwargs",
]
