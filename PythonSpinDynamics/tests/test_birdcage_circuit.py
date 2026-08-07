"""Resonant birdcage circuit, loss, tolerance, and drive tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.fields.birdcage import (
    BirdcageGeometry,
    birdcage_field_metrics,
    solve_birdcage_field,
)
from spin_dynamics.fields.birdcage_circuit import (
    BirdcageCircuit,
    birdcage_quadrature_port_voltages,
    tuned_high_pass_birdcage,
    tuned_low_pass_birdcage,
)


def _geometry() -> BirdcageGeometry:
    return BirdcageGeometry(
        radius=0.15,
        length=0.30,
        n_rungs=16,
        ring_segments_per_section=8,
    )


@pytest.mark.parametrize(
    "factory, architecture",
    (
        (tuned_low_pass_birdcage, "low_pass"),
        (tuned_high_pass_birdcage, "high_pass"),
    ),
)
def test_tuned_uniform_cages_match_analytical_frequencies(
    factory,
    architecture: str,
) -> None:
    target_hz = 63.87e6
    circuit = factory(
        _geometry(),
        target_hz,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )
    analysis = circuit.modal_analysis()
    fundamental = analysis.azimuthal_modes(1)

    assert circuit.architecture == architecture
    assert len(fundamental) == 2
    assert analysis.splitting_hz(1) < 1.0e-4
    for mode in fundamental:
        assert mode.frequency_hz == pytest.approx(target_hz, rel=1.0e-12)
        assert mode.frequency_hz == pytest.approx(
            circuit.uniform_mode_frequency_hz(1),
            rel=1.0e-12,
        )
        assert mode.quality_factor > 100.0
        assert mode.currents.max_kcl_residual_a < 1.0e-12


def test_modal_frequencies_match_uniform_ladder_formula_for_all_modes() -> None:
    circuit = tuned_low_pass_birdcage(
        _geometry(),
        63.87e6,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
    )
    analysis = circuit.modal_analysis()
    for azimuthal_index in range(1, circuit.geometry.n_rungs // 2 + 1):
        expected = circuit.uniform_mode_frequency_hz(azimuthal_index)
        for mode in analysis.azimuthal_modes(azimuthal_index):
            assert mode.frequency_hz == pytest.approx(expected, rel=1.0e-12)


def test_band_pass_components_follow_same_uniform_modal_formula() -> None:
    circuit = BirdcageCircuit(
        geometry=_geometry(),
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_capacitance_f=20.0e-12,
        end_ring_capacitance_f=200.0e-12,
    )
    fundamental = circuit.modal_analysis().azimuthal_modes(1)

    assert circuit.architecture == "band_pass"
    assert len(fundamental) == 2
    for mode in fundamental:
        assert mode.frequency_hz == pytest.approx(
            circuit.uniform_mode_frequency_hz(1),
            rel=1.0e-12,
        )


def test_component_tolerance_splits_fundamental_pair() -> None:
    ideal = tuned_low_pass_birdcage(
        _geometry(),
        63.87e6,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )
    capacitance = np.array(ideal.rung_capacitance_f, copy=True)
    capacitance[0] *= 1.03
    perturbed = BirdcageCircuit(
        geometry=ideal.geometry,
        rung_inductance_h=ideal.rung_inductance_h,
        end_ring_inductance_h=ideal.end_ring_inductance_h,
        rung_capacitance_f=capacitance,
        rung_resistance_ohm=ideal.rung_resistance_ohm,
        end_ring_resistance_ohm=ideal.end_ring_resistance_ohm,
    )

    assert ideal.modal_analysis().splitting_hz(1) < 1.0e-4
    assert perturbed.modal_analysis().splitting_hz(1) > 10.0e3


def test_split_pair_midpoint_can_reverse_quadrature_handedness() -> None:
    geometry = _geometry()
    target_hz = 63.87e6
    ideal = tuned_low_pass_birdcage(
        geometry,
        target_hz,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )
    capacitance = np.array(ideal.rung_capacitance_f, copy=True)
    capacitance[0] *= 1.03
    perturbed = BirdcageCircuit(
        geometry=geometry,
        rung_inductance_h=ideal.rung_inductance_h,
        end_ring_inductance_h=ideal.end_ring_inductance_h,
        rung_capacitance_f=capacitance,
        rung_resistance_ohm=ideal.rung_resistance_ohm,
        end_ring_resistance_ohm=ideal.end_ring_resistance_ohm,
    )
    source = birdcage_quadrature_port_voltages(geometry, handedness=1)
    perturbed_family = perturbed.modal_analysis().azimuthal_modes(1)
    midpoint_hz = np.mean([mode.frequency_hz for mode in perturbed_family])
    ideal_field = solve_birdcage_field(
        geometry,
        ideal.solve_drive(target_hz, source).currents,
        geometry.center,
    )
    perturbed_field = solve_birdcage_field(
        geometry,
        perturbed.solve_drive(midpoint_hz, source).currents,
        geometry.center,
    )
    perturbed_metrics = birdcage_field_metrics(
        perturbed_field,
        target_handedness=1,
    )

    amplitude_ratio = abs(perturbed_field.b1_plus_t) / abs(ideal_field.b1_plus_t)
    assert 0.20 < amplitude_ratio < 0.35
    assert perturbed_metrics.circularity_db < 0.0


def test_driven_solution_conserves_real_power_and_drives_circular_field() -> None:
    geometry = _geometry()
    target_hz = 63.87e6
    circuit = tuned_low_pass_birdcage(
        geometry,
        target_hz,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )
    source = birdcage_quadrature_port_voltages(
        geometry,
        voltage_v=1.0,
        handedness=1,
    )
    driven = circuit.solve_drive(target_hz, source)
    center = solve_birdcage_field(
        geometry,
        driven.currents,
        geometry.center,
    )
    metrics = birdcage_field_metrics(center, target_handedness=1)

    assert driven.supplied_power_w == pytest.approx(
        driven.dissipated_power_w,
        rel=1.0e-11,
    )
    assert driven.currents.max_kcl_residual_a < 1.0e-12
    assert metrics.circularity > 0.999
    assert metrics.circularity_db > 30.0


def test_single_port_reports_input_impedance() -> None:
    circuit = tuned_low_pass_birdcage(
        _geometry(),
        63.87e6,
        rung_inductance_h=180.0e-9,
        end_ring_inductance_h=35.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )
    source = np.zeros(circuit.geometry.n_rungs, dtype=np.complex128)
    source[0] = 1.0
    solution = circuit.solve_drive(63.87e6, source)

    assert np.isfinite(solution.input_impedance_ohm)
    assert solution.input_impedance_ohm.real > 0.0


def test_circuit_validation_rejects_unphysical_values() -> None:
    geometry = _geometry()
    with pytest.raises(ValueError, match="capacitance"):
        BirdcageCircuit(
            geometry=geometry,
            rung_inductance_h=1.0e-7,
            end_ring_inductance_h=1.0e-8,
        )
    with pytest.raises(ValueError, match="positive"):
        BirdcageCircuit(
            geometry=geometry,
            rung_inductance_h=-1.0,
            end_ring_inductance_h=1.0e-8,
            rung_capacitance_f=1.0e-12,
        )
    with pytest.raises(ValueError, match="divisible by four"):
        birdcage_quadrature_port_voltages(
            BirdcageGeometry(radius=0.1, length=0.2, n_rungs=10)
        )
