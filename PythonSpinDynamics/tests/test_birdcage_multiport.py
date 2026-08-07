"""Loaded, matched, reciprocal multiport birdcage tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.fields.birdcage import BirdcageGeometry
from spin_dynamics.fields.birdcage_circuit import tuned_low_pass_birdcage
from spin_dynamics.fields.birdcage_multiport import (
    BirdcageBranchLoading,
    BirdcageLoadedCircuit,
    BirdcageMultiport,
    birdcage_branch_mutual_inductance_matrix,
    birdcage_conductive_loading_resistance,
    calibrate_birdcage_conductor_quality_factor,
    design_independent_l_match,
    retune_loaded_birdcage,
    solve_birdcage_receive_sensitivities,
)
from spin_dynamics.workflows.receiver_arrays import roemer_combine


def _geometry() -> BirdcageGeometry:
    return BirdcageGeometry(
        radius=0.12,
        length=0.24,
        n_rungs=8,
        ring_segments_per_section=3,
    )


def _circuit():
    return tuned_low_pass_birdcage(
        _geometry(),
        42.58e6,
        rung_inductance_h=150.0e-9,
        end_ring_inductance_h=30.0e-9,
        rung_resistance_ohm=0.08,
        end_ring_resistance_ohm=0.015,
    )


def test_geometry_mutual_matrix_is_reciprocal_and_cyclic() -> None:
    geometry = _geometry()
    mutual = birdcage_branch_mutual_inductance_matrix(geometry)

    assert mutual.shape == (3 * geometry.n_rungs, 3 * geometry.n_rungs)
    np.testing.assert_allclose(mutual, mutual.T, atol=1.0e-18)
    np.testing.assert_allclose(np.diag(mutual), 0.0)
    rung_block = mutual[: geometry.n_rungs, : geometry.n_rungs]
    np.testing.assert_allclose(
        rung_block[0],
        np.roll(rung_block[1], -1),
        atol=1.0e-16,
    )


def test_conductive_loading_is_psd_and_scales_as_frequency_squared() -> None:
    geometry = _geometry()
    axis = np.linspace(-0.03, 0.03, 3)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack((xx, yy, zz), axis=-1)
    volume = float((axis[1] - axis[0]) ** 3)
    low = birdcage_conductive_loading_resistance(
        geometry,
        10.0e6,
        points,
        conductivity_s_per_m=0.5,
        cell_volume_m3=volume,
    )
    high = birdcage_conductive_loading_resistance(
        geometry,
        20.0e6,
        points,
        conductivity_s_per_m=0.5,
        cell_volume_m3=volume,
    )

    assert float(np.min(np.linalg.eigvalsh(low))) > -1.0e-12
    np.testing.assert_allclose(high, 4.0 * low, rtol=1.0e-12, atol=1.0e-15)


def test_loading_reduces_q_and_retuning_restores_target_frequency() -> None:
    circuit = _circuit()
    size = 3 * circuit.geometry.n_rungs
    extra_loss = 0.02 * np.eye(size)
    loading = BirdcageBranchLoading(size, resistance_ohm=extra_loss)
    loaded = BirdcageLoadedCircuit(circuit, loading)
    target_hz = 42.58e6
    retuned = retune_loaded_birdcage(circuit, loading, target_hz)

    unloaded_mode = circuit.modal_analysis().azimuthal_modes(1)[0]
    loaded_mode = loaded.modal_analysis().azimuthal_modes(1)[0]
    retuned_modes = retuned.modal_analysis().azimuthal_modes(1)
    assert loaded_mode.quality_factor < unloaded_mode.quality_factor
    for mode in retuned_modes:
        assert mode.frequency_hz == pytest.approx(target_hz, rel=1.0e-12)


def test_conductor_loss_calibration_sets_explicit_unloaded_q() -> None:
    circuit = _circuit()
    size = 3 * circuit.geometry.n_rungs
    mutual = 0.05 * birdcage_branch_mutual_inductance_matrix(circuit.geometry)
    loading = BirdcageBranchLoading(size, inductance_coupling_h=mutual)
    retuned = retune_loaded_birdcage(circuit, loading, 42.58e6)
    calibrated = calibrate_birdcage_conductor_quality_factor(retuned, 180.0)

    modes = calibrated.modal_analysis().azimuthal_modes(1)
    for mode in modes:
        assert mode.frequency_hz == pytest.approx(42.58e6, rel=1.0e-12)
        assert mode.quality_factor == pytest.approx(180.0, rel=1.0e-12)


def test_port_impedance_is_reciprocal_and_realizes_requested_currents() -> None:
    circuit = _circuit()
    frequency = 42.58e6
    ports = BirdcageMultiport(circuit, (0, 2), labels=("x", "y"))
    impedance = ports.impedance_matrix_ohm(frequency)
    requested = np.array([0.7 + 0.2j, -0.3j])
    driven = ports.solve_port_currents(frequency, requested)
    realized = driven.currents.rung_currents_a[[0, 2]]

    np.testing.assert_allclose(impedance, impedance.T, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(realized, requested, rtol=1.0e-11, atol=1.0e-11)
    assert driven.supplied_power_w == pytest.approx(
        driven.dissipated_power_w,
        rel=1.0e-11,
    )


def test_single_port_l_match_transforms_to_fifty_ohms() -> None:
    frequency = 42.58e6
    ports = BirdcageMultiport(_circuit(), (0,))
    coil_impedance = ports.impedance_matrix_ohm(frequency)
    matching = design_independent_l_match(coil_impedance, frequency)
    input_impedance = matching.input_impedance_ohm(coil_impedance)
    scattering = matching.scattering_matrix(coil_impedance)

    np.testing.assert_allclose(input_impedance, [[50.0]], atol=1.0e-10)
    assert abs(scattering[0, 0]) < 1.0e-12
    assert len(matching.component_summary()) == 1


def test_receive_maps_noise_and_roemer_reconstruction_share_channel_basis() -> None:
    _geometry()
    frequency = 42.58e6
    ports = BirdcageMultiport(_circuit(), (0, 2), labels=("x", "y"))
    matching = design_independent_l_match(
        ports.impedance_matrix_ohm(frequency),
        frequency,
    )
    axis = np.linspace(-0.05, 0.05, 9)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    phantom = ((xx / 0.045) ** 2 + (yy / 0.040) ** 2 <= 1.0).astype(np.float64)
    maps = solve_birdcage_receive_sensitivities(
        ports,
        frequency,
        points,
        matching=matching,
        normalization_weights=phantom,
    )
    sensitivities = maps.normalized_complex
    channel_images = (
        sensitivities[..., np.newaxis] * phantom[np.newaxis, ..., np.newaxis]
    )
    covariance = maps.noise_correlation
    reconstruction = roemer_combine(
        channel_images,
        sensitivities,
        covariance,
    )[..., 0]

    assert maps.b1_minus_t_per_a.shape == (2, 9, 9)
    assert maps.noise_covariance_v2_per_hz.shape == (2, 2)
    assert float(np.min(np.linalg.eigvalsh(maps.noise_covariance_v2_per_hz))) > -1e-30
    np.testing.assert_allclose(reconstruction, phantom, atol=1.0e-12)
