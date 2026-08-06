"""Coupled receiver-network, multiport PEEC, and workflow tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from spin_dynamics.experiment import (
    Acquisition,
    CPMGImaging,
    Experiment,
    Hardware,
    ImagingPlane,
    Phantom,
    ReceiverNetwork,
    RxArray,
    RxCoil,
    Sample,
    SolenoidCoil,
)
from spin_dynamics.fields import (
    Conductor,
    extract_impedance,
    extract_multiport_impedance,
)
from spin_dynamics.receiver_network import (
    BOLTZMANN_J_PER_K,
    ActiveReceiverNetwork,
    LNAInputModel,
    analyze_receiver_coupling_sweep,
    coupled_resonant_modes,
    covariance_to_correlation,
    mutual_cancellation_capacitance,
    optimal_channel_snr,
    scale_noise_covariance,
    shared_capacitor_mesh_impedance,
)
from spin_dynamics.workflows import run_ideal_receiver_array_cpmg_imaging


def _coupled_network(*, bandwidth_hz: float = 2.0) -> ReceiverNetwork:
    return ReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array(
            [[2.0 + 20.0j, 0.4 + 6.0j], [0.4 + 6.0j, 2.5 + 20.0j]],
            dtype=np.complex128,
        ),
        series_impedance_ohm=np.diag([-20.0j, -20.0j]),
        load_impedance_ohm=np.array([50.0, 60.0]),
        temperature_k=300.0,
        noise_bandwidth_hz=bandwidth_hz,
    )


def _geometric_maps() -> np.ndarray:
    return np.array(
        [
            [[1.0, 1.0], [0.2, 0.2]],
            [[0.2j, 0.2j], [1.0j, 1.0j]],
        ],
        dtype=np.complex128,
    )


def test_matched_one_port_transfer_and_johnson_noise() -> None:
    network = ReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array([[50.0 + 3.0j]]),
        series_impedance_ohm=np.array([[-3.0j]]),
        load_impedance_ohm=50.0,
        temperature_k=300.0,
        noise_bandwidth_hz=2.0,
    )
    result = network.solve(np.ones((1, 2, 2), dtype=np.complex128))

    np.testing.assert_allclose(result.transfer_matrix, [[0.5]])
    np.testing.assert_allclose(result.effective_sensitivities, 0.5)
    expected = 100.0 * BOLTZMANN_J_PER_K * 300.0 * 2.0
    np.testing.assert_allclose(result.noise_covariance_v2, [[expected]])


def test_unmatched_one_port_noise_uses_parallel_output_resistance() -> None:
    network = ReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array([[25.0]]),
        load_impedance_ohm=75.0,
        temperature_k=300.0,
        noise_bandwidth_hz=2.0,
    )
    result = network.solve(np.ones((1, 1), dtype=np.complex128))

    output_resistance = 25.0 * 75.0 / (25.0 + 75.0)
    expected = (
        4.0
        * BOLTZMANN_J_PER_K
        * 300.0
        * 2.0
        * output_resistance
    )
    np.testing.assert_allclose(result.noise_covariance_v2, [[expected]])


def test_loaded_maps_and_noise_follow_same_transfer_matrix() -> None:
    network = _coupled_network()
    maps = _geometric_maps()
    result = network.solve(maps)

    expected = (
        result.transfer_matrix @ maps.reshape(2, -1)
    ).reshape(maps.shape)
    np.testing.assert_allclose(result.effective_sensitivities, expected)
    np.testing.assert_allclose(
        result.noise_covariance_v2,
        result.noise_covariance_v2.conj().T,
    )
    assert np.min(np.linalg.eigvalsh(result.noise_covariance_v2)) >= -1e-30
    assert abs(result.noise_correlation[0, 1]) > 0.0


def test_preamp_voltage_and_current_noise_are_added_at_output() -> None:
    voltage_density = np.array([[4.0e-18, 0.5e-18], [0.5e-18, 3.0e-18]])
    current_density = np.diag([2.0e-24, 3.0e-24])
    base = _coupled_network(bandwidth_hz=5.0)
    network = dataclasses.replace(
        base,
        preamp_voltage_noise_covariance_v2_per_hz=voltage_density,
        preamp_current_noise_covariance_a2_per_hz=current_density,
    )
    result = network.solve(_geometric_maps())

    np.testing.assert_allclose(
        result.preamp_voltage_noise_covariance_v2,
        5.0 * voltage_density,
    )
    expected_current = (
        5.0
        * result.output_impedance_ohm
        @ current_density
        @ result.output_impedance_ohm.conj().T
    )
    np.testing.assert_allclose(
        result.preamp_current_noise_covariance_v2,
        expected_current,
    )
    np.testing.assert_allclose(
        result.noise_covariance_v2,
        result.passive_noise_covariance_v2
        + result.preamp_voltage_noise_covariance_v2
        + result.preamp_current_noise_covariance_v2,
    )


def test_active_lna_excludes_input_resistance_johnson_noise() -> None:
    model = LNAInputModel(
        input_resistance_ohm=75.0,
        voltage_gain_v_per_v=2.0,
    )
    network = ActiveReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array([[25.0]]),
        lna_input_models=model,
        temperature_k=300.0,
        noise_bandwidth_hz=2.0,
    )
    result = network.solve(np.ones((1, 2), dtype=np.complex128))

    input_transfer = 75.0 / 100.0
    expected = (
        2.0**2
        * input_transfer**2
        * 4.0
        * BOLTZMANN_J_PER_K
        * 300.0
        * 25.0
        * 2.0
    )
    np.testing.assert_allclose(result.input_transfer_matrix, [[input_transfer]])
    np.testing.assert_allclose(result.transfer_matrix, [[1.5]])
    np.testing.assert_allclose(result.source_noise_covariance_v2, [[expected]])
    np.testing.assert_allclose(result.noise_covariance_v2, [[expected]])
    np.testing.assert_allclose(result.noise_figure_db, [0.0], atol=1.0e-12)


def test_active_lna_voltage_current_cross_and_downstream_noise() -> None:
    model = LNAInputModel(
        input_resistance_ohm=80.0,
        voltage_noise_density_v_per_sqrt_hz=1.0e-9,
        current_noise_density_a_per_sqrt_hz=2.0e-12,
        voltage_current_noise_correlation=0.25 + 0.1j,
        voltage_gain_v_per_v=10.0,
        output_noise_density_v_per_sqrt_hz=3.0e-9,
    )
    bandwidth = 5.0
    network = ActiveReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array([[20.0]]),
        lna_input_models=model,
        temperature_k=300.0,
        noise_bandwidth_hz=bandwidth,
    )
    result = network.solve(np.ones((1, 1), dtype=np.complex128))

    node_impedance = 20.0 * 80.0 / (20.0 + 80.0)
    gain_squared = 10.0**2
    voltage = gain_squared * bandwidth * (1.0e-9) ** 2
    current = (
        gain_squared * bandwidth * node_impedance**2 * (2.0e-12) ** 2
    )
    cross_spectrum = (0.25 + 0.1j) * 1.0e-9 * 2.0e-12
    cross = (
        gain_squared
        * bandwidth
        * 2.0
        * np.real(cross_spectrum * node_impedance)
    )
    downstream = bandwidth * (3.0e-9) ** 2
    np.testing.assert_allclose(result.input_node_impedance_ohm, [[node_impedance]])
    np.testing.assert_allclose(result.lna_voltage_noise_covariance_v2, [[voltage]])
    np.testing.assert_allclose(result.lna_current_noise_covariance_v2, [[current]])
    np.testing.assert_allclose(result.lna_cross_noise_covariance_v2, [[cross]])
    np.testing.assert_allclose(result.downstream_noise_covariance_v2, [[downstream]])
    np.testing.assert_allclose(
        result.noise_covariance_v2,
        result.source_noise_covariance_v2
        + voltage
        + current
        + cross
        + downstream,
    )


def test_active_lna_multichannel_maps_covariance_and_model_count() -> None:
    passive = _coupled_network(bandwidth_hz=10.0)
    models = (
        LNAInputModel(
            input_resistance_ohm=50.0,
            voltage_noise_density_v_per_sqrt_hz=0.5e-9,
            current_noise_density_a_per_sqrt_hz=1.0e-12,
            voltage_gain_v_per_v=4.0,
        ),
        LNAInputModel(
            input_resistance_ohm=1.0e5,
            input_capacitance_f=2.0e-12,
            voltage_noise_density_v_per_sqrt_hz=0.8e-9,
            current_noise_density_a_per_sqrt_hz=5.0e-15,
            voltage_current_noise_correlation=-0.2j,
            voltage_gain_v_per_v=3.0,
        ),
    )
    network = ActiveReceiverNetwork(
        frequency_hz=passive.frequency_hz,
        coil_impedance_ohm=passive.coil_impedance_ohm,
        series_impedance_ohm=passive.series_impedance_ohm,
        lna_input_models=models,
        noise_bandwidth_hz=passive.noise_bandwidth_hz,
    )
    result = network.solve(_geometric_maps())

    assert result.effective_sensitivities.shape == _geometric_maps().shape
    assert result.noise_covariance_v2.shape == (2, 2)
    np.testing.assert_allclose(
        result.noise_covariance_v2,
        result.noise_covariance_v2.conj().T,
    )
    assert np.min(np.linalg.eigvalsh(result.noise_covariance_v2)) >= -1.0e-30
    snr = optimal_channel_snr(
        result.effective_sensitivities,
        result.noise_covariance_v2,
    )
    assert isinstance(snr, np.ndarray)
    assert snr.shape == _geometric_maps().shape[1:]
    assert np.all(snr > 0.0)

    with pytest.raises(ValueError, match="must contain 2"):
        ActiveReceiverNetwork(
            frequency_hz=1.0e6,
            coil_impedance_ohm=np.eye(2),
            lna_input_models=(models[0],),
        )

def test_lna_parallel_input_capacitance_and_validation() -> None:
    model = LNAInputModel(
        input_resistance_ohm=1.0e6,
        input_capacitance_f=5.0e-12,
    )
    frequency = 10.0e6
    expected = 1.0 / (1.0 / 1.0e6 + 1j * 2.0 * np.pi * frequency * 5.0e-12)
    assert model.input_impedance_ohm(frequency) == pytest.approx(expected)

    with pytest.raises(ValueError, match="magnitude <= 1"):
        LNAInputModel(
            input_resistance_ohm=50.0,
            voltage_current_noise_correlation=1.01,
        )
    with pytest.raises(ValueError, match="finite and positive"):
        LNAInputModel(input_resistance_ohm=0.0)


def test_optimal_channel_snr_matches_quadratic_form() -> None:
    signal = np.array([1.0 + 0.5j, -0.25j])
    covariance = np.array([[2.0, 0.25j], [-0.25j, 1.0]])
    expected = np.sqrt(
        np.real(signal.conj() @ np.linalg.solve(covariance, signal))
    )
    assert optimal_channel_snr(signal, covariance) == pytest.approx(expected)


def test_noise_scaling_preserves_correlation_shape() -> None:
    covariance = _coupled_network().solve(_geometric_maps()).noise_covariance_v2
    scaled = scale_noise_covariance(covariance, 0.03)

    assert np.mean(np.real(np.diag(scaled))) == pytest.approx(0.03**2)
    np.testing.assert_allclose(
        covariance_to_correlation(scaled),
        covariance_to_correlation(covariance),
    )


def test_two_loop_resonance_splitting_matches_analytic_modes() -> None:
    inductance = 10.0e-6
    mutual = 2.0e-6
    capacitance = 100.0e-12
    frequencies, modes = coupled_resonant_modes(
        np.array([[inductance, mutual], [mutual, inductance]]),
        capacitance,
    )
    expected = np.sort(
        1.0
        / (
            2.0
            * np.pi
            * np.sqrt(capacitance * np.array([inductance + mutual, inductance - mutual]))
        )
    )

    np.testing.assert_allclose(frequencies, expected)
    np.testing.assert_allclose(np.abs(modes[0]), np.abs(modes[1]))
    assert np.sign(modes[0, 0]) == np.sign(modes[1, 0])
    assert np.sign(modes[0, 1]) != np.sign(modes[1, 1])


def _analytic_cancellation_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    target_frequency = 2.0e6
    frequencies = np.array([1.8e6, target_frequency, 2.2e6])
    omega = 2.0 * np.pi * frequencies
    inductance = 10.0e-6
    mutual = 2.0e-6
    resistance = np.array([[2.0, 0.2], [0.2, 2.0]])
    inductance_matrix = np.array(
        [[inductance, mutual], [mutual, inductance]]
    )
    coil = (
        resistance[np.newaxis, :, :]
        + 1j * omega[:, np.newaxis, np.newaxis] * inductance_matrix
    )

    baseline_capacitance = 1.0 / (
        (2.0 * np.pi * target_frequency) ** 2 * inductance
    )
    baseline_tuning = np.array(
        [
            np.diag([1.0 / (1j * value * baseline_capacitance)] * 2)
            for value in omega
        ]
    )
    branch_capacitance = mutual_cancellation_capacitance(
        mutual,
        target_frequency,
    )
    branch = shared_capacitor_mesh_impedance(
        frequencies,
        branch_capacitance,
    )
    cancelled_capacitance = 1.0 / (
        (2.0 * np.pi * target_frequency) ** 2 * (inductance - mutual)
    )
    cancelled_tuning = np.array(
        [
            np.diag([1.0 / (1j * value * cancelled_capacitance)] * 2)
            for value in omega
        ]
    )
    return (
        frequencies,
        coil + baseline_tuning,
        coil + cancelled_tuning + branch,
        branch_capacitance,
    )


def test_shared_capacitor_cancels_mutual_reactance_but_not_resistance() -> None:
    frequencies, before, after, capacitance = _analytic_cancellation_case()
    target = 1

    assert capacitance == pytest.approx(
        1.0 / ((2.0 * np.pi * frequencies[target]) ** 2 * 2.0e-6)
    )
    assert np.imag(after[target, 0, 1]) == pytest.approx(0.0, abs=1e-14)
    assert np.real(after[target, 0, 1]) == pytest.approx(0.2)
    assert abs(after[0, 0, 1]) > abs(after[target, 0, 1])
    np.testing.assert_allclose(before, before.transpose(0, 2, 1))
    np.testing.assert_allclose(after, after.transpose(0, 2, 1))


def test_shared_capacitor_mesh_orientation_and_loss_are_physical() -> None:
    frequency = np.array([2.0e6])
    same = shared_capacitor_mesh_impedance(
        frequency,
        3.0e-9,
        n_ports=3,
        ports=(0, 2),
        series_resistance_ohm=0.1,
    )
    opposite = shared_capacitor_mesh_impedance(
        frequency,
        3.0e-9,
        n_ports=3,
        ports=(0, 2),
        branch_signs=(1, -1),
        series_resistance_ohm=0.1,
    )

    assert same.shape == (1, 3, 3)
    np.testing.assert_allclose(same, same.transpose(0, 2, 1))
    assert same[0, 0, 2] == -opposite[0, 0, 2]
    assert np.min(np.linalg.eigvalsh(np.real(same[0]))) >= -1e-14


@pytest.mark.smoke
def test_receiver_coupling_sweep_matches_single_frequency_network() -> None:
    frequencies, before, after, _ = _analytic_cancellation_case()
    result = analyze_receiver_coupling_sweep(
        frequencies,
        before,
        after,
        load_impedance_ohm=50.0,
        temperature_k=300.0,
        noise_bandwidth_hz=2.0,
    )
    target = 1
    network = ReceiverNetwork(
        frequency_hz=frequencies[target],
        coil_impedance_ohm=after[target],
        load_impedance_ohm=50.0,
        temperature_k=300.0,
        noise_bandwidth_hz=2.0,
    )
    single = network.solve(np.ones((2, 1), dtype=np.complex128))

    np.testing.assert_allclose(
        result.transfer_matrix_after[target],
        single.transfer_matrix,
    )
    np.testing.assert_allclose(
        result.passive_noise_covariance_after_v2[target],
        single.passive_noise_covariance_v2,
    )
    assert result.isolation_improvement_db[target] > 40.0
    assert abs(result.noise_correlation_after[target, 0, 1]) > 0.0


def test_receiver_cancellation_helpers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        mutual_cancellation_capacitance(0.0, 1.0e6)
    with pytest.raises(ValueError, match="branch_signs"):
        shared_capacitor_mesh_impedance(
            [1.0e6],
            1.0e-9,
            branch_signs=(1, 0),
        )
    with pytest.raises(ValueError, match="distinct valid"):
        shared_capacitor_mesh_impedance(
            [1.0e6],
            1.0e-9,
            ports=(0, 0),
        )
    with pytest.raises(ValueError, match="distinct valid"):
        shared_capacitor_mesh_impedance(
            [1.0e6],
            1.0e-9,
            ports=(0.0, 1.0),
        )
    with pytest.raises(ValueError, match="distinct valid"):
        analyze_receiver_coupling_sweep(
            [1.0e6],
            np.eye(2),
            np.eye(2),
            drive_port=0,
            victim_port=0,
        )


def _parallel_wires() -> tuple[Conductor, Conductor]:
    kwargs = dict(wire_radius=5.0e-4, n_radial=1, n_angular=4)
    first = Conductor(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        **kwargs,
    )
    second = Conductor(
        np.array([[0.0, 0.02, 0.0], [0.1, 0.02, 0.0]]),
        **kwargs,
    )
    return first, second


def test_multiport_peec_single_port_matches_scalar_extraction() -> None:
    conductor, _ = _parallel_wires()
    frequency = 1.0e6
    scalar = extract_impedance(conductor, [frequency])
    multiport = extract_multiport_impedance((conductor,), [frequency])

    expected = scalar.resistance[0] + 1j * 2.0 * np.pi * frequency * scalar.inductance[0]
    np.testing.assert_allclose(multiport.impedance[0, 0, 0], expected)
    np.testing.assert_allclose(
        multiport.dc_inductance[0, 0],
        scalar.dc_inductance,
    )
    np.testing.assert_allclose(
        multiport.dc_resistance[0, 0],
        scalar.dc_resistance,
    )


def test_multiport_peec_is_reciprocal_passive_and_coupled() -> None:
    result = extract_multiport_impedance(_parallel_wires(), [1.0e6])
    impedance = result.impedance[0]

    np.testing.assert_allclose(impedance, impedance.T)
    assert np.min(np.linalg.eigvalsh(result.resistance[0])) >= -1e-12
    assert abs(result.inductance[0, 0, 1]) > 0.0
    np.testing.assert_allclose(
        result.dc_inductance,
        result.dc_inductance.T,
    )


def test_replaced_conductor_does_not_reuse_cached_geometry() -> None:
    first, _ = _parallel_wires()
    first.subfilaments()
    translated = dataclasses.replace(
        first,
        path_points=first.path_points + np.array([0.0, 0.05, 0.0]),
    )

    first_start = first.subfilaments()[0][0][0][0]
    translated_start = translated.subfilaments()[0][0][0][0]
    np.testing.assert_allclose(
        translated_start - first_start,
        [0.0, 0.05, 0.0],
    )
    assert translated._cache is not first._cache


def test_receiver_workflow_applies_network_and_scaled_correlated_noise() -> None:
    network = _coupled_network()
    maps = _geometric_maps()
    expected = network.solve(maps)
    result = run_ideal_receiver_array_cpmg_imaging(
        np.ones((2, 2)),
        receiver_sensitivities=maps,
        receiver_network=network,
        num_echoes=1,
        ny=1,
        maxoffs=0.1,
        sense_acceleration=2,
        sense_axis=0,
        noise_std=0.02,
        noise_seed=7,
    )

    np.testing.assert_allclose(
        result.geometric_receiver_sensitivities,
        maps,
    )
    np.testing.assert_allclose(
        result.receiver_sensitivities,
        expected.effective_sensitivities,
    )
    np.testing.assert_allclose(
        result.receiver_transfer_matrix,
        expected.transfer_matrix,
    )
    np.testing.assert_allclose(
        result.receiver_network_noise_covariance_v2,
        expected.noise_covariance_v2,
    )
    assert np.mean(np.real(np.diag(result.noise_covariance))) == pytest.approx(
        0.02**2
    )
    assert result.channel_kspace_noisy is not None
    assert result.sense_image is not None
    assert result.sense_rank is not None
    assert np.all(result.sense_rank == 2)

    with pytest.raises(ValueError, match="derives channel covariance"):
        run_ideal_receiver_array_cpmg_imaging(
            np.ones((2, 2)),
            receiver_sensitivities=maps,
            receiver_network=network,
            noise_covariance=np.eye(2),
            num_echoes=1,
            ny=1,
            maxoffs=0.1,
        )


@pytest.mark.smoke
def test_receiver_network_experiment_round_trip_plan_and_run() -> None:
    network = _coupled_network()
    receivers = RxArray(
        (
            RxCoil(
                SolenoidCoil(
                    radius_m=0.015,
                    length_m=0.03,
                    turns=8,
                    axis="x",
                )
            ),
            RxCoil(
                SolenoidCoil(
                    radius_m=0.015,
                    length_m=0.03,
                    turns=8,
                    axis="y",
                )
            ),
        ),
        network=network,
    )
    experiment = Experiment(
        sequence=CPMGImaging(num_echoes=1, ny=1, maxoffs=0.1),
        sample=Sample(phantom=Phantom(np.ones((2, 2)))),
        hardware=Hardware(
            rx_coil=receivers,
            plane=ImagingPlane(plane="xy"),
        ),
        acquisition=Acquisition(
            receiver_noise_std=0.01,
            receiver_noise_seed=4,
        ),
    )

    assert Experiment.from_json(experiment.to_json()) == experiment
    assert experiment.plan(estimate=False).ok
    result = experiment.run().result
    assert result.receiver_transfer_matrix.shape == (2, 2)
    assert result.receiver_network_noise_covariance_v2.shape == (2, 2)
    assert result.geometric_receiver_sensitivities.shape == (2, 2, 2)
    assert result.channel_kspace_noisy is not None

    invalid = dataclasses.replace(
        experiment,
        acquisition=Acquisition(receiver_noise_covariance=np.eye(2)),
    )
    plan = invalid.plan(estimate=False)
    assert not plan.ok
    assert any("derives receiver noise covariance" in error for error in plan.errors)


def test_receiver_network_validation_rejects_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="reciprocal"):
        ReceiverNetwork(
            1.0e6,
            np.array([[1.0, 1.0], [0.0, 1.0]]),
            50.0,
        )
    with pytest.raises(ValueError, match="passive"):
        ReceiverNetwork(
            1.0e6,
            np.array([[-1.0 + 1.0j]]),
            50.0,
        )
    with pytest.raises(ValueError, match="port count"):
        RxArray(
            (
                RxCoil(SolenoidCoil(0.01, 0.02, 2, axis="x")),
                RxCoil(SolenoidCoil(0.01, 0.02, 2, axis="y")),
            ),
            network=ReceiverNetwork(1.0e6, np.array([[1.0 + 1.0j]]), 50.0),
        )
