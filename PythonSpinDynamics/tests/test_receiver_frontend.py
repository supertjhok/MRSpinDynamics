"""Passive receiver two-port and preamplifier-decoupling tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.receiver_frontend import (
    PassiveTwoPort,
    ideal_transformer_two_port,
    identity_two_port,
    series_impedance_two_port,
    shunt_admittance_two_port,
    transmission_line_two_port,
)
from spin_dynamics.receiver_network import (
    BOLTZMANN_J_PER_K,
    ActiveReceiverNetwork,
    LNAInputModel,
    analyze_active_receiver_front_end_sweep,
)


def test_two_port_factories_transform_loads_and_cascade() -> None:
    through = identity_two_port()
    assert through.transformed_load_impedance_ohm(50.0) == pytest.approx(50.0)
    np.testing.assert_allclose(abs(through.s_parameters[1, 0]), 1.0)

    transformer = ideal_transformer_two_port(3.0)
    assert transformer.transformed_load_impedance_ohm(10.0) == pytest.approx(90.0)
    series = series_impedance_two_port(5.0 + 2.0j)
    shunt = shunt_admittance_two_port(0.01)
    cascade = series.cascade(shunt)
    np.testing.assert_allclose(cascade.abcd, series.abcd @ shunt.abcd)


def test_quarter_wave_line_transforms_low_lna_impedance_high() -> None:
    line = transmission_line_two_port(
        50.0,
        np.pi / 2.0,
    )
    transformed = line.transformed_load_impedance_ohm(2.0)
    assert transformed == pytest.approx(50.0**2 / 2.0)
    assert line.matched_insertion_loss_db == pytest.approx(0.0, abs=1.0e-12)
    assert line.is_lossless
    assert not transmission_line_two_port(
        50.0,
        np.pi / 2.0,
        attenuation_db=0.1,
    ).is_lossless


def test_lossy_matched_line_obeys_friis_noise_limit() -> None:
    attenuation_db = 1.5
    temperature_k = 300.0
    line = transmission_line_two_port(
        50.0,
        0.7,
        attenuation_db=attenuation_db,
    )
    network = ActiveReceiverNetwork(
        frequency_hz=1.0e6,
        coil_impedance_ohm=np.array([[50.0]]),
        lna_input_models=LNAInputModel(input_resistance_ohm=50.0),
        front_end_two_ports=line,
        temperature_k=temperature_k,
        noise_bandwidth_hz=1.0,
    )
    result = network.solve(np.ones((1, 1), dtype=np.complex128))
    power_transmission = 10.0 ** (-attenuation_db / 10.0)
    matched_noise = BOLTZMANN_J_PER_K * temperature_k * 50.0

    np.testing.assert_allclose(
        result.source_noise_covariance_v2,
        [[matched_noise * power_transmission]],
    )
    np.testing.assert_allclose(
        result.front_end_noise_covariance_v2,
        [[matched_noise * (1.0 - power_transmission)]],
    )
    np.testing.assert_allclose(result.noise_covariance_v2, [[matched_noise]])
    np.testing.assert_allclose(result.noise_figure_db, [attenuation_db])


def test_front_end_sweep_reports_preamplifier_decoupling() -> None:
    frequencies = np.array([0.9e6, 1.0e6, 1.1e6])
    source = np.broadcast_to(
        np.array([[5.0, 1.0 + 8.0j], [1.0 + 8.0j, 5.0]]),
        (frequencies.size, 2, 2),
    ).copy()
    low_impedance_lna = LNAInputModel(
        input_resistance_ohm=2.0,
        voltage_noise_density_v_per_sqrt_hz=0.5e-9,
        current_noise_density_a_per_sqrt_hz=2.0e-12,
    )
    front_ends = tuple(
        (
            transmission_line_two_port(
                50.0,
                np.pi / 2.0 * frequency / 1.0e6,
                attenuation_db=0.2,
            ),
        )
        * 2
        for frequency in frequencies
    )
    result = analyze_active_receiver_front_end_sweep(
        frequencies,
        source,
        lna_input_models=low_impedance_lna,
        front_end_two_ports=front_ends,
        noise_bandwidth_hz=10.0e3,
    )

    assert result.source_load_impedance_ohm[1, 0, 0].real > 500.0
    assert result.isolation_db[1] > result.isolation_db[0]
    assert result.isolation_db[1] > result.isolation_db[2]
    assert np.all(np.isfinite(result.noise_figure_db))
    assert np.all(
        np.linalg.eigvalsh(result.front_end_noise_covariance_v2) >= -1.0e-30
    )


def test_two_port_validation_rejects_active_or_nonreciprocal_inputs() -> None:
    with pytest.raises(ValueError, match="passive"):
        series_impedance_two_port(-1.0)
    with pytest.raises(ValueError, match="determinant one"):
        PassiveTwoPort(np.array([[1.0, 0.0], [0.0, 2.0]]))
