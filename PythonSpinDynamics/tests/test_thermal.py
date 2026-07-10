"""Tests for spin_dynamics.thermal (Phases 0-1: materials, sources, network)."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.fields import coils
from spin_dynamics.fields.quasistatic import eddy_currents, reflected_resistance
from spin_dynamics.parameters import set_params_tuned_orig
from spin_dynamics.thermal import (
    AIR,
    COPPER_THERMAL,
    MUSCLE_TISSUE,
    STEFAN_BOLTZMANN,
    ThermalLink,
    ThermalMaterial,
    ThermalNetwork,
    ThermalNode,
    WATER_THERMAL,
    ConstantSource,
    DutyCycledSource,
    average_coil_power,
    coil_joule_power,
    conduction_conductance,
    convection_conductance,
    cylindrical_shell_conductance,
    duty_cycle_from_pulse_params,
    gradient_waveform_power,
    radiation_link,
    sar_power_from_loading,
    sar_source_from_eddy,
    transmit_coil_current,
)


class TestThermalMaterials:
    def test_diffusivity_and_capacity(self) -> None:
        m = ThermalMaterial("test", conductivity=10.0, density=2000.0, specific_heat=500.0)
        assert m.volumetric_heat_capacity == pytest.approx(1e6)
        assert m.diffusivity == pytest.approx(1e-5)
        assert m.heat_capacity(2e-6) == pytest.approx(2.0)

    def test_presets_are_physical(self) -> None:
        # Copper conducts ~3 orders better than tissue; water stores the most
        # heat per volume of the common presets; air diffuses fastest of the
        # insulators (low volumetric capacity).
        assert COPPER_THERMAL.conductivity / MUSCLE_TISSUE.conductivity > 500
        assert WATER_THERMAL.volumetric_heat_capacity > MUSCLE_TISSUE.volumetric_heat_capacity
        assert AIR.diffusivity > WATER_THERMAL.diffusivity

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            ThermalMaterial("bad", conductivity=-1.0, density=1.0, specific_heat=1.0)
        with pytest.raises(ValueError):
            ThermalMaterial("bad", conductivity=1.0, density=1.0, specific_heat=1.0, emissivity=2.0)
        with pytest.raises(ValueError):
            COPPER_THERMAL.heat_capacity(-1.0)


class TestSources:
    def test_source_average_power(self) -> None:
        assert ConstantSource("x", 2.0).average_power == 2.0
        pulsed = DutyCycledSource("x", peak_power=100.0, duty_cycle=0.25)
        assert pulsed.average_power == pytest.approx(25.0)
        with pytest.raises(ValueError):
            DutyCycledSource("x", peak_power=1.0, duty_cycle=1.5)
        with pytest.raises(ValueError):
            ConstantSource("x", -1.0)

    def test_coil_joule_power_amplitude_vs_rms(self) -> None:
        # I_rms = I_peak / sqrt(2) must give the same dissipation.
        p_peak = coil_joule_power(2.0, 5.0)
        p_rms = coil_joule_power(2.0 / np.sqrt(2.0), 5.0, rms=True)
        assert p_peak == pytest.approx(10.0)
        assert p_rms == pytest.approx(p_peak)

    def test_transmit_coil_current(self) -> None:
        # Rotating-frame B1 needs twice the linear field: I = 2 B1 / B1_hat.
        assert transmit_coil_current(1e-3, 5e-4) == pytest.approx(4.0)

    def test_duty_cycle_from_tuned_cpmg_params(self) -> None:
        _params, _sp, pp = set_params_tuned_orig(numpts=9)
        # tref = [75us, 50us, 75us], aref = [0, 1, 0] -> duty = 50/200.
        assert duty_cycle_from_pulse_params(pp) == pytest.approx(0.25)

    def test_energy_bookkeeping(self) -> None:
        # Average power x repetition time == pulse energy x pulses. One 180
        # pulse of duration t_p per cycle of duration t_cycle.
        current, resistance = 3.0, 2.0
        t_p, t_cycle = 50e-6, 200e-6
        source = average_coil_power(current, resistance, t_p / t_cycle)
        pulse_energy = coil_joule_power(current, resistance) * t_p
        assert source.average_power * t_cycle == pytest.approx(pulse_energy)

    def test_gradient_waveform_power(self) -> None:
        t = np.linspace(0.0, 1e-3, 2001)
        const = gradient_waveform_power(np.full(t.size, 2.0), t, 5.0)
        assert const.power == pytest.approx(20.0)  # I^2 R, no sinusoidal 1/2
        # Sine waveform: mean square = I0^2 / 2 over whole periods.
        i = 2.0 * np.sin(2 * np.pi * 5e3 * t)
        sine = gradient_waveform_power(i, t, 5.0)
        assert sine.power == pytest.approx(10.0, rel=1e-3)
        # Duty dilution by repetition time.
        diluted = gradient_waveform_power(np.full(t.size, 2.0), t, 5.0, repetition_time=4e-3)
        assert diluted.power == pytest.approx(5.0)
        with pytest.raises(ValueError):
            gradient_waveform_power(np.full(t.size, 2.0), t, 5.0, repetition_time=1e-4)


@pytest.fixture(scope="module")
def eddy_setup():
    coil = coils.solenoid(radius=0.05, length=0.3, turns=200, axis="z", n_segments=48)
    nx = 13
    ax = np.linspace(-0.02, 0.02, nx)
    dx = ax[1] - ax[0]
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    mask = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2) <= 0.015
    return coil, grid, mask, (dx, dx, dx)


class TestSARSources:
    def test_sar_density_integrates_to_total_power(self, eddy_setup) -> None:
        coil, grid, mask, spacing = eddy_setup
        res = eddy_currents(
            grid, coil, dcurrent_dt=1e4, conductivity=1e2, mask=mask,
            spacing=spacing, charge_correction=True,
        )
        source, density = sar_source_from_eddy(res, duty_cycle=1.0)
        cell_volume = float(np.prod(spacing))
        assert float(np.sum(density)) * cell_volume == pytest.approx(res.power, rel=1e-9)
        assert source.average_power == pytest.approx(res.power, rel=1e-12)
        # Explicit-conductivity path agrees with the J.E path.
        source2, density2 = sar_source_from_eddy(res, duty_cycle=1.0, conductivity=1e2)
        np.testing.assert_allclose(density2, density, rtol=1e-12)
        assert source2.average_power == source.average_power

    def test_duty_cycle_scales_sar(self, eddy_setup) -> None:
        coil, grid, mask, spacing = eddy_setup
        res = eddy_currents(
            grid, coil, dcurrent_dt=1e4, conductivity=1e2, mask=mask,
            spacing=spacing, charge_correction=True,
        )
        full, density_full = sar_source_from_eddy(res, duty_cycle=1.0)
        quarter, density_quarter = sar_source_from_eddy(res, duty_cycle=0.25)
        assert quarter.average_power == pytest.approx(0.25 * full.average_power)
        np.testing.assert_allclose(density_quarter, 0.25 * density_full, rtol=1e-12)

    def test_volume_sar_matches_circuit_loading(self, eddy_setup) -> None:
        # Fluctuation of nothing here -- pure bookkeeping identity: the
        # volume-integrated eddy power at dI/dt = omega * I0 must equal the
        # circuit-side P = I0^2 R_reflected / 2 (same correction flag).
        coil, grid, mask, spacing = eddy_setup
        frequency = 1e6
        omega = 2 * np.pi * frequency
        i0 = 2.0
        sigma = 10.0
        res = eddy_currents(
            grid, coil, dcurrent_dt=omega * i0, conductivity=sigma, mask=mask,
            spacing=spacing, charge_correction=False,
        )
        r_reflected = reflected_resistance(
            grid, coil, frequency=frequency, conductivity=sigma, mask=mask, spacing=spacing
        )
        circuit = sar_power_from_loading(i0, r_reflected)
        volume_source, _ = sar_source_from_eddy(res)
        assert circuit.average_power == pytest.approx(volume_source.average_power, rel=1e-9)


class TestConductanceHelpers:
    def test_formulas(self) -> None:
        assert conduction_conductance(10.0, 2e-4, 1e-2) == pytest.approx(0.2)
        assert convection_conductance(25.0, 4e-3) == pytest.approx(0.1)
        shell = cylindrical_shell_conductance(0.25, 0.05, 5e-3, 10e-3)
        assert shell == pytest.approx(2 * np.pi * 0.25 * 0.05 / np.log(2.0))
        with pytest.raises(ValueError):
            cylindrical_shell_conductance(0.25, 0.05, 10e-3, 5e-3)

    def test_radiation_link_coefficient(self) -> None:
        link = radiation_link("a", "b", emissivity=0.5, area=2e-3)
        assert link.radiation_coefficient == pytest.approx(0.5 * STEFAN_BOLTZMANN * 2e-3)
        with pytest.raises(ValueError):
            radiation_link("a", "b", emissivity=0.0, area=1.0)


class TestThermalNetwork:
    def _single_node(self, *, power: float = 2.0, g: float = 0.1, c: float = 50.0):
        nodes = [
            ThermalNode("coil", heat_capacity=c, initial_temperature=293.15),
            ThermalNode("ambient", heat_capacity=None, initial_temperature=293.15),
        ]
        links = [ThermalLink("coil", "ambient", conductance=g)]
        return ThermalNetwork(nodes, links, sources={"coil": power})

    def test_single_node_steady_state(self) -> None:
        net = self._single_node(power=2.0, g=0.1)
        steady = net.steady_state()
        assert steady["coil"] == pytest.approx(293.15 + 20.0)
        assert steady["ambient"] == pytest.approx(293.15)

    def test_single_node_exponential_transient(self) -> None:
        power, g, c = 2.0, 0.1, 50.0
        tau = c / g
        net = self._single_node(power=power, g=g, c=c)
        times = np.linspace(0.0, 3.0 * tau, 121)
        result = net.transient(times)
        expected = 293.15 + (power / g) * (1.0 - np.exp(-times / tau))
        np.testing.assert_allclose(result.temperatures["coil"], expected, rtol=1e-6, atol=1e-6)

    def test_rk4_fallback_matches_analytic(self) -> None:
        power, g, c = 2.0, 0.1, 50.0
        tau = c / g
        net = self._single_node(power=power, g=g, c=c)
        times = np.linspace(0.0, 2.0 * tau, 41)
        caps = np.array([c, np.inf])
        trajectory = net._transient_rk4(times, caps, None)
        expected = 293.15 + (power / g) * (1.0 - np.exp(-times / tau))
        np.testing.assert_allclose(trajectory[:, 0], expected, rtol=1e-6)

    def test_two_node_chain_temperature_drops(self) -> None:
        # bath <-G1- former <-G2- coil with P into the coil: in steady state
        # the full P crosses both links, dropping P/G across each.
        p, g1, g2 = 1.5, 0.3, 0.05
        nodes = [
            ThermalNode("coil", heat_capacity=10.0),
            ThermalNode("former", heat_capacity=40.0),
            ThermalNode("bath", heat_capacity=None, initial_temperature=290.0),
        ]
        links = [
            ThermalLink("coil", "former", conductance=g2),
            ThermalLink("former", "bath", conductance=g1),
        ]
        net = ThermalNetwork(nodes, links, sources={"coil": p})
        steady = net.steady_state()
        assert steady["former"] - steady["bath"] == pytest.approx(p / g1)
        assert steady["coil"] - steady["former"] == pytest.approx(p / g2)

    def test_radiation_only_steady_state(self) -> None:
        coeff_area, emissivity, p = 1e-3, 0.8, 5.0
        nodes = [
            ThermalNode("hot", heat_capacity=1.0, initial_temperature=350.0),
            ThermalNode("walls", heat_capacity=None, initial_temperature=293.15),
        ]
        links = [radiation_link("hot", "walls", emissivity=emissivity, area=coeff_area)]
        net = ThermalNetwork(nodes, links, sources={"hot": p})
        steady = net.steady_state()
        expected = (p / (emissivity * STEFAN_BOLTZMANN * coeff_area) + 293.15**4) ** 0.25
        assert steady["hot"] == pytest.approx(expected, rel=1e-9)

    def test_adiabatic_node_heats_linearly(self) -> None:
        c, p = 20.0, 4.0
        nodes = [
            ThermalNode("blob", heat_capacity=c, initial_temperature=300.0),
            ThermalNode("bath", heat_capacity=None),
        ]
        # A vanishingly weak link keeps the network well-posed while staying
        # effectively adiabatic over the simulated window.
        links = [ThermalLink("blob", "bath", conductance=1e-12)]
        net = ThermalNetwork(nodes, links, sources={"blob": p})
        times = np.linspace(0.0, 10.0, 21)
        result = net.transient(times)
        np.testing.assert_allclose(
            result.temperatures["blob"], 300.0 + p * times / c, rtol=1e-8
        )

    def test_transient_approaches_steady_state(self) -> None:
        net = self._single_node(power=1.0, g=0.2, c=5.0)
        steady = net.steady_state()
        result = net.transient(np.linspace(0.0, 10.0 * 5.0 / 0.2, 61))
        assert result.final()["coil"] == pytest.approx(steady["coil"], rel=1e-6)

    def test_sources_accept_source_objects(self) -> None:
        pulsed = DutyCycledSource("rf", peak_power=8.0, duty_cycle=0.25)
        extra = ConstantSource("gradient", 1.0)
        nodes = [
            ThermalNode("coil", heat_capacity=10.0),
            ThermalNode("bath", heat_capacity=None),
        ]
        links = [ThermalLink("coil", "bath", conductance=0.5)]
        net = ThermalNetwork(nodes, links, sources={"coil": [pulsed, extra]})
        steady = net.steady_state()
        assert steady["coil"] - 293.15 == pytest.approx((2.0 + 1.0) / 0.5)

    def test_validation_errors(self) -> None:
        nodes = [
            ThermalNode("a", heat_capacity=1.0),
            ThermalNode("bath", heat_capacity=None),
        ]
        links = [ThermalLink("a", "bath", conductance=1.0)]
        with pytest.raises(ValueError):
            ThermalNetwork(nodes, links, sources={"missing": 1.0})
        with pytest.raises(ValueError):
            ThermalNetwork(nodes, [ThermalLink("a", "nope", conductance=1.0)])
        with pytest.raises(ValueError):
            ThermalLink("a", "a", conductance=1.0)
        with pytest.raises(ValueError):
            ThermalLink("a", "b")
