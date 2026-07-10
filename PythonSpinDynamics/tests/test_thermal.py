"""Tests for spin_dynamics.thermal (Phases 0-1: materials, sources, network)."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.fields import coils
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER, ConductorMaterial
from spin_dynamics.fields.quasistatic import eddy_currents, reflected_resistance
from spin_dynamics.noise import tuned_probe_output_noise_density
from spin_dynamics.parameters import set_params_tuned_orig
from spin_dynamics.radiation_damping import KB
from spin_dynamics.sample import sphere_geometry, water_sample
from spin_dynamics.spin_noise import SampleCoupling
from spin_dynamics.thermal import (
    Conduction1D,
    PerfusionModel,
    CoupledCoilDrive,
    CoupledSAR,
    ThermalCoupling,
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


class TestThermalCoupling:
    T_BATH = 293.15

    def _factory(self, g: float = 0.05, c: float = 20.0, t_bath: float | None = None):
        bath = self.T_BATH if t_bath is None else t_bath

        def make(sources):
            nodes = [
                ThermalNode("coil", heat_capacity=c, initial_temperature=bath),
                ThermalNode("bath", heat_capacity=None, initial_temperature=bath),
            ]
            links = [ThermalLink("coil", "bath", conductance=g)]
            return ThermalNetwork(nodes, links, sources)

        return make

    def _drive(self, *, current=1.0, duty=0.25, r_ref=2.0, exponent=1.0):
        return CoupledCoilDrive(
            node="coil",
            current=current,
            duty_cycle=duty,
            material=ANNEALED_COPPER,
            reference_resistance=r_ref,
            reference_temperature=self.T_BATH,
            resistance_exponent=exponent,
        )

    def test_fixed_point_matches_closed_form(self) -> None:
        # Linear tempco, exponent 1: P(T) = P0 (1 + alpha (T - T_bath)) and
        # T - T_bath = P(T)/G has the closed form (P0/G) / (1 - P0 alpha / G).
        g = 0.05
        drive = self._drive(current=1.0, duty=0.25, r_ref=2.0, exponent=1.0)
        coupling = ThermalCoupling(self._factory(g=g), drive)
        result = coupling.fixed_point()
        p0 = 0.5 * 1.0**2 * 2.0 * 0.25
        alpha = ANNEALED_COPPER.temp_coefficient
        expected_rise = (p0 / g) / (1.0 - p0 * alpha / g)
        assert result.temperatures["coil"] - self.T_BATH == pytest.approx(
            expected_rise, rel=1e-7
        )
        assert result.coil_resistance == pytest.approx(
            2.0 * (1.0 + alpha * expected_rise), rel=1e-7
        )
        assert result.coil_power == pytest.approx(
            g * expected_rise, rel=1e-7
        )  # steady state: input equals loss

    def test_thermal_runaway_raises(self) -> None:
        # Feedback gain P0 * alpha / G > 1: no steady state exists.
        hot = ConductorMaterial("runaway", 17e-9, temp_coefficient=0.5)
        drive = CoupledCoilDrive(
            node="coil",
            current=2.0,
            duty_cycle=1.0,
            material=hot,
            reference_resistance=2.0,
            reference_temperature=self.T_BATH,
            resistance_exponent=1.0,
        )
        coupling = ThermalCoupling(self._factory(g=0.05), drive)
        with pytest.raises(RuntimeError, match="runaway|converge"):
            coupling.fixed_point(max_iterations=50)

    def test_march_approaches_fixed_point(self) -> None:
        g, c = 0.05, 20.0
        drive = self._drive()
        coupling = ThermalCoupling(self._factory(g=g, c=c), drive)
        steady = coupling.fixed_point()
        tau = c / g
        times = np.linspace(0.0, 12.0 * tau, 241)
        marched = coupling.march(times, update_every=4)
        assert marched.transient is not None
        coil_t = marched.transient.temperatures["coil"]
        assert np.all(np.diff(coil_t) > -1e-9)  # monotone warm-up
        assert marched.temperatures["coil"] == pytest.approx(
            steady.temperatures["coil"], rel=1e-4
        )

    def test_sample_and_sar_coupling(self) -> None:
        def make(sources):
            nodes = [
                ThermalNode("coil", heat_capacity=20.0, initial_temperature=self.T_BATH),
                ThermalNode("sample", heat_capacity=5.0, initial_temperature=self.T_BATH),
                ThermalNode("bath", heat_capacity=None, initial_temperature=self.T_BATH),
            ]
            links = [
                ThermalLink("coil", "bath", conductance=0.1),
                ThermalLink("coil", "sample", conductance=0.02),
                ThermalLink("sample", "bath", conductance=0.05),
            ]
            return ThermalNetwork(nodes, links, sources)

        sample = water_sample(sphere_geometry(2.5e-3), temperature=self.T_BATH, t2=0.1)
        sar = CoupledSAR(
            node="sample",
            reference_power=0.05,
            reference_temperature=self.T_BATH,
            tempco=0.02,
        )
        coupling = ThermalCoupling(
            make,
            self._drive(current=2.0, duty=0.25, r_ref=1.0, exponent=0.5),
            sar=sar,
            sample=sample,
            sample_node="sample",
        )
        result = coupling.fixed_point()
        assert result.temperatures["sample"] > self.T_BATH
        assert result.sample is not None
        assert result.sample.temperature == pytest.approx(result.temperatures["sample"])
        # Warmer sample -> lower Curie-law magnetization.
        assert result.sample.magnetization_density(1.0) < sample.magnetization_density(1.0)
        # SAR grew with its positive tempco.
        assert result.sample_power > 0.05

    def test_probe_updates_expose_temperature_r_and_q(self) -> None:
        drive = self._drive()
        coupling = ThermalCoupling(self._factory(), drive)
        result = coupling.fixed_point()
        w0, inductance = 2 * np.pi * 1e6, 10e-6
        updates = result.probe_updates(coil_node="coil", inductance=inductance, omega0=w0)
        assert updates["T"] == pytest.approx(result.temperatures["coil"])
        assert updates["R"] == pytest.approx(result.coil_resistance)
        assert updates["Q"] == pytest.approx(w0 * inductance / result.coil_resistance)

    def test_cryoprobe_two_temperature_noise(self) -> None:
        # Coil cooled to a 77 K bath, sample held near 300 K: the coupled
        # temperatures feed the two-temperature noise model and the cold coil
        # must beat the room-temperature coil on baseline noise density.
        cryo_copper = ConductorMaterial(
            "rrr copper", 17.241e-9, temp_coefficient=3.93e-3,
            residual_resistivity_ratio=50.0,
        )

        def make(sources):
            nodes = [
                ThermalNode("coil", heat_capacity=5.0, initial_temperature=77.0),
                ThermalNode("sample", heat_capacity=5.0, initial_temperature=300.0),
                ThermalNode("cryostat", heat_capacity=None, initial_temperature=77.0),
                ThermalNode("room", heat_capacity=None, initial_temperature=300.0),
            ]
            links = [
                ThermalLink("coil", "cryostat", conductance=0.5),
                ThermalLink("sample", "room", conductance=0.5),
                ThermalLink("coil", "sample", conductance=0.002),  # weak leak
            ]
            return ThermalNetwork(nodes, links, sources)

        drive = CoupledCoilDrive(
            node="coil",
            current=0.5,
            duty_cycle=0.1,
            material=cryo_copper,
            reference_resistance=1.0,
            reference_temperature=293.15,
            resistance_exponent=0.5,
        )
        sample = water_sample(sphere_geometry(2.5e-3), temperature=300.0, t2=0.05)
        coupling = ThermalCoupling(
            make, drive, sample=sample, sample_node="sample",
        )
        result = coupling.fixed_point()
        t_coil = result.temperatures["coil"]
        assert t_coil < 100.0  # coil stays cold
        assert result.temperatures["sample"] > 295.0  # sample stays warm
        assert result.coil_resistance < 1.0  # cryogenic R drop

        # Feed the coupled state into the tuned-probe noise density.
        f0, L = 1e6, 10e-6
        w0 = 2 * np.pi * f0
        sp_cold = {
            "k": KB, "T": t_coil, "L": L, "R": result.coil_resistance,
            "C": 1 / (w0**2 * L), "Cin": 1e-15, "Rin": 1e12, "Rd": 1e12,
            "vn": 0.0, "in_": 0.0, "w0": w0,
            "del_w": np.linspace(-2.0, 2.0, 801),
        }
        sp_warm = dict(sp_cold, T=293.15, R=1.0)
        pp = {"T_90": 25e-6}
        coupling_b = SampleCoupling(r_n0=0.05, t2=0.05, temperature=result.sample.temperature)
        cold, _ = tuned_probe_output_noise_density(sp_cold, pp, sample=coupling_b)
        warm, _ = tuned_probe_output_noise_density(sp_warm, pp, sample=coupling_b)
        edge = 5
        assert cold[edge] < warm[edge]  # cold coil lowers the baseline


class TestConduction1D:
    def test_slab_uniform_source_parabola(self) -> None:
        L, k, q, t0 = 0.02, 0.5, 1e4, 300.0
        n = 201
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=1e6, source=q,
            inner_bc=("temperature", t0), outer_bc=("temperature", t0),
        )
        prof = c.steady_state().temperature
        assert prof.max() - t0 == pytest.approx(q * L**2 / (8 * k), rel=1e-3)

    def test_cylinder_uniform_source_center_rise(self) -> None:
        a, k, q, t0 = 0.015, 0.5, 1e4, 300.0
        n = 200
        r = np.linspace(a / (2 * n), a - a / (2 * n), n)
        c = Conduction1D(
            r, geometry="cylinder", conductivity=k, rho_cp=1e6, source=q,
            inner_bc=("insulated",), outer_bc=("temperature", t0),
        )
        prof = c.steady_state().temperature
        assert prof.max() - t0 == pytest.approx(q * a**2 / (4 * k), rel=1e-4)

    def test_sphere_uniform_source_center_rise(self) -> None:
        a, k, q, t0 = 0.015, 0.5, 1e4, 300.0
        n = 200
        r = np.linspace(a / (2 * n), a - a / (2 * n), n)
        c = Conduction1D(
            r, geometry="sphere", conductivity=k, rho_cp=1e6, source=q,
            inner_bc=("insulated",), outer_bc=("temperature", t0),
        )
        prof = c.steady_state().temperature
        assert prof.max() - t0 == pytest.approx(q * a**2 / (6 * k), rel=1e-3)

    def test_convection_bc_matches_lumped(self) -> None:
        # Thin slab, low Biot: convection BC steady state -> nearly uniform
        # T = T_inf + q*L/(h) (all heat leaves one face; other insulated).
        L, k, q, h, t_inf = 5e-3, 20.0, 2e4, 50.0, 293.0
        n = 101
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=1e6, source=q,
            inner_bc=("insulated",), outer_bc=("convection", h, t_inf),
        )
        prof = c.steady_state().temperature
        # Total flux per area q*L leaves through the film: dT_film = q L / h.
        assert prof.min() - t_inf == pytest.approx(q * L / h, rel=1e-2)

    def test_perfusion_uniform_limit(self) -> None:
        # Insulated slab, uniform source, perfusion sink -> uniform
        # T = T_a + q / (w_b c_b).
        L, k, q = 0.02, 0.5, 1e4
        n = 151
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        perf = PerfusionModel(
            blood_perfusion=2.0, blood_specific_heat=3600.0, arterial_temperature=310.0
        )
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=1e6, source=q, perfusion=perf,
            inner_bc=("insulated",), outer_bc=("insulated",),
        )
        prof = c.steady_state().temperature
        assert prof.mean() == pytest.approx(310.0 + q / (2.0 * 3600.0), rel=1e-6)
        assert prof.max() - prof.min() < 1e-6  # uniform

    def test_perfusion_penetration_depth(self) -> None:
        # With conduction + perfusion and a hot wall, the steady profile decays
        # into the tissue with length sqrt(k / (w_b c_b)).
        k = 0.5
        perf = PerfusionModel(blood_perfusion=4.0, blood_specific_heat=3600.0,
                              arterial_temperature=310.0)
        depth = np.sqrt(k / perf.sink_rate)
        L = 12.0 * depth
        n = 400
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=1e6, source=0.0, perfusion=perf,
            inner_bc=("temperature", 320.0), outer_bc=("temperature", 310.0),
        )
        prof = c.steady_state().temperature
        excess = prof - 310.0
        # At one penetration depth the excess is ~1/e of the wall excess.
        i_depth = np.argmin(np.abs(r - r[0] - depth))
        assert excess[i_depth] / excess[0] == pytest.approx(np.exp(-1.0), rel=0.1)

    def test_transient_relaxes_to_steady_state(self) -> None:
        L, k, q, t0 = 0.02, 0.5, 5e3, 300.0
        n = 81
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        rho_cp = 1e6
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=rho_cp, source=q,
            inner_bc=("temperature", t0), outer_bc=("temperature", t0),
        )
        steady = c.steady_state().temperature
        tau = rho_cp * L**2 / k
        res = c.transient(np.linspace(0.0, 5.0 * tau, 200), initial_temperature=t0)
        np.testing.assert_allclose(res.final(), steady, rtol=1e-3, atol=1e-2)

    def test_transient_conserves_energy_adiabatic(self) -> None:
        # Fully insulated slab with uniform source: mean T rises linearly at
        # q / rho_cp regardless of conduction.
        L, k, q = 0.02, 0.5, 1e4
        n = 61
        r = np.linspace(L / (2 * n), L - L / (2 * n), n)
        rho_cp = 1e6
        c = Conduction1D(
            r, geometry="slab", conductivity=k, rho_cp=rho_cp, source=q,
            inner_bc=("insulated",), outer_bc=("insulated",),
        )
        times = np.linspace(0.0, 100.0, 51)
        res = c.transient(times, initial_temperature=300.0)
        mean_t = res.temperature.mean(axis=1)
        np.testing.assert_allclose(mean_t, 300.0 + q * times / rho_cp, rtol=1e-6)

    def test_validation_errors(self) -> None:
        with pytest.raises(ValueError):
            Conduction1D(np.array([0.0, 1.0]), conductivity=1.0, rho_cp=1.0)  # too few
        with pytest.raises(ValueError):
            Conduction1D(
                np.array([0.0, 1.0, 3.0]), conductivity=1.0, rho_cp=1.0
            )  # non-uniform
        with pytest.raises(ValueError):
            PerfusionModel(blood_perfusion=-1.0)


class TestThermalFlow:
    def test_flow_conductance_value(self) -> None:
        from spin_dynamics.thermal import flow_conductance

        # G = rho c_p Q; water at 1 mL/s.
        g = flow_conductance(997.0, 4184.0, 1e-6)
        assert g == pytest.approx(997.0 * 4184.0 * 1e-6)
        with pytest.raises(ValueError):
            flow_conductance(-1.0, 4184.0, 1e-6)

    def test_lumped_flow_cooling_closed_form(self) -> None:
        # Node with power P, advective link to an inlet bath: steady
        # T = T_in + P / (rho c_p Q).
        from spin_dynamics.thermal import flow_conductance

        rho, cp, q, p, t_in = 997.0, 4184.0, 5e-7, 2.0, 295.0
        g = flow_conductance(rho, cp, q)
        nodes = [
            ThermalNode("sample", heat_capacity=10.0, initial_temperature=t_in),
            ThermalNode("inlet", heat_capacity=None, initial_temperature=t_in),
        ]
        links = [ThermalLink("sample", "inlet", conductance=g)]
        net = ThermalNetwork(nodes, links, sources={"sample": p})
        steady = net.steady_state()
        assert steady["sample"] - t_in == pytest.approx(p / g)

    def _adv_diff_analytic(self, x, L, T0, TL, Pe):
        return T0 + (TL - T0) * (np.exp(Pe * x / L) - 1.0) / (np.exp(Pe) - 1.0)

    def test_advection_diffusion_matches_analytic(self) -> None:
        L, k, rho_cp, T0, TL = 0.1, 0.5, 1e6, 300.0, 320.0
        Pe = 5.0
        v = Pe * k / (rho_cp * L)
        n = 400
        x = np.linspace(L / (2 * n), L - L / (2 * n), n)
        c = Conduction1D(
            x, geometry="slab", conductivity=k, rho_cp=rho_cp, velocity=v,
            inner_bc=("temperature", T0), outer_bc=("temperature", TL),
        )
        prof = c.steady_state().temperature
        analytic = self._adv_diff_analytic(x, L, T0, TL, Pe)
        np.testing.assert_allclose(prof, analytic, atol=0.15)  # <1% of the 20 K span

    def test_zero_velocity_reduces_to_conduction(self) -> None:
        L, k, rho_cp, T0, TL = 0.1, 0.5, 1e6, 300.0, 320.0
        n = 100
        x = np.linspace(L / (2 * n), L - L / (2 * n), n)
        kw = dict(geometry="slab", conductivity=k, rho_cp=rho_cp,
                  inner_bc=("temperature", T0), outer_bc=("temperature", TL))
        no_v = Conduction1D(x, **kw).steady_state().temperature
        zero_v = Conduction1D(x, velocity=0.0, **kw).steady_state().temperature
        np.testing.assert_allclose(no_v, zero_v)
        # Pure conduction is the linear profile.
        np.testing.assert_allclose(no_v, T0 + (TL - T0) * x / L, atol=0.05)

    def test_high_peclet_pushes_boundary_layer_downstream(self) -> None:
        L, k, rho_cp, T0, TL = 0.1, 0.5, 1e6, 300.0, 320.0
        Pe = 40.0
        v = Pe * k / (rho_cp * L)
        n = 600
        x = np.linspace(L / (2 * n), L - L / (2 * n), n)
        prof = Conduction1D(
            x, geometry="slab", conductivity=k, rho_cp=rho_cp, velocity=v,
            inner_bc=("temperature", T0), outer_bc=("temperature", TL),
        ).steady_state().temperature
        # Interior stays near the inlet T0; the rise is confined near the outlet.
        assert prof[n // 2] - T0 < 0.05 * (TL - T0)
        assert prof[-1] > 0.5 * (T0 + TL)

    def test_reverse_flow_mirrors(self) -> None:
        L, k, rho_cp, T0, TL = 0.1, 0.5, 1e6, 300.0, 320.0
        Pe = 5.0
        v = Pe * k / (rho_cp * L)
        n = 400
        x = np.linspace(L / (2 * n), L - L / (2 * n), n)
        # Forward flow, inlet at inner end (T0).
        fwd = Conduction1D(
            x, geometry="slab", conductivity=k, rho_cp=rho_cp, velocity=v,
            inner_bc=("temperature", T0), outer_bc=("temperature", TL),
        ).steady_state().temperature
        # Reverse flow with swapped inlet: mirror image.
        rev = Conduction1D(
            x, geometry="slab", conductivity=k, rho_cp=rho_cp, velocity=-v,
            inner_bc=("temperature", TL), outer_bc=("temperature", T0),
        ).steady_state().temperature
        np.testing.assert_allclose(fwd, rev[::-1], atol=1e-9)

    def test_advection_validation_errors(self) -> None:
        n = 50
        x = np.linspace(1e-3, 0.1, n)
        with pytest.raises(ValueError):
            Conduction1D(x, geometry="cylinder", conductivity=1.0, rho_cp=1e6,
                         velocity=1e-3, inner_bc=("temperature", 300.0))
        with pytest.raises(ValueError):
            Conduction1D(x, geometry="slab", conductivity=1.0, rho_cp=1e6,
                         velocity=1e-3, inner_bc=("insulated",))
