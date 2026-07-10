"""Tests for the sample layer and the physical spin-noise models."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.noise import (
    matched_probe_output_noise_density,
    tuned_probe_output_noise_density,
    untuned_probe_output_noise_density,
)
from spin_dynamics.radiation_damping import (
    HBAR,
    KB,
    PROTON_GAMMA,
    RadiationDampingProbe,
    proton_thermal_magnetization_density,
    radiation_damping_time,
)
from spin_dynamics.sample import (
    Sample,
    cuboid_geometry,
    cylinder_geometry,
    sphere_geometry,
    water_sample,
)
from spin_dynamics.spin_noise import (
    SampleCoupling,
    SpinNoiseSource,
    estimate_spin_noise_spectrum,
    sample_coupling_from_radiation_damping,
    sample_coupling_from_sample,
    sample_resistance_on_resonance,
    simulate_spin_noise,
    spin_noise_output_psd,
    spin_noise_source_from_sample,
)


class TestSampleLayer:
    def test_geometry_volumes(self) -> None:
        assert cylinder_geometry(0.01, 0.02).volume == pytest.approx(
            np.pi * 0.01**2 * 0.02
        )
        assert sphere_geometry(0.01).volume == pytest.approx(4 / 3 * np.pi * 0.01**3)
        assert cuboid_geometry(0.01, 0.02, 0.03).volume == pytest.approx(6e-6)

    def test_water_magnetization_matches_rd_helper(self) -> None:
        sample = water_sample(sphere_geometry(2.5e-3), temperature=300.0)
        b0 = 1.5
        assert sample.magnetization_density(b0) == pytest.approx(
            proton_thermal_magnetization_density(
                b0, proton_concentration_mol_per_liter=111.0, temperature_kelvin=300.0
            ),
            rel=1e-12,
        )

    def test_polarization_consistent_with_magnetization(self) -> None:
        # M0 = n * gamma*hbar*I * p for the general-I Curie forms.
        sample = water_sample(sphere_geometry(1e-3))
        b0 = 1.0
        m0 = sample.magnetization_density(b0)
        expected = sample.spin_density * sample.gamma * HBAR * 0.5 * sample.polarization(b0)
        assert m0 == pytest.approx(expected, rel=1e-12)

    def test_normalized_fluctuation_spin_half_form(self) -> None:
        # For spin 1/2 the normalized fluctuation reduces to 1/(sqrt(N)*p).
        sample = water_sample(sphere_geometry(1e-3))
        b0 = 1.0
        n = sample.number_of_spins
        p = sample.polarization(b0)
        assert sample.normalized_transverse_fluctuation(b0) == pytest.approx(
            1.0 / (np.sqrt(n) * p), rel=1e-12
        )

    def test_fluctuation_moment_scales_sqrt_n(self) -> None:
        small = water_sample(sphere_geometry(1e-3))
        big = water_sample(sphere_geometry(1e-2))
        ratio = big.transverse_fluctuation_moment() / small.transverse_fluctuation_moment()
        assert ratio == pytest.approx(np.sqrt(big.number_of_spins / small.number_of_spins))
        # Spin-1/2 closed form: sqrt(N) * gamma*hbar/2.
        assert small.transverse_fluctuation_moment() == pytest.approx(
            np.sqrt(small.number_of_spins) * PROTON_GAMMA * HBAR / 2.0, rel=1e-12
        )

    def test_sample_and_coil_temperatures_are_distinct(self) -> None:
        cold = water_sample(sphere_geometry(1e-3), temperature=77.0)
        assert cold.temperature == 77.0
        warm = cold.with_temperature(300.0)
        assert warm.temperature == 300.0
        assert cold.magnetization_density(1.0) > warm.magnetization_density(1.0)

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            cylinder_geometry(-1.0, 1.0)
        with pytest.raises(ValueError):
            Sample(
                name="bad",
                geometry=sphere_geometry(1e-3),
                spin_density=1e28,
                temperature=-1.0,
            )
        with pytest.raises(ValueError):
            Sample(
                name="bad",
                geometry=sphere_geometry(1e-3),
                spin_density=1e28,
                temperature=300.0,
                spin=0.3,
            )


def _coupling_ingredients() -> dict:
    return {
        "fill_factor": 0.6,
        "gamma": PROTON_GAMMA,
        "t2": 0.1,
        "omega0": 2 * np.pi * 1e6,
        "inductance": 10e-6,
        "q": 50.0,
    }


class TestSampleCoupling:
    def test_rn0_matches_radiation_damping_identity(self) -> None:
        # R_n0 = R_coil * T2 / (2 * Trd) with the package Trd convention.
        ing = _coupling_ingredients()
        m0 = proton_thermal_magnetization_density(ing["omega0"] / PROTON_GAMMA)
        r_coil = ing["omega0"] * ing["inductance"] / ing["q"]
        trd = radiation_damping_time(ing["gamma"], ing["fill_factor"], m0, ing["q"])
        r_n0 = sample_resistance_on_resonance(
            fill_factor=ing["fill_factor"],
            gamma=ing["gamma"],
            magnetization_density=m0,
            t2=ing["t2"],
            omega0=ing["omega0"],
            inductance=ing["inductance"],
        )
        assert r_n0 == pytest.approx(r_coil * ing["t2"] / (2 * trd), rel=1e-12)
        via_trd = sample_coupling_from_radiation_damping(
            trd=trd, coil_resistance=r_coil, t2=ing["t2"], temperature=300.0
        )
        assert via_trd.r_n0 == pytest.approx(r_n0, rel=1e-12)

    def test_impedance_lineshape(self) -> None:
        coupling = SampleCoupling(r_n0=50.0, t2=0.1, temperature=300.0)
        dw = np.linspace(-200.0, 200.0, 401)
        zn = coupling.z_n(dw)
        # Absorptive Lorentzian real part, peak on resonance, HWHM at dw = 1/T2.
        assert np.real(zn[200]) == pytest.approx(50.0)
        hw = np.real(coupling.z_n(np.array([1.0 / coupling.t2])))[0]
        assert hw == pytest.approx(25.0, rel=1e-12)
        # Dispersive odd imaginary part.
        assert np.imag(zn[200]) == pytest.approx(0.0, abs=1e-12)
        assert np.imag(zn[250]) == pytest.approx(-np.imag(zn[150]), rel=1e-9)
        # Offset shifts the peak.
        shifted = SampleCoupling(r_n0=50.0, t2=0.1, temperature=300.0, offset=50.0)
        assert np.real(shifted.z_n(np.array([50.0])))[0] == pytest.approx(50.0)

    def test_from_sample_constructor(self) -> None:
        sample = water_sample(sphere_geometry(2.5e-3), t2=0.1)
        b0 = (2 * np.pi * 1e6) / PROTON_GAMMA
        coupling = sample_coupling_from_sample(
            sample,
            field_tesla=b0,
            fill_factor=0.6,
            inductance=10e-6,
        )
        expected = sample_resistance_on_resonance(
            fill_factor=0.6,
            gamma=sample.gamma,
            magnetization_density=sample.magnetization_density(b0),
            t2=0.1,
            omega0=sample.gamma * b0,
            inductance=10e-6,
        )
        assert coupling.r_n0 == pytest.approx(expected, rel=1e-12)
        assert coupling.temperature == sample.temperature


def _tuned_sp(numpts: int = 4001, maxoffs: float = 4.0) -> dict:
    f0 = 1e6
    L = 10e-6
    Q = 50.0
    w0 = 2 * np.pi * f0
    return {
        "k": KB,
        "T": 300.0,
        "L": L,
        "R": w0 * L / Q,
        "C": 1 / (w0**2 * L),
        "Cin": 1e-15,
        "Rin": 1e12,
        "Rd": 1e12,
        "vn": 0.0,
        "in_": 0.0,
        "w0": w0,
        "del_w": np.linspace(-maxoffs, maxoffs, numpts),
    }


_PP = {"T_90": 25e-6}
_W1MAX = (np.pi / 2) / _PP["T_90"]


class TestOptionBNoiseDensities:
    def test_no_sample_is_backward_compatible(self) -> None:
        sp = _tuned_sp()
        base, f = tuned_probe_output_noise_density(sp, _PP)
        again, f2 = tuned_probe_output_noise_density(sp, _PP, sample=None)
        np.testing.assert_allclose(base, again)
        np.testing.assert_allclose(f, f2)

    def test_equilibrium_dip_at_uniform_temperature(self) -> None:
        # A sample at the coil temperature loads the resonant peak: the
        # on-resonance output noise density must drop (dip), consistent with
        # 4kT*Re(Z_out) for a passive network at uniform temperature.
        sp = _tuned_sp()
        t2 = 200.0 / _W1MAX  # narrow line well inside the offset grid
        coupling = SampleCoupling(r_n0=10.0, t2=t2, temperature=sp["T"])
        base, _ = tuned_probe_output_noise_density(sp, _PP)
        with_sample, _ = tuned_probe_output_noise_density(sp, _PP, sample=coupling)
        center = len(sp["del_w"]) // 2
        assert with_sample[center] < base[center]
        # Far off resonance the absorptive part vanishes; only the small
        # dispersive tail X_n remains (sub-percent detuning shift).
        np.testing.assert_allclose(with_sample[0], base[0], rtol=1e-2)

    def test_hot_spins_make_a_bump(self) -> None:
        sp = _tuned_sp()
        t2 = 200.0 / _W1MAX
        hot = SampleCoupling(r_n0=10.0, t2=t2, temperature=100.0 * sp["T"])
        base, _ = tuned_probe_output_noise_density(sp, _PP)
        with_sample, _ = tuned_probe_output_noise_density(sp, _PP, sample=hot)
        center = len(sp["del_w"]) // 2
        assert with_sample[center] > base[center]

    def test_cold_sample_dip_deeper_than_equilibrium(self) -> None:
        sp = _tuned_sp()
        t2 = 200.0 / _W1MAX
        base, _ = tuned_probe_output_noise_density(
            sp, _PP, sample=SampleCoupling(r_n0=10.0, t2=t2, temperature=sp["T"])
        )
        cold, _ = tuned_probe_output_noise_density(
            sp, _PP, sample=SampleCoupling(r_n0=10.0, t2=t2, temperature=1e-3)
        )
        center = len(sp["del_w"]) // 2
        assert cold[center] < base[center]

    def test_sample_feature_has_lorentzian_scale(self) -> None:
        # For a weak coupling (R_n0 << R) at uniform temperature the net
        # feature is a *dip* of magnitude ~ 4*k*T*R_n(dw)*|tf|^2 (the sample
        # emits R_n worth of noise but removes 2*R_n worth of coil-noise
        # transfer), so the dip must inherit the absorptive Lorentzian shape.
        sp = _tuned_sp(numpts=16001, maxoffs=2.0)
        t2 = 100.0 / _W1MAX
        coupling = SampleCoupling(r_n0=1e-3, t2=t2, temperature=300.0)
        base, f = tuned_probe_output_noise_density(sp, _PP)
        with_sample, _ = tuned_probe_output_noise_density(sp, _PP, sample=coupling)
        dip = base - with_sample
        center = len(sp["del_w"]) // 2
        # HWHM of the feature at dw*t2 = 1 (in physical rad/s).
        dw_phys = sp["del_w"] * _W1MAX
        idx_hwhm = np.argmin(np.abs(dw_phys - 1.0 / t2))
        assert dip[center] > 0
        assert dip[idx_hwhm] == pytest.approx(dip[center] / 2, rel=0.08)

    def test_untuned_and_matched_accept_sample(self) -> None:
        f0 = 1e6
        w0 = 2 * np.pi * f0
        L = 10e-6
        Q = 50.0
        sp_untuned = {
            "k": KB,
            "T": 300.0,
            "L": L,
            "R": w0 * L / Q,
            "C": 1e-12,
            "Cin": 1e-15,
            "Rin": 1e12,
            "Rd": 1e12,
            "Rdup": 0.1,
            "Nrx": 4.0,
            "krx": 0.95,
            "L1": 1e-6,
            "R1": 0.05,
            "vn": 0.0,
            "in_": 0.0,
            "w0": w0,
            "del_w": np.linspace(-4.0, 4.0, 2001),
        }
        t2 = 200.0 / _W1MAX
        hot = SampleCoupling(r_n0=1.0, t2=t2, temperature=3000.0)
        base, _ = untuned_probe_output_noise_density(sp_untuned, _PP)
        bumped, _ = untuned_probe_output_noise_density(sp_untuned, _PP, sample=hot)
        center = len(sp_untuned["del_w"]) // 2
        assert bumped[center] > base[center]
        assert bumped[0] == pytest.approx(base[0], rel=1e-2)

        del_w = np.linspace(-4.0, 4.0, 2001)
        sp_matched = {
            "k": KB,
            "T": 300.0,
            "L": L,
            "f0": f0,
            "Q": Q,
            "NF": 0.0,
            "Rin": 50.0,
            "del_w": del_w,
        }
        tf1 = np.ones(del_w.size, dtype=np.complex128)
        base_m, _ = matched_probe_output_noise_density(sp_matched, _PP, tf1=tf1)
        hot_m, _ = matched_probe_output_noise_density(
            sp_matched, _PP, tf1=tf1, sample=hot
        )
        cold_m, _ = matched_probe_output_noise_density(
            sp_matched,
            _PP,
            tf1=tf1,
            sample=SampleCoupling(r_n0=1.0, t2=t2, temperature=1e-3),
        )
        assert hot_m[center] > base_m[center]
        assert cold_m[center] < base_m[center]


def _probe(trd_scale: float = 1.0) -> RadiationDampingProbe:
    # Direct construction: choose fill/Q/M0 so trd lands at a convenient value.
    gamma = PROTON_GAMMA
    q = 50.0
    fill = 0.5
    trd_target = 0.02 * trd_scale
    m0 = 2.0 / (gamma * (4e-7 * np.pi) * fill * q * trd_target)
    return RadiationDampingProbe(
        gamma=gamma,
        omega0=2 * np.pi * 1e6,
        q=q,
        fill_factor=fill,
        equilibrium_magnetization=m0,
    )


class TestOptionCStochastic:
    def test_stationary_variance_matches_two_bath_balance(self) -> None:
        probe = _probe()
        t2 = 0.01
        source = SpinNoiseSource(
            m_rms=1e-5, t2=t2, sample_temperature=300.0, coil_temperature=300.0
        )
        dt = 2e-4
        time = np.arange(0.0, 20.0, dt)
        res = simulate_spin_noise(time, probe, source, seed=7)
        burn = int(5 * max(t2, probe.trd) / dt)
        var = np.mean(np.abs(res.mxy[burn:]) ** 2)
        gamma_rate = 1.0 / t2 + 1.0 / probe.trd
        expected = (
            source.spin_bath_force_density() + source.coil_force_density(probe.trd)
        ) / (2.0 * gamma_rate)
        assert var == pytest.approx(expected, rel=0.1)

    def test_autocorrelation_decays_at_total_rate(self) -> None:
        probe = _probe()
        t2 = 0.01
        source = SpinNoiseSource(
            m_rms=1e-5, t2=t2, sample_temperature=300.0, coil_temperature=0.0
        )
        dt = 2e-4
        time = np.arange(0.0, 40.0, dt)
        res = simulate_spin_noise(time, probe, source, seed=11)
        burn = int(5 * t2 / dt)
        m = res.mxy[burn:]
        gamma_rate = 1.0 / t2 + 1.0 / probe.trd
        lag = int(round(0.5 / gamma_rate / dt))
        r0 = np.mean(np.abs(m) ** 2)
        rlag = np.real(np.mean(m[lag:] * np.conj(m[:-lag])))
        assert rlag / r0 == pytest.approx(np.exp(-gamma_rate * lag * dt), abs=0.06)

    def test_cold_coil_zero_t2_bath_gives_pure_ou_spectrum(self) -> None:
        # Coil at zero temperature: only the spin bath drives; the mxy PSD is
        # the OU Lorentzian of total width 1/T2 + 1/Trd and integrates to the
        # stationary variance.
        probe = _probe()
        t2 = 0.01
        source = SpinNoiseSource(
            m_rms=1e-5, t2=t2, sample_temperature=300.0, coil_temperature=0.0
        )
        dt = 2e-4
        time = np.arange(0.0, 80.0, dt)
        res = simulate_spin_noise(time, probe, source, seed=3)
        burn = int(5 * t2 / dt)
        gamma_rate = 1.0 / t2 + 1.0 / probe.trd
        block = int(round(8.0 / gamma_rate / dt))
        m = res.mxy[burn:]
        freqs, psd = estimate_spin_noise_spectrum(m, dt, block_samples=block)
        df = freqs[1] - freqs[0]
        trimmed = m[: (m.size // block) * block]
        var = np.mean(np.abs(trimmed) ** 2)
        # Parseval: the two-sided PSD integrates to the signal power.
        assert np.sum(psd) * df == pytest.approx(var, rel=1e-9)
        # Peak density: S(0) = sigma_s^2 / Gamma^2.
        expected_peak = source.spin_bath_force_density() / gamma_rate**2
        center = np.argmin(np.abs(freqs))
        measured_peak = np.mean(psd[center - 2 : center + 3])
        assert measured_peak == pytest.approx(expected_peak, rel=0.25)

    def test_emf_spectrum_matches_analytic_dip(self) -> None:
        # Equilibrium (Tc = Ts): the receiver-node PSD must reproduce the
        # analytic linear-response curve, including the on-resonance dip
        # S(0)/S_e = Trd/(T2+Trd).
        probe = _probe()
        t2 = 0.01
        source = SpinNoiseSource(
            m_rms=1e-5, t2=t2, sample_temperature=300.0, coil_temperature=300.0
        )
        dt = 5e-4
        time = np.arange(0.0, 100.0, dt)
        res = simulate_spin_noise(time, probe, source, seed=19)
        burn = int(5 * max(t2, probe.trd) / dt)
        gamma_rate = 1.0 / t2 + 1.0 / probe.trd
        block = int(round(12.0 / gamma_rate / dt))
        freqs, psd = estimate_spin_noise_spectrum(res.emf[burn:], dt, block_samples=block)
        analytic = spin_noise_output_psd(freqs, source=source, trd=probe.trd)
        # On-resonance analytic depth is exactly S_e * Trd / (T2 + Trd).
        center = np.argmin(np.abs(freqs))
        s_e = source.coil_force_density(probe.trd) * probe.trd**2
        assert analytic[center] == pytest.approx(
            s_e * probe.trd / (t2 + probe.trd), rel=1e-9
        )
        # The simulated feature depth must track the analytic curve.
        edge = np.abs(freqs) > 5 * gamma_rate / (2 * np.pi)
        measured_ratio = np.mean(psd[center - 2 : center + 3]) / np.mean(psd[edge])
        analytic_ratio = analytic[center] / np.mean(analytic[edge])
        assert measured_ratio == pytest.approx(analytic_ratio, rel=0.2)
        assert analytic_ratio < 1.0  # equilibrium dip

    def test_hot_spins_flip_dip_to_bump(self) -> None:
        probe = _probe()
        t2 = 0.01
        # Bump threshold is Ts/Tc > 2 + T2/Trd; go well beyond it.
        ratio = 10.0 * (2.0 + t2 / probe.trd)
        source = SpinNoiseSource(
            m_rms=1e-5,
            t2=t2,
            sample_temperature=300.0 * ratio,
            coil_temperature=300.0,
        )
        freqs = np.linspace(-200.0, 200.0, 2001)
        psd = spin_noise_output_psd(freqs, source=source, trd=probe.trd)
        center = np.argmin(np.abs(freqs))
        assert psd[center] > psd[0]

    def test_spin_offset_moves_the_line(self) -> None:
        probe = _probe()
        offset_hz = 40.0
        source = SpinNoiseSource(
            m_rms=1e-5,
            t2=0.01,
            sample_temperature=300.0,
            coil_temperature=0.0,
            spin_offset=2 * np.pi * offset_hz,
        )
        freqs = np.linspace(-200.0, 200.0, 4001)
        psd = spin_noise_output_psd(freqs, source=source, trd=probe.trd)
        assert freqs[np.argmax(psd)] == pytest.approx(offset_hz, abs=0.5)

    def test_zero_noise_recovers_deterministic_damping(self) -> None:
        # m_rms = 0 and cold coil: a tipped magnetization must decay at
        # 1/T2 + 1/Trd (small-tip linear regime) with no stochastic component.
        probe = _probe()
        t2 = 0.01
        source = SpinNoiseSource(
            m_rms=0.0, t2=t2, sample_temperature=300.0, coil_temperature=0.0
        )
        dt = 1e-4
        time = np.arange(0.0, 0.05, dt)
        res = simulate_spin_noise(
            time, probe, source, initial_mxy=1e-3 + 0j, seed=1, max_step=1e-5
        )
        gamma_rate = 1.0 / t2 + 1.0 / probe.trd
        expected = 1e-3 * np.exp(-gamma_rate * time)
        np.testing.assert_allclose(np.abs(res.mxy), expected, rtol=0.02)

    def test_damping_is_phase_independent(self) -> None:
        # The Bloom-form feedback must damp every transverse phase equally --
        # this is exactly where a conj(mxy) feedback would anti-damp.
        probe = _probe()
        source = SpinNoiseSource(
            m_rms=0.0, t2=np.inf, sample_temperature=300.0, coil_temperature=0.0
        )
        dt = 1e-4
        time = np.arange(0.0, 0.05, dt)
        finals = []
        for phase in (0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2):
            res = simulate_spin_noise(
                time,
                probe,
                source,
                initial_mxy=1e-3 * np.exp(1j * phase),
                seed=1,
            )
            assert np.abs(res.mxy[-1]) < 1e-3  # damped, never amplified
            finals.append(np.abs(res.mxy[-1]))
        np.testing.assert_allclose(finals, finals[0], rtol=1e-9)

    def test_spin_noise_seeds_maser_growth(self) -> None:
        # Inverted, pumped sample above threshold: spin noise alone (no
        # deterministic seed) must ignite the maser.
        probe = _probe(trd_scale=0.1)
        t2 = 0.01
        t1 = 0.05
        assert 1.0 / probe.trd > 1.0 / t2  # above threshold for |mz| = 1
        source = SpinNoiseSource(
            m_rms=1e-6, t2=t2, sample_temperature=300.0, coil_temperature=0.0
        )
        dt = 1e-4
        time = np.arange(0.0, 0.6, dt)
        res = simulate_spin_noise(
            time,
            probe,
            source,
            initial_mxy=0.0 + 0.0j,
            initial_mz=-1.0,
            t1=t1,
            equilibrium_mz=-1.0,
            seed=23,
        )
        assert np.max(np.abs(res.mxy)) > 1e3 * source.m_rms

    def test_source_from_sample(self) -> None:
        sample = water_sample(sphere_geometry(1e-3), t2=0.1)
        b0 = 0.1
        source = spin_noise_source_from_sample(
            sample, field_tesla=b0, coil_temperature=77.0
        )
        assert source.m_rms == pytest.approx(
            sample.normalized_transverse_fluctuation(b0), rel=1e-12
        )
        assert source.t2 == sample.t2
        assert source.sample_temperature == sample.temperature
        assert source.effective_coil_temperature == 77.0

    def test_estimator_validation_and_seed_reproducibility(self) -> None:
        probe = _probe()
        source = SpinNoiseSource(
            m_rms=1e-5, t2=0.01, sample_temperature=300.0
        )
        time = np.linspace(0.0, 0.1, 201)
        a = simulate_spin_noise(time, probe, source, seed=5)
        b = simulate_spin_noise(time, probe, source, seed=5)
        np.testing.assert_array_equal(a.mxy, b.mxy)
        np.testing.assert_array_equal(a.coil_noise, b.coil_noise)
        with pytest.raises(ValueError):
            simulate_spin_noise(
                time, probe, source, seed=5, rng=np.random.default_rng(1)
            )
        with pytest.raises(ValueError):
            estimate_spin_noise_spectrum(np.zeros(10), -1.0, block_samples=4)
        with pytest.raises(ValueError):
            estimate_spin_noise_spectrum(np.zeros(3), 1.0, block_samples=8)
