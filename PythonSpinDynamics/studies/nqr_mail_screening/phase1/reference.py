"""Absolute spin-1 -> moment -> solenoid EMF -> real ADC reference.

Study-local composition of library APIs; no instrument control or parcel model.
Run from PythonSpinDynamics with --output .tmp/nqr_mail_screening_phase1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict, replace
import hashlib
import json
from pathlib import Path
from numbers import Integral
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from spin_dynamics.nqr.crossover import boltzmann_populations
from spin_dynamics.nqr.hamiltonians import diagonalize_site
from spin_dynamics.nqr.isotopes import QUADRUPOLAR_ISOTOPES
from spin_dynamics.nqr.systems import QuadrupolarSite
from spin_dynamics.nqr.operators import spin_matrices
from spin_dynamics.nqr.orientations import OrientationSample
from spin_dynamics.nqr.pulses import SelectivePulse, apply_selective_pulse
from spin_dynamics.nqr.sequences import SLSESequence
from spin_dynamics.nqr.simulation import equilibrium_density, simulate_slse
from spin_dynamics.fields.coils import solenoid
from spin_dynamics.fields.magnetostatics import MU0, biot_savart
from spin_dynamics.detection.base import snr_from_field_noise_psd

H = 6.62607015e-34
KB = 1.380649e-23
NA = 6.02214076e23
GAMMA_HZ_T = QUADRUPOLAR_ISOTOPES["14N"].gamma_hz_per_t
HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ReferenceConfig:
    handling_time_s: float = 2.0  # delegated study overhead, outside RF segment
    mass_kg: float = 0.001
    molar_mass_kg_mol: float = 0.37293
    sites_per_molecule: float = 1.0  # aniline only, not both molecular nitrogens
    isotope_fraction: float = 0.9964  # declared natural-abundance approximation
    crystalline_fraction: float = 1.0
    temperature_k: float = (
        293.15  # assumed room temperature, not measured source temperature
    )
    radius_m: float = 0.03
    length_m: float = 0.06
    turns: int = 10
    segments_per_turn: int = 256
    powder_points: int = 32
    flip_angle_rad: float = np.pi / 2
    pulse_duration_s: float = 50e-6
    blanking_s: float = 100e-6
    record_duration_s: float = 0.001
    sample_rate_hz: float = 16e6
    coil_resistance_ohm: float = 2.0
    coil_temperature_k: float = 293.15
    receiver_noise_v_rt_hz: float = 1e-9
    voltage_gain: float = 10000.0
    adc_bits: int = 16
    adc_full_scale_v: float = 2.0  # total bipolar span, [-1,1) V
    monte_carlo_trials: int = 2048
    seed: int = 104729

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite numeric")
            if key != "seed" and value <= 0:
                raise ValueError(f"{key} must be positive")
        for key in (
            "turns",
            "segments_per_turn",
            "powder_points",
            "adc_bits",
            "monte_carlo_trials",
            "seed",
        ):
            value = getattr(self, key)
            if not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        if not 2 <= self.adc_bits <= 24:
            raise ValueError("ADC bits must be 2..24")
        if self.powder_points < 4 or self.segments_per_turn < 16:
            raise ValueError("insufficient reference integration resolution")
        if self.isotope_fraction > 1 or self.crystalline_fraction > 1:
            raise ValueError("fractions must not exceed one")
        if self.monte_carlo_trials < 64:
            raise ValueError("at least 64 Monte Carlo trials required")


def material_reference():
    data = json.loads((HERE.parent / "phase0/literature_supplement.json").read_text())
    lines = {r["line_id"]: r["values"] for r in data["materials"]}
    x = lines["malone2025:aniline:nu_x"]
    y = lines["malone2025:aniline:nu_y"]
    q = (x["frequency_hz"] + y["frequency_hz"]) / 2
    eta = 3 * (x["frequency_hz"] - y["frequency_hz"]) / (2 * q)
    site = QuadrupolarSite(
        quadrupole_frequency_hz=q,
        eta=eta,
        gamma_hz_per_t=GAMMA_HZ_T,
        label="fentanyl-HCl aniline",
    )
    eig = diagonalize_site(site)
    transition = min(
        eig.transitions, key=lambda t: abs(t.frequency_hz - x["frequency_hz"])
    )
    return eig, transition, x


def thermal_deviation(levels_hz, temperature_k):
    """Exact diagonal Gibbs deviation, avoiding subtraction of ~1/3 populations.

    Library Gibbs probabilities are audited separately. Using expm1 here preserves
    the ~1e-7 signal when the identity part is propagated by unitary matrices.
    """
    levels = np.asarray(levels_hz, dtype=float)
    if levels.shape != (3,) or not np.all(np.isfinite(levels)):
        raise ValueError("need three finite spin-1 energy levels in Hz")
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("temperature must be positive")
    x = -H * (levels - levels.min()) / (KB * temperature_k)
    d = np.expm1(x)
    return np.diag((d - d.mean()) / (3 + d.sum())).astype(complex)


def high_temperature_scale(levels_hz, temperature_k):
    """Multiplier for the library's max-normalized trace-zero equilibrium."""
    levels = np.asarray(levels_hz)
    return H * np.max(np.abs(levels - levels.mean())) / (3 * KB * temperature_k)


def nuclei_count(cfg):
    return (
        cfg.mass_kg
        / cfg.molar_mass_kg_mol
        * NA
        * cfg.sites_per_molecule
        * cfg.isotope_fraction
        * cfg.crystalline_fraction
    )


def powder_pulse(cfg, initial_deviation=None):
    """Return selected-line moment phasor: m(t)=Re[mhat exp(-i omega t)]."""
    eig, transition, line = material_reference()
    rho0 = (
        thermal_deviation(eig.levels_hz, cfg.temperature_k)
        if initial_deviation is None
        else initial_deviation
    )
    axis = int(np.argmax(np.abs(transition.dipole_vector)))
    mu, weights = np.polynomial.legendre.leggauss(cfg.powder_points)
    weights = weights / 2
    ops = spin_matrices(1)
    spin_ops = (ops.ix, ops.iy, ops.iz)
    pulse = SelectivePulse(
        transition.label,
        cfg.pulse_duration_s,
        cfg.flip_angle_rad / (2 * np.pi * cfg.pulse_duration_s),
    )
    moment = 0j
    samples = []
    moment_per_nucleus = H * GAMMA_HZ_T  # hbar * gamma_rad, NOT hbar * gamma_Hz
    for u, w in zip(mu, weights):
        direction = np.zeros(3)
        direction[axis] = u
        direction[(axis + 1) % 3] = np.sqrt(1 - u * u)
        samples.append(OrientationSample(direction, float(w)))
        rho = apply_selective_pulse(rho0, transition, pulse, b1_direction_pas=direction)
        op = (
            eig.eigenvectors.conj().T
            @ sum(d * o for d, o in zip(direction, spin_ops))
            @ eig.eigenvectors
        )
        # Hermitian conjugate coherence supplies the other half of the real RF waveform.
        moment += (
            w
            * 2
            * moment_per_nucleus
            * op[transition.lower, transition.upper]
            * rho[transition.upper, transition.lower]
        )
    moment *= nuclei_count(cfg)
    return complex(moment), eig, transition, line, tuple(samples), pulse


def analytic_powder_moment(cfg, eig, transition):
    """Independent two-level rotation and analytic isotropic angular integral."""
    p = boltzmann_populations(eig.levels_hz, cfg.temperature_k)
    delta = p[transition.lower] - p[transition.upper]
    a = cfg.flip_angle_rad
    angular = (
        (np.sin(a) - a * np.cos(a)) / (a * a) if abs(a) > 1e-4 else a / 3 - a**3 / 30
    )
    # Sign follows the selective-pulse Hamiltonian convention of this reference.
    return -1j * nuclei_count(cfg) * H * GAMMA_HZ_T * delta * angular


def coil_couplings(cfg):
    """Numerical Tx unit-current field vs independent dipole-flux line integral."""
    seg = solenoid(
        radius=cfg.radius_m,
        length=cfg.length_m,
        turns=cfg.turns,
        n_segments=cfg.segments_per_turn,
    )
    beta = float(biot_savart(np.zeros((1, 3)), seg, 1.0)[0, 2])
    z = (
        np.linspace(-cfg.length_m / 2, cfg.length_m / 2, cfg.turns)
        if cfg.turns > 1
        else np.array([0.0])
    )
    exact = float(
        np.sum(MU0 * cfg.radius_m**2 / (2 * (cfg.radius_m**2 + z * z) ** 1.5))
    )
    # Stokes: Phi = integral A_dipole dot dl on each circular wire;
    # A = mu0/(4pi) m cross r / r^3. Independent of the Tx solver.
    phi = 0.0
    theta = (np.arange(4096) + 0.5) * 2 * np.pi / 4096
    for zi in z:
        r = np.stack(
            [
                cfg.radius_m * np.cos(theta),
                cfg.radius_m * np.sin(theta),
                np.full_like(theta, zi),
            ],
            axis=-1,
        )
        dl = (
            np.stack(
                [
                    -cfg.radius_m * np.sin(theta),
                    cfg.radius_m * np.cos(theta),
                    np.zeros_like(theta),
                ],
                axis=-1,
            )
            * 2
            * np.pi
            / 4096
        )
        potential = (
            MU0
            / (4 * np.pi)
            * np.cross(np.array([0.0, 0.0, 1.0]), r)
            / np.linalg.norm(r, axis=-1)[:, None] ** 3
        )
        phi += float(np.sum(potential * dl))
    return beta, exact, phi


@dataclass(frozen=True)
class PreparedState:
    """Material-specific state supplied by a transfer model or calibration.

    State is at the end of buildup/transfer. Ramp time is charged to cycle time;
    any ramp relaxation must already be in populations. Settling relaxes here.
    """

    populations: tuple[float, float, float]
    buildup_s: float
    ramp_s: float
    settling_s: float
    energy_j: float
    source_reference: str
    evidence_class: str

    def prepare(self, equilibrium, t1_s):
        p = np.asarray(self.populations, dtype=float)
        if (
            p.shape != (3,)
            or not np.all(np.isfinite(p))
            or np.any(p < 0)
            or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("prepared populations must be a probability vector")
        for value in (self.buildup_s, self.ramp_s, self.settling_s, self.energy_j):
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    "pre-polarization costs must be finite and nonnegative"
                )
        if not self.source_reference or self.evidence_class not in {
            "predicted",
            "literature",
            "measured_calibration",
            "fitted",
        }:
            raise ValueError("prepared state requires source and evidence class")
        if not np.isfinite(t1_s) or t1_s <= 0:
            raise ValueError("positive settling relaxation time required")
        eq = np.real(np.diag(equilibrium)) + 1 / 3
        final = eq + (p - eq) * np.exp(-self.settling_s / t1_s)
        return np.diag(final - 1 / 3).astype(
            complex
        ), self.buildup_s + self.ramp_s + self.settling_s


def adc_quantize(voltage, cfg):
    """Real, signed mid-tread ADC with explicit clipping; return volts and codes."""
    step = cfg.adc_full_scale_v / 2**cfg.adc_bits
    code = np.rint(np.asarray(voltage) / step)
    lo, hi = -(2 ** (cfg.adc_bits - 1)), 2 ** (cfg.adc_bits - 1) - 1
    clipped = np.count_nonzero((code < lo) | (code > hi))
    code = np.clip(code, lo, hi).astype(np.int32)
    return code * step, code, int(clipped)


def build_reference(cfg):
    moment, eig, tr, line, orientations, pulse = powder_pulse(cfg)
    beta, exact_beta, dipole_flux = coil_couplings(cfg)
    if cfg.sample_rate_hz <= 2 * tr.frequency_hz:
        raise ValueError("RF ADC sample rate must exceed carrier Nyquist limit")
    n = int(round(cfg.record_duration_s * cfg.sample_rate_hz))
    if n < 8:
        raise ValueError("need at least 8 RF samples")
    t = cfg.blanking_s + np.arange(n) / cfg.sample_rate_hz
    rate = -1 / line["t2_star_s"] - 2j * np.pi * tr.frequency_hz
    # Differentiate the entire damped moment, not just the carrier.
    moment_t = moment * np.exp(rate * t)
    emf = np.real(-beta * rate * moment_t)
    direct = analytic_powder_moment(cfg, eig, tr)
    emf_independent = np.real(-dipole_flux * rate * direct * np.exp(rate * t))
    voltage = cfg.voltage_gain * emf
    coil_psd = 4 * KB * cfg.coil_temperature_k * cfg.coil_resistance_ohm
    amp_psd = cfg.receiver_noise_v_rt_hz**2
    output_psd = cfg.voltage_gain**2 * (coil_psd + amp_psd)  # one-sided, real RF
    variance = output_psd * cfg.sample_rate_hz / 2
    step = cfg.adc_full_scale_v / 2**cfg.adc_bits
    quant_variance = step**2 / 12  # valid with analog-noise dither, verified by MC
    energy = float(voltage @ voltage)
    if np.sqrt(variance) < 10 * step:
        raise ValueError(
            "reference ADC noise model requires at least 10 LSB analog dither"
        )
    snr = np.sqrt(
        energy / (variance + quant_variance)
    )  # known real waveform / real noise SD
    # Parseval path via the package's field-noise matched filter, TWO-sided PSD.
    freq = np.fft.fftfreq(n, 1 / cfg.sample_rate_hz)
    spectrum = np.fft.fft(voltage) / cfg.sample_rate_hz
    full_psd = np.full(n, (variance + quant_variance) / cfg.sample_rate_hz)
    spectral_snr = snr_from_field_noise_psd(
        spectrum, freq, full_psd, df=cfg.sample_rate_hz / n
    ).snr
    # Equivalent uniform-flux field coordinate, not local dipole B at the coil center.
    effective_area = cfg.turns * np.pi * cfg.radius_m**2
    h_field = cfg.voltage_gain * 2 * np.pi * tr.frequency_hz * effective_area
    field_snr = snr_from_field_noise_psd(
        spectrum / h_field, freq, full_psd / h_field**2, df=cfg.sample_rate_hz / n
    ).snr
    ht = high_temperature_scale(eig.levels_hz, cfg.temperature_k)
    rho_ht = equilibrium_density(eig.levels_hz) * ht
    rho_exact = thermal_deviation(eig.levels_hz, cfg.temperature_k)
    slse = SLSESequence(pulse, 0.001, 4)
    rel = simulate_slse(eig.site, slse, orientations=orientations)
    absolute = simulate_slse(
        eig.site, slse, orientations=orientations, initial_density=rho_exact
    )
    slse_error = np.linalg.norm(
        absolute.echo_amplitudes - ht * rel.echo_amplitudes
    ) / np.linalg.norm(absolute.echo_amplitudes)
    # Independent limiting checks with only the stated nuisance variable changed.
    double_mass = powder_pulse(replace(cfg, mass_kg=2 * cfg.mass_kg))[0]
    double_temperature = powder_pulse(
        replace(cfg, temperature_k=2 * cfg.temperature_k)
    )[0]
    refined_powder = powder_pulse(replace(cfg, powder_points=2 * cfg.powder_points))[0]
    refined_beta = coil_couplings(
        replace(cfg, segments_per_turn=2 * cfg.segments_per_turn)
    )[0]
    dt_fd = 1 / (tr.frequency_hz * 4096)
    phi_plus = np.real(dipole_flux * direct * np.exp(rate * (t + dt_fd)))
    phi_minus = np.real(dipole_flux * direct * np.exp(rate * (t - dt_fd)))
    emf_fd = -(phi_plus - phi_minus) / (2 * dt_fd)
    report = {
        "schema_version": "0.1.0",
        "evidence_class": "predicted",
        "scope": "single-line ideal selective pulse/FID, point sample at reference-solenoid center; no scanner claim",
        "configuration": asdict(cfg),
        "material_line_id": line["line_id"],
        "budget": {
            "nuclei_at_addressed_site": nuclei_count(cfg),
            "frequency_hz": tr.frequency_hz,
            "equilibrium_population_difference": float(
                np.real(rho_exact[tr.lower, tr.lower] - rho_exact[tr.upper, tr.upper])
            ),
            "normalized_density_to_physical_scale": ht,
            "moment_peak_am2": abs(moment),
            "coil_b_per_i_t_a": beta,
            "coil_b_per_i_analytic_t_a": exact_beta,
            "flux_peak_wb_at_pulse_end": abs(beta * moment),
            "emf_peak_v_at_pulse_end": abs(beta * rate * moment),
            "receiver_peak_v_in_valid_window": float(np.max(np.abs(voltage))),
            "coil_noise_one_sided_v2_hz": coil_psd,
            "amplifier_noise_one_sided_v2_hz": amp_psd,
            "receiver_noise_rms_v": np.sqrt(variance),
            "adc_step_v": step,
            "adc_quantization_variance_v2": quant_variance,
            "matched_filter_snr_real": snr,
            "spectral_snr_real": spectral_snr,
            "equivalent_field_snr_real": field_snr,
            "pulse_b_peak_t": cfg.flip_angle_rad
            / (2 * np.pi * cfg.pulse_duration_s * GAMMA_HZ_T * tr.strength),
            "pulse_current_peak_a": cfg.flip_angle_rad
            / (2 * np.pi * cfg.pulse_duration_s * GAMMA_HZ_T * tr.strength * beta),
            "physical_pulse_blanking_record_s": cfg.pulse_duration_s
            + cfg.blanking_s
            + n / cfg.sample_rate_hz,
        },
        "checks": {
            "mass_linearity_relative_error": float(abs(double_mass / (2 * moment) - 1)),
            "inverse_temperature_relative_error": float(
                abs(2 * double_temperature / moment - 1)
            ),
            "powder_refinement_relative_error": float(abs(refined_powder / moment - 1)),
            "coil_refinement_relative_error": float(abs(refined_beta / beta - 1)),
            "faraday_finite_difference_relative_error": float(
                np.linalg.norm(emf_fd - emf_independent)
                / np.linalg.norm(emf_independent)
            ),
            "density_high_temperature_relative_error": float(
                np.linalg.norm(rho_exact - rho_ht) / np.linalg.norm(rho_exact)
            ),
            "moment_independent_relative_error": float(
                abs(moment - direct) / abs(moment)
            ),
            "coil_biot_savart_vs_dipole_flux_relative_error": abs(beta - dipole_flux)
            / abs(dipole_flux),
            "adc_waveform_independent_relative_error": float(
                np.linalg.norm(emf - emf_independent) / np.linalg.norm(emf_independent)
            ),
            "slse_absolute_vs_scaled_relative_error": float(slse_error),
            "time_vs_spectral_snr_relative_error": abs(snr - spectral_snr) / snr,
            "voltage_vs_field_snr_relative_error": abs(snr - field_snr) / snr,
        },
        "prepolarization": {
            "status": "state-and-cost interface implemented; material transfer not parameterized",
            "enhancement_factor": None,
            "library_model": "spin_dynamics.nqr.polarization_enhancement.simulate_adiabatic_polarization_transfer",
            "notes": "Adapter requires material-specific prepared populations and all costs. No fentanyl transfer efficiency or gain invented.",
        },
        "limitations": [
            "Reference coil coupling and electronics are predicted/assumed, not measured calibration.",
            "Room temperature assigned 293.15 K; exact source temperature was not reported in imported table.",
            "FID uses literature T2-star; effective CPMG decay is not substituted for SLSE/SORC T2.",
            "Ideal selective RWA pulse neglects relaxation during RF; no loading, resonator, RFI or overload transient modeled.",
            "High-impedance ideal flat-gain voltage receiver, white analog noise to Nyquist; ADC quantization checked with analog dither.",
            "No parcel-size coil optimization, material detection limit, ROC/AUC or empirical calibration claim.",
        ],
    }
    arrays = {
        "time_after_pulse_s": t,
        "moment_am2": np.real(moment_t),
        "pickup_flux_wb": np.real(beta * moment_t),
        "emf_v": emf,
        "receiver_clean_v": voltage,
    }
    return report, arrays


def monte_carlo(cfg, report, arrays):
    clean = arrays["receiver_clean_v"]
    unit = clean / np.linalg.norm(clean)
    sigma = report["budget"]["receiver_noise_rms_v"]
    rng = np.random.default_rng(cfg.seed)
    adc_scores = []
    clipped = 0
    for start in range(0, cfg.monte_carlo_trials, 32):
        count = min(32, cfg.monte_carlo_trials - start)
        noise = rng.normal(0, sigma, (count, clean.size))
        adc, codes, clips = adc_quantize(clean + noise, cfg)
        adc_scores.extend((adc - clean) @ unit)
        clipped += clips
        if start == 0:
            arrays["receiver_noisy_v"] = clean + noise[0]
            arrays["receiver_noise_v"] = noise[0]
            arrays["adc_codes"] = codes[0]
            arrays["adc_reconstructed_v"] = adc[0]
    adc_scores = np.asarray(adc_scores)
    expected = np.sqrt(sigma * sigma + report["budget"]["adc_quantization_variance_v2"])
    snr = np.linalg.norm(clean) / np.std(adc_scores, ddof=1)
    grouped = adc_scores[: len(adc_scores) // 4 * 4].reshape(-1, 4).mean(axis=1)
    ratio = np.std(adc_scores, ddof=1) / np.std(grouped, ddof=1)
    return {
        "trials": cfg.monte_carlo_trials,
        "seed": cfg.seed,
        "adc_clipped_samples": clipped,
        "predicted_score_noise_sd_v": expected,
        "measured_score_noise_sd_v": float(np.std(adc_scores, ddof=1)),
        "noise_sd_relative_error": abs(
            float(np.std(adc_scores, ddof=1)) / expected - 1
        ),
        "noise_mean_standard_errors": float(
            np.mean(adc_scores) / (expected / np.sqrt(len(adc_scores)))
        ),
        "measured_matched_filter_snr_real": float(snr),
        "four_average_snr_ratio": float(ratio),
        "four_average_ratio_relative_error": abs(float(ratio) / 2 - 1),
    }


def result_record(cfg, report):
    """Canonical single-reference record; no detector decision or ROC claim."""
    phase0 = HERE.parent / "phase0"
    requirements = json.loads((phase0 / "requirements.json").read_text())
    materials = json.loads((phase0 / "materials.json").read_text())
    current_limit = next(
        r["value"]
        for r in requirements["requirements"]
        if r["id"] == "req.hardware.max_coil_current"
    )
    current = report["budget"]["pulse_current_peak_a"]
    return {
        "schema_version": "0.1.0",
        "study_id": "nqr-mail-screening",
        "record_id": "phase1-absolute-reference",
        "record_kind": "simulation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "predicted",
        "requirements_version": requirements["requirements_version"],
        "material_library_version": materials["library_version"],
        "model_fidelity": "level_a",
        "scenario": {
            "scenario_id": "phase1.center-point.thermal",
            "hypothesis": "target_present",
            "material_ids": ["material.target.fentanyl_hcl"],
            "parcel_class": "point_reference",
            "random_seed": cfg.seed,
            "parameters": [
                {
                    "name": "salt_mass",
                    "value": cfg.mass_kg,
                    "units": "kg",
                    "evidence_class": "predicted",
                    "uncertainty": None,
                },
                {
                    "name": "temperature",
                    "value": cfg.temperature_k,
                    "units": "K",
                    "evidence_class": "predicted",
                    "uncertainty": None,
                },
            ],
        },
        "design": {
            "design_id": "phase1-analytic-reference",
            "hardware_id": "small-solenoid-ideal-receiver",
            "protocol_id": "selective-pulse-fid",
            "decision_rule_id": "not_evaluated",
        },
        "acquisition": {
            "physical_time_s": cfg.handling_time_s
            + report["budget"]["physical_pulse_blanking_record_s"],
            "shot_count": 1,
            "carrier_frequencies_hz": [report["budget"]["frequency_hz"]],
            "stopping_reason": "sequence_complete",
        },
        "signal": {
            "domain": "adc_voltage",
            "units": "V",
            "data_reference": "reference_waveforms.npz#adc_reconstructed_v",
            "components": [
                {
                    "name": name,
                    "data_reference": "reference_waveforms.npz#" + key,
                    "evidence_class": "predicted",
                }
                for name, key in [
                    ("clean_nqr", "receiver_clean_v"),
                    ("receiver_noise", "receiver_noise_v"),
                    ("total", "adc_reconstructed_v"),
                ]
            ],
        },
        "constraints": [
            {
                "requirement_id": "req.hardware.max_coil_current",
                "value": current,
                "limit": current_limit,
                "units": "A",
                "comparison": "<=",
                "margin": current_limit - current,
                "passed": current <= current_limit,
                "evidence_class": "predicted",
            }
        ],
        "decision": {
            "status": "not_evaluated",
            "score": None,
            "threshold": None,
            "predicted_label": None,
            "truth_label": "target_present",
        },
        "provenance": {
            "repository_commit": report["provenance"]["repository_base_commit"],
            "input_records": [
                "reference_report.json",
                "repo:PythonSpinDynamics/studies/nqr_mail_screening/phase0/literature_supplement.json",
            ],
            "model_versions": {
                "numpy": np.__version__,
                "reference_source_sha256": report["provenance"][
                    "reference_source_sha256"
                ],
            },
            "notes": "Source hash identifies working code beyond base commit. One synthetic ADC realization; MC trials are independent verification repetitions, not extra physical shots. Time includes assumed 2 s handling plus RF segment; initial thermal equilibrium is stipulated. Only current is constrained here; no claim all hardware constraints passed. RFI and ringdown are absent in this reference.",
        },
    }


def plot_reference(report, arrays, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    t = arrays["time_after_pulse_s"]
    axes[0, 0].plot((t[:128] - t[0]) * 1e6, arrays["emf_v"][:128] * 1e9)
    axes[0, 0].set(
        xlabel="Time from first valid sample (us)",
        ylabel="Pickup EMF (nV)",
        title="Real RF signal, first 8 us",
    )
    axes[0, 1].plot(
        t[::128] * 1e3, np.abs(arrays["receiver_clean_v"][::128]) * 1e6, ".", ms=2
    )
    axes[0, 1].set(
        xlabel="Time after pulse (ms)",
        ylabel="Absolute receiver sample (uV)",
        title="Clean FID samples",
    )
    axes[1, 0].plot(
        (t[:128] - t[0]) * 1e6, arrays["adc_reconstructed_v"][:128] * 1e3, lw=0.7
    )
    axes[1, 0].set(
        xlabel="Time from first valid sample (us)",
        ylabel="ADC reconstructed voltage (mV)",
        title="One seeded noisy realization",
    )
    axes[1, 1].bar(
        ["Analytic", "2048-trial MC"],
        [
            report["budget"]["matched_filter_snr_real"],
            report["monte_carlo"]["measured_matched_filter_snr_real"],
        ],
        color=["#205fa0", "#318b65"],
    )
    axes[1, 1].set(
        ylabel="Known-waveform real SNR", title="Matched-filter verification"
    )
    fig.suptitle(
        "Phase 1 absolute reference — 1 g point-equivalent sample; ideal receiver",
        fontsize=12,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "phase0/validate_phase0.py"),
            "--require-ready",
        ],
        check=True,
    )
    cfg = ReferenceConfig()
    report, arrays = build_reference(cfg)
    report["monte_carlo"] = monte_carlo(cfg, report, arrays)
    report["provenance"] = {
        "repository_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "numpy_version": np.__version__,
        "reference_source_sha256": hashlib.sha256(
            Path(__file__).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest(),
        "literature_source_sha256": hashlib.sha256(
            (HERE.parent / "phase0/literature_supplement.json")
            .read_bytes()
            .replace(b"\r\n", b"\n")
        ).hexdigest(),
    }
    limits = {
        key: (1e-4 if "coil_" in key or "waveform" in key else 1e-5)
        for key in report["checks"]
    }
    limits["time_vs_spectral_snr_relative_error"] = 1e-10
    limits["voltage_vs_field_snr_relative_error"] = 1e-10
    report["acceptance_limits"] = limits
    report["numerical_gate_passed"] = (
        all(report["checks"][k] <= v for k, v in limits.items())
        and report["monte_carlo"]["noise_sd_relative_error"] < 0.06
        and report["monte_carlo"]["four_average_ratio_relative_error"] < 0.12
        and report["monte_carlo"]["adc_clipped_samples"] == 0
        and abs(report["monte_carlo"]["noise_mean_standard_errors"]) < 5
    )
    report["monte_carlo_acceptance"] = {
        "noise_sd_relative_error": 0.06,
        "four_average_ratio_relative_error": 0.12,
        "absolute_mean_standard_errors": 5,
        "clipped_samples": 0,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(args.output / "reference_waveforms.npz", **arrays)
    record = result_record(cfg, report)
    from jsonschema import Draft202012Validator, FormatChecker

    schema = json.loads((HERE.parent / "phase0/result_record.schema.json").read_text())
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(record)
    (args.output / "result_record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    if args.plot:
        plot_reference(report, arrays, args.output / "reference.png")
    print(
        json.dumps(
            {
                "gate_passed": report["numerical_gate_passed"],
                "budget": report["budget"],
                "checks": report["checks"],
                "monte_carlo": report["monte_carlo"],
            },
            indent=2,
        )
    )
    if not report["numerical_gate_passed"]:
        raise SystemExit("Phase 1 numerical gate failed; inspect report")


if __name__ == "__main__":
    main()
