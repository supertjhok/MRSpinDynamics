"""Close the finite-pulse single-line budget before Phase 2.

I/Q integrate-and-dump reference receiver: independent nonoverlapping windows,
flat input noise. This is not the later tuned/colored/RFI receiver model.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from pulsed_study import Config, simulate, material_reference
from reference import ReferenceConfig, build_reference, monte_carlo, adc_quantize


def audit():
    cfg = Config(echoes=3)
    row, t, v, valid, d = simulate(cfg, return_details=True)
    _, line, _ = material_reference()
    flux = d["moment_am2"] * d["receive_beta_t_per_a"]
    independent_voltage = 2j * np.pi * line.frequency_hz * flux
    w = d["exposure_s"][valid]
    gain = 10000.0
    adc = ReferenceConfig(monte_carlo_trials=512)
    lsb = adc.adc_full_scale_v / (2**adc.adc_bits)
    sigma2 = gain**2 * d["noise_psd_one_sided_v2_hz"] / w
    clean = gain * v[valid]
    # Both I and Q ADCs have variance S_one-sided / integration time.
    x = np.r_[clean.real, clean.imag]
    variance = np.tile(sigma2 + lsb**2 / 12, 2)
    whitened = x / np.sqrt(variance)
    snr = np.linalg.norm(whitened)
    spectral = np.sqrt(np.sum(np.abs(np.fft.fft(whitened, norm="ortho")) ** 2))
    template = whitened / snr
    rng = np.random.default_rng(170141)
    scores = []
    clipped = 0
    for _ in range(32):
        noise = rng.normal(size=(64, len(x))) * np.sqrt(np.tile(sigma2, 2))
        quantized, _, nclip = adc_quantize(noise + x, adc)
        clipped += nclip
        scores.extend((quantized / np.sqrt(variance)) @ template)
    scores = np.asarray(scores)
    checks = {
        "moment_flux_voltage_relative_error": float(
            np.linalg.norm(v - independent_voltage) / np.linalg.norm(v)
        ),
        "time_frequency_snr_relative_error": float(abs(snr / spectral - 1)),
        "adc_vs_input_snr_relative_error": float(
            abs(snr / row["single_shot_snr_1g"] - 1)
        ),
        "monte_carlo_noise_sd_error": float(abs(scores.std(ddof=1) - 1)),
        "monte_carlo_mean_standard_errors": float(
            abs(scores.mean() - snr) * np.sqrt(len(scores))
        ),
        "sqrt_four_averages_error": float(
            abs(
                scores.std(ddof=1) / scores.reshape(-1, 4).mean(axis=1).std(ddof=1) / 2
                - 1
            )
        ),
        "clipped_samples": int(clipped),
    }
    reference_cfg = ReferenceConfig(monte_carlo_trials=512)
    reference, arrays = build_reference(reference_cfg)
    reference_mc = monte_carlo(reference_cfg, reference, arrays)
    limits = {
        "moment_flux_voltage_relative_error": 1e-12,
        "time_frequency_snr_relative_error": 1e-12,
        "adc_vs_input_snr_relative_error": 0.001,
        "monte_carlo_noise_sd_error": 0.06,
        "monte_carlo_mean_standard_errors": 5.0,
        "sqrt_four_averages_error": 0.12,
        "clipped_samples": 0,
    }
    passed = (
        all(checks[k] <= v for k, v in limits.items())
        and max(reference["checks"].values()) < 1e-4
    )
    preparation = simulate(replace(cfg, prepolarization=True))[0]
    # Explicit illustrative actuator budget, kept separate from RF spin physics.
    parcel_mass = 0.1
    efficiency = 0.5
    force = 1.0
    distance = (
        cfg.magnet_length_m / 2 + cfg.magnet_coil_spacing_m + cfg.coil_length_m / 2
    )
    mechanical_energy = (
        0.5 * parcel_mass * cfg.transport_velocity_m_s**2 + force * distance
    ) / efficiency
    return {
        "gate1_passed": bool(passed),
        "scope": "absolute single-line finite-pulse reference under declared receiver and material assumptions",
        "checks": checks,
        "tolerances": limits,
        "ideal_reference_checks": reference["checks"],
        "ideal_reference_monte_carlo": reference_mc,
        "finite_pulse_budget": row,
        "prepared_budget": preparation,
        "transport_energy_assumption": {
            "parcel_mass_kg": parcel_mass,
            "drive_efficiency": efficiency,
            "drag_force_n": force,
            "electrical_energy_j": mechanical_energy,
            "permanent_magnet_hold_power_w": 0.0,
            "scope": "one acceleration, no regenerative credit; illustrative, not measured",
        },
        "receiver": {
            "gain": gain,
            "adc_bits": adc.adc_bits,
            "adc_span_v": adc.adc_full_scale_v,
            "integration_windows": len(w),
            "acquired_s": float(w.sum()),
            "snr_1g": snr,
            "architecture": "two-channel I/Q integrate-and-dump reference with white input noise and dithered ADC",
        },
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "gate1_audit.py",
                "pulsed_study.py",
                "reference.py",
                "geometry.py",
            )
        },
    }


if __name__ == "__main__":
    result = audit()
    Path(__file__).with_name("gate1_report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"gate1_passed": result["gate1_passed"], "checks": result["checks"]},
            indent=2,
        )
    )
    if not result["gate1_passed"]:
        raise SystemExit(1)
