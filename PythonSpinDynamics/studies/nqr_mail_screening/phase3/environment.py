"""Explicit synthetic RFI priors and pulse-energy thermal accounting."""

from pathlib import Path
import sys
import json

import numpy as np
from spin_dynamics.interference.cancellers import (
    fit_gated_ridge_fir,
    adaptive_lms_canceller,
)
from spin_dynamics.thermal import ThermalNode, ThermalLink, ThermalNetwork

from acquisition import Receiver, receiver

PHASE2 = Path(__file__).resolve().parents[1] / "phase2"
sys.path.insert(0, str(PHASE2))
from aperture import candidates, field  # noqa: E402
from spatial_loading import combined_incident_integral  # noqa: E402

sys.path.remove(str(PHASE2))


def uniform_field_area(candidate):
    # Close the explicit port with a straight return. Lead routing is part of
    # this assumed common-mode response, and must be replaced by installed leads.
    p = candidates(128)[candidate].path_points
    closed = np.vstack((p, p[0]))
    return 0.5 * np.cross(closed[:-1], closed[1:]).sum(axis=0)


def rfi_sources(time, family, seed=735):
    rng = np.random.default_rng(seed)
    amplitude = 1 + 0.25 * np.sin(2 * np.pi * 80 * time)
    tone = amplitude * np.exp(2j * np.pi * 1800 * time)
    comb = sum(
        np.exp(2j * np.pi * f * time + 1j * p)
        for f, p in zip((3200, 4200, 5200), rng.uniform(0, 2 * np.pi, 3))
    ) / np.sqrt(3)
    impulses = np.zeros(len(time), complex)
    for centre in (0.0015, 0.0075, 0.012):
        impulses += (
            5
            * np.exp(-(((time - centre) / 30e-6) ** 2))
            * np.exp(2j * np.pi * 8000 * time)
        )
    refs = np.array([tone, comb, impulses])
    coefficients = np.array([1.0 + 0.3j, 0.7 - 0.2j, 0.4j])[:, None]
    if family == "drift":
        coefficients = coefficients * (1 + 0.7 * time / time[-1])
    scale = 0.004 if family == "overload" else 2e-6
    primary = scale * np.sum(coefficients * refs, axis=0)
    return primary, scale * refs


def cancellation_trial(record, family, method, cfg=Receiver(), seed=735):
    t, signal = record["time_s"], record["signal_v"]
    rfi, refs = rfi_sources(t, family, seed)
    # Axial spatially uniform magnetic interferer; gradiometer suppression is
    # computed from its actual signed loop area, not an assumed SNR advantage.
    area = uniform_field_area(record["candidate"])
    reference_area = uniform_field_area("enclosing_helix")
    magnetic_ratio = abs(area[2] / reference_area[2])
    if family == "nearby_source":
        point = np.array([[0.6, 0.2, 0.1]])
        coils = candidates(128)
        magnetic_ratio = abs(
            field(coils[record["candidate"]], point)[0, 2]
            / field(coils["enclosing_helix"], point)[0, 2]
        )
    rfi *= magnetic_ratio
    if method == "passive_20db":
        rfi *= 0.1  # synthetic shield attenuation; no assumed coil-Q change
    rng = np.random.default_rng(seed)
    psd = max(b["loaded_equivalent_resistance_ohm"] for b in record["blocks"])
    psd = 4 * 1.380649e-23 * record["temperature_k"] * psd + 1e-18
    noise = np.sqrt(psd * cfg.fs_hz / 2) * (
        rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))
    )
    leakage = record["tx_leakage_before_isolation_v"] * 10 ** (-cfg.isolation_db / 20)
    # Reference pickup is independently digitized, including its own receiver noise.
    reference_outputs = []
    reference_clipped = 0
    for ref in refs:
        n = np.sqrt(psd * cfg.fs_hz / 2) * (
            rng.normal(size=len(t)) + 1j * rng.normal(size=len(t))
        )
        out, _, margin = receiver(ref + n, np.ones(len(t), bool), cfg)
        reference_outputs.append(out / 2e-6)
        reference_clipped += margin["adc_clipped_samples"]
    references = np.array(reference_outputs)
    clean, _, _ = receiver(signal, record["receive_mask"], cfg)
    noisy, noisy_valid, noisy_margin = receiver(
        signal + noise + leakage, record["receive_mask"], cfg
    )
    primary, valid, margin = receiver(
        signal + noise + leakage + rfi, record["receive_mask"], cfg
    )
    fit = record["fit_mask"]
    if method in ("single_reference", "multi_reference", "protected_adaptive"):
        x = references[:1] if method == "single_reference" else references
        if method == "protected_adaptive":
            out = adaptive_lms_canceller(primary, x, fit, step=0.05)
            cleaned = out.cleaned

            def apply(y, z):
                return adaptive_lms_canceller(y, z, fit, step=0.05).cleaned
        else:
            model = fit_gated_ridge_fir(primary, x, fit, taps=3, ridge=1e-5)
            cleaned = model.apply(primary, x).cleaned

            def apply(y, z):
                return model.apply(y, z).cleaned

        # Test the digital canceller separately from quantization: the physical
        # 1 g signal is below an ADC LSB. A known template is injected AFTER ADC,
        # with zero calibration-window support. Never use this as a mass claim.
        template = signal / max(np.max(abs(signal)), 1e-30) * 1e-6
        template[fit] = 0
        difference = apply(primary + template, x) - apply(primary, x)
        injection_error = np.linalg.norm(difference - template) / np.linalg.norm(
            template
        )
        # A frozen fit alone cannot protect signal leaked into reference coils.
        contaminated = x + 0.1 * template[None, :] / 2e-6
        leaked = apply(primary + template, contaminated) - apply(primary, x)
        leakage_bias = np.linalg.norm(leaked - template) / np.linalg.norm(template)
    else:
        cleaned = primary.copy()
        injection_error = 0.0
        leakage_bias = None
    select = valid & noisy_valid
    residual = cleaned[select] - noisy[select]
    before = primary[select] - noisy[select]
    suppression = 10 * np.log10(
        max(np.vdot(before, before).real, 1e-50)
        / max(np.vdot(residual, residual).real, 1e-50)
    )
    summary = {
        "family": family,
        "method": method,
        "seed": seed,
        **margin,
        "noise_only_receiver": noisy_margin,
        "reference_clipped_samples": reference_clipped,
        "uniform_axial_field_pickup_ratio": float(magnetic_ratio),
        "residual_suppression_db": float(suppression)
        if np.linalg.norm(before) > 1e-20
        else None,
        "uncancelled_rfi_rms_v": float(np.sqrt(np.mean(abs(before) ** 2)))
        if len(before)
        else None,
        "residual_rms_v": float(np.sqrt(np.mean(abs(residual) ** 2)))
        if len(residual)
        else None,
        "protected_injection_relative_error": float(injection_error),
        "reference_signal_leakage_relative_bias": None
        if leakage_bias is None
        else float(leakage_bias),
        "fit_signal_overlap_samples": int(np.count_nonzero(fit & (abs(signal) > 0))),
        "receiver_feasible": bool(
            margin["adc_clipped_samples"] == 0
            and (reference_clipped == 0 or method in ("none", "passive_20db"))
        ),
        "reference_leakage_requires_rejection_or_calibration": bool(
            leakage_bias is not None and leakage_bias > 0.01
        ),
        "input_noise_psd_one_sided_v2_hz": psd,
    }
    arrays = {
        "clean_adc_v": clean,
        "noisy_adc_v": noisy,
        "rfi_adc_v": primary,
        "cancelled_adc_v": cleaned,
        "receiver_valid": valid,
        "references_adc_scaled": references,
    }
    return summary, arrays


def thermal_trial(record, phase2_root):
    """Pulse-wise network solve plus repetition steady state and peak bounds.

    Pulse+tail average powers conserve each pulse's analytic I^2 integral.
    Adiabatic E/C bounds cover within-pulse temperature ripple. The parcel is
    replaced each cycle; its steady-state node is used only as a conservative
    repeated-parcel bound, not a claim of physical parcel recirculation.
    """
    candidate = record["candidate"]
    wire = candidates(128)[candidate]
    length = np.linalg.norm(np.diff(wire.path_points, axis=0), axis=1).sum()
    copper_capacity = length * np.pi * 0.0015**2 * 8960 * 385
    capacities = {"coil": copper_capacity, "damper": 50.0, "parcel": 0.021 * 1200}
    ambient = record["temperature_k"]
    links = [
        ThermalLink("coil", "ambient", 0.5),
        ThermalLink("damper", "ambient", 1.0),
        ThermalLink("parcel", "ambient", 0.05),
    ]

    def network(temperatures, powers):
        nodes = [ThermalNode(k, c, temperatures[k]) for k, c in capacities.items()]
        nodes += [ThermalNode("ambient", None, ambient)]
        return ThermalNetwork(nodes, links, powers)

    report = json.loads((phase2_root / "gate2_report.json").read_text())
    refinements = report["candidates"][candidate]["spatial_loading"]["refinements"]
    net = record["network"]
    w = 2 * np.pi * 3304300
    zw = net["R"][2] + 1j * w * net["L"][2]
    # sigma<=1e-4: the uniform-contrast upper loss bound is sigma*int|Einc|^2.
    g_bound = max(
        1e-4
        * combined_incident_integral(
            r["refinements"][-1]["electric_integral_m"],
            r["refinements"][-1]["a_squared"],
            r["refinements"][-1]["e_dot_a"],
            3304300,
            zw,
        )
        for r in refinements
    )
    temps = {k: ambient for k in capacities}
    energies = {k: 0.0 for k in capacities}
    peak_ripple = {k: 0.0 for k in capacities}
    previous = 0.0
    trajectory = [(0.0, *temps.values())]
    for a, b, tau, voltage in record["rf_events"]:
        if a > previous:
            temps.update(
                network(temps, {}).transient(np.array([0, a - previous])).final()
            )
        duration = b - a
        fraction = 1 - np.exp(-duration / tau)
        effective = (
            duration
            - 2 * tau * fraction
            + tau / 2 * (1 - np.exp(-2 * duration / tau))
            + fraction**2 * tau / 2 * (1 - np.exp(-16))
        )
        block = max(
            (row for row in record["blocks"] if row["block_start_s"] <= a + 1e-12),
            key=lambda row: row["block_start_s"],
        )
        current = block["config"]["current_peak_a"]
        total_r = block["loaded_equivalent_resistance_ohm"]
        wire_r = block["coil_ac_resistance_ohm"]
        pulse_energy = {
            "coil": 0.5 * current**2 * wire_r * effective,
            "damper": 0.5 * current**2 * (total_r - wire_r) * effective,
            "parcel": 0.5 * voltage**2 * g_bound * effective,
        }
        interval = duration + 8 * tau
        temps.update(
            network(temps, {k: e / interval for k, e in pulse_energy.items()})
            .transient(np.array([0, interval]))
            .final()
        )
        for k, e in pulse_energy.items():
            energies[k] += e
            peak_ripple[k] = max(peak_ripple[k], e / capacities[k])
        previous = b + 8 * tau
        trajectory.append((previous, *(temps[k] for k in capacities)))
    cycle = max(b["cycle_s"] for b in record["blocks"])
    steady = network(
        {k: ambient for k in capacities}, {k: e / cycle for k, e in energies.items()}
    ).steady_state()
    component_peak = max(steady[k] + peak_ripple[k] for k in ("coil", "damper"))
    parcel_rise_bound = energies["parcel"] / capacities["parcel"]
    return {
        "energies_j": energies,
        "rf_energy_balance_relative_error": abs(
            energies["coil"] + energies["damper"] - record["rf_energy_j"]
        )
        / record["rf_energy_j"],
        "heat_capacities_j_k": capacities,
        "thermal_conductances_w_k": {"coil": 0.5, "damper": 1.0, "parcel": 0.05},
        "parcel_loss_conductance_upper_s": g_bound,
        "component_peak_bound_k": component_peak,
        "component_margin_k": 353.15 - component_peak,
        "parcel_rise_bound_k": parcel_rise_bound,
        "parcel_margin_k": 2 - parcel_rise_bound,
        "serial_cycle_s": cycle,
        "steady_temperatures_k": steady,
        "feasible": bool(component_peak < 353.15 and parcel_rise_bound < 2),
    }, np.array(trajectory)
