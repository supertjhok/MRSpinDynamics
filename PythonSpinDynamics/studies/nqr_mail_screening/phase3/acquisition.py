"""Stateful finite-pulse records and a causal, input-referred IQ receiver.

All RF voltages are peak phasors; complex IQ white noise has E|n|^2=S1*fs.
The independent real and imaginary quadratures each have variance S1*fs/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import lfilter

PHASE1 = Path(__file__).resolve().parents[1] / "phase1"
sys.path.insert(0, str(PHASE1))
from pulsed_study import Config, simulate, material_reference  # noqa: E402

sys.path.remove(str(PHASE1))


@dataclass(frozen=True)
class Receiver:
    fs_hz: float = 200000.0
    bandwidth_hz: float = 15000.0
    resonator_bandwidth_hz: float = 3304300.0 / 60.0
    gain: float = 1000.0
    rail_v: float = 1.0
    adc_bits: int = 18
    isolation_db: float = 140.0
    overload_recovery_s: float = 200e-6

    def __post_init__(self):
        if (
            min(
                self.fs_hz,
                self.bandwidth_hz,
                self.resonator_bandwidth_hz,
                self.gain,
                self.rail_v,
            )
            <= 0
        ):
            raise ValueError("receiver scales must be positive")
        if self.bandwidth_hz >= self.fs_hz / 2 or not 2 <= self.adc_bits <= 24:
            raise ValueError("receiver bandwidth/ADC out of range")
        if self.overload_recovery_s < 0:
            raise ValueError("negative recovery")


def lowpass(x, bandwidth, fs):
    pole = np.exp(-2 * np.pi * bandwidth / fs)
    return lfilter([1 - pole], [1, -pole], x)


def receiver(x, allowed, cfg=Receiver()):
    """Causal two-pole anti-alias chain, component clipping and recovery mask.

    The switch leaves finite TX leakage at the LNA input; invalid samples still
    drive filter and overload state. No digital cleaning can restore clipped data.
    """
    resonated = lowpass(x, cfg.resonator_bandwidth_hz, cfg.fs_hz)
    analog = (
        lowpass(
            lowpass(resonated, cfg.bandwidth_hz, cfg.fs_hz), cfg.bandwidth_hz, cfg.fs_hz
        )
        * cfg.gain
    )
    overload = (abs(analog.real) >= cfg.rail_v) | (abs(analog.imag) >= cfg.rail_v)
    valid = np.asarray(allowed, bool).copy()
    remaining = 0
    hold = int(np.ceil(cfg.overload_recovery_s * cfg.fs_hz))
    for i, hit in enumerate(overload):
        if hit:
            remaining = hold + 1
        if remaining:
            valid[i] = False
            remaining -= 1
    lsb = 2 * cfg.rail_v / 2**cfg.adc_bits
    upper = cfg.rail_v - lsb
    quantized = np.clip(
        np.round(analog.real / lsb) * lsb, -cfg.rail_v, upper
    ) + 1j * np.clip(np.round(analog.imag / lsb) * lsb, -cfg.rail_v, upper)
    return (
        quantized / cfg.gain,
        valid,
        {
            "adc_clipped_samples": int(np.count_nonzero(overload)),
            "receiver_rail_margin_v": float(
                cfg.rail_v - max(abs(analog.real).max(), abs(analog.imag).max())
            ),
            "valid_receive_samples": int(np.count_nonzero(valid)),
            "input_referred_lsb_v": lsb / cfg.gain,
            "overload_recovery_s": cfg.overload_recovery_s,
        },
    )


def phase2_inputs(root, candidate):
    report = json.loads((root / "gate2_report.json").read_text())
    if not report["gate2_passed"]:
        raise ValueError("Phase 2 gate has not passed")
    data = np.load(root / f"{candidate}_converged_map.npz")
    grid = tuple(data[k] for k in ("x_m", "y_m", "z_m"))
    interp = RegularGridInterpolator(grid, data["tx_t_a"], bounds_error=True)
    direction = interp([[0, 0, -0.115]])[0]
    direction /= np.linalg.norm(direction)

    def profile(points):
        vectors = interp(points)
        axial = vectors @ direction
        transverse = np.linalg.norm(vectors - axial[:, None] * direction, axis=1)
        if candidate == "axial_gradiometer":
            return vectors
        if np.max(transverse / np.maximum(abs(axial), 1e-30)) > 0.05:
            raise ValueError("fixed-axis trajectory exceeds 5%; use vector profile")
        return axial

    network = next(
        r
        for r in report["candidates"][candidate]["network_refinements"]
        if r["path"] == 64 and r["perimeter"] == 32 and r["formulation"] == "chain"
    )
    return profile, network


def schedule(
    root,
    candidate="enclosing_helix",
    temperature_k=293.15,
    prepolarization=False,
    resolution=1,
    fs_hz=200000.0,
    position_m=(0.0, 0.0, 0.0),
    magnet_dwell_s=1.0,
    coil_dwell_s=0.025,
    transfer_velocity_m_s=5.0,
    coil_stop_fraction=0.25,
    settling_s=0.005,
    repetitions=1,
):
    """One stationary 1 g packet after magnet dwell and transfer; phase-reset SLSE-x, FID-y, SORC-x, SLSE-x.

    Each packet temperature is an independent ensemble member. Both transitions
    share x-line Gaussian EFG disorder and x T1: a declared common-site prior,
    not independent measurements of y linewidth or SLSE/SORC relaxation.
    """
    profile, network = phase2_inputs(root, candidate)
    if not np.isfinite(magnet_dwell_s) or magnet_dwell_s < 0:
        raise ValueError("magnet dwell must be finite and nonnegative")
    if not np.isfinite(settling_s) or settling_s < 0:
        raise ValueError("settling time must be finite and nonnegative")
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    baseline = settling_s + 0.005
    gap = 0.001
    elapsed = 0.0
    # Uniform frequency grids recur after 1/df. Keep that artificial recurrence
    # beyond 1.5 times the complete observation, including retained-state repeats.
    _, _, material = material_reference()
    sigma = np.sqrt(2 * (1 - material["t2_star_s"] / 0.0144)) / (
        2 * np.pi * material["t2_star_s"]
    )
    observation = coil_dwell_s - baseline
    offset_points = disorder_grid_points(sigma, observation, resolution)
    history = None
    blocks = []
    segments = []
    events = []
    total_energy = 0.0
    for index, (sequence, line) in enumerate(
        (("SLSE", "x"), ("FID", "y"), ("SORC", "x"), ("SLSE", "x")) * repetitions
    ):
        recovery = gap if index else 0.0
        elapsed += recovery
        cfg = Config(
            sequence=sequence,
            line_id=line,
            echoes=4,
            powder_theta=4 * resolution,
            powder_phi=8 * resolution,
            offset_points=offset_points,
            pulse_steps=16 * resolution,
            dt_s=10e-6 / resolution,
            temperature_k=temperature_k,
            position_m=position_m,
            readout_start_fraction=coil_stop_fraction,
            motion_mode="stop_transfer_stop",
            preparation_s=magnet_dwell_s,
            coil_dwell_s=coil_dwell_s,
            transport_velocity_m_s=transfer_velocity_m_s,
            settling_s=baseline,
            prepolarization=prepolarization and index == 0,
        )
        frequency_index = 2 if line == "x" else 0
        electrical = (
            network["L"][frequency_index],
            network["R"][frequency_index] * (1 + 0.00393 * (temperature_k - 293.15)),
            network["C"],
        )
        post_acquisition = 0.0
        if index == 4 * repetitions - 1:
            sequence_time = cfg.echoes * cfg.spacing_s + cfg.excitation_s
            post_acquisition = coil_dwell_s - baseline - elapsed - sequence_time
            if post_acquisition < 0.002 - 1e-12:
                raise ValueError(
                    "coil dwell must contain all acquisitions and a 2 ms final observation guard"
                )
        row, t, v, valid, detail = simulate(
            cfg,
            return_details=True,
            initial_state=history,
            recovery_s=recovery,
            field_profile=profile,
            electrical=electrical,
            post_acquisition_s=post_acquisition,
        )
        row["coil_residence_remaining_s"] -= elapsed
        history = detail["history"]
        block_start = baseline + elapsed
        segments.append((block_start + t, v, valid, detail["exposure_s"]))
        for a, b in detail["rf_events_s"]:
            events.append(
                (
                    block_start + a,
                    block_start + b,
                    row["rf_envelope_time_constant_s"],
                    row["coil_reactive_voltage_peak_v"],
                )
            )
        row["block_start_s"] = block_start
        row["recovery_before_s"] = recovery
        row["history_deviation_from_equilibrium"] = float(
            np.linalg.norm(
                detail["start_state"][:, :9] - detail["equilibrium_state"][:, :9]
            )
        )
        blocks.append(row)
        total_energy += row["rf_energy_j"]
        elapsed += row["rf_train_s"]
    if baseline + elapsed + 0.002 > coil_dwell_s:
        raise ValueError(
            "coil dwell must contain settling, calibration, all RF/recovery blocks and 2 ms final guard"
        )
    t = np.arange(int(np.ceil(coil_dwell_s * fs_hz))) / fs_hz
    signal = np.zeros(t.size, complex)
    allowed = np.zeros(t.size, bool)
    # Interpolation is only within each free acquisition run; never across RF.
    # Spin receive samples include the blanked interval, so receiver memory sees
    # the freely evolving signal before acquisition opens.
    for times, volts, valid, exposure in segments:
        breaks = np.r_[
            0, np.flatnonzero(np.diff(times) > 1.5 * 10e-6 / resolution) + 1, len(times)
        ]
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            select = (t >= times[lo]) & (t <= times[hi - 1])
            signal[select] = np.interp(t[select], times[lo:hi], volts[lo:hi])
        for end, width in zip(times, exposure):
            allowed |= (t > end - width) & (t <= end)
    # Explicit quiet calibration before the first RF event. Entire acquisition
    # and final tail are protected, independently of target amplitude.
    fit = (t >= settling_s + 0.0005) & (t < baseline - 0.0001)
    leakage = np.zeros(t.size, complex)
    for a, b, tau, voltage in events:
        on = (t >= a) & (t < b)
        tail = (t >= b) & (t <= b + 8 * tau)
        leakage[on] += voltage * (1 - np.exp(-(t[on] - a) / tau))
        leakage[tail] += (
            voltage * (1 - np.exp(-(b - a) / tau)) * np.exp(-(t[tail] - b) / tau)
        )
    return {
        "time_s": t,
        "signal_v": signal,
        "receive_mask": allowed,
        "fit_mask": fit,
        "tx_leakage_before_isolation_v": leakage,
        "blocks": blocks,
        "rf_events": events,
        "rf_energy_j": total_energy,
        "duration_s": coil_dwell_s,
        "motion": motion_timeline(cfg, baseline, baseline + elapsed),
        "temperature_k": temperature_k,
        "candidate": candidate,
        "network": network,
        "sample_mass_kg": 0.001,
    }


def motion_timeline(cfg, rf_start, rf_end):
    """Piecewise constant-speed transport with ideal stops; no acceleration ramps.

    Coordinates are relative to the magnet centre. Coil-local record time zero
    is arrival at the measurement stop. Dwell includes settling and all guards.
    """
    magnet_start = -cfg.magnet_length_m / 2
    coil_stop = (
        cfg.magnet_coil_spacing_m
        + (cfg.readout_start_fraction - 0.5) * cfg.coil_length_m
    )
    coil_exit = cfg.magnet_coil_spacing_m + cfg.coil_length_m / 2
    clock = 0.0
    stages = []
    for name, start, stop, duration in (
        ("entry", magnet_start, 0.0, -magnet_start / cfg.transport_velocity_m_s),
        ("magnet_dwell", 0.0, 0.0, cfg.preparation_s),
        (
            "polarization_transfer",
            0.0,
            coil_stop,
            coil_stop / cfg.transport_velocity_m_s,
        ),
        ("coil_dwell", coil_stop, coil_stop, cfg.coil_dwell_s),
        (
            "exit",
            coil_stop,
            coil_exit,
            (coil_exit - coil_stop) / cfg.transport_velocity_m_s,
        ),
        ("handling", coil_exit, coil_exit, 2.0),
    ):
        stages.append(
            {
                "stage": name,
                "start_s": clock,
                "end_s": clock + duration,
                "start_z_m": start,
                "end_z_m": stop,
            }
        )
        clock += duration
    arrival = stages[3]["start_s"]
    return {
        "stages": stages,
        "cycle_s": clock,
        "coil_arrival_s": arrival,
        "rf_start_s": arrival + rf_start,
        "rf_end_s": arrival + rf_end,
        "model": "ideal stops with constant-speed entry, transfer and exit; acceleration ramps not included",
    }


def disorder_grid_points(sigma_hz, observation_s, refinement=1):
    """Resolve a +/-4 sigma grid beyond 1.5 observation times before recurrence."""
    count = max(81, int(np.ceil(12 * sigma_hz * max(observation_s, 0.0))) + 1)
    count += (count + 1) % 2
    return refinement * (count - 1) + 1
