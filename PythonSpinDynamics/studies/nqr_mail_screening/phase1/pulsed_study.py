"""Finite-pulse engineering comparison using the package's full spin dynamics.

Run from PythonSpinDynamics; see PULSED_MODEL.md for assumptions and limitations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr
from geometry import for_config, spacing_sweep

from spin_dynamics.fields.coils import solenoid
from spin_dynamics.fields.coil_properties import solenoid_properties
from spin_dynamics.fields.magnetostatics import biot_savart
from spin_dynamics.nqr.full_dynamics import (
    pulse_hamiltonian,
    static_hamiltonian_rotating,
)
from spin_dynamics.nqr.orientations import powder_average_grid
from spin_dynamics.nqr.polarization_enhancement import (
    CylindricalSampleGeometry,
    HalbachPrepolarizationMagnet,
    LinearTransportMotion,
    PolarizationEnhancedNQRSample,
    simulate_adiabatic_polarization_transfer,
)
from spin_dynamics.relaxation import (
    NQRRelaxationModel,
    liouville_superoperator,
    relaxation_superoperator,
)
from reference import (
    H,
    KB,
    GAMMA_HZ_T,
    ReferenceConfig,
    material_reference,
    nuclei_count,
    thermal_deviation,
)


@dataclass(frozen=True)
class Config:
    sequence: str = "SLSE"
    coil_radius_m: float = 0.18
    coil_length_m: float = 0.46
    coil_turns: int = 6
    wire_diameter_m: float = 0.003
    magnet_radius_m: float = 0.23
    magnet_length_m: float = 0.46
    magnet_rod_width_m: float = 0.08
    magnet_remanence_t: float = 1.15
    magnet_coil_spacing_m: float = 1.2  # centre to centre, along z
    readout_start_fraction: float = 0.25  # fraction of coil length from entrance

    current_peak_a: float = 8.0
    loaded_q: float = 30.0
    pulse_steps: int = 16
    excitation_s: float = 30e-6
    pulse_s: float = 60e-6
    spacing_s: float = 600e-6
    echoes: int = 10
    dt_s: float = 20e-6
    blank_s: float = 100e-6
    t2_s: float = 0.0144  # provisional CPMG prior, NOT intrinsic measured T2
    detuning_hz: float = 0.0
    temperature_k: float = 293.15
    powder_theta: int = 4
    powder_phi: int = 8
    offset_points: int = 41
    position_m: tuple = (0.0, 0.0, 0.0)
    preparation_s: float = 0.0
    prepolarization: bool = False
    transport_velocity_m_s: float = 5.0
    proton_t1_s: float = 1.0
    coupling_hz: float = 1000.0
    settling_s: float = 0.005

    def __post_init__(self):
        for name in (
            "coil_radius_m",
            "coil_length_m",
            "wire_diameter_m",
            "magnet_radius_m",
            "magnet_length_m",
            "magnet_rod_width_m",
            "magnet_remanence_t",
            "magnet_coil_spacing_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.coil_radius_m < np.hypot(0.17, 0.0175) or self.coil_length_m < 0.43:
            raise ValueError("coil must fit the declared clear mail volume")
        if 2 * self.magnet_radius_m - self.magnet_rod_width_m < 0.34:
            raise ValueError("magnet must clear the mail width")
        if not 0 <= self.readout_start_fraction < 1:
            raise ValueError("readout start fraction must be in [0,1)")
        if (
            min(self.pulse_steps, self.coil_turns, self.powder_theta, self.powder_phi)
            < 1
            or self.offset_points < 3
        ):
            raise ValueError(
                "integration counts must be positive; offsets need >=3 points"
            )
        if self.sequence not in ("SLSE", "SORC"):
            raise ValueError("sequence must be SLSE or SORC")
        for name in (
            "excitation_s",
            "pulse_s",
            "spacing_s",
            "dt_s",
            "t2_s",
            "temperature_k",
            "transport_velocity_m_s",
            "proton_t1_s",
            "coupling_hz",
            "loaded_q",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        if (
            min(self.current_peak_a, self.preparation_s, self.settling_s, self.blank_s)
            < 0
        ):
            raise ValueError("current and times cannot be negative")
        if self.spacing_s <= self.pulse_s + 2 * self.blank_s or self.echoes < 1:
            raise ValueError("pulse spacing must leave a receive window")


def affine_generator(hamiltonian, relaxation, equilibrium):
    """Thermal source is -R rho_eq, including during RF (not -L rho_eq)."""
    out = np.zeros((10, 10), complex)
    out[:9, :9] = liouville_superoperator(hamiltonian, relaxation)
    out[:9, 9] = -relaxation_superoperator(3, relaxation) @ equilibrium.reshape(
        9, order="F"
    )
    return out


@lru_cache(maxsize=32)
def hardware(radius=0.18, length=0.46, turns=6, wire=0.003):
    eig, line, _ = material_reference()
    properties = solenoid_properties(
        diameter=2 * radius,
        length=length,
        turns=turns,
        wire_diameter=wire,
        frequency=line.frequency_hz,
    )
    if (
        not np.isfinite(properties.inductance_effective)
        or properties.inductance_effective <= 0
        or properties.self_resonant_frequency <= line.frequency_hz
    ):
        raise ValueError("coil is outside its usable lumped-circuit regime")
    segments = solenoid(radius=radius, length=length, turns=turns, n_segments=128)
    return properties, segments


@lru_cache(maxsize=32)
def magnet_profile(radius=0.23, length=0.46, width=0.08, remanence=1.15, stop=1.2):
    # 0.38 m clear bore: envelope width plus clearance fits this study magnet.
    magnet = HalbachPrepolarizationMagnet(
        center_radius=radius,
        length=length,
        rod_width=width,
        remanence=remanence,
        n_cross=3,
        n_length=21,
    )
    z = np.linspace(-length / 2 - 0.02, stop + 0.02, 701)
    points = np.column_stack((z * 0, z * 0, z))
    return z, magnet.b0_magnitude(points)


def prepared_density(cfg):
    eig, line, values = material_reference()
    equilibrium = thermal_deviation(eig.levels_hz, cfg.temperature_k)
    if not cfg.prepolarization and cfg.preparation_s == 0:
        return equilibrium, {"enhancement": 1.0, "preparation_and_transport_s": 0.0}
    stop = (
        cfg.magnet_coil_spacing_m
        + (cfg.readout_start_fraction - 0.5) * cfg.coil_length_m
        - cfg.transport_velocity_m_s * cfg.settling_s
    )
    if stop <= 0:
        raise ValueError("settling distance extends before the magnet centre")
    z, field = magnet_profile(
        cfg.magnet_radius_m,
        cfg.magnet_length_m,
        cfg.magnet_rod_width_m,
        cfg.magnet_remanence_t,
        stop,
    )
    # Proton longitudinal buildup on the incoming half-magnet path, in field
    # units. Fold this and an optional centre dwell into the worked model's
    # effective preparation time; its crossing-efficiency model is unchanged.
    incoming = np.linspace(-cfg.magnet_length_m / 2, 0.0, 301)
    incoming_field = np.interp(incoming, z, field)
    step = (incoming[1] - incoming[0]) / cfg.transport_velocity_m_s
    polarized_field = 0.0
    for target in (incoming_field[:-1] + incoming_field[1:]) / 2:
        polarized_field = target + (polarized_field - target) * np.exp(
            -step / cfg.proton_t1_s
        )
    centre_field = float(np.interp(0.0, z, field))
    polarized_field = centre_field + (polarized_field - centre_field) * np.exp(
        -cfg.preparation_s / cfg.proton_t1_s
    )
    fraction = np.clip(polarized_field / centre_field, 0.0, 1.0 - 1e-15)
    effective_preparation = -cfg.proton_t1_s * np.log1p(-fraction)
    sample = PolarizationEnhancedNQRSample(
        (line.frequency_hz,),
        ("aniline x",),
        protons_per_molecule=29,
        nitrogens_per_molecule=2,
        proton_t1_seconds=cfg.proton_t1_s,
        nitrogen_t1_seconds=values["t1_s"],
        proton_linewidth_hz=80000.0,
        proton_nitrogen_coupling_hz=cfg.coupling_hz,
    )
    result = simulate_adiabatic_polarization_transfer(
        lambda points: np.interp(points[..., 2], z, field),
        sample,
        CylindricalSampleGeometry(
            length=0.02, diameter=0.008, axial_points=5, radial_rings=0
        ),
        LinearTransportMotion(0.0, stop, cfg.transport_velocity_m_s),
        prepolarization_time_seconds=effective_preparation,
        path_points=501,
    )
    enhancement = 1 + (float(result.practical_enhancement[0]) - 1) * np.exp(
        -cfg.settling_s / values["t1_s"]
    )
    state = equilibrium.copy()
    low, high = line.lower, line.upper
    change = 0.5 * (enhancement - 1) * (state[low, low] - state[high, high])
    state[low, low] += change
    state[high, high] -= change
    if np.min(np.linalg.eigvalsh(state + np.eye(3) / 3)) < 0:
        raise ValueError("unphysical prepared density")
    return state, {
        "enhancement": enhancement,
        "preparation_and_transport_s": cfg.preparation_s
        + result.travel_time_seconds
        + cfg.settling_s,
        "max_field_t": float(field.max()),
        "incoming_proton_polarization_fraction": float(fraction),
        "effective_preparation_s": float(effective_preparation),
        "end_field_t": float(np.interp(stop, z, field)),
        "mean_transfer_efficiency": float(np.mean(result.transfer_efficiency)),
        "crossing_positions_m": result.crossing_positions.tolist(),
        "model": "built-in adiabatic transfer; single-line population embedding",
    }


def simulate(cfg=Config(), *, return_details=False):
    eig, line, values = material_reference()
    properties, segments = hardware(
        cfg.coil_radius_m, cfg.coil_length_m, cfg.coil_turns, cfg.wire_diameter_m
    )
    start_z = (cfg.readout_start_fraction - 0.5) * cfg.coil_length_m
    path_times = np.linspace(0.0, cfg.echoes * cfg.spacing_s + cfg.excitation_s, 101)
    path_positions = np.tile(np.asarray(cfg.position_m), (101, 1))
    path_positions[:, 2] += start_z + cfg.transport_velocity_m_s * path_times
    path_beta = np.linalg.norm(
        biot_savart(path_positions, segments, current=1.0), axis=1
    )

    def beta_at(time):
        return float(np.interp(time, path_times, path_beta))

    beta = beta_at(0.0)
    requested_train = cfg.echoes * cfg.spacing_s + (
        cfg.excitation_s if cfg.sequence == "SLSE" else 0.0
    )
    residence = (
        (1 - cfg.readout_start_fraction)
        * cfg.coil_length_m
        / cfg.transport_velocity_m_s
    )
    if requested_train > residence:
        raise ValueError(
            "pulse train extends past the coil exit; shorten train or reduce speed"
        )
    equilibrium = thermal_deviation(eig.levels_hz, cfg.temperature_k)
    initial, preparation = prepared_density(cfg)
    relaxation = NQRRelaxationModel(t1_seconds=values["t1_s"], t2_seconds=cfg.t2_s)
    carrier = line.frequency_hz + cfg.detuning_hz
    resistance = max(
        properties.ac_resistance, properties.reactance_effective / cfg.loaded_q
    )
    tau = properties.inductance_effective * 2 / resistance
    if 8 * tau >= min(cfg.blank_s, (cfg.spacing_s - cfg.pulse_s) / 2):
        raise ValueError("ringdown exceeds the allocated blanking/free interval")
    # Gaussian static disorder: its envelope reaches 1/e at the measured T2*.
    # Homogeneous decay is included separately; this is an explicit line-shape prior.
    sigma = np.sqrt(2 * max(0.0, 1 - values["t2_star_s"] / cfg.t2_s)) / (
        2 * np.pi * values["t2_star_s"]
    )
    if sigma <= 0:
        raise ValueError("T2 must exceed measured T2-star for this disorder model")
    offsets = np.linspace(-4 * sigma, 4 * sigma, cfg.offset_points)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    free, pulse_x, pulse_y, detectors, ensemble_weights = [], [], [], [], []
    for orientation in powder_average_grid(cfg.powder_theta, cfg.powder_phi):
        for offset, weight in zip(offsets, weights):
            # Shift both upper-band transitions by a common static disorder offset.
            disorder = np.diag(
                2
                * np.pi
                * offset
                * np.round((eig.levels_hz - eig.levels_hz.min()) / carrier)
            )
            h0 = static_hamiltonian_rotating(eig, carrier) + disorder
            free.append(affine_generator(h0, relaxation, equilibrium))
            for phase, output in ((0.0, pulse_x), (np.pi / 2, pulse_y)):
                # B_peak cos(wt) has co-rotating field B_peak/2.
                hp = (
                    pulse_hamiltonian(
                        eig,
                        nutation_hz=GAMMA_HZ_T * beta * cfg.current_peak_a / 2,
                        rf_frequency_hz=carrier,
                        phase=phase,
                        b1_direction_pas=orientation.b1_direction_pas,
                    )
                    + disorder
                )
                output.append(affine_generator(hp, relaxation, equilibrium))
            from spin_dynamics.nqr.full_dynamics import rf_operator_eigenbasis

            op = rf_operator_eigenbasis(eig, orientation.b1_direction_pas)
            detector = np.zeros(10, complex)
            detector[line.upper + 3 * line.lower] = op[line.lower, line.upper]
            detectors.append(detector)
            ensemble_weights.append(orientation.weight * weight)
    free = np.array(free)
    pulse_x, pulse_y = np.array(pulse_x), np.array(pulse_y)
    detectors, ensemble_weights = np.array(detectors), np.array(ensemble_weights)
    state = np.tile(np.r_[initial.reshape(9, order="F"), 1.0], (len(free), 1))
    cache = {}

    def advance(generator, duration, key):
        nonlocal state
        if key[0] == "free":
            if key not in cache:
                cache[key] = expm(generator * duration)
            matrix = cache[key]
        else:
            matrix = expm(generator * duration)
        state = np.einsum("nij,nj->ni", matrix, state)

    times, signal, valid, receive_beta, exposure = [], [], [], [], []
    time = 0.0
    last_pulse_end = -np.inf
    rf_on_s = 0.0
    pending_tail = 0.0

    def pulse(generator, duration, key):
        nonlocal time, last_pulse_end, rf_on_s, pending_tail
        # Delivered current follows a first-order resonator, including its RF tail.
        field_scale = beta_at(time) / beta
        key = (key, round(field_scale, 12))
        for k in range(cfg.pulse_steps):
            fraction = field_scale * (
                1 - np.exp(-(k + 0.5) * duration / (cfg.pulse_steps * tau))
            )
            advance(
                free + fraction * (generator - free),
                duration / cfg.pulse_steps,
                (key, "rise", k),
            )
        time += duration
        last_pulse_end = time
        end_fraction = 1 - np.exp(-duration / tau)
        for k in range(cfg.pulse_steps):
            fraction = (
                field_scale * end_fraction * np.exp(-(k + 0.5) * 8 / cfg.pulse_steps)
            )
            advance(
                free + fraction * (generator - free),
                8 * tau / cfg.pulse_steps,
                (key, "tail", k),
            )
        time += 8 * tau
        pending_tail = 8 * tau
        rf_on_s += (
            duration
            - 2 * tau * end_fraction
            + tau / 2 * (1 - np.exp(-2 * duration / tau))
            + end_fraction**2 * tau / 2 * (1 - np.exp(-16))
        )

    def acquire(duration):
        nonlocal time, pending_tail
        duration -= pending_tail
        pending_tail = 0.0
        count = int(round(duration / cfg.dt_s))
        step = duration / count
        for _ in range(count):
            advance(free, step, ("free", step))
            time += step
            times.append(time)
            receive_beta.append(beta_at(time))
            signal.append(
                np.sum(ensemble_weights * np.einsum("ni,ni->n", detectors, state))
            )
            exposure.append(min(step, max(0.0, time - last_pulse_end - cfg.blank_s)))
            valid.append(exposure[-1] > 0.0)

    if cfg.sequence == "SLSE":
        pulse(pulse_x, cfg.excitation_s, "excitation")
        for _ in range(cfg.echoes):
            acquire((cfg.spacing_s - cfg.pulse_s) / 2)
            pulse(pulse_y, cfg.pulse_s, "refocus")
            acquire((cfg.spacing_s - cfg.pulse_s) / 2)
    else:
        # Transient same-phase SORC train; no free steady-state initialization.
        for _ in range(cfg.echoes):
            pulse(pulse_x, cfg.pulse_s, "sorc")
            acquire(cfg.spacing_s - cfg.pulse_s)
    times, valid = np.array(times), np.array(valid)
    moment = 2 * nuclei_count(ReferenceConfig()) * H * GAMMA_HZ_T * np.array(signal)
    voltage = 1j * 2 * np.pi * line.frequency_hz * np.asarray(receive_beta) * moment
    # Input-referred, flat receiver bound; SNR^2=int |V_phasor|^2/S_one-sided dt.
    psd = 4 * KB * cfg.temperature_k * resistance + 1e-18
    snr2 = float(np.sum(np.abs(voltage) ** 2 * np.asarray(exposure)) / psd)
    recovery = 5 * values["t1_s"]
    # Serial one-parcel path; transport continues through the remaining coil.
    path_time = (
        cfg.magnet_length_m / 2 + cfg.magnet_coil_spacing_m + cfg.coil_length_m / 2
    ) / cfg.transport_velocity_m_s
    cycle = cfg.preparation_s + path_time + 2.0
    voltage_peak = cfg.current_peak_a * properties.reactance_effective
    geometry_check = for_config(cfg)
    row = {
        "eligible_for_comparison": bool(
            geometry_check["passes"]
            and cfg.current_peak_a <= 50
            and voltage_peak <= 2000
        ),
        "config": asdict(cfg),
        "prepolarization": preparation,
        "zeeman_spacing_constraint": geometry_check,
        "beta_t_per_a": beta,
        "coil_ac_resistance_ohm": properties.ac_resistance,
        "loaded_equivalent_resistance_ohm": resistance,
        "rf_envelope_time_constant_s": tau,
        "coil_inductance_h": properties.inductance_effective,
        "coil_self_resonance_hz": properties.self_resonant_frequency,
        "coil_reactive_voltage_peak_v": voltage_peak,
        "within_provisional_current_voltage_limits": bool(
            cfg.current_peak_a <= 50 and voltage_peak <= 2000
        ),
        "rf_energy_j": 0.5 * cfg.current_peak_a**2 * resistance * rf_on_s,
        "rf_train_s": time,
        "reference_five_t1_recovery_s": recovery,
        "cycle_s": cycle,
        "coil_residence_remaining_s": residence,
        "readout_start_z_relative_coil_m": start_z,
        "readout_end_z_relative_coil_m": start_z + time * cfg.transport_velocity_m_s,
        "beta_end_t_per_a": float(receive_beta[-1]),
        "geometry_status": "design variables; not selected hardware",
        "timing_model": "serial path from magnet entrance through coil exit plus 2 s handling; optional preparation dwell",
        "single_shot_snr_1g": np.sqrt(snr2),
        "snr_per_sqrt_second_1g": np.sqrt(snr2 / cycle),
        "gaussian_oracle_auc_1g_single_shot": float(ndtr(np.sqrt(snr2 / 2))),
        "max_density_trace_error": float(
            np.max(np.abs(state[:, [0, 4, 8]].sum(axis=1)))
        ),
    }
    if return_details:
        return (
            row,
            times,
            voltage,
            valid,
            {
                "moment_am2": moment,
                "receive_beta_t_per_a": np.asarray(receive_beta),
                "exposure_s": np.asarray(exposure),
                "noise_psd_one_sided_v2_hz": psd,
            },
        )
    return row, times, voltage, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".tmp/nqr_pulsed"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--config", type=Path, help="JSON Config overrides; run one scenario"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configurations = [
        replace(Config(), sequence=sequence, preparation_s=prep)
        for sequence in ("SLSE", "SORC")
        for prep in (0.0, 1.0)
    ]
    if not args.quick:
        configurations += [
            replace(Config(), prepolarization=True),
            replace(Config(), coil_radius_m=0.22),
            replace(Config(), detuning_hz=1500.0),
            replace(Config(), t2_s=0.003),
            replace(Config(), current_peak_a=30.0),
            replace(Config(), preparation_s=1.0, transport_velocity_m_s=1.0),
            replace(Config(), preparation_s=1.0, coupling_hz=300.0),
        ]
    if args.config:
        configurations = [Config(**json.loads(args.config.read_text()))]
    rows = []
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cfg in enumerate(configurations):
        row, times, voltage, valid = simulate(cfg)
        rows.append(row)
        np.savez_compressed(
            args.output / f"waveform_{i}.npz",
            time_s=times,
            voltage_v=voltage,
            valid=valid,
        )
        if i < 4:
            ax.plot(
                times[valid] * 1e3,
                np.abs(voltage[valid]) * 1e9,
                label=f"{cfg.sequence}, preparation {cfg.preparation_s:g} s",
            )
        print(
            i,
            cfg.sequence,
            cfg.preparation_s,
            row["snr_per_sqrt_second_1g"],
            flush=True,
        )
    ax.set(
        xlabel="Time (ms)",
        ylabel="Receive EMF magnitude (nV), 1 g",
        title="Finite-pulse trains; blanked samples excluded",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "pulsed_comparison.png", dpi=160)
    report = {
        "status": "engineering sensitivity model; no scanner ROC validation",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "supporting_source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("geometry.py", "reference.py")
        },
        "spacing_sweep": spacing_sweep(Config()),
        "rows": rows,
    }
    (args.output / "pulsed_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
