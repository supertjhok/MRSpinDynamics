"""Compare conventional preamplifier decoupling and receiver front ends."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import extract_multiport_impedance, helical_solenoid
from spin_dynamics.workflows import (
    ActiveReceiverNetwork,
    LNAInputModel,
    PassiveTwoPort,
    analyze_active_receiver_front_end_sweep,
    identity_two_port,
    mutual_cancellation_capacitance,
    shared_capacitor_mesh_impedance,
    transmission_line_two_port,
)


def _series_capacitor_sweep(
    frequencies_hz: np.ndarray,
    capacitances_f: np.ndarray,
) -> np.ndarray:
    omega = 2.0 * np.pi * frequencies_hz
    result = np.zeros(
        (frequencies_hz.size, capacitances_f.size, capacitances_f.size),
        dtype=np.complex128,
    )
    for index, value in enumerate(omega):
        result[index] = np.diag(1.0 / (1j * value * capacitances_f))
    return result


def _coupled_coils(frequencies_hz: np.ndarray):
    first = helical_solenoid(
        diameter=0.024,
        length=0.025,
        turns=4,
        wire_radius=0.4e-3,
        n_per_turn=12,
        n_radial=1,
        n_angular=4,
        axis="x",
    )
    second = dataclasses.replace(
        first,
        path_points=first.path_points + np.array([0.0, 0.014, 0.0]),
    )
    return extract_multiport_impedance((first, second), frequencies_hz)


def _line_sweep(
    frequencies_hz: np.ndarray,
    target_frequency_hz: float,
    phase_at_target_rad: float,
    attenuation_db: float,
) -> tuple[tuple[PassiveTwoPort, PassiveTwoPort], ...]:
    return tuple(
        (
            transmission_line_two_port(
                50.0,
                phase_at_target_rad * frequency / target_frequency_hz,
                attenuation_db=attenuation_db
                * np.sqrt(frequency / target_frequency_hz),
            ),
        )
        * 2
        for frequency in frequencies_hz
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 50-ohm loading, low-Z preamplifier decoupling, "
            "on-coil high-Z reception, and combined passive cancellation."
        )
    )
    parser.add_argument("--frequency-mhz", type=float, default=2.0)
    parser.add_argument("--span-percent", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument("--cable-loss-db", type=float, default=0.25)
    parser.add_argument("--low-z-ohm", type=float, default=2.0)
    parser.add_argument("--tolerance-points", type=int, default=41)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.frequency_mhz <= 0.0:
        raise ValueError("--frequency-mhz must be positive")
    if not 0.0 < args.span_percent < 100.0:
        raise ValueError("--span-percent must lie between 0 and 100")
    if args.points < 3 or args.points % 2 == 0:
        raise ValueError("--points must be an odd integer of at least 3")
    if args.cable_loss_db < 0.0:
        raise ValueError("--cable-loss-db must be non-negative")
    if args.low_z_ohm <= 0.0:
        raise ValueError("--low-z-ohm must be positive")
    if args.tolerance_points < 5:
        raise ValueError("--tolerance-points must be at least 5")
    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    target_frequency = args.frequency_mhz * 1.0e6
    fractional_span = args.span_percent / 100.0
    frequencies = np.linspace(
        target_frequency * (1.0 - fractional_span),
        target_frequency * (1.0 + fractional_span),
        args.points,
    )
    target_index = args.points // 2
    omega0 = 2.0 * np.pi * target_frequency
    peec = _coupled_coils(frequencies)
    sample_loss_ohm = np.array([[6.0, 0.05], [0.05, 6.0]])
    coil_impedance = peec.impedance + sample_loss_ohm[np.newaxis, :, :]
    inductance0 = peec.inductance[target_index]
    mutual_inductance = float(inductance0[0, 1])

    baseline_capacitances = 1.0 / (omega0**2 * np.diag(inductance0))
    source_baseline = coil_impedance + _series_capacitor_sweep(
        frequencies,
        baseline_capacitances,
    )
    cancellation_capacitance = mutual_cancellation_capacitance(
        mutual_inductance,
        target_frequency,
    )
    branch_signs = (1, 1) if mutual_inductance > 0.0 else (1, -1)
    cancelled_inductances = np.diag(inductance0) - abs(mutual_inductance)
    cancelled_capacitances = 1.0 / (omega0**2 * cancelled_inductances)
    source_cancelled = (
        coil_impedance
        + _series_capacitor_sweep(frequencies, cancelled_capacitances)
        + shared_capacitor_mesh_impedance(
            frequencies,
            cancellation_capacitance,
            branch_signs=branch_signs,
            series_resistance_ohm=0.02,
        )
    )

    matched_lna = LNAInputModel(
        input_resistance_ohm=50.0,
        voltage_noise_density_v_per_sqrt_hz=0.45e-9,
        current_noise_density_a_per_sqrt_hz=2.0e-12,
        voltage_gain_v_per_v=100.0,
        output_noise_density_v_per_sqrt_hz=2.0e-9,
    )
    low_z_lna = dataclasses.replace(
        matched_lna,
        input_resistance_ohm=args.low_z_ohm,
    )
    high_z_lna = LNAInputModel(
        input_resistance_ohm=1.0e6,
        input_capacitance_f=5.0e-12,
        voltage_noise_density_v_per_sqrt_hz=1.2e-9,
        current_noise_density_a_per_sqrt_hz=5.0e-15,
        voltage_gain_v_per_v=100.0,
        output_noise_density_v_per_sqrt_hz=2.0e-9,
    )
    cable = _line_sweep(
        frequencies,
        target_frequency,
        np.pi / 2.0,
        args.cable_loss_db,
    )
    identity = tuple((identity_two_port(),) * 2 for _ in frequencies)
    architectures = {
        "Remote matched 50 ohm": analyze_active_receiver_front_end_sweep(
            frequencies,
            source_baseline,
            lna_input_models=matched_lna,
            front_end_two_ports=cable,
            noise_bandwidth_hz=10.0e3,
        ),
        "Low-Z + quarter-wave": analyze_active_receiver_front_end_sweep(
            frequencies,
            source_baseline,
            lna_input_models=low_z_lna,
            front_end_two_ports=cable,
            noise_bandwidth_hz=10.0e3,
        ),
        "On-coil high-Z": analyze_active_receiver_front_end_sweep(
            frequencies,
            source_baseline,
            lna_input_models=high_z_lna,
            front_end_two_ports=identity,
            noise_bandwidth_hz=10.0e3,
        ),
        "Passive + low-Z": analyze_active_receiver_front_end_sweep(
            frequencies,
            source_cancelled,
            lna_input_models=low_z_lna,
            front_end_two_ports=cable,
            noise_bandwidth_hz=10.0e3,
        ),
    }
    colors = {
        "Remote matched 50 ohm": "tab:blue",
        "Low-Z + quarter-wave": "tab:orange",
        "On-coil high-Z": "tab:green",
        "Passive + low-Z": "tab:red",
    }

    phase_errors_deg = np.linspace(
        -25.0,
        25.0,
        args.tolerance_points,
    )
    input_resistances = np.geomspace(0.5, 10.0, args.tolerance_points)
    isolation_grid = np.empty(
        (input_resistances.size, phase_errors_deg.size),
    )
    noise_figure_grid = np.empty_like(isolation_grid)
    target_source = source_baseline[target_index]
    unit_maps = np.eye(2, dtype=np.complex128)
    drive = np.array([1.0, 0.0], dtype=np.complex128)
    for row, resistance in enumerate(input_resistances):
        model = dataclasses.replace(low_z_lna, input_resistance_ohm=resistance)
        for column, phase_error in enumerate(phase_errors_deg):
            line = transmission_line_two_port(
                50.0,
                np.pi / 2.0 + np.deg2rad(phase_error),
                attenuation_db=args.cable_loss_db,
            )
            network = ActiveReceiverNetwork(
                frequency_hz=target_frequency,
                coil_impedance_ohm=target_source,
                lna_input_models=model,
                front_end_two_ports=line,
                noise_bandwidth_hz=10.0e3,
            )
            solution = network.solve(unit_maps)
            currents = np.linalg.solve(network.total_impedance_ohm, drive)
            isolation_grid[row, column] = -20.0 * np.log10(
                abs(currents[1] / currents[0])
            )
            noise_figure_grid[row, column] = float(
                np.mean(solution.noise_figure_db)
            )

    cable_losses = np.linspace(0.0, 2.0, args.tolerance_points)
    cable_noise_figures: dict[str, np.ndarray] = {}
    for label, model in (
        ("Remote matched 50 ohm", matched_lna),
        ("Low-Z + quarter-wave", low_z_lna),
    ):
        values = np.empty_like(cable_losses)
        for index, loss in enumerate(cable_losses):
            line = transmission_line_two_port(
                50.0,
                np.pi / 2.0,
                attenuation_db=loss,
            )
            solution = ActiveReceiverNetwork(
                frequency_hz=target_frequency,
                coil_impedance_ohm=target_source,
                lna_input_models=model,
                front_end_two_ports=line,
                noise_bandwidth_hz=10.0e3,
            ).solve(unit_maps)
            values[index] = float(np.mean(solution.noise_figure_db))
        cable_noise_figures[label] = values

    normalized_frequency = frequencies / target_frequency
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for label, result in architectures.items():
        color = colors[label]
        axes[0, 0].semilogy(
            normalized_frequency,
            np.abs(result.source_load_impedance_ohm[:, 0, 0]),
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            normalized_frequency,
            result.isolation_db,
            label=label,
            color=color,
        )
        axes[0, 2].plot(
            normalized_frequency,
            np.abs(result.input_transfer_matrix[:, 0, 0]),
            label=label,
            color=color,
        )
        axes[0, 3].plot(
            normalized_frequency,
            np.mean(result.noise_figure_db, axis=1),
            label=label,
            color=color,
        )
    for axis in axes[0]:
        axis.axvline(1.0, color="0.5", linestyle=":")
        axis.set_xlabel(r"$f/f_0$")
        axis.grid(alpha=0.25)
    axes[0, 0].set_title("Load transformed to coil port")
    axes[0, 0].set_ylabel(r"$|Z_{load,coil}|$ (ohm)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("Induced-current isolation")
    axes[0, 1].set_ylabel("dB")
    axes[0, 2].set_title("Open-circuit voltage transfer")
    axes[0, 2].set_ylabel(r"$|V_{LNA}/V_{oc}|$")
    axes[0, 3].set_title("Mean receiver noise figure")
    axes[0, 3].set_ylabel("dB")

    extent = (
        phase_errors_deg[0],
        phase_errors_deg[-1],
        input_resistances[0],
        input_resistances[-1],
    )
    isolation_artist = axes[1, 0].imshow(
        isolation_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Low-Z isolation robustness")
    axes[1, 0].set_xlabel("Cable phase error (degree)")
    axes[1, 0].set_ylabel(r"$R_{in}$ (ohm)")
    figure.colorbar(isolation_artist, ax=axes[1, 0], label="dB")
    noise_artist = axes[1, 1].imshow(
        noise_figure_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Low-Z noise-figure robustness")
    axes[1, 1].set_xlabel("Cable phase error (degree)")
    axes[1, 1].set_ylabel(r"$R_{in}$ (ohm)")
    figure.colorbar(noise_artist, ax=axes[1, 1], label="dB")
    for label, values in cable_noise_figures.items():
        axes[1, 2].plot(
            cable_losses,
            values,
            label=label,
            color=colors[label],
        )
    axes[1, 2].set_title("Pre-LNA cable-loss penalty")
    axes[1, 2].set_xlabel("Matched cable loss (dB)")
    axes[1, 2].set_ylabel("Mean noise figure (dB)")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend(fontsize=8)

    component_names = ("source", "front end", "LNA")
    x_positions = np.arange(len(component_names))
    width = 0.8 / len(architectures)
    for index, (label, result) in enumerate(architectures.items()):
        components = (
            result.source_noise_covariance_v2[target_index],
            result.front_end_noise_covariance_v2[target_index],
            result.lna_noise_covariance_v2[target_index],
        )
        rms = [
            np.sqrt(np.mean(np.real(np.diag(value))) / 10.0e3)
            for value in components
        ]
        axes[1, 3].bar(
            x_positions + (index - 1.5) * width,
            rms,
            width=width,
            label=label,
            color=colors[label],
        )
    axes[1, 3].set_yscale("log")
    axes[1, 3].set_xticks(x_positions, component_names)
    axes[1, 3].set_title("Target output-noise components")
    axes[1, 3].set_ylabel(r"V/$\sqrt{Hz}$")
    axes[1, 3].legend(fontsize=6)
    figure.suptitle(
        "Phase 4.5 preamplifier decoupling: transformation, loss, and tolerance"
    )

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)

    matched_isolation = architectures["Remote matched 50 ohm"].isolation_db[
        target_index
    ]
    low_z_isolation = architectures["Low-Z + quarter-wave"].isolation_db[
        target_index
    ]
    combined_isolation = architectures["Passive + low-Z"].isolation_db[target_index]
    low_z_noise_figure = float(
        np.mean(
            architectures["Low-Z + quarter-wave"].noise_figure_db[target_index]
        )
    )
    print(
        "low-Z preamplifier isolation improvement: "
        f"{low_z_isolation - matched_isolation:.3f} dB"
    )
    print(
        "combined passive/preamplifier isolation improvement: "
        f"{combined_isolation - matched_isolation:.3f} dB"
    )
    print(f"low-Z nominal mean noise figure: {low_z_noise_figure:.3f} dB")
    print(
        "quarter-wave transformed load magnitude: "
        f"{abs(architectures['Low-Z + quarter-wave'].source_load_impedance_ohm[target_index, 0, 0]):.3f} ohm"
    )


if __name__ == "__main__":
    main()
