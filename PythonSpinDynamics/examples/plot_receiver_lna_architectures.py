"""Compare matched 50-ohm and on-coil high-input-impedance LNAs."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.experiment import (
    Hardware,
    ImagingPlane,
    Phantom,
    RxArray,
    RxCoil,
    SolenoidCoil,
    solve_receive_sensitivities,
)
from spin_dynamics.fields import extract_multiport_impedance, helical_solenoid
from spin_dynamics.workflows import (
    ActiveReceiverNetwork,
    LNAInputModel,
    optimal_channel_snr,
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


def _coil_geometry() -> tuple[SolenoidCoil, SolenoidCoil]:
    first = SolenoidCoil(
        radius_m=0.012,
        length_m=0.025,
        turns=4,
        center_m=(0.0, -0.007, 0.0),
        axis="x",
        n_segments=12,
    )
    return first, dataclasses.replace(first, center_m=(0.0, 0.007, 0.0))


def _peec_conductors(geometries: tuple[SolenoidCoil, ...]):
    conductors = []
    for geometry in geometries:
        conductor = helical_solenoid(
            diameter=2.0 * geometry.radius_m,
            length=geometry.length_m,
            turns=geometry.turns,
            wire_radius=0.4e-3,
            n_per_turn=geometry.n_segments,
            n_radial=1,
            n_angular=4,
            axis=geometry.axis,
        )
        conductors.append(
            dataclasses.replace(
                conductor,
                path_points=conductor.path_points + np.asarray(geometry.center_m),
            )
        )
    return tuple(conductors)


def _phantom(pixels: int) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, pixels)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    rho = (x**2 + y**2 <= 0.82**2).astype(np.float64)
    rho += 0.6 * (((x + 0.3) / 0.2) ** 2 + ((y - 0.2) / 0.15) ** 2 <= 1.0)
    return rho


def _network(
    frequency_hz: float,
    coil_impedance_ohm: np.ndarray,
    series_impedance_ohm: np.ndarray,
    model: LNAInputModel,
    bandwidth_hz: float,
) -> ActiveReceiverNetwork:
    return ActiveReceiverNetwork(
        frequency_hz=frequency_hz,
        coil_impedance_ohm=coil_impedance_ohm,
        series_impedance_ohm=series_impedance_ohm,
        lna_input_models=model,
        temperature_k=293.15,
        noise_bandwidth_hz=bandwidth_hz,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare active matched-50-ohm and high-input-impedance Rx front ends."
        )
    )
    parser.add_argument("--frequency-mhz", type=float, default=2.0)
    parser.add_argument("--span-percent", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=61)
    parser.add_argument("--pixels", type=int, default=31)
    parser.add_argument("--noise-bandwidth-khz", type=float, default=10.0)
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
    if args.pixels < 5:
        raise ValueError("--pixels must be at least 5")
    if args.noise_bandwidth_khz <= 0.0:
        raise ValueError("--noise-bandwidth-khz must be positive")

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
    bandwidth_hz = args.noise_bandwidth_khz * 1.0e3
    geometries = _coil_geometry()
    peec = extract_multiport_impedance(_peec_conductors(geometries), frequencies)
    sample_loss_ohm = np.array([[6.0, 0.4], [0.4, 6.0]])
    coil_impedance = peec.impedance + sample_loss_ohm[np.newaxis, :, :]
    omega0 = 2.0 * np.pi * target_frequency
    tuning_capacitances = 1.0 / (
        omega0**2 * np.diag(peec.inductance[target_index])
    )
    series_impedance = _series_capacitor_sweep(
        frequencies,
        tuning_capacitances,
    )

    # Illustrative device-level inputs, not specifications for a particular LNA.
    matched_lna = LNAInputModel(
        input_resistance_ohm=50.0,
        voltage_noise_density_v_per_sqrt_hz=0.45e-9,
        current_noise_density_a_per_sqrt_hz=2.0e-12,
        voltage_gain_v_per_v=100.0,
        output_noise_density_v_per_sqrt_hz=2.0e-9,
    )
    high_impedance_lna = LNAInputModel(
        input_resistance_ohm=1.0e6,
        input_capacitance_f=5.0e-12,
        voltage_noise_density_v_per_sqrt_hz=1.2e-9,
        current_noise_density_a_per_sqrt_hz=5.0e-15,
        voltage_gain_v_per_v=100.0,
        output_noise_density_v_per_sqrt_hz=2.0e-9,
    )
    models = {
        "Matched 50 ohm": matched_lna,
        "On-coil high-Z": high_impedance_lna,
    }
    colors = {"Matched 50 ohm": "tab:blue", "On-coil high-Z": "tab:orange"}
    transfer: dict[str, np.ndarray] = {}
    coupling: dict[str, np.ndarray] = {}
    noise_figure: dict[str, np.ndarray] = {}
    input_impedance: dict[str, np.ndarray] = {}
    target_networks: dict[str, ActiveReceiverNetwork] = {}
    unit_maps = np.ones((2, 1), dtype=np.complex128)
    drive = np.array([1.0, 0.0], dtype=np.complex128)
    for label, model in models.items():
        transfer_values = np.empty(args.points)
        coupling_values = np.empty(args.points)
        noise_figure_values = np.empty(args.points)
        input_values = np.empty(args.points)
        for index, frequency in enumerate(frequencies):
            network = _network(
                frequency,
                coil_impedance[index],
                series_impedance[index],
                model,
                bandwidth_hz,
            )
            solution = network.solve(unit_maps)
            currents = np.linalg.solve(network.total_impedance_ohm, drive)
            transfer_values[index] = abs(solution.input_transfer_matrix[0, 0])
            coupling_values[index] = abs(currents[1] / currents[0])
            noise_figure_values[index] = float(np.mean(solution.noise_figure_db))
            input_values[index] = abs(network.input_impedance_ohm[0, 0])
            if index == target_index:
                target_networks[label] = network
        transfer[label] = transfer_values
        coupling[label] = coupling_values
        noise_figure[label] = noise_figure_values
        input_impedance[label] = input_values

    rho = _phantom(args.pixels)
    hardware = Hardware(
        rx_coil=RxArray(tuple(RxCoil(geometry) for geometry in geometries)),
        plane=ImagingPlane(plane="xy", extent_m=(0.014, 0.014)),
    )
    geometric_maps = solve_receive_sensitivities(
        Phantom(rho),
        hardware,
    ).normalized_complex
    target_solutions = {
        label: network.solve(geometric_maps)
        for label, network in target_networks.items()
    }
    snr_maps = {
        label: np.asarray(
            optimal_channel_snr(
                solution.effective_sensitivities,
                solution.noise_covariance_v2,
            )
        )
        for label, solution in target_solutions.items()
    }
    matched_snr = snr_maps["Matched 50 ohm"]
    high_z_snr = snr_maps["On-coil high-Z"]
    snr_limit = max(float(np.max(matched_snr)), float(np.max(high_z_snr)))
    snr_ratio = np.divide(
        high_z_snr,
        matched_snr,
        out=np.full_like(high_z_snr, np.nan),
        where=matched_snr > 0.0,
    )

    normalized_frequency = frequencies / target_frequency
    figure, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for label in models:
        color = colors[label]
        axes[0, 0].semilogy(
            normalized_frequency,
            input_impedance[label],
            label=label,
            color=color,
        )
        axes[0, 1].plot(
            normalized_frequency,
            transfer[label],
            label=label,
            color=color,
        )
        axes[0, 2].plot(
            normalized_frequency,
            20.0 * np.log10(coupling[label]),
            label=label,
            color=color,
        )
        axes[0, 3].plot(
            normalized_frequency,
            noise_figure[label],
            label=label,
            color=color,
        )
    for axis in axes[0]:
        axis.axvline(1.0, color="0.5", linestyle=":")
        axis.set_xlabel(r"$f/f_0$")
        axis.grid(alpha=0.25)
    axes[0, 0].set_title(r"LNA input magnitude $|Z_{in}|$")
    axes[0, 0].set_ylabel("ohm")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Open-circuit signal transfer")
    axes[0, 1].set_ylabel(r"$|V_{in}/V_{oc}|$")
    axes[0, 2].set_title("Induced-current coupling")
    axes[0, 2].set_ylabel("dB")
    axes[0, 3].set_title("Mean active-chain noise figure")
    axes[0, 3].set_ylabel("dB")

    labels = tuple(models)
    component_names = ("source", r"$e_n$", r"$i_n$", "downstream")
    x_positions = np.arange(len(component_names))
    width = 0.34
    for offset, label in enumerate(labels):
        solution = target_solutions[label]
        component_covariances = (
            solution.source_noise_covariance_v2,
            solution.lna_voltage_noise_covariance_v2,
            solution.lna_current_noise_covariance_v2,
            solution.downstream_noise_covariance_v2,
        )
        amplitudes = [
            np.sqrt(np.mean(np.real(np.diag(value))) / bandwidth_hz)
            for value in component_covariances
        ]
        axes[1, 0].bar(
            x_positions + (offset - 0.5) * width,
            amplitudes,
            width=width,
            label=label,
            color=colors[label],
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(x_positions, component_names)
    axes[1, 0].set_title("Target output-noise components")
    axes[1, 0].set_ylabel(r"V/$\sqrt{Hz}$")
    axes[1, 0].legend(fontsize=8)

    image_panels = (
        (matched_snr, "Matched 50-ohm array SNR", 0.0, snr_limit, "viridis"),
        (high_z_snr, "On-coil high-Z array SNR", 0.0, snr_limit, "viridis"),
        (snr_ratio, "High-Z / matched SNR", 0.5, 1.5, "coolwarm"),
    )
    for axis, (values, title, vmin, vmax, cmap) in zip(axes[1, 1:], image_panels):
        artist = axis.imshow(
            values,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axis.contour(rho > 0.0, levels=[0.5], colors="white", linewidths=0.5)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(artist, ax=axis, shrink=0.8)
    figure.suptitle(
        "Phase 4.5 active front ends: loading, coupling, noise, and array SNR"
    )

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)

    mask = rho > 0.0
    target_coupling_improvement = 20.0 * np.log10(
        coupling["Matched 50 ohm"][target_index]
        / coupling["On-coil high-Z"][target_index]
    )
    print(
        "matched 50-ohm mean noise figure: "
        f"{noise_figure['Matched 50 ohm'][target_index]:.3f} dB"
    )
    print(
        "high-Z mean noise figure: "
        f"{noise_figure['On-coil high-Z'][target_index]:.3f} dB"
    )
    print(
        "high-Z induced-current isolation improvement: "
        f"{target_coupling_improvement:.3f} dB"
    )
    print(
        "median in-object high-Z/matched SNR ratio: "
        f"{float(np.nanmedian(snr_ratio[mask])):.3f}"
    )


if __name__ == "__main__":
    main()