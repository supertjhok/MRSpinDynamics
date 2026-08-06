"""Study resonant cancellation of mutual inductance between two Rx coils."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import extract_multiport_impedance, helical_solenoid
from spin_dynamics.workflows import (
    analyze_receiver_coupling_sweep,
    mutual_cancellation_capacitance,
    shared_capacitor_mesh_impedance,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency-mhz", type=float, default=2.0)
    parser.add_argument("--span-percent", type=float, default=20.0)
    parser.add_argument("--points", type=int, default=121)
    parser.add_argument("--tolerance-percent", type=float, default=5.0)
    parser.add_argument("--load-ohm", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None
    if args.frequency_mhz <= 0.0:
        raise ValueError("--frequency-mhz must be positive")
    if not 0.0 < args.span_percent < 100.0:
        raise ValueError("--span-percent must lie between 0 and 100")
    if args.points < 3 or args.points % 2 == 0:
        raise ValueError("--points must be an odd integer of at least 3")
    if not 0.0 <= args.tolerance_percent < 100.0:
        raise ValueError("--tolerance-percent must lie in [0, 100)")
    if args.load_ohm <= 0.0:
        raise ValueError("--load-ohm must be positive")

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

    # PEEC supplies conductor loss. This reciprocal term illustrates shared
    # sample loss, whose mutual resistance survives reactive cancellation.
    sample_loss_ohm = np.array([[6.0, 0.05], [0.05, 6.0]])
    coil_impedance = peec.impedance + sample_loss_ohm[np.newaxis, :, :]
    inductance0 = peec.inductance[target_index]
    mutual_inductance = float(inductance0[0, 1])
    mutual_magnitude = abs(mutual_inductance)
    branch_capacitance = mutual_cancellation_capacitance(
        mutual_inductance,
        target_frequency,
    )
    branch_signs = (1, 1) if mutual_inductance > 0.0 else (1, -1)

    baseline_capacitances = 1.0 / (
        omega0**2 * np.diag(inductance0)
    )
    cancelled_inductances = np.diag(inductance0) - mutual_magnitude
    if np.any(cancelled_inductances <= 0.0):
        raise ValueError("mutual inductance is too large for this mesh tuning")
    cancelled_capacitances = 1.0 / (
        omega0**2 * cancelled_inductances
    )
    source_before = coil_impedance + _series_capacitor_sweep(
        frequencies,
        baseline_capacitances,
    )
    nominal_branch = shared_capacitor_mesh_impedance(
        frequencies,
        branch_capacitance,
        branch_signs=branch_signs,
        series_resistance_ohm=0.02,
    )
    source_after = (
        coil_impedance
        + _series_capacitor_sweep(frequencies, cancelled_capacitances)
        + nominal_branch
    )
    nominal = analyze_receiver_coupling_sweep(
        frequencies,
        source_before,
        source_after,
        load_impedance_ohm=args.load_ohm,
        noise_bandwidth_hz=10.0e3,
    )

    tolerance = args.tolerance_percent / 100.0
    tolerance_results = []
    for fraction in (-tolerance, 0.0, tolerance):
        branch = shared_capacitor_mesh_impedance(
            frequencies,
            branch_capacitance * (1.0 + fraction),
            branch_signs=branch_signs,
            series_resistance_ohm=0.02,
        )
        source = (
            coil_impedance
            + _series_capacitor_sweep(frequencies, cancelled_capacitances)
            + branch
        )
        tolerance_results.append(
            (
                fraction,
                analyze_receiver_coupling_sweep(
                    frequencies,
                    source_before,
                    source,
                    load_impedance_ohm=args.load_ohm,
                    noise_bandwidth_hz=10.0e3,
                ),
            )
        )

    normalized_frequency = frequencies / target_frequency
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].semilogy(
        normalized_frequency,
        np.abs(nominal.mutual_impedance_before_ohm),
        label="before",
    )
    axes[0, 0].semilogy(
        normalized_frequency,
        np.abs(nominal.mutual_impedance_after_ohm),
        label="after",
    )
    axes[0, 0].set_title(r"Residual mutual impedance $|Z_{21}|$")
    axes[0, 0].set_ylabel("ohm")
    axes[0, 0].legend()

    axes[0, 1].plot(
        normalized_frequency,
        20.0 * np.log10(nominal.coupling_ratio_before),
        label="before",
    )
    axes[0, 1].plot(
        normalized_frequency,
        20.0 * np.log10(nominal.coupling_ratio_after),
        label="after",
    )
    axes[0, 1].set_title("Induced-current coupling")
    axes[0, 1].set_ylabel("dB")
    axes[0, 1].legend()

    for fraction, result in tolerance_results:
        axes[0, 2].plot(
            normalized_frequency,
            result.isolation_improvement_db,
            label=f"{100.0 * fraction:+.1f}% C",
        )
    axes[0, 2].set_title("Isolation improvement and tolerance")
    axes[0, 2].set_ylabel("dB")
    axes[0, 2].legend()

    axes[1, 0].semilogy(
        normalized_frequency,
        np.abs(nominal.transfer_matrix_before[:, 0, 1]),
        label="before",
    )
    axes[1, 0].semilogy(
        normalized_frequency,
        np.abs(nominal.transfer_matrix_after[:, 0, 1]),
        label="after",
    )
    axes[1, 0].set_title("Loaded sensitivity mixing")
    axes[1, 0].set_ylabel(r"$|H_{01}|$")
    axes[1, 0].legend()

    axes[1, 1].plot(
        normalized_frequency,
        np.abs(nominal.noise_correlation_before[:, 0, 1]),
        label="before",
    )
    axes[1, 1].plot(
        normalized_frequency,
        np.abs(nominal.noise_correlation_after[:, 0, 1]),
        label="after",
    )
    axes[1, 1].set_title("Passive noise correlation")
    axes[1, 1].set_ylabel(r"$|\rho_{01}|$")
    axes[1, 1].legend()

    axes[1, 2].plot(
        normalized_frequency,
        np.imag(nominal.mutual_impedance_before_ohm),
        label="before",
    )
    axes[1, 2].plot(
        normalized_frequency,
        np.imag(nominal.mutual_impedance_after_ohm),
        label="after",
    )
    axes[1, 2].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 2].set_title(r"Mutual reactance $\operatorname{Im} Z_{21}$")
    axes[1, 2].set_ylabel("ohm")
    axes[1, 2].legend()

    for axis in axes.flat:
        axis.axvline(1.0, color="0.5", linestyle=":", linewidth=1.0)
        axis.set_xlabel(r"$f/f_0$")
        axis.grid(alpha=0.2)
    figure.suptitle(
        "Phase 4.5 resonant receiver decoupling: "
        f"f0={args.frequency_mhz:g} MHz, "
        f"Cd={branch_capacitance * 1.0e9:.3f} nF"
    )
    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)

    residual = nominal.mutual_impedance_after_ohm[target_index]
    print(f"mutual inductance: {mutual_inductance * 1.0e6:.6f} uH")
    print(
        "cancellation capacitance: "
        f"{branch_capacitance * 1.0e9:.6f} nF"
    )
    print(
        "target isolation improvement: "
        f"{nominal.isolation_improvement_db[target_index]:.3f} dB"
    )
    print(
        "residual mutual impedance: "
        f"{residual.real:.6f} {residual.imag:+.6f}j ohm"
    )
    print(
        "residual noise correlation: "
        f"{abs(nominal.noise_correlation_after[target_index, 0, 1]):.6f}"
    )


if __name__ == "__main__":
    main()
