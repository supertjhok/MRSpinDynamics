"""Plot EPM programming-pulse, calibration, and thermal diagnostics.

The example keeps three evidence layers separate: archived peak-current values,
configuration-specific inferred circuit parameters, and the coarse published
demagnetization envelope.  It does not claim a universal AlNiCo hysteresis law.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import (  # noqa: E402
    ALNICO5_AC500,
    ProgrammingPulse,
    PulseThermalState,
    RemanenceState,
    archived_igbt_pulse_cases,
    finite_cylinder_on_axis_field,
    published_demagnetization_calibration,
    variable_field_nmr_rod,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot capacitor/H-bridge/RLC EPM programming pulses, archived "
            "peak-current comparisons, empirical remanence, and pulse heating."
        )
    )
    parser.add_argument(
        "--command-fraction",
        type=float,
        default=0.17,
        help="Saturate-then-demagnetize command fraction in [0, 1] (default: 0.17).",
    )
    parser.add_argument(
        "--pulse-count",
        type=int,
        default=60,
        help="Number of pulses in the thermal train (default: 60).",
    )
    parser.add_argument(
        "--cooling-s",
        type=float,
        default=0.02,
        help="Cooling interval between thermal-train pulses (default: 0.02 s).",
    )
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _check_arguments(args: argparse.Namespace) -> None:
    if not 0.0 <= args.command_fraction <= 1.0:
        raise ValueError("--command-fraction must lie in [0, 1]")
    if args.pulse_count < 1:
        raise ValueError("--pulse-count must be at least one")
    if not np.isfinite(args.cooling_s) or args.cooling_s < 0.0:
        raise ValueError("--cooling-s must be finite and non-negative")


def main() -> None:
    args = _parser().parse_args()
    _check_arguments(args)

    cases = archived_igbt_pulse_cases()
    waveforms = tuple(case.run() for case in cases)
    measured_peaks = np.asarray([case.measured_peak_current_a for case in cases])
    modeled_peaks = np.asarray([waveform.peak_current_a for waveform in waveforms])

    calibration = published_demagnetization_calibration()
    initial_state = RemanenceState(0.33, branch="positive_saturation")
    duty = np.linspace(0.0, 1.0, 101)
    retained = np.empty_like(duty)
    uncertainty = np.empty_like(duty)
    rod = variable_field_nmr_rod()
    surface_position = 0.5 * rod.length_m + 1.0e-3
    surface_field = np.empty_like(duty)
    surface_uncertainty = np.empty_like(duty)
    negative_driver = cases[0].driver
    base_times = waveforms[0].times_s
    for index, command in enumerate(duty):
        pulse = ProgrammingPulse(
            220.0,
            50.0e-6,
            polarity=-1,
            command_fraction=float(command),
        )
        waveform = negative_driver.simulate(base_times, pulse)
        result = calibration.apply(initial_state, waveform)
        retained[index] = result.final_state.remanence_t
        uncertainty[index] = result.final_state.uncertainty_t
        programmed = rod.with_state(result.final_state)
        surface_field[index] = finite_cylinder_on_axis_field(
            surface_position,
            programmed,
        )
        surface_uncertainty[index] = (
            abs(surface_field[index] / retained[index]) * uncertainty[index]
            if retained[index]
            else abs(
                finite_cylinder_on_axis_field(
                    surface_position,
                    rod.with_state(RemanenceState(uncertainty[index])),
                )
            )
        )

    selected_pulse = ProgrammingPulse(
        220.0,
        50.0e-6,
        polarity=-1,
        command_fraction=args.command_fraction,
    )
    selected_waveform = negative_driver.simulate(base_times, selected_pulse)
    selected_result = calibration.apply(initial_state, selected_waveform)

    state_sensitive_driver = replace(
        cases[0].driver,
        state_inductance_fraction_at_saturation=0.25,
        label="illustrative state-dependent L",
    )
    unprogrammed = RemanenceState(0.0)
    retained_state = RemanenceState(0.33, branch="positive_saturation")
    zero_state_waveform = state_sensitive_driver.simulate(
        base_times,
        cases[0].pulse,
        state=unprogrammed,
        material=ALNICO5_AC500,
    )
    biased_state_waveform = state_sensitive_driver.simulate(
        base_times,
        cases[0].pulse,
        state=retained_state,
        material=ALNICO5_AC500,
        bias_field_a_per_m=-10.0e3,
    )

    thermal_driver = replace(
        cases[0].driver,
        coil_heat_capacity_j_per_k=20.0,
        coil_thermal_conductance_w_per_k=0.5,
        driver_heat_capacity_j_per_k=10.0,
        driver_thermal_conductance_w_per_k=0.5,
        switching_energy_per_transition_j=0.01,
    )
    thermal_state = PulseThermalState()
    pulse_numbers = np.arange(1, args.pulse_count + 1)
    coil_temperature = np.empty(args.pulse_count)
    driver_temperature = np.empty(args.pulse_count)
    cumulative_energy = np.empty(args.pulse_count)
    for index in range(args.pulse_count):
        waveform = thermal_driver.simulate(
            base_times,
            cases[0].pulse,
            initial_thermal_state=thermal_state,
            max_step_s=cases[0].pulse.duration_s / 40.0,
        )
        thermal_state = waveform.final_thermal_state
        coil_temperature[index] = thermal_state.coil_temperature_k
        driver_temperature[index] = thermal_state.driver_temperature_k
        cumulative_energy[index] = (
            thermal_state.cumulative_coil_energy_j
            + thermal_state.cumulative_driver_energy_j
        )
        thermal_state = thermal_driver.cool_thermal_state(
            thermal_state,
            args.cooling_s,
        )

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4), constrained_layout=True)
    fig.suptitle(
        "Electropermanent-magnet programming — circuit before hysteresis",
        fontsize=15,
    )

    ax = axes[0, 0]
    for case, waveform in zip(cases, waveforms):
        ax.plot(
            waveform.times_s * 1e6,
            waveform.current_a,
            label=case.pulse.label.replace("archive ", ""),
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(title="configuration-specific archived pulses", xlabel="time (µs)", ylabel="current (A)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    primary = waveforms[0]
    line_current = ax.plot(
        primary.times_s * 1e6,
        primary.current_a,
        color="tab:blue",
        label="current",
    )[0]
    ax.set(xlabel="time (µs)", ylabel="current (A)", title="220-V capacitor discharge and recovery")
    ax.grid(alpha=0.25)
    voltage_axis = ax.twinx()
    line_voltage = voltage_axis.plot(
        primary.times_s * 1e6,
        primary.capacitor_voltage_v,
        color="tab:orange",
        label="capacitor voltage",
    )[0]
    voltage_axis.set_ylabel("capacitor voltage (V)", color="tab:orange")
    ax.axvline(primary.pulse.duration_s * 1e6, color="0.4", linestyle=":")
    ax.legend([line_current, line_voltage], ["current", "capacitor voltage"], fontsize=8)

    ax = axes[0, 2]
    limit = 1.08 * max(np.max(measured_peaks), np.max(modeled_peaks))
    ax.plot([0.0, limit], [0.0, limit], color="0.5", linestyle="--", label="identity")
    ax.scatter(measured_peaks, modeled_peaks, s=55, color="tab:green", zorder=3)
    for case, measured, modeled in zip(cases, measured_peaks, modeled_peaks):
        ax.annotate(
            f"{case.pulse.capacitor_voltage_v:.0f} V",
            (measured, modeled),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set(
        xlim=(0.0, limit),
        ylim=(0.0, limit),
        xlabel="archived peak current (A)",
        ylabel="modeled peak current (A)",
        title="trace-level regression target",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(duty, retained, color="tab:purple", label="retained Br")
    ax.fill_between(
        duty,
        retained - uncertainty,
        retained + uncertainty,
        color="tab:purple",
        alpha=0.18,
        label="stated uncertainty",
    )
    ax.axhline(0.0, color="0.4", linewidth=0.8)
    ax.axvline(0.17, color="0.4", linestyle=":", label="reported zero ≈ 17%")
    ax.scatter(
        [args.command_fraction],
        [selected_result.final_state.remanence_t],
        color="black",
        zorder=4,
        label="selected command",
    )
    ax.set(
        xlabel="saturate-then-demagnetize command fraction",
        ylabel="effective retained Br (T)",
        title="coarse published envelope — not raw data",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(
        zero_state_waveform.times_s * 1e6,
        zero_state_waveform.internal_h_a_per_m / 1e3,
        label=f"Br=0, L={zero_state_waveform.inductance_h * 1e6:.1f} µH",
    )
    ax.plot(
        biased_state_waveform.times_s * 1e6,
        biased_state_waveform.internal_h_a_per_m / 1e3,
        label=(
            f"Br=0.33 T, L={biased_state_waveform.inductance_h * 1e6:.1f} µH, "
            "Hbias=-10 kA/m"
        ),
    )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="time (µs)",
        ylabel="nominal internal H (kA/m)",
        title="explicit state-L and neighbor-field hooks",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.plot(
        pulse_numbers,
        1e3 * (coil_temperature - thermal_driver.ambient_temperature_k),
        label="coil ΔT",
    )
    ax.plot(
        pulse_numbers,
        1e3 * (driver_temperature - thermal_driver.ambient_temperature_k),
        label="driver ΔT",
    )
    ax.set(
        xlabel="pulse number",
        ylabel="temperature rise (mK)",
        title=f"pulse train with {args.cooling_s * 1e3:.0f}-ms cooling gaps",
    )
    energy_axis = ax.twinx()
    energy_line = energy_axis.plot(
        pulse_numbers,
        cumulative_energy,
        color="black",
        linestyle="--",
        label="cumulative loss",
    )[0]
    energy_axis.set_ylabel("cumulative electrical loss (J)")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines + [energy_line], labels + ["cumulative loss"], fontsize=7)
    ax.grid(alpha=0.25)

    selected_index = int(np.argmin(np.abs(duty - args.command_fraction)))
    print("Electropermanent programming-pulse model")
    for case, waveform in zip(cases, waveforms):
        error = 100.0 * (
            waveform.peak_current_a - case.measured_peak_current_a
        ) / case.measured_peak_current_a
        print(
            f"  {case.pulse.label}: modeled {waveform.peak_current_a:.1f} A, "
            f"archived {case.measured_peak_current_a:.1f} A ({error:+.2f}%)"
        )
    print(
        f"  selected duty {args.command_fraction:.3f}: "
        f"Br={selected_result.final_state.remanence_t:+.3f} ± "
        f"{selected_result.final_state.uncertainty_t:.3f} T, "
        f"surface B={surface_field[selected_index] * 1e3:+.1f} ± "
        f"{surface_uncertainty[selected_index] * 1e3:.1f} mT"
    )
    print(
        f"  220-V pulse: peak H={selected_waveform.peak_internal_h_a_per_m / 1e3:.1f} kA/m, "
        f"coil loss={selected_waveform.coil_energy_j:.3f} J, "
        f"driver loss={selected_waveform.driver_energy_j:.3f} J"
    )
    print(
        f"  pulse train: total loss={cumulative_energy[-1]:.2f} J, "
        f"max coil rise={1e3 * np.max(coil_temperature - thermal_driver.ambient_temperature_k):.1f} mK, "
        f"max driver rise={1e3 * np.max(driver_temperature - thermal_driver.ambient_temperature_k):.1f} mK"
    )
    print("  calibration evidence: inferred coarse anchors; raw demag samples remain unavailable")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
