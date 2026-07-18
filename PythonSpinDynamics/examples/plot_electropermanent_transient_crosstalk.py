"""Plot transient programming-coil cross-talk in a two-element EPM array."""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import (  # noqa: E402
    EvidenceRecord,
    MutualProgrammingCircuit,
    ProgrammingPulse,
    TransientCoupledEPMProgrammer,
    archived_igbt_pulse_cases,
    illustrative_alnico_return_point_model,
    neighbor_coupling_matrix,
    variable_field_nmr_rod,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict induced neighbor current, voltage, internal field, and "
            "return-point-state disturbance during one EPM programming pulse."
        )
    )
    parser.add_argument(
        "--mutual-coefficient",
        type=float,
        default=0.08,
        help="Signed winding coupling coefficient M/sqrt(L1 L2) (default: 0.08).",
    )
    parser.add_argument(
        "--leakage-fraction",
        type=float,
        default=0.20,
        help="Cross-winding H/I relative to local H/I (default: 0.20).",
    )
    parser.add_argument(
        "--pulse-voltage-v",
        type=float,
        default=100.0,
        help="Target capacitor voltage (default: 100 V).",
    )
    parser.add_argument(
        "--pulse-duration-us",
        type=float,
        default=30.0,
        help="Negative programming gate duration (default: 30 us).",
    )
    parser.add_argument(
        "--sweep-samples",
        type=int,
        default=13,
        help="Leakage-fraction sweep samples (default: 13).",
    )
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if not np.isfinite(args.mutual_coefficient) or abs(args.mutual_coefficient) >= 0.8:
        raise ValueError("--mutual-coefficient must have magnitude below 0.8")
    if not np.isfinite(args.leakage_fraction) or not 0.0 <= args.leakage_fraction <= 0.5:
        raise ValueError("--leakage-fraction must lie in [0, 0.5]")
    if not np.isfinite(args.pulse_voltage_v) or not 0.0 < args.pulse_voltage_v <= 650.0:
        raise ValueError("--pulse-voltage-v must lie in (0, 650]")
    if not np.isfinite(args.pulse_duration_us) or args.pulse_duration_us <= 0.0:
        raise ValueError("--pulse-duration-us must be positive")
    if args.sweep_samples < 5:
        raise ValueError("--sweep-samples must be at least five")


def _make_programmer(
    mutual_coefficient: float,
    leakage_fraction: float,
    sources,
    driver,
    model,
    remanence_coupling,
) -> TransientCoupledEPMProgrammer:
    inferred = EvidenceRecord(
        source="plot_electropermanent_transient_crosstalk.py input",
        classification="inferred",
        detail=(
            "Illustrative mutual-inductance and winding-field leakage; replace "
            "with measured array matrices for quantitative predictions"
        ),
    )
    circuit = MutualProgrammingCircuit.from_coupling_coefficients(
        (driver, driver),
        np.asarray(
            [
                [0.0, mutual_coefficient],
                [mutual_coefficient, 0.0],
            ]
        ),
        field_coupling_fractions=np.asarray(
            [
                [1.0, leakage_fraction],
                [leakage_fraction, 1.0],
            ]
        ),
        label="illustrative two-EPM transient coupling",
        evidence=(inferred,),
    )
    return TransientCoupledEPMProgrammer(
        sources=sources,
        hysteresis_models=(model, model),
        remanence_coupling_a_per_m_per_t=remanence_coupling,
        circuit=circuit,
    )


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    sources = (
        variable_field_nmr_rod(
            center_m=(-0.04, 0.0, 0.0),
            effective_remanence_t=0.33,
        ),
        variable_field_nmr_rod(
            center_m=(0.04, 0.0, 0.0),
            effective_remanence_t=0.33,
        ),
    )
    model = illustrative_alnico_return_point_model()
    driver = replace(
        archived_igbt_pulse_cases()[0].driver,
        state_inductance_fraction_at_saturation=0.25,
    )
    remanence_coupling = neighbor_coupling_matrix(
        sources,
        n_cross=5,
        n_length=31,
    )
    pulse = ProgrammingPulse(
        args.pulse_voltage_v,
        args.pulse_duration_us * 1e-6,
        polarity=-1,
        label="target negative programming command",
    )
    times = np.linspace(0.0, max(180e-6, 6.0 * pulse.duration_s), 901)
    programmer = _make_programmer(
        args.mutual_coefficient,
        args.leakage_fraction,
        sources,
        driver,
        model,
        remanence_coupling,
    )
    result = programmer.program(
        0,
        times,
        pulse,
        tolerance_t=2e-6,
        relaxation=0.8,
        max_step_s=pulse.duration_s / 75.0,
    )

    leakage_sweep = np.linspace(0.0, 0.35, args.sweep_samples)
    target_final = np.empty_like(leakage_sweep)
    neighbor_final = np.empty_like(leakage_sweep)
    neighbor_peak_current = np.empty_like(leakage_sweep)
    for sample, leakage in enumerate(leakage_sweep):
        sweep_programmer = _make_programmer(
            args.mutual_coefficient,
            float(leakage),
            sources,
            driver,
            model,
            remanence_coupling,
        )
        sweep = sweep_programmer.program(
            0,
            times[::2],
            pulse,
            tolerance_t=3e-6,
            relaxation=0.8,
            max_step_s=pulse.duration_s / 60.0,
        )
        target_final[sample] = sweep.final_states[0].remanence.remanence_t
        neighbor_final[sample] = sweep.final_states[1].remanence.remanence_t
        neighbor_peak_current[sample] = sweep.waveform.peak_current_a[1]

    time_us = times * 1e6
    waveform = result.waveform
    colors = ("tab:blue", "tab:orange")
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Transient cross-talk during two-element EPM programming", fontsize=16)

    geometry = axes[0, 0]
    for index, (x_mm, color, label) in enumerate(
        ((-40.0, colors[0], "commanded target"), (40.0, colors[1], "inactive neighbor"))
    ):
        geometry.add_patch(Circle((x_mm, 0.0), 12.7, color=color, alpha=0.75, label=label))
        for radius in (16.5, 19.5):
            geometry.add_patch(Circle((x_mm, 0.0), radius, fill=False, color=color, linewidth=1.2))
        geometry.text(x_mm, -27.0, f"channel {index}", ha="center")
    geometry.annotate(
        f"M: k={args.mutual_coefficient:.2f}",
        xy=(22.0, 16.0),
        xytext=(-22.0, 16.0),
        arrowprops={"arrowstyle": "<->", "color": "0.25"},
        ha="center",
    )
    geometry.annotate(
        "negative pulse",
        xy=(-40.0, 0.0),
        xytext=(-40.0, 31.0),
        arrowprops={"arrowstyle": "->", "color": colors[0]},
        ha="center",
    )
    geometry.set(xlim=(-68, 68), ylim=(-35, 38), xlabel="x (mm)", ylabel="schematic y (mm)")
    geometry.set_aspect("equal")
    geometry.set_title("two rods and closed recovery windings")
    geometry.legend(loc="lower center", fontsize=8)
    geometry.grid(alpha=0.2)

    current_ax = axes[0, 1]
    current_ax.plot(time_us, waveform.currents_a[:, 0], color=colors[0], label="target")
    current_ax.plot(time_us, waveform.currents_a[:, 1], color=colors[1], label="neighbor")
    current_ax.axvline(pulse.duration_s * 1e6, color="0.3", linestyle=":", label="turn-off")
    current_ax.set(xlabel="time (us)", ylabel="winding current (A)", title="mutually coupled currents")
    current_ax.legend(fontsize=8)
    current_ax.grid(alpha=0.25)

    voltage_ax = axes[0, 2]
    voltage_ax.plot(time_us, waveform.applied_voltage_v[:, 0], color=colors[0], label="target applied")
    voltage_ax.plot(
        time_us,
        waveform.mutual_induced_voltage_v[:, 1],
        color=colors[1],
        label="neighbor induced",
    )
    voltage_ax.set(xlabel="time (us)", ylabel="voltage (V)", title="drive and induced neighbor voltage")
    voltage_ax.legend(fontsize=8)
    voltage_ax.grid(alpha=0.25)

    field_ax = axes[1, 0]
    field_ax.plot(time_us, waveform.internal_h_a_per_m[:, 0] / 1e3, color=colors[0], label="target")
    field_ax.plot(time_us, waveform.internal_h_a_per_m[:, 1] / 1e3, color=colors[1], label="neighbor")
    field_ax.axhline(-model.thresholds_a_per_m[0] / 1e3, color="0.3", linestyle="--", label="first -play threshold")
    field_ax.set(xlabel="time (us)", ylabel="internal H (kA/m)", title="winding plus retained-state interaction")
    field_ax.legend(fontsize=8)
    field_ax.grid(alpha=0.25)

    state_ax = axes[1, 1]
    state_ax.plot(time_us, result.remanence_t[:, 0], color=colors[0], label="target")
    state_ax.plot(time_us, result.remanence_t[:, 1], color=colors[1], label="neighbor")
    state_ax.set(xlabel="time (us)", ylabel="retained Br (T)", title="pulse-driven return-point trajectories")
    state_ax.legend(fontsize=8)
    state_ax.grid(alpha=0.25)

    sweep_ax = axes[1, 2]
    sweep_ax.plot(leakage_sweep, target_final, "o-", color=colors[0], label="target final Br")
    sweep_ax.plot(leakage_sweep, neighbor_final, "o-", color=colors[1], label="neighbor final Br")
    sweep_ax.axvline(args.leakage_fraction, color="0.3", linestyle=":")
    sweep_ax.set(
        xlabel="cross-winding field fraction",
        ylabel="final retained Br (T)",
        title="disturbance threshold and retained-state error",
    )
    current_twin = sweep_ax.twinx()
    current_twin.plot(leakage_sweep, neighbor_peak_current, "s--", color="tab:green", label="neighbor peak current")
    current_twin.set_ylabel("neighbor peak current (A)", color="tab:green")
    lines, labels = sweep_ax.get_legend_handles_labels()
    extra_lines, extra_labels = current_twin.get_legend_handles_labels()
    sweep_ax.legend(lines + extra_lines, labels + extra_labels, fontsize=7, loc="best")
    sweep_ax.grid(alpha=0.25)

    initial_energy = waveform.initial_electrical_energy_j
    relative_energy_error = abs(waveform.electrical_energy_balance_error_j) / initial_energy
    print("Transient two-element EPM programming")
    print(f"  mutual coefficient k={args.mutual_coefficient:+.3f} (inferred)")
    print(f"  cross-winding H/I fraction={args.leakage_fraction:.3f} (inferred)")
    print(
        "  peak currents: "
        f"target={waveform.peak_current_a[0]:.1f} A, "
        f"neighbor={waveform.peak_current_a[1]:.1f} A"
    )
    print(f"  neighbor peak induced voltage={waveform.peak_mutual_induced_voltage_v[1]:.1f} V")
    print(
        "  peak internal H: "
        f"target={waveform.peak_internal_h_a_per_m[0] / 1e3:.1f} kA/m, "
        f"neighbor={waveform.peak_internal_h_a_per_m[1] / 1e3:.1f} kA/m"
    )
    print(
        "  retained-state changes: "
        f"target={result.remanence_change_t[0]:+.4f} T, "
        f"neighbor={result.remanence_change_t[1]:+.4f} T"
    )
    print(f"  disturbed neighbor indices={result.disturbed_indices}")
    print(f"  fixed-point iterations={result.iterations}, residual={result.residual_t:.2e} T")
    print(f"  electrical energy residual={100 * relative_energy_error:.3f}%")
    print("  calibration status: mutual and leakage matrices require measurement")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
