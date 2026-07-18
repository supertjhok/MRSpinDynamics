"""Compare electrothermal control strategies for an electromagnet B0 source.

The example builds an air-core solenoid, derives its central ``B/I`` from the
Biot-Savart geometry, and predicts coupled current, copper temperature,
resistance, supply voltage, Joule power, and B0.  Constant-voltage operation is
compared with the three stabilization paths in Section 11.2 of the measurements
textbook: thermal compensation, current feedback, and direct field lock.

Run with ``--output figure.png`` to save the plot; otherwise display it.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-field-mt", type=float, default=30.0)
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--ramp-s", type=float, default=5.0)
    parser.add_argument("--voltage-limit-v", type=float, default=18.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parse_args()
    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields import coils
    from spin_dynamics.thermal import (
        ElectromagnetControl,
        ElectrothermalElectromagnet,
    )

    segments = coils.solenoid(
        radius=0.12,
        length=0.24,
        turns=144,
        axis="z",
        n_segments=32,
    )
    magnet = ElectrothermalElectromagnet.from_segments(
        segments,
        inductance_h=25.0e-3,
        reference_resistance_ohm=0.20,
        heat_capacity_j_per_k=15.0e3,
        thermal_conductance_w_per_k=8.0,
    )
    target_field = args.target_field_mt * 1.0e-3
    target_current = target_field / magnet.field_sensitivity_t_per_a
    reference_voltage = target_current * magnet.reference_resistance_ohm
    duration = args.duration_min * 60.0
    early = np.linspace(0.0, min(20.0, duration), 161)
    slow = np.linspace(0.0, duration, 451)
    times = np.unique(np.r_[early, slow])
    ramp = np.clip(times / args.ramp_s, 0.0, 1.0)
    limits = (-args.voltage_limit_v, args.voltage_limit_v)

    cases = {
        "constant voltage": magnet.simulate(
            times,
            reference_voltage * ramp,
            control=ElectromagnetControl("voltage", voltage_limits_v=limits),
        ),
        "temperature feedback": magnet.simulate(
            times,
            target_current * ramp,
            control=ElectromagnetControl(
                "temperature_compensated",
                voltage_limits_v=limits,
            ),
        ),
        "current feedback": magnet.simulate(
            times,
            target_current * ramp,
            control=ElectromagnetControl(
                "current",
                response_time_s=0.5,
                voltage_limits_v=limits,
            ),
        ),
        "field lock": magnet.simulate(
            times,
            target_field * ramp,
            control=ElectromagnetControl(
                "field",
                response_time_s=0.5,
                voltage_limits_v=limits,
            ),
        ),
    }

    print("Electrothermal electromagnet B0 source")
    print(f"  B/I at center: {1e3 * magnet.field_sensitivity_t_per_a:.4f} mT/A")
    print(f"  target: {1e3 * target_field:.2f} mT at {target_current:.2f} A")
    print(
        f"  nominal poles: electrical={magnet.electrical_time_constant_s:.3f} s, "
        f"thermal={magnet.thermal_time_constant_s / 60.0:.1f} min"
    )
    for name, result in cases.items():
        drift = 100.0 * (result.field_t[-1] / target_field - 1.0)
        print(
            f"  {name}: B0={1e3 * result.field_t[-1]:.3f} mT, "
            f"drift={drift:+.2f}%, T={result.temperature_k[-1] - 273.15:.1f} C, "
            f"R={result.resistance_ohm[-1]:.3f} ohm, "
            f"P={result.power_w[-1]:.1f} W"
        )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True)
    time_min = times / 60.0
    styles = ("-", "--", "-.", ":")
    for (name, result), style in zip(cases.items(), styles):
        axes[0, 0].plot(
            time_min,
            1.0e3 * result.field_t,
            style,
            lw=1.8,
            label=name,
        )
        axes[0, 1].plot(
            time_min,
            result.temperature_k - 273.15,
            style,
            lw=1.8,
        )
        axes[1, 0].plot(time_min, result.current_a, style, lw=1.8)
        axes[1, 1].plot(time_min, result.voltage_v, style, lw=1.8)

    axes[0, 0].axhline(1.0e3 * target_field, color="black", lw=0.9, alpha=0.6)
    axes[0, 0].set_title("B0 field")
    axes[0, 0].set_ylabel("field (mT)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("Coil temperature")
    axes[0, 1].set_ylabel("temperature (C)")
    axes[1, 0].set_title("Coil current")
    axes[1, 0].set_ylabel("current (A)")
    axes[1, 1].set_title("Supply voltage")
    axes[1, 1].set_ylabel("voltage (V)")
    for ax in axes[1, :]:
        ax.set_xlabel("time (min)")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.22)
    fig.suptitle(
        "Electrothermal B0 electromagnet: voltage drift and feedback stabilization\n"
        f"{1e3 * target_field:.0f} mT air-core solenoid, "
        f"thermal pole {magnet.thermal_time_constant_s / 60.0:.1f} min",
        fontsize=13,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170, bbox_inches="tight")
        print(f"  saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
