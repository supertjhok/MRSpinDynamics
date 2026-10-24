"""Thermal recovery and microwave pumping of an explicit electron–nuclear pair.

Run from PythonSpinDynamics with opt/plot extras:
python examples/plot_overhauser_coupled_dynamics.py --output .tmp/dnp_coupled.png

Illustrative 1:1 spin pair at 0.35 T, 298.15 K; this is not a dilute TEMPOL/water
ensemble. Transition rates are stated explicitly in s^-1: w0=0.05, w2=0.3,
w1n=0.1, w1e=0.02. Intrinsic electron/nuclear T1 are 100 us/2.5 s and pure
electron dephasing is 20000 s^-1. Saturation emerges from coherent driving.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.dnp import (
    CoupledElectronNuclearModel,
    ElectronNuclearTransitionRates,
    MicrowaveDrive,
    cw_overhauser,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--csv", type=Path, help="Pump/recovery data for the strongest drive"
    )
    parser.add_argument("--points", type=int, default=160)
    args = parser.parse_args()
    if args.points < 3:
        parser.error("points must be at least 3")
    model = CoupledElectronNuclearModel(
        0.35,
        transitions=ElectronNuclearTransitionRates(0.05, 0.3, 0.1, 0.02),
    )
    times = np.linspace(0, 8, args.points)
    early_times = np.linspace(0, 1e-3, args.points)
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    fig.suptitle(
        "Coupled electron–nuclear Overhauser dynamics: thermal Lindblad reference\n"
        "One electron + one proton | 0.35 T | 298.15 K | T₁e = 100 μs | T₁n,background = 2.5 s\n"
        "w₀ = 0.05, w₂ = 0.3, w₁n = 0.1, w₁e = 0.02 s⁻¹ | γ φ,e = 20000 s⁻¹",
        fontsize=11,
    )
    for nutation in [0, 1e4, 5e4]:
        drive = MicrowaveDrive(nutation)
        steady = model.steady_state(drive)
        early = model.simulate(early_times, drive=drive)
        pump = model.simulate(times, drive=drive)
        (line,) = axes[0, 0].plot(
            early_times * 1e3,
            1 - early.electron_polarization / early.thermal_electron_polarization,
            label=f"ω₁ = {nutation / 1e3:g} krad/s",
        )
        color = line.get_color()
        axes[0, 1].plot(
            times,
            pump.nuclear_polarization * 1e6,
            color=color,
            label=f"ω₁ = {nutation / 1e3:g} krad/s",
        )
        print(
            f"omega1={nutation:g} rad/s; saturation={steady.electron_saturation:.6f}; "
            f"E={steady.nuclear_enhancement:.6f}; Pn={steady.nuclear_polarization * 1e6:.3f} ppm"
        )
    recovery = model.simulate(
        times, drive=drive.off(), initial_density=pump.density_matrices[-1]
    )
    axes[1, 0].plot(times, recovery.nuclear_polarization * 1e6, label="Microwaves off")
    axes[1, 0].axhline(
        recovery.thermal_nuclear_polarization * 1e6,
        color="0.4",
        ls=":",
        label="Gibbs thermal target",
    )
    detunings = np.linspace(-3e5, 3e5, 151)
    enhancements, cw_enhancements = [], []
    cw_rates = model.transitions.nuclear_cw_rates(
        background_r1_per_second=1 / model.nuclear_t1_seconds
    )
    for delta in detunings:
        steady = model.steady_state(MicrowaveDrive(5e4, float(delta)))
        enhancements.append(steady.nuclear_enhancement)
        cw = cw_overhauser(
            cw_rates,
            field_tesla=model.field_tesla,
            saturation_factor=steady.electron_saturation,
        )
        cw_enhancements.append(float(cw.enhancement))
    axes[1, 1].plot(detunings / 1e3, enhancements, label="Coupled density matrix")
    axes[1, 1].plot(
        detunings / 1e3, cw_enhancements, "--", label="High-T CW with actual electron s"
    )
    axes[0, 0].set(
        title="Electron saturation develops dynamically",
        xlabel="Pump time (ms)",
        ylabel="1 − P e / P e,thermal",
    )
    axes[0, 1].set(
        title="Nuclear polarization buildup",
        xlabel="Pump time (s)",
        ylabel="Signed proton polarization (ppm)",
    )
    axes[1, 0].set(
        title="Thermal recovery after pumping",
        xlabel="Recovery time (s)",
        ylabel="Signed proton polarization (ppm)",
    )
    axes[1, 1].set(
        title="Microwave detuning profile, ω₁ = 50 krad/s",
        xlabel="Detuning (krad/s)",
        ylabel="Nuclear enhancement E",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            np.column_stack(
                [
                    times,
                    pump.electron_polarization,
                    pump.nuclear_polarization,
                    recovery.electron_polarization,
                    recovery.nuclear_polarization,
                ]
            ),
            delimiter=",",
            comments="",
            header="time_seconds,pump_electron_polarization,"
            "pump_nuclear_polarization,recovery_electron_polarization,recovery_nuclear_polarization",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
