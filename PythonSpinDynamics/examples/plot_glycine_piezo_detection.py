"""Estimate piezoelectric/acoustic NQR detection sensitivity for glycine.

The default workflow uses the local repository data:

* glycine crystal density and formula weight from
  ``QuadrupolarDFT/structures/Glycine/189379.cif``;
* ``14N`` glycine NQR frequencies, linewidths, T1, and T2 from
  ``NQRDatabase/data/exports/nqr.sqlite``.

The default strain-to-transition couplings use the first-pass glycine
finite-strain ABINIT result in ``QuadrupolarDFT/runs/glycine_strain``.  Those
numbers are exploratory because the static DFT EFG does not yet reproduce the
measured glycine asymmetry, but they replace the previous order-of-magnitude
``1 MHz/strain`` placeholder with realistic DFT-scale values.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nqr import (  # noqa: E402
    PiezoNQRCoupling,
    glycine_crystal_from_cif,
    load_glycine_nqr_lines_from_sqlite,
    simulate_piezoelectric_nqr_detection,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = REPO_ROOT / "NQRDatabase" / "data" / "exports" / "nqr.sqlite"
DEFAULT_CIF = REPO_ROOT / "QuadrupolarDFT" / "structures" / "Glycine" / "189379.cif"
DFT_DRIVE_HZ_PER_STRAIN = {
    # Mapped from the strongest DFT projected coupling for each experimental
    # single-quantum line: y/lower <- yy on DFT 0-1; x/upper <- xz on DFT 0-2.
    "y": 1.844477580370626e6,
    "x": 2.1893189700453524e6,
}
DFT_SOURCE_NOTE = (
    "QuadrupolarDFT glycine strain run: C_Q=1592.5 kHz, eta=0.1948; "
    "NQRDatabase glycine: C_Q=1193 kHz, eta=0.528."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cif", type=Path, default=DEFAULT_CIF)
    parser.add_argument("--thickness-mm", type=float, default=0.5)
    parser.add_argument("--area-mm2", type=float, default=25.0)
    parser.add_argument("--mechanical-q", type=float, default=100.0)
    parser.add_argument("--d-eff-pm-v", type=float, default=6.1)
    parser.add_argument("--epsilon-r", type=float, default=10.0)
    parser.add_argument("--sound-velocity", type=float, default=2500.0)
    parser.add_argument(
        "--drive-source",
        choices=("dft", "manual"),
        default="dft",
        help="Use DFT-derived per-line drive couplings or one manual value.",
    )
    parser.add_argument(
        "--drive-hz-per-strain",
        type=float,
        default=1.0e6,
        help="Manual drive coupling in Hz/strain when --drive-source=manual.",
    )
    parser.add_argument("--voltage", type=float, default=10.0,
                        help="Report voltage in V RMS.")
    parser.add_argument("--voltage-min", type=float, default=0.03,
                        help="Minimum sweep voltage in V RMS.")
    parser.add_argument("--voltage-max", type=float, default=100.0,
                        help="Maximum sweep voltage in V RMS.")
    parser.add_argument("--points", type=int, default=80)
    parser.add_argument("--integration", type=float, default=10.0,
                        help="Lock-in integration time in seconds.")
    parser.add_argument("--spin-enhancement", type=float, default=1.0,
                        help="Population enhancement over 300 K thermal NQR.")
    parser.add_argument("--power-noise", type=float, default=1.0e-15,
                        help="Power readout noise in W/sqrt(Hz).")
    parser.add_argument("--fractional-noise", type=float, default=1.0e-6,
                        help="Fractional acoustic loading noise in 1/sqrt(Hz).")
    parser.add_argument("--output", type=Path, help="Optional output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.points < 2:
        raise SystemExit("--points must be at least 2")

    crystal = glycine_crystal_from_cif(
        args.cif,
        thickness_m=args.thickness_mm * 1e-3,
        electrode_area_m2=args.area_mm2 * 1e-6,
        d_eff_m_per_v=args.d_eff_pm_v * 1e-12,
        relative_permittivity=args.epsilon_r,
        sound_velocity_m_s=args.sound_velocity,
        mechanical_q=args.mechanical_q,
    )
    lines = load_glycine_nqr_lines_from_sqlite(args.database)
    voltages = np.geomspace(args.voltage_min, args.voltage_max, args.points)
    sweep = {
        line.transition_label: [
            simulate_piezoelectric_nqr_detection(
                crystal,
                line,
                coupling_for_line(args, line),
                voltage_rms=float(voltage),
                spin_temperature_enhancement=args.spin_enhancement,
                power_noise_density_w_per_sqrt_hz=args.power_noise,
                fractional_noise_density_per_sqrt_hz=args.fractional_noise,
                integration_time_seconds=args.integration,
            )
            for voltage in voltages
        ]
        for line in lines
    }

    print_report(args, crystal, lines)
    report_results = [
        simulate_piezoelectric_nqr_detection(
            crystal,
            line,
            coupling_for_line(args, line),
            voltage_rms=args.voltage,
            spin_temperature_enhancement=args.spin_enhancement,
            power_noise_density_w_per_sqrt_hz=args.power_noise,
            fractional_noise_density_per_sqrt_hz=args.fractional_noise,
            integration_time_seconds=args.integration,
        )
        for line in lines
    ]
    print_result_table(report_results)

    plt = load_matplotlib(headless=args.output is not None)
    fig = plot_sweep(plt, voltages, sweep)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


def coupling_for_line(args, line) -> PiezoNQRCoupling:
    """Return the drive coupling for one experimental glycine line."""

    if args.drive_source == "manual":
        return PiezoNQRCoupling(drive_hz_per_strain=args.drive_hz_per_strain)
    drive = DFT_DRIVE_HZ_PER_STRAIN.get(line.transition_label)
    if drive is None:
        drive = args.drive_hz_per_strain
    return PiezoNQRCoupling(drive_hz_per_strain=drive)


def print_report(args, crystal, lines) -> None:
    print("Glycine piezoelectric NQR detection estimate")
    print(f"  CIF: {args.cif}")
    print(f"  NQR database: {args.database}")
    print(f"  active volume: {crystal.volume_m3 * 1e9:.3f} mm^3")
    print(f"  14N spin count: {crystal.quadrupolar_spin_count:.3e}")
    print(f"  capacitance: {crystal.capacitance_f * 1e12:.2f} pF")
    print(
        "  piezo drive: "
        f"d_eff={crystal.d_eff_m_per_v * 1e12:.2f} pm/V, "
        f"Q={crystal.mechanical_q:g}, "
        f"source={args.drive_source}"
    )
    if args.drive_source == "dft":
        print(f"  DFT drive note: {DFT_SOURCE_NOTE}")
        for line in lines:
            coupling = coupling_for_line(args, line)
            print(
                f"    {line.transition_label} {line.frequency_hz / 1e3:.1f} kHz: "
                f"drive={coupling.drive_hz_per_strain:.3e} Hz/strain"
            )
    else:
        print(f"    manual drive={args.drive_hz_per_strain:.3e} Hz/strain")
    print(f"  lines loaded: {len(lines)}")


def print_result_table(results) -> None:
    print("  at requested voltage:")
    for result in results:
        print(
            f"    {result.line.transition_label} "
            f"{result.line.frequency_hz / 1e3:7.1f} kHz: "
            f"strain={result.strain_peak:.2e}, "
            f"Rabi={result.rabi_hz:.2f} Hz, "
            f"s={result.saturation_parameter:.2e}, "
            f"Pspin={result.spin_absorbed_power_w:.2e} W, "
            f"dQ/Q={result.fractional_q_shift:.2e}, "
            f"SNRp={result.power_snr:.2g}, "
            f"SNRq={result.fractional_snr:.2g}"
        )


def plot_sweep(plt, voltages, sweep):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    colors = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}

    for label, results in sweep.items():
        color = colors.get(label, None)
        freq = results[0].line.frequency_hz / 1e3
        legend = f"{label} {freq:.0f} kHz"
        axes[0, 0].loglog(
            voltages,
            [item.strain_peak for item in results],
            label=legend,
            color=color,
        )
        axes[0, 1].loglog(
            voltages,
            [item.rabi_hz for item in results],
            label=legend,
            color=color,
        )
        axes[1, 0].loglog(
            voltages,
            [item.spin_absorbed_power_w for item in results],
            label=legend,
            color=color,
        )
        axes[1, 1].loglog(
            voltages,
            [item.fractional_q_shift for item in results],
            label=legend,
            color=color,
        )

    axes[0, 0].set_ylabel("peak strain")
    axes[0, 1].set_ylabel("quadrupolar Rabi rate (Hz)")
    axes[1, 0].set_ylabel("spin absorbed power (W)")
    axes[1, 1].set_ylabel("fractional acoustic loading")
    for ax in axes.ravel():
        ax.set_xlabel("drive voltage (V RMS)")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Glycine piezoelectric/acoustic NQR detection model")
    return fig


if __name__ == "__main__":
    main()
