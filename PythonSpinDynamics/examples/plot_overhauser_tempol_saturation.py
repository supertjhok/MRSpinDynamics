"""Illustrative 14N TEMPOL/water CW saturation and proton polarization.

Retune microwaves to the central hyperfine line at each field. Compare
concentrations including nitrogen relaxation, population-swap Heisenberg
exchange and independently supplied transverse broadening. All electron
parameters are illustrative inputs, NOT measured TEMPOL calibration or a
prediction of their field/temperature dependence. Resolved lines only.

Run: python examples/plot_overhauser_tempol_saturation.py --output .tmp/tempol.png
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib
from plot_overhauser_tempol_water import find_optimal_field

add_src_to_path()

from spin_dynamics.dnp import (
    NitroxideSaturationModel,
    TranslationalDiffusionModel,
    cw_overhauser,
    translational_relaxation_rates,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path, help="Field curves plus <stem>_optima.csv")
    parser.add_argument(
        "--concentrations-mm", type=float, nargs="+", default=[0.1, 1, 5, 10, 25]
    )
    parser.add_argument(
        "--exchange-coefficient",
        type=float,
        default=2e9,
        help="Population-correlation exchange coefficient, M^-1 s^-1",
    )
    parser.add_argument(
        "--broadening-coefficient",
        type=float,
        default=1e9,
        help="Additional 1/T2 per mol/L, M^-1 s^-1",
    )
    parser.add_argument(
        "--nitrogen-rate",
        type=float,
        default=1e4,
        help="Adjacent nitrogen transition rate per direction, s^-1",
    )
    parser.add_argument("--t1e-us", type=float, default=0.5)
    parser.add_argument("--t2e-us", type=float, default=0.1)
    parser.add_argument("--hyperfine-mhz", type=float, default=45)
    parser.add_argument(
        "--nutation-mrad-s",
        type=float,
        default=20,
        help="Co-rotating angular nutation in million rad/s",
    )
    parser.add_argument("--field-min", type=float, default=0.02)
    parser.add_argument("--field-max", type=float, default=10)
    args = parser.parse_args()
    concentrations = np.asarray(args.concentrations_mm) * 1e-3
    if not np.all(np.isfinite(concentrations) & (concentrations > 0)):
        parser.error("concentrations must be finite and positive")
    if not 0 < args.field_min < args.field_max:
        parser.error("require 0 < field-min < field-max")
    model = NitroxideSaturationModel(
        hyperfine_hz=args.hyperfine_mhz * 1e6,
        electron_t1_seconds=args.t1e_us * 1e-6,
        electron_t2_seconds=args.t2e_us * 1e-6,
        nitrogen_transition_per_second=args.nitrogen_rate,
        exchange_coefficient_per_molar_second=args.exchange_coefficient,
        transverse_broadening_per_molar_second=args.broadening_coefficient,
    )
    omega = args.nutation_mrad_s * 1e6
    diffusion = TranslationalDiffusionModel(0.39e-9, 2.8e-9)
    fields = np.geomspace(args.field_min, args.field_max, 241)
    concentration_axis = np.geomspace(1e-5, max(0.025, concentrations.max()), 201)
    sat = model.steady_state(
        field_tesla=0.35,
        concentration_molar=concentration_axis,
        nutation_rad_per_s=omega,
    )
    isolated = replace(model, exchange_coefficient_per_molar_second=0).steady_state(
        field_tesla=0.35,
        concentration_molar=concentration_axis,
        nutation_rad_per_s=omega,
    )
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].semilogx(
        concentration_axis * 1e3,
        sat.saturation_factor,
        label="Exchange + nitrogen relaxation",
    )
    axes[0, 0].semilogx(
        concentration_axis * 1e3,
        isolated.saturation_factor,
        "--",
        label="Exchange off; same broadening",
    )
    for index, m in enumerate(model.nitrogen_m):
        axes[0, 1].semilogx(
            concentration_axis * 1e3,
            sat.line_saturation[:, index],
            label=f"mN = {m:g}",
            linestyle=["-", "--", ":"][index],
        )
    rows, optima = [], []
    detuning_mhz = np.linspace(-80, 80, 481)
    for concentration in concentrations:

        def polarization(field):
            saturation = model.steady_state(
                field_tesla=field,
                concentration_molar=concentration,
                nutation_rad_per_s=omega,
            ).saturation_factor
            rates = translational_relaxation_rates(
                field, diffusion, concentration, background_r1_per_second=0.4
            )
            return cw_overhauser(
                rates, field_tesla=field, saturation_factor=saturation
            ).steady_state_polarization

        saturation = model.steady_state(
            field_tesla=fields,
            concentration_molar=concentration,
            nutation_rad_per_s=omega,
        )
        values = polarization(fields)
        optimum_field, optimum_p, boundary = find_optimal_field(
            polarization, args.field_min, args.field_max
        )
        label = f"{concentration * 1e3:g} mM"
        (line,) = axes[1, 0].semilogx(fields, values * 1e6, label=label)
        axes[1, 0].plot(optimum_field, optimum_p * 1e6, "o", color=line.get_color())
        spectrum = model.steady_state(
            field_tesla=0.35,
            concentration_molar=concentration,
            nutation_rad_per_s=omega,
            detuning_rad_per_s=2 * np.pi * detuning_mhz * 1e6,
        )
        axes[1, 1].plot(detuning_mhz, spectrum.saturation_factor, label=label)
        rows.extend(
            zip(
                np.full(len(fields), concentration * 1e3),
                fields,
                saturation.saturation_factor,
                values,
            )
        )
        optima.append(
            [
                concentration * 1e3,
                float(saturation.saturation_factor[0]),
                optimum_field,
                optimum_p,
                int(boundary),
            ]
        )
        print(
            f"{label}: s={float(saturation.saturation_factor[0]):.6f}; max |P| at {optimum_field:.6f} T, P={optimum_p * 1e6:.3f} ppm"
            + (" (interval boundary)" if boundary else "")
        )
    axes[0, 0].set(
        xlabel="TEMPOL concentration (mM)", ylabel="Effective saturation", ylim=(0, 1)
    )
    axes[0, 1].set(
        xlabel="TEMPOL concentration (mM)",
        ylabel="Conditional line saturation",
        ylim=(0, 1),
    )
    axes[1, 0].set(xlabel="Field (T)", ylabel="Proton polarization (ppm)")
    axes[1, 1].set(
        xlabel="Electron detuning / 2π (MHz)",
        ylabel="Effective saturation",
        ylim=(0, 1),
    )
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "TEMPOL/water: resolved 14N lines and concentration-dependent saturation\nIllustrative electron parameters; 298.15 K, FFHS water motion, T10 = 2.5 s",
        fontsize=12,
    )
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            rows,
            delimiter=",",
            comments="",
            header="concentration_mm,field_tesla,saturation_factor,proton_polarization",
        )
        np.savetxt(
            args.csv.with_name(args.csv.stem + "_optima.csv"),
            optima,
            delimiter=",",
            comments="",
            header="concentration_mm,saturation_factor,optimal_field_tesla,proton_polarization,boundary_optimum",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
