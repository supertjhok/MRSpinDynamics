"""Compare TEMPOL/water concentrations and maximize |proton polarization|.

Run from PythonSpinDynamics (requires the opt and plot extras):
python examples/plot_overhauser_tempol_water.py --output .tmp/tempol_water.png

Defaults: 0.1, 1, 5, 10, 25 mM TEMPOL, 298.15 K, d=0.39 nm,
D_relative=2.8e-9 m^2/s, radical-free T10=2.5 s, effective saturation s=0.8.
These are illustrative assumptions, not fits to experimental samples. Diffusion,
T10 and saturation are held constant with field and concentration. Electron
irradiation is assumed retuned at each field. Optima maximize total |P| within
the requested field interval, including its endpoints and thermal polarization.

Context: DOI 10.1021/ja0783207 and DOI 10.1016/j.pnmrs.2013.06.001.
The FFHS model with prescribed saturation does not calculate concentration-
dependent hyperfine-line mixing, radical clustering, viscosity or heating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.dnp import (
    TranslationalDiffusionModel,
    cw_overhauser,
    translational_relaxation_rates,
)


def find_optimal_field(polarization_at_field, field_min_t, field_max_t):
    """Maximize |P| in a bounded interval, retaining sign and a boundary flag.

    Search 513 logarithmic samples, refine every sampled local maximum in
    log(field), then compare with both endpoints. This resolves smooth FFHS
    dispersion independently of plot resolution, not arbitrary narrow peaks.
    """
    if (
        not np.isfinite([field_min_t, field_max_t]).all()
        or not 0 < field_min_t < field_max_t
    ):
        raise ValueError("require finite 0 < field_min_t < field_max_t")
    logs = np.linspace(np.log(field_min_t), np.log(field_max_t), 513)
    scores = np.abs(polarization_at_field(np.exp(logs)))
    if scores.shape != logs.shape or not np.isfinite(scores).all():
        raise ValueError(
            "polarization evaluator must return finite values matching the field axis"
        )
    candidates = [field_min_t, field_max_t]
    peaks = (
        np.flatnonzero((scores[1:-1] > scores[:-2]) & (scores[1:-1] >= scores[2:])) + 1
    )
    for index in peaks:
        optimum = minimize_scalar(
            lambda log_field: -abs(float(polarization_at_field(np.exp(log_field)))),
            bounds=(logs[index - 1], logs[index + 1]),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not optimum.success:
            raise RuntimeError(f"field optimization failed: {optimum.message}")
        candidates.extend([float(np.exp(logs[index])), float(np.exp(optimum.x))])
    field = max(candidates, key=lambda b: abs(float(polarization_at_field(b))))
    polarization = float(polarization_at_field(field))
    return field, polarization, field in (field_min_t, field_max_t)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--csv", type=Path, help="Field-resolved data; also writes <stem>_optima.csv"
    )
    parser.add_argument(
        "--optima-csv", type=Path, help="Override the optimum-summary CSV path"
    )
    concentrations = parser.add_mutually_exclusive_group()
    concentrations.add_argument(
        "--concentration-mm", type=float, help="Single-concentration compatibility mode"
    )
    concentrations.add_argument(
        "--concentrations-mm",
        type=float,
        nargs="+",
        help="Concentrations in mM (default: 0.1 1 5 10 25)",
    )
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--distance-nm", type=float, default=0.39)
    parser.add_argument("--diffusion-m2-s", type=float, default=2.8e-9)
    parser.add_argument("--t10-s", type=float, default=2.5)
    parser.add_argument("--saturation", type=float, default=0.8)
    parser.add_argument("--field-min-t", type=float, default=0.02)
    parser.add_argument("--field-max-t", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=240)
    args = parser.parse_args()
    if (
        args.points < 2
        or not np.isfinite([args.field_min_t, args.field_max_t]).all()
        or not 0 < args.field_min_t < args.field_max_t
    ):
        parser.error(
            "require at least two points and finite 0 < field-min-t < field-max-t"
        )
    if not np.isfinite(args.t10_s) or args.t10_s <= 0:
        parser.error("t10-s must be positive and finite")
    concentration_values = (
        [args.concentration_mm]
        if args.concentration_mm is not None
        else args.concentrations_mm
        if args.concentrations_mm is not None
        else [0.1, 1, 5, 10, 25]
    )
    if not np.isfinite(concentration_values).all() or np.any(
        np.array(concentration_values) < 0
    ):
        parser.error("concentrations must be finite and non-negative")
    optima_path = args.optima_csv
    if args.csv:
        optima_path = optima_path or args.csv.with_name(args.csv.stem + "_optima.csv")
    if args.csv and optima_path and args.csv.resolve() == optima_path.resolve():
        parser.error("field data and optimum summary require different CSV paths")

    model = TranslationalDiffusionModel(args.distance_nm * 1e-9, args.diffusion_m2_s)
    fields = np.geomspace(args.field_min_t, args.field_max_t, args.points)
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    fig.suptitle(
        "CW Overhauser DNP: TEMPOL concentration comparison in water\n"
        f"{args.temperature_k:g} K | d = {args.distance_nm:g} nm | "
        f"D relative = {args.diffusion_m2_s / 1e-9:g} × 10⁻⁹ m²/s | "
        f"τ D = {model.diffusion_time_seconds * 1e12:.1f} ps\n"
        f"Illustrative FFHS model; T₁₀ = {args.t10_s:g} s and "
        f"effective saturation s = {args.saturation:g} held constant",
        fontsize=12,
    )
    data_rows, optima_rows = [], []
    print(
        f"TEMPOL/water: maximize |P| within {fields[0]:g}..{fields[-1]:g} T; "
        f"s={args.saturation:g} held constant"
    )
    print("concentration_mM  optimal_field_T  signed_P_ppm  abs_P_ppm  location")
    for concentration in concentration_values:

        def evaluate(field, saturation=args.saturation):
            rates = translational_relaxation_rates(
                field,
                model,
                concentration * 1e-3,
                background_r1_per_second=1 / args.t10_s,
            )
            return cw_overhauser(
                rates,
                field_tesla=field,
                saturation_factor=saturation,
                temperature_kelvin=args.temperature_k,
            )

        cw, ideal = evaluate(fields), evaluate(fields, 1)
        best_field, best_polarization, boundary = find_optimal_field(
            lambda field: evaluate(field).steady_state_polarization,
            args.field_min_t,
            args.field_max_t,
        )
        best = evaluate(best_field)
        label = f"{concentration:g} mM"
        (line,) = axes[0, 0].plot(fields, cw.enhancement, label=label)
        color = line.get_color()
        axes[0, 1].plot(
            fields, cw.steady_state_polarization * 1e6, color=color, label=label
        )
        axes[0, 1].plot(best_field, best_polarization * 1e6, "*", color=color, ms=11)
        axes[0, 1].annotate(
            f"{best_field:.2f} T" + (" (edge)" if boundary else ""),
            (best_field, best_polarization * 1e6),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
            horizontalalignment="right" if best_field == args.field_max_t else "left",
        )
        axes[1, 0].plot(fields, cw.leakage_factor, color=color, label=label)
        axes[1, 1].plot(fields, cw.rates.t1_seconds, color=color, label=label)
        columns = [
            np.full_like(fields, concentration),
            fields,
            cw.enhancement,
            ideal.enhancement,
            cw.steady_state_polarization,
            cw.thermal_polarization,
            cw.coupling_factor,
            cw.leakage_factor,
            cw.saturation_factor,
            cw.rates.t1_seconds,
            cw.rates.self_relaxation_per_second,
            cw.rates.cross_relaxation_per_second,
        ]
        data_rows.append(np.column_stack(columns))
        optima_rows.append(
            [
                concentration,
                best_field,
                best_polarization,
                abs(best_polarization),
                float(best.enhancement),
                float(best.leakage_factor),
                float(best.rates.t1_seconds),
                int(boundary),
                args.field_min_t,
                args.field_max_t,
            ]
        )
        print(
            f"{concentration:16g}  {best_field:15.6f}  {best_polarization * 1e6:12.3f}  "
            f"{abs(best_polarization) * 1e6:9.3f}  "
            + ("boundary" if boundary else "interior")
        )

    axes[0, 0].axhline(1, color="0.65", lw=1, ls=":")
    axes[0, 0].set(title="Proton enhancement", ylabel="E = P steady / P thermal")
    axes[0, 1].plot(
        fields, cw.thermal_polarization * 1e6, ":", color="0.3", label="Thermal"
    )
    axes[0, 1].plot([], [], "*", color="0.3", ms=10, label="Maximum |P| in range")
    axes[0, 1].set(
        title="Absolute polarization and optimal field",
        ylabel="Signed polarization (ppm)",
    )
    axes[1, 0].plot(
        fields, cw.coupling_factor, "--", color="0.3", label="Coupling ξ (all C)"
    )
    axes[1, 0].set(
        title="Leakage f and common coupling factor ξ",
        ylabel="Dimensionless factor",
        ylim=(0, 1),
    )
    axes[1, 1].axhline(args.t10_s, ls="--", color="0.55", label="Radical-free T₁₀")
    axes[1, 1].set(title="Nuclear buildup / recovery time", ylabel="Time (s)")
    for ax in axes.flat:
        ax.set(xscale="log", xlabel="Polarization and detection field B₀ (T)")
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=8)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            args.csv,
            np.vstack(data_rows),
            delimiter=",",
            comments="",
            header="concentration_mM,field_tesla,enhancement,full_saturation_enhancement,"
            "steady_polarization,thermal_polarization,coupling_factor,"
            "leakage_factor,saturation_factor,t1_seconds,"
            "self_rate_per_second,cross_rate_per_second",
        )
    if optima_path:
        optima_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            optima_path,
            optima_rows,
            delimiter=",",
            comments="",
            header="concentration_mM,optimal_field_tesla,steady_polarization,"
            "absolute_polarization,enhancement,leakage_factor,t1_seconds,at_boundary,"
            "field_min_tesla,field_max_tesla",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
