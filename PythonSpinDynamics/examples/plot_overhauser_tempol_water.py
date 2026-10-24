"""Illustrative liquid-state CW Overhauser DNP of TEMPOL in water versus field.

Run from PythonSpinDynamics:
python examples/plot_overhauser_tempol_water.py --output .tmp/tempol_water.png

Defaults: 10 mM TEMPOL, 298.15 K, d=0.39 nm, D_relative=2.8e-9 m^2/s,
radical-free T10=2.5 s, effective electron saturation s=0.8. These are explicit
illustrative assumptions, not a fit to a particular experimental sample.
Diffusion and T10 are held constant with field; the electron resonance is
assumed retuned at every field to maintain the chosen effective saturation.

Context: Field Dependent Dynamic Nuclear Polarization with Radicals in Aqueous
Solution, DOI 10.1021/ja0783207. The ~54 ps bulk-water diffusion scale is discussed
by Franck et al., DOI 10.1016/j.pnmrs.2013.06.001. The FFHS approximation and
supplied saturation do not predict nitroxide hyperfine-line mixing or heating.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.dnp import (
    TranslationalDiffusionModel,
    cw_overhauser,
    translational_relaxation_rates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path, help="Optional field-resolved model data")
    parser.add_argument("--concentration-mm", type=float, default=10.0)
    parser.add_argument("--temperature-k", type=float, default=298.15)
    parser.add_argument("--distance-nm", type=float, default=0.39)
    parser.add_argument("--diffusion-m2-s", type=float, default=2.8e-9)
    parser.add_argument("--t10-s", type=float, default=2.5)
    parser.add_argument("--saturation", type=float, default=0.8)
    parser.add_argument("--field-min-t", type=float, default=0.02)
    parser.add_argument("--field-max-t", type=float, default=10.0)
    parser.add_argument("--points", type=int, default=240)
    args = parser.parse_args()
    if args.points < 2 or not 0 < args.field_min_t < args.field_max_t:
        parser.error("require at least two points and 0 < field-min-t < field-max-t")
    if not np.isfinite(args.t10_s) or args.t10_s <= 0:
        parser.error("t10-s must be positive and finite")

    model = TranslationalDiffusionModel(args.distance_nm * 1e-9, args.diffusion_m2_s)
    fields = np.geomspace(args.field_min_t, args.field_max_t, args.points)
    rates = translational_relaxation_rates(
        fields,
        model,
        args.concentration_mm * 1e-3,
        background_r1_per_second=1 / args.t10_s,
    )
    cw = cw_overhauser(
        rates,
        field_tesla=fields,
        saturation_factor=args.saturation,
        temperature_kelvin=args.temperature_k,
    )
    ideal = cw_overhauser(
        rates,
        field_tesla=fields,
        saturation_factor=1,
        temperature_kelvin=args.temperature_k,
    )
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    fig.suptitle(
        f"CW Overhauser DNP: {args.concentration_mm:g} mM TEMPOL in water\n"
        f"{args.temperature_k:g} K | d = {args.distance_nm:g} nm | "
        f"D relative = {args.diffusion_m2_s / 1e-9:g} × 10⁻⁹ m²/s | "
        f"τ D = {model.diffusion_time_seconds * 1e12:.1f} ps\n"
        f"Illustrative FFHS model; T₁₀ = {args.t10_s:g} s and "
        f"effective saturation s = {args.saturation:g} held constant",
        fontsize=12,
    )
    axes[0, 0].plot(fields, cw.enhancement, label=f"Effective s = {args.saturation:g}")
    axes[0, 0].plot(
        fields, ideal.enhancement, "--", color="0.55", label="Full saturation s = 1"
    )
    axes[0, 0].axhline(1, color="0.65", lw=1, ls=":", label="Microwaves off: E = 1")
    axes[0, 0].set(title="Proton enhancement", ylabel="E = P steady / P thermal")
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].plot(fields, cw.steady_state_polarization * 1e6, label="CW steady state")
    axes[0, 1].plot(
        fields, cw.thermal_polarization * 1e6, ":", label="Thermal equilibrium"
    )
    axes[0, 1].set(title="Absolute proton polarization", ylabel="Polarization (ppm)")
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(fields, cw.coupling_factor, label="Coupling ξ")
    axes[1, 0].plot(fields, cw.leakage_factor, label="Leakage f")
    axes[1, 0].set(
        title="Transfer and relaxation factors",
        ylabel="Dimensionless factor",
        ylim=(0, 1),
    )
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].plot(fields, rates.t1_seconds, label="T₁ with TEMPOL")
    axes[1, 1].axhline(
        args.t10_s, ls="--", color="0.55", label="Assumed radical-free T₁₀"
    )
    axes[1, 1].set(title="Nuclear buildup / recovery time", ylabel="Time (s)")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set(xscale="log", xlabel="Polarization and detection field B₀ (T)")
        ax.grid(alpha=0.2, which="both")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            fields,
            cw.enhancement,
            ideal.enhancement,
            cw.steady_state_polarization,
            cw.thermal_polarization,
            cw.coupling_factor,
            cw.leakage_factor,
            cw.saturation_factor,
            rates.t1_seconds,
            rates.self_relaxation_per_second,
            rates.cross_relaxation_per_second,
        ]
        np.savetxt(
            args.csv,
            np.column_stack(columns),
            delimiter=",",
            comments="",
            header="field_tesla,enhancement,full_saturation_enhancement,"
            "steady_polarization,thermal_polarization,coupling_factor,"
            "leakage_factor,saturation_factor,t1_seconds,"
            "self_rate_per_second,cross_rate_per_second",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    print(
        f"TEMPOL/water: tau_D={model.diffusion_time_seconds * 1e12:.3f} ps; "
        f"field={fields[0]:g}..{fields[-1]:g} T; "
        f"E={cw.enhancement[0]:.3f}..{cw.enhancement[-1]:.3f}; "
        f"s={args.saturation:g} (assumed constant)"
    )


if __name__ == "__main__":
    main()
