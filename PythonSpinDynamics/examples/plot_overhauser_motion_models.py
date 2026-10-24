"""Compare molecular-motion channels in the liquid-state CW Overhauser model.

Run: python examples/plot_overhauser_motion_models.py --output .tmp/dnp_motion.png

All numbers are illustrative proton-reservoir parameters at 298.15 K, not a
TEMPOL fit. FFHS: 10 mM, d=0.39 nm, D=2.8e-9 m^2/s. Bound dipolar pairs:
r=0.9 nm, rank-2 tau=100 ps, 0.01 independent pairs per observed nucleus.
Scalar contact fluctuations: RMS delta_A=0.3 MHz, tau=5 ps, fraction=0.02.
The mixed model sums independent rate contributions without cross correlations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.dnp import (
    MotionBasedCWModel,
    RotationalDipolarModel,
    ScalarContactModel,
    TranslationalDiffusionModel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--saturation", type=float, default=0.8)
    args = parser.parse_args()
    diffusion = TranslationalDiffusionModel(0.39e-9, 2.8e-9)
    rotation = RotationalDipolarModel(0.9e-9, 100e-12, pair_count=0.01)
    scalar = ScalarContactModel(0.3e6, 5e-12, contact_fraction=0.02)
    mixed = MotionBasedCWModel(diffusion, 0.01, (rotation,), (scalar,))
    models = {
        "FFHS translation": MotionBasedCWModel(diffusion, 0.01),
        "Bound-vector rotation": MotionBasedCWModel(rotational_models=(rotation,)),
        "Scalar contact": MotionBasedCWModel(scalar_models=(scalar,)),
        "Independent mixed rates": mixed,
    }
    fields = np.geomspace(0.01, 10, 220)
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    fig.suptitle(
        "Motion-based CW Overhauser DNP: translation, rotation and scalar contact\n"
        f"Illustrative independent proton channels | 298.15 K | s = {args.saturation:g} | T₁₀ = 2.5 s\n"
        "FFHS: 10 mM, τ D = 54.3 ps | bound pairs: 0.01, τ₂ = 100 ps | "
        "contact: 0.3 MHz RMS, 5 ps, fraction 0.02",
        fontsize=11,
    )
    rows = []
    for label, model in models.items():
        cw = model.cw(
            fields, saturation_factor=args.saturation, background_r1_per_second=0.4
        )
        (line,) = axes[0, 0].plot(fields, cw.enhancement, label=label)
        axes[0, 1].plot(
            fields,
            cw.steady_state_polarization * 1e6,
            color=line.get_color(),
            label=label,
        )
        rows.append(
            np.column_stack(
                [
                    fields,
                    cw.enhancement,
                    cw.steady_state_polarization,
                    cw.coupling_factor,
                    cw.leakage_factor,
                ]
            )
        )
        print(
            f"{label}: E at {fields[0]:g} T = {cw.enhancement[0]:.3f}; "
            f"at {fields[-1]:g} T = {cw.enhancement[-1]:.3f}"
        )
    for label, rates in mixed.rate_components(fields).items():
        if label == "background":
            continue
        (line,) = axes[1, 0].plot(fields, rates.self_relaxation_per_second, label=label)
        axes[1, 1].plot(
            fields,
            rates.cross_relaxation_per_second,
            color=line.get_color(),
            label=label,
        )
    total = mixed.relaxation_rates(fields)
    axes[1, 0].plot(
        fields, total.self_relaxation_per_second, "k--", label="Total paramagnetic"
    )
    axes[1, 1].plot(
        fields, total.cross_relaxation_per_second, "k--", label="Total cross rate"
    )
    axes[0, 0].axhline(1, color="0.5", ls=":", lw=1)
    axes[1, 1].axhline(0, color="0.5", lw=1)
    axes[0, 0].set(title="CW enhancement from motion", ylabel="E = P / P thermal")
    axes[0, 1].set(
        title="Absolute proton polarization", ylabel="Signed polarization (ppm)"
    )
    axes[1, 0].set(
        title="Mixed-model self-relaxation contributions",
        ylabel="R ρ (s⁻¹)",
        yscale="log",
    )
    axes[1, 1].set(
        title="Mixed-model cross-relaxation contributions", ylabel="R σ (s⁻¹)"
    )
    for ax in axes.flat:
        ax.set(xscale="log", xlabel="Field B₀ (T)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        data = np.vstack(
            [
                np.column_stack([np.full(fields.size, index), row])
                for index, row in enumerate(rows)
            ]
        )
        np.savetxt(
            args.csv,
            data,
            delimiter=",",
            comments="",
            header="model_index,field_tesla,enhancement,polarization,coupling_factor,leakage_factor",
        )
        print("CSV model_index order: " + "; ".join(models))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
