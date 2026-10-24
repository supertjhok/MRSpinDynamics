"""Liquid-state CW ODNP: diffusion, field/power sweeps, buildup and recovery.

Run: python examples/plot_overhauser_dnp.py --output .tmp/overhauser_dnp.png
Parameters describe an illustrative uniform proton/radical solution. Saturation
is supplied calibration, not a prediction of nitroxide hyperfine-line mixing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.dnp import (
    OverhauserRates,
    TranslationalDiffusionModel,
    cw_overhauser,
    polarization_buildup,
)
from spin_dynamics.workflows.dnp import run_overhauser_power_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concentration-mm", type=float, default=1.0)
    parser.add_argument("--distance-nm", type=float, default=0.3)
    parser.add_argument("--diffusion-m2-s", type=float, default=3e-9)
    parser.add_argument("--s-max", type=float, default=0.8)
    args = parser.parse_args()
    model = TranslationalDiffusionModel(args.distance_nm * 1e-9, args.diffusion_m2_s)
    powers = np.linspace(0, 3, 100)
    sweep = run_overhauser_power_sweep(
        [0.1, 0.35, 1.0],
        powers,
        model=model,
        concentration_molar=args.concentration_mm * 1e-3,
        half_saturation_power_watts=0.3,
        maximum_saturation=args.s_max,
        background_r1_per_second=1 / 2.5,
    )
    plt = load_matplotlib(headless=args.output is not None)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for index, field in enumerate(sweep.field_tesla):
        axes[0, 0].plot(powers, sweep.cw.enhancement[:, index], label=f"{field:g} T")
        axes[0, 1].plot(
            powers,
            sweep.cw.steady_state_polarization[:, index] * 1e6,
            label=f"{field:g} T",
        )
    axes[0, 0].set(
        xlabel="Microwave power (W)",
        ylabel="E = P / P thermal",
        title="CW enhancement (baseline E = 1)",
    )
    axes[0, 1].set(
        xlabel="Microwave power (W)",
        ylabel="Absolute polarization (ppm)",
        title="Enhancement and absolute polarization differ",
    )
    axes[0, 0].legend()
    axes[0, 1].legend()

    selected = cw_overhauser(
        OverhauserRates(
            sweep.cw.rates.background_r1_per_second[-1, 1],
            sweep.cw.rates.self_relaxation_per_second[-1, 1],
            sweep.cw.rates.cross_relaxation_per_second[-1, 1],
        ),
        field_tesla=0.35,
        saturation_factor=sweep.cw.saturation_factor[-1, 1],
    )
    times = np.linspace(0, 10, 200)
    pump = polarization_buildup(selected, times)
    recover = polarization_buildup(selected, times, pumping=False)
    axes[1, 0].plot(times, pump.polarization * 1e6, label="Microwaves on")
    axes[1, 0].plot(times, recover.polarization * 1e6, label="Microwaves off")
    axes[1, 0].set(
        xlabel="Time (s)",
        ylabel="Polarization (ppm)",
        title="0.35 T: single-reservoir buildup / recovery",
    )
    axes[1, 0].legend()
    fields = np.geomspace(0.01, 3, 120)
    for tau_ps in [30, 54, 100, 300]:
        motion = TranslationalDiffusionModel(0.3e-9, (0.3e-9) ** 2 / (tau_ps * 1e-12))
        curve = run_overhauser_power_sweep(
            fields,
            [1],
            model=motion,
            concentration_molar=0.001,
            half_saturation_power_watts=0.3,
            background_r1_per_second=0.4,
        )
        axes[1, 1].semilogx(fields, curve.cw.coupling_factor[0], label=f"{tau_ps} ps")
    axes[1, 1].set(
        xlabel="Field (T)",
        ylabel="Coupling factor xi",
        title="FFHS diffusion dispersion, tau D = d² / D",
    )
    axes[1, 1].legend()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        plt.close(fig)
    else:
        plt.show()
    print(
        f"tau_D={model.diffusion_time_seconds * 1e12:.3f} ps; "
        f"xi={float(selected.coupling_factor):.6f}; "
        f"f={float(selected.leakage_factor):.6f}; "
        f"E={float(selected.enhancement):.3f}; "
        f"T1={float(selected.rates.t1_seconds):.3f} s"
    )


if __name__ == "__main__":
    main()
