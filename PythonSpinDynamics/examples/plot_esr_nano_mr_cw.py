"""Cross-list pure ESR, defect ODMR, and resolved-nucleus CW models.

The left panel verifies the spin-1/2 zero-ZFS bridge.  The right panel uses the
exact nano-MR cluster backend to resolve a nearby proton coupled to an NV
sensor.  Run ``python examples/plot_esr_nano_mr_cw.py --help`` for options.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.esr import (  # noqa: E402
    ESRSpinSystem,
    simulate_frequency_spectrum,
    spectrum_from_lines,
)
from spin_dynamics.nano_mr import (  # noqa: E402
    ResolvedNucleus,
    ResolvedSpinCluster,
    defect_sensor_from_esr,
    diagonalize_sensor,
    diamond_nv_minus,
    simulate_resolved_cw_spectrum,
)


def _normalized(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    return values / scale if scale > 0.0 else values


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=20.0)
    parser.add_argument("--g", type=float, default=2.0028)
    parser.add_argument("--proton-distance-nm", type=float, default=2.0)
    parser.add_argument("--broadening-khz", type=float, default=3.0)
    parser.add_argument("--points", type=int, default=2001)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    field_tesla = args.field_mt * 1.0e-3
    b0 = np.array([0.0, 0.0, field_tesla])
    broadening_hz = args.broadening_khz * 1.0e3

    pure_system = ESRSpinSystem(g_tensor=args.g, label="spin-1/2 center")
    pure = simulate_frequency_spectrum(
        pure_system,
        field_tesla,
        broadening_hz=broadening_hz,
        points=args.points,
        span_hz=20.0 * broadening_hz,
    )
    promoted_sensor = defect_sensor_from_esr(pure_system, depth_nm=2.0)
    promoted = diagonalize_sensor(promoted_sensor, b0)
    promoted_spectrum = spectrum_from_lines(
        pure.frequencies_hz,
        [line.frequency_hz for line in promoted.transitions],
        [line.strength for line in promoted.transitions],
        width=broadening_hz,
    )

    nv = diamond_nv_minus(depth_nm=3.0)
    proton = ResolvedNucleus.from_isotope(
        "1H",
        [0.0, 0.0, args.proton_distance_nm],
        label="surface proton",
    )
    cluster = ResolvedSpinCluster(
        nv,
        (proton,),
        sensor_position_lab_nm=[0.0, 0.0, 0.0],
    )
    bare_nv = diagonalize_sensor(nv, b0, drive_direction_lab=[1.0, 0.0, 0.0])
    center_hz = min(line.frequency_hz for line in bare_nv.transitions)
    cluster_axis = np.linspace(center_hz - 75.0e3, center_hz + 75.0e3, args.points)
    cluster_result = simulate_resolved_cw_spectrum(
        cluster,
        b0,
        frequencies_hz=cluster_axis,
        broadening_hz=broadening_hz,
    )
    bare_spectrum = spectrum_from_lines(
        cluster_axis,
        [line.frequency_hz for line in bare_nv.transitions],
        [line.strength for line in bare_nv.transitions],
        width=broadening_hz,
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True)
    frequency_offset = pure.lines[0].frequency_hz
    axes[0].plot(
        1.0e-3 * (pure.frequencies_hz - frequency_offset),
        _normalized(pure.spectrum),
        linewidth=2.5,
        label="pure ESR",
    )
    axes[0].plot(
        1.0e-3 * (pure.frequencies_hz - frequency_offset),
        _normalized(promoted_spectrum),
        linestyle="--",
        linewidth=1.8,
        label="zero-ZFS defect bridge",
    )
    axes[0].set_title("Compatible Spin-1/2 CW Models")
    axes[0].set_xlabel("Offset from resonance (kHz)")
    axes[0].set_ylabel("Normalized absorption")
    axes[0].legend()

    axes[1].plot(
        1.0e-3 * (cluster_axis - center_hz),
        _normalized(bare_spectrum),
        color="0.55",
        linestyle="--",
        label="bare NV branch",
    )
    axes[1].plot(
        1.0e-3 * (cluster_axis - center_hz),
        _normalized(cluster_result.spectrum),
        color="tab:purple",
        linewidth=2.0,
        label="NV + resolved proton",
    )
    axes[1].set_title("Exact Resolved-Cluster CW ODMR")
    axes[1].set_xlabel("Offset from bare NV transition (kHz)")
    axes[1].set_ylabel("Normalized absorption")
    axes[1].legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
