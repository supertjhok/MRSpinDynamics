"""Compare ground-state ODMR transitions of diamond NV and 4H-SiC PL6 sensors.

Run ``python examples/plot_nano_mr_defect_odmr.py --help`` for adjustable field
range and output options.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    diamond_nv_minus,
    diagonalize_sensor,
    sic_pl6,
)


def transition_branches(sensor, fields_tesla: np.ndarray) -> np.ndarray:
    """Return the two strongest transverse-drive branches over an axial field."""

    branches = []
    for field in fields_tesla:
        result = diagonalize_sensor(
            sensor,
            field * sensor.axis_lab,
            drive_direction_lab=sensor.frame.x_axis_lab,
        )
        branches.append(
            sorted(item.frequency_hz for item in result.transitions)[:2]
        )
    return np.asarray(branches, dtype=np.float64)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=12.0)
    parser.add_argument("--points", type=int, default=241)
    parser.add_argument("--nv-depth-nm", type=float, default=5.0)
    parser.add_argument("--pl6-depth-nm", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    fields = np.linspace(-args.field_mt, args.field_mt, args.points) * 1.0e-3
    sensors = (
        diamond_nv_minus(depth_nm=args.nv_depth_nm),
        sic_pl6(depth_nm=args.pl6_depth_nm),
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    for axis, sensor in zip(axes, sensors):
        branches = transition_branches(sensor, fields)
        axis.plot(1.0e3 * fields, 1.0e-9 * branches[:, 0], label="lower branch")
        axis.plot(1.0e3 * fields, 1.0e-9 * branches[:, 1], label="upper branch")
        axis.set_title(f"{sensor.material} {sensor.defect}")
        axis.set_xlabel("Axial static field (mT)")
        axis.set_ylabel("Transition frequency (GHz)")
        axis.legend()

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
