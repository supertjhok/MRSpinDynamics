"""Plot radiation-damping FID envelopes for several flip angles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.workflows import run_radiation_damping_fid


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this example. Install the optional "
            "plot dependency, for example: pip install matplotlib"
        ) from exc
    return plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--field-tesla", type=float, default=1.0)
    parser.add_argument("--polarization-scale", type=float, default=250.0)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="matched")
    parser.add_argument("--points", type=int, default=600)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    plt = _load_matplotlib()
    angles = np.deg2rad([10.0, 60.0, 100.0, 140.0])
    first = run_radiation_damping_fid(
        probe=args.probe,
        fill_factor=args.fill_factor,
        field_tesla=args.field_tesla,
        polarization_scale=args.polarization_scale,
        flip_angle=float(angles[0]),
        num_points=args.points,
    )
    trd = first.probe.trd

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)
    for theta in angles:
        result = run_radiation_damping_fid(
            probe=args.probe,
            fill_factor=args.fill_factor,
            field_tesla=args.field_tesla,
            polarization_scale=args.polarization_scale,
            flip_angle=float(theta),
            duration_seconds=first.time_seconds[-1],
            num_points=args.points,
        )
        label = f"{np.rad2deg(theta):.0f} deg"
        axes[0].plot(result.time_seconds / trd, result.envelope, label=label)
        axes[0].plot(
            result.time_seconds / trd,
            result.analytic_envelope,
            linestyle="--",
            color=axes[0].lines[-1].get_color(),
        )
        axes[1].plot(result.time_seconds / trd, result.mz, label=label)

    axes[0].set_title("Radiation-Damping FID Envelope")
    axes[0].set_xlabel("Time / Trd")
    axes[0].set_ylabel("|mxy|")
    axes[0].legend()
    axes[1].set_title("Longitudinal Recovery from Back-Action")
    axes[1].set_xlabel("Time / Trd")
    axes[1].set_ylabel("mz")
    axes[1].legend()
    fig.suptitle(f"{args.probe.capitalize()} probe, Q={first.probe.q:.0f}, Trd={trd:.3g} s")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
