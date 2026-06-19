"""Plot Brownian diffusion during a simple CPMG train in a static gradient."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.motion import (
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence




def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=900, help="Walker count.")
    parser.add_argument("--num-echoes", type=int, default=18, help="Number of echoes.")
    parser.add_argument(
        "--echo-spacing",
        type=float,
        default=0.08,
        help="Echo spacing.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=10,
        help="Substeps per interval.",
    )
    parser.add_argument(
        "--gradient",
        type=float,
        default=45.0,
        help="Static x gradient.",
    )
    parser.add_argument(
        "--excitation-duration",
        type=float,
        default=0.002,
        help="Rectangular 90-degree pulse duration.",
    )
    parser.add_argument(
        "--refocusing-duration",
        type=float,
        default=0.004,
        help="Rectangular 180-degree pulse duration.",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        nargs="+",
        default=[0.0, 0.0015, 0.0045],
        help="Diffusion coefficients in map-coordinate units squared per time.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    return parser.parse_args()


def _make_fields():
    x_axis = np.linspace(-1.0, 1.0, 80)
    z_axis = np.linspace(-0.25, 0.25, 12)
    return make_motion_field_maps_2d(x_axis, z_axis)


def _initialize_walkers(num_particles: int, diffusion: float, seed: int):
    if num_particles <= 0:
        raise ValueError("num_particles must be positive")
    side = int(np.ceil(np.sqrt(num_particles)))
    x_axis = np.linspace(-0.65, 0.65, side)
    z_axis = np.linspace(-0.05, 0.05, side)
    rho = np.ones((side, side), dtype=np.float64)
    ensemble = initialize_ensemble_from_density(
        rho,
        x_axis,
        z_axis,
        diffusion_coefficient=float(diffusion),
        seed=seed,
        jitter=True,
    )
    if ensemble.num_particles > num_particles:
        keep = slice(0, int(num_particles))
        ensemble = ensemble.__class__(
            positions=ensemble.positions[keep],
            magnetization=ensemble.magnetization[:, keep],
            weights=ensemble.weights[keep],
            diffusion_coefficient=ensemble.diffusion_coefficient[keep],
        )
    magnetization = ensemble.magnetization.copy()
    magnetization[0, :] = 1.0
    magnetization[1:, :] = 0.0
    return ensemble.with_updates(magnetization=magnetization)


def _run_case(args: argparse.Namespace, diffusion: float, fields, case_index: int):
    rng = np.random.default_rng(args.seed + 1009 * case_index)
    ensemble = _initialize_walkers(
        args.num_particles,
        diffusion,
        args.seed + case_index,
    )
    start_positions = ensemble.positions.copy()
    result = run_motion_cpmg_sequence(
        ensemble,
        fields,
        num_echoes=args.num_echoes,
        echo_spacing=args.echo_spacing,
        excitation_duration=args.excitation_duration,
        refocusing_duration=args.refocusing_duration,
        gradient=(args.gradient, 0.0),
        rng=rng,
        substeps_per_interval=args.substeps,
    )

    return {
        "diffusion": float(diffusion),
        "echo_times": result.sample_times,
        "echo_values": result.signal,
        "start_positions": start_positions,
        "end_positions": result.final_ensemble.positions.copy(),
    }


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 0 or args.echo_spacing <= 0.0:
        raise SystemExit("--num-echoes and --echo-spacing must be positive")
    if args.substeps <= 0:
        raise SystemExit("--substeps must be positive")
    if args.excitation_duration <= 0.0 or args.refocusing_duration <= 0.0:
        raise SystemExit(
            "--excitation-duration and --refocusing-duration must be positive"
        )
    if args.echo_spacing < args.refocusing_duration:
        raise SystemExit("--echo-spacing must be at least --refocusing-duration")
    if any(value < 0.0 for value in args.diffusion):
        raise SystemExit("--diffusion values must be non-negative")

    plt = load_matplotlib()
    fields = _make_fields()
    rows = [
        _run_case(args, diffusion, fields, idx)
        for idx, diffusion in enumerate(args.diffusion)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rows)))

    for row, color in zip(rows, colors):
        normalized = np.abs(row["echo_values"]) / max(
            np.abs(row["echo_values"][0]),
            np.finfo(float).eps,
        )
        axes[0, 0].plot(
            np.arange(1, args.num_echoes + 1),
            normalized,
            marker="o",
            color=color,
            label=f"D={row['diffusion']:g}",
        )
        axes[0, 1].plot(
            row["echo_times"],
            np.unwrap(np.angle(row["echo_values"])),
            color=color,
        )

    strongest = rows[-1]
    stride = max(1, strongest["start_positions"].shape[0] // 500)
    axes[1, 0].scatter(
        strongest["start_positions"][::stride, 0],
        strongest["start_positions"][::stride, 1],
        s=6,
        alpha=0.35,
        label="start",
    )
    axes[1, 0].scatter(
        strongest["end_positions"][::stride, 0],
        strongest["end_positions"][::stride, 1],
        s=6,
        alpha=0.35,
        label="end",
    )

    x_axis = fields.x_axis
    gradient_profile = args.gradient * x_axis
    axes[1, 1].plot(x_axis, gradient_profile, color="tab:red")

    axes[0, 0].set_title("CPMG Echo Attenuation")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("normalized |echo|")
    axes[0, 0].legend()

    axes[0, 1].set_title("Echo Phase")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("unwrapped phase")

    axes[1, 0].set_title(f"Walker Cloud for D={strongest['diffusion']:g}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("z")
    axes[1, 0].legend()
    axes[1, 0].set_aspect("equal", adjustable="box")

    axes[1, 1].set_title("Static Gradient Profile")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("B0 offset")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
