"""Plot Brownian diffusion during a simple CPMG train in a static gradient."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path

add_src_to_path()

from spin_dynamics.motion import (
    advect_diffuse_positions,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for this example. Install it with "
            "`python -m pip install -e .[plot]`."
        ) from exc
    return plt


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


def _propagate_phase_interval(
    positions: np.ndarray,
    phase: np.ndarray,
    diffusion: np.ndarray,
    bounds,
    duration: float,
    substeps: int,
    rng: np.random.Generator,
    gradient: float,
    sign: float,
):
    dt = float(duration) / int(substeps)
    current_positions = positions
    current_phase = phase
    for _ in range(int(substeps)):
        current_positions = advect_diffuse_positions(
            current_positions,
            dt,
            diffusion_coefficient=diffusion,
            rng=rng,
            bounds=bounds,
            boundary="reflect",
        )
        current_phase = (
            current_phase
            + float(sign) * float(gradient) * current_positions[:, 0] * dt
        )
    return current_positions, current_phase


def _run_case(args: argparse.Namespace, diffusion: float, fields, case_index: int):
    rng = np.random.default_rng(args.seed + 1009 * case_index)
    ensemble = _initialize_walkers(
        args.num_particles,
        diffusion,
        args.seed + case_index,
    )
    start_positions = ensemble.positions.copy()
    echo_values = np.zeros(args.num_echoes, dtype=np.complex128)
    echo_times = args.echo_spacing * (np.arange(args.num_echoes, dtype=np.float64) + 1)
    positions = ensemble.positions.copy()
    phase = np.zeros(ensemble.num_particles, dtype=np.float64)
    half_echo = 0.5 * args.echo_spacing
    for echo_index in range(args.num_echoes):
        # The ideal CPMG toggling function flips static-gradient phase at every
        # 180-degree pulse. Fixed particles refocus exactly at the echo;
        # diffusing particles do not retrace the same field history.
        sign = 1.0 if echo_index % 2 == 0 else -1.0
        positions, phase = _propagate_phase_interval(
            positions,
            phase,
            ensemble.diffusion_coefficient,
            fields.bounds,
            half_echo,
            args.substeps,
            rng,
            args.gradient,
            sign,
        )
        positions, phase = _propagate_phase_interval(
            positions,
            phase,
            ensemble.diffusion_coefficient,
            fields.bounds,
            half_echo,
            args.substeps,
            rng,
            args.gradient,
            -sign,
        )
        echo_values[echo_index] = np.sum(ensemble.weights * np.exp(-1j * phase))

    return {
        "diffusion": float(diffusion),
        "echo_times": echo_times,
        "echo_values": echo_values,
        "start_positions": start_positions,
        "end_positions": positions.copy(),
    }


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 0 or args.echo_spacing <= 0.0:
        raise SystemExit("--num-echoes and --echo-spacing must be positive")
    if args.substeps <= 0:
        raise SystemExit("--substeps must be positive")
    if any(value < 0.0 for value in args.diffusion):
        raise SystemExit("--diffusion values must be non-negative")

    plt = _load_matplotlib()
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
