"""Plot linear motion of a spin packet through static B0/B1 maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.motion import (
    free_precession_with_motion_step,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
    receive_signal,
)




def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=70, help="Number of x map samples.")
    parser.add_argument("--nz", type=int, default=48, help="Number of z map samples.")
    parser.add_argument("--steps", type=int, default=160, help="Motion time steps.")
    parser.add_argument(
        "--total-time",
        type=float,
        default=1.2,
        help="Simulation time.",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=1.35,
        help="Packet x velocity.",
    )
    parser.add_argument("--t2", type=float, default=1.6, help="Transverse decay time.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    return parser.parse_args()


def _make_inside_out_maps(nx: int, nz: int):
    x_axis = np.linspace(-1.0, 1.0, nx)
    z_axis = np.linspace(-0.8, 0.8, nz)
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    radius = np.sqrt(xx**2 + zz**2)

    # The receive map mimics an inside-out tool with a sensitive shell outside
    # the tool body rather than a uniform receive sensitivity.
    b1_rx = np.exp(-0.5 * ((radius - 0.58) / 0.12) ** 2)
    b1_tx = 0.85 + 0.15 * np.exp(-0.5 * (radius / 0.7) ** 2)

    # A static inhomogeneous B0 map makes packet motion visible in signal phase.
    b0 = 7.0 * xx + 2.0 * np.exp(-((xx - 0.25) ** 2 + (zz + 0.15) ** 2) / 0.08)
    return x_axis, z_axis, b0, b1_tx, b1_rx


def _initialize_packet(x_axis: np.ndarray, z_axis: np.ndarray):
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    rho = np.exp(-((xx + 0.75) ** 2 / 0.025 + zz**2 / 0.04))
    ensemble = initialize_ensemble_from_density(
        rho,
        x_axis,
        z_axis,
        walkers_per_cell=1,
    )
    magnetization = ensemble.magnetization.copy()
    magnetization[1, :] = 1.0
    magnetization[2, :] = 1.0
    return ensemble.with_updates(magnetization=magnetization)


def main() -> None:
    args = _parse_args()
    if args.nx < 3 or args.nz < 3:
        raise SystemExit("--nx and --nz must be at least 3")
    if args.steps <= 0 or args.total_time <= 0.0:
        raise SystemExit("--steps and --total-time must be positive")

    plt = load_matplotlib()
    x_axis, z_axis, b0, b1_tx, b1_rx = _make_inside_out_maps(args.nx, args.nz)
    fields = make_motion_field_maps_2d(
        x_axis,
        z_axis,
        b0_map=b0,
        b1_tx_map=b1_tx,
        b1_rx_map=b1_rx,
    )
    ensemble = _initialize_packet(x_axis, z_axis)

    dt = args.total_time / args.steps
    velocity = np.array([args.velocity, 0.0], dtype=np.float64)
    times = np.linspace(0.0, args.total_time, args.steps + 1)
    signal = np.zeros(times.size, dtype=np.complex128)
    center = np.zeros((times.size, 2), dtype=np.float64)
    snapshots: list[np.ndarray] = []
    snapshot_indices = {0, args.steps // 2, args.steps}

    for step, time in enumerate(times):
        signal[step] = receive_signal(ensemble, fields)
        center[step] = np.average(
            ensemble.positions,
            axis=0,
            weights=np.maximum(ensemble.weights, 0.0),
        )
        if step in snapshot_indices:
            snapshots.append(ensemble.positions.copy())
        if step == args.steps:
            break
        ensemble = free_precession_with_motion_step(
            ensemble,
            fields,
            dt,
            velocity=velocity,
            time=time,
            t1=np.inf,
            t2=args.t2,
            boundary="clip",
        )

    extent = [z_axis[0], z_axis[-1], x_axis[0], x_axis[-1]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    b0_img = axes[0, 0].imshow(b0, origin="lower", extent=extent, aspect="auto")
    fig.colorbar(b0_img, ax=axes[0, 0], label="B0 offset")
    axes[0, 0].plot(center[:, 1], center[:, 0], color="white", linewidth=2)

    b1_img = axes[0, 1].imshow(b1_rx, origin="lower", extent=extent, aspect="auto")
    fig.colorbar(b1_img, ax=axes[0, 1], label="B1 receive")
    for positions, label in zip(snapshots, ["start", "middle", "end"]):
        stride = max(1, positions.shape[0] // 300)
        axes[0, 1].scatter(
            positions[::stride, 1],
            positions[::stride, 0],
            s=5,
            alpha=0.35,
            label=label,
        )
    axes[0, 1].legend(loc="upper right")

    axes[1, 0].plot(times, np.abs(signal), color="tab:blue")
    axes[1, 1].plot(times, np.unwrap(np.angle(signal)), color="tab:orange")
    axes[1, 1].plot(times, center[:, 0], color="tab:green", linestyle="--")

    axes[0, 0].set_title("Static B0 Map and Packet Path")
    axes[0, 1].set_title("Inside-Out Receive Lobe")
    axes[1, 0].set_title("Received Signal Magnitude")
    axes[1, 1].set_title("Signal Phase and Packet X")
    axes[1, 1].legend(["phase", "center x"], loc="best")

    for ax in axes[0, :]:
        ax.set_xlabel("z")
        ax.set_ylabel("x")
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("|signal|")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("phase / position")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
