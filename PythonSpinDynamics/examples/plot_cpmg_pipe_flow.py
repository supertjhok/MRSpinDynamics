"""Simulate CPMG detection of laminar flow through a polarized pipe.

The example uses the moving-isochromat machinery directly. A cylindrical pipe is
reduced to an axisymmetric ``(x, z)`` slice: ``x`` is the transverse coordinate
across the pipe diameter and ``z`` is the downstream flow direction. The spin
density is weighted by the circular chord length, while the flow velocity is the
Poiseuille profile

    v_z(x) = 2 * v_mean * (1 - (x / R)^2).

Spins are polarized in an upstream static magnet, then translated into
downstream transmit/receive coils for a finite CPMG train. Increasing the mean
velocity reduces the upstream residence time, moves spins through the B0/B1 maps
during the train, and therefore lowers the first echo and dephases later echoes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.motion import (  # noqa: E402
    ParticleEnsemble,
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences.motion import run_motion_cpmg_sequence  # noqa: E402


@dataclass(frozen=True)
class PipeFlowFields:
    x_axis: np.ndarray
    z_axis: np.ndarray
    b0_hz: np.ndarray
    b1_tx: np.ndarray
    b1_rx: np.ndarray
    polarizer_profile: np.ndarray


@dataclass(frozen=True)
class FlowCaseResult:
    mean_velocity: float
    initial_polarization: float
    echo_times: np.ndarray
    echo_values: np.ndarray
    start_positions: np.ndarray
    end_positions: np.ndarray

    @property
    def echo_magnitudes(self) -> np.ndarray:
        return np.abs(self.echo_values)

    @property
    def normalized_decay(self) -> np.ndarray:
        denom = max(float(self.echo_magnitudes[0]), np.finfo(float).eps)
        return self.echo_magnitudes / denom


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe-radius-mm", type=float, default=2.0)
    parser.add_argument("--num-x", type=int, default=45, help="Transverse samples.")
    parser.add_argument("--num-z", type=int, default=9, help="Initial packet samples.")
    parser.add_argument("--walkers-per-cell", type=int, default=1)
    parser.add_argument("--num-echoes", type=int, default=14)
    parser.add_argument("--echo-spacing-ms", type=float, default=4.0)
    parser.add_argument("--excitation-us", type=float, default=120.0)
    parser.add_argument("--refocusing-us", type=float, default=240.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument(
        "--velocity",
        type=float,
        nargs="+",
        default=[0.005, 0.03, 0.10, 0.25],
        help="Mean pipe velocities in m/s.",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=1.2,
        help="Longitudinal build-up time in the polarizing magnet (s).",
    )
    parser.add_argument(
        "--t2",
        type=float,
        default=0.18,
        help="Intrinsic transverse decay during the CPMG train (s).",
    )
    parser.add_argument(
        "--polarizer-length-mm",
        type=float,
        default=80.0,
        help="Upstream polarizing magnet length.",
    )
    parser.add_argument(
        "--polarizer-center-mm",
        type=float,
        default=-58.0,
        help="Polarizing magnet center relative to the transmit coil.",
    )
    parser.add_argument(
        "--coil-width-mm",
        type=float,
        default=14.0,
        help="Transmit/receive coil Gaussian width.",
    )
    parser.add_argument(
        "--initial-packet-width-mm",
        type=float,
        default=5.0,
        help="Axial width of the packet at the first RF pulse.",
    )
    parser.add_argument(
        "--z-extent-mm",
        type=float,
        default=115.0,
        help="Half-width of the displayed/sampled axial field map.",
    )
    parser.add_argument(
        "--fringe-hz",
        type=float,
        default=260.0,
        help="Residual polarizer fringe offset scale in Hz.",
    )
    parser.add_argument(
        "--axial-gradient-hz-per-m",
        type=float,
        default=8000.0,
        help="Linear downstream B0 gradient near the coil.",
    )
    parser.add_argument(
        "--radial-spread-hz",
        type=float,
        default=55.0,
        help="Static radial B0 spread across the pipe.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG path.")
    return parser.parse_args()


def laminar_pipe_velocity(radius: float, mean_velocity: float):
    """Return a Poiseuille velocity callback for ``run_motion_sequence``."""

    radius = float(radius)
    mean_velocity = float(mean_velocity)
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if mean_velocity < 0.0:
        raise ValueError("mean_velocity must be non-negative")

    def velocity(positions: np.ndarray, _time: float) -> np.ndarray:
        x = np.asarray(positions, dtype=np.float64)[:, 0]
        profile = np.clip(1.0 - (x / radius) ** 2, 0.0, None)
        values = np.zeros_like(positions, dtype=np.float64)
        values[:, 1] = 2.0 * mean_velocity * profile
        return values

    return velocity


def make_pipe_boundary(radius: float):
    """Reflect at the pipe wall and clip only at the far axial map edges."""

    radius = float(radius)

    def boundary(
        positions: np.ndarray,
        *,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        **_: object,
    ) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float64).copy()
        period = 2.0 * radius
        folded = np.mod(pos[:, 0] + radius, 2.0 * period)
        pos[:, 0] = np.where(folded <= period, folded, 2.0 * period - folded) - radius
        if bounds is not None:
            pos[:, 1] = np.clip(pos[:, 1], bounds[1][0], bounds[1][1])
        return pos

    return boundary


def make_pipe_flow_fields(
    *,
    radius: float,
    z_extent: float,
    num_x: int,
    num_z_map: int,
    polarizer_center: float,
    polarizer_length: float,
    coil_width: float,
    fringe_hz: float,
    axial_gradient_hz_per_m: float,
    radial_spread_hz: float,
) -> PipeFlowFields:
    """Build static B0, transmit, receive, and polarizer maps."""

    x_axis = np.linspace(-radius, radius, int(num_x))
    z_axis = np.linspace(-z_extent, z_extent, int(num_z_map))
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    radius_norm = np.clip(np.abs(xx) / radius, 0.0, 1.0)

    polarizer_sigma = max(polarizer_length / 2.355, np.finfo(float).eps)
    polarizer_profile = np.exp(-0.5 * ((zz - polarizer_center) / polarizer_sigma) ** 2)
    detector_fringe = float(fringe_hz) * polarizer_profile
    detector_fringe -= float(fringe_hz) * np.exp(
        -0.5 * ((0.0 - polarizer_center) / polarizer_sigma) ** 2
    )
    b0_hz = (
        detector_fringe
        + float(axial_gradient_hz_per_m) * zz
        + float(radial_spread_hz) * radius_norm**2
    )

    tx = np.exp(-0.5 * (zz / coil_width) ** 2) * (1.0 - 0.10 * radius_norm**2)
    rx_center = 0.35 * coil_width
    rx = np.exp(-0.5 * ((zz - rx_center) / (1.25 * coil_width)) ** 2)
    rx *= 1.0 - 0.18 * radius_norm**2
    return PipeFlowFields(
        x_axis=x_axis,
        z_axis=z_axis,
        b0_hz=b0_hz,
        b1_tx=np.clip(tx, 0.0, None),
        b1_rx=np.clip(rx, 0.0, None),
        polarizer_profile=polarizer_profile,
    )


def initialize_pipe_packet(
    *,
    radius: float,
    num_x: int,
    num_z: int,
    packet_width: float,
    walkers_per_cell: int,
    mean_velocity: float,
    polarizer_length: float,
    t1: float,
    seed: int,
) -> ParticleEnsemble:
    """Initialize a polarized axisymmetric pipe packet at the detector coil."""

    x_axis = np.linspace(-radius, radius, int(num_x))
    z_axis = np.linspace(-1.5 * packet_width, 1.5 * packet_width, int(num_z))
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    chord_weight = 2.0 * np.sqrt(np.clip(radius**2 - xx**2, 0.0, None))
    packet = np.exp(-0.5 * (zz / packet_width) ** 2)
    density = chord_weight * packet
    ensemble = initialize_ensemble_from_density(
        density,
        x_axis,
        z_axis,
        walkers_per_cell=int(walkers_per_cell),
        seed=seed,
        jitter=walkers_per_cell > 1,
    )

    x = ensemble.positions[:, 0]
    profile = np.clip(1.0 - (x / radius) ** 2, 0.0, None)
    local_velocity = 2.0 * float(mean_velocity) * profile
    residence = np.divide(
        polarizer_length,
        local_velocity,
        out=np.full_like(local_velocity, np.inf),
        where=local_velocity > 0.0,
    )
    polarization = 1.0 - np.exp(-residence / float(t1))
    mag = ensemble.magnetization.copy()
    mag[0, :] = polarization
    mag[1:, :] = 0.0
    return ensemble.with_updates(magnetization=mag)


def run_flow_case(
    *,
    fields: PipeFlowFields,
    radius: float,
    mean_velocity: float,
    num_x: int,
    num_z: int,
    packet_width: float,
    walkers_per_cell: int,
    polarizer_length: float,
    t1: float,
    t2: float,
    num_echoes: int,
    echo_spacing: float,
    excitation_duration: float,
    refocusing_duration: float,
    substeps: int,
    seed: int,
) -> FlowCaseResult:
    """Run one moving-isochromat CPMG flow case."""

    ensemble = initialize_pipe_packet(
        radius=radius,
        num_x=num_x,
        num_z=num_z,
        packet_width=packet_width,
        walkers_per_cell=walkers_per_cell,
        mean_velocity=mean_velocity,
        polarizer_length=polarizer_length,
        t1=t1,
        seed=seed,
    )
    initial_polarization = float(
        np.sum(ensemble.weights * np.real(ensemble.magnetization[0, :]))
        / np.sum(ensemble.weights)
    )
    maps = make_motion_field_maps_2d(
        fields.x_axis,
        fields.z_axis,
        b0_map=2.0 * np.pi * fields.b0_hz,
        b1_tx_map=fields.b1_tx,
        b1_rx_map=fields.b1_rx,
    )
    result = run_motion_cpmg_sequence(
        ensemble,
        maps,
        num_echoes=num_echoes,
        echo_spacing=echo_spacing,
        excitation_duration=excitation_duration,
        refocusing_duration=refocusing_duration,
        velocity=laminar_pipe_velocity(radius, mean_velocity),
        rng=np.random.default_rng(seed + 7919),
        t1=np.inf,
        t2=t2,
        boundary=make_pipe_boundary(radius),
        substeps_per_interval=substeps,
    )
    return FlowCaseResult(
        mean_velocity=float(mean_velocity),
        initial_polarization=initial_polarization,
        echo_times=result.sample_times,
        echo_values=result.signal,
        start_positions=ensemble.positions.copy(),
        end_positions=result.final_ensemble.positions.copy(),
    )


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "--pipe-radius-mm": args.pipe_radius_mm,
        "--num-x": args.num_x,
        "--num-z": args.num_z,
        "--walkers-per-cell": args.walkers_per_cell,
        "--num-echoes": args.num_echoes,
        "--echo-spacing-ms": args.echo_spacing_ms,
        "--excitation-us": args.excitation_us,
        "--refocusing-us": args.refocusing_us,
        "--substeps": args.substeps,
        "--t1": args.t1,
        "--t2": args.t2,
        "--polarizer-length-mm": args.polarizer_length_mm,
        "--coil-width-mm": args.coil_width_mm,
        "--initial-packet-width-mm": args.initial_packet_width_mm,
        "--z-extent-mm": args.z_extent_mm,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    if args.num_x < 3 or args.num_z < 2:
        raise SystemExit("--num-x must be at least 3 and --num-z at least 2")
    if any(v < 0.0 for v in args.velocity):
        raise SystemExit("--velocity values must be non-negative")
    if args.echo_spacing_ms * 1e-3 < args.refocusing_us * 1e-6:
        raise SystemExit("--echo-spacing-ms must be at least --refocusing-us")


def _plot_results(
    plt,
    args: argparse.Namespace,
    fields: PipeFlowFields,
    rows: list[FlowCaseResult],
) -> None:
    colors = plt.cm.plasma(np.linspace(0.12, 0.88, len(rows)))
    extent_mm = [
        fields.z_axis[0] * 1e3,
        fields.z_axis[-1] * 1e3,
        fields.x_axis[0] * 1e3,
        fields.x_axis[-1] * 1e3,
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    b0_img = axes[0, 0].imshow(
        fields.b0_hz,
        origin="lower",
        extent=extent_mm,
        aspect="auto",
        cmap="coolwarm",
    )
    fig.colorbar(b0_img, ax=axes[0, 0], label="B0 offset (Hz)")
    axes[0, 0].contour(
        fields.z_axis * 1e3,
        fields.x_axis * 1e3,
        fields.b1_tx,
        levels=[0.25, 0.5, 0.75],
        colors="white",
        linewidths=0.8,
    )
    axes[0, 0].plot(
        fields.z_axis * 1e3,
        fields.polarizer_profile[fields.x_axis.size // 2] * args.pipe_radius_mm,
        color="black",
        linewidth=1.5,
        label="scaled polarizer",
    )

    velocity_summary = []
    first_echoes = []
    last_over_first = []
    for row, color in zip(rows, colors):
        label = f"{row.mean_velocity * 1e2:g} cm/s"
        echo_numbers = np.arange(1, row.echo_magnitudes.size + 1)
        axes[0, 1].plot(
            echo_numbers,
            row.echo_magnitudes,
            marker="o",
            color=color,
            label=label,
        )
        axes[1, 0].plot(
            echo_numbers,
            row.normalized_decay,
            marker="o",
            color=color,
        )
        velocity_summary.append(row.mean_velocity)
        first_echoes.append(row.echo_magnitudes[0])
        last_over_first.append(row.normalized_decay[-1])

    velocity_summary = np.asarray(velocity_summary, dtype=np.float64)
    first_echoes = np.asarray(first_echoes, dtype=np.float64)
    last_over_first = np.asarray(last_over_first, dtype=np.float64)
    axes[1, 1].plot(
        velocity_summary * 1e2,
        first_echoes / max(first_echoes[0], np.finfo(float).eps),
        marker="o",
        label="first echo / slow case",
    )
    axes[1, 1].plot(
        velocity_summary * 1e2,
        last_over_first,
        marker="s",
        label="last echo / first echo",
    )
    axes[1, 1].plot(
        velocity_summary * 1e2,
        [row.initial_polarization for row in rows],
        marker="^",
        label="mean initial Mz",
    )

    fastest = rows[-1]
    stride = max(1, fastest.start_positions.shape[0] // 350)
    axes[0, 0].scatter(
        fastest.start_positions[::stride, 1] * 1e3,
        fastest.start_positions[::stride, 0] * 1e3,
        s=5,
        alpha=0.35,
        color="tab:green",
        label="packet start",
    )
    axes[0, 0].scatter(
        fastest.end_positions[::stride, 1] * 1e3,
        fastest.end_positions[::stride, 0] * 1e3,
        s=5,
        alpha=0.35,
        color="tab:orange",
        label="packet end",
    )
    axes[0, 0].legend(loc="upper right")

    axes[0, 0].set_title("Pipe, Polarizer Fringe, and Coil B1")
    axes[0, 0].set_xlabel("z downstream (mm)")
    axes[0, 0].set_ylabel("x across pipe (mm)")
    axes[0, 1].set_title("Absolute CPMG Echo Magnitudes")
    axes[0, 1].set_xlabel("echo number")
    axes[0, 1].set_ylabel("|echo|")
    axes[0, 1].legend(title="mean velocity")
    axes[1, 0].set_title("Motion-Induced Echo Decay")
    axes[1, 0].set_xlabel("echo number")
    axes[1, 0].set_ylabel("|echo| / first echo")
    axes[1, 1].set_title("Velocity Sweep Diagnostics")
    axes[1, 1].set_xlabel("mean velocity (cm/s)")
    axes[1, 1].set_ylabel("relative signal")
    axes[1, 1].set_ylim(bottom=0.0)
    axes[1, 1].legend(loc="best")

    fig.suptitle(
        f"Laminar Pipe Flow CPMG: R={args.pipe_radius_mm:g} mm, "
        f"TE={args.echo_spacing_ms:g} ms"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    plt = load_matplotlib()

    radius = args.pipe_radius_mm * 1e-3
    polarizer_length = args.polarizer_length_mm * 1e-3
    fields = make_pipe_flow_fields(
        radius=radius,
        z_extent=args.z_extent_mm * 1e-3,
        num_x=max(args.num_x, 31),
        num_z_map=241,
        polarizer_center=args.polarizer_center_mm * 1e-3,
        polarizer_length=polarizer_length,
        coil_width=args.coil_width_mm * 1e-3,
        fringe_hz=args.fringe_hz,
        axial_gradient_hz_per_m=args.axial_gradient_hz_per_m,
        radial_spread_hz=args.radial_spread_hz,
    )
    rows = [
        run_flow_case(
            fields=fields,
            radius=radius,
            mean_velocity=velocity,
            num_x=args.num_x,
            num_z=args.num_z,
            packet_width=args.initial_packet_width_mm * 1e-3,
            walkers_per_cell=args.walkers_per_cell,
            polarizer_length=polarizer_length,
            t1=args.t1,
            t2=args.t2,
            num_echoes=args.num_echoes,
            echo_spacing=args.echo_spacing_ms * 1e-3,
            excitation_duration=args.excitation_us * 1e-6,
            refocusing_duration=args.refocusing_us * 1e-6,
            substeps=args.substeps,
            seed=args.seed + idx * 1009,
        )
        for idx, velocity in enumerate(args.velocity)
    ]

    print("Laminar pipe-flow CPMG")
    print("mean_velocity_cm_s  initial_Mz  first_echo  last_over_first")
    for row in rows:
        print(
            f"{row.mean_velocity * 1e2:18.6g}  "
            f"{row.initial_polarization:10.6g}  "
            f"{row.echo_magnitudes[0]:10.6g}  "
            f"{row.normalized_decay[-1]:15.6g}"
        )

    _plot_results(plt, args, fields, rows)


if __name__ == "__main__":
    main()
