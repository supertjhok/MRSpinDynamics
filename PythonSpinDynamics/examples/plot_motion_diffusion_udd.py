"""Compare CPMG and UDD random-walker signals with relaxation and diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.motion import (  # noqa: E402
    initialize_ensemble_from_density,
    make_motion_field_maps_2d,
)
from spin_dynamics.sequences import (  # noqa: E402
    cpmg_pulse_times,
    run_motion_cpmg_sequence,
    run_motion_udd_sequence,
    udd_pulse_times,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=500, help="Walker count.")
    parser.add_argument(
        "--pulses",
        type=int,
        default=8,
        help="Number of refocusing pi pulses in each sequence.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.12,
        help="Minimum post-excitation evolution window.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=0.72,
        help="Maximum post-excitation evolution window.",
    )
    parser.add_argument(
        "--duration-points",
        type=int,
        default=7,
        help="Number of total-duration samples.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Substeps per free-precession interval.",
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
        default=0.00025,
        help="Rectangular 90-degree pulse duration.",
    )
    parser.add_argument(
        "--refocusing-duration",
        type=float,
        default=0.0005,
        help="Rectangular 180-degree pulse duration.",
    )
    parser.add_argument(
        "--diffusion",
        type=float,
        nargs="+",
        default=[0.0, 0.0015, 0.0045],
        help="Diffusion coefficients in map-coordinate units squared per time.",
    )
    parser.add_argument("--t1", type=float, default=5.0, help="T1 relaxation time.")
    parser.add_argument("--t2", type=float, default=1.0, help="T2 relaxation time.")
    parser.add_argument(
        "--fluctuation-amplitude",
        type=float,
        default=1500.0,
        help="Sinusoidal x-gradient fluctuation amplitude.",
    )
    parser.add_argument(
        "--fluctuation-frequency",
        type=float,
        default=0.35,
        help="B0-gradient fluctuation frequency in cycles per time unit.",
    )
    parser.add_argument(
        "--fluctuation-phase",
        type=float,
        default=0.0,
        help="B0-gradient fluctuation phase in radians.",
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


def _run_case(
    args: argparse.Namespace,
    fields,
    *,
    sequence: str,
    diffusion: float,
    duration: float,
    case_index: int,
):
    seed = args.seed + 1009 * case_index
    ensemble = _initialize_walkers(args.num_particles, diffusion, seed)
    rng = np.random.default_rng(seed)
    detuning = lambda time, positions: (
        args.fluctuation_amplitude
        * positions[:, 0]
        * np.cos(
            2.0 * np.pi * args.fluctuation_frequency * time
            + args.fluctuation_phase
        )
    )
    common = dict(
        ensemble=ensemble,
        fields=fields,
        excitation_duration=args.excitation_duration,
        refocusing_duration=args.refocusing_duration,
        gradient=(args.gradient, 0.0),
        rng=rng,
        t1=args.t1,
        t2=args.t2,
        detuning_waveform=detuning,
        substeps_per_interval=args.substeps,
    )
    if sequence == "UDD":
        result = run_motion_udd_sequence(
            num_pulses=args.pulses,
            total_duration=duration,
            **common,
        )
    elif sequence == "CPMG":
        result = run_motion_cpmg_sequence(
            num_echoes=args.pulses,
            echo_spacing=duration / args.pulses,
            **common,
        )
    else:
        raise ValueError("sequence must be 'CPMG' or 'UDD'")
    return result.signal[-1], result.sample_times[-1], result.final_ensemble.positions


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_particles <= 0:
        raise SystemExit("--num-particles must be positive")
    if args.pulses <= 0:
        raise SystemExit("--pulses must be positive")
    if args.duration_points < 2:
        raise SystemExit("--duration-points must be at least 2")
    if args.min_duration <= 0.0 or args.max_duration <= args.min_duration:
        raise SystemExit("--max-duration must be greater than positive --min-duration")
    if args.substeps <= 0:
        raise SystemExit("--substeps must be positive")
    if args.excitation_duration <= 0.0 or args.refocusing_duration <= 0.0:
        raise SystemExit(
            "--excitation-duration and --refocusing-duration must be positive"
        )
    if args.min_duration / args.pulses < args.refocusing_duration:
        raise SystemExit(
            "--min-duration / --pulses must be at least --refocusing-duration"
        )
    if any(value < 0.0 for value in args.diffusion):
        raise SystemExit("--diffusion values must be non-negative")
    if args.t1 <= 0.0 and not np.isinf(args.t1):
        raise SystemExit("--t1 must be positive or inf")
    if args.t2 <= 0.0 and not np.isinf(args.t2):
        raise SystemExit("--t2 must be positive or inf")
    if args.fluctuation_frequency < 0.0:
        raise SystemExit("--fluctuation-frequency must be non-negative")


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    plt = load_matplotlib()
    fields = _make_fields()
    durations = np.linspace(args.min_duration, args.max_duration, args.duration_points)
    sequences = ("CPMG", "UDD")
    rows = []
    final_positions = None

    for diffusion_index, diffusion in enumerate(args.diffusion):
        for sequence_index, sequence in enumerate(sequences):
            signals = []
            echo_times = []
            for duration_index, duration in enumerate(durations):
                case_index = (
                    10_000 * diffusion_index
                    + 100 * sequence_index
                    + duration_index
                )
                signal, echo_time, positions = _run_case(
                    args,
                    fields,
                    sequence=sequence,
                    diffusion=diffusion,
                    duration=float(duration),
                    case_index=case_index,
                )
                signals.append(signal)
                echo_times.append(echo_time)
                if diffusion_index == len(args.diffusion) - 1 and sequence == "UDD":
                    final_positions = positions
            rows.append(
                {
                    "sequence": sequence,
                    "diffusion": float(diffusion),
                    "diffusion_index": diffusion_index,
                    "signals": np.asarray(signals, dtype=np.complex128),
                    "echo_times": np.asarray(echo_times, dtype=np.float64),
                }
            )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(args.diffusion)))
    linestyles = {"CPMG": "-", "UDD": "--"}

    for row in rows:
        color = colors[row["diffusion_index"]]
        magnitude = np.abs(row["signals"])
        magnitude /= max(magnitude[0], np.finfo(float).eps)
        label = f"{row['sequence']}, D={row['diffusion']:g}"
        axes[0, 0].plot(
            durations,
            magnitude,
            linestyle=linestyles[row["sequence"]],
            marker="o",
            color=color,
            label=label,
        )
        axes[0, 1].plot(
            durations,
            np.unwrap(np.angle(row["signals"])),
            linestyle=linestyles[row["sequence"]],
            marker="o",
            color=color,
        )

    cpmg_times = cpmg_pulse_times(args.pulses, args.max_duration)
    udd_times = udd_pulse_times(args.pulses, args.max_duration)
    axes[1, 0].vlines(cpmg_times / args.max_duration, 0.0, 0.82, color="tab:blue")
    axes[1, 0].vlines(udd_times / args.max_duration, 0.18, 1.0, color="tab:orange")

    if final_positions is not None:
        stride = max(1, final_positions.shape[0] // 600)
        axes[1, 1].scatter(
            final_positions[::stride, 0],
            final_positions[::stride, 1],
            s=6,
            alpha=0.35,
            color="tab:green",
        )

    axes[0, 0].set_title("Final Signal Attenuation")
    axes[0, 0].set_xlabel("post-excitation duration")
    axes[0, 0].set_ylabel("normalized |signal|")
    axes[0, 0].legend(fontsize="small")

    axes[0, 1].set_title("Final Signal Phase")
    axes[0, 1].set_xlabel("post-excitation duration")
    axes[0, 1].set_ylabel("unwrapped phase")

    axes[1, 0].set_title(f"Pulse Timing at T={args.max_duration:g}")
    axes[1, 0].set_xlabel("time / T")
    axes[1, 0].set_yticks([])
    axes[1, 0].set_xlim(0.0, 1.0)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].text(0.02, 0.74, "CPMG", color="tab:blue")
    axes[1, 0].text(0.02, 0.92, "UDD", color="tab:orange")

    axes[1, 1].set_title(f"Final UDD Walker Cloud, D={args.diffusion[-1]:g}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("z")
    axes[1, 1].set_aspect("equal", adjustable="box")

    fig.suptitle(
        f"Random-Walker CPMG vs UDD ({args.pulses} pi pulses, "
        f"T1={args.t1:g}, T2={args.t2:g}, "
        f"delta={args.fluctuation_amplitude:g})"
    )

    print("Random-walker CPMG vs UDD")
    print(f"pulses: {args.pulses}")
    print(f"durations: {durations[0]:.6g} ... {durations[-1]:.6g}")
    print(
        "fluctuation: "
        f"amplitude={args.fluctuation_amplitude:g}, "
        f"frequency={args.fluctuation_frequency:g}"
    )
    for row in rows:
        final = row["signals"][-1]
        print(
            f"{row['sequence']} D={row['diffusion']:g}: "
            f"final |signal|={abs(final):.12g}, phase={np.angle(final):.12g}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
