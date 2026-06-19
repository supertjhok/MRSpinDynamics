"""Plot circuit-model radiation damping for several probe detunings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import run_radiation_damping_fid




def _instantaneous_frequency(time: np.ndarray, signal: np.ndarray) -> np.ndarray:
    phase = np.unwrap(np.angle(signal))
    return np.gradient(phase, time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=["tuned", "matched"], default="matched")
    parser.add_argument("--fill-factor", type=float, default=0.7)
    parser.add_argument("--field-tesla", type=float, default=1.0)
    parser.add_argument("--polarization-scale", type=float, default=250.0)
    parser.add_argument("--flip-angle", type=float, default=np.pi / 3)
    parser.add_argument("--max-detuning", type=float, default=2.0e4)
    parser.add_argument("--num-detunings", type=int, default=5)
    parser.add_argument("--points", type=int, default=700)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_detunings < 2:
        raise SystemExit("--num-detunings must be at least 2")

    plt = load_matplotlib()
    detunings = np.linspace(-args.max_detuning, args.max_detuning, args.num_detunings)
    reference = run_radiation_damping_fid(
        probe=args.probe,
        fill_factor=args.fill_factor,
        field_tesla=args.field_tesla,
        polarization_scale=args.polarization_scale,
        flip_angle=args.flip_angle,
        detuning=0.0,
        model="circuit",
        num_points=args.points,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), constrained_layout=True)
    for detuning in detunings:
        result = run_radiation_damping_fid(
            probe=args.probe,
            fill_factor=args.fill_factor,
            field_tesla=args.field_tesla,
            polarization_scale=args.polarization_scale,
            flip_angle=args.flip_angle,
            detuning=float(detuning),
            duration_seconds=reference.time_seconds[-1],
            model="circuit",
            num_points=args.points,
        )
        time_trd = result.time_seconds / result.probe.trd
        label = f"{detuning / 1e3:.1f} krad/s"
        axes[0].plot(time_trd, result.envelope, label=label)
        axes[1].plot(
            time_trd,
            _instantaneous_frequency(result.time_seconds, result.mxy) / 1e3,
            label=label,
        )

    axes[0].set_title("Circuit Radiation-Damping Envelope")
    axes[0].set_xlabel("Time / Trd")
    axes[0].set_ylabel("|mxy|")
    axes[0].legend(title="detuning")
    axes[1].set_title("Rotating-Frame Frequency Pulling")
    axes[1].set_xlabel("Time / Trd")
    axes[1].set_ylabel("d phase / dt (krad/s)")
    axes[1].legend(title="detuning")
    fig.suptitle(
        f"{args.probe.capitalize()} probe, Q={reference.probe.q:.0f}, "
        f"tau={reference.probe.resonator_time_constant:.3g} s"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
