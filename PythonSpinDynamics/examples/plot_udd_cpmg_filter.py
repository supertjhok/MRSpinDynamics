"""Compare UDD and CPMG rejection of low-frequency detuning noise."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.sequences import (  # noqa: E402
    cpmg_pulse_times,
    dephasing_filter_function,
    toggling_frame_integral,
    udd_pulse_times,
)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    cumulative = np.zeros_like(y, dtype=np.float64)
    if y.size > 1:
        dx = np.diff(x)
        cumulative[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return cumulative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pulses",
        type=int,
        default=8,
        help="Number of refocusing pi pulses in each sequence.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Total evolution window in seconds.",
    )
    parser.add_argument(
        "--min-omega-t",
        type=float,
        default=1e-3,
        help="Minimum normalized angular frequency omega*T.",
    )
    parser.add_argument(
        "--max-omega-t",
        type=float,
        default=20.0,
        help="Maximum normalized angular frequency omega*T.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=900,
        help="Number of frequency samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    args = parser.parse_args()

    if args.pulses < 0:
        raise SystemExit("--pulses must be non-negative")
    if args.duration <= 0.0:
        raise SystemExit("--duration must be positive")
    if args.min_omega_t <= 0.0 or args.max_omega_t <= args.min_omega_t:
        raise SystemExit("--max-omega-t must be greater than positive --min-omega-t")
    if args.points < 2:
        raise SystemExit("--points must be at least 2")

    plt = load_matplotlib(headless=args.output is not None)

    cpmg_times = cpmg_pulse_times(args.pulses, args.duration)
    udd_times = udd_pulse_times(args.pulses, args.duration)
    omega_t = np.logspace(
        np.log10(args.min_omega_t),
        np.log10(args.max_omega_t),
        args.points,
        dtype=np.float64,
    )
    omega = omega_t / args.duration

    cpmg_response = np.abs(
        toggling_frame_integral(omega, cpmg_times, args.duration)
    ) / args.duration
    udd_response = np.abs(
        toggling_frame_integral(omega, udd_times, args.duration)
    ) / args.duration
    cpmg_filter = dephasing_filter_function(omega, cpmg_times, args.duration)
    udd_filter = dephasing_filter_function(omega, udd_times, args.duration)
    cpmg_cumulative = _cumulative_trapezoid(cpmg_response**2, omega_t)
    udd_cumulative = _cumulative_trapezoid(udd_response**2, omega_t)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    axes[0].vlines(
        cpmg_times / args.duration,
        0.0,
        0.82,
        color="tab:blue",
        label="CPMG",
    )
    axes[0].vlines(
        udd_times / args.duration,
        0.18,
        1.0,
        color="tab:orange",
        label="UDD",
    )
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_yticks([])
    axes[0].set_xlabel("time / T")
    axes[0].set_title("Pulse Placement")
    axes[0].legend(loc="upper center")

    axes[1].loglog(omega_t, cpmg_response, label="CPMG", color="tab:blue")
    axes[1].loglog(omega_t, udd_response, label="UDD", color="tab:orange")
    axes[1].set_xlabel("omega*T")
    axes[1].set_ylabel("|phase response| / T")
    axes[1].set_title("Sinusoidal Detuning Response")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    axes[2].loglog(omega_t, cpmg_cumulative, label="CPMG", color="tab:blue")
    axes[2].loglog(omega_t, udd_cumulative, label="UDD", color="tab:orange")
    axes[2].set_xlabel("upper band edge omega*T")
    axes[2].set_ylabel("cumulative low-frequency pickup")
    axes[2].set_title("Integrated Low-Frequency Noise")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend()

    fig.suptitle(
        f"UDD vs CPMG Filter Functions ({args.pulses} pi pulses, T={args.duration:g} s)"
    )

    low_index = 0
    print("UDD vs CPMG low-frequency filter comparison")
    print(f"pulses: {args.pulses}")
    print(f"lowest omega*T: {omega_t[low_index]:.6g}")
    print(f"CPMG filter: {cpmg_filter[low_index]:.12g}")
    print(f"UDD filter: {udd_filter[low_index]:.12g}")
    if cpmg_filter[low_index] > 0.0:
        ratio = udd_filter[low_index] / cpmg_filter[low_index]
        print(f"UDD/CPMG filter ratio: {ratio:.12g}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
