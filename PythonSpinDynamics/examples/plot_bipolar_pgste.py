"""Background-gradient suppression with the bipolar 13-interval PGSTE.

A constant background gradient -- for example the internal gradient from
magnetic-susceptibility contrast in porous media -- biases an ordinary
pulsed-gradient diffusion measurement through the cross-term between the applied
and background gradients. The Cotts 13-interval alternating PGSTE cancels that
cross-term, so its apparent diffusion coefficient is unbiased by the background.

This example shows two things. The left panel traces the toggling-frame
dephasing wavevectors for the 13-interval sequence: the applied wavevector is
parked during storage and refocuses to zero, and the continuously present
background wavevector is refocused by the 180 pulses. The right panel sweeps the
background gradient and plots the apparent diffusion coefficient recovered from
a b-value sweep, for the monopolar stimulated echo (biased) and the 13-interval
sequence (flat at the true value).

Run with ``--output bipolar_pgste.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8
DIFFUSION = 2.3e-9


@dataclass(frozen=True)
class BipolarSimulation:
    """Wavevector trajectories and apparent-diffusion suppression curves."""

    times: np.ndarray
    q_applied: np.ndarray
    q_background: np.ndarray
    background_gradients: np.ndarray
    apparent_thirteen: np.ndarray
    apparent_monopolar: np.ndarray
    diffusion: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show how the bipolar 13-interval PGSTE suppresses the apparent "
            "diffusion bias from a constant background gradient, compared with a "
            "monopolar stimulated echo."
        )
    )
    parser.add_argument(
        "--gradient-duration-ms",
        type=float,
        default=2.0,
        help="Diffusion-gradient lobe duration (ms).",
    )
    parser.add_argument(
        "--half-echo-time-ms",
        type=float,
        default=6.0,
        help="Spacing from a lobe start to its 180 pulse centre (ms).",
    )
    parser.add_argument(
        "--storage-time-ms",
        type=float,
        default=40.0,
        help="Stimulated-echo storage interval (ms).",
    )
    parser.add_argument(
        "--max-gradient",
        type=float,
        default=0.2,
        help="Maximum applied gradient in the b-value sweep (T/m).",
    )
    parser.add_argument(
        "--gradient-points",
        type=int,
        default=9,
        help="Number of applied gradients in the b-value sweep.",
    )
    parser.add_argument(
        "--max-background",
        type=float,
        default=0.06,
        help="Maximum background gradient swept (T/m).",
    )
    parser.add_argument(
        "--background-points",
        type=int,
        default=13,
        help="Number of background-gradient values.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _wavevector_trajectory(intervals):
    times = [0.0]
    q_applied = [0.0]
    q_background = [0.0]
    qa = 0.0
    qb = 0.0
    t = 0.0
    for interval in intervals:
        h = interval.duration
        if h <= 0.0:
            continue
        if interval.sign == 0:
            slope_a = 0.0
            slope_b = 0.0
        else:
            slope_a = GAMMA * interval.sign * interval.applied_gradient
            slope_b = GAMMA * interval.sign
        t += h
        qa += slope_a * h
        qb += slope_b * h
        times.append(t)
        q_applied.append(qa)
        q_background.append(qb)
    return np.array(times), np.array(q_applied), np.array(q_background)


def _apparent_diffusion(runner, *, background, gradients, diffusion, kwargs) -> float:
    b_values = []
    log_signal = []
    for g in gradients:
        result = runner(
            gradient_amplitude=float(g),
            diffusion_coefficient=diffusion,
            background_gradient=float(background),
            **kwargs,
        )
        b_values.append(result.b_value)
        log_signal.append(np.log(result.diffusion_attenuation))
    slope = np.polyfit(np.array(b_values), np.array(log_signal), 1)[0]
    return -float(slope)


def _simulate(args: argparse.Namespace) -> BipolarSimulation:
    from spin_dynamics.workflows.bipolar import (
        cotts_thirteen_interval_intervals,
        run_cotts_thirteen_interval_moment,
        run_monopolar_pgste_moment,
    )

    timing = dict(
        gradient_duration=float(args.gradient_duration_ms) * 1e-3,
        half_echo_time=float(args.half_echo_time_ms) * 1e-3,
        storage_time=float(args.storage_time_ms) * 1e-3,
    )

    intervals = cotts_thirteen_interval_intervals(gradient_amplitude=0.1, **timing)
    times, q_applied, q_background = _wavevector_trajectory(intervals)

    gradients = np.linspace(0.0, float(args.max_gradient), int(args.gradient_points))
    backgrounds = np.linspace(
        0.0, float(args.max_background), int(args.background_points)
    )
    apparent_thirteen = np.array(
        [
            _apparent_diffusion(
                run_cotts_thirteen_interval_moment,
                background=g0,
                gradients=gradients,
                diffusion=DIFFUSION,
                kwargs=timing,
            )
            for g0 in backgrounds
        ]
    )
    apparent_monopolar = np.array(
        [
            _apparent_diffusion(
                run_monopolar_pgste_moment,
                background=g0,
                gradients=gradients,
                diffusion=DIFFUSION,
                kwargs=timing,
            )
            for g0 in backgrounds
        ]
    )
    return BipolarSimulation(
        times=times,
        q_applied=q_applied,
        q_background=q_background,
        background_gradients=backgrounds,
        apparent_thirteen=apparent_thirteen,
        apparent_monopolar=apparent_monopolar,
        diffusion=DIFFUSION,
    )


def _plot_results(plt, sim: BipolarSimulation):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    axes[0].plot(
        sim.times * 1e3,
        sim.q_applied,
        marker="o",
        ms=3,
        label="applied wavevector",
        color="#2a7f3f",
    )
    axes[0].plot(
        sim.times * 1e3,
        sim.q_background,
        marker="s",
        ms=3,
        label="background wavevector",
        color="#b03060",
    )
    axes[0].axhline(0.0, color="gray", lw=0.6)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("toggling-frame q (rad/m)")
    axes[0].set_title("13-interval wavevectors")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        sim.background_gradients * 1e3,
        sim.apparent_monopolar / sim.diffusion,
        marker="o",
        ms=4,
        color="#b03060",
        label="monopolar PGSTE",
    )
    axes[1].plot(
        sim.background_gradients * 1e3,
        sim.apparent_thirteen / sim.diffusion,
        marker="o",
        ms=4,
        color="#2a7f3f",
        label="13-interval APGSTE",
    )
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("background gradient g0 (mT/m)")
    axes[1].set_ylabel("apparent D / true D")
    axes[1].set_title("Background-gradient suppression")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    if args.gradient_points < 2 or args.background_points < 1:
        raise SystemExit("need at least 2 gradient points and 1 background point")

    plt = load_matplotlib(headless=bool(args.output))
    sim = _simulate(args)

    from spin_dynamics.phase_cycling import diff_stebp_phase_cycle

    cycle = diff_stebp_phase_cycle()
    print(f"phase cycle: {cycle.name} ({cycle.num_steps} steps, Bruker diff_stebp)")
    print("apparent D / true D (background g0 sweep):")
    for g0, mono, thirteen in zip(
        sim.background_gradients, sim.apparent_monopolar, sim.apparent_thirteen
    ):
        print(
            f"  g0 = {g0 * 1e3:5.1f} mT/m -> monopolar {mono / sim.diffusion:5.3f}, "
            f"13-interval {thirteen / sim.diffusion:5.3f}"
        )

    fig = _plot_results(plt, sim)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
