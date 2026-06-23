"""Compare CPMG echo decays with diffusion and absolute-phase increments.

The four cases share the same matched-probe CPMG diffusion workflow and toggle
two effects independently: free diffusion in the constant background gradient,
and echo-to-echo absolute RF phase advance for the refocusing pulses.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.workflows import MatchedDiffusionCPMGResult  # noqa: E402
from spin_dynamics.workflows import run_matched_diffusion_cpmg  # noqa: E402


def _phase_aligned_spec(
    *,
    phase_step_cycles: float,
    echo_spacing_seconds: float,
    diffusion_time_seconds: float,
    t90_seconds: float,
    phase_bins: int | None,
    initial_refocus_phase_rad: float,
) -> dict[str, float | int | None]:
    phase_step = 1.0 if phase_step_cycles == 0.0 else float(phase_step_cycles)
    rf_frequency_hz = phase_step / float(echo_spacing_seconds)
    first_refocus = _first_cpmg_refocus_start_seconds(
        echo_spacing_seconds=echo_spacing_seconds,
        diffusion_time_seconds=diffusion_time_seconds,
        t90_seconds=t90_seconds,
    )
    return {
        "rf_frequency_hz": rf_frequency_hz,
        "rf_phase_at_zero_rad": (
            float(initial_refocus_phase_rad)
            - 2.0 * np.pi * rf_frequency_hz * first_refocus
        ),
        "phase_bins": phase_bins,
    }


def _first_cpmg_refocus_start_seconds(
    *,
    echo_spacing_seconds: float,
    diffusion_time_seconds: float,
    t90_seconds: float,
) -> float:
    t180 = 2.0 * float(t90_seconds)
    encoding_gap = (np.pi / 2.0) * (
        float(diffusion_time_seconds) - 0.5 * float(t90_seconds) - 0.5 * t180
    ) / float(t90_seconds)
    tfp = (np.pi / 2.0) * (
        float(echo_spacing_seconds) - t180
    ) / (2.0 * float(t90_seconds))
    time_scale = 2.0 * float(t90_seconds) / np.pi
    encoding_start = float(t90_seconds) - time_scale + encoding_gap * time_scale
    return encoding_start + t180 + encoding_gap * time_scale + tfp * time_scale


def _run_case(
    *,
    numpts: int,
    num_echoes: int,
    q_value: float,
    echo_spacing_seconds: float,
    diffusion_time_seconds: float,
    diffusion_coefficient: float,
    t90_seconds: float,
    dz: float,
    absolute_phase: dict[str, float | int | None] | None,
    num_workers: int | None,
    auto_refine_grid: bool,
    rephase_action: str,
) -> MatchedDiffusionCPMGResult:
    return run_matched_diffusion_cpmg(
        numpts=int(numpts),
        num_echoes=int(num_echoes),
        q_value=float(q_value),
        echo_spacing_seconds=float(echo_spacing_seconds),
        diffusion_time=float(diffusion_time_seconds),
        diffusion_coefficient=float(diffusion_coefficient),
        t90_seconds=float(t90_seconds),
        dz=float(dz),
        num_workers=num_workers,
        auto_refine_grid=auto_refine_grid,
        rephase_action=rephase_action,
        q_stability_action="warn",
        absolute_phase=absolute_phase,
    )


def _normalized_integrals(
    result: MatchedDiffusionCPMGResult,
    reference: MatchedDiffusionCPMGResult,
) -> np.ndarray:
    scale = float(np.abs(reference.echo_integrals[0]))
    return np.abs(result.echo_integrals) / scale


def _relative_delta(
    result: MatchedDiffusionCPMGResult,
    reference: MatchedDiffusionCPMGResult,
) -> np.ndarray:
    denominator = np.maximum(np.abs(reference.echo_integrals), np.finfo(float).tiny)
    return np.abs(result.echo_integrals - reference.echo_integrals) / denominator


def _case_table(args: argparse.Namespace) -> list[tuple[str, str, MatchedDiffusionCPMGResult]]:
    echo_spacing = args.echo_spacing_us * 1.0e-6
    diffusion_time = args.diffusion_time_us * 1.0e-6
    t90 = args.t90_us * 1.0e-6
    phase_spec = _phase_aligned_spec(
        phase_step_cycles=args.phase_step,
        echo_spacing_seconds=echo_spacing,
        diffusion_time_seconds=diffusion_time,
        t90_seconds=t90,
        phase_bins=args.phase_bins,
        initial_refocus_phase_rad=args.initial_phase,
    )
    common = {
        "numpts": args.numpts,
        "num_echoes": args.num_echoes,
        "q_value": args.q_value,
        "echo_spacing_seconds": echo_spacing,
        "diffusion_time_seconds": diffusion_time,
        "t90_seconds": t90,
        "dz": args.dz_um * 1.0e-6,
        "num_workers": None if args.workers is None else int(args.workers),
        "auto_refine_grid": bool(args.auto_refine_grid),
        "rephase_action": args.rephase_action,
    }
    return [
        (
            "No diffusion, synchronized RF",
            "0.30",
            _run_case(
                **common,
                diffusion_coefficient=0.0,
                absolute_phase=None,
            ),
        ),
        (
            "Diffusion only",
            "C0",
            _run_case(
                **common,
                diffusion_coefficient=args.diffusion_coefficient,
                absolute_phase=None,
            ),
        ),
        (
            "Absolute phase only",
            "C1",
            _run_case(
                **common,
                diffusion_coefficient=0.0,
                absolute_phase=phase_spec,
            ),
        ),
        (
            "Diffusion + absolute phase",
            "C3",
            _run_case(
                **common,
                diffusion_coefficient=args.diffusion_coefficient,
                absolute_phase=phase_spec,
            ),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numpts", type=int, default=65, help="Offset grid size.")
    parser.add_argument("--num-echoes", type=int, default=8, help="Number of echoes.")
    parser.add_argument("--q-value", type=float, default=50.0, help="Matched-probe Q.")
    parser.add_argument(
        "--diffusion-coefficient",
        type=float,
        default=2.0e-8,
        help="Diffusion coefficient for diffusion-on cases, in SI units.",
    )
    parser.add_argument(
        "--echo-spacing-us",
        type=float,
        default=1000.0,
        help="CPMG echo spacing in microseconds.",
    )
    parser.add_argument(
        "--diffusion-time-us",
        type=float,
        default=1000.0,
        help="Diffusion encoding block duration in microseconds.",
    )
    parser.add_argument("--t90-us", type=float, default=100.0, help="90-degree pulse length.")
    parser.add_argument(
        "--dz-um",
        type=float,
        default=50.0,
        help="Slice thickness used to set the normalized offset span, in micrometers.",
    )
    parser.add_argument(
        "--phase-step",
        type=float,
        default=0.25,
        help="Absolute RF phase advance per CPMG echo, in cycles.",
    )
    parser.add_argument(
        "--phase-bins",
        type=int,
        default=16,
        help="Optional absolute-phase bins for pulse-shape reuse.",
    )
    parser.add_argument(
        "--initial-phase",
        type=float,
        default=0.0,
        help="First CPMG refocusing absolute phase, in radians.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Isochromat workers.")
    parser.add_argument(
        "--no-auto-refine-grid",
        dest="auto_refine_grid",
        action="store_false",
        help="Keep the requested numpts even when the offset grid may rephase.",
    )
    parser.add_argument(
        "--rephase-action",
        choices=["warn", "ignore", "raise"],
        default="raise",
        help="Action if auto-refinement is disabled and the grid may rephase.",
    )
    parser.add_argument(
        "--phase-effect-tolerance",
        type=float,
        default=1.0e-3,
        help="Warn when the matched-probe absolute-phase residual is below this fraction.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path.")
    parser.set_defaults(auto_refine_grid=True)
    args = parser.parse_args()
    if args.diffusion_coefficient < 0:
        raise SystemExit("--diffusion-coefficient must be non-negative")
    if args.dz_um <= 0:
        raise SystemExit("--dz-um must be positive")

    plt = load_matplotlib(headless=args.output is not None)
    cases = _case_table(args)
    reference = cases[0][2]
    print(f"effective num offsets: {reference.del_w.size}")
    phase_only = cases[2][2]
    diffusion_only = cases[1][2]
    combined = cases[3][2]
    phase_delta = _relative_delta(phase_only, reference)
    combined_phase_delta = _relative_delta(combined, diffusion_only)
    max_phase_delta = float(max(np.max(phase_delta), np.max(combined_phase_delta)))
    print(f"max absolute-phase residual: {max_phase_delta:.6g}")
    if max_phase_delta < args.phase_effect_tolerance:
        warnings.warn(
            "The matched-probe pulse-shape model is nearly invariant to absolute "
            "RF phase for these settings; the absolute-phase curves will overlap. "
            "Use the Mandal 2015 tuned/untuned examples to inspect the stronger "
            "probe-solved phase sensitivity.",
            RuntimeWarning,
            stacklevel=2,
        )
    echo_numbers = np.arange(1, args.num_echoes + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for label, color, result in cases:
        norm = _normalized_integrals(result, reference)
        axes[0, 0].plot(echo_numbers, norm, marker="o", label=label, color=color)
        axes[0, 1].semilogy(
            echo_numbers,
            np.maximum(norm, np.finfo(float).tiny),
            marker="o",
            label=label,
            color=color,
        )
    axes[1, 0].semilogy(
        echo_numbers,
        np.maximum(phase_delta, np.finfo(float).tiny),
        marker="o",
        color="C1",
        label="absolute phase only vs synchronized",
    )
    axes[1, 0].semilogy(
        echo_numbers,
        np.maximum(combined_phase_delta, np.finfo(float).tiny),
        marker="o",
        color="C3",
        label="combined vs diffusion only",
    )

    phase_case = cases[-1][2].absolute_phase
    if phase_case is not None:
        axes[1, 1].plot(
            echo_numbers,
            np.mod(phase_case.refocus_absolute_phase_rad / (2.0 * np.pi), 1.0),
            marker="o",
            label="CPMG refocusing",
        )
        if phase_case.encoding_absolute_phase_rad is not None:
            axes[1, 1].axhline(
                float(
                    np.mod(
                        phase_case.encoding_absolute_phase_rad[0] / (2.0 * np.pi),
                        1.0,
                    )
                ),
                color="0.35",
                linestyle="--",
                label="diffusion pi pulse",
            )

    axes[0, 0].set_title("Echo Integral Decay")
    axes[0, 0].set_xlabel("Echo number")
    axes[0, 0].set_ylabel("|integral| / baseline echo 1")
    axes[0, 0].legend(fontsize="small")

    axes[0, 1].set_title("Echo Integral Decay, Log Scale")
    axes[0, 1].set_xlabel("Echo number")
    axes[0, 1].set_ylabel("|integral| / baseline echo 1")
    axes[0, 1].legend(fontsize="small")

    axes[1, 0].set_title("Absolute-Phase Residual")
    axes[1, 0].set_xlabel("Echo number")
    axes[1, 0].set_ylabel("relative |delta integral|")
    axes[1, 0].legend(fontsize="small")

    axes[1, 1].set_title("Absolute RF Phase Schedule")
    axes[1, 1].set_xlabel("Echo number")
    axes[1, 1].set_ylabel("Cycles modulo 1")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend(fontsize="small")

    fig.suptitle(
        "Matched CPMG Echo Decays: Diffusion and Absolute-Phase Advance"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
