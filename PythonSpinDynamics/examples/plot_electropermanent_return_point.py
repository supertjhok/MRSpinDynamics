"""Plot return-point hysteresis and two-element EPM programming interaction."""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import (  # noqa: E402
    CoupledEPMProgrammer,
    ProgrammingPulse,
    RemanenceState,
    archived_igbt_pulse_cases,
    electropermanent_field,
    illustrative_alnico_return_point_model,
    neighbor_coupling_matrix,
    variable_field_nmr_rod,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an illustrative play-operator return-point model and a "
            "geometry-derived two-EPM neighbor-programming calculation."
        )
    )
    parser.add_argument(
        "--separation-mm",
        type=float,
        default=80.0,
        help="Center-to-center rod separation (default: 80 mm).",
    )
    parser.add_argument(
        "--pulse-voltage-v",
        type=float,
        default=100.0,
        help="Negative programming-pulse capacitor voltage (default: 100 V).",
    )
    parser.add_argument(
        "--pulse-duration-us",
        type=float,
        default=30.0,
        help="Programming gate duration (default: 30 us).",
    )
    parser.add_argument(
        "--neighbor-samples",
        type=int,
        default=17,
        help="Neighbor-remanence sweep samples (default: 17).",
    )
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _ramp_path(points: tuple[float, ...], samples_per_segment: int) -> np.ndarray:
    segments = []
    for start, stop in zip(points[:-1], points[1:]):
        segment = np.linspace(start, stop, samples_per_segment)
        segments.append(segment if not segments else segment[1:])
    return np.concatenate(segments)


def _validate(args: argparse.Namespace) -> None:
    if not np.isfinite(args.separation_mm) or args.separation_mm <= 30.0:
        raise ValueError("--separation-mm must exceed 30 mm")
    if not np.isfinite(args.pulse_voltage_v) or not 0.0 < args.pulse_voltage_v <= 650.0:
        raise ValueError("--pulse-voltage-v must lie in (0, 650]")
    if not np.isfinite(args.pulse_duration_us) or args.pulse_duration_us <= 0.0:
        raise ValueError("--pulse-duration-us must be positive")
    if args.neighbor_samples < 5:
        raise ValueError("--neighbor-samples must be at least five")


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    model = illustrative_alnico_return_point_model()

    major_h = _ramp_path((0.0, 400e3, -400e3, 400e3), 301)
    major_initial = RemanenceState(-0.33, branch="negative_saturation")
    major = model.propagate(major_initial, major_h)

    minor_h = _ramp_path(
        (0.0, -120e3, 60e3, -120e3, 140e3, -200e3, 140e3),
        151,
    )
    minor_initial = RemanenceState(0.33, branch="positive_saturation")
    minor = model.propagate(minor_initial, minor_h)
    first_return_index = 150
    second_return_index = 3 * 150
    closure_error = abs(
        minor.remanence_t[first_return_index]
        - minor.remanence_t[second_return_index]
    )

    half_separation = 0.5e-3 * args.separation_mm
    target = variable_field_nmr_rod(
        center_m=(-half_separation, 0.0, 0.0),
        effective_remanence_t=0.33,
    )
    positive_neighbor = variable_field_nmr_rod(
        center_m=(half_separation, 0.0, 0.0),
        effective_remanence_t=0.33,
    )
    coupling = neighbor_coupling_matrix(
        (target, positive_neighbor),
        n_cross=5,
        n_length=31,
    )
    base_driver = archived_igbt_pulse_cases()[0].driver
    driver = replace(
        base_driver,
        state_inductance_fraction_at_saturation=0.25,
        label="illustrative state-dependent L",
    )
    pulse = ProgrammingPulse(
        args.pulse_voltage_v,
        args.pulse_duration_us * 1e-6,
        polarity=-1,
        label="partial negative programming",
    )
    times = np.linspace(0.0, max(180e-6, pulse.duration_s + 120e-6), 701)

    neighbor_states = np.linspace(-0.33, 0.33, args.neighbor_samples)
    final_target = np.empty_like(neighbor_states)
    sweep_results = []
    for index, neighbor_br in enumerate(neighbor_states):
        branch = (
            "positive_saturation"
            if neighbor_br == 0.33
            else "negative_saturation"
            if neighbor_br == -0.33
            else "partial"
        )
        neighbor = positive_neighbor.with_state(
            RemanenceState(float(neighbor_br), branch=branch)
        )
        system = CoupledEPMProgrammer(
            sources=(target, neighbor),
            drivers=(driver, driver),
            hysteresis_models=(model, model),
            coupling_a_per_m_per_t=coupling,
        )
        result = system.program(
            0,
            times,
            pulse,
            relaxation=0.8,
            tolerance_t=2e-6,
            max_step_s=pulse.duration_s / 50.0,
        )
        final_target[index] = result.final_states[0].remanence.remanence_t
        sweep_results.append(result)

    negative_result = sweep_results[0]
    zero_result = sweep_results[int(np.argmin(np.abs(neighbor_states)))]
    positive_result = sweep_results[-1]

    x_profile = np.linspace(-0.085, 0.085, 241)
    profile_gap_m = 10e-3
    z_face = 0.5 * target.length_m + profile_gap_m
    profile_points = np.column_stack(
        (x_profile, np.zeros_like(x_profile), np.full_like(x_profile, z_face))
    )
    profile_negative = electropermanent_field(
        profile_points,
        negative_result.final_sources,
        n_cross=5,
        n_length=31,
    )[:, 2]
    profile_positive = electropermanent_field(
        profile_points,
        positive_result.final_sources,
        n_cross=5,
        n_length=31,
    )[:, 2]

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.4), constrained_layout=True)
    fig.suptitle(
        "Electropermanent return-point memory and neighbor-coupled programming",
        fontsize=15,
    )

    ax = axes[0, 0]
    ax.plot(major_h / 1e3, major.remanence_t, color="tab:purple")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.axvline(0.0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="applied H (kA/m)",
        ylabel="retained coordinate Br (T)",
        title="illustrative weighted-play major loop",
    )
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(minor_h / 1e3, minor.remanence_t, color="tab:blue")
    ax.scatter(
        [minor_h[first_return_index] / 1e3, minor_h[second_return_index] / 1e3],
        [minor.remanence_t[first_return_index], minor.remanence_t[second_return_index]],
        color=("tab:orange", "black"),
        zorder=4,
        label="closed return point",
    )
    ax.set(
        xlabel="applied H (kA/m)",
        ylabel="retained coordinate Br (T)",
        title=f"nested minor loops; closure error {closure_error:.1e} T",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    radius_mm = target.radius_m * 1e3
    for center_x, color, label in (
        (-half_separation * 1e3, "tab:blue", "programmed target"),
        (half_separation * 1e3, "tab:orange", "saturated neighbor"),
    ):
        ax.add_patch(Circle((center_x, 0.0), radius_mm, color=color, alpha=0.75))
        ax.annotate(
            "Br along +z",
            xy=(center_x, 0.0),
            xytext=(center_x, 1.8 * radius_mm),
            ha="center",
            arrowprops=dict(arrowstyle="->", color=color),
            fontsize=8,
        )
    ax.plot([], [], color="tab:blue", linewidth=8, label="target")
    ax.plot([], [], color="tab:orange", linewidth=8, label="neighbor")
    ax.set_aspect("equal")
    ax.set(
        xlim=(-half_separation * 1e3 - 2.2 * radius_mm, half_separation * 1e3 + 2.2 * radius_mm),
        ylim=(-2.2 * radius_mm, 2.8 * radius_mm),
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"two-rod coupling K01={coupling[0, 1] / 1e3:.1f} kA/m/T",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    for label, result, color in (
        ("neighbor Br=-0.33 T", negative_result, "tab:blue"),
        ("neighbor Br=0", zero_result, "0.35"),
        ("neighbor Br=+0.33 T", positive_result, "tab:red"),
    ):
        ax.plot(
            result.waveform.times_s * 1e6,
            result.waveform.internal_h_a_per_m / 1e3,
            color=color,
            label=label,
        )
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="time (µs)",
        ylabel="target internal H (kA/m)",
        title="same command, different static neighbor bias",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(neighbor_states, final_target, marker="o", color="tab:green")
    ax.axvline(0.0, color="0.5", linewidth=0.8)
    ax.set(
        xlabel="neighbor retained Br (T)",
        ylabel="target final Br (T)",
        title="neighbor-dependent programming outcome",
    )
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.plot(x_profile * 1e3, profile_negative * 1e3, label="neighbor initially -0.33 T")
    ax.plot(x_profile * 1e3, profile_positive * 1e3, label="neighbor initially +0.33 T")
    ax.axvline(-half_separation * 1e3, color="tab:blue", linestyle=":")
    ax.axvline(half_separation * 1e3, color="tab:orange", linestyle=":")
    ax.set(
        xlabel=f"x at {profile_gap_m * 1e3:.0f} mm beyond rod faces (mm)",
        ylabel="combined Bz (mT)",
        title="static field consequence after programming",
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    print("Electropermanent return-point and neighbor-coupling model")
    print(
        f"  play model: {len(model.thresholds_a_per_m)} operators, "
        f"Br,sat={model.saturation_remanence_t:.3f} T, evidence={model.evidence[0].classification}"
    )
    print(f"  nested return-point closure error: {closure_error:.3e} T")
    print(
        f"  separation={args.separation_mm:.1f} mm, "
        f"K01={coupling[0, 1] / 1e3:.2f} kA/m/T"
    )
    for label, result in (
        ("negative neighbor", negative_result),
        ("zero neighbor", zero_result),
        ("positive neighbor", positive_result),
    ):
        print(
            f"  {label}: bias={result.neighbor_bias_a_per_m / 1e3:+.2f} kA/m, "
            f"target Br={result.final_states[0].remanence.remanence_t:+.3f} T, "
            f"iterations={result.iterations}, residual={result.residual_t:.2e} T"
        )
    print("  calibration status: illustrative; no measured minor-loop fit is claimed")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
