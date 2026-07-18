"""Closed-loop EPM particle imaging and magnetic-aggregate delivery."""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import illustrative_hybrid_epm_array  # noqa: E402
from spin_dynamics.workflows import (  # noqa: E402
    EPMTherapyControllerConfig,
    SuperparamagneticParticle,
    build_epm_nonlinear_encoding,
    magnetic_force_from_gradient,
    random_epm_encoding_states,
    run_epm_image_guided_controller,
    simple_tissue_phantom,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a particle cloud from nonlinear EPM images, program a "
            "transport field from that estimate, and verify delivery by imaging."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=14, help="Image/map size.")
    parser.add_argument("--encodings", type=int, default=256, help="Imaging states per cycle.")
    parser.add_argument("--cycles", type=int, default=4, help="Maximum controller cycles.")
    parser.add_argument(
        "--transport-min",
        type=float,
        default=24.0,
        help="Transport duration per cycle (default: 24 min).",
    )
    parser.add_argument(
        "--capture-goal",
        type=float,
        default=0.70,
        help="Early-stop capture fraction (default: 0.70).",
    )
    parser.add_argument("--particles", type=int, default=80, help="Aggregate count.")
    parser.add_argument("--seed", type=int, default=13, help="Imaging/motion seed.")
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.matrix_size < 10:
        raise ValueError("--matrix-size must be at least 10")
    if args.encodings < args.matrix_size**2:
        raise ValueError("--encodings must be at least the voxel count")
    if args.cycles < 1:
        raise ValueError("--cycles must be positive")
    if not np.isfinite(args.transport_min) or args.transport_min <= 0.0:
        raise ValueError("--transport-min must be positive")
    if not np.isfinite(args.capture_goal) or not 0.0 < args.capture_goal <= 1.0:
        raise ValueError("--capture-goal must be in (0, 1]")
    if args.particles < 10:
        raise ValueError("--particles must be at least 10")


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    rng = np.random.default_rng(args.seed)
    phantom = simple_tissue_phantom(args.matrix_size, field_of_view_m=0.040)
    array = illustrative_hybrid_epm_array(panel_gap_m=0.150)
    basis = array.build_field_basis(phantom.points_m, n_cross=1, n_length=7)
    states = random_epm_encoding_states(
        basis,
        args.encodings,
        amplitude_fraction=0.60,
        seed=args.seed,
    )
    encoding = build_epm_nonlinear_encoding(
        basis,
        states,
        image_shape=phantom.shape,
        phase_encoding_s=300e-6,
    )
    expected = phantom.spin_echo_image(repetition_time_s=1.2, echo_time_s=0.040)
    particle = SuperparamagneticParticle(
        magnetic_core_radius_m=12e-6,
        hydrodynamic_radius_m=15e-6,
        volume_susceptibility=1.4,
        saturation_magnetization_a_m=4.5e5,
        fluid_viscosity_pa_s=1.5e-3,
        temperature_k=310.0,
        magnetic_volume_fraction=0.60,
        label="illustrative 30 um hydrodynamic aggregate",
    )
    initial = np.column_stack(
        (
            rng.normal(-0.014, 0.35e-3, args.particles),
            rng.normal(-0.5e-3, 2.2e-3, args.particles),
        )
    )
    config = EPMTherapyControllerConfig(
        max_cycles=args.cycles,
        capture_goal=args.capture_goal,
        imaging_window_s=90.0,
        programming_window_s=0.25,
        transport_window_s=60.0 * args.transport_min,
        transport_time_step_s=5.0,
        verification_imaging_window_s=60.0,
        target_radius_m=4.2e-3,
        transport_bias_field_t=2.0e-3,
        transport_gradient_t_m=0.150,
        localization_threshold_fraction=0.90,
        imaging_snr_db=35.0,
        particle_imaging_snr_db=32.0,
        seed=args.seed + 1,
    )
    result = run_epm_image_guided_controller(
        encoding,
        expected,
        phantom.x_m,
        phantom.y_m,
        particle,
        initial,
        config=config,
        background_velocity_m_s=(2.5e-6, 0.0),
    )

    extent = 1e3 * np.asarray(
        (phantom.x_m[0], phantom.x_m[-1], phantom.y_m[0], phantom.y_m[-1])
    )
    x_grid, y_grid = np.meshgrid(phantom.x_m, phantom.y_m, indexing="xy")
    colors = plt.cm.viridis(np.linspace(0.12, 0.92, len(result.cycles)))
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    fig.suptitle(
        "Closed-loop particle-state estimation and magnetic transport",
        fontsize=15,
        y=0.99,
    )

    ax = axes[0, 0]
    last_cycle = result.cycles[-1]
    image = ax.imshow(
        last_cycle.imaging.reconstructed_image,
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    for cycle, color in zip(result.cycles, colors, strict=True):
        ax.plot(1e3 * cycle.target_center_m[0], 1e3 * cycle.target_center_m[1], "x", color=color, ms=8, mew=2)
    ax.contour(1e3 * phantom.x_m, 1e3 * phantom.y_m, phantom.target_mask, levels=(0.5,), colors="cyan", linewidths=1.4)
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="fresh target localization each cycle")
    fig.colorbar(image, ax=ax, label="last-cycle reconstructed signal")

    ax = axes[0, 1]
    particle_result = last_cycle.particle_imaging_after
    particle_image = ax.imshow(
        particle_result.estimate.concentration_image,
        origin="lower",
        extent=extent,
        cmap="inferno",
    )
    estimated_positions = particle_result.estimate.positions_m
    ax.scatter(
        1e3 * estimated_positions[:, 0],
        1e3 * estimated_positions[:, 1],
        marker="x",
        color="cyan",
        s=18,
        alpha=0.65,
        label="image-resolved estimate",
    )
    ax.scatter(
        1e3 * result.final_positions_m[:, 0],
        1e3 * result.final_positions_m[:, 1],
        facecolors="none",
        edgecolors="white",
        s=16,
        alpha=0.55,
        label="hidden truth (score only)",
    )
    ax.add_patch(
        Circle(
            1e3 * last_cycle.target_center_m,
            1e3 * config.target_radius_m,
            fill=False,
            color="lime",
            lw=1.8,
        )
    )
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="particle reconstruction and positions")
    ax.legend(fontsize=7, loc="upper left")
    fig.colorbar(particle_image, ax=ax, label="recovered particle signal")

    ax = axes[0, 2]
    force_map = last_cycle.force_map
    field_image = ax.imshow(1e3 * force_map.field_magnitude_t, origin="lower", extent=extent, cmap="viridis")
    force = magnetic_force_from_gradient(
        particle,
        force_map.field_magnitude_t,
        force_map.grad_b_squared_t2_m,
    )
    norm = np.linalg.norm(force, axis=-1)
    safe = np.where(norm > 0.0, norm, 1.0)
    stride = max(1, args.matrix_size // 7)
    ax.quiver(
        1e3 * x_grid[::stride, ::stride],
        1e3 * y_grid[::stride, ::stride],
        (force[..., 0] / safe)[::stride, ::stride],
        (force[..., 1] / safe)[::stride, ::stride],
        color="white",
        scale=17,
        width=0.005,
    )
    ax.add_patch(Circle(1e3 * last_cycle.target_center_m, 1e3 * config.target_radius_m, fill=False, color="cyan", lw=1.8))
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="last re-aimed transport field and force")
    fig.colorbar(field_image, ax=ax, label="|B| (mT)")

    ax = axes[1, 0]
    ax.imshow(phantom.tissue_labels > 0, origin="lower", extent=extent, cmap="Greys", alpha=0.18)
    selected = np.arange(0, args.particles, max(1, args.particles // 24))
    for cycle, color in zip(result.cycles, colors, strict=True):
        for index in selected:
            path = cycle.transport.positions_m[:, index]
            ax.plot(1e3 * path[:, 0], 1e3 * path[:, 1], color=color, alpha=0.55, lw=0.75)
    ax.scatter(1e3 * result.final_positions_m[:, 0], 1e3 * result.final_positions_m[:, 1], c=np.where(result.final_captured, "tab:red", "tab:blue"), s=12, alpha=0.75)
    for cycle, color in zip(result.cycles, colors, strict=True):
        start = 1e3 * cycle.source_centroid_m
        delta = 1e3 * (cycle.target_center_m - cycle.source_centroid_m)
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            color=color,
            width=0.08,
            head_width=0.65,
            length_includes_head=True,
        )
    ax.add_patch(Circle(1e3 * last_cycle.target_center_m, 1e3 * config.target_radius_m, fill=False, color="tab:red", lw=2))
    ax.set(
        xlim=extent[:2],
        ylim=extent[2:],
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="image-estimated centroids re-aim transport",
    )
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    mode_colors = {"imaging": "tab:purple", "programming": "tab:orange", "transport": "tab:green"}
    mode_rows = {"imaging": 2, "programming": 1, "transport": 0}
    for interval in result.intervals:
        ax.broken_barh(
            [(interval.start_s / 60.0, interval.duration_s / 60.0)],
            (mode_rows[interval.mode] - 0.35, 0.70),
            facecolors=mode_colors[interval.mode],
        )
        if interval.mode == "programming":
            ax.axvline(interval.start_s / 60.0, color="tab:orange", lw=2.2)
    ax.set_yticks((0, 1, 2), ("transport", "program", "imaging"))
    ax.set(xlabel="wall-clock time (min)", title="explicit controller mode timeline")
    capture_axis = ax.twinx()
    cycle_end = np.asarray([cycle.end_s for cycle in result.cycles]) / 60.0
    capture_axis.step(
        cycle_end,
        100 * result.capture_fraction_by_cycle,
        where="post",
        color="0.35",
        lw=1.5,
        linestyle="--",
        label="hidden truth",
    )
    capture_axis.step(
        cycle_end,
        100 * result.estimated_capture_fraction_by_cycle,
        where="post",
        color="tab:red",
        lw=2,
        label="image estimate",
    )
    capture_axis.set_ylabel("capture (%)", color="tab:red")
    capture_axis.set_ylim(0.0, 100.0)
    capture_axis.legend(fontsize=7, loc="center right")

    ax = axes[1, 2]
    indices = np.arange(1, len(result.cycles) + 1)
    centroid_error_mm = 1e3 * result.particle_centroid_error_by_cycle_m
    capture_error_percent = 100 * np.asarray(
        [cycle.particle_imaging_after.capture_fraction_error for cycle in result.cycles]
    )
    bars = ax.bar(indices - 0.18, centroid_error_mm, width=0.36, color="tab:blue")
    ax.set(
        xlabel="controller cycle",
        ylabel="centroid error (mm)",
        title="particle-state estimator error against hidden truth",
    )
    error_axis = ax.twinx()
    error_bars = error_axis.bar(
        indices + 0.18,
        capture_error_percent,
        width=0.36,
        color="tab:orange",
    )
    error_axis.axhline(0.0, color="0.4", lw=0.8)
    error_axis.set_ylabel("occupancy error (points)", color="tab:orange")
    ax.legend(
        [bars, error_bars],
        ["centroid error", "occupancy estimate - truth"],
        fontsize=7,
    )
    ax.set_xticks(indices)
    ax.grid(axis="y", alpha=0.2)

    localized = result.localized_targets_m
    localization_spread_um = 1e6 * float(np.max(np.linalg.norm(localized - np.mean(localized, axis=0), axis=1)))
    print("Closed-loop EPM particle-imaging therapy controller")
    print(f"  cycles executed={len(result.cycles)}/{config.max_cycles}; stop={result.stop_reason}")
    print(f"  total wall time={result.total_time_s / 60.0:.2f} min")
    print(
        "  image-estimated capture by cycle="
        f"{100 * result.estimated_capture_fraction_by_cycle} percent"
    )
    print(f"  final image-estimated capture={100 * result.estimated_capture_fraction:.1f}%")
    print(f"  final hidden-truth capture={100 * result.capture_fraction:.1f}% (score only)")
    print(
        "  particle-centroid error by cycle="
        f"{np.round(1e3 * result.particle_centroid_error_by_cycle_m, 3)} mm"
    )
    print(
        "  final estimated particle count="
        f"{result.cycles[-1].particle_imaging_after.estimate.particle_count}"
    )
    print(f"  target-localization spread={localization_spread_um:.2f} um")
    print(
        "  transport-state remanence L1 variation="
        f"{[round(c.transport_remanence_variation_t, 3) for c in result.cycles]} channel T"
    )
    print(f"  total imaging+transport remanence variation={result.total_remanence_variation_t:.2f} channel T")
    print(
        "  peak force by cycle="
        f"{[round(1e12 * c.transport.peak_force_n, 3) for c in result.cycles]} pN"
    )
    print("  pulse energy unavailable: requires calibrated 72-channel programming schedule")
    print("  next: calibrate particle contrast and add tissue/vascular transport")

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
