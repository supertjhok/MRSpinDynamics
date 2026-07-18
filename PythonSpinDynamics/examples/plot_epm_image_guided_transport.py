"""Image-guided magnetic-particle transport with a hybrid EPM array.

The example first reconstructs the off-center target in the simple tissue
phantom from nonlinear retained-state encodings.  That image-derived centroid
then determines an affine transport objective.  A superparamagnetic aggregate
is advanced with magnetic force, Stokes drag, background flow, Brownian
diffusion, reflecting tissue-map boundaries, and irreversible target capture.

The array geometry and particle aggregate are illustrative engineering cases,
not a clinical device or dose model.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import (  # noqa: E402
    illustrative_hybrid_epm_array,
    synthesize_transport_state,
)
from spin_dynamics.workflows import (  # noqa: E402
    SuperparamagneticParticle,
    build_epm_nonlinear_encoding,
    magnetic_force_map_2d,
    random_epm_encoding_states,
    run_epm_nonlinear_imaging,
    simple_tissue_phantom,
    simulate_magnetophoretic_transport,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a simple tissue target, synthesize an EPM transport "
            "state, and compare guided nanoparticle trajectories with flow only."
        )
    )
    parser.add_argument("--matrix-size", type=int, default=14, help="Square image/map size.")
    parser.add_argument("--encodings", type=int, default=256, help="Nonlinear imaging states.")
    parser.add_argument(
        "--transport-gradient-mt-per-m",
        type=float,
        default=150.0,
        help="Requested affine transport-gradient magnitude (default: 150 mT/m).",
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        default=75.0,
        help="Transport burst duration (default: 75 min).",
    )
    parser.add_argument("--particles", type=int, default=80, help="Particle aggregates.")
    parser.add_argument("--seed", type=int, default=8, help="Imaging/transport seed.")
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if args.matrix_size < 10:
        raise ValueError("--matrix-size must be at least 10")
    if args.encodings < args.matrix_size**2:
        raise ValueError("--encodings must be at least the voxel count")
    if not np.isfinite(args.transport_gradient_mt_per_m) or args.transport_gradient_mt_per_m <= 0:
        raise ValueError("--transport-gradient-mt-per-m must be positive")
    if not np.isfinite(args.duration_min) or args.duration_min <= 0:
        raise ValueError("--duration-min must be positive")
    if args.particles < 10:
        raise ValueError("--particles must be at least 10")


def _weighted_centroid(image: np.ndarray, x_m: np.ndarray, y_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
    xx, yy = np.meshgrid(x_m, y_m, indexing="xy")
    weights = np.where(mask, np.maximum(image, 0.0), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("localized target mask has zero image weight")
    return np.asarray((np.sum(weights * xx) / total, np.sum(weights * yy) / total))


def _zero_magnetic_particle(particle: SuperparamagneticParticle) -> SuperparamagneticParticle:
    return SuperparamagneticParticle(
        magnetic_core_radius_m=particle.magnetic_core_radius_m,
        hydrodynamic_radius_m=particle.hydrodynamic_radius_m,
        volume_susceptibility=0.0,
        saturation_magnetization_a_m=particle.saturation_magnetization_a_m,
        fluid_viscosity_pa_s=particle.fluid_viscosity_pa_s,
        temperature_k=particle.temperature_k,
        magnetic_volume_fraction=particle.magnetic_volume_fraction,
        label="matched nonmagnetic control",
    )


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
    imaging = run_epm_nonlinear_imaging(
        encoding,
        expected,
        regularization=1e-4,
        snr_db=35.0,
        seed=args.seed + 1,
    )
    tissue_mean = float(np.mean(expected[phantom.tissue_labels == 1]))
    target_mean = float(np.mean(expected[phantom.target_mask]))
    threshold = 0.5 * (tissue_mean + target_mean)
    reconstructed_target = imaging.reconstructed_image > threshold
    target_center = _weighted_centroid(
        imaging.reconstructed_image,
        phantom.x_m,
        phantom.y_m,
        reconstructed_target,
    )
    true_center = _weighted_centroid(expected, phantom.x_m, phantom.y_m, phantom.target_mask)

    injection_center = np.asarray((-0.014, -0.0005))
    direction = target_center - injection_center
    direction /= np.linalg.norm(direction)
    requested_gradient = args.transport_gradient_mt_per_m * 1e-3 * direction
    transport_state = synthesize_transport_state(
        basis,
        bias_field_t=2.0e-3,
        gradient_t_per_m=(requested_gradient[0], requested_gradient[1], 0.0),
        center_m=(0.0, 0.0, 0.0),
        field_direction=(0.0, 0.0, 1.0),
        regularization=1e-10,
        tolerance_t=1e-10,
    )
    transport_field = basis.field_vectors(transport_state.remanence_t).reshape(
        phantom.shape + (3,)
    )
    force_map = magnetic_force_map_2d(phantom.x_m, phantom.y_m, transport_field)

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
            rng.normal(injection_center[0], 0.35e-3, args.particles),
            rng.normal(injection_center[1], 1.5e-3, args.particles),
        )
    )
    target_radius = 0.105 * 0.040
    common_transport = dict(
        duration_s=args.duration_min * 60.0,
        time_step_s=5.0,
        target_center_m=target_center,
        target_radius_m=target_radius,
        background_velocity_m_s=(2.5e-6, 0.0),
        boundary="reflect",
        seed=args.seed + 2,
    )
    guided = simulate_magnetophoretic_transport(
        force_map,
        particle,
        initial,
        **common_transport,
    )
    flow_only = simulate_magnetophoretic_transport(
        force_map,
        _zero_magnetic_particle(particle),
        initial,
        **common_transport,
    )

    x_grid, y_grid = np.meshgrid(phantom.x_m, phantom.y_m, indexing="xy")
    extent = 1e3 * np.asarray(
        (phantom.x_m[0], phantom.x_m[-1], phantom.y_m[0], phantom.y_m[-1])
    )
    force_nodes = magnetic_force_from_gradient_for_plot(force_map, particle)
    force_norm = np.linalg.norm(force_nodes, axis=-1)
    force_safe = np.where(force_norm > 0.0, force_norm, 1.0)

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), constrained_layout=True)
    fig.suptitle("Image-guided magnetic aggregate transport with a hybrid EPM array", fontsize=15)

    ax = axes[0, 0]
    half_gap_mm = 75.0
    ax.add_patch(Rectangle((-50, -half_gap_mm - 5), 100, 10, color="0.3", alpha=0.35))
    ax.add_patch(Rectangle((-50, half_gap_mm - 5), 100, 10, color="0.3", alpha=0.35))
    for y_mm in (-half_gap_mm, half_gap_mm):
        for x_mm in (-35.0, 0.0, 35.0):
            ax.plot(x_mm, y_mm, "s", ms=10, color="tab:purple")
    ax.add_patch(Ellipse((0, 0), 40, 34, fill=False, lw=2, color="tab:green"))
    ax.plot(1e3 * target_center[0], 1e3 * target_center[1], "r*", ms=12)
    ax.set(xlim=(-58, 58), ylim=(-92, 92), xlabel="x (mm)", ylabel="panel axis (mm)", title="two 3x3 hybrid panels and tissue ROI")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    image = ax.imshow(imaging.reconstructed_image, origin="lower", extent=extent, cmap="magma")
    ax.contour(1e3 * phantom.x_m, 1e3 * phantom.y_m, reconstructed_target, levels=(0.5,), colors="cyan", linewidths=1.5)
    ax.plot(1e3 * true_center[0], 1e3 * true_center[1], "w+", ms=11, mew=2, label="true")
    ax.plot(1e3 * target_center[0], 1e3 * target_center[1], "cx", ms=9, mew=2, label="localized")
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"nonlinear EPM reconstruction; NRMSE {100 * imaging.nrmse:.2f}%")
    ax.legend(fontsize=8)
    fig.colorbar(image, ax=ax, label="relative spin-echo signal")

    ax = axes[0, 2]
    field_image = ax.imshow(1e3 * force_map.field_magnitude_t, origin="lower", extent=extent, cmap="viridis")
    stride = max(1, args.matrix_size // 7)
    ax.quiver(
        1e3 * x_grid[::stride, ::stride],
        1e3 * y_grid[::stride, ::stride],
        (force_nodes[..., 0] / force_safe)[::stride, ::stride],
        (force_nodes[..., 1] / force_safe)[::stride, ::stride],
        color="white",
        scale=17,
        width=0.005,
    )
    ax.add_patch(Circle(1e3 * target_center, 1e3 * target_radius, fill=False, color="cyan", lw=1.7))
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title="transport |B| (mT) and force direction")
    fig.colorbar(field_image, ax=ax, label="|B| (mT)")

    ax = axes[1, 0]
    ax.imshow(phantom.tissue_labels > 0, origin="lower", extent=extent, cmap="Greys", alpha=0.18)
    for trajectory in guided.positions_m[:, :: max(1, args.particles // 24), :].transpose(1, 0, 2):
        ax.plot(1e3 * trajectory[:, 0], 1e3 * trajectory[:, 1], color="tab:blue", alpha=0.65, lw=0.8)
    ax.scatter(1e3 * initial[:, 0], 1e3 * initial[:, 1], s=9, color="0.35", label="injected")
    ax.scatter(1e3 * guided.positions_m[-1, :, 0], 1e3 * guided.positions_m[-1, :, 1], s=12, c=np.where(guided.captured, "tab:red", "tab:blue"), alpha=0.75)
    ax.add_patch(Circle(1e3 * target_center, 1e3 * target_radius, fill=False, color="tab:red", lw=2))
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"guided trajectories; {100 * guided.capture_fraction:.1f}% captured")
    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    ax.plot(guided.time_s / 60.0, 100 * guided.cumulative_capture_fraction, lw=2, label="EPM guided")
    ax.plot(flow_only.time_s / 60.0, 100 * flow_only.cumulative_capture_fraction, lw=2, label="flow only")
    ax.set(xlabel="transport time (min)", ylabel="cumulative capture (%)", title="image-selected target capture")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    force_values = np.linalg.norm(guided.magnetic_force_n, axis=-1)
    speed_values = np.linalg.norm(guided.magnetic_velocity_m_s, axis=-1)
    ax.plot(guided.time_s / 60.0, 1e12 * np.median(force_values, axis=1), label="median force (pN)")
    ax.plot(guided.time_s / 60.0, 1e6 * np.median(speed_values, axis=1), label="median magnetic speed (um/s)")
    ax.set(xlabel="transport time (min)", ylabel="force or speed", title="overdamped transport diagnostics")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    affine = np.linalg.lstsq(
        np.column_stack((np.ones(phantom.points_m.shape[0]), phantom.points_m)),
        transport_state.predicted_projected_field_t,
        rcond=None,
    )[0]
    achieved_gradient = affine[1:3]
    localization_error_mm = 1e3 * float(np.linalg.norm(target_center - true_center))
    print("Image-guided EPM nanoparticle transport")
    print(f"  imaging: condition={encoding.condition_number:.3f}, NRMSE={100 * imaging.nrmse:.3f}%")
    print(f"  target localization error={localization_error_mm:.3f} mm")
    print(
        "  requested/achieved in-plane gradient="
        f"{1e3 * np.linalg.norm(requested_gradient):.2f}/"
        f"{1e3 * np.linalg.norm(achieved_gradient):.2f} mT/m"
    )
    print(
        f"  |B| range={1e3 * np.min(force_map.field_magnitude_t):.3f}-"
        f"{1e3 * np.max(force_map.field_magnitude_t):.3f} mT"
    )
    print(
        f"  aggregate: magnetic radius={1e6 * particle.magnetic_core_radius_m:.1f} um, "
        f"hydrodynamic diameter={2e6 * particle.hydrodynamic_radius_m:.1f} um, "
        f"D={particle.diffusion_coefficient_m2_s:.3e} m^2/s"
    )
    print(
        f"  force median/peak={1e12 * guided.median_force_n:.3f}/"
        f"{1e12 * guided.peak_force_n:.3f} pN; "
        f"peak magnetic speed={1e6 * guided.peak_magnetic_speed_m_s:.3f} um/s"
    )
    print(
        f"  target capture guided/flow-only={100 * guided.capture_fraction:.1f}/"
        f"{100 * flow_only.capture_fraction:.1f}%"
    )
    print("  next: close the loop with particle-sensitive imaging and state estimation")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


def magnetic_force_from_gradient_for_plot(force_map, particle):
    from spin_dynamics.workflows import magnetic_force_from_gradient

    return magnetic_force_from_gradient(
        particle,
        force_map.field_magnitude_t,
        force_map.grad_b_squared_t2_m,
    )


if __name__ == "__main__":
    main()
