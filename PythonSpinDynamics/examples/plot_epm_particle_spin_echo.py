"""Physical particle contrast from an EPM-field spin-echo acquisition.

The example deliberately keeps the earlier directly detectable particle image
out of the forward model.  Magnetic aggregates perturb the EPM B0 field; a
finite-pulse spin-warp sequence then produces signal loss, distortion, and
diffusion-mediated dephasing in surrounding tissue.  A paired particle-free
reference converts the negative contrast into a localization image.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spin_dynamics.fields import illustrative_hybrid_epm_array
from spin_dynamics.workflows import (
    SuperparamagneticParticle,
    build_epm_nonlinear_encoding,
    random_epm_encoding_states,
    run_epm_particle_susceptibility_spin_echo,
    simple_tissue_phantom,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-size", type=int, default=16)
    parser.add_argument("--particles", type=int, default=24)
    parser.add_argument("--core-radius-um", type=float, default=100.0)
    parser.add_argument("--subvoxel-grid", type=int, default=3)
    parser.add_argument("--walkers-per-voxel", type=int, default=4)
    parser.add_argument(
        "--snr-db",
        type=float,
        default=None,
        help="Optional complex k-space SNR; omitted to isolate sequence contrast.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/images/example_epm_particle_spin_echo.png"),
    )
    return parser


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parser().parse_args()
    rng = np.random.default_rng(args.seed)
    phantom = simple_tissue_phantom(args.matrix_size, field_of_view_m=0.040)
    array = illustrative_hybrid_epm_array(panel_gap_m=0.150)
    basis = array.build_field_basis(phantom.points_m, n_cross=1, n_length=7)
    states = random_epm_encoding_states(
        basis,
        max(96, 2 * phantom.proton_density.size),
        amplitude_fraction=0.60,
        seed=args.seed,
    )
    encoding = build_epm_nonlinear_encoding(
        basis,
        states,
        image_shape=phantom.shape,
        phase_encoding_s=300e-6,
    )
    state_index = int(
        np.argmax(np.mean(np.abs(encoding.projected_fields_t), axis=1))
    )
    target_indices = np.argwhere(phantom.target_mask)
    target_center = np.asarray(
        (
            np.mean(phantom.x_m[target_indices[:, 1]]),
            np.mean(phantom.y_m[target_indices[:, 0]]),
        )
    )
    positions = np.column_stack(
        (
            rng.normal(-7e-3, 2.5e-3, args.particles),
            rng.normal(0.0, 3.0e-3, args.particles),
        )
    )
    positions[:, 0] = np.clip(positions[:, 0], phantom.x_m[1], phantom.x_m[-2])
    positions[:, 1] = np.clip(positions[:, 1], phantom.y_m[1], phantom.y_m[-2])
    particle = SuperparamagneticParticle(
        magnetic_core_radius_m=args.core_radius_um * 1e-6,
        hydrodynamic_radius_m=1.25 * args.core_radius_um * 1e-6,
        volume_susceptibility=1.4,
        saturation_magnetization_a_m=4.5e5,
        fluid_viscosity_pa_s=1.5e-3,
        temperature_k=310.0,
        magnetic_volume_fraction=0.60,
        label="illustrative magnetic aggregate",
    )
    result = run_epm_particle_susceptibility_spin_echo(
        encoding,
        positions,
        phantom.x_m,
        phantom.y_m,
        particle,
        phantom.proton_density,
        phantom.t1_s,
        phantom.t2_s,
        target_center_m=target_center,
        target_radius_m=4.2e-3,
        imaging_state_index=state_index,
        subvoxel_grid_size=args.subvoxel_grid,
        water_walkers_per_voxel=args.walkers_per_voxel,
        support_threshold_fraction=0.08,
        snr_db=args.snr_db,
        seed=args.seed + 1,
    )

    extent = (
        phantom.x_m[0] * 1e3,
        phantom.x_m[-1] * 1e3,
        phantom.y_m[0] * 1e3,
        phantom.y_m[-1] * 1e3,
    )
    mean_delta = np.mean(result.particle_delta_b0_samples_t, axis=0)
    reference_magnitude = np.abs(result.reference_image)
    particle_magnitude = np.abs(result.particle_image)
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 8.0), constrained_layout=True)
    panels = (
        (result.background_field_t * 1e3, "EPM imaging B0", "mT", "viridis"),
        (mean_delta * 1e6, "Particle-induced ΔB0", "µT", "coolwarm"),
        (reference_magnitude, "Reference spin echo", "signal", "gray"),
        (particle_magnitude, "Particle-present spin echo", "signal", "gray"),
        (result.signed_contrast_image, "Reference − particle", "signal", "coolwarm"),
        (result.estimate.contrast_image, "Recovered particle contrast", "signal", "magma"),
    )
    for axis, (image, title, label, cmap) in zip(axes.ravel(), panels):
        artist = axis.imshow(image, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        figure.colorbar(artist, ax=axis, shrink=0.82, label=label)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    contrast_axis = axes[1, 2]
    contrast_axis.scatter(
        positions[:, 0] * 1e3,
        positions[:, 1] * 1e3,
        s=18,
        facecolors="none",
        edgecolors="cyan",
        linewidths=0.8,
        label="hidden truth",
    )
    if result.estimate.positions_m.size:
        contrast_axis.scatter(
            result.estimate.positions_m[:, 0] * 1e3,
            result.estimate.positions_m[:, 1] * 1e3,
            marker="x",
            s=44,
            color="white",
            linewidths=1.2,
            label="resolved foci",
        )
    contrast_axis.legend(loc="upper left", fontsize=8)
    figure.suptitle(
        "Susceptibility-aware EPM spin echo: "
        f"TE={result.echo_time_s * 1e3:.2f} ms, "
        f"D={result.particle_acquisition.diffusion_coefficient:.2g} m²/s, "
        f"centroid error={result.centroid_error_m * 1e3:.2f} mm, "
        + ("no receiver noise" if args.snr_db is None else f"SNR={args.snr_db:g} dB"),
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"wrote {args.output}")
    print(f"imaging state: {state_index}")
    print(f"B0 range: {np.ptp(result.background_field_t) * 1e3:.3f} mT")
    print(
        "particle dB0 range: "
        f"{np.min(result.particle_delta_b0_samples_t) * 1e6:.3f} to "
        f"{np.max(result.particle_delta_b0_samples_t) * 1e6:.3f} uT"
    )
    print(f"resolved susceptibility foci: {result.estimate.resolved_focus_count}")


if __name__ == "__main__":
    main()
