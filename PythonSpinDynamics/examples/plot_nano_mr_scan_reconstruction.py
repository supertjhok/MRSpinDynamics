"""Reconstruct a sparse planar nano-MRI sample from a scanning NV raster.

The example builds a statistical field-variance forward operator, adds seeded
measurement noise, performs nonnegative density inversion, and refines two
point positions nonlinearly. Run
``python examples/plot_nano_mr_scan_reconstruction.py --help``.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.nano_mr import (  # noqa: E402
    NuclearBathSpecies,
    build_voxel_density_forward_operator,
    localize_point_sources,
    nuclear_voxel_variance_amplitudes,
    planar_voxel_grid,
    raster_scan,
    reconstruct_nonnegative_density,
)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-mt", type=float, default=20.0)
    parser.add_argument("--sensor-depth-nm", type=float, default=8.0)
    parser.add_argument("--noise-fraction", type=float, default=0.004)
    parser.add_argument("--regularization", type=float, default=1.0e-6)
    parser.add_argument(
        "--correlated-noise",
        action="store_true",
        help="use a seeded spatial/temporal covariance and generalized least squares",
    )
    parser.add_argument("--correlation-length-nm", type=float, default=3.0)
    parser.add_argument("--correlation-time-us", type=float, default=20.0)
    parser.add_argument("--shot-interval-us", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.field_mt < 0.0:
        parser.error("--field-mt must be non-negative")
    if args.sensor_depth_nm <= 0.0:
        parser.error("--sensor-depth-nm must be positive")
    if args.noise_fraction <= 0.0:
        parser.error("--noise-fraction must be positive")
    if args.regularization < 0.0:
        parser.error("--regularization must be non-negative")
    if args.correlation_length_nm <= 0.0:
        parser.error("--correlation-length-nm must be positive")
    if args.correlation_time_us <= 0.0:
        parser.error("--correlation-time-us must be positive")
    if args.shot_interval_us <= 0.0:
        parser.error("--shot-interval-us must be positive")

    axis = np.array([np.sqrt(2.0 / 3.0), 0.0, np.sqrt(1.0 / 3.0)])
    scan_axis = np.linspace(-12.0, 12.0, 25)
    source_axis = np.linspace(-10.0, 10.0, 17)
    scan = raster_scan(
        scan_axis,
        scan_axis,
        z_nm=-args.sensor_depth_nm,
        sensor_axis_lab=axis,
    )
    grid = planar_voxel_grid(
        source_axis,
        source_axis,
        z_nm=0.0,
        thickness_nm=2.0,
    )
    proton = NuclearBathSpecies.from_isotope("1H")
    operator = build_voxel_density_forward_operator(
        scan,
        grid,
        proton,
        field_tesla=args.field_mt * 1.0e-3,
        field_axis_lab=axis,
        minimum_distance_nm=args.sensor_depth_nm,
    )

    true_indices = ((6, 5), (11, 12))
    true_density = np.zeros(grid.shape)
    true_density[true_indices[0]] = 6.0e28
    true_density[true_indices[1]] = 4.0e28
    clean = operator.predict(true_density)
    noise_std = args.noise_fraction * float(np.max(clean))
    generator = np.random.default_rng(args.seed)
    noise_covariance = None
    if args.correlated_noise:
        positions = scan.positions_lab_nm
        displacement = (
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        )
        distance_nm = np.linalg.norm(displacement, axis=2)
        times_seconds = (
            np.arange(clean.size) * args.shot_interval_us * 1.0e-6
        )
        lag_seconds = np.abs(
            times_seconds[:, np.newaxis] - times_seconds[np.newaxis, :]
        )
        correlation = np.exp(
            -distance_nm / args.correlation_length_nm
            -lag_seconds / (args.correlation_time_us * 1.0e-6)
        )
        correlation = 0.8 * correlation + 0.2 * np.eye(clean.size)
        noise_covariance = noise_std**2 * correlation
        eigenvalues, eigenvectors = np.linalg.eigh(noise_covariance)
        measurement_noise = eigenvectors @ (
            np.sqrt(np.maximum(eigenvalues, 0.0))
            * generator.standard_normal(clean.size)
        )
    else:
        measurement_noise = generator.normal(
            scale=noise_std,
            size=clean.size,
        )
    measured = clean + measurement_noise
    reconstruction = reconstruct_nonnegative_density(
        operator,
        measured,
        regularization=args.regularization,
        regularization_order=1,
        noise_std=None if noise_covariance is not None else noise_std,
        noise_covariance=noise_covariance,
    )

    true_positions = np.vstack(
        [
            grid.positions_lab_nm[
                np.ravel_multi_index(index, grid.shape)
            ]
            for index in true_indices
        ]
    )
    true_amplitudes = nuclear_voxel_variance_amplitudes(
        proton,
        args.field_mt * 1.0e-3,
        np.array([true_density[index] for index in true_indices]),
        grid.voxel_volumes_nm3[
            [np.ravel_multi_index(index, grid.shape) for index in true_indices]
        ],
    )
    point_fit = localize_point_sources(
        scan,
        measured,
        true_positions + np.array([[1.0, -0.8, 0.0], [-0.8, 1.0, 0.0]]),
        response_kind="transverse_variance",
        field_axis_lab=axis,
        initial_amplitudes=0.8 * true_amplitudes,
        bounds_lab_nm=((-11.0, 11.0), (-11.0, 11.0), (-0.5, 0.5)),
        minimum_distance_nm=args.sensor_depth_nm,
        noise_std=None if noise_covariance is not None else noise_std,
        noise_covariance=noise_covariance,
    )

    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=args.output is not None)
    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    extent = [
        source_axis[0],
        source_axis[-1],
        source_axis[0],
        source_axis[-1],
    ]
    density_scale = 1.0e28
    image0 = axes[0, 0].imshow(
        true_density / density_scale,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    axes[0, 0].scatter(
        true_positions[:, 0],
        true_positions[:, 1],
        marker="+",
        s=100,
        color="cyan",
        label="true points",
    )
    axes[0, 0].set(title="True planar spin density", xlabel="x (nm)", ylabel="y (nm)")
    axes[0, 0].legend(loc="upper left")
    fig.colorbar(image0, ax=axes[0, 0], label=r"Density ($10^{28}$ m$^{-3}$)")

    scan_image = scan.reshape_raster(measured) * 1.0e18
    image1 = axes[0, 1].imshow(
        scan_image,
        origin="lower",
        extent=[scan_axis[0], scan_axis[-1], scan_axis[0], scan_axis[-1]],
        cmap="viridis",
    )
    axes[0, 1].set(
        title=(
            f"{'Correlated' if args.correlated_noise else 'Independent'} "
            f"NV raster (seed {args.seed})"
        ),
        xlabel="sensor x (nm)",
        ylabel="sensor y (nm)",
    )
    fig.colorbar(image1, ax=axes[0, 1], label=r"Field variance (nT$^2$)")

    image2 = axes[1, 0].imshow(
        reconstruction.density / density_scale,
        origin="lower",
        extent=extent,
        cmap="magma",
        interpolation="nearest",
    )
    axes[1, 0].scatter(
        point_fit.positions_lab_nm[:, 0],
        point_fit.positions_lab_nm[:, 1],
        marker="x",
        s=75,
        color="cyan",
        label="nonlinear point fit",
    )
    axes[1, 0].set(
        title=(
            "Nonnegative generalized least squares"
            if args.correlated_noise
            else "Nonnegative regularized reconstruction"
        ),
        xlabel="x (nm)",
        ylabel="y (nm)",
    )
    axes[1, 0].legend(loc="upper left")
    fig.colorbar(image2, ax=axes[1, 0], label=r"Density ($10^{28}$ m$^{-3}$)")

    image3 = axes[1, 1].imshow(
        reconstruction.standard_deviation / density_scale,
        origin="lower",
        extent=extent,
        cmap="cividis",
    )
    direct_rms = np.sqrt(
        np.mean(
            np.sum(
                (point_fit.positions_lab_nm[:, :2] - true_positions[:, :2]) ** 2,
                axis=1,
            )
        )
    )
    swapped_rms = np.sqrt(
        np.mean(
            np.sum(
                (point_fit.positions_lab_nm[::-1, :2] - true_positions[:, :2]) ** 2,
                axis=1,
            )
        )
    )
    localization_rms = min(direct_rms, swapped_rms)
    axes[1, 1].set(
        title=(
            "Linearized density uncertainty\n"
            f"point RMS error = {localization_rms:.2f} nm"
        ),
        xlabel="x (nm)",
        ylabel="y (nm)",
    )
    fig.colorbar(image3, ax=axes[1, 1], label=r"1$\sigma$ ($10^{28}$ m$^{-3}$)")

    # Save reproducibly for batch runs; otherwise keep the figure interactive.
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
