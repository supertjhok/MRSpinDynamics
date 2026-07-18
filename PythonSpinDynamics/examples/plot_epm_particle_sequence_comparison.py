"""Compare MR sequences for susceptibility-based particle localization.

Spin echo, Cartesian gradient echo, GRE phase-gradient positive contrast, and
center-out radial short-TE imaging use the same EPM B0 state, particle dipole
fields, tissue maps, subvoxel water exclusion, and diffusion realization.  The
comparison reports localization errors rather than assuming that the sequence
with the strongest-looking contrast is the most accurate.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import illustrative_hybrid_epm_array
from spin_dynamics.workflows import (
    SuperparamagneticParticle,
    build_epm_nonlinear_encoding,
    random_epm_encoding_states,
    run_epm_particle_susceptibility_imaging,
    simple_tissue_phantom,
)


SEQUENCES = ("spin_echo", "gradient_echo", "phase_gradient", "radial_ute")
LABELS = {
    "spin_echo": "Spin echo",
    "gradient_echo": "Gradient echo",
    "phase_gradient": "GRE phase gradient",
    "radial_ute": "Radial short TE",
}
COLORS = ("#4477AA", "#EE6677", "#228833", "#CCBB44")


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
        help="Optional per-sequence complex acquisition SNR.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parser().parse_args()
    plt = load_matplotlib(headless=args.output is not None)
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
    common = dict(
        target_center_m=target_center,
        target_radius_m=4.2e-3,
        imaging_state_index=state_index,
        subvoxel_grid_size=args.subvoxel_grid,
        water_walkers_per_voxel=args.walkers_per_voxel,
        support_threshold_fraction=0.08,
        snr_db=args.snr_db,
        seed=args.seed + 1,
    )
    results = {
        sequence: run_epm_particle_susceptibility_imaging(
            encoding,
            positions,
            phantom.x_m,
            phantom.y_m,
            particle,
            phantom.proton_density,
            phantom.t1_s,
            phantom.t2_s,
            sequence=sequence,
            **common,
        )
        for sequence in SEQUENCES
    }

    extent = (
        phantom.x_m[0] * 1e3,
        phantom.x_m[-1] * 1e3,
        phantom.y_m[0] * 1e3,
        phantom.y_m[-1] * 1e3,
    )
    figure = plt.figure(figsize=(14.5, 8.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 4, height_ratios=(1.35, 0.8))
    for column, sequence in enumerate(SEQUENCES):
        result = results[sequence]
        axis = figure.add_subplot(grid[0, column])
        contrast = result.estimate.contrast_image
        normalized = contrast / max(float(np.max(contrast)), np.finfo(float).eps)
        artist = axis.imshow(
            normalized,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        axis.scatter(
            positions[:, 0] * 1e3,
            positions[:, 1] * 1e3,
            s=20,
            facecolors="none",
            edgecolors="cyan",
            linewidths=0.8,
            label="hidden truth",
        )
        if result.estimate.positions_m.size:
            axis.scatter(
                result.estimate.positions_m[:, 0] * 1e3,
                result.estimate.positions_m[:, 1] * 1e3,
                marker="x",
                s=42,
                color="white",
                linewidths=1.1,
                label="resolved foci",
            )
        axis.scatter(
            result.estimate.centroid_m[0] * 1e3,
            result.estimate.centroid_m[1] * 1e3,
            marker="+",
            s=90,
            color="lime",
            linewidths=1.8,
            label="estimated centroid",
        )
        axis.set_title(
            f"{LABELS[sequence]}\n"
            f"TE={result.echo_time_s * 1e6:.0f} µs, "
            f"centroid={result.centroid_error_m * 1e3:.2f} mm"
        )
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        if column == 0:
            axis.legend(loc="upper left", fontsize=7)
        figure.colorbar(artist, ax=axis, shrink=0.78, label="normalized contrast")

    indices = np.arange(len(SEQUENCES))
    centroid_errors = np.asarray(
        [results[name].centroid_error_m * 1e3 for name in SEQUENCES]
    )
    chamfer_errors = np.asarray(
        [results[name].focus_chamfer_error_m * 1e3 for name in SEQUENCES]
    )
    echo_times = np.asarray([results[name].echo_time_s * 1e6 for name in SEQUENCES])
    focus_counts = np.asarray(
        [results[name].estimate.resolved_focus_count for name in SEQUENCES]
    )
    summaries = (
        (centroid_errors, "Centroid error", "mm", False),
        (chamfer_errors, "Symmetric focus error", "mm", False),
        (echo_times, "Center-k-space echo time", "µs", True),
        (focus_counts, "Resolved contrast foci", "count", False),
    )
    for column, (values, title, ylabel, logarithmic) in enumerate(summaries):
        axis = figure.add_subplot(grid[1, column])
        axis.bar(indices, values, color=COLORS)
        axis.set_xticks(indices, ("SE", "GRE", "PGM", "UTE"))
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if logarithmic:
            axis.set_yscale("log")
        if column == 3:
            axis.axhline(args.particles, color="black", ls="--", lw=1, label="particles")
            axis.legend(fontsize=8)
    figure.suptitle(
        "Particle-localization sequence comparison under one EPM B0 state"
        + (" (no receiver noise)" if args.snr_db is None else f" (SNR={args.snr_db:g} dB)"),
        fontsize=14,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"wrote {args.output}")
    else:
        plt.show()
    plt.close(figure)
    for sequence in SEQUENCES:
        result = results[sequence]
        print(
            f"{sequence:>14}: TE={result.echo_time_s * 1e6:7.1f} us, "
            f"centroid={result.centroid_error_m * 1e3:5.2f} mm, "
            f"focus={result.focus_chamfer_error_m * 1e3:5.2f} mm, "
            f"foci={result.estimate.resolved_focus_count}"
        )


if __name__ == "__main__":
    main()
