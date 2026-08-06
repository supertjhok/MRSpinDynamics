"""Visualize ideal birdcage modes, quadrature polarization, and B1 uniformity."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields import (
    BirdcageGeometry,
    birdcage_field_metrics,
    birdcage_linear_mode,
    birdcage_quadrature_mode,
    solve_birdcage_field,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the prescribed-current fundamental and quadrature modes "
            "of a cylindrical birdcage coil."
        )
    )
    parser.add_argument("--radius-cm", type=float, default=15.0)
    parser.add_argument("--length-cm", type=float, default=30.0)
    parser.add_argument("--rungs", type=int, default=16)
    parser.add_argument("--ring-segments", type=int, default=8)
    parser.add_argument("--grid", type=int, default=61)
    parser.add_argument("--roi-fraction", type=float, default=0.4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _plot_geometry(axis, geometry: BirdcageGeometry) -> None:
    for start, end in geometry.rung_segments():
        points = np.stack((start, end))
        axis.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue")
    for sections in (
        geometry.positive_end_ring_sections,
        geometry.negative_end_ring_sections,
    ):
        for section in sections:
            points = np.asarray(
                [section[0][0], *(end for _, end in section)],
            )
            axis.plot(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                color="tab:orange",
            )
    radius = geometry.radius
    half_length = geometry.length / 2.0
    axis.set(
        xlim=(-1.1 * radius, 1.1 * radius),
        ylim=(-1.1 * radius, 1.1 * radius),
        zlim=(-1.1 * half_length, 1.1 * half_length),
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="z (m)",
        title="Explicit rung/end-ring geometry",
    )
    axis.set_box_aspect((2.0 * radius, 2.0 * radius, geometry.length))


def _center_polarization_plot(
    axis,
    cosine_field: np.ndarray,
    sine_field: np.ndarray,
    quadrature_field: np.ndarray,
) -> None:
    scale = max(
        np.linalg.norm(cosine_field[:2]),
        np.linalg.norm(sine_field[:2]),
    )
    cosine = np.real(cosine_field[:2]) / scale
    sine = np.real(sine_field[:2]) / scale
    phase = np.linspace(0.0, 2.0 * np.pi, 181)
    trajectory = np.real(
        quadrature_field[np.newaxis, :2]
        * np.exp(1.0j * phase[:, np.newaxis])
    )
    trajectory /= scale
    axis.axhline(0.0, color="0.8", linewidth=0.8)
    axis.axvline(0.0, color="0.8", linewidth=0.8)
    axis.quiver(
        [0.0, 0.0],
        [0.0, 0.0],
        [cosine[0], sine[0]],
        [cosine[1], sine[1]],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=("tab:blue", "tab:orange"),
    )
    axis.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="tab:green",
        label="quadrature trajectory",
    )
    axis.set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        xlabel=r"$B_x$ (normalized)",
        ylabel=r"$B_y$ (normalized)",
        title="Degenerate linear modes → circular B1",
        aspect="equal",
    )
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)


def main() -> None:
    args = _parse_args()
    if args.radius_cm <= 0.0 or args.length_cm <= 0.0:
        raise ValueError("radius and length must be positive")
    if args.rungs < 4:
        raise ValueError("--rungs must be at least 4")
    if args.ring_segments < 1:
        raise ValueError("--ring-segments must be at least 1")
    if args.grid < 9:
        raise ValueError("--grid must be at least 9")
    if not 0.0 < args.roi_fraction < 0.75:
        raise ValueError("--roi-fraction must lie between 0 and 0.75")
    plt = load_matplotlib(headless=args.output is not None)
    assert plt is not None

    geometry = BirdcageGeometry(
        radius=args.radius_cm * 1.0e-2,
        length=args.length_cm * 1.0e-2,
        n_rungs=args.rungs,
        ring_segments_per_section=args.ring_segments,
    )
    cosine_mode = birdcage_linear_mode(geometry, label="cosine")
    sine_mode = birdcage_linear_mode(
        geometry,
        azimuthal_phase_rad=np.pi / 2.0,
        label="sine",
    )
    quadrature_mode = birdcage_quadrature_mode(
        geometry,
        handedness=1,
        label="B1+ quadrature",
    )

    center = np.asarray([geometry.center])
    cosine_center = solve_birdcage_field(
        geometry,
        cosine_mode,
        center,
    ).field_t[0]
    sine_center = solve_birdcage_field(
        geometry,
        sine_mode,
        center,
    ).field_t[0]
    quadrature_center = solve_birdcage_field(
        geometry,
        quadrature_mode,
        center,
    ).field_t[0]

    half_extent = 0.72 * geometry.radius
    coordinate = np.linspace(-half_extent, half_extent, args.grid)
    xx, yy = np.meshgrid(coordinate, coordinate, indexing="ij")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    solution = solve_birdcage_field(geometry, quadrature_mode, points)
    roi = xx**2 + yy**2 <= (args.roi_fraction * geometry.radius) ** 2
    metrics = birdcage_field_metrics(
        solution,
        roi_mask=roi,
        target_handedness=1,
    )

    b1_plus_ut = np.abs(solution.b1_plus_t) * 1.0e6
    roi_mean_ut = metrics.mean_b1_t * 1.0e6
    deviation_percent = 100.0 * (b1_plus_ut / roi_mean_ut - 1.0)
    counter_ratio = np.abs(solution.b1_minus_t) / np.maximum(
        np.abs(solution.b1_plus_t),
        np.finfo(np.float64).tiny,
    )
    circularity_db = -20.0 * np.log10(
        np.maximum(counter_ratio, 1.0e-6),
    )
    outside_cage = xx**2 + yy**2 >= (0.98 * geometry.radius) ** 2
    b1_plus_ut = np.ma.masked_where(outside_cage, b1_plus_ut)
    deviation_percent = np.ma.masked_where(outside_cage, deviation_percent)
    circularity_db = np.ma.masked_where(outside_cage, circularity_db)

    figure = plt.figure(figsize=(15, 8), constrained_layout=True)
    geometry_axis = figure.add_subplot(2, 3, 1, projection="3d")
    _plot_geometry(geometry_axis, geometry)

    current_axis = figure.add_subplot(2, 3, 2)
    azimuth_deg = np.rad2deg(geometry.azimuth_rad)
    current_axis.plot(
        azimuth_deg,
        np.real(cosine_mode.rung_currents_a),
        "o-",
        label="cosine mode",
    )
    current_axis.plot(
        azimuth_deg,
        np.real(sine_mode.rung_currents_a),
        "s-",
        label="sine mode",
    )
    current_axis.plot(
        azimuth_deg,
        np.abs(quadrature_mode.rung_currents_a),
        "k--",
        label="quadrature magnitude",
    )
    current_axis.set(
        xlabel="Rung azimuth (degree)",
        ylabel="Rung current (A)",
        title="Fundamental current modes",
    )
    current_axis.grid(alpha=0.25)
    current_axis.legend(fontsize=8)

    polarization_axis = figure.add_subplot(2, 3, 3)
    _center_polarization_plot(
        polarization_axis,
        cosine_center,
        sine_center,
        quadrature_center,
    )

    extent_cm = (
        coordinate[0] * 100.0,
        coordinate[-1] * 100.0,
        coordinate[0] * 100.0,
        coordinate[-1] * 100.0,
    )
    field_axis = figure.add_subplot(2, 3, 4)
    field_artist = field_axis.imshow(
        b1_plus_ut.T,
        origin="lower",
        extent=extent_cm,
        cmap="viridis",
    )
    field_axis.contour(
        xx.T * 100.0,
        yy.T * 100.0,
        roi.T.astype(float),
        levels=(0.5,),
        colors="white",
        linewidths=1.0,
    )
    field_axis.set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title=r"Quadrature $|B_1^+|$ ($\mu$T/A)",
        aspect="equal",
    )
    figure.colorbar(field_artist, ax=field_axis)

    uniformity_axis = figure.add_subplot(2, 3, 5)
    limit = max(
        2.0,
        float(np.percentile(np.abs(deviation_percent.compressed()), 98.0)),
    )
    uniformity_artist = uniformity_axis.imshow(
        deviation_percent.T,
        origin="lower",
        extent=extent_cm,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    uniformity_axis.set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title="Deviation from central-ROI mean (%)",
        aspect="equal",
    )
    figure.colorbar(uniformity_artist, ax=uniformity_axis)

    circularity_axis = figure.add_subplot(2, 3, 6)
    circularity_artist = circularity_axis.imshow(
        circularity_db.T,
        origin="lower",
        extent=extent_cm,
        cmap="magma",
        vmin=20.0,
        vmax=60.0,
    )
    circularity_axis.set(
        xlabel="x (cm)",
        ylabel="y (cm)",
        title=r"Co-/counter-rotating B1 ratio (dB)",
        aspect="equal",
    )
    figure.colorbar(circularity_artist, ax=circularity_axis)
    figure.suptitle(
        "Phase 5 birdcage reference: geometry, degenerate modes, and quadrature"
    )

    if args.output is None:
        plt.show()
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=160)
        plt.close(figure)

    cosine_norm = float(np.linalg.norm(cosine_center))
    sine_norm = float(np.linalg.norm(sine_center))
    degeneracy_error = abs(cosine_norm - sine_norm) / cosine_norm
    orthogonality = abs(np.vdot(cosine_center, sine_center)) / (
        cosine_norm * sine_norm
    )
    print(f"maximum end-ring KCL residual: {quadrature_mode.max_kcl_residual_a:.3e} A")
    print(f"fundamental center-field degeneracy error: {degeneracy_error:.3e}")
    print(f"fundamental center-field normalized inner product: {orthogonality:.3e}")
    print(f"central-ROI B1+ coefficient of variation: {100.0 * metrics.coefficient_of_variation:.3f} percent")
    print(f"central-ROI peak-to-peak nonuniformity: {100.0 * metrics.peak_to_peak_nonuniformity:.3f} percent")
    print(f"central-ROI circular isolation: {metrics.circularity_db:.3f} dB")


if __name__ == "__main__":
    main()
