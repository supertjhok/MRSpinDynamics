"""Plot hybrid EPM array field bases and constrained operating modes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spin_dynamics.fields import (  # noqa: E402
    illustrative_hybrid_epm_array,
    synthesize_field_off_state,
    synthesize_transport_state,
    synthesize_uniform_imaging_state,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the illustrative 18-sub-unit/72-element hybrid EPM array "
            "and synthesize imaging, field-off, and transport states."
        )
    )
    parser.add_argument(
        "--panel-gap-mm",
        type=float,
        default=150.0,
        help="Opposing panel center-to-center gap (default: 150 mm).",
    )
    parser.add_argument(
        "--imaging-field-mt",
        type=float,
        default=2.0,
        help="Uniform imaging-mode Bz target (default: 2 mT).",
    )
    parser.add_argument(
        "--transport-gradient-mt-per-m",
        type=float,
        default=50.0,
        help="Transport-mode dBz/dx target (default: 50 mT/m).",
    )
    parser.add_argument(
        "--map-points",
        type=int,
        default=41,
        help="Points per in-plane map axis (default: 41).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "numpy", "scipy"),
        default="auto",
        help="Bounded least-squares backend (default: auto).",
    )
    parser.add_argument("--output", type=Path, help="Optional output image path.")
    return parser


def _validate(args: argparse.Namespace) -> None:
    if not np.isfinite(args.panel_gap_mm) or args.panel_gap_mm <= 80.0:
        raise ValueError("--panel-gap-mm must exceed 80 mm")
    if not np.isfinite(args.imaging_field_mt) or args.imaging_field_mt <= 0.0:
        raise ValueError("--imaging-field-mt must be positive")
    if not np.isfinite(args.transport_gradient_mt_per_m):
        raise ValueError("--transport-gradient-mt-per-m must be finite")
    if args.map_points < 21:
        raise ValueError("--map-points must be at least 21")


def _roi_points(radius_m: float = 0.020, points_per_axis: int = 5) -> np.ndarray:
    axis = np.linspace(-radius_m, radius_m, points_per_axis)
    return np.asarray(
        [
            (x, y, z)
            for x in axis
            for y in axis
            for z in axis
            if x * x + y * y + z * z <= radius_m**2 + 1e-15
        ]
    )


def _mode_map(basis, result, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    field = basis.field_vectors(result.remanence_t).reshape(*shape, 3)
    return field, field[..., 2]


def main() -> None:
    args = _parser().parse_args()
    _validate(args)
    array = illustrative_hybrid_epm_array(panel_gap_m=args.panel_gap_mm * 1e-3)
    roi_points = _roi_points()
    roi_basis = array.build_field_basis(roi_points, n_cross=3, n_length=9)
    common = {
        "field_direction": (0.0, 0.0, 1.0),
        "regularization": 1e-10,
        "backend": args.backend,
        "tolerance_t": 1e-11,
    }
    imaging = synthesize_uniform_imaging_state(
        roi_basis,
        args.imaging_field_mt * 1e-3,
        **common,
    )
    field_off = synthesize_field_off_state(roi_basis, **common)
    transport = synthesize_transport_state(
        roi_basis,
        bias_field_t=args.imaging_field_mt * 1e-3,
        gradient_t_per_m=(args.transport_gradient_mt_per_m * 1e-3, 0.0, 0.0),
        center_m=(0.0, 0.0, 0.0),
        **common,
    )

    axis = np.linspace(-0.030, 0.030, args.map_points)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    plane_points = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    plane_basis = array.build_field_basis(plane_points, n_cross=3, n_length=9)
    imaging_field, imaging_bz = _mode_map(plane_basis, imaging, xx.shape)
    off_field, off_bz = _mode_map(plane_basis, field_off, xx.shape)
    transport_field, transport_bz = _mode_map(plane_basis, transport, xx.shape)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Hybrid electropermanent array: cached field basis and operating modes", fontsize=16)

    geometry = axes[0, 0]
    half_gap_mm = 0.5 * args.panel_gap_mm
    geometry.add_patch(Rectangle((-52, -half_gap_mm - 8), 104, 16, color="0.35", alpha=0.35))
    geometry.add_patch(Rectangle((-52, half_gap_mm - 8), 104, 16, color="0.35", alpha=0.35))
    for z_mm in (-half_gap_mm, half_gap_mm):
        for x_mm in (-35.0, 0.0, 35.0):
            geometry.plot(x_mm, z_mm, "s", color="tab:purple", markersize=10)
            geometry.text(x_mm, z_mm + (12 if z_mm < 0 else -12), "3 rows", ha="center", fontsize=8)
    geometry.add_patch(Circle((0.0, 0.0), 20.0, fill=False, color="tab:green", linewidth=2))
    geometry.text(0.0, 0.0, "40 mm\ncontrol ROI", ha="center", va="center", color="tab:green")
    geometry.annotate(
        f"{args.panel_gap_mm:.0f} mm gap",
        xy=(48, -half_gap_mm),
        xytext=(48, half_gap_mm),
        arrowprops={"arrowstyle": "<->"},
        ha="center",
        va="center",
    )
    geometry.set(
        xlim=(-62, 62),
        ylim=(-half_gap_mm - 25, half_gap_mm + 25),
        xlabel="x (mm)",
        ylabel="z (mm)",
        title="two 3x3 panels; each square = fixed NdFeB + 4 AlNiCo",
    )
    geometry.set_aspect("equal")
    geometry.grid(alpha=0.2)

    extent = 1e3 * np.asarray([axis[0], axis[-1], axis[0], axis[-1]])
    image_plot = axes[0, 1].imshow(1e3 * imaging_bz, origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].contour(1e3 * xx, 1e3 * yy, 1e3 * imaging_bz, colors="white", linewidths=0.6)
    axes[0, 1].set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"imaging mode Bz (mT); {imaging.homogeneity_ppm:.0f} ppm ROI",
    )
    fig.colorbar(image_plot, ax=axes[0, 1], label="Bz (mT)")

    off_plot = axes[0, 2].imshow(1e6 * off_bz, origin="lower", extent=extent, cmap="coolwarm")
    axes[0, 2].set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=f"field-off residual Bz (uT); RMS {1e6 * field_off.rms_error_t:.2f} uT",
    )
    fig.colorbar(off_plot, ax=axes[0, 2], label="Bz (uT)")

    magnitude_squared = np.sum(transport_field**2, axis=-1)
    d_by, d_bx = np.gradient(magnitude_squared, axis, axis)
    force_norm = np.hypot(d_bx, d_by)
    safe = np.where(force_norm > 0.0, force_norm, 1.0)
    transport_plot = axes[1, 0].imshow(
        1e3 * np.linalg.norm(transport_field, axis=-1),
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    stride = max(1, args.map_points // 12)
    axes[1, 0].quiver(
        1e3 * xx[::stride, ::stride],
        1e3 * yy[::stride, ::stride],
        (d_bx / safe)[::stride, ::stride],
        (d_by / safe)[::stride, ::stride],
        color="cyan",
        scale=18,
        width=0.004,
    )
    axes[1, 0].set(
        xlabel="x (mm)",
        ylabel="y (mm)",
        title="transport |B| (mT) and normalized grad(|B|^2)",
    )
    fig.colorbar(transport_plot, ax=axes[1, 0], label="|B| (mT)")

    states = axes[1, 1]
    channels = np.arange(72)
    states.plot(channels, imaging.remanence_t, ".", label="imaging", alpha=0.8)
    states.plot(channels, field_off.remanence_t, ".", label="field off", alpha=0.8)
    states.plot(channels, transport.remanence_t, ".", label="transport", alpha=0.8)
    states.axhline(0.33, color="0.3", linestyle=":")
    states.axhline(-0.33, color="0.3", linestyle=":")
    for boundary in (36,):
        states.axvline(boundary - 0.5, color="0.5", linestyle="--")
    states.set(
        xlabel="programmable AlNiCo channel",
        ylabel="synthesized retained Br (T)",
        title="bounded 72-element operating states",
        ylim=(-0.36, 0.36),
    )
    states.legend(fontsize=8)
    states.grid(alpha=0.2)

    profile = axes[1, 2]
    center_row = args.map_points // 2
    transport_target = args.imaging_field_mt + args.transport_gradient_mt_per_m * axis
    profile.plot(1e3 * axis, 1e3 * imaging_bz[center_row], label="imaging achieved")
    profile.plot(1e3 * axis, np.full_like(axis, args.imaging_field_mt), "--", label="imaging target")
    profile.plot(1e3 * axis, 1e3 * off_bz[center_row], label="field-off achieved")
    profile.plot(1e3 * axis, 1e3 * transport_bz[center_row], label="transport achieved")
    profile.plot(1e3 * axis, transport_target, "--", label="transport target")
    profile.set(
        xlabel="x at y=z=0 (mm)",
        ylabel="Bz (mT)",
        title="target and achieved centerline fields",
    )
    profile.legend(fontsize=7, ncol=2)
    profile.grid(alpha=0.25)

    fit = np.linalg.lstsq(
        np.column_stack((np.ones(roi_points.shape[0]), roi_points)),
        transport.predicted_projected_field_t,
        rcond=None,
    )[0]
    achieved_gradient = fit[1:]
    basis_megabytes = (
        roi_basis.fixed_field_t.nbytes + roi_basis.programmable_field_t_per_t.nbytes
    ) / 2**20
    print("Hybrid EPM array state synthesis")
    print(f"  hierarchy: {len(array.subunits)} sub-units, {len(array.programmable_elements)} controls")
    print(f"  evidence: hierarchy={array.evidence[0].classification}, dimensions={array.evidence[1].classification}")
    print(f"  cached ROI basis: {roi_points.shape[0]} points, {basis_megabytes:.3f} MiB")
    print(
        f"  imaging: mean={1e3 * imaging.mean_projected_field_t:.4f} mT, "
        f"RMS error={1e6 * imaging.rms_error_t:.2f} uT, "
        f"homogeneity={imaging.homogeneity_ppm:.0f} ppm"
    )
    print(f"  field off: RMS residual={1e6 * field_off.rms_error_t:.3f} uT")
    print(
        "  transport achieved gradient="
        f"({1e3 * achieved_gradient[0]:.2f}, {1e3 * achieved_gradient[1]:.2f}, "
        f"{1e3 * achieved_gradient[2]:.2f}) mT/m"
    )
    print(
        "  saturated controls (imaging/off/transport)="
        f"{np.count_nonzero(np.isclose(np.abs(imaging.remanence_t), 0.33, atol=1e-5))}/"
        f"{np.count_nonzero(np.isclose(np.abs(field_off.remanence_t), 0.33, atol=1e-5))}/"
        f"{np.count_nonzero(np.isclose(np.abs(transport.remanence_t), 0.33, atol=1e-5))}"
    )
    print(
        "  synthesis backends="
        f"{imaging.backend}/{field_off.backend}/{transport.backend}; "
        f"basis condition={imaging.condition_number:.2e}"
    )
    print("  next: nonlinear encoding and nanoparticle force/trajectory integration")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
