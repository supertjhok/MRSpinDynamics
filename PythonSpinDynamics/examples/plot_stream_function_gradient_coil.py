"""Design and plot a cylindrical MRI gradient coil from its stream function.

The example constructs a transverse y-gradient target, reuses one Biot-Savart
sensitivity matrix for an L-curve regularization sweep, extracts equally spaced
periodic stream-function contours, and verifies the resulting thin-wire winding
against the target field. The four panels show the regularization trade-off, the
unwrapped stream function, the 3-D winding paths, and target-versus-realized Bz.

Run with ``--output figure.png`` to save the plot; otherwise display it.
"""
# Follow the example from physical inputs through simulation to plotted diagnostics.
# Quantities use SI units unless a variable name or CLI help states otherwise.


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-cm", type=float, default=8.0)
    parser.add_argument("--length-cm", type=float, default=18.0)
    parser.add_argument("--target-radius-cm", type=float, default=3.0)
    parser.add_argument("--gradient-mt-m", type=float, default=10.0)
    parser.add_argument("--n-phi", type=int, default=32)
    parser.add_argument("--n-z", type=int, default=17)
    parser.add_argument("--target-points", type=int, default=7)
    parser.add_argument("--path-points", type=int, default=23)
    parser.add_argument("--turns-per-polarity", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _plot_unwrapped(ax, contour, **kwargs) -> None:
    path = contour.phi_z.copy()
    phi = np.rad2deg(path[:, 0])
    z_cm = 100.0 * path[:, 1]
    jumps = np.flatnonzero(np.abs(np.diff(phi)) > 180.0) + 1
    for indices in np.split(np.arange(phi.size), jumps):
        if indices.size > 1:
            ax.plot(phi[indices], z_cm[indices], **kwargs)


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parse_args()
    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields import (
        CylindricalWindingSurface,
        build_cylindrical_gradient_system,
        extract_winding_contours,
        linear_gradient_target,
        solve_regularization_path,
        spherical_target_points,
        winding_segments,
    )
    from spin_dynamics.fields.magnetostatics import biot_savart

    surface = CylindricalWindingSurface(
        radius=args.radius_cm * 1.0e-2,
        length=args.length_cm * 1.0e-2,
        n_phi=int(args.n_phi),
        n_z=int(args.n_z),
    )
    target_points = spherical_target_points(
        args.target_radius_cm * 1.0e-2,
        points_per_axis=int(args.target_points),
    )
    target = linear_gradient_target(
        target_points,
        gradient=(0.0, args.gradient_mt_m * 1.0e-3, 0.0),
    )
    system = build_cylindrical_gradient_system(surface, target_points)
    regularizations = np.logspace(-20.0, -9.0, int(args.path_points))
    path = solve_regularization_path(system, target, regularizations)
    result = path.selected_result

    peak_stream = float(np.max(np.abs(result.stream_function_a)))
    current_per_turn = peak_stream / max(int(args.turns_per_polarity), 1)
    contours = extract_winding_contours(
        result,
        current_per_turn_a=current_per_turn,
    )
    segments = winding_segments(contours)
    winding_bz = biot_savart(
        target_points,
        segments,
        current=current_per_turn,
    )[:, 2]
    winding_correlation = float(np.corrcoef(target, winding_bz)[0, 1])
    winding_scale = float(np.dot(winding_bz, target) / np.dot(target, target))

    print("Cylindrical stream-function gradient-coil design")
    print(
        f"  source mesh: {surface.n_phi} x {surface.n_z} = "
        f"{surface.segment_count} azimuthal elements"
    )
    print(f"  target points: {target_points.shape[0]}")
    print(f"  selected alpha: {path.selected_regularization:.3e} T^2/A^2")
    print(f"  continuous-current relative RMS error: {result.relative_rms_error:.3%}")
    print(
        f"  extracted contours: {len(contours)}, "
        f"current/turn={current_per_turn:.3f} A"
    )
    print(
        f"  winding field: correlation={winding_correlation:.6f}, "
        f"gradient scale={winding_scale:.4f}"
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig = plt.figure(figsize=(13.5, 9.0))
    grid = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.28)

    ax_l = fig.add_subplot(grid[0, 0])
    ax_l.loglog(
        path.weighted_residual_norms_t,
        path.current_norms_a,
        "o-",
        color="C0",
        lw=1.4,
        ms=4,
    )
    ax_l.plot(
        path.weighted_residual_norms_t[path.selected_index],
        path.current_norms_a[path.selected_index],
        "o",
        color="C3",
        ms=9,
        label=f"L-curve corner, alpha={path.selected_regularization:.1e}",
    )
    ax_l.set_title("Regularization trade-off")
    ax_l.set_xlabel("weighted field residual norm (T)")
    ax_l.set_ylabel("segment-current norm (A)")
    ax_l.grid(True, which="both", alpha=0.22)
    ax_l.legend(fontsize=8)

    ax_psi = fig.add_subplot(grid[0, 1])
    phi_degrees = np.rad2deg(np.r_[surface.phi, 2.0 * np.pi])
    psi_periodic = np.vstack(
        [result.stream_function_a, result.stream_function_a[0:1]]
    )
    image = ax_psi.pcolormesh(
        phi_degrees,
        100.0 * result.stream_z,
        psi_periodic.T,
        shading="auto",
        cmap="coolwarm",
    )
    for contour in contours:
        _plot_unwrapped(ax_psi, contour, color="black", lw=0.8)
    ax_psi.set_title("Stream function and winding contours")
    ax_psi.set_xlabel("azimuth (degrees)")
    ax_psi.set_ylabel("z (cm)")
    fig.colorbar(image, ax=ax_psi, label="stream function (A)")

    ax_3d = fig.add_subplot(grid[1, 0], projection="3d")
    for contour in contours:
        ax_3d.plot(
            100.0 * contour.points[:, 0],
            100.0 * contour.points[:, 1],
            100.0 * contour.points[:, 2],
            color="C0" if contour.level_a > 0.0 else "C3",
            lw=1.0,
        )
    ax_3d.set_title("Extracted cylindrical windings")
    ax_3d.set_xlabel("x (cm)")
    ax_3d.set_ylabel("y (cm)")
    ax_3d.set_zlabel("z (cm)")
    ax_3d.set_box_aspect((1.0, 1.0, surface.length / (2.0 * surface.radius)))
    ax_3d.view_init(elev=22.0, azim=-58.0)

    ax_fit = fig.add_subplot(grid[1, 1])
    limit = max(float(np.max(np.abs(target))), float(np.max(np.abs(winding_bz))))
    ax_fit.plot([-limit, limit], [-limit, limit], "k--", lw=1.0, label="ideal")
    ax_fit.scatter(target, result.predicted_field_t, s=20, alpha=0.7, label="optimized sheet")
    ax_fit.scatter(target, winding_bz, s=16, alpha=0.65, label="contour winding")
    ax_fit.set_title("Target-field realization")
    ax_fit.set_xlabel("target Bz (T)")
    ax_fit.set_ylabel("realized Bz (T)")
    ax_fit.grid(True, alpha=0.22)
    ax_fit.legend(fontsize=8)

    fig.suptitle(
        f"{args.gradient_mt_m:g} mT/m y-gradient on a "
        f"{args.radius_cm:g} cm-radius cylinder",
        fontsize=14,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170, bbox_inches="tight")
        print(f"  saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
