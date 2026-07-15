"""Design, realize, and assess an actively shielded z-gradient coil.

The primary and shield cylinders are solved jointly.  The example then extracts
physical winding contours, reports electrical and force/torque estimates, and
compares the continuous active design with an otherwise identical unshielded
primary coil.  Run with ``--output figure.png`` to save the plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-radius-cm", type=float, default=5.0)
    parser.add_argument("--shield-radius-cm", type=float, default=6.5)
    parser.add_argument("--primary-length-cm", type=float, default=12.0)
    parser.add_argument("--shield-length-cm", type=float, default=15.0)
    parser.add_argument("--target-radius-cm", type=float, default=1.5)
    parser.add_argument("--sample-shell-radius-cm", type=float, default=8.5)
    parser.add_argument("--sample-shell-length-cm", type=float, default=18.0)
    parser.add_argument("--gradient-mt-m", type=float, default=10.0)
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--n-z", type=int, default=13)
    parser.add_argument("--turns-per-polarity", type=int, default=8)
    parser.add_argument("--wire-diameter-mm", type=float, default=1.0)
    parser.add_argument("--b0-t", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _stream_panel(ax, surface, stream, z, title: str):
    phi = np.rad2deg(np.r_[surface.phi, 2.0 * np.pi])
    periodic = np.vstack([stream, stream[0:1]])
    image = ax.pcolormesh(
        phi,
        100.0 * z,
        periodic.T,
        shading="auto",
        cmap="coolwarm",
    )
    ax.set_title(title)
    ax.set_xlabel("azimuth (degrees)")
    ax.set_ylabel("z (cm)")
    return image


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields import (
        CylindricalWindingSurface,
        build_actively_shielded_gradient_system,
        build_cylindrical_gradient_system,
        cylindrical_shield_points,
        extract_actively_shielded_winding,
        gradient_coil_engineering_metrics,
        linear_gradient_target,
        solve_actively_shielded_gradient_coil,
        solve_gradient_coil,
        spherical_target_points,
    )

    primary = CylindricalWindingSurface(
        args.primary_radius_cm * 1.0e-2,
        args.primary_length_cm * 1.0e-2,
        int(args.n_phi),
        int(args.n_z),
    )
    shield = CylindricalWindingSurface(
        args.shield_radius_cm * 1.0e-2,
        args.shield_length_cm * 1.0e-2,
        int(args.n_phi),
        int(args.n_z),
    )
    target_points = spherical_target_points(
        args.target_radius_cm * 1.0e-2,
        points_per_axis=5,
    )
    shell_points = cylindrical_shield_points(
        args.sample_shell_radius_cm * 1.0e-2,
        args.sample_shell_length_cm * 1.0e-2,
        n_phi=max(int(args.n_phi), 16),
        n_z=17,
    )
    target = linear_gradient_target(
        target_points,
        gradient=(0.0, 0.0, args.gradient_mt_m * 1.0e-3),
    )
    system = build_actively_shielded_gradient_system(
        primary,
        shield,
        target_points,
        shell_points,
    )
    result = solve_actively_shielded_gradient_coil(
        system,
        target,
        regularization=1.0e-15,
        shield_weights=5.0,
    )
    unshielded_system = build_cylindrical_gradient_system(primary, target_points)
    unshielded = solve_gradient_coil(
        unshielded_system,
        target,
        regularization=1.0e-15,
    )
    unshielded_shell = (
        system.shield_sensitivity[:, : primary.segment_count]
        @ unshielded.segment_currents_a.ravel()
    )

    stream_peak = min(
        float(np.max(np.abs(result.primary_stream_function_a))),
        float(np.max(np.abs(result.shield_stream_function_a))),
    )
    current_per_turn = stream_peak / max(int(args.turns_per_polarity), 1)
    winding = extract_actively_shielded_winding(
        result,
        current_per_turn_a=current_per_turn,
    )
    metrics = gradient_coil_engineering_metrics(
        winding,
        target_points,
        target_field_t=target,
        shield_points=shell_points,
        wire_radius=0.5e-3 * args.wire_diameter_mm,
        background_field=(0.0, 0.0, args.b0_t),
    )
    suppression = float(
        np.sqrt(np.mean(unshielded_shell**2))
        / result.shield_weighted_rms_field_t
    )

    print("Actively shielded stream-function gradient-coil design")
    print(f"  target relative RMS error: {result.target_relative_rms_error:.3%}")
    print(f"  continuous-sheet exterior suppression: {suppression:.3g} x")
    print(
        f"  realized winding: {metrics.electrical.contour_count} contours, "
        f"{metrics.electrical.wire_length_m:.3f} m wire"
    )
    print(
        f"  Rdc={metrics.electrical.dc_resistance_ohm:.4f} ohm, "
        f"Lest={1.0e6 * metrics.electrical.estimated_inductance_h:.2f} uH, "
        f"P={metrics.electrical.ohmic_power_w:.3f} W"
    )
    print(
        f"  net force={np.linalg.norm(metrics.mechanical.net_force_n):.3e} N, "
        f"net torque={np.linalg.norm(metrics.mechanical.net_torque_nm):.3e} N m"
    )

    fig = plt.figure(figsize=(13.5, 9.0))
    grid = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.28)
    ax_primary = fig.add_subplot(grid[0, 0])
    image = _stream_panel(
        ax_primary,
        primary,
        result.primary_stream_function_a,
        result.primary_stream_z,
        "Primary stream function",
    )
    fig.colorbar(image, ax=ax_primary, label="stream function (A)")

    ax_shield = fig.add_subplot(grid[0, 1])
    image = _stream_panel(
        ax_shield,
        shield,
        result.shield_stream_function_a,
        result.shield_stream_z,
        "Shield stream function",
    )
    fig.colorbar(image, ax=ax_shield, label="stream function (A)")

    ax_winding = fig.add_subplot(grid[1, 0], projection="3d")
    for winding_set, color in (
        (winding.primary, "C0"),
        (winding.shield, "C3"),
    ):
        for contour in winding_set.contours:
            ax_winding.plot(*(100.0 * contour.points).T, color=color, lw=0.75)
    ax_winding.set_title("Realized primary (blue) and shield (red)")
    ax_winding.set_xlabel("x (cm)")
    ax_winding.set_ylabel("y (cm)")
    ax_winding.set_zlabel("z (cm)")
    ax_winding.set_box_aspect((1.0, 1.0, shield.length / (2.0 * shield.radius)))
    ax_winding.view_init(elev=22.0, azim=-58.0)

    ax_leak = fig.add_subplot(grid[1, 1])
    polar_z = shell_points[:, 2] / (0.5 * args.sample_shell_length_cm * 1.0e-2)
    order = np.argsort(polar_z)
    ax_leak.plot(
        polar_z[order],
        1.0e6 * unshielded_shell[order],
        "o-",
        ms=3,
        lw=1.0,
        label="unshielded primary",
    )
    ax_leak.plot(
        polar_z[order],
        1.0e6 * result.predicted_shield_field_t[order],
        "o-",
        ms=3,
        lw=1.0,
        label="active primary + shield",
    )
    ax_leak.set_title(f"Exterior field: {suppression:.2g}x RMS suppression")
    ax_leak.set_xlabel("shell coordinate z/r")
    ax_leak.set_ylabel("Bz (uT)")
    ax_leak.grid(True, alpha=0.22)
    ax_leak.legend(fontsize=8)

    fig.suptitle(
        f"{args.gradient_mt_m:g} mT/m actively shielded z-gradient",
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
