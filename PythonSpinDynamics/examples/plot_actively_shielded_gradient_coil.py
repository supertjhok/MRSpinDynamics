"""Design an actively shielded XYZ gradient set and predict its transients.

The x/y saddle coils share one primary cylinder and one active-shield cylinder;
the z rings use nested layers.  A larger thin conducting cylinder is reduced to
one geometry-derived dominant eddy-current mode per gradient axis.  The plot
shows all six stream functions and the predicted time-domain response before
and after active shielding and pre-emphasis.

Run with ``--output figure.png`` to save the plot; otherwise display it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()


@dataclass(frozen=True)
class AxisDesign:
    name: str
    direction: tuple[float, float, float]
    result: object
    winding: object
    shell_mode: object
    primary_driver: object
    active_driver: object


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xy-primary-radius-cm", type=float, default=5.0)
    parser.add_argument("--z-primary-radius-cm", type=float, default=4.6)
    parser.add_argument("--z-shield-radius-cm", type=float, default=6.1)
    parser.add_argument("--xy-shield-radius-cm", type=float, default=6.5)
    parser.add_argument("--control-radius-cm", type=float, default=7.2)
    parser.add_argument("--conductor-radius-cm", type=float, default=8.0)
    parser.add_argument("--primary-length-cm", type=float, default=12.0)
    parser.add_argument("--shield-length-cm", type=float, default=15.0)
    parser.add_argument("--control-length-cm", type=float, default=18.0)
    parser.add_argument("--conductor-length-cm", type=float, default=20.0)
    parser.add_argument("--conductor-thickness-mm", type=float, default=1.5)
    parser.add_argument(
        "--conductor-resistivity",
        type=float,
        default=2.82e-8,
        help="outer-cylinder resistivity in ohm.m (default: aluminium)",
    )
    parser.add_argument("--target-radius-cm", type=float, default=1.5)
    parser.add_argument("--gradient-mt-m", type=float, default=10.0)
    parser.add_argument("--n-phi", type=int, default=24)
    parser.add_argument("--n-z", type=int, default=13)
    parser.add_argument("--turns-per-primary-peak", type=int, default=8)
    parser.add_argument("--shell-mode-contours", type=int, default=5)
    parser.add_argument("--shield-weight", type=float, default=0.1)
    parser.add_argument("--tau-rl-us", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _surface(radius_cm: float, length_cm: float, n_phi: int, n_z: int):
    from spin_dynamics.fields import CylindricalWindingSurface

    return CylindricalWindingSurface(
        radius_cm * 1.0e-2,
        length_cm * 1.0e-2,
        int(n_phi),
        int(n_z),
    )


def _stream_panel(ax, result, winding_set, title: str):
    surface = winding_set.surface
    stream = (
        result.primary_stream_function_a
        if surface is result.system.primary_surface
        else result.shield_stream_function_a
    )
    z_axis = (
        result.primary_stream_z
        if surface is result.system.primary_surface
        else result.shield_stream_z
    )
    phi = np.rad2deg(np.r_[surface.phi, 2.0 * np.pi])
    periodic = np.vstack([stream, stream[0:1]])
    image = ax.pcolormesh(
        phi,
        100.0 * z_axis,
        periodic.T,
        shading="auto",
        cmap="coolwarm",
    )
    for contour in winding_set.contours:
        path = contour.phi_z.copy()
        path_phi = np.rad2deg(path[:, 0])
        path_z = 100.0 * path[:, 1]
        jumps = np.flatnonzero(np.abs(np.diff(path_phi)) > 180.0) + 1
        for indices in np.split(np.arange(path_phi.size), jumps):
            if indices.size > 1:
                ax.plot(path_phi[indices], path_z[indices], color="black", lw=0.35)
    ax.set_title(title)
    ax.set_xlim(0.0, 360.0)
    ax.set_xlabel("azimuth (degrees)")
    ax.set_ylabel("z (cm)")
    return image


def _causal_preemphasis(driver, target: np.ndarray, dt: float, limit: float = 4.0):
    """Causally invert the driver's first-order cascade, then apply a bound."""

    desired = np.asarray(target, dtype=np.float64).copy()
    for alpha, tau in reversed(driver.eddy_terms):
        decay = float(np.exp(-dt / tau))
        state = 0.0
        inverse = np.empty_like(desired)
        denominator = 1.0 - alpha * decay
        for index, value in enumerate(desired):
            command = (value - alpha * decay * state) / denominator
            inverse[index] = command
            state = decay * state + (1.0 - decay) * command
        desired = inverse
    decay = float(np.exp(-dt / driver.tau_rl))
    command = np.empty_like(desired)
    previous_output = 0.0
    for index, value in enumerate(desired):
        command[index] = (value - decay * previous_output) / (1.0 - decay)
        previous_output = value
    return np.clip(command, -float(limit), float(limit))


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields import (
        build_actively_shielded_gradient_system,
        build_cylindrical_gradient_system,
        cylindrical_shell_eddy_mode,
        cylindrical_shield_points,
        extract_actively_shielded_winding,
        linear_gradient_target,
        solve_actively_shielded_gradient_coil,
        solve_gradient_coil,
        spherical_target_points,
    )

    primary_xy = _surface(
        args.xy_primary_radius_cm,
        args.primary_length_cm,
        args.n_phi,
        args.n_z,
    )
    shield_xy = _surface(
        args.xy_shield_radius_cm,
        args.shield_length_cm,
        args.n_phi,
        args.n_z,
    )
    primary_z = _surface(
        args.z_primary_radius_cm,
        args.primary_length_cm,
        args.n_phi,
        args.n_z,
    )
    shield_z = _surface(
        args.z_shield_radius_cm,
        args.shield_length_cm,
        args.n_phi,
        args.n_z,
    )
    outer_conductor = _surface(
        args.conductor_radius_cm,
        args.conductor_length_cm,
        max(int(args.n_phi), 24),
        int(args.n_z) + 2,
    )
    target_points = spherical_target_points(
        args.target_radius_cm * 1.0e-2,
        points_per_axis=5,
    )
    control_points = cylindrical_shield_points(
        args.control_radius_cm * 1.0e-2,
        args.control_length_cm * 1.0e-2,
        n_phi=max(int(args.n_phi), 16),
        n_z=17,
    )
    transverse_system = build_actively_shielded_gradient_system(
        primary_xy,
        shield_xy,
        target_points,
        control_points,
    )
    longitudinal_system = build_actively_shielded_gradient_system(
        primary_z,
        shield_z,
        target_points,
        control_points,
    )
    conductor_system = build_cylindrical_gradient_system(
        outer_conductor,
        target_points,
    )

    axis_specs = (
        ("x", (1.0, 0.0, 0.0), transverse_system),
        ("y", (0.0, 1.0, 0.0), transverse_system),
        ("z", (0.0, 0.0, 1.0), longitudinal_system),
    )
    designs: list[AxisDesign] = []
    for name, direction, system in axis_specs:
        gradient = tuple(
            args.gradient_mt_m * 1.0e-3 * component for component in direction
        )
        target = linear_gradient_target(target_points, gradient)
        result = solve_actively_shielded_gradient_coil(
            system,
            target,
            regularization=1.0e-15,
            shield_weights=args.shield_weight,
        )
        primary_peak = float(np.max(np.abs(result.primary_stream_function_a)))
        turn_current = primary_peak / max(int(args.turns_per_primary_peak), 1)
        winding = extract_actively_shielded_winding(
            result,
            current_per_turn_a=turn_current,
        )

        conductor_result = solve_gradient_coil(
            conductor_system,
            target,
            regularization=1.0e-15,
        )
        conductor_peak = float(
            np.max(np.abs(conductor_result.stream_function_a))
        )
        shell_mode = cylindrical_shell_eddy_mode(
            conductor_result,
            current_per_turn_a=(
                conductor_peak / max(int(args.shell_mode_contours), 1)
            ),
            thickness_m=args.conductor_thickness_mm * 1.0e-3,
            resistivity_ohm_m=args.conductor_resistivity,
        )
        driver_options = {
            "gradient_direction": direction,
            "tau_rl": args.tau_rl_us * 1.0e-6,
            "oversample": 1,
        }
        primary_driver = shell_mode.to_gradient_driver(
            winding.primary,
            **driver_options,
        )
        active_driver = shell_mode.to_gradient_driver(
            winding,
            **driver_options,
        )
        designs.append(
            AxisDesign(
                name=name,
                direction=direction,
                result=result,
                winding=winding,
                shell_mode=shell_mode,
                primary_driver=primary_driver,
                active_driver=active_driver,
            )
        )

    dt = 5.0e-6
    samples = 440
    on = slice(20, 300)
    target_step = np.zeros(samples)
    target_step[on] = 1.0

    print("Actively shielded XYZ gradient set with outer-conductor transients")
    print("  x/y share the same primary and active-shield cylinders")
    for design in designs:
        primary_spectrum = design.shell_mode.spectrum(
            design.winding.primary,
            gradient_direction=design.direction,
        )
        active_spectrum = design.shell_mode.spectrum(
            design.winding,
            gradient_direction=design.direction,
        )
        print(
            f"  {design.name}: fit={design.result.target_relative_rms_error:.3%}, "
            f"turns={len(design.winding.primary.contours)}+"
            f"{len(design.winding.shield.contours)}, "
            f"tau_eddy={1e3 * design.shell_mode.time_constant_s:.3f} ms, "
            f"alpha primary={primary_spectrum.alpha[0]:.4f}, "
            f"active={active_spectrum.alpha[0]:.4f}"
        )
    print(
        f"  outer conductor: r={args.conductor_radius_cm:g} cm, "
        f"length={args.conductor_length_cm:g} cm, "
        f"thickness={args.conductor_thickness_mm:g} mm"
    )

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.0))
    time_ms = np.arange(samples) * dt * 1.0e3
    for column, design in enumerate(designs):
        image = _stream_panel(
            axes[0, column],
            design.result,
            design.winding.primary,
            f"{design.name.upper()} primary stream function",
        )
        fig.colorbar(image, ax=axes[0, column], label="stream function (A)")
        image = _stream_panel(
            axes[1, column],
            design.result,
            design.winding.shield,
            f"{design.name.upper()} active-shield stream function",
        )
        fig.colorbar(image, ax=axes[1, column], label="stream function (A)")

        primary_response = np.asarray(
            design.primary_driver.apply(target_step, dt, xp=np)
        )
        active_response = np.asarray(
            design.active_driver.apply(target_step, dt, xp=np)
        )
        command = _causal_preemphasis(design.active_driver, target_step, dt)
        corrected = np.asarray(design.active_driver.apply(command, dt, xp=np))
        ax = axes[2, column]
        ax.plot(time_ms, target_step, "k--", lw=1.0, label="target")
        ax.plot(time_ms, primary_response, lw=1.2, label="primary only")
        ax.plot(time_ms, active_response, lw=1.3, label="active shield")
        ax.plot(time_ms, corrected, lw=1.1, label="active + pre-emphasis")
        active_alpha = sum(alpha for alpha, _tau in design.active_driver.eddy_terms)
        ax.set_title(
            f"{design.name.upper()} response: "
            f"tau={1e3 * design.shell_mode.time_constant_s:.2f} ms, "
            f"alpha={active_alpha:.3f}"
        )
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("normalized gradient")
        ax.set_ylim(-0.08, 1.18)
        ax.grid(True, alpha=0.22)
        if column == 0:
            ax.legend(fontsize=8, loc="lower center")

    fig.suptitle(
        "Complete actively shielded XYZ set with a conducting outer cylinder\n"
        "x/y share cylindrical layers; z uses a nested ring layer",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170, bbox_inches="tight")
        print(f"  saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
