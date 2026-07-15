"""Model AlNiCo electropermanent rods and a close-packed magnet bundle.

This first EPM example is intentionally magnetostatic.  It validates the hard
part needed before pulse programming is credible: retained remanence is kept
separate from nominal material Br, individual rods remain explicit, and a
bundle may be compared with a faster area-equivalent cylinder.

The six panels show geometry, partial-remanence states, a B0 map, the bundle
reduction error, the field contribution from a neighboring bundle, and
cubature convergence against the exact finite-cylinder axial field.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the static field, provenance, bundle reduction, and numerical "
            "convergence of a documented 37-rod AlNiCo electropermanent magnet."
        )
    )
    parser.add_argument(
        "--remanence-t",
        type=float,
        default=0.42,
        help="Illustrative retained remanence of every bundle rod (T).",
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=61,
        help="Grid samples per dimension in the bundle field map.",
    )
    parser.add_argument(
        "--n-cross",
        type=int,
        default=3,
        help="Cubature samples across each rod diameter.",
    )
    parser.add_argument(
        "--n-length",
        type=int,
        default=13,
        help="Cubature samples along each rod.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path. If omitted, show the plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.pixels < 11:
        raise ValueError("--pixels must be at least 11")
    if not 0.0 <= args.remanence_t <= 1.27:
        raise ValueError("--remanence-t must lie between 0 and nominal material Br")
    plt = load_matplotlib(headless=bool(args.output))
    from matplotlib.patches import Circle

    from spin_dynamics.fields.electropermanent import (
        EvidenceRecord,
        RemanenceState,
        electropermanent_field,
        finite_cylinder_on_axis_field,
        sample_electropermanent_field,
        variable_field_nmr_rod,
        weinberg_37_rod_bundle,
    )

    state = RemanenceState(
        args.remanence_t,
        branch="partial",
        calibration_id="illustrative-static-example",
        evidence=(
            EvidenceRecord(
                source="plot_electropermanent_magnet.py input",
                classification="inferred",
                detail="Illustrative state; not a fitted pulse-programming result",
            ),
        ),
    )
    bundle = weinberg_37_rod_bundle(state=state)
    equivalent = bundle.equivalent_cylinder(label="area-equivalent bundle")
    published = variable_field_nmr_rod()

    face_z = 0.5 * bundle.rods[0].length_m
    x_axis = np.linspace(-0.03, 0.03, args.pixels)
    z_axis = np.linspace(face_z + 0.004, face_z + 0.08, args.pixels)
    field_maps = sample_electropermanent_field(
        (x_axis, z_axis),
        (bundle,),
        n_cross=args.n_cross,
        n_length=args.n_length,
        chunk_size=256,
    )

    axial_gap = np.linspace(0.001, 0.10, 121)
    axial_points = np.column_stack(
        [np.zeros_like(axial_gap), np.zeros_like(axial_gap), face_z + axial_gap]
    )
    explicit_axial = electropermanent_field(
        axial_points,
        (bundle,),
        n_cross=max(3, args.n_cross),
        n_length=max(15, args.n_length),
        chunk_size=128,
    )[:, 2]
    equivalent_axial = finite_cylinder_on_axis_field(
        face_z + axial_gap,
        equivalent,
    )
    relative_rms = float(
        np.sqrt(np.mean((explicit_axial - equivalent_axial) ** 2))
        / np.max(np.abs(explicit_axial))
    )

    separation = 2.6 * bundle.outer_radius_m
    primary = equivalent.with_state(state)
    primary = type(primary)(
        center_m=(-0.5 * separation, 0.0, 0.0),
        axis=primary.axis,
        length_m=primary.length_m,
        radius_m=primary.radius_m,
        material=primary.material,
        state=primary.state,
        coil=primary.coil,
        label="primary bundle",
        evidence=primary.evidence,
    )
    neighbor = type(primary)(
        center_m=(0.5 * separation, 0.0, 0.0),
        axis=primary.axis,
        length_m=primary.length_m,
        radius_m=primary.radius_m,
        material=primary.material,
        state=RemanenceState(-0.75 * args.remanence_t, branch="partial"),
        coil=primary.coil,
        label="opposed neighbor",
        evidence=primary.evidence,
    )
    neighbor_x = np.linspace(-0.06, 0.06, 181)
    neighbor_z = face_z + 0.012
    neighbor_points = np.column_stack(
        [neighbor_x, np.zeros_like(neighbor_x), np.full_like(neighbor_x, neighbor_z)]
    )
    primary_field = electropermanent_field(
        neighbor_points,
        (primary,),
        n_cross=7,
        n_length=31,
    )[:, 2]
    combined_field = electropermanent_field(
        neighbor_points,
        (primary, neighbor),
        n_cross=7,
        n_length=31,
    )[:, 2]
    primary_axis_index = int(np.argmin(np.abs(neighbor_x + 0.5 * separation)))
    neighbor_perturbation = float(
        (combined_field[primary_axis_index] - primary_field[primary_axis_index])
        / primary_field[primary_axis_index]
    )

    convergence_orders = np.asarray([3, 5, 7, 11])
    convergence_gap = 0.04
    convergence_position = 0.5 * published.length_m + convergence_gap
    convergence_point = np.asarray([[0.0, 0.0, convergence_position]])
    exact = finite_cylinder_on_axis_field(convergence_position, published)
    errors = []
    for order in convergence_orders:
        numerical = published.field_vectors(
            convergence_point,
            n_cross=int(order),
            n_length=int(6 * order + 1),
        )[0, 2]
        errors.append(abs((numerical - exact) / exact))
    errors = np.asarray(errors)

    surface_field = finite_cylinder_on_axis_field(
        0.5 * published.length_m + 0.001,
        published,
    )
    print("Electropermanent AlNiCo magnet benchmark")
    print(
        f"  material: {bundle.rods[0].material.name}; "
        f"retained Br={args.remanence_t:.3f} T "
        f"({state.fraction_of(bundle.rods[0].material):.1%} of nominal)"
    )
    print(
        f"  bundle: {len(bundle.rods)} rods, outer diameter "
        f"{2.0 * bundle.outer_radius_m * 1e3:.2f} mm, "
        f"moment {np.linalg.norm(bundle.magnetic_moment_am2):.2f} A m^2"
    )
    print(
        f"  published rod inferred surface field: {surface_field * 1e3:.1f} mT"
    )
    print(f"  area-equivalent cylinder axial RMS error: {relative_rms:.1%}")
    print(
        "  opposed-neighbor field contribution at primary axis: "
        f"{neighbor_perturbation:+.1%}"
    )
    print(
        "  geometry evidence: "
        f"{bundle.evidence[0].classification} — {bundle.evidence[0].source}"
    )
    print("  programming-pulse and hysteresis dynamics: not enabled in this phase")

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8))

    ax = axes[0, 0]
    for rod in bundle.rods:
        center = np.asarray(rod.center_m)
        ax.add_patch(
            Circle(
                (center[0] * 1e3, center[1] * 1e3),
                rod.radius_m * 1e3,
                facecolor="tab:orange",
                edgecolor="0.2",
                linewidth=0.6,
            )
        )
    limit = 1.15 * bundle.outer_radius_m * 1e3
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_title("documented 37-rod cross-section")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax = axes[0, 1]
    profile_gap = np.linspace(0.001, 0.08, 160)
    profile_position = 0.5 * published.length_m + profile_gap
    for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
        profile_rod = published.with_state(
            RemanenceState(
                fraction * published.state.remanence_t,
                branch="partial",
            )
        )
        profile = finite_cylinder_on_axis_field(profile_position, profile_rod)
        ax.plot(profile_gap * 1e3, profile * 1e3, label=f"{fraction:+.1f} state")
    ax.set_title("retained-state axial profiles")
    ax.set_xlabel("distance from face (mm)")
    ax.set_ylabel("B axial (mT)")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[0, 2]
    extent = [
        x_axis[0] * 1e3,
        x_axis[-1] * 1e3,
        (z_axis[0] - face_z) * 1e3,
        (z_axis[-1] - face_z) * 1e3,
    ]
    image = ax.imshow(
        (1e3 * field_maps.b0_projected).T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
    )
    ax.set_title("explicit-bundle Bz outside face (mT)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("distance from face (mm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(axial_gap * 1e3, explicit_axial * 1e3, label="37 explicit rods")
    ax.plot(
        axial_gap * 1e3,
        equivalent_axial * 1e3,
        linestyle="--",
        label="area-equivalent cylinder",
    )
    ax.set_title(f"bundle reduction (normalized RMS error {relative_rms:.1%})")
    ax.set_xlabel("distance from face (mm)")
    ax.set_ylabel("Bz (mT)")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(neighbor_x * 1e3, primary_field * 1e3, label="primary only")
    ax.plot(neighbor_x * 1e3, combined_field * 1e3, label="with opposed neighbor")
    ax.axvline(-0.5 * separation * 1e3, color="0.4", linestyle=":")
    ax.axvline(0.5 * separation * 1e3, color="0.4", linestyle=":")
    ax.set_title("neighbor contribution before hysteresis coupling")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("Bz (mT)")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    ax.semilogy(convergence_orders, 100.0 * errors, marker="o")
    ax.set_title("cubature error vs exact finite cylinder")
    ax.set_xlabel("samples across diameter")
    ax.set_ylabel("relative axial-field error (%)")
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle(
        "Electropermanent magnet static model — geometry before pulse programming",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=160)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
