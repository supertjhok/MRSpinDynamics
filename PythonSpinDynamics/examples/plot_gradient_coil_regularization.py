"""Explore the Tikhonov/L-curve trade-off in cylindrical gradient-coil design.

The same Biot-Savart sensitivity matrix is solved over a logarithmic alpha grid.
The figure shows how regularization trades target-field accuracy for smaller,
smoother current amplitudes, the discrete L-curve corner, and stream functions
from weak, selected, and strong regularization.

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
    parser.add_argument("--n-phi", type=int, default=28)
    parser.add_argument("--n-z", type=int, default=15)
    parser.add_argument("--path-points", type=int, default=23)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# Keep orchestration in one entry point so helper functions remain reusable.
def main() -> None:
    args = _parse_args()
    # Load Matplotlib only after choosing interactive or headless output mode.
    plt = load_matplotlib(headless=bool(args.output))

    from spin_dynamics.fields import (
        CylindricalWindingSurface,
        build_cylindrical_gradient_system,
        linear_gradient_target,
        solve_regularization_path,
        spherical_target_points,
    )

    surface = CylindricalWindingSurface(0.08, 0.18, args.n_phi, args.n_z)
    points = spherical_target_points(0.03, points_per_axis=7)
    target = linear_gradient_target(points, (0.0, 10.0e-3, 0.0))
    system = build_cylindrical_gradient_system(surface, points)
    path = solve_regularization_path(
        system,
        target,
        np.logspace(-20.0, -9.0, args.path_points),
    )

    alphas = path.regularizations_t2_per_a2
    errors = np.asarray([result.relative_rms_error for result in path.results])
    print("Gradient-coil regularization path")
    print(f"  selected alpha: {path.selected_regularization:.3e} T^2/A^2")
    print(
        f"  selected error/current norm: "
        f"{path.selected_result.relative_rms_error:.3%} / "
        f"{path.selected_result.current_norm_a:.3f} A"
    )

    # Assemble the observables and diagnostics for side-by-side interpretation.
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    ax = axes[0, 0]
    ax.loglog(alphas, errors, "o-", color="C3", label="relative RMS field error")
    ax.set_xlabel("regularization alpha (T^2/A^2)")
    ax.set_ylabel("relative RMS error")
    ax.grid(True, which="both", alpha=0.22)
    ax2 = ax.twinx()
    ax2.loglog(alphas, path.current_norms_a, "s-", color="C0", label="current norm")
    ax2.set_ylabel("segment-current norm (A)")
    ax.set_title("Bias-current trade-off")

    ax = axes[0, 1]
    ax.loglog(path.weighted_residual_norms_t, path.current_norms_a, "o-", lw=1.4)
    selected = path.selected_index
    ax.plot(
        path.weighted_residual_norms_t[selected],
        path.current_norms_a[selected],
        "o",
        color="C3",
        ms=9,
        label="selected corner",
    )
    ax.set_xlabel("weighted residual norm (T)")
    ax.set_ylabel("segment-current norm (A)")
    ax.set_title("Discrete L-curve")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(fontsize=8)

    representative = (0, selected, len(path.results) - 1)
    labels = ("weak", "L-curve", "strong")
    colors = ("C0", "C2", "C3")
    ax = axes[1, 0]
    for index, label, color in zip(representative, labels, colors, strict=True):
        result = path.results[index]
        profile = result.stream_function_a[int(surface.n_phi) // 4]
        ax.plot(
            100.0 * result.stream_z,
            profile,
            color=color,
            lw=1.6,
            label=f"{label}: {alphas[index]:.1e}",
        )
    ax.set_xlabel("z (cm)")
    ax.set_ylabel("stream function at phi=90 degrees (A)")
    ax.set_title("Axial smoothing of the current distribution")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.semilogx(alphas, path.l_curve_curvature, "o-", color="C4")
    ax.axvline(path.selected_regularization, color="C3", ls="--", lw=1.2)
    ax.set_xlabel("regularization alpha (T^2/A^2)")
    ax.set_ylabel("L-curve curvature")
    ax.set_title("Automated corner diagnostic")
    ax.grid(True, which="both", alpha=0.22)

    fig.suptitle("Cylindrical gradient-coil regularization path", fontsize=14)
    fig.tight_layout()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=170, bbox_inches="tight")
        print(f"  saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
