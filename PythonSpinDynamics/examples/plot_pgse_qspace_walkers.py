"""Validate finite-pulse PGSE walker data through q-space reconstruction.

An elliptical pore supplies a non-circular ground truth. The example acquires a
two-dimensional angular-q grid with explicit restricted-diffusion walkers,
compares it with the short-gradient-pulse ``|F(q)|^2`` limit, and reconstructs
both the pore autocorrelation and a support-constrained pore-shape estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


# Keep CLI choices together so scientific defaults are easy to find and override.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixels", type=int, default=12)
    parser.add_argument("--walkers-per-cell", type=int, default=32)
    parser.add_argument("--gradient-duration", type=float, default=0.3e-3)
    parser.add_argument("--diffusion-time", type=float, default=24.0e-3)
    parser.add_argument("--diffusion", type=float, default=1.0e-9)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


# Follow the user workflow: parse inputs, build the model, run, then report.
def main() -> None:
    from spin_dynamics.motion import make_elliptical_reflector
    from spin_dynamics.workflows import (
        acquire_pgse_qspace_walkers,
        phase_retrieve_qspace_magnitude,
        pore_form_factor_from_density,
        qspace_axes_from_real_space,
        reconstruct_qspace_image,
    )

    args = _parse_args()
    if args.pixels < 8 or args.walkers_per_cell <= 0:
        raise SystemExit("--pixels must be at least 8 and walkers-per-cell positive")
    axis = (np.arange(args.pixels, dtype=np.float64) - args.pixels // 2) * 1.0e-6
    xx, zz = np.meshgrid(axis, axis, indexing="ij")
    semi_axes = (3.2e-6, 2.1e-6)
    rho = ((xx / semi_axes[0]) ** 2 + (zz / semi_axes[1]) ** 2 <= 1.0).astype(float)
    qx, qz = qspace_axes_from_real_space(axis, axis)
    ideal_intensity = np.abs(pore_form_factor_from_density(rho)) ** 2
    acquired = acquire_pgse_qspace_walkers(
        rho,
        axis,
        axis,
        qx,
        qz,
        gradient_duration=args.gradient_duration,
        diffusion_time=args.diffusion_time,
        diffusion_coefficient=args.diffusion,
        walkers_per_cell=args.walkers_per_cell,
        seed=args.seed,
        boundary=make_elliptical_reflector((0.0, 0.0), semi_axes),
        substeps_per_interval=5,
    )
    autocorrelation = reconstruct_qspace_image(
        acquired.intensity, qx, qz, data_kind="intensity", clip_negative=True
    )
    support = (xx / 4.5e-6) ** 2 + (zz / 3.2e-6) ** 2 <= 1.0
    retrieved = phase_retrieve_qspace_magnitude(
        acquired.intensity,
        qx,
        qz,
        support=support,
        input_is_intensity=True,
        iterations=220,
        er_iterations=60,
        seed=3,
    )
    correlation = float(
        np.corrcoef(acquired.intensity.ravel(), ideal_intensity.ravel())[0, 1]
    )
    print("finite-pulse walker-to-q-space validation")
    print(f"grid: {args.pixels} x {args.pixels}; walkers/cell: {args.walkers_per_cell}")
    print(
        f"delta: {args.gradient_duration * 1e3:.3f} ms; "
        f"Delta: {args.diffusion_time * 1e3:.1f} ms"
    )
    print(f"walker vs short-pulse q-space correlation: {correlation:.3f}")
    print(f"phase-retrieval residual: {retrieved.residual:.3e}")

    plt = load_matplotlib(headless=bool(args.output))
    fig, axes = plt.subplots(1, 5, figsize=(15.8, 3.4), constrained_layout=True)
    extent = [axis[0] * 1e6, axis[-1] * 1e6, axis[0] * 1e6, axis[-1] * 1e6]
    q_cycles = qx / (2.0 * np.pi) * 1e-6
    q_extent = [q_cycles[0], q_cycles[-1], q_cycles[0], q_cycles[-1]]
    axes[0].imshow(rho, origin="lower", extent=extent, cmap="gray_r")
    axes[0].set_title("True elliptical pore")
    axes[1].imshow(
        np.log10(np.maximum(ideal_intensity, 1e-5)),
        origin="lower",
        extent=q_extent,
        cmap="magma",
    )
    axes[1].set_title("Short-pulse |F(q)|²")
    axes[2].imshow(
        np.log10(np.maximum(acquired.intensity, 1e-5)),
        origin="lower",
        extent=q_extent,
        cmap="magma",
    )
    axes[2].set_title("Finite-pulse walkers")
    axes[3].imshow(autocorrelation.image, origin="lower", extent=extent, cmap="viridis")
    axes[3].set_title("Walker autocorrelation")
    recovered = retrieved.density / max(float(retrieved.density.max()), 1e-12)
    axes[4].imshow(recovered, origin="lower", extent=extent, cmap="gray_r")
    axes[4].set_title("Walker pore estimate")
    for index, ax in enumerate(axes):
        ax.set_aspect("equal")
        ax.set_xlabel("qz (1/µm)" if index in (1, 2) else "z (µm)")
        ax.set_ylabel("qx (1/µm)" if index in (1, 2) else "x (µm)")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
