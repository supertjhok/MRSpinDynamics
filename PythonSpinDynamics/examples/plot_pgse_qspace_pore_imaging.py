"""Estimate a circular pore image from its 2D q-space diffraction response.

The companion ``plot_pgse_circular_pore_diffraction.py`` example shows the
one-dimensional diffraction minima from a circular pore. This example takes the
next inverse step: a two-dimensional q-space response is converted back into a
real-space pore estimate.

In the ideal short-gradient-pulse, long-diffusion-time limit the response is the
pore form factor. If the complex phase is preserved, the pore density is just an
inverse Fourier transform. Conventional diffusion diffraction usually measures
only ``|F(q)|^2``; its direct inverse is the pore autocorrelation, so estimating
the pore shape itself requires a constraint such as finite support and
non-negativity. The final panels demonstrate that phase-retrieval step for both
ideal and finite-SNR q-space measurements.
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
            "Reconstruct a circular pore image from a 2D q-space "
            "diffusion-diffraction response."
        )
    )
    parser.add_argument(
        "--pixels",
        type=int,
        default=64,
        help="Number of image/q-space samples per axis.",
    )
    parser.add_argument(
        "--pore-radius",
        type=float,
        default=5.0e-6,
        help="Circular pore radius (m).",
    )
    parser.add_argument(
        "--fov-factor",
        type=float,
        default=5.0,
        help="Real-space field of view as a multiple of the pore diameter.",
    )
    parser.add_argument(
        "--support-factor",
        type=float,
        default=1.35,
        help="Phase-retrieval support radius as a multiple of the true pore radius.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=280,
        help="Hybrid-input-output phase-retrieval iterations.",
    )
    parser.add_argument(
        "--er-iterations",
        type=int,
        default=60,
        help="Final error-reduction cleanup iterations.",
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=50.0,
        help=(
            "Finite-SNR q-space intensity case. Defined as peak |F(q)|^2 divided "
            "by the Gaussian noise standard deviation; use inf to disable noise."
        ),
    )
    parser.add_argument("--seed", type=int, default=4, help="Phase seed.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output PNG path. If omitted, show the plot.",
    )
    return parser.parse_args()


def _make_disc(radius: float, pixels: int, fov_factor: float):
    fov = float(fov_factor) * 2.0 * float(radius)
    dx = fov / int(pixels)
    axis = (np.arange(int(pixels), dtype=np.float64) - int(pixels) // 2) * dx
    xx, zz = np.meshgrid(axis, axis, indexing="ij")
    rho = (xx**2 + zz**2 <= float(radius) ** 2).astype(float)
    return rho, axis, axis


def _add_intensity_noise(
    intensity: np.ndarray,
    *,
    snr: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    if np.isinf(snr):
        return intensity.copy(), 0.0
    sigma = float(np.max(intensity)) / float(snr)
    rng = np.random.default_rng(seed)
    noisy = intensity + rng.normal(scale=sigma, size=intensity.shape)
    return np.maximum(noisy, 0.0), sigma


def _plot(
    plt,
    *,
    args,
    rho,
    qx,
    qz,
    intensity,
    noisy_intensity,
    autocorr,
    support,
    retrieved,
    noisy_retrieved,
):
    radius_um = args.pore_radius * 1e6
    x_um = retrieved.x_axis * 1e6
    z_um = retrieved.z_axis * 1e6
    q_cycles_um = qx / (2.0 * np.pi) * 1e-6
    extent_xz = [z_um[0], z_um[-1], x_um[0], x_um[-1]]
    extent_q = [q_cycles_um[0], q_cycles_um[-1], q_cycles_um[0], q_cycles_um[-1]]
    zz_um, xx_um = np.meshgrid(z_um, x_um, indexing="xy")

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.0), constrained_layout=True)
    axes = axes.ravel()

    axes[0].imshow(rho, origin="lower", extent=extent_xz, cmap="gray_r")
    axes[0].set_title(f"True pore, a = {radius_um:.1f} um")
    axes[0].set_xlabel("z (um)")
    axes[0].set_ylabel("x (um)")

    im1 = axes[1].imshow(
        np.log10(np.maximum(intensity, 1e-8)),
        origin="lower",
        extent=extent_q,
        cmap="magma",
        vmin=-5,
        vmax=0,
    )
    axes[1].set_title("Ideal intensity |F(q)|^2")
    axes[1].set_xlabel("q_z (1/um)")
    axes[1].set_ylabel("q_x (1/um)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="log10 intensity")

    im2 = axes[2].imshow(
        np.log10(np.maximum(noisy_intensity, 1e-8)),
        origin="lower",
        extent=extent_q,
        cmap="magma",
        vmin=-5,
        vmax=0,
    )
    snr_label = "infinite" if np.isinf(args.snr) else f"{args.snr:g}"
    axes[2].set_title(f"Finite-SNR intensity, SNR={snr_label}")
    axes[2].set_xlabel("q_z (1/um)")
    axes[2].set_ylabel("q_x (1/um)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="log10 intensity")

    axes[3].imshow(autocorr.image, origin="lower", extent=extent_xz, cmap="viridis")
    axes[3].contour(
        zz_um, xx_um, support.astype(float), levels=[0.5],
        colors="white", linewidths=0.8,
    )
    axes[3].set_title("Direct inverse: autocorrelation")
    axes[3].set_xlabel("z displacement (um)")
    axes[3].set_ylabel("x displacement (um)")

    recovered = retrieved.density / max(float(retrieved.density.max()), 1e-12)
    axes[4].imshow(recovered, origin="lower", extent=extent_xz, cmap="gray_r")
    axes[4].contour(
        zz_um, xx_um, support.astype(float), levels=[0.5],
        colors="#1f77b4", linewidths=0.8,
    )
    axes[4].set_title(f"Ideal retrieval, residual {retrieved.residual:.1e}")
    axes[4].set_xlabel("z (um)")
    axes[4].set_ylabel("x (um)")

    noisy = noisy_retrieved.density / max(float(noisy_retrieved.density.max()), 1e-12)
    axes[5].imshow(noisy, origin="lower", extent=extent_xz, cmap="gray_r")
    axes[5].contour(
        zz_um, xx_um, support.astype(float), levels=[0.5],
        colors="#1f77b4", linewidths=0.8,
    )
    axes[5].set_title(f"Finite-SNR retrieval, residual {noisy_retrieved.residual:.1e}")
    axes[5].set_xlabel("z (um)")
    axes[5].set_ylabel("x (um)")

    for ax in axes:
        ax.set_aspect("equal")
    return fig


def main() -> None:
    from spin_dynamics.workflows import (
        phase_retrieve_qspace_magnitude,
        pore_form_factor_from_density,
        qspace_axes_from_real_space,
        reconstruct_qspace_image,
    )

    args = _parse_args()
    if args.pixels < 8:
        raise SystemExit("--pixels must be at least 8")
    if args.pore_radius <= 0.0:
        raise SystemExit("--pore-radius must be positive")
    if args.fov_factor <= 2.0:
        raise SystemExit("--fov-factor must exceed 2")
    if args.snr <= 0.0:
        raise SystemExit("--snr must be positive")

    plt = load_matplotlib(headless=bool(args.output))
    rho, x_axis, z_axis = _make_disc(args.pore_radius, args.pixels, args.fov_factor)
    qx, qz = qspace_axes_from_real_space(x_axis, z_axis)
    form = pore_form_factor_from_density(rho)
    intensity = np.abs(form) ** 2
    noisy_intensity, noise_sigma = _add_intensity_noise(
        intensity, snr=float(args.snr), seed=int(args.seed) + 1000
    )
    autocorr = reconstruct_qspace_image(intensity, qx, qz, data_kind="intensity")

    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    support_radius = float(args.support_factor) * float(args.pore_radius)
    support = xx**2 + zz**2 <= support_radius**2
    retrieved = phase_retrieve_qspace_magnitude(
        intensity,
        qx,
        qz,
        support=support,
        input_is_intensity=True,
        iterations=int(args.iterations),
        er_iterations=int(args.er_iterations),
        seed=int(args.seed),
    )
    noisy_retrieved = phase_retrieve_qspace_magnitude(
        noisy_intensity,
        qx,
        qz,
        support=support,
        input_is_intensity=True,
        iterations=int(args.iterations),
        er_iterations=int(args.er_iterations),
        seed=int(args.seed) + 1,
    )

    print("q-space pore imaging")
    print(f"pore radius: {args.pore_radius * 1e6:.2f} um")
    print(f"pixels: {args.pixels} x {args.pixels}")
    print(f"finite-SNR intensity: SNR={args.snr:g}, noise sigma={noise_sigma:.3e}")
    print("direct intensity inversion returns the pore autocorrelation")
    print(
        "ideal phase retrieval estimate: "
        f"{retrieved.iterations} iterations, residual {retrieved.residual:.3e}"
    )
    print(
        "finite-SNR phase retrieval estimate: "
        f"{noisy_retrieved.iterations} iterations, residual {noisy_retrieved.residual:.3e}"
    )

    fig = _plot(
        plt,
        args=args,
        rho=rho,
        qx=qx,
        qz=qz,
        intensity=intensity,
        noisy_intensity=noisy_intensity,
        autocorr=autocorr,
        support=support,
        retrieved=retrieved,
        noisy_retrieved=noisy_retrieved,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
