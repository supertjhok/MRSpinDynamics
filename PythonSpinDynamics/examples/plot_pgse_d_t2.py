"""Simulate PGSE-prepared CPMG data and recover a D-T2 map with 2D ILT."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic PGSE-prepared CPMG data, then recover a D-T2 "
            "distribution with the separable 2D inverse Laplace transform."
        )
    )
    parser.add_argument(
        "--snr",
        type=float,
        default=80.0,
        help="Synthetic data SNR, defined as clean RMS divided by noise RMS.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-4,
        help="Tikhonov regularization strength used on both ILT axes.",
    )
    parser.add_argument(
        "--regularization-order",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Penalty order: 0 amplitude, 1 slope, or 2 curvature.",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help=(
            "Use an unconstrained least-squares ILT. By default the example "
            "uses non-negative ILT, which requires SciPy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for synthetic measurement noise.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _add_noise(data: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    if snr <= 0.0:
        raise ValueError("snr must be positive")
    signal_rms = float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))
    return data + rng.normal(scale=signal_rms / float(snr), size=data.shape)


def _sparse_distribution(
    diffusion_axis: np.ndarray,
    t2_axis: np.ndarray,
    components: list[tuple[float, float, float]],
) -> np.ndarray:
    distribution = np.zeros((diffusion_axis.size, t2_axis.size), dtype=np.float64)
    for diffusion, t2, amplitude in components:
        d_idx = int(np.argmin(np.abs(diffusion_axis - diffusion)))
        t2_idx = int(np.argmin(np.abs(t2_axis - t2)))
        distribution[d_idx, t2_idx] += amplitude
    return distribution


def _pgse_b_axis() -> tuple[np.ndarray, np.ndarray]:
    from spin_dynamics.workflows import run_pgse_moment

    gradients = np.linspace(0.0, 0.32, 22)
    b_values = np.array(
        [
            run_pgse_moment(
                gradient_amplitude=float(gradient),
                gradient_duration=2.5e-3,
                diffusion_time=28.0e-3,
                diffusion_coefficient=0.0,
                num_echoes=1,
            ).b_value
            for gradient in gradients
        ],
        dtype=np.float64,
    )
    return gradients, b_values


def _simulate_pgse_cpmg_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[float, float, float]],
]:
    from spin_dynamics.analysis import diffusion_kernel, t2_kernel

    _gradients, b_values = _pgse_b_axis()
    echo_times = np.linspace(3.0e-3, 90.0e-3, 28)
    diffusion_axis = np.linspace(0.15e-9, 2.8e-9, 48)
    t2_axis = np.logspace(np.log10(2.0e-3), np.log10(180.0e-3), 52)
    components = [
        (0.55e-9, 16.0e-3, 1.0),
        (1.65e-9, 62.0e-3, 0.58),
    ]
    true_distribution = _sparse_distribution(diffusion_axis, t2_axis, components)
    data = (
        diffusion_kernel(b_values, diffusion_axis)
        @ true_distribution
        @ t2_kernel(echo_times, t2_axis).T
    )
    return data, b_values, echo_times, diffusion_axis, t2_axis, components


def _plot_results(
    plt,
    b_values: np.ndarray,
    echo_times: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    diffusion_axis: np.ndarray,
    t2_axis: np.ndarray,
    recovered: np.ndarray,
    components: list[tuple[float, float, float]],
    *,
    snr: float,
    regularization: float,
    nonnegative: bool,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))

    axes[0].plot(b_values * 1e-9, clean[:, 0], label="clean")
    axes[0].plot(b_values * 1e-9, noisy[:, 0], ".", markersize=4, label="noisy")
    axes[0].set_xlabel("b (10^9 s/m^2)")
    axes[0].set_ylabel("first echo amplitude")
    axes[0].set_title("PGSE attenuation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize="small")

    image = axes[1].imshow(
        noisy,
        origin="lower",
        aspect="auto",
        extent=[
            echo_times[0] * 1e3,
            echo_times[-1] * 1e3,
            b_values[0] * 1e-9,
            b_values[-1] * 1e-9,
        ],
        cmap="magma",
    )
    axes[1].set_xlabel("echo time (ms)")
    axes[1].set_ylabel("b (10^9 s/m^2)")
    axes[1].set_title(f"PGSE-CPMG data, SNR {snr:g}")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    display = recovered if nonnegative else np.clip(recovered, 0.0, None)
    normalized = display / max(float(np.max(display)), np.finfo(float).eps)
    mesh = axes[2].pcolormesh(
        t2_axis * 1e3,
        diffusion_axis * 1e9,
        normalized,
        shading="auto",
        cmap="viridis",
    )
    for diffusion, t2, _amplitude in components:
        axes[2].plot(t2 * 1e3, diffusion * 1e9, "wx", markersize=8, mew=1.6)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("T2 (ms)")
    axes[2].set_ylabel("D (10^-9 m^2/s)")
    solver = "NNLS" if nonnegative else "LS preview"
    axes[2].set_title(f"Recovered D-T2 ({solver})")
    fig.colorbar(mesh, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    nonnegative = not args.unconstrained
    if nonnegative and not _has_scipy():
        print(
            "SciPy is not installed; falling back to --unconstrained. "
            "Install the opt extra for non-negative ILT."
        )
        nonnegative = False
    plt = load_matplotlib()

    from spin_dynamics.analysis import invert_d_t2

    clean, b_values, echo_times, diffusion_axis, t2_axis, components = (
        _simulate_pgse_cpmg_data()
    )
    rng = np.random.default_rng(args.seed)
    noisy = _add_noise(clean, args.snr, rng)
    result = invert_d_t2(
        noisy,
        b_values,
        echo_times,
        diffusion_axis,
        t2_axis,
        regularization=(args.regularization, args.regularization),
        regularization_order=args.regularization_order,
        nonnegative=nonnegative,
    )

    fig = _plot_results(
        plt,
        b_values,
        echo_times,
        clean,
        noisy,
        diffusion_axis,
        t2_axis,
        result.distribution,
        components,
        snr=args.snr,
        regularization=args.regularization,
        nonnegative=nonnegative,
    )
    print(f"b range: {b_values[0]:.3e} to {b_values[-1]:.3e} s/m^2")
    print(f"relative ILT residual: {result.residual_norm / np.linalg.norm(noisy):.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
