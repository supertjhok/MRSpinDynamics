"""Plot synthetic 1D and 2D inverse Laplace transform recoveries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()




def _check_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise SystemExit(
            "SciPy is required for non-negative inverse Laplace transforms. "
            "Install the opt extra or run: python -m pip install scipy"
        ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic T1, T2, T1-T2, and D-T2 inverse Laplace "
            "transform examples at one or more SNR levels."
        )
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["t2", "t1", "t1-t2", "d-t2"],
        choices=["t2", "t1", "t1-t2", "d-t2"],
        help="ILT examples to include in the figure.",
    )
    parser.add_argument(
        "--snr-levels",
        nargs="+",
        type=float,
        default=[100.0, 40.0, 15.0],
        help="Signal-to-noise ratios for the synthetic noisy data.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=5e-4,
        help="Manual Tikhonov regularization strength for every inversion.",
    )
    parser.add_argument(
        "--auto-regularization",
        action="store_true",
        help="Select regularization by matching residual norm to each SNR.",
    )
    parser.add_argument(
        "--auto-strength-min",
        type=float,
        default=1e-8,
        help="Smallest regularization strength tried in automatic mode.",
    )
    parser.add_argument(
        "--auto-strength-max",
        type=float,
        default=1e1,
        help="Largest regularization strength tried in automatic mode.",
    )
    parser.add_argument(
        "--auto-strength-count",
        type=int,
        default=37,
        help="Number of logarithmic strengths tried in automatic mode.",
    )
    parser.add_argument(
        "--regularization-order",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Penalty order: 0 amplitude, 1 slope, or 2 curvature.",
    )
    parser.add_argument(
        "--t1-mode",
        choices=["saturation", "inversion"],
        default="inversion",
        help="T1 kernel used by the T1 and T1-T2 examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for synthetic noise.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _add_noise(data: np.ndarray, snr: float, rng: np.random.Generator) -> np.ndarray:
    if snr <= 0.0:
        raise ValueError("SNR levels must be positive")
    # The example defines SNR using clean RMS divided by additive noise RMS.
    signal_rms = float(np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2)))
    sigma = signal_rms / snr
    return data + rng.normal(scale=sigma, size=data.shape)


def _auto_strengths(args: argparse.Namespace) -> np.ndarray:
    from spin_dynamics.analysis import default_regularization_strengths

    return default_regularization_strengths(
        args.auto_strength_min,
        args.auto_strength_max,
        args.auto_strength_count,
    )


def _sparse_1d(axis: np.ndarray, components: list[tuple[float, float]]) -> np.ndarray:
    # Put idealized delta-like components on the nearest grid points.
    distribution = np.zeros(axis.size, dtype=np.float64)
    for value, amplitude in components:
        distribution[int(np.argmin(np.abs(axis - value)))] += amplitude
    return distribution


def _sparse_2d(
    axis1: np.ndarray,
    axis2: np.ndarray,
    components: list[tuple[float, float, float]],
) -> np.ndarray:
    # Components are stored as (axis1 value, axis2 value, amplitude).
    distribution = np.zeros((axis1.size, axis2.size), dtype=np.float64)
    for value1, value2, amplitude in components:
        idx1 = int(np.argmin(np.abs(axis1 - value1)))
        idx2 = int(np.argmin(np.abs(axis2 - value2)))
        distribution[idx1, idx2] += amplitude
    return distribution


def _plot_1d(
    ax,
    axis: np.ndarray,
    true_dist: np.ndarray,
    recovered: np.ndarray,
    title: str,
) -> None:
    ax.semilogx(
        axis,
        true_dist / np.max(true_dist),
        color="0.25",
        linestyle="--",
        label="true",
    )
    peak = max(float(np.max(recovered)), np.finfo(float).eps)
    ax.semilogx(axis, recovered / peak, color="tab:blue", label="recovered")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize="small")


def _plot_2d(
    ax,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    recovered: np.ndarray,
    components: list[tuple[float, float, float]],
    title: str,
    *,
    y_log: bool,
) -> None:
    image = recovered / max(float(np.max(recovered)), np.finfo(float).eps)
    mesh = ax.pcolormesh(x_axis, y_axis, image, shading="auto", cmap="viridis")
    for y_value, x_value, _ in components:
        ax.plot(x_value, y_value, marker="x", color="white", markersize=7, mew=1.5)
    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.set_title(title)
    return mesh


def _run_t2(snr: float, rng: np.random.Generator, args: argparse.Namespace):
    from spin_dynamics.analysis import select_regularization_1d, t2_kernel, invert_t2

    echo_times = np.linspace(0.0005, 0.09, 40)
    t2_axis = np.logspace(-4, -1, 60)
    true_dist = _sparse_1d(t2_axis, [(0.006, 1.0), (0.03, 0.45)])
    clean = t2_kernel(echo_times, t2_axis) @ true_dist
    noisy = _add_noise(clean, snr, rng)
    if args.auto_regularization:
        # Automatic mode scans lambda values and picks from the noise target.
        selection = select_regularization_1d(
            noisy,
            echo_times,
            t2_axis,
            snr=snr,
            kernel="t2",
            strengths=_auto_strengths(args),
            regularization_order=args.regularization_order,
        )
        return t2_axis, true_dist, selection.result, selection.selected_strength
    result = invert_t2(
        noisy,
        echo_times,
        t2_axis,
        regularization=args.regularization,
        regularization_order=args.regularization_order,
    )
    return t2_axis, true_dist, result, args.regularization


def _run_t1(
    snr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
):
    from spin_dynamics.analysis import invert_t1, select_regularization_1d, t1_kernel

    recovery_times = np.linspace(0.0002, 0.05, 40)
    t1_axis = np.logspace(-4, -1, 60)
    true_dist = _sparse_1d(t1_axis, [(0.004, 1.0), (0.02, 0.55)])
    clean = t1_kernel(recovery_times, t1_axis, mode=args.t1_mode) @ true_dist
    noisy = _add_noise(clean, snr, rng)
    kernel = "t1_ir" if args.t1_mode == "inversion" else "t1"
    if args.auto_regularization:
        # The selector uses the same kernel as the manual inversion path.
        selection = select_regularization_1d(
            noisy,
            recovery_times,
            t1_axis,
            snr=snr,
            kernel=kernel,
            strengths=_auto_strengths(args),
            regularization_order=args.regularization_order,
        )
        return t1_axis, true_dist, selection.result, selection.selected_strength
    result = invert_t1(
        noisy,
        recovery_times,
        t1_axis,
        mode=args.t1_mode,
        regularization=args.regularization,
        regularization_order=args.regularization_order,
    )
    return t1_axis, true_dist, result, args.regularization


def _run_t1_t2(
    snr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
):
    from spin_dynamics.analysis import (
        invert_t1_t2,
        select_regularization_2d,
        t1_kernel,
        t2_kernel,
    )

    recovery_times = np.linspace(0.0004, 0.045, 20)
    echo_times = np.linspace(0.0005, 0.06, 18)
    t1_axis = np.logspace(-4, -1, 24)
    t2_axis = np.logspace(-4, -1, 22)
    components = [(0.004, 0.007, 1.0), (0.025, 0.032, 0.5)]
    true_dist = _sparse_2d(t1_axis, t2_axis, components)
    clean = (
        t1_kernel(recovery_times, t1_axis, mode=args.t1_mode)
        @ true_dist
        @ t2_kernel(echo_times, t2_axis).T
    )
    noisy = _add_noise(clean, snr, rng)
    kernel1 = "t1_ir" if args.t1_mode == "inversion" else "t1"
    if args.auto_regularization:
        # For 2D maps this scans one shared lambda scale for both axes.
        selection = select_regularization_2d(
            noisy,
            recovery_times,
            echo_times,
            t1_axis,
            t2_axis,
            snr=snr,
            kernel1=kernel1,
            kernel2="t2",
            strengths=_auto_strengths(args),
            regularization_order=args.regularization_order,
        )
        return (
            t1_axis,
            t2_axis,
            components,
            selection.result,
            selection.selected_strength,
        )
    result = invert_t1_t2(
        noisy,
        recovery_times,
        echo_times,
        t1_axis,
        t2_axis,
        t1_mode=args.t1_mode,
        regularization=(args.regularization, args.regularization),
        regularization_order=args.regularization_order,
    )
    return t1_axis, t2_axis, components, result, args.regularization


def _run_d_t2(snr: float, rng: np.random.Generator, args: argparse.Namespace):
    from spin_dynamics.analysis import (
        diffusion_kernel,
        invert_d_t2,
        select_regularization_2d,
        t2_kernel,
    )

    b_values = np.linspace(0.0, 4.0e9, 20)
    echo_times = np.linspace(0.0005, 0.06, 18)
    diffusion_axis = np.linspace(0.2e-9, 2.8e-9, 24)
    t2_axis = np.logspace(-4, -1, 22)
    components = [(0.8e-9, 0.008, 1.0), (1.8e-9, 0.028, 0.45)]
    true_dist = _sparse_2d(diffusion_axis, t2_axis, components)
    clean = (
        diffusion_kernel(b_values, diffusion_axis)
        @ true_dist
        @ t2_kernel(echo_times, t2_axis).T
    )
    noisy = _add_noise(clean, snr, rng)
    if args.auto_regularization:
        # D-T2 uses the same separable selector with a diffusion first kernel.
        selection = select_regularization_2d(
            noisy,
            b_values,
            echo_times,
            diffusion_axis,
            t2_axis,
            snr=snr,
            kernel1="diffusion",
            kernel2="t2",
            strengths=_auto_strengths(args),
            regularization_order=args.regularization_order,
        )
        return (
            diffusion_axis,
            t2_axis,
            components,
            selection.result,
            selection.selected_strength,
        )
    result = invert_d_t2(
        noisy,
        b_values,
        echo_times,
        diffusion_axis,
        t2_axis,
        regularization=(args.regularization, args.regularization),
        regularization_order=args.regularization_order,
    )
    return diffusion_axis, t2_axis, components, result, args.regularization


def main() -> None:
    args = _parse_args()
    _check_scipy()
    plt = load_matplotlib()

    rng = np.random.default_rng(args.seed)
    cases = list(dict.fromkeys(args.cases))
    snr_levels = [float(snr) for snr in args.snr_levels]
    fig, axes = plt.subplots(
        len(snr_levels),
        len(cases),
        figsize=(4.0 * len(cases), 3.1 * len(snr_levels)),
        squeeze=False,
    )

    for row, snr in enumerate(snr_levels):
        for col, case in enumerate(cases):
            ax = axes[row, col]
            if case == "t2":
                axis, true_dist, result, strength = _run_t2(snr, rng, args)
                _plot_1d(
                    ax,
                    axis,
                    true_dist,
                    result.distribution,
                    f"T2, SNR {snr:g}, lambda {strength:.1e}",
                )
                ax.set_xlabel("T2 (s)")
                ax.set_ylabel("normalized amplitude")
            elif case == "t1":
                axis, true_dist, result, strength = _run_t1(snr, rng, args)
                _plot_1d(
                    ax,
                    axis,
                    true_dist,
                    result.distribution,
                    f"T1, SNR {snr:g}, lambda {strength:.1e}",
                )
                ax.set_xlabel("T1 (s)")
                ax.set_ylabel("normalized amplitude")
            elif case == "t1-t2":
                t1_axis, t2_axis, components, result, strength = _run_t1_t2(
                    snr, rng, args
                )
                _plot_2d(
                    ax,
                    t2_axis,
                    t1_axis,
                    result.distribution,
                    components,
                    f"T1-T2, SNR {snr:g}, lambda {strength:.1e}",
                    y_log=True,
                )
                ax.set_xlabel("T2 (s)")
                ax.set_ylabel("T1 (s)")
            elif case == "d-t2":
                d_axis, t2_axis, components, result, strength = _run_d_t2(
                    snr, rng, args
                )
                _plot_2d(
                    ax,
                    t2_axis,
                    d_axis * 1e9,
                    result.distribution,
                    [(d * 1e9, t2, amp) for d, t2, amp in components],
                    f"D-T2, SNR {snr:g}, lambda {strength:.1e}",
                    y_log=False,
                )
                ax.set_xlabel("T2 (s)")
                ax.set_ylabel("D (10^-9 m^2/s)")
            else:  # pragma: no cover - argparse prevents this
                raise ValueError(case)

    fig.tight_layout()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
