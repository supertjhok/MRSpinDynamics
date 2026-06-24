"""Restricted diffusion in a pore with the stochastic random-walker PGSE backend.

The analytical (moment) PGSE backend assumes free, unrestricted Gaussian
diffusion, so its attenuation is always the straight Stejskal-Tanner line
``E(b) = exp(-b * D)``. The random-walker backend instead moves explicit
isochromats and reflects them off hard walls, which is exactly what is needed to
model spins trapped inside a pore.

This example confines walkers to a slab pore of width ``L`` along the gradient
axis (reflecting walls) and contrasts it with free diffusion run through the same
engine. It reproduces the two classic signatures of restricted diffusion:

1. The echo attenuation ``E(b)`` bends *below* the free-diffusion line: as the
   pore shrinks relative to the diffusion length the spins cannot dephase as
   much, so they retain more signal.
2. The apparent diffusion coefficient ``D_app = -ln(E)/b`` drops below the free
   value as the diffusion time grows, because walkers increasingly feel the
   walls and their mean-squared displacement saturates.

A walker displacement histogram makes the mechanism explicit: free diffusion
spreads as a Gaussian, while the pore clamps the distribution to its width.

Run with ``--output figure.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


# Physical constants and shared sequence timing (SI units).
GAMMA = 2.675e8  # rad/(s*T), proton gyromagnetic ratio
D_FREE = 2.3e-9  # m^2/s, bulk water at room temperature
GRADIENT_DURATION = 2.0e-3  # delta (s)
Z_HALF_WIDTH = 0.5e-6  # half-thickness of the (gradient-free) slab in z

# Pore widths probed in the b-sweep. ``np.inf`` is the free-diffusion control.
PORE_CONDITIONS: list[tuple[str, float]] = [
    ("free", np.inf),
    ("20 um pore", 20.0e-6),
    ("10 um pore", 10.0e-6),
    ("6 um pore", 6.0e-6),
]
PORE_COLORS = ["k", "#1f77b4", "#2ca02c", "#d62728"]
FREE_INIT_WIDTH = 12.0e-6  # starting spread for the free control (result is invariant)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stochastic random-walker PGSE in a reflecting pore, showing how "
            "restricted diffusion bends the attenuation curve and lowers the "
            "apparent diffusion coefficient relative to free diffusion."
        )
    )
    parser.add_argument(
        "--num-cells",
        type=int,
        default=24,
        help="Spatial cells across the pore used to seed walkers along x.",
    )
    parser.add_argument(
        "--walkers-per-cell",
        type=int,
        default=160,
        help="Random walkers per spatial cell. Higher means smoother curves.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Diffusion substeps per sequence interval (trajectory refinement).",
    )
    parser.add_argument(
        "--diffusion-time",
        type=float,
        default=25.0e-3,
        help="Diffusion time Delta (s) used for the b-sweep panel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed. Reused across gradients so each curve is smooth.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _run_pore_pgse(
    *,
    pore_width: float,
    gradient_amplitude: float,
    diffusion_time: float,
    num_cells: int,
    walkers_per_cell: int,
    substeps: int,
    seed: int,
):
    """Run one random-walker PGSE experiment in a slab pore (or free medium).

    A finite ``pore_width`` builds explicit field maps whose bounds are the pore
    walls, so walkers reflect off them. ``np.inf`` leaves ``fields=None`` and the
    workflow picks a box far larger than the diffusion length -> free diffusion.
    """

    from spin_dynamics.motion import make_motion_field_maps_2d
    from spin_dynamics.workflows import run_pgse_walkers

    restricted = np.isfinite(pore_width)
    width = pore_width if restricted else FREE_INIT_WIDTH
    half = 0.5 * float(width)
    x_axis = np.linspace(-half, half, int(num_cells))
    z_axis = np.array([-Z_HALF_WIDTH, Z_HALF_WIDTH])
    rho = np.ones((x_axis.size, z_axis.size), dtype=np.float64)

    # Reflecting walls only exist when we hand the engine a field-map box that
    # coincides with the pore. Free diffusion uses the default (very large) box.
    fields = make_motion_field_maps_2d(x_axis, z_axis) if restricted else None

    return run_pgse_walkers(
        rho=rho,
        x_axis=x_axis,
        z_axis=z_axis,
        fields=fields,
        gradient_amplitude=float(gradient_amplitude),
        gradient_duration=GRADIENT_DURATION,
        diffusion_time=float(diffusion_time),
        diffusion_coefficient=D_FREE,
        gamma=GAMMA,
        gradient_axis="x",
        walkers_per_cell=int(walkers_per_cell),
        seed=int(seed),
        jitter=True,
        boundary="reflect",
        substeps_per_interval=int(substeps),
    )


def _sweep_attenuation(args: argparse.Namespace):
    """Echo attenuation E(b) versus pore size at fixed diffusion time."""

    gradients = np.linspace(0.0, 0.30, 10)
    b_values = np.zeros_like(gradients)
    attenuation: dict[str, np.ndarray] = {}

    for label, pore_width in PORE_CONDITIONS:
        echo = np.zeros_like(gradients)
        for index, gradient in enumerate(gradients):
            result = _run_pore_pgse(
                pore_width=pore_width,
                gradient_amplitude=float(gradient),
                diffusion_time=args.diffusion_time,
                num_cells=args.num_cells,
                walkers_per_cell=args.walkers_per_cell,
                substeps=args.substeps,
                seed=args.seed,
            )
            echo[index] = float(np.abs(result.signal[0]))
            b_values[index] = result.b_value
        attenuation[label] = echo / max(echo[0], np.finfo(float).eps)
    return gradients, b_values, attenuation


def _sweep_apparent_diffusion(args: argparse.Namespace):
    """Apparent diffusion D_app(Delta) for free diffusion versus a 10 um pore."""

    diffusion_times = np.linspace(10.0e-3, 45.0e-3, 8)
    probe_gradient = 0.22  # T/m, keeps b*D in a well-conditioned range
    conditions = [("free", np.inf), ("10 um pore", 10.0e-6)]
    d_app: dict[str, np.ndarray] = {}

    for label, pore_width in conditions:
        values = np.zeros_like(diffusion_times)
        for index, delta_big in enumerate(diffusion_times):
            # Baseline (b = 0) and gradient-on echoes share walker paths (same
            # seed), so their ratio isolates the diffusion-driven attenuation.
            baseline = _run_pore_pgse(
                pore_width=pore_width,
                gradient_amplitude=0.0,
                diffusion_time=float(delta_big),
                num_cells=args.num_cells,
                walkers_per_cell=args.walkers_per_cell,
                substeps=args.substeps,
                seed=args.seed,
            )
            probe = _run_pore_pgse(
                pore_width=pore_width,
                gradient_amplitude=probe_gradient,
                diffusion_time=float(delta_big),
                num_cells=args.num_cells,
                walkers_per_cell=args.walkers_per_cell,
                substeps=args.substeps,
                seed=args.seed,
            )
            echo0 = float(np.abs(baseline.signal[0]))
            echo = float(np.abs(probe.signal[0]))
            attenuation = max(echo / max(echo0, np.finfo(float).eps), np.finfo(float).eps)
            values[index] = -np.log(attenuation) / probe.b_value
        d_app[label] = values
    return diffusion_times, d_app


def _displacement_histogram(args: argparse.Namespace):
    """Net walker displacement along the gradient axis: free versus 6 um pore."""

    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, pore_width in (("free", np.inf), ("6 um pore", 6.0e-6)):
        result = _run_pore_pgse(
            pore_width=pore_width,
            gradient_amplitude=0.0,  # trajectories are independent of gradient
            diffusion_time=args.diffusion_time,
            num_cells=args.num_cells,
            walkers_per_cell=args.walkers_per_cell,
            substeps=args.substeps,
            seed=args.seed,
        )
        start = result.initial_ensemble.positions[:, 0]
        end = result.sequence.final_ensemble.positions[:, 0]
        weights = result.initial_ensemble.weights
        samples[label] = ((end - start) * 1e6, weights)  # micrometers
    return samples


def _plot_results(
    plt,
    *,
    b_values: np.ndarray,
    attenuation: dict[str, np.ndarray],
    diffusion_times: np.ndarray,
    d_app: dict[str, np.ndarray],
    displacements: dict[str, tuple[np.ndarray, np.ndarray]],
    diffusion_time: float,
):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    # Panel 1: attenuation curves bending below the free Stejskal-Tanner line.
    b_axis = b_values * 1e-9
    for (label, _width), color in zip(PORE_CONDITIONS, PORE_COLORS):
        axes[0].semilogy(
            b_axis, attenuation[label], "o-", color=color, markersize=4, label=label
        )
    free_line = np.exp(-b_values * D_FREE)
    axes[0].semilogy(
        b_axis, free_line, "k--", linewidth=1.2, alpha=0.7, label="exp(-b D) theory"
    )
    axes[0].set_xlabel("b (10^9 s/m^2)")
    axes[0].set_ylabel("E = |S(b)| / |S(0)|")
    axes[0].set_title(f"PGSE attenuation, Delta = {diffusion_time * 1e3:.0f} ms")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize="small")

    # Panel 2: apparent diffusion coefficient falling with diffusion time.
    delta_axis = diffusion_times * 1e3
    axes[1].plot(
        delta_axis, d_app["free"] / D_FREE, "ks-", markersize=4, label="free"
    )
    axes[1].plot(
        delta_axis,
        d_app["10 um pore"] / D_FREE,
        "o-",
        color="#2ca02c",
        markersize=4,
        label="10 um pore",
    )
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("diffusion time Delta (ms)")
    axes[1].set_ylabel("D_app / D_free")
    axes[1].set_title("Apparent diffusion vs. diffusion time")
    axes[1].set_ylim(0.0, 1.15)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize="small")

    # Panel 3: displacement distribution, Gaussian (free) vs. clamped (pore).
    bins = np.linspace(-18.0, 18.0, 61)
    for label, color in (("free", "k"), ("6 um pore", "#d62728")):
        disp, weights = displacements[label]
        axes[2].hist(
            disp,
            bins=bins,
            weights=weights,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=label,
        )
    # Net displacement in a 6 um pore is bounded wall-to-wall by +/- L.
    axes[2].axvline(-6.0, color="#d62728", linestyle=":", linewidth=1.0)
    axes[2].axvline(6.0, color="#d62728", linestyle=":", linewidth=1.0)
    axes[2].set_xlabel("net displacement along gradient (um)")
    axes[2].set_ylabel("probability density")
    axes[2].set_title("Walker displacement distribution")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize="small")

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    plt = load_matplotlib(headless=bool(args.output))

    gradients, b_values, attenuation = _sweep_attenuation(args)
    diffusion_times, d_app = _sweep_apparent_diffusion(args)
    displacements = _displacement_histogram(args)

    diffusion_length = np.sqrt(2.0 * D_FREE * args.diffusion_time)
    print(f"b range: {b_values[0]:.3e} to {b_values[-1]:.3e} s/m^2")
    print(f"free-diffusion length sqrt(2 D Delta): {diffusion_length * 1e6:.1f} um")
    for label, _width in PORE_CONDITIONS:
        print(f"  E(b_max) [{label}]: {attenuation[label][-1]:.3f}")
    print(
        "D_app/D_free at longest Delta: "
        f"free {d_app['free'][-1] / D_FREE:.2f}, "
        f"10 um pore {d_app['10 um pore'][-1] / D_FREE:.2f}"
    )

    fig = _plot_results(
        plt,
        b_values=b_values,
        attenuation=attenuation,
        diffusion_times=diffusion_times,
        d_app=d_app,
        displacements=displacements,
        diffusion_time=args.diffusion_time,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
