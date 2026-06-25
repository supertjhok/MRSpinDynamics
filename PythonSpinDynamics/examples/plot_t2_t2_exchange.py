"""T2-T2 relaxation exchange spectroscopy (REXSY) with Bloch-McConnell sites.

REXSY encodes transverse relaxation, stores magnetization along z for a mixing
interval during which spins may change site, then encodes transverse relaxation
again. Spins that keep the same T2 across the mixing interval give peaks on the
diagonal of the recovered T2-T2 map; spins that hop to a site with a different
T2 give off-diagonal cross peaks whose intensity grows with the exchange rate.
This is the relaxation analogue of the diffusion-exchange (DEXSY) example.

This script builds an analytic two-site Bloch-McConnell system, simulates the
encode-mix-detect data with ``simulate_relaxation_exchange_2d``, and inverts it
to a T2-T2 map with the non-negative 2D inverse Laplace transform. It also prints
the longitudinal mixing propagator so the exchanged fraction is explicit.

Run with ``--output t2_t2_exchange.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


@dataclass(frozen=True)
class RexsySimulation:
    """Simulated REXSY data, recovered T2-T2 map, and diagnostics."""

    data: np.ndarray
    encode_times: np.ndarray
    detect_times: np.ndarray
    t2_axis: np.ndarray
    recovered: np.ndarray
    mixing_propagator: np.ndarray
    t2_fast: float
    t2_slow: float
    residual_fraction: float
    nonnegative: bool


def _has_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except ImportError:
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate a two-site T2-T2 relaxation exchange (REXSY) data set and "
            "invert it into a T2-T2 map showing diagonal and exchange cross peaks."
        )
    )
    parser.add_argument(
        "--t2-fast-ms",
        type=float,
        default=10.0,
        help="Transverse relaxation time of the fast-relaxing site (ms).",
    )
    parser.add_argument(
        "--t2-slow-ms",
        type=float,
        default=200.0,
        help="Transverse relaxation time of the slow-relaxing site (ms).",
    )
    parser.add_argument(
        "--exchange-rate",
        type=float,
        default=8.0,
        help="Symmetric site exchange rate constant in s^-1 (0 for no exchange).",
    )
    parser.add_argument(
        "--population-fast",
        type=float,
        default=0.5,
        help="Equilibrium population fraction of the fast-relaxing site.",
    )
    parser.add_argument(
        "--mixing-time-ms",
        type=float,
        default=60.0,
        help="Longitudinal mixing interval during which exchange occurs (ms).",
    )
    parser.add_argument(
        "--encode-points",
        type=int,
        default=28,
        help="Number of encode echo times along the first dimension.",
    )
    parser.add_argument(
        "--detect-points",
        type=int,
        default=28,
        help="Number of detect echo times along the second dimension.",
    )
    parser.add_argument(
        "--t2-points",
        type=int,
        default=48,
        help="Number of T2-axis points used for the 2D inversion.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-3,
        help="Tikhonov regularization strength applied on both T2 axes.",
    )
    parser.add_argument(
        "--regularization-order",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Penalty order for the inverse Laplace transform.",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help=(
            "Use unconstrained least squares for the inversion. By default the "
            "example uses non-negative ILT, which requires SciPy."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _simulate(args: argparse.Namespace, *, nonnegative: bool) -> RexsySimulation:
    from spin_dynamics.analysis import invert_t2_t2
    from spin_dynamics.exchange import simulate_relaxation_exchange_2d, two_site_exchange

    t2_fast = float(args.t2_fast_ms) * 1e-3
    t2_slow = float(args.t2_slow_ms) * 1e-3
    if t2_fast <= 0.0 or t2_slow <= 0.0:
        raise ValueError("T2 values must be positive")
    population_fast = float(args.population_fast)
    if not 0.0 < population_fast < 1.0:
        raise ValueError("population-fast must lie strictly between 0 and 1")

    system = two_site_exchange(
        offset_a_hz=0.0,
        offset_b_hz=0.0,
        k_ab_hz=float(args.exchange_rate),
        population_a=population_fast,
        t2_a_seconds=t2_fast,
        t2_b_seconds=t2_slow,
        labels=("fast", "slow"),
    )

    encode = np.linspace(0.0, 6.0 * t2_fast, int(args.encode_points))
    detect = np.linspace(0.0, 4.0 * t2_slow, int(args.detect_points))
    result = simulate_relaxation_exchange_2d(
        system,
        encode,
        detect,
        mixing_time=float(args.mixing_time_ms) * 1e-3,
    )

    t2_axis = np.logspace(
        np.log10(0.3 * t2_fast), np.log10(3.0 * t2_slow), int(args.t2_points)
    )
    ilt = invert_t2_t2(
        result.data,
        encode,
        detect,
        t2_axis,
        regularization=float(args.regularization),
        regularization_order=int(args.regularization_order),
        nonnegative=nonnegative,
    )
    residual_fraction = ilt.residual_norm / max(
        float(np.linalg.norm(result.data)), np.finfo(float).eps
    )
    return RexsySimulation(
        data=result.data,
        encode_times=encode,
        detect_times=detect,
        t2_axis=t2_axis,
        recovered=ilt.distribution,
        mixing_propagator=result.mixing_propagator,
        t2_fast=t2_fast,
        t2_slow=t2_slow,
        residual_fraction=residual_fraction,
        nonnegative=nonnegative,
    )


def _plot_results(plt, sim: RexsySimulation):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))

    image = axes[0].imshow(
        sim.data,
        origin="lower",
        extent=[
            sim.detect_times[0] * 1e3,
            sim.detect_times[-1] * 1e3,
            sim.encode_times[0] * 1e3,
            sim.encode_times[-1] * 1e3,
        ],
        aspect="auto",
        cmap="magma",
    )
    axes[0].set_xlabel("detect echo time (ms)")
    axes[0].set_ylabel("encode echo time (ms)")
    axes[0].set_title("REXSY signal S(t1, t2)")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    display = sim.recovered if sim.nonnegative else np.clip(sim.recovered, 0.0, None)
    display = display / max(float(np.max(display)), np.finfo(float).eps)
    mesh = axes[1].pcolormesh(
        sim.t2_axis * 1e3,
        sim.t2_axis * 1e3,
        display,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("detect T2 (ms)")
    axes[1].set_ylabel("encode T2 (ms)")
    for value in (sim.t2_fast, sim.t2_slow):
        axes[1].axvline(value * 1e3, color="white", lw=0.6, alpha=0.4)
        axes[1].axhline(value * 1e3, color="white", lw=0.6, alpha=0.4)
    solver = "NNLS" if sim.nonnegative else "LS preview"
    axes[1].set_title(f"Recovered T2-T2 map ({solver})")
    fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].text(
        0.5,
        -0.24,
        f"relative ILT residual: {sim.residual_fraction:.3f}",
        ha="center",
        va="top",
        transform=axes[1].transAxes,
    )

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

    plt = load_matplotlib(headless=bool(args.output))
    sim = _simulate(args, nonnegative=nonnegative)

    print(f"fast site T2: {sim.t2_fast * 1e3:.1f} ms")
    print(f"slow site T2: {sim.t2_slow * 1e3:.1f} ms")
    print(
        "exchanged fraction during mixing: "
        f"fast->slow {sim.mixing_propagator[1, 0]:.3f}, "
        f"slow->fast {sim.mixing_propagator[0, 1]:.3f}"
    )
    print(f"relative ILT residual: {sim.residual_fraction:.3f}")

    fig = _plot_results(plt, sim)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
