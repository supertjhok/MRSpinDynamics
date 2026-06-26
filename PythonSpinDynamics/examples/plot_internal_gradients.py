"""Susceptibility-induced internal gradients in a packed-grain pore space.

Magnetic-susceptibility contrast between solid grains and pore fluid sets up
internal field gradients that are usually the dominant inhomogeneity in
porous-media NMR. This example builds the internal off-resonance field for an
array of cylindrical grains, summarizes the pore-space internal-gradient
distribution, and then drives a CPMG echo train of diffusing walkers through the
internal field with no applied gradient. Because the decay comes purely from
diffusion through the internal gradient, longer echo spacings decay faster --
the classic g_internal signature that motivates background-gradient suppression.

Run with ``--output internal_gradients.png`` to save, or omit it to show.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675222005e8
DIFFUSION = 2.3e-9


@dataclass(frozen=True)
class InternalGradientSimulation:
    """Internal field, gradient distribution, and CPMG decays."""

    x_axis: np.ndarray
    z_axis: np.ndarray
    offresonance_hz: np.ndarray
    inclusion_mask: np.ndarray
    bin_edges: np.ndarray
    histogram: np.ndarray
    gradient_rms: float
    gradient_mean: float
    echo_spacings: np.ndarray
    echo_numbers: np.ndarray
    decays: list[np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the internal susceptibility field of a packed-grain pore "
            "space, summarize its internal-gradient distribution, and show "
            "echo-spacing-dependent CPMG decay from diffusion in that field."
        )
    )
    parser.add_argument(
        "--grain-radius-um",
        type=float,
        default=12.0,
        help="Cylindrical grain radius in micrometres.",
    )
    parser.add_argument(
        "--susceptibility",
        type=float,
        default=1.0e-6,
        help="Grain-minus-fluid SI volume susceptibility contrast.",
    )
    parser.add_argument(
        "--b0-tesla",
        type=float,
        default=2.0,
        help="Static field strength in tesla.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=161,
        help="Field-map samples along each axis.",
    )
    parser.add_argument(
        "--walkers",
        type=int,
        default=600,
        help="Number of diffusing walkers in the CPMG train.",
    )
    parser.add_argument(
        "--num-echoes",
        type=int,
        default=24,
        help="Number of CPMG echoes.",
    )
    parser.add_argument(
        "--echo-spacings-ms",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 6.0],
        help="Echo spacings (ms) to compare.",
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=6,
        help="Motion substeps per CPMG interval.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for walker placement and Brownian steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _grain_centers(half: float, radius: float) -> list[tuple[float, float]]:
    # a 3x3 grain lattice with the central grain removed to leave a connected pore
    pitch = 2.6 * radius
    centers = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if i == 0 and j == 0:
                continue
            centers.append((i * pitch, j * pitch))
    return [c for c in centers if abs(c[0]) <= half and abs(c[1]) <= half]


def _build_field(args: argparse.Namespace):
    from spin_dynamics.susceptibility import (
        CylindricalInclusion,
        internal_gradient_distribution,
        susceptibility_offresonance_map,
    )

    radius = float(args.grain_radius_um) * 1e-6
    half = 4.0 * radius
    axis = np.linspace(-half, half, int(args.grid))
    inclusions = [
        CylindricalInclusion(cx, cz, radius)
        for cx, cz in _grain_centers(half, radius)
    ]
    field = susceptibility_offresonance_map(
        axis,
        axis,
        inclusions,
        b0_tesla=float(args.b0_tesla),
        susceptibility_difference=float(args.susceptibility),
        gamma=GAMMA,
    )
    distribution = internal_gradient_distribution(field, bins=48)
    return field, distribution, inclusions, half


def _place_walkers(args, inclusions, half, radius):
    from spin_dynamics.motion import ParticleEnsemble

    rng = np.random.default_rng(args.seed)
    margin = half - 0.05 * radius
    positions = np.empty((0, 2), dtype=np.float64)
    while positions.shape[0] < int(args.walkers):
        trial = rng.uniform(-margin, margin, size=(2 * int(args.walkers), 2))
        keep = np.ones(trial.shape[0], dtype=bool)
        for inc in inclusions:
            dx = trial[:, 0] - inc.center_x
            dz = trial[:, 1] - inc.center_z
            keep &= (dx * dx + dz * dz) > inc.radius**2
        positions = np.vstack((positions, trial[keep]))
    positions = positions[: int(args.walkers)]

    weights = np.full(positions.shape[0], 1.0 / positions.shape[0])
    magnetization = np.zeros((3, positions.shape[0]), dtype=np.complex128)
    magnetization[0, :] = 1.0
    diffusion = np.full(positions.shape[0], DIFFUSION)
    return ParticleEnsemble(positions, magnetization, weights, diffusion)


def _grain_boundary(inclusions):
    from spin_dynamics.motion import apply_boundary

    def boundary(positions, *, previous_positions=None, bounds=None, **_kwargs):
        pos = np.array(positions, dtype=np.float64, copy=True)
        if bounds is not None:
            pos = apply_boundary(pos, bounds, "reflect")
        prev = previous_positions
        for inc in inclusions:
            dx = pos[:, 0] - inc.center_x
            dz = pos[:, 1] - inc.center_z
            r2 = dx * dx + dz * dz
            inside = r2 < inc.radius**2
            if not np.any(inside):
                continue
            if prev is not None:
                pos[inside] = prev[inside]  # reflect by rejecting the step
            else:
                r = np.sqrt(r2[inside])
                r = np.where(r == 0.0, inc.radius, r)
                scale = inc.radius / r
                pos[inside, 0] = inc.center_x + dx[inside] * scale
                pos[inside, 1] = inc.center_z + dz[inside] * scale
        return pos

    return boundary


def _run_cpmg(args, field, inclusions, half, radius):
    from spin_dynamics.sequences.motion import run_motion_cpmg_sequence
    from spin_dynamics.susceptibility import make_susceptibility_field_maps

    maps = make_susceptibility_field_maps(field)
    boundary = _grain_boundary(inclusions)
    spacings = np.asarray(args.echo_spacings_ms, dtype=np.float64) * 1e-3
    decays = []
    for spacing in spacings:
        ensemble = _place_walkers(args, inclusions, half, radius)
        result = run_motion_cpmg_sequence(
            ensemble,
            maps,
            num_echoes=int(args.num_echoes),
            echo_spacing=float(spacing),
            excitation_duration=min(40e-6, 0.2 * spacing),
            refocusing_duration=min(80e-6, 0.3 * spacing),
            gradient=(0.0, 0.0),
            rng=np.random.default_rng(args.seed + 1),
            boundary=boundary,
            substeps_per_interval=int(args.substeps),
        )
        amplitude = np.abs(result.signal)
        amplitude = amplitude / max(amplitude[0], np.finfo(float).eps)
        decays.append(amplitude)
    return spacings, np.arange(1, int(args.num_echoes) + 1), decays


def _simulate(args: argparse.Namespace) -> InternalGradientSimulation:
    field, distribution, inclusions, half = _build_field(args)
    radius = float(args.grain_radius_um) * 1e-6
    spacings, echo_numbers, decays = _run_cpmg(args, field, inclusions, half, radius)
    return InternalGradientSimulation(
        x_axis=field.x_axis,
        z_axis=field.z_axis,
        offresonance_hz=field.offresonance_hz,
        inclusion_mask=field.inclusion_mask,
        bin_edges=distribution.bin_edges,
        histogram=distribution.histogram,
        gradient_rms=distribution.rms,
        gradient_mean=distribution.mean,
        echo_spacings=spacings,
        echo_numbers=echo_numbers,
        decays=decays,
    )


def _plot_results(plt, sim: InternalGradientSimulation):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    display = np.where(sim.inclusion_mask, np.nan, sim.offresonance_hz)
    extent = [
        sim.z_axis[0] * 1e6,
        sim.z_axis[-1] * 1e6,
        sim.x_axis[0] * 1e6,
        sim.x_axis[-1] * 1e6,
    ]
    limit = float(np.nanmax(np.abs(display)))
    image = axes[0].imshow(
        display,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[0].set_xlabel("z (um)")
    axes[0].set_ylabel("x (um)")
    axes[0].set_title("Internal off-resonance (Hz)")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    centers = 0.5 * (sim.bin_edges[:-1] + sim.bin_edges[1:])
    width = np.diff(sim.bin_edges)
    axes[1].bar(centers * 1e3, sim.histogram, width=width * 1e3, color="#3b6ea5")
    axes[1].axvline(
        sim.gradient_rms * 1e3,
        color="crimson",
        lw=1.4,
        label=f"rms {sim.gradient_rms * 1e3:.2f} mT/m",
    )
    axes[1].set_xlabel("internal gradient |g| (mT/m)")
    axes[1].set_ylabel("pore-space weight")
    axes[1].set_title("Internal-gradient distribution")
    axes[1].legend()

    for spacing, decay in zip(sim.echo_spacings, sim.decays):
        axes[2].semilogy(
            sim.echo_numbers,
            decay,
            marker="o",
            ms=3,
            label=f"TE = {spacing * 1e3:.1f} ms",
        )
    axes[2].set_xlabel("echo number")
    axes[2].set_ylabel("normalized |echo|")
    axes[2].set_title("CPMG decay in the internal field")
    axes[2].legend()

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 0:
        raise SystemExit("--num-echoes must be positive")
    if any(s <= 0.0 for s in args.echo_spacings_ms):
        raise SystemExit("--echo-spacings-ms must be positive")

    plt = load_matplotlib(headless=bool(args.output))
    sim = _simulate(args)

    print(f"internal-gradient rms:  {sim.gradient_rms * 1e3:.3f} mT/m")
    print(f"internal-gradient mean: {sim.gradient_mean * 1e3:.3f} mT/m")
    print("final |echo| by echo spacing:")
    for spacing, decay in zip(sim.echo_spacings, sim.decays):
        print(f"  TE = {spacing * 1e3:.1f} ms -> {decay[-1]:.3f}")

    fig = _plot_results(plt, sim)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
