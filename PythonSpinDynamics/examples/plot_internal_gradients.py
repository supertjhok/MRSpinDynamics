"""Susceptibility-induced internal gradients in a packed-grain pore space.

Magnetic-susceptibility contrast between solid grains and pore fluid sets up
internal field gradients that are usually the dominant inhomogeneity in
porous-media NMR. This example builds the internal off-resonance field for an
array of cylindrical grains, summarizes the pore-space internal-gradient
distribution, and then shows the diagnostic that actually distinguishes an
internal gradient from an applied one.

Note on what is and is not a signature. The acceleration of CPMG decay with
echo spacing T_E is the general diffusion-in-a-gradient effect (the decay rate
grows like G^2 D T_E^2); it occurs for a uniform applied or static gradient just
as much as for an internal one, so it is not by itself evidence of internal
gradients. What *is* a signature of susceptibility-induced gradients is their
scaling with the static field: because G_internal is proportional to
delta_chi * B0, the diffusion-induced decay rate scales as B0^2, whereas an
applied gradient is set by hardware and is independent of B0. The third panel
runs a CPMG train at fixed T_E for several B0 values and recovers that B0^2
slope. The broad pore-space gradient *distribution* in the second panel is a
second distinguishing feature -- a uniform gradient would be a single value.

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
    """Internal field, gradient distribution, and B0 scaling of the decay."""

    x_axis: np.ndarray
    z_axis: np.ndarray
    offresonance_hz: np.ndarray
    inclusion_mask: np.ndarray
    bin_edges: np.ndarray
    histogram: np.ndarray
    gradient_rms: float
    gradient_mean: float
    reference_b0: float
    echo_spacing: float
    b0_values: np.ndarray
    decay_rates: np.ndarray
    b0_scaling_slope: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the internal susceptibility field of a packed-grain pore "
            "space, summarize its internal-gradient distribution, and show the "
            "B0^2 scaling of diffusion-induced CPMG decay that distinguishes an "
            "internal gradient from an applied one."
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
        help="Reference field for the displayed field and gradient panels (T).",
    )
    parser.add_argument(
        "--b0-values-tesla",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
        help="Static fields (T) used for the B0-scaling CPMG sweep.",
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
        help="Number of diffusing walkers in each CPMG train.",
    )
    parser.add_argument(
        "--num-echoes",
        type=int,
        default=20,
        help="Number of CPMG echoes per train.",
    )
    parser.add_argument(
        "--echo-spacing-ms",
        type=float,
        default=4.0,
        help="Fixed CPMG echo spacing for the B0 sweep (ms).",
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


def _inclusions(args: argparse.Namespace):
    from spin_dynamics.susceptibility import CylindricalInclusion

    radius = float(args.grain_radius_um) * 1e-6
    half = 4.0 * radius
    centers = _grain_centers(half, radius)
    inclusions = [CylindricalInclusion(cx, cz, radius) for cx, cz in centers]
    return inclusions, half, radius


def _build_field(args: argparse.Namespace, inclusions, half, *, b0: float):
    from spin_dynamics.susceptibility import susceptibility_offresonance_map

    axis = np.linspace(-half, half, int(args.grid))
    return susceptibility_offresonance_map(
        axis,
        axis,
        inclusions,
        b0_tesla=float(b0),
        susceptibility_difference=float(args.susceptibility),
        gamma=GAMMA,
    )


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
                pos[inside] = prev[inside]  # reject the step into the grain
            else:
                r = np.sqrt(r2[inside])
                r = np.where(r == 0.0, inc.radius, r)
                scale = inc.radius / r
                pos[inside, 0] = inc.center_x + dx[inside] * scale
                pos[inside, 1] = inc.center_z + dz[inside] * scale
        return pos

    return boundary


def _decay_rate(args, field, inclusions, half, radius) -> float:
    """Run a fixed-TE CPMG train and fit the diffusion-induced decay rate (1/s)."""

    from spin_dynamics.sequences.motion import run_motion_cpmg_sequence
    from spin_dynamics.susceptibility import make_susceptibility_field_maps

    maps = make_susceptibility_field_maps(field)
    spacing = float(args.echo_spacing_ms) * 1e-3
    ensemble = _place_walkers(args, inclusions, half, radius)
    result = run_motion_cpmg_sequence(
        ensemble,
        maps,
        num_echoes=int(args.num_echoes),
        echo_spacing=spacing,
        excitation_duration=min(40e-6, 0.2 * spacing),
        refocusing_duration=min(80e-6, 0.3 * spacing),
        gradient=(0.0, 0.0),
        rng=np.random.default_rng(args.seed + 1),
        boundary=_grain_boundary(inclusions),
        substeps_per_interval=int(args.substeps),
    )
    amplitude = np.abs(result.signal)
    amplitude = amplitude / max(amplitude[0], np.finfo(float).eps)
    times = np.asarray(result.sample_times, dtype=np.float64)
    # T1=T2=inf, so the only decay is diffusion in the internal gradient
    slope = np.polyfit(times, np.log(np.clip(amplitude, 1e-12, None)), 1)[0]
    return float(-slope)


def _simulate(args: argparse.Namespace) -> InternalGradientSimulation:
    from spin_dynamics.susceptibility import internal_gradient_distribution

    inclusions, half, radius = _inclusions(args)
    reference = _build_field(args, inclusions, half, b0=float(args.b0_tesla))
    distribution = internal_gradient_distribution(reference, bins=48)

    b0_values = np.asarray(args.b0_values_tesla, dtype=np.float64)
    rates = np.array(
        [
            _decay_rate(
                args,
                _build_field(args, inclusions, half, b0=float(b0)),
                inclusions,
                half,
                radius,
            )
            for b0 in b0_values
        ]
    )
    positive = rates > 0.0
    if np.count_nonzero(positive) >= 2:
        slope = float(
            np.polyfit(np.log(b0_values[positive]), np.log(rates[positive]), 1)[0]
        )
    else:
        slope = float("nan")

    return InternalGradientSimulation(
        x_axis=reference.x_axis,
        z_axis=reference.z_axis,
        offresonance_hz=reference.offresonance_hz,
        inclusion_mask=reference.inclusion_mask,
        bin_edges=distribution.bin_edges,
        histogram=distribution.histogram,
        gradient_rms=distribution.rms,
        gradient_mean=distribution.mean,
        reference_b0=float(args.b0_tesla),
        echo_spacing=float(args.echo_spacing_ms) * 1e-3,
        b0_values=b0_values,
        decay_rates=rates,
        b0_scaling_slope=slope,
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
    axes[0].set_title(f"Internal off-resonance (Hz) at {sim.reference_b0:g} T")
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

    axes[2].loglog(
        sim.b0_values,
        sim.decay_rates,
        "o",
        ms=6,
        color="#2a7f3f",
        label="simulated R2,diff",
    )
    # reference B0^2 slope anchored at the first point
    ref = sim.decay_rates[0] * (sim.b0_values / sim.b0_values[0]) ** 2
    axes[2].loglog(sim.b0_values, ref, "--", color="gray", label="slope 2 (B0^2)")
    axes[2].set_xlabel("B0 (T)")
    axes[2].set_ylabel("diffusion decay rate (1/s)")
    axes[2].set_title(f"B0 scaling (fit slope {sim.b0_scaling_slope:.2f})")
    axes[2].legend()
    axes[2].text(
        0.5,
        -0.26,
        "internal gradient: R2,diff ~ B0^2; an applied gradient would be flat",
        ha="center",
        va="top",
        transform=axes[2].transAxes,
        fontsize=9,
    )

    fig.tight_layout()
    return fig


def main() -> None:
    args = _parse_args()
    if args.num_echoes <= 0:
        raise SystemExit("--num-echoes must be positive")
    if args.echo_spacing_ms <= 0.0:
        raise SystemExit("--echo-spacing-ms must be positive")
    if any(b <= 0.0 for b in args.b0_values_tesla):
        raise SystemExit("--b0-values-tesla must be positive")

    plt = load_matplotlib(headless=bool(args.output))
    sim = _simulate(args)

    print(f"internal-gradient rms at {sim.reference_b0:g} T:  "
          f"{sim.gradient_rms * 1e3:.3f} mT/m")
    print(f"internal-gradient mean at {sim.reference_b0:g} T: "
          f"{sim.gradient_mean * 1e3:.3f} mT/m")
    print(f"fixed echo spacing: {sim.echo_spacing * 1e3:.1f} ms")
    print("diffusion decay rate vs B0:")
    for b0, rate in zip(sim.b0_values, sim.decay_rates):
        print(f"  B0 = {b0:.2f} T -> R2,diff = {rate:.3f} 1/s")
    print(f"B0-scaling log-log slope: {sim.b0_scaling_slope:.2f} (expect ~2)")

    fig = _plot_results(plt, sim)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
        print(f"saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
