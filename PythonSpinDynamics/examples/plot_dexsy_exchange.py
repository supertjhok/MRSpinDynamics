"""Finite-pulse DEXSY-style diffusion-diffusion exchange map.

DEXSY measures a two-dimensional diffusion correlation: one diffusion encoding,
a mixing interval, then a second diffusion encoding. Spins that remain in the
same compartment give diagonal D-D peaks. Spins that cross a semi-permeable
membrane during the mixing interval give off-diagonal exchange peaks.

This example runs an explicit finite-pulse two-PGSE sequence with moving
walkers. The sample is a two-compartment slab: a narrow left compartment and a
wider right compartment share the same microscopic diffusion coefficient, but
the finite PGSE blocks see different apparent diffusivities because the narrow
side is more restricted. A semi-permeable internal plane lets walkers exchange
during the mixing interval.

Run with ``--output dexsy.png`` to save, or omit it to show interactively.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from _source_path import add_src_to_path, load_matplotlib


add_src_to_path()


GAMMA = 2.675e8
DIFFUSION = 2.3e-9
LEFT_WIDTH = 6.0e-6
RIGHT_WIDTH = 24.0e-6
Z_HALF_WIDTH = 0.5e-6
EXCITATION_DURATION = 80.0e-6
REFOCUSING_DURATION = 160.0e-6
GRADIENT_DURATION = 1.5e-3
DIFFUSION_TIME = 18.0e-3


@dataclass(frozen=True)
class DexsySimulation:
    """Finite-pulse DEXSY data and trajectory exchange diagnostics."""

    data: np.ndarray
    b_values: np.ndarray
    diffusion_axis: np.ndarray
    recovered: np.ndarray
    prediction: np.ndarray
    transition: np.ndarray
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
            "Run a finite-pulse two-PGSE DEXSY-style exchange simulation with "
            "semi-permeable compartments, then invert S(b1,b2) into a D-D map."
        )
    )
    parser.add_argument(
        "--walkers-per-cell",
        type=int,
        default=80,
        help="Random walkers per spatial cell.",
    )
    parser.add_argument(
        "--cells-left",
        type=int,
        default=8,
        help="Spatial cells across the narrow left compartment.",
    )
    parser.add_argument(
        "--cells-right",
        type=int,
        default=20,
        help="Spatial cells across the wider right compartment.",
    )
    parser.add_argument(
        "--mixing-time",
        type=float,
        default=90.0e-3,
        help="Exchange mixing interval between the two PGSE blocks (s).",
    )
    parser.add_argument(
        "--exchange-rate",
        type=float,
        default=55.0,
        help=(
            "Membrane transmission rate in s^-1. Use 0 for no exchange and "
            "larger values for stronger off-diagonal peaks."
        ),
    )
    parser.add_argument(
        "--substeps",
        type=int,
        default=8,
        help="Motion substeps per finite sequence interval.",
    )
    parser.add_argument(
        "--b-points",
        type=int,
        default=13,
        help="Number of b-values along each DEXSY dimension.",
    )
    parser.add_argument(
        "--b-max",
        type=float,
        default=2.1e9,
        help="Maximum per-block b-value on each axis (s/m^2).",
    )
    parser.add_argument(
        "--d-points",
        type=int,
        default=44,
        help="Number of diffusion-axis points used for the D-D inversion.",
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=3.0e-4,
        help="Tikhonov regularization strength used on both D-D axes.",
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
            "Use unconstrained least squares for the ILT. By default the "
            "example uses non-negative ILT, which requires SciPy."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for walker initialization and Brownian steps.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the output PNG. If omitted, show the plot.",
    )
    return parser.parse_args()


def _b_values_and_gradients(num_points: int, b_max: float) -> tuple[np.ndarray, np.ndarray]:
    from spin_dynamics.workflows import pgse_b_value

    if num_points < 2:
        raise ValueError("b-points must be at least 2")
    b_values = np.linspace(0.0, float(b_max), int(num_points), dtype=np.float64)
    unit_b = pgse_b_value(
        1.0,
        GRADIENT_DURATION,
        DIFFUSION_TIME,
        gamma=GAMMA,
    )
    gradients = np.sqrt(np.divide(b_values, unit_b, out=np.zeros_like(b_values), where=unit_b > 0.0))
    return b_values, gradients


def _make_ensemble(args: argparse.Namespace):
    from spin_dynamics.motion import ParticleEnsemble

    if args.walkers_per_cell <= 0:
        raise ValueError("walkers-per-cell must be positive")
    if args.cells_left <= 0 or args.cells_right <= 0:
        raise ValueError("cells-left and cells-right must be positive")

    rng = np.random.default_rng(args.seed)
    left_centers = np.linspace(
        -LEFT_WIDTH + 0.5 * LEFT_WIDTH / args.cells_left,
        -0.5 * LEFT_WIDTH / args.cells_left,
        int(args.cells_left),
    )
    right_centers = np.linspace(
        0.5 * RIGHT_WIDTH / args.cells_right,
        RIGHT_WIDTH - 0.5 * RIGHT_WIDTH / args.cells_right,
        int(args.cells_right),
    )
    x_centers = np.concatenate((left_centers, right_centers))
    base = np.column_stack((x_centers, np.zeros_like(x_centers)))
    positions = np.repeat(base, int(args.walkers_per_cell), axis=0)
    dx = np.concatenate(
        (
            np.full(args.cells_left, LEFT_WIDTH / args.cells_left),
            np.full(args.cells_right, RIGHT_WIDTH / args.cells_right),
        )
    )
    widths = np.column_stack((dx, np.full_like(dx, 2.0 * Z_HALF_WIDTH)))
    widths = np.repeat(widths, int(args.walkers_per_cell), axis=0)
    positions = positions + rng.uniform(-0.5, 0.5, size=positions.shape) * widths
    positions[:, 0] = np.clip(positions[:, 0], -LEFT_WIDTH, RIGHT_WIDTH)
    positions[:, 1] = np.clip(positions[:, 1], -Z_HALF_WIDTH, Z_HALF_WIDTH)

    weights = np.full(positions.shape[0], 1.0 / positions.shape[0], dtype=np.float64)
    magnetization = np.zeros((3, positions.shape[0]), dtype=np.complex128)
    magnetization[0, :] = 1.0
    diffusion = np.full(positions.shape[0], DIFFUSION, dtype=np.float64)
    return ParticleEnsemble(positions, magnetization, weights, diffusion)


def _make_fields():
    from spin_dynamics.motion import make_motion_field_maps_2d

    return make_motion_field_maps_2d(
        [-LEFT_WIDTH, RIGHT_WIDTH],
        [-Z_HALF_WIDTH, Z_HALF_WIDTH],
    )


def _make_boundary(exchange_rate: float):
    from spin_dynamics.motion import make_semipermeable_plane

    return make_semipermeable_plane(
        0.0,
        exchange_rate=float(exchange_rate),
        axis="x",
        outer_boundary="reflect",
    )


def _pgse_block(
    *,
    gradient: tuple[float, float],
    index: int,
    acquire: bool,
    substeps: int,
):
    from spin_dynamics.sequences.motion import MotionSequenceStep

    gap = 0.5 * (DIFFUSION_TIME - GRADIENT_DURATION - REFOCUSING_DURATION)
    if gap <= 0.0:
        raise ValueError("diffusion timing leaves no positive PGSE gap")
    return [
        MotionSequenceStep(
            duration=GRADIENT_DURATION,
            gradient=gradient,
            substeps=substeps,
            label=f"encoding_{index}_lobe_1",
        ),
        MotionSequenceStep(
            duration=gap,
            substeps=substeps,
            label=f"encoding_{index}_gap_1",
        ),
        MotionSequenceStep(
            duration=REFOCUSING_DURATION,
            rf_amplitude=np.pi / REFOCUSING_DURATION,
            rf_phase=0.0,
            substeps=max(1, substeps),
            label=f"encoding_{index}_180",
        ),
        MotionSequenceStep(
            duration=gap,
            substeps=substeps,
            label=f"encoding_{index}_gap_2",
        ),
        MotionSequenceStep(
            duration=GRADIENT_DURATION,
            gradient=gradient,
            acquire=acquire,
            num_samples=1 if acquire else 0,
            substeps=substeps,
            label="dexsy_echo" if acquire else f"encoding_{index}_lobe_2",
        ),
    ]


def _make_dexsy_steps(
    gradient1: float,
    gradient2: float,
    *,
    mixing_time: float,
    substeps: int,
):
    from spin_dynamics.sequences.motion import MotionSequenceStep

    if mixing_time < 0.0:
        raise ValueError("mixing-time must be non-negative")
    steps = [
        MotionSequenceStep(
            duration=EXCITATION_DURATION,
            rf_amplitude=(0.5 * np.pi) / EXCITATION_DURATION,
            rf_phase=np.pi / 2,
            substeps=max(1, substeps),
            label="excitation_90",
        )
    ]
    steps.extend(
        _pgse_block(
            gradient=(GAMMA * float(gradient1), 0.0),
            index=1,
            acquire=False,
            substeps=substeps,
        )
    )
    if mixing_time > 0.0:
        steps.append(
            MotionSequenceStep(
                duration=float(mixing_time),
                substeps=substeps,
                label="exchange_mixing",
            )
        )
    steps.extend(
        _pgse_block(
            gradient=(GAMMA * float(gradient2), 0.0),
            index=2,
            acquire=True,
            substeps=substeps,
        )
    )
    return tuple(steps)


def _run_sequence_signal(
    args: argparse.Namespace,
    *,
    gradient1: float,
    gradient2: float,
) -> complex:
    from spin_dynamics.sequences.motion import run_motion_sequence

    ensemble = _make_ensemble(args)
    fields = _make_fields()
    boundary = _make_boundary(args.exchange_rate)
    steps = _make_dexsy_steps(
        gradient1,
        gradient2,
        mixing_time=float(args.mixing_time),
        substeps=int(args.substeps),
    )
    sequence = run_motion_sequence(
        ensemble,
        fields,
        steps,
        rng=np.random.default_rng(args.seed),
        t1=np.inf,
        t2=np.inf,
        boundary=boundary,
        default_substeps=int(args.substeps),
    )
    return complex(sequence.signal[-1])


def _simulate_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    b_values, gradients = _b_values_and_gradients(args.b_points, args.b_max)
    data = np.zeros((b_values.size, b_values.size), dtype=np.float64)
    baseline = abs(_run_sequence_signal(args, gradient1=0.0, gradient2=0.0))
    baseline = max(baseline, np.finfo(float).eps)
    for i, g1 in enumerate(gradients):
        for j, g2 in enumerate(gradients):
            signal = _run_sequence_signal(args, gradient1=float(g1), gradient2=float(g2))
            data[i, j] = abs(signal) / baseline
    data[0, 0] = 1.0
    return np.clip(data, 0.0, None), b_values


def _advance_positions(
    positions: np.ndarray,
    *,
    duration: float,
    substeps: int,
    boundary,
    rng: np.random.Generator,
) -> np.ndarray:
    from spin_dynamics.motion import advect_diffuse_positions

    if duration <= 0.0:
        return positions.copy()
    dt = float(duration) / int(substeps)
    pos = positions.copy()
    bounds = ((-LEFT_WIDTH, RIGHT_WIDTH), (-Z_HALF_WIDTH, Z_HALF_WIDTH))
    for step in range(int(substeps)):
        pos = advect_diffuse_positions(
            pos,
            dt,
            diffusion_coefficient=DIFFUSION,
            rng=rng,
            time=step * dt,
            bounds=bounds,
            boundary=boundary,
        )
    return pos


def _trajectory_transition(args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    ensemble = _make_ensemble(args)
    boundary = _make_boundary(args.exchange_rate)
    start_side = ensemble.positions[:, 0] >= 0.0
    block_duration = (
        2.0 * GRADIENT_DURATION
        + 2.0 * 0.5 * (DIFFUSION_TIME - GRADIENT_DURATION - REFOCUSING_DURATION)
        + REFOCUSING_DURATION
    )
    after_first = _advance_positions(
        ensemble.positions,
        duration=block_duration,
        substeps=max(1, 5 * int(args.substeps)),
        boundary=boundary,
        rng=rng,
    )
    before_second = _advance_positions(
        after_first,
        duration=float(args.mixing_time),
        substeps=max(1, int(args.substeps)),
        boundary=boundary,
        rng=rng,
    )
    second_side = before_second[:, 0] >= 0.0
    matrix = np.zeros((2, 2), dtype=np.float64)
    for start_fast in (False, True):
        row = 1 if start_fast else 0
        mask = start_side == start_fast
        denom = float(np.mean(mask))
        if denom == 0.0:
            continue
        for second_fast in (False, True):
            col = 1 if second_fast else 0
            matrix[row, col] = float(np.mean(mask & (second_side == second_fast))) / denom
    return matrix


def _invert_dd(
    data: np.ndarray,
    b_values: np.ndarray,
    diffusion_axis: np.ndarray,
    *,
    regularization: float,
    regularization_order: int,
    nonnegative: bool,
):
    from spin_dynamics.analysis import invert_laplace_2d

    return invert_laplace_2d(
        data,
        b_values,
        b_values,
        diffusion_axis,
        diffusion_axis,
        kernel1="diffusion",
        kernel2="diffusion",
        regularization=(float(regularization), float(regularization)),
        regularization_order=int(regularization_order),
        nonnegative=nonnegative,
    )


def _simulate(args: argparse.Namespace, *, nonnegative: bool) -> DexsySimulation:
    data, b_values = _simulate_data(args)
    diffusion_axis = np.linspace(0.05e-9, 2.6e-9, int(args.d_points))
    result = _invert_dd(
        data,
        b_values,
        diffusion_axis,
        regularization=float(args.regularization),
        regularization_order=int(args.regularization_order),
        nonnegative=nonnegative,
    )
    residual_fraction = result.residual_norm / max(
        float(np.linalg.norm(data)), np.finfo(float).eps
    )
    return DexsySimulation(
        data=data,
        b_values=b_values,
        diffusion_axis=diffusion_axis,
        recovered=result.distribution,
        prediction=result.prediction,
        transition=_trajectory_transition(args),
        residual_fraction=residual_fraction,
        nonnegative=nonnegative,
    )


def _plot_results(plt, sim: DexsySimulation):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))

    b_extent = [
        sim.b_values[0] * 1e-9,
        sim.b_values[-1] * 1e-9,
        sim.b_values[0] * 1e-9,
        sim.b_values[-1] * 1e-9,
    ]
    image = axes[0].imshow(
        sim.data,
        origin="lower",
        extent=b_extent,
        aspect="auto",
        cmap="magma",
    )
    axes[0].set_xlabel("b2 (10^9 s/m^2)")
    axes[0].set_ylabel("b1 (10^9 s/m^2)")
    axes[0].set_title("Finite-pulse DEXSY signal")
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

    display = sim.recovered if sim.nonnegative else np.clip(sim.recovered, 0.0, None)
    display = display / max(float(np.max(display)), np.finfo(float).eps)
    mesh = axes[1].pcolormesh(
        sim.diffusion_axis * 1e9,
        sim.diffusion_axis * 1e9,
        display,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_xlabel("D2 (10^-9 m^2/s)")
    axes[1].set_ylabel("D1 (10^-9 m^2/s)")
    solver = "NNLS" if sim.nonnegative else "LS preview"
    axes[1].set_title(f"Recovered D-D map ({solver})")
    fig.colorbar(mesh, ax=axes[1], fraction=0.046, pad=0.04)

    trans = axes[2].imshow(
        sim.transition,
        origin="upper",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
    )
    axes[2].set_xticks([0, 1], labels=["narrow", "wide"])
    axes[2].set_yticks([0, 1], labels=["narrow", "wide"])
    axes[2].set_xlabel("compartment before encoding 2")
    axes[2].set_ylabel("compartment before encoding 1")
    axes[2].set_title("Trajectory exchange probabilities")
    for row in range(2):
        for col in range(2):
            axes[2].text(
                col,
                row,
                f"{sim.transition[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(trans, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].text(
        0.5,
        -0.22,
        f"relative ILT residual: {sim.residual_fraction:.3f}",
        ha="center",
        va="top",
        transform=axes[2].transAxes,
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

    print(f"b range: {sim.b_values[0]:.3e} to {sim.b_values[-1]:.3e} s/m^2")
    print(f"microscopic D: {DIFFUSION:.2e} m^2/s")
    print(
        "exchange probabilities during mixing: "
        f"narrow->wide {sim.transition[0, 1]:.2f}, "
        f"wide->narrow {sim.transition[1, 0]:.2f}"
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
