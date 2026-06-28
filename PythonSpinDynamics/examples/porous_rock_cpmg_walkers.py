"""3D porous-rock CPMG with explicit restricted-diffusion walkers.

This example models the workload that fixed isochromat grids cannot: millions
of water walkers diffuse through a voxelized cylindrical rock core, reject steps
into solid voxels, sample susceptibility-driven B0 offsets, and relax with a
pore-size-dependent surface-relaxation T2. The default parameters are a large
benchmark-style challenge intended for the JAX backend.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np

from _source_path import add_src_to_path, load_matplotlib

add_src_to_path()

from spin_dynamics.fields.domain import SpatialDomain  # noqa: E402
from spin_dynamics.motion import (  # noqa: E402
    ParticleEnsemble,
    apply_boundary,
    make_motion_field_maps,
)
from spin_dynamics.sequences.motion import (  # noqa: E402
    MotionSequenceStep,
    make_motion_cpmg_sequence,
    run_motion_sequence,
)

try:  # noqa: E402
    from spin_dynamics import _numba_motion as numba_motion
except Exception:  # pragma: no cover - optional dependency import guard
    numba_motion = None
try:  # noqa: E402
    from spin_dynamics import _jax_motion as jax_motion
except Exception:  # pragma: no cover - optional dependency import guard
    jax_motion = None


GAMMA_PROTON = 2.675222005e8  # rad / (s T)
D_WATER = 2.3e-9  # m^2 / s


@dataclass(frozen=True)
class PorousCore:
    domain: SpatialDomain
    pore_mask: np.ndarray
    pore_radius: np.ndarray
    b0_offset: np.ndarray
    t1_map: np.ndarray
    t2_map: np.ndarray
    diffusion_map: np.ndarray
    porosity: float
    pore_radii: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=int, default=96, help="x/y cells across the core."
    )
    parser.add_argument(
        "--z-cells", type=int, default=160, help="z cells along the core."
    )
    parser.add_argument(
        "--pores", type=int, default=900, help="Synthetic spherical pores."
    )
    parser.add_argument("--walkers-per-voxel", type=int, default=10)
    parser.add_argument("--num-echoes", type=int, default=480)
    parser.add_argument("--echo-spacing-ms", type=float, default=2.0)
    parser.add_argument("--substeps", type=int, default=6)
    parser.add_argument("--core-radius-mm", type=float, default=1.5)
    parser.add_argument("--core-length-mm", type=float, default=6.0)
    parser.add_argument("--pore-radius-min-um", type=float, default=45.0)
    parser.add_argument("--pore-radius-max-um", type=float, default=420.0)
    parser.add_argument("--micro-tortuosity", type=float, default=2.4)
    parser.add_argument("--surface-relaxivity-um-s", type=float, default=240.0)
    parser.add_argument("--bulk-t2-s", type=float, default=0.18)
    parser.add_argument("--bulk-t1-s", type=float, default=2.2)
    parser.add_argument(
        "--throats",
        type=int,
        default=-1,
        help="Connecting throat count. Negative uses about 1.5 per pore body.",
    )
    parser.add_argument("--b0-mt", type=float, default=50.0)
    parser.add_argument("--susceptibility-ppm", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print feasibility estimates without running the sequence.",
    )
    parser.add_argument(
        "--backend",
        choices=["numpy", "numba", "jax", "auto"],
        default="auto",
        help="Walker compute backend. 'auto' prefers JAX, then Numba.",
    )
    parser.add_argument(
        "--benchmark-backends",
        action="store_true",
        help="Run NumPy and Numba backends and print timings.",
    )
    parser.add_argument(
        "--jax-rng",
        choices=["on-the-fly", "precomputed"],
        default="on-the-fly",
        help=(
            "JAX Brownian increment source. Use on-the-fly for large runs; "
            "precomputed is useful for NumPy/JAX trajectory parity checks."
        ),
    )
    parser.add_argument(
        "--structure-points",
        type=int,
        default=15000,
        help="Maximum pore voxels to scatter in the structure plot.",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional NPZ output."
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional PNG path for the particle D-T2 distribution.",
    )
    return parser.parse_args()


def _cell_edges(axis: np.ndarray) -> np.ndarray:
    if axis.size == 1:
        return np.array([axis[0] - 0.5, axis[0] + 0.5], dtype=np.float64)
    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return edges


def _positions_to_indices(
    domain: SpatialDomain,
    positions: np.ndarray,
) -> tuple[np.ndarray, ...]:
    indices = []
    for axis, coord in zip(domain.axes, positions.T):
        edges = _cell_edges(axis)
        idx = np.searchsorted(edges, coord, side="right") - 1
        indices.append(np.clip(idx, 0, axis.size - 1).astype(np.int64))
    return tuple(indices)


def _sample_map(
    domain: SpatialDomain,
    values: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    return values[_positions_to_indices(domain, positions)]


def _reflect_box(
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
) -> np.ndarray:
    return apply_boundary(positions, bounds, "reflect")


def make_voxel_pore_boundary(domain: SpatialDomain, pore_mask: np.ndarray):
    """Return a 3D no-flux boundary for a voxel pore mask.

    Proposed moves that land in solid are rejected to the previous position.
    This is a finite-step approximation to reflecting walls; use small Brownian
    hops compared with the voxel width and pore throat size.
    """

    mask = np.asarray(pore_mask, dtype=bool)
    bounds = domain.bounds

    def boundary(
        positions: np.ndarray,
        *,
        previous_positions: np.ndarray | None = None,
        **_: object,
    ) -> np.ndarray:
        pos = _reflect_box(np.asarray(positions, dtype=np.float64), bounds)
        idx = _positions_to_indices(domain, pos)
        in_pore = mask[idx]
        if np.all(in_pore):
            return pos
        if previous_positions is None:
            pos[~in_pore] = 0.0
            return pos
        prev = _reflect_box(np.asarray(previous_positions, dtype=np.float64), bounds)
        pos[~in_pore] = prev[~in_pore]
        return pos

    return boundary


def _sample_pore_radii(
    rng: np.random.Generator,
    count: int,
    r_min: float,
    r_max: float,
) -> np.ndarray:
    """Sample a truncated multimodal pore-size distribution."""

    weights = np.array([0.50, 0.35, 0.15], dtype=np.float64)
    medians = np.array([65.0e-6, 145.0e-6, 330.0e-6], dtype=np.float64)
    sigmas = np.array([0.30, 0.34, 0.24], dtype=np.float64)
    medians = np.clip(medians, 1.05 * r_min, 0.95 * r_max)
    counts = rng.multinomial(int(count), weights / np.sum(weights))
    samples = []
    for mode_count, median, sigma in zip(counts, medians, sigmas):
        mode_values = np.empty(0, dtype=np.float64)
        while mode_values.size < mode_count:
            draw = rng.lognormal(np.log(median), sigma, max(mode_count, 32))
            draw = draw[(draw >= r_min) & (draw <= r_max)]
            mode_values = np.concatenate((mode_values, draw))
        samples.append(mode_values[:mode_count])
    radii = np.concatenate(samples)
    rng.shuffle(radii)
    return radii


def _nearest_axis_index(axis: np.ndarray, value: float) -> int:
    return int(np.clip(np.searchsorted(axis, value), 0, axis.size - 1))


def _axis_window(axis: np.ndarray, lower: float, upper: float) -> slice:
    start = int(np.searchsorted(axis, lower, side="left"))
    stop = int(np.searchsorted(axis, upper, side="right"))
    return slice(max(0, start), min(axis.size, stop))


def _paint_sphere(
    domain: SpatialDomain,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    cylinder: np.ndarray,
    pore_mask: np.ndarray,
    owner_radius: np.ndarray,
    owner_factor: np.ndarray,
    constriction: np.ndarray,
    wall_margin: np.ndarray,
    center: np.ndarray,
    radius: float,
    factor: float,
) -> None:
    xs = _axis_window(domain.axes[0], center[0] - radius, center[0] + radius)
    ys = _axis_window(domain.axes[1], center[1] - radius, center[1] + radius)
    zs = _axis_window(domain.axes[2], center[2] - radius, center[2] + radius)
    xx, yy, zz = (grid[xs, ys, zs] for grid in grids)
    dist = np.sqrt(
        (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    )
    inside = cylinder[xs, ys, zs] & (dist <= radius)
    margin = radius - dist
    update = inside & (margin > wall_margin[xs, ys, zs])
    pore_mask[xs, ys, zs] |= inside
    owner_radius[xs, ys, zs][update] = radius
    owner_factor[xs, ys, zs][update] = factor
    constriction[xs, ys, zs][update] = 1.0
    wall_margin[xs, ys, zs][update] = margin[update]

    if not np.any(inside):
        ix = _nearest_axis_index(domain.axes[0], center[0])
        iy = _nearest_axis_index(domain.axes[1], center[1])
        iz = _nearest_axis_index(domain.axes[2], center[2])
        if cylinder[ix, iy, iz]:
            pore_mask[ix, iy, iz] = True
            owner_radius[ix, iy, iz] = radius
            owner_factor[ix, iy, iz] = factor
            constriction[ix, iy, iz] = 1.0
            wall_margin[ix, iy, iz] = radius


def _paint_throat(
    domain: SpatialDomain,
    grids: tuple[np.ndarray, np.ndarray, np.ndarray],
    cylinder: np.ndarray,
    pore_mask: np.ndarray,
    owner_radius: np.ndarray,
    owner_factor: np.ndarray,
    constriction: np.ndarray,
    wall_margin: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    body_radius: float,
    factor: float,
) -> None:
    lower = np.minimum(start, end) - radius
    upper = np.maximum(start, end) + radius
    xs = _axis_window(domain.axes[0], lower[0], upper[0])
    ys = _axis_window(domain.axes[1], lower[1], upper[1])
    zs = _axis_window(domain.axes[2], lower[2], upper[2])
    if xs.start == xs.stop or ys.start == ys.stop or zs.start == zs.stop:
        return
    xx, yy, zz = (grid[xs, ys, zs] for grid in grids)
    segment = end - start
    segment_norm = float(np.dot(segment, segment))
    if segment_norm <= 0.0:
        return
    rel_x = xx - start[0]
    rel_y = yy - start[1]
    rel_z = zz - start[2]
    t = (rel_x * segment[0] + rel_y * segment[1] + rel_z * segment[2])
    t = np.clip(t / segment_norm, 0.0, 1.0)
    near_x = start[0] + t * segment[0]
    near_y = start[1] + t * segment[1]
    near_z = start[2] + t * segment[2]
    dist = np.sqrt((xx - near_x) ** 2 + (yy - near_y) ** 2 + (zz - near_z) ** 2)
    inside = cylinder[xs, ys, zs] & (dist <= radius)
    margin = radius - dist
    update = inside & (margin > wall_margin[xs, ys, zs])
    throat_constriction = np.clip(radius / max(body_radius, radius), 0.18, 0.95)
    pore_mask[xs, ys, zs] |= inside
    owner_radius[xs, ys, zs][update] = radius
    owner_factor[xs, ys, zs][update] = factor
    constriction[xs, ys, zs][update] = throat_constriction
    wall_margin[xs, ys, zs][update] = margin[update]


def build_porous_core(args: argparse.Namespace) -> PorousCore:
    rng = np.random.default_rng(args.seed)
    radius = args.core_radius_mm * 1.0e-3
    length = args.core_length_mm * 1.0e-3
    x = np.linspace(-radius, radius, int(args.grid))
    y = np.linspace(-radius, radius, int(args.grid))
    z = np.linspace(-0.5 * length, 0.5 * length, int(args.z_cells))
    domain = SpatialDomain((x, y, z))
    xx, yy, zz = domain.meshgrid()
    cylinder = xx**2 + yy**2 <= radius**2

    r_min = args.pore_radius_min_um * 1.0e-6
    r_max = args.pore_radius_max_um * 1.0e-6
    pore_radii = _sample_pore_radii(rng, int(args.pores), r_min, r_max)
    angles = rng.uniform(0.0, 2.0 * np.pi, int(args.pores))
    center_r = radius * np.sqrt(rng.uniform(0.0, 0.92, int(args.pores)))
    centers = np.column_stack(
        (
            center_r * np.cos(angles),
            center_r * np.sin(angles),
            rng.uniform(-0.46 * length, 0.46 * length, int(args.pores)),
        )
    )
    pore_factor = rng.normal(0.0, 1.0, int(args.pores))

    pore_mask = np.zeros(domain.shape, dtype=bool)
    owner_radius = np.zeros(domain.shape, dtype=np.float64)
    owner_factor = np.zeros(domain.shape, dtype=np.float64)
    constriction = np.zeros(domain.shape, dtype=np.float64)
    wall_margin = np.zeros(domain.shape, dtype=np.float64)
    for center, pore_radius, factor in zip(centers, pore_radii, pore_factor):
        _paint_sphere(
            domain,
            (xx, yy, zz),
            cylinder,
            pore_mask,
            owner_radius,
            owner_factor,
            constriction,
            wall_margin,
            center,
            pore_radius,
            factor,
        )

    throat_count = int(args.throats)
    if throat_count < 0:
        throat_count = int(round(1.5 * int(args.pores)))
    for _ in range(throat_count):
        i = int(rng.integers(0, centers.shape[0]))
        candidates = rng.choice(centers.shape[0], size=min(32, centers.shape[0]), replace=False)
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            continue
        distances = np.linalg.norm(centers[candidates] - centers[i], axis=1)
        j = int(candidates[np.argmin(distances)])
        body_radius = min(pore_radii[i], pore_radii[j])
        throat_radius = body_radius * rng.uniform(0.22, 0.55)
        throat_radius = max(throat_radius, 0.55 * min(np.mean(np.diff(x)), np.mean(np.diff(z))))
        _paint_throat(
            domain,
            (xx, yy, zz),
            cylinder,
            pore_mask,
            owner_radius,
            owner_factor,
            constriction,
            wall_margin,
            centers[i],
            centers[j],
            throat_radius,
            body_radius,
            0.5 * (pore_factor[i] + pore_factor[j]),
        )

    if not np.any(pore_mask):
        raise SystemExit("generated no pore voxels; increase --pores or pore radius")

    porosity = float(np.count_nonzero(pore_mask) / np.count_nonzero(cylinder))
    pore_radius_map = np.where(pore_mask, np.maximum(owner_radius, r_min), np.inf)
    constriction_map = np.where(pore_mask, np.maximum(constriction, 0.18), 1.0)
    near_wall = np.exp(
        -wall_margin / np.maximum(0.18 * pore_radius_map, np.finfo(float).eps)
    )
    near_wall = np.where(pore_mask, near_wall, 0.0)
    relaxation_radius = pore_radius_map * (0.35 + 0.65 * constriction_map)
    relaxation_radius = relaxation_radius / (1.0 + 2.0 * near_wall)
    relaxation_radius = np.where(
        pore_mask,
        np.maximum(relaxation_radius, 0.30 * r_min),
        np.inf,
    )
    rho2 = args.surface_relaxivity_um_s * 1.0e-6
    inv_t2 = (1.0 / args.bulk_t2_s) + 3.0 * rho2 / relaxation_radius
    t2_map = np.where(pore_mask, 1.0 / inv_t2, np.inf)
    t1_map = np.where(pore_mask, float(args.bulk_t1_s), np.inf)

    r_ref = np.median(pore_radii)
    tortuosity_map = (
        float(args.micro_tortuosity)
        * (r_ref / pore_radius_map) ** 0.25
        * np.exp(0.22 * owner_factor)
        / constriction_map**1.35
    )
    tortuosity_map = np.where(pore_mask, np.clip(tortuosity_map, 1.25, 9.0), np.inf)
    diffusion_map = np.where(pore_mask, D_WATER / tortuosity_map, 0.0)

    b0_t = args.b0_mt * 1.0e-3
    chi = args.susceptibility_ppm * 1.0e-6
    normalized_z = zz / max(length, np.finfo(float).eps)
    wall_scale = np.exp(
        -wall_margin / np.maximum(0.25 * pore_radius_map, np.finfo(float).eps)
    )
    wall_scale = np.where(pore_mask, wall_scale, 0.0)
    geometric_term = (
        0.35 * (xx / radius) ** 2
        - 0.20 * (yy / radius) ** 2
        + 0.10 * normalized_z
    )
    b0_offset = GAMMA_PROTON * b0_t * chi * (
        geometric_term + 0.65 * wall_scale * owner_factor
    )
    b0_offset = np.where(pore_mask, b0_offset, 0.0)

    return PorousCore(
        domain=domain,
        pore_mask=pore_mask,
        pore_radius=pore_radius_map,
        b0_offset=b0_offset,
        t1_map=t1_map,
        t2_map=t2_map,
        diffusion_map=diffusion_map,
        porosity=porosity,
        pore_radii=pore_radii,
    )


def seed_walkers(
    core: PorousCore,
    walkers_per_voxel: int,
    seed: int,
) -> tuple[ParticleEnsemble, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grids = core.domain.meshgrid()
    pore_indices = np.flatnonzero(core.pore_mask.ravel())
    unraveled = np.unravel_index(pore_indices, core.domain.shape)
    centers = np.column_stack([grid.ravel()[pore_indices] for grid in grids])
    positions = np.repeat(centers, int(walkers_per_voxel), axis=0)
    widths = np.column_stack(
        [
            (_cell_edges(axis)[1:] - _cell_edges(axis)[:-1])[unraveled[axis_index]]
            for axis_index, axis in enumerate(core.domain.axes)
        ]
    )
    widths = np.repeat(widths, int(walkers_per_voxel), axis=0)
    positions = positions + rng.uniform(-0.45, 0.45, size=positions.shape) * widths
    positions = make_voxel_pore_boundary(core.domain, core.pore_mask)(
        positions,
        previous_positions=np.repeat(centers, int(walkers_per_voxel), axis=0),
    )

    n = positions.shape[0]
    magnetization = np.zeros((3, n), dtype=np.complex128)
    magnetization[0, :] = 1.0
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    diffusion = _sample_map(core.domain, core.diffusion_map, positions)
    ensemble = ParticleEnsemble(
        positions=positions,
        magnetization=magnetization,
        weights=weights,
        diffusion_coefficient=diffusion,
    )
    t1 = _sample_map(core.domain, core.t1_map, positions)
    t2 = _sample_map(core.domain, core.t2_map, positions)
    return ensemble, t1, t2


def _workflow_size(args: argparse.Namespace, walkers: int) -> tuple[int, float]:
    steps = make_motion_cpmg_sequence(
        int(args.num_echoes),
        args.echo_spacing_ms * 1.0e-3,
        excitation_duration=100.0e-6,
        refocusing_duration=200.0e-6,
        gradient=(0.0, 0.0, 0.0),
        substeps_per_interval=int(args.substeps),
    )
    updates = int(sum(step.substeps or args.substeps for step in steps) * walkers)
    # Conservative NumPy walker rate for 3D field sampling, RNG, and boundary checks.
    estimated_seconds = updates / 1.8e6
    return updates, estimated_seconds


def _estimate_backend_seconds(
    updates: int,
    *,
    jax_devices: list[str],
) -> dict[str, float]:
    """Return rough host-specific estimates from the benchmarked walker rates."""

    has_cuda = any("cuda" in device.lower() or "gpu" in device.lower() for device in jax_devices)
    rates = {
        "numpy": 1.8e6,
        "numba": 12.0e6,
        "jax": 130.0e6 if has_cuda else 10.0e6,
    }
    return {name: updates / rate for name, rate in rates.items()}


def _percentile_summary(values: np.ndarray, scale: float = 1.0) -> str:
    percentiles = np.percentile(
        np.asarray(values, dtype=np.float64) * scale,
        [1, 5, 25, 50, 75, 95, 99],
    )
    labels = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
    return ", ".join(
        f"{label}={value:.3g}" for label, value in zip(labels, percentiles)
    )


def _distribution_estimates(core: PorousCore) -> dict[str, np.ndarray]:
    pore = core.pore_mask
    radius = core.pore_radius[pore]
    t2 = core.t2_map[pore]
    diffusion = core.diffusion_map[pore]
    tortuosity = D_WATER / np.maximum(diffusion, np.finfo(float).tiny)
    corr = np.corrcoef(np.log(diffusion), np.log(t2))[0, 1]
    return {
        "body_radius_um": core.pore_radii * 1.0e6,
        "voxel_radius_um": radius * 1.0e6,
        "t2_ms": t2 * 1.0e3,
        "diffusion_um2_ms": diffusion * 1.0e9,
        "tortuosity": tortuosity,
        "log_d_log_t2_corr": np.asarray(corr),
    }


def _print_distribution_estimates(core: PorousCore) -> None:
    estimates = _distribution_estimates(core)
    print("analytical/geometry-only distribution estimates")
    print(
        "  input pore-body radius (um): "
        f"{_percentile_summary(estimates['body_radius_um'])}"
    )
    print(
        "  voxel local pore radius (um): "
        f"{_percentile_summary(estimates['voxel_radius_um'])}"
    )
    print(
        "  expected T2 from surface relaxation (ms): "
        f"{_percentile_summary(estimates['t2_ms'])}"
    )
    print(
        "  expected D from tortuosity (um^2/ms): "
        f"{_percentile_summary(estimates['diffusion_um2_ms'])}"
    )
    print(
        "  effective tortuosity: "
        f"{_percentile_summary(estimates['tortuosity'])}"
    )
    print(f"  corr(log D, log T2): {estimates['log_d_log_t2_corr']:.3f}")


def _flatten_steps(
    steps: tuple[MotionSequenceStep, ...],
    default_substeps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_values = []
    rf_amp = []
    rf_phase = []
    acquire = []
    for step in steps:
        substeps = int(default_substeps if step.substeps is None else step.substeps)
        for index in range(substeps):
            dt_values.append(float(step.duration) / substeps)
            rf_amp.append(float(step.rf_amplitude))
            rf_phase.append(float(step.rf_phase))
            acquire.append(1 if step.acquire and index == substeps - 1 else 0)
    return (
        np.asarray(dt_values, dtype=np.float64),
        np.asarray(rf_amp, dtype=np.float64),
        np.asarray(rf_phase, dtype=np.float64),
        np.asarray(acquire, dtype=np.uint8),
    )


def _run_numpy_backend(
    ensemble: ParticleEnsemble,
    core: PorousCore,
    steps: tuple[MotionSequenceStep, ...],
    t1: np.ndarray,
    t2: np.ndarray,
    args: argparse.Namespace,
):
    fields = make_motion_field_maps(core.domain, b0_map=core.b0_offset)
    boundary = make_voxel_pore_boundary(core.domain, core.pore_mask)
    return run_motion_sequence(
        ensemble,
        fields,
        steps,
        rng=np.random.default_rng(args.seed),
        t1=t1,
        t2=t2,
        boundary=boundary,
        default_substeps=int(args.substeps),
    ).signal


def _run_numba_backend(
    ensemble: ParticleEnsemble,
    core: PorousCore,
    steps: tuple[MotionSequenceStep, ...],
    t1: np.ndarray,
    t2: np.ndarray,
    args: argparse.Namespace,
):
    if numba_motion is None or not numba_motion.NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requires the optional 'numba' extra")
    dt_values, rf_amp, rf_phase, acquire = _flatten_steps(steps, int(args.substeps))
    rng = np.random.default_rng(args.seed)
    normals = rng.normal(
        size=(dt_values.size, ensemble.num_particles, 3)
    ).astype(np.float64)
    signal, _positions = numba_motion.cpmg_voxel_walkers_core(
        np.asarray(ensemble.positions, dtype=np.float64),
        np.asarray(ensemble.weights, dtype=np.float64),
        np.asarray(ensemble.diffusion_coefficient, dtype=np.float64),
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
        np.asarray(core.b0_offset, dtype=np.float64),
        np.asarray(core.pore_mask, dtype=np.bool_),
        np.asarray(core.domain.axes[0], dtype=np.float64),
        np.asarray(core.domain.axes[1], dtype=np.float64),
        np.asarray(core.domain.axes[2], dtype=np.float64),
        dt_values,
        rf_amp,
        rf_phase,
        acquire,
        normals,
    )
    return signal


def _run_jax_backend(
    ensemble: ParticleEnsemble,
    core: PorousCore,
    steps: tuple[MotionSequenceStep, ...],
    t1: np.ndarray,
    t2: np.ndarray,
    args: argparse.Namespace,
):
    if jax_motion is None or not jax_motion.JAX_AVAILABLE:
        raise RuntimeError("JAX backend requires the optional 'jax' extra")
    dt_values, rf_amp, rf_phase, acquire = _flatten_steps(steps, int(args.substeps))
    if args.jax_rng == "on-the-fly":
        signal, _positions = jax_motion.cpmg_voxel_walkers_core_prng(
            np.asarray(ensemble.positions, dtype=np.float64),
            np.asarray(ensemble.weights, dtype=np.float64),
            np.asarray(ensemble.diffusion_coefficient, dtype=np.float64),
            np.asarray(t1, dtype=np.float64),
            np.asarray(t2, dtype=np.float64),
            np.asarray(core.b0_offset, dtype=np.float64),
            np.asarray(core.pore_mask, dtype=np.bool_),
            np.asarray(core.domain.axes[0], dtype=np.float64),
            np.asarray(core.domain.axes[1], dtype=np.float64),
            np.asarray(core.domain.axes[2], dtype=np.float64),
            dt_values,
            rf_amp,
            rf_phase,
            acquire,
            int(args.seed),
        )
        return signal
    rng = np.random.default_rng(args.seed)
    normals = rng.normal(
        size=(dt_values.size, ensemble.num_particles, 3)
    ).astype(np.float64)
    signal, _positions = jax_motion.cpmg_voxel_walkers_core(
        np.asarray(ensemble.positions, dtype=np.float64),
        np.asarray(ensemble.weights, dtype=np.float64),
        np.asarray(ensemble.diffusion_coefficient, dtype=np.float64),
        np.asarray(t1, dtype=np.float64),
        np.asarray(t2, dtype=np.float64),
        np.asarray(core.b0_offset, dtype=np.float64),
        np.asarray(core.pore_mask, dtype=np.bool_),
        np.asarray(core.domain.axes[0], dtype=np.float64),
        np.asarray(core.domain.axes[1], dtype=np.float64),
        np.asarray(core.domain.axes[2], dtype=np.float64),
        dt_values,
        rf_amp,
        rf_phase,
        acquire,
        normals,
    )
    return signal


def _time_backend(name: str, run) -> tuple[np.ndarray, float]:
    if name in {"numba", "jax"}:
        run()  # compile/warm up outside the timed interval
    start = time.perf_counter()
    signal = run()
    return signal, time.perf_counter() - start


def _plot_summary(
    output: Path,
    core: PorousCore,
    diffusion: np.ndarray,
    t2: np.ndarray,
    echo_times: np.ndarray,
    normalized_echo: np.ndarray,
    structure_points: int,
) -> None:
    plt = load_matplotlib(headless=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.8, 8.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax_structure = fig.add_subplot(gs[0, 0], projection="3d")
    ax_dt2 = fig.add_subplot(gs[0, 1])
    ax_decay = fig.add_subplot(gs[1, 0])
    ax_pores = fig.add_subplot(gs[1, 1])

    pore_indices = np.column_stack(np.nonzero(core.pore_mask))
    if pore_indices.shape[0] > structure_points > 0:
        rng = np.random.default_rng(12345)
        chosen = rng.choice(pore_indices.shape[0], size=structure_points, replace=False)
        pore_indices = pore_indices[chosen]
    x_axis, y_axis, z_axis = core.domain.axes
    px = x_axis[pore_indices[:, 0]] * 1.0e3
    py = y_axis[pore_indices[:, 1]] * 1.0e3
    pz = z_axis[pore_indices[:, 2]] * 1.0e3
    color = core.pore_radius[tuple(pore_indices.T)] * 1.0e6
    scatter = ax_structure.scatter(
        px,
        py,
        pz,
        c=color,
        s=1.2,
        alpha=0.45,
        cmap="viridis",
        linewidths=0,
    )
    fig.colorbar(scatter, ax=ax_structure, shrink=0.72, label="local pore radius (um)")
    ax_structure.set_xlabel("x (mm)")
    ax_structure.set_ylabel("y (mm)")
    ax_structure.set_zlabel("z (mm)")
    ax_structure.set_title("Voxelized pore structure")
    ax_structure.view_init(elev=22, azim=-55)

    d_axis = diffusion * 1.0e9  # m^2/s -> um^2/ms
    t2_axis = t2 * 1.0e3
    hist = ax_dt2.hist2d(
        t2_axis,
        d_axis,
        bins=42,
        cmap="magma",
    )
    fig.colorbar(hist[3], ax=ax_dt2, label="walkers")
    ax_dt2.set_xlabel("T2 (ms)")
    ax_dt2.set_ylabel("D (um^2/ms)")
    ax_dt2.set_title("Pore-weighted D-T2 map")
    ax_decay.plot(echo_times * 1.0e3, normalized_echo, marker="o", ms=2.4)
    ax_decay.set_xlabel("echo time (ms)")
    ax_decay.set_ylabel("normalized echo")
    ax_decay.set_ylim(0.0, 1.05)
    ax_decay.set_title("Walker CPMG decay")
    ax_pores.hist(core.pore_radii * 1.0e6, bins=36, color="#4c78a8")
    ax_pores.set_xlabel("synthetic pore radius (um)")
    ax_pores.set_ylabel("count")
    ax_pores.set_title("Input pore-size distribution")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = _parse_args()
    if args.grid < 4 or args.z_cells < 4:
        raise SystemExit("--grid and --z-cells must be at least 4")
    if args.walkers_per_voxel <= 0 or args.num_echoes <= 0 or args.substeps <= 0:
        raise SystemExit("walkers, echoes, and substeps must be positive")
    if args.micro_tortuosity <= 0.0:
        raise SystemExit("--micro-tortuosity must be positive")

    core = build_porous_core(args)
    ensemble, t1, t2 = seed_walkers(core, args.walkers_per_voxel, args.seed)
    updates, estimated_seconds = _workflow_size(args, ensemble.num_particles)
    min_spacing = min(float(np.mean(np.diff(axis))) for axis in core.domain.axes)
    dt = 0.5 * args.echo_spacing_ms * 1.0e-3 / max(1, int(args.substeps))
    rms_hop = float(np.sqrt(2.0 * np.mean(ensemble.diffusion_coefficient) * dt))

    print("3D porous-rock CPMG walker feasibility")
    print(f"domain: {core.domain.shape} voxels, porosity {core.porosity:.3f}")
    print(
        "pore radii: "
        f"{np.min(core.pore_radii)*1e6:.0f}-{np.max(core.pore_radii)*1e6:.0f} um"
    )
    print(
        f"walkers: {ensemble.num_particles} "
        f"({args.walkers_per_voxel} per pore voxel)"
    )
    print(
        f"walker-updates: {updates:,}   "
        f"estimated runtime: {estimated_seconds:.1f} s"
    )
    print(
        f"rms Brownian hop/substep: {rms_hop*1e6:.1f} um "
        f"({rms_hop/min_spacing:.2f} voxels)"
    )
    _print_distribution_estimates(core)
    numba_available = bool(numba_motion is not None and numba_motion.NUMBA_AVAILABLE)
    jax_available = bool(jax_motion is not None and jax_motion.JAX_AVAILABLE)
    print(f"numba walker backend available: {numba_available}")
    jax_devices = [] if not jax_available else jax_motion.devices()
    print(f"jax walker backend available: {jax_available} devices={jax_devices}")
    estimates = _estimate_backend_seconds(updates, jax_devices=jax_devices)
    print(
        "rough backend estimates: "
        + ", ".join(f"{name} {seconds/60:.1f} min" for name, seconds in estimates.items())
    )
    if args.estimate_only:
        return

    steps = make_motion_cpmg_sequence(
        int(args.num_echoes),
        args.echo_spacing_ms * 1.0e-3,
        excitation_duration=100.0e-6,
        refocusing_duration=200.0e-6,
        gradient=(0.0, 0.0, 0.0),
        substeps_per_interval=int(args.substeps),
    )

    runners = {
        "numpy": lambda: _run_numpy_backend(ensemble, core, steps, t1, t2, args),
    }
    if numba_available:
        runners["numba"] = lambda: _run_numba_backend(
            ensemble, core, steps, t1, t2, args
        )
    if jax_available:
        runners["jax"] = lambda: _run_jax_backend(
            ensemble, core, steps, t1, t2, args
        )
    requested_backend = args.backend
    if requested_backend == "auto":
        if jax_available:
            requested_backend = "jax"
        elif numba_available:
            requested_backend = "numba"
        else:
            requested_backend = "numpy"
    if requested_backend not in runners:
        raise SystemExit(f"backend '{requested_backend}' is not available")

    timings: dict[str, float] = {}
    signals: dict[str, np.ndarray] = {}
    if args.benchmark_backends:
        for name, runner in runners.items():
            signals[name], timings[name] = _time_backend(name, runner)
            print(f"{name} runtime: {timings[name]:.3f} s")
        echo = signals[requested_backend]
        elapsed = timings[requested_backend]
        if "numpy" in signals and "numba" in signals:
            n = min(signals["numpy"].size, signals["numba"].size)
            rel = np.linalg.norm(signals["numba"][:n] - signals["numpy"][:n])
            rel /= max(np.linalg.norm(signals["numpy"][:n]), np.finfo(float).eps)
            speedup = timings["numpy"] / timings["numba"]
            print(f"numba speedup vs numpy: {speedup:.2f}x")
            print(f"trajectory-to-trajectory relative signal delta: {rel:.3e}")
        if "numpy" in signals and "jax" in signals:
            speedup = timings["numpy"] / timings["jax"]
            print(f"jax speedup vs numpy: {speedup:.2f}x")
            if args.jax_rng == "precomputed":
                n = min(signals["numpy"].size, signals["jax"].size)
                rel = np.linalg.norm(signals["jax"][:n] - signals["numpy"][:n])
                rel /= max(np.linalg.norm(signals["numpy"][:n]), np.finfo(float).eps)
                print(f"trajectory-to-trajectory relative signal delta: {rel:.3e}")
            else:
                print("trajectory parity skipped: JAX used on-the-fly PRNG")
    else:
        echo, elapsed = _time_backend(requested_backend, runners[requested_backend])

    magnitude = np.abs(echo)
    normalized = magnitude / max(float(magnitude[0]), np.finfo(float).eps)
    echo_times = args.echo_spacing_ms * 1.0e-3 * (
        np.arange(normalized.size, dtype=np.float64) + 1.0
    )
    print(f"actual runtime: {elapsed:.2f} s")
    print(f"backend: {requested_backend}")
    print(
        f"first/last normalized echo: {normalized[0]:.4g} / {normalized[-1]:.4g}"
    )
    print(f"median T2 from seeded pores: {np.median(t2)*1e3:.1f} ms")
    print(
        "B0 offset rms in pore space: "
        f"{np.std(core.b0_offset[core.pore_mask])/(2*np.pi):.1f} Hz"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            echo=echo,
            echo_times=echo_times,
            normalized_echo=normalized,
            porosity=np.asarray(core.porosity),
            pore_radii=core.pore_radii,
            t2_particles=t2,
            diffusion_particles=ensemble.diffusion_coefficient,
            backend=np.asarray(requested_backend),
            elapsed_seconds=np.asarray(elapsed),
            walker_updates=np.asarray(updates),
        )
        print(f"saved: {args.output}")
    if args.plot_output is not None:
        _plot_summary(
            args.plot_output,
            core,
            ensemble.diffusion_coefficient,
            t2,
            echo_times,
            normalized,
            int(args.structure_points),
        )
        print(f"plot saved: {args.plot_output}")
    else:
        print("plot not written; pass --plot-output PATH to save the structure/D-T2 figure")


if __name__ == "__main__":
    main()
