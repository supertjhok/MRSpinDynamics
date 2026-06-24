"""Lagrangian isochromat motion helpers for advection and diffusion physics.

The existing arbitrary-pulse kernels treat each isochromat as fixed in space.
This module provides opt-in building blocks for simulations where spins move
through spatial B0/B1 maps between sequence updates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from spin_dynamics.core.rotations import MatrixElements, rf_matrix_elements


BoundaryMode = Literal["reflect", "periodic", "clip"]
BoundaryFn = Callable[[np.ndarray], np.ndarray]
# A boundary is either one of the rectangular-box modes or a callable that maps
# ``(num_particles, 2)`` positions to confined positions (e.g. a curved pore).
Boundary = BoundaryMode | BoundaryFn
Velocity = np.ndarray | Callable[[np.ndarray, float], np.ndarray] | None


@dataclass(frozen=True)
class MotionFieldMaps2D:
    """Two-dimensional field maps used by moving isochromats.

    Axes and maps use the same orientation as the imaging field maps:
    map rows follow `x_axis` and columns follow `z_axis`. `b0_map` is a
    normalized angular off-resonance map, while `b1_tx_map` and `b1_rx_map`
    are relative transmit and receive sensitivities.
    """

    x_axis: np.ndarray
    z_axis: np.ndarray
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray

    @property
    def bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            (float(self.x_axis[0]), float(self.x_axis[-1])),
            (float(self.z_axis[0]), float(self.z_axis[-1])),
        )

    def sample(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        """Bilinearly sample field maps at `(x, z)` positions."""

        pos = _positions2d(positions)
        return {
            "b0": _bilinear_sample(self.b0_map, self.x_axis, self.z_axis, pos),
            "b1_tx": _bilinear_sample(self.b1_tx_map, self.x_axis, self.z_axis, pos),
            "b1_rx": _bilinear_sample(self.b1_rx_map, self.x_axis, self.z_axis, pos),
        }


@dataclass(frozen=True)
class ParticleEnsemble:
    """Moving isochromat ensemble.

    `positions` has shape `(num_particles, 2)` for `(x, z)`. `magnetization`
    has shape `(3, num_particles)` and follows the existing kernel convention:
    row 0 is longitudinal magnetization, row 1 is `M-`, and row 2 is `M+`.
    """

    positions: np.ndarray
    magnetization: np.ndarray
    weights: np.ndarray
    diffusion_coefficient: np.ndarray

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[0])

    def with_updates(
        self,
        *,
        positions: np.ndarray | None = None,
        magnetization: np.ndarray | None = None,
    ) -> "ParticleEnsemble":
        """Return a copy with updated positions and/or magnetization."""

        return replace(
            self,
            positions=self.positions if positions is None else positions,
            magnetization=(
                self.magnetization if magnetization is None else magnetization
            ),
        )


def make_motion_field_maps_2d(
    x_axis: Iterable[float] | np.ndarray,
    z_axis: Iterable[float] | np.ndarray,
    *,
    b0_map: Iterable[float] | np.ndarray | None = None,
    b0_vector_map: Iterable[float] | np.ndarray | None = None,
    b1_tx_map: Iterable[float] | np.ndarray | None = None,
    b1_tx_vector_map: Iterable[float] | np.ndarray | None = None,
    b1_rx_map: Iterable[float] | np.ndarray | None = None,
    b1_rx_vector_map: Iterable[float] | np.ndarray | None = None,
) -> MotionFieldMaps2D:
    """Validate and assemble two-dimensional field maps."""

    x = _strict_axis(x_axis, "x_axis", min_size=2)
    z = _strict_axis(z_axis, "z_axis", min_size=2)
    shape = (x.size, z.size)
    b0 = (
        np.zeros(shape, dtype=np.float64)
        if b0_map is None
        else _map2d(b0_map, "b0_map")
    )
    if b0_vector_map is not None:
        _vector_map(b0_vector_map, "b0_vector_map")
    if b1_tx_map is not None and b1_tx_vector_map is not None:
        raise ValueError("provide either b1_tx_map or b1_tx_vector_map, not both")
    if b1_rx_map is not None and b1_rx_vector_map is not None:
        raise ValueError("provide either b1_rx_map or b1_rx_vector_map, not both")
    b1_tx = (
        np.ones(shape, dtype=np.float64)
        if b1_tx_map is None
        else _map2d(b1_tx_map, "b1_tx_map")
    )
    if b1_tx_vector_map is not None:
        if b0_vector_map is None:
            raise ValueError("b1_tx_vector_map requires b0_vector_map")
        b1_tx = transverse_b1_magnitude(b0_vector_map, b1_tx_vector_map)
    if b1_rx_vector_map is not None:
        if b0_vector_map is None:
            raise ValueError("b1_rx_vector_map requires b0_vector_map")
        b1_rx = transverse_b1_magnitude(b0_vector_map, b1_rx_vector_map)
    else:
        b1_rx = b1_tx.copy() if b1_rx_map is None else _map2d(b1_rx_map, "b1_rx_map")
    for name, arr in {
        "b0_map": b0,
        "b1_tx_map": b1_tx,
        "b1_rx_map": b1_rx,
    }.items():
        if arr.shape != shape:
            raise ValueError(f"{name} must have shape (len(x_axis), len(z_axis))")
    if np.any(b1_tx < 0.0) or np.any(b1_rx < 0.0):
        raise ValueError("B1 maps must be non-negative")
    return MotionFieldMaps2D(x, z, b0, b1_tx, b1_rx)


def transverse_b1_magnitude(
    b0_vector_map: Iterable[float] | np.ndarray,
    b1_vector_map: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Return the local B1 magnitude perpendicular to the local B0 direction.

    The last axis contains vector components. B0 vectors are real-valued field
    directions; B1 vectors may be real or complex. The returned scalar map is
    `|B1 - (B1 dot b0_hat) b0_hat|`.
    """

    b0 = _vector_map(b0_vector_map, "b0_vector_map").astype(np.float64)
    b1 = np.asarray(b1_vector_map)
    if b1.shape != b0.shape:
        raise ValueError("b1_vector_map must have the same shape as b0_vector_map")
    if not np.all(np.isfinite(b1)):
        raise ValueError("b1_vector_map must contain finite values")
    b0_norm = np.linalg.norm(b0, axis=-1)
    if np.any(b0_norm <= 0.0):
        raise ValueError("b0_vector_map must not contain zero vectors")
    b0_hat = b0 / b0_norm[..., np.newaxis]
    parallel = np.sum(b1 * b0_hat, axis=-1)
    perpendicular = b1 - parallel[..., np.newaxis] * b0_hat
    return np.sqrt(np.sum(np.abs(perpendicular) ** 2, axis=-1)).astype(np.float64)


def initialize_ensemble_from_density(
    rho: Iterable[float] | np.ndarray,
    x_axis: Iterable[float] | np.ndarray,
    z_axis: Iterable[float] | np.ndarray,
    *,
    walkers_per_cell: int = 1,
    diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0,
    seed: int | None = None,
    jitter: bool = False,
) -> ParticleEnsemble:
    """Create a walker ensemble from a two-dimensional spin-density map."""

    density = _map2d(rho, "rho")
    x = _strict_axis(x_axis, "x_axis", min_size=1)
    z = _strict_axis(z_axis, "z_axis", min_size=1)
    if density.shape != (x.size, z.size):
        raise ValueError("rho must have shape (len(x_axis), len(z_axis))")
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")

    xx, zz = np.meshgrid(x, z, indexing="ij")
    base_positions = np.column_stack((xx.ravel(), zz.ravel()))
    positions = np.repeat(base_positions, int(walkers_per_cell), axis=0)

    if jitter:
        rng = np.random.default_rng(seed)
        dx = _cell_widths(x)
        dz = _cell_widths(z)
        widths = np.column_stack((dx.repeat(z.size), np.tile(dz, x.size)))
        widths = np.repeat(widths, int(walkers_per_cell), axis=0)
        positions = positions + rng.uniform(-0.5, 0.5, size=positions.shape) * widths
        positions = apply_boundary(positions, ((x[0], x[-1]), (z[0], z[-1])), "clip")

    weights = np.repeat(density.ravel() / int(walkers_per_cell), int(walkers_per_cell))
    diffusion = _particle_values(
        diffusion_coefficient,
        density.size,
        "diffusion_coefficient",
    )
    diffusion = np.repeat(diffusion, int(walkers_per_cell))
    magnetization = np.zeros((3, positions.shape[0]), dtype=np.complex128)
    magnetization[0, :] = 1.0
    return ParticleEnsemble(
        positions=positions.astype(np.float64),
        magnetization=magnetization,
        weights=weights.astype(np.float64),
        diffusion_coefficient=diffusion.astype(np.float64),
    )


def advect_diffuse_positions(
    positions: np.ndarray,
    dt: float,
    *,
    velocity: Velocity = None,
    diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0,
    rng: np.random.Generator | None = None,
    time: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    boundary: Boundary = "reflect",
) -> np.ndarray:
    """Advance positions with deterministic advection and Brownian diffusion."""

    pos = _positions2d(positions)
    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    updated = pos + _velocity_array(velocity, pos, float(time)) * float(dt)
    diffusion = _particle_values(
        diffusion_coefficient,
        pos.shape[0],
        "diffusion_coefficient",
    )
    if np.any(diffusion < 0.0):
        raise ValueError("diffusion_coefficient must be non-negative")
    if np.any(diffusion > 0.0) and dt > 0.0:
        generator = np.random.default_rng() if rng is None else rng
        sigma = np.sqrt(2.0 * diffusion * float(dt))
        updated = updated + generator.normal(size=pos.shape) * sigma[:, np.newaxis]
    if bounds is not None:
        updated = apply_boundary(updated, bounds, boundary)
    return updated


def move_ensemble(
    ensemble: ParticleEnsemble,
    dt: float,
    *,
    velocity: Velocity = None,
    rng: np.random.Generator | None = None,
    time: float = 0.0,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    boundary: Boundary = "reflect",
) -> ParticleEnsemble:
    """Return an ensemble with advected/diffused positions."""

    positions = advect_diffuse_positions(
        ensemble.positions,
        dt,
        velocity=velocity,
        diffusion_coefficient=ensemble.diffusion_coefficient,
        rng=rng,
        time=time,
        bounds=bounds,
        boundary=boundary,
    )
    return ensemble.with_updates(positions=positions)


def apply_free_precession(
    ensemble: ParticleEnsemble,
    dt: float,
    off_resonance: Iterable[float] | np.ndarray,
    *,
    t1: float | Iterable[float] | np.ndarray = np.inf,
    t2: float | Iterable[float] | np.ndarray = np.inf,
    mth: float | Iterable[float] | np.ndarray = 1.0,
) -> ParticleEnsemble:
    """Apply relaxation and off-resonance precession to each particle."""

    if dt < 0.0:
        raise ValueError("dt must be non-negative")
    num_particles = ensemble.num_particles
    omega = _particle_values(off_resonance, num_particles, "off_resonance")
    t1_arr = _positive_particle_values(t1, num_particles, "t1")
    t2_arr = _positive_particle_values(t2, num_particles, "t2")
    mth_arr = _particle_values(mth, num_particles, "mth")

    mag = np.array(ensemble.magnetization, dtype=np.complex128, copy=True)
    e1 = np.exp(-float(dt) / t1_arr)
    e2 = np.exp(-float(dt) / t2_arr)
    mag[0, :] = e1 * mag[0, :] + mth_arr * (1.0 - e1)
    mag[1, :] = e2 * np.exp(-1j * omega * float(dt)) * mag[1, :]
    mag[2, :] = e2 * np.exp(1j * omega * float(dt)) * mag[2, :]
    return ensemble.with_updates(magnetization=mag)


def apply_rf_rotation(
    ensemble: ParticleEnsemble,
    duration: float,
    phase: float,
    amplitude: float,
    off_resonance: Iterable[float] | np.ndarray,
    *,
    b1_tx: float | Iterable[float] | np.ndarray = 1.0,
) -> ParticleEnsemble:
    """Apply a rectangular RF rotation using local B1 transmit scaling."""

    if duration < 0.0:
        raise ValueError("duration must be non-negative")
    num_particles = ensemble.num_particles
    omega = _particle_values(off_resonance, num_particles, "off_resonance")
    b1 = _particle_values(b1_tx, num_particles, "b1_tx")
    if np.any(b1 < 0.0):
        raise ValueError("b1_tx must be non-negative")
    mat = rf_matrix_elements(
        omega,
        float(amplitude) * b1,
        float(duration),
        float(phase),
    )
    return _apply_matrix_elements(
        ensemble,
        mat,
        np.zeros(num_particles, dtype=np.complex128),
    )


def free_precession_with_motion_step(
    ensemble: ParticleEnsemble,
    fields: MotionFieldMaps2D,
    dt: float,
    *,
    velocity: Velocity = None,
    rng: np.random.Generator | None = None,
    time: float = 0.0,
    gradient: tuple[float, float] = (0.0, 0.0),
    t1: float | Iterable[float] | np.ndarray = np.inf,
    t2: float | Iterable[float] | np.ndarray = np.inf,
    mth: float | Iterable[float] | np.ndarray = 1.0,
    boundary: Boundary = "reflect",
) -> ParticleEnsemble:
    """Move particles and apply a first-order free-precession update.

    For accurate motion through sharply varying maps, split long sequence
    intervals into smaller calls to this helper.
    """

    moved = move_ensemble(
        ensemble,
        dt,
        velocity=velocity,
        rng=rng,
        time=time,
        bounds=fields.bounds,
        boundary=boundary,
    )
    sampled = fields.sample(moved.positions)
    grad = np.asarray(gradient, dtype=np.float64).reshape(2)
    gradient_offset = moved.positions @ grad
    return apply_free_precession(
        moved,
        dt,
        sampled["b0"] + gradient_offset,
        t1=t1,
        t2=t2,
        mth=mth,
    )


def receive_signal(
    ensemble: ParticleEnsemble,
    fields: MotionFieldMaps2D | None = None,
) -> complex:
    """Sum weighted received transverse magnetization over particles."""

    receive_weight = 1.0
    if fields is not None:
        receive_weight = fields.sample(ensemble.positions)["b1_rx"]
    return complex(
        np.sum(ensemble.weights * receive_weight * ensemble.magnetization[1, :])
    )


def make_circular_reflector(
    center: tuple[float, float],
    radius: float,
) -> BoundaryFn:
    """Return a reflecting-wall boundary callback for a circular pore.

    The returned function maps walker positions so that any walker stepping
    outside the circle is folded back along its radial direction, the curved
    analogue of the rectangular ``"reflect"`` mode. Pass it as the ``boundary``
    argument to ``run_motion_sequence`` (or ``run_pgse_walkers``) to model
    isotropically restricted diffusion inside a disc.

    Accurate reflection assumes the per-substep diffusion length stays well
    below ``radius``; refine ``substeps_per_interval`` if walkers routinely
    overshoot the wall by more than a small fraction of the radius.
    """

    cx = float(center[0])
    cz = float(center[1])
    r = float(radius)
    if r <= 0.0:
        raise ValueError("radius must be positive")

    def reflect(positions: np.ndarray) -> np.ndarray:
        pos = _positions2d(positions).copy()
        dx = pos[:, 0] - cx
        dz = pos[:, 1] - cz
        dist = np.hypot(dx, dz)
        # Fold the radius into [0, r] with a reflecting wall at r, mirroring the
        # triangle-wave fold used by the rectangular "reflect" mode.
        period = 2.0 * r
        folded_mod = np.mod(dist, period)
        folded = np.where(folded_mod <= r, folded_mod, period - folded_mod)
        scale = np.divide(folded, dist, out=np.ones_like(dist), where=dist > 0.0)
        pos[:, 0] = cx + dx * scale
        pos[:, 1] = cz + dz * scale
        return pos

    return reflect


def apply_boundary(
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    mode: Boundary,
) -> np.ndarray:
    """Apply boundary conditions to two-dimensional positions.

    ``mode`` is one of the rectangular-box modes ``"reflect"``, ``"periodic"``,
    or ``"clip"``, or a callable mapping positions to confined positions (in
    which case ``bounds`` is ignored), as produced by ``make_circular_reflector``.
    """

    pos = _positions2d(positions).copy()
    if callable(mode):
        return _positions2d(mode(pos))
    for dim, (lower, upper) in enumerate(bounds):
        lo = float(lower)
        hi = float(upper)
        if hi < lo:
            raise ValueError("each bounds entry must satisfy lower <= upper")
        if hi == lo:
            pos[:, dim] = lo
            continue
        if mode == "clip":
            pos[:, dim] = np.clip(pos[:, dim], lo, hi)
        elif mode == "periodic":
            pos[:, dim] = lo + np.mod(pos[:, dim] - lo, hi - lo)
        elif mode == "reflect":
            width = hi - lo
            folded = np.mod(pos[:, dim] - lo, 2.0 * width)
            pos[:, dim] = lo + np.where(folded <= width, folded, 2.0 * width - folded)
        else:
            raise ValueError("boundary must be 'reflect', 'periodic', or 'clip'")
    return pos


def _apply_matrix_elements(
    ensemble: ParticleEnsemble,
    mat: MatrixElements,
    mlong: np.ndarray,
) -> ParticleEnsemble:
    tmp = ensemble.magnetization
    mag = np.zeros_like(tmp)
    mag[0, :] = (
        mat.R_00 * tmp[0, :]
        + mat.R_0m * tmp[1, :]
        + mat.R_0p * tmp[2, :]
        + mlong
    )
    mag[1, :] = mat.R_m0 * tmp[0, :] + mat.R_mm * tmp[1, :] + mat.R_mp * tmp[2, :]
    mag[2, :] = mat.R_p0 * tmp[0, :] + mat.R_pm * tmp[1, :] + mat.R_pp * tmp[2, :]
    return ensemble.with_updates(magnetization=mag)


def _bilinear_sample(
    values: np.ndarray,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    x = np.clip(positions[:, 0], x_axis[0], x_axis[-1])
    z = np.clip(positions[:, 1], z_axis[0], z_axis[-1])
    ix1 = np.searchsorted(x_axis, x, side="right")
    iz1 = np.searchsorted(z_axis, z, side="right")
    ix1 = np.clip(ix1, 1, x_axis.size - 1)
    iz1 = np.clip(iz1, 1, z_axis.size - 1)
    ix0 = ix1 - 1
    iz0 = iz1 - 1
    x0 = x_axis[ix0]
    x1 = x_axis[ix1]
    z0 = z_axis[iz0]
    z1 = z_axis[iz1]
    tx = np.divide(x - x0, x1 - x0, out=np.zeros_like(x), where=(x1 != x0))
    tz = np.divide(z - z0, z1 - z0, out=np.zeros_like(z), where=(z1 != z0))

    v00 = values[ix0, iz0]
    v10 = values[ix1, iz0]
    v01 = values[ix0, iz1]
    v11 = values[ix1, iz1]
    return (
        (1.0 - tx) * (1.0 - tz) * v00
        + tx * (1.0 - tz) * v10
        + (1.0 - tx) * tz * v01
        + tx * tz * v11
    )


def _strict_axis(
    values: Iterable[float] | np.ndarray,
    name: str,
    *,
    min_size: int,
) -> np.ndarray:
    axis = np.asarray(values, dtype=np.float64).reshape(-1)
    if axis.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _map2d(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _vector_map(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim < 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (..., 3)")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _positions2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("positions must have shape (num_particles, 2)")
    if not np.all(np.isfinite(arr)):
        raise ValueError("positions must contain finite values")
    return arr


def _particle_values(
    values: float | Iterable[float] | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
        if arr.size != size:
            raise ValueError(f"{name} must be scalar or have one value per particle")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _positive_particle_values(
    values: float | Iterable[float] | np.ndarray,
    size: int,
    name: str,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(size, float(arr), dtype=np.float64)
    else:
        arr = arr.reshape(-1)
        if arr.size != size:
            raise ValueError(f"{name} must be scalar or have one value per particle")
    if np.any(np.isnan(arr)):
        raise ValueError(f"{name} must not contain NaN")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive")
    return arr


def _velocity_array(
    velocity: Velocity,
    positions: np.ndarray,
    time: float,
) -> np.ndarray:
    if velocity is None:
        return np.zeros_like(positions)
    values = velocity(positions, time) if callable(velocity) else velocity
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("velocity vector must have two components")
        arr = np.tile(arr, (positions.shape[0], 1))
    if arr.shape != positions.shape:
        raise ValueError("velocity must have shape (2,) or positions.shape")
    if not np.all(np.isfinite(arr)):
        raise ValueError("velocity must contain finite values")
    return arr


def _cell_widths(axis: np.ndarray) -> np.ndarray:
    if axis.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return np.diff(edges)
