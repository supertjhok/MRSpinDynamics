"""Lagrangian isochromat motion helpers for advection and diffusion physics.

The existing arbitrary-pulse kernels treat each isochromat as fixed in space.
This module provides opt-in building blocks for simulations where spins move
through spatial B0/B1 maps between sequence updates.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from spin_dynamics.core.rotations import MatrixElements, rf_matrix_elements
from spin_dynamics.fields import positions as _fields_positions
from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.interpolate import dlinear_sample as _dlinear_sample


BoundaryMode = Literal["reflect", "periodic", "clip"]
BoundaryFn = Callable[..., np.ndarray]
# A boundary is either one of the rectangular-box modes or a callable that maps
# ``(num_particles, ndim)`` positions to confined positions. Context-aware
# callables may also accept ``previous_positions``, ``dt``, ``rng``, ``time``,
# and ``bounds`` keyword arguments.
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
class MotionFieldMaps:
    """N-dimensional (1-, 2-, or 3-D) field maps for moving isochromats.

    The dimension-agnostic counterpart to :class:`MotionFieldMaps2D`: ``b0_map``
    (normalized angular off-resonance) and the relative ``b1_tx_map``/
    ``b1_rx_map`` share the shape of ``domain``. ``sample`` multilinearly
    interpolates at ``(num_particles, d)`` positions; ``bounds`` returns the
    per-axis extent the boundary handling needs. Either container can be passed
    to ``run_motion_sequence`` -- it only requires ``sample`` and ``bounds``.
    """

    domain: SpatialDomain
    b0_map: np.ndarray
    b1_tx_map: np.ndarray
    b1_rx_map: np.ndarray

    @property
    def ndim(self) -> int:
        return self.domain.ndim

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return self.domain.bounds

    def sample(self, positions: np.ndarray) -> dict[str, np.ndarray]:
        axes = self.domain.axes
        return {
            "b0": _dlinear_sample(self.b0_map, axes, positions),
            "b1_tx": _dlinear_sample(self.b1_tx_map, axes, positions),
            "b1_rx": _dlinear_sample(self.b1_rx_map, axes, positions),
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


def make_motion_field_maps(
    axes: SpatialDomain | Sequence[Iterable[float] | np.ndarray],
    *,
    b0_map: Iterable[float] | np.ndarray | None = None,
    b1_tx_map: Iterable[float] | np.ndarray | None = None,
    b1_rx_map: Iterable[float] | np.ndarray | None = None,
) -> MotionFieldMaps:
    """Assemble 1-, 2-, or 3-D motion field maps over ``axes``.

    ``axes`` is either a :class:`SpatialDomain` or a sequence of strictly
    increasing coordinate axes. Missing ``b0`` defaults to zero off-resonance and
    missing ``b1`` maps to uniform unit sensitivity. ``b1_rx`` defaults to
    ``b1_tx``.
    """

    domain = axes if isinstance(axes, SpatialDomain) else SpatialDomain(tuple(axes))
    shape = domain.shape
    b0 = np.zeros(shape) if b0_map is None else np.asarray(b0_map, dtype=np.float64)
    b1_tx = np.ones(shape) if b1_tx_map is None else np.asarray(b1_tx_map, dtype=np.float64)
    b1_rx = b1_tx.copy() if b1_rx_map is None else np.asarray(b1_rx_map, dtype=np.float64)
    for name, arr in (("b0_map", b0), ("b1_tx_map", b1_tx), ("b1_rx_map", b1_rx)):
        if arr.shape != shape:
            raise ValueError(f"{name} must have the same shape as the domain")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
    if np.any(b1_tx < 0.0) or np.any(b1_rx < 0.0):
        raise ValueError("B1 maps must be non-negative")
    return MotionFieldMaps(domain=domain, b0_map=b0, b1_tx_map=b1_tx, b1_rx_map=b1_rx)


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
    return _ensemble_from_axes(
        density,
        (x, z),
        walkers_per_cell=walkers_per_cell,
        diffusion_coefficient=diffusion_coefficient,
        seed=seed,
        jitter=jitter,
    )


def initialize_ensemble_from_domain(
    domain: SpatialDomain,
    rho: Iterable[float] | np.ndarray,
    *,
    walkers_per_cell: int = 1,
    diffusion_coefficient: float | Iterable[float] | np.ndarray = 0.0,
    seed: int | None = None,
    jitter: bool = False,
) -> ParticleEnsemble:
    """Create a walker ensemble from a 1-, 2-, or 3-D spin-density volume.

    The dimension-agnostic counterpart to ``initialize_ensemble_from_density``:
    ``rho`` must match ``domain.shape`` and walkers are seeded at the voxel
    centers of ``domain``. For a two-axis domain this is identical to the
    ``(x_axis, z_axis)`` entry point.
    """

    density = np.asarray(rho, dtype=np.float64)
    if density.shape != domain.shape:
        raise ValueError("rho must have the same shape as the domain")
    if not np.all(np.isfinite(density)):
        raise ValueError("rho must contain finite values")
    return _ensemble_from_axes(
        density,
        domain.axes,
        walkers_per_cell=walkers_per_cell,
        diffusion_coefficient=diffusion_coefficient,
        seed=seed,
        jitter=jitter,
    )


def _ensemble_from_axes(
    density: np.ndarray,
    axes: tuple[np.ndarray, ...],
    *,
    walkers_per_cell: int,
    diffusion_coefficient: float | Iterable[float] | np.ndarray,
    seed: int | None,
    jitter: bool,
) -> ParticleEnsemble:
    if walkers_per_cell <= 0:
        raise ValueError("walkers_per_cell must be positive")
    ndim = len(axes)
    shape = tuple(int(a.size) for a in axes)

    grids = np.meshgrid(*axes, indexing="ij")
    base_positions = np.column_stack([grid.ravel() for grid in grids])
    positions = np.repeat(base_positions, int(walkers_per_cell), axis=0)

    if jitter:
        rng = np.random.default_rng(seed)
        width_columns = []
        for k, axis in enumerate(axes):
            widths_k = _cell_widths(axis)
            reshaped = widths_k.reshape(
                tuple(shape[j] if j == k else 1 for j in range(ndim))
            )
            width_columns.append(np.broadcast_to(reshaped, shape).ravel())
        widths = np.column_stack(width_columns)
        widths = np.repeat(widths, int(walkers_per_cell), axis=0)
        positions = positions + rng.uniform(-0.5, 0.5, size=positions.shape) * widths
        bounds = tuple((float(a[0]), float(a[-1])) for a in axes)
        positions = apply_boundary(positions, bounds, "clip")

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

    pos = _positions_nd(positions)
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
    generator = rng
    if np.any(diffusion > 0.0) and dt > 0.0:
        generator = np.random.default_rng() if generator is None else generator
        sigma = np.sqrt(2.0 * diffusion * float(dt))
        updated = updated + generator.normal(size=pos.shape) * sigma[:, np.newaxis]
    if bounds is not None:
        if callable(boundary) and generator is None:
            generator = np.random.default_rng()
        updated = apply_boundary(
            updated,
            bounds,
            boundary,
            previous_positions=pos,
            rng=generator,
            time=time,
            dt=dt,
        )
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


def make_elliptical_reflector(
    center: tuple[float, float],
    semi_axes: tuple[float, float],
) -> BoundaryFn:
    """Return a reflecting-wall boundary callback for an elliptical pore.

    ``semi_axes`` are the ``(x, z)`` half-widths of the ellipse. Reflection is
    performed in normalized coordinates ``(x/ax, z/az)``, where the ellipse maps
    to the unit circle, so a uniform spin density stays uniform inside the pore.
    This is the anisotropic generalization of ``make_circular_reflector`` and is
    the geometry that makes double diffusion encoding (DDE) sensitive to
    microscopic anisotropy.

    As with the circular reflector, accurate reflection assumes the per-substep
    diffusion length stays well below the smaller semi-axis.
    """

    cx = float(center[0])
    cz = float(center[1])
    ax = float(semi_axes[0])
    az = float(semi_axes[1])
    if ax <= 0.0 or az <= 0.0:
        raise ValueError("semi-axes must be positive")

    def reflect(positions: np.ndarray) -> np.ndarray:
        pos = _positions2d(positions).copy()
        u = (pos[:, 0] - cx) / ax
        v = (pos[:, 1] - cz) / az
        radius = np.hypot(u, v)
        # Fold the normalized radius into the unit disc (ellipse boundary at 1).
        folded_mod = np.mod(radius, 2.0)
        folded = np.where(folded_mod <= 1.0, folded_mod, 2.0 - folded_mod)
        scale = np.divide(folded, radius, out=np.ones_like(radius), where=radius > 0.0)
        pos[:, 0] = cx + u * scale * ax
        pos[:, 1] = cz + v * scale * az
        return pos

    return reflect


def make_semipermeable_plane(
    interface: float,
    exchange_rate: float,
    *,
    axis: Literal["x", "z"] = "x",
    outer_boundary: BoundaryMode = "reflect",
) -> BoundaryFn:
    """Return a stochastic semi-permeable internal plane boundary.

    The membrane is the line ``x = interface`` or ``z = interface`` inside the
    rectangular simulation bounds. Walkers that do not cross the membrane are
    left alone. Walkers that do cross transmit with probability
    ``1 - exp(-exchange_rate * dt)`` and otherwise reflect from the membrane.

    ``exchange_rate`` has units of inverse simulation time. Use ``0`` for an
    impermeable internal wall and ``np.inf`` for a freely transmitting
    interface. The returned boundary is intended for the motion helpers, which
    provide the previous positions, time step, and random generator needed for
    stochastic exchange.
    """

    membrane = float(interface)
    rate = float(exchange_rate)
    if not np.isfinite(membrane):
        raise ValueError("interface must be finite")
    if np.isnan(rate) or rate < 0.0:
        raise ValueError("exchange_rate must be non-negative")
    if axis not in {"x", "z"}:
        raise ValueError("axis must be 'x' or 'z'")
    if outer_boundary not in {"reflect", "periodic", "clip"}:
        raise ValueError("outer_boundary must be 'reflect', 'periodic', or 'clip'")
    dim = 0 if axis == "x" else 1

    def exchange(
        positions: np.ndarray,
        *,
        previous_positions: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        dt: float = 0.0,
        bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
        **_: object,
    ) -> np.ndarray:
        pos = _positions2d(positions).copy()
        if previous_positions is not None:
            prev = _positions2d(previous_positions)
            if prev.shape != pos.shape:
                raise ValueError("previous_positions must match positions.shape")
            left_prev = prev[:, dim] < membrane
            left_next = pos[:, dim] < membrane
            crossing = left_prev != left_next
            if np.any(crossing):
                if dt < 0.0:
                    raise ValueError("dt must be non-negative")
                if rate == np.inf:
                    transmit_probability = 1.0
                elif dt == 0.0:
                    transmit_probability = 0.0
                else:
                    transmit_probability = -np.expm1(-rate * float(dt))
                generator = np.random.default_rng() if rng is None else rng
                transmitted = np.zeros(pos.shape[0], dtype=bool)
                transmitted[crossing] = (
                    generator.random(int(np.count_nonzero(crossing)))
                    < transmit_probability
                )
                blocked = crossing & ~transmitted
                pos[blocked, dim] = 2.0 * membrane - pos[blocked, dim]
        if bounds is None:
            return pos
        return apply_boundary(pos, bounds, outer_boundary)

    return exchange


def apply_boundary(
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    mode: Boundary,
    *,
    previous_positions: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    time: float = 0.0,
    dt: float = 0.0,
) -> np.ndarray:
    """Apply boundary conditions to two-dimensional positions.

    ``mode`` is one of the rectangular-box modes ``"reflect"``, ``"periodic"``,
    or ``"clip"``, or a callable mapping positions to confined positions. Plain
    callables such as ``make_circular_reflector`` only receive positions;
    context-aware callables such as ``make_semipermeable_plane`` also receive
    previous positions, the time step, random generator, current time, and
    rectangular bounds when their signature accepts those keywords.
    """

    pos = _positions_nd(positions).copy()
    if callable(mode):
        return _positions_nd(
            _call_boundary(
                mode,
                pos,
                bounds,
                previous_positions=previous_positions,
                rng=rng,
                time=time,
                dt=dt,
            ),
            pos.shape[1],
        )
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


def _call_boundary(
    callback: BoundaryFn,
    positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float]],
    *,
    previous_positions: np.ndarray | None,
    rng: np.random.Generator | None,
    time: float,
    dt: float,
) -> np.ndarray:
    context = {
        "previous_positions": previous_positions,
        "rng": rng,
        "time": time,
        "dt": dt,
        "bounds": bounds,
    }
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(positions, **context)
    params = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if accepts_kwargs:
        return callback(positions, **context)
    accepted = {
        name: value
        for name, value in context.items()
        if name in params
        and params[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return callback(positions, **accepted)


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
    """Bilinearly sample a 2-D map (thin shim over the d-linear sampler)."""

    return _dlinear_sample(values, (x_axis, z_axis), positions)


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


def _positions_nd(values: np.ndarray, ndim: int | None = None) -> np.ndarray:
    return _fields_positions.positions_nd(values, ndim)


def _positions2d(values: np.ndarray) -> np.ndarray:
    return _fields_positions.positions_nd(values, 2)


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
    return _fields_positions.velocity_array(velocity, positions, time)


def _cell_widths(axis: np.ndarray) -> np.ndarray:
    if axis.size == 1:
        return np.ones(1, dtype=np.float64)
    edges = np.empty(axis.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    return np.diff(edges)
