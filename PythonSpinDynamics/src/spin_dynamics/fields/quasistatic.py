"""Quasistatic induced-E-field and eddy-current solver (first-order / Born).

Adds the E-field the Biot-Savart B solver cannot see: in the magnetoquasistatic
regime a time-varying coil current induces ``E = -dA/dt - grad(phi)``, and in a
conductor ``J = sigma*E`` drives eddy currents and resistive power loss. The
vector potential ``A`` is computed from the same filament segments as
``magnetostatics.biot_savart``.

**First-order (Born) model -- limitations (see docs/field_solvers_quasistatic.md):**
this uses the *primary* coil field only and ignores the secondary (reaction)
field of the eddy currents. It is valid when the eddy currents do not
significantly perturb the source -- low conductivity, thin samples, or skin
depth ``sqrt(2/(mu0 sigma omega)) >> sample size``. There is **no skin-effect
shielding**, so at high conductivity/frequency it over-estimates the interior
field and deposited power. The ``phi`` charge correction enforces
``J.n = 0`` in a finite sample but is still first order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from spin_dynamics.fields.magnetostatics import MU0, circular_loop
from spin_dynamics.fields.validity import (
    QuasistaticAssessment,
    QuasistaticRegion,
    ValidityPolicy,
    apply_validity_policy,
    assess_quasistatic_validity,
)

Segment = tuple[Sequence[float], Sequence[float]]


def _characteristic_extent(points: np.ndarray, minimum: float = 0.0) -> float:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if values.size == 0:
        return float(minimum)
    return float(max(np.linalg.norm(np.ptp(values, axis=0)), minimum))


def _conductive_validity_assessment(
    grid_points: np.ndarray,
    segments: Sequence[Segment],
    *,
    mask: np.ndarray,
    spacing: Sequence[float],
    conductivity: float,
    frequency: float,
    relative_permittivity: float,
    relative_permeability: float,
    validity_length_m: float | None,
) -> QuasistaticAssessment:
    segment_points = np.asarray(
        [point for segment in segments for point in segment],
        dtype=np.float64,
    )
    minimum = float(np.max(np.asarray(spacing, dtype=np.float64)))
    masked_points = np.asarray(grid_points, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    length = (
        _characteristic_extent(masked_points, minimum)
        if validity_length_m is None
        else float(validity_length_m)
    )
    region = QuasistaticRegion(
        name="conductive sample",
        characteristic_length_m=length,
        relative_permittivity=relative_permittivity,
        conductivity_s_per_m=conductivity,
        relative_permeability=relative_permeability,
        born_approximation=True,
    )
    coil_extent = _characteristic_extent(segment_points)
    return assess_quasistatic_validity(
        frequency,
        coil_extent_m=coil_extent if coil_extent > 0.0 else None,
        regions=(region,),
    )


def vector_potential(
    points: np.ndarray,
    segments: Sequence[Segment],
    current: float = 1.0,
) -> np.ndarray:
    """Magnetic vector potential ``A`` (T*m) of straight current segments.

    ``points`` has shape ``(..., 3)`` (m); ``segments`` a sequence of
    ``(start, end)`` endpoints (m); ``current`` in amperes. Each straight
    segment contributes the closed form
    ``A = mu0 I/(4 pi) * e_hat * ln((R1+R2+L)/(R1+R2-L))`` (``e_hat`` along the
    segment), whose curl reproduces :func:`magnetostatics.biot_savart`.
    """

    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    total = np.zeros_like(pts)
    for start, end in segments:
        x1 = np.asarray(start, dtype=np.float64)
        x2 = np.asarray(end, dtype=np.float64)
        length = np.linalg.norm(x2 - x1)
        if length == 0.0:
            continue
        e = (x2 - x1) / length
        r1 = np.linalg.norm(pts - x1, axis=-1)
        r2 = np.linalg.norm(pts - x2, axis=-1)
        denom = r1 + r2 - length
        num = r1 + r2 + length
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.where(denom > 0.0, np.log(num / np.where(denom > 0.0, denom, 1.0)), 0.0)
        total = total + (MU0 * current / (4.0 * np.pi)) * log_term[..., np.newaxis] * e
    return total


def induced_efield(
    points: np.ndarray,
    segments: Sequence[Segment],
    dcurrent_dt: float,
) -> np.ndarray:
    """Induced (inductive) E-field ``E = -dA/dt`` (V/m) for current slew ``dI/dt``.

    ``A`` is linear in current, so ``E = -A_unit * dI/dt`` where ``A_unit`` is the
    per-ampere vector potential. For a sinusoidal drive ``I = I0 cos(2 pi f t)``
    the peak slew is ``dcurrent_dt = 2 pi f I0``.
    """

    return -vector_potential(points, segments, current=1.0) * float(dcurrent_dt)


def mutual_inductance(loop_a: Sequence[Segment], loop_b: Sequence[Segment]) -> float:
    """Mutual inductance ``M`` (H) between two filamentary loops.

    ``M = oint_a A_b . dl_a`` with ``A_b`` the unit-current vector potential of
    ``loop_b`` -- evaluated as the sum over ``loop_a``'s segments of
    ``A_b(midpoint) . segment_vector``. Symmetric in a/b up to discretization.
    """

    mids = np.array([0.5 * (np.asarray(s) + np.asarray(e)) for s, e in loop_a])
    vecs = np.array([np.asarray(e) - np.asarray(s) for s, e in loop_a])
    a_b = vector_potential(mids, loop_b, current=1.0)
    return float(np.sum(a_b * vecs))


def self_inductance_circular(radius: float, wire_radius: float) -> float:
    """Self-inductance (H) of a circular loop, ``mu0 a (ln(8a/r_w) - 2)``.

    The uniform-surface-current thin-wire formula (Maxwell); ``a`` is the loop
    radius and ``r_w`` the wire radius (``r_w << a``).
    """

    if radius <= 0 or wire_radius <= 0 or wire_radius >= radius:
        raise ValueError("require 0 < wire_radius < radius")
    return float(MU0 * radius * (np.log(8.0 * radius / wire_radius) - 2.0))


def eddy_power(
    e_field: np.ndarray,
    conductivity: np.ndarray | float,
    cell_volume: float,
    *,
    time_average: bool = True,
) -> float:
    """Resistive eddy power ``P = k * integral sigma |E|^2 dV`` (W).

    ``e_field`` is the induced field amplitude ``(..., 3)`` (V/m); ``conductivity``
    is scalar or per-cell (S/m); ``cell_volume`` the grid cell volume (m^3).
    ``time_average=True`` gives ``k = 1/2`` (cycle average for a sinusoidal drive
    whose amplitude is ``e_field``); ``False`` gives the instantaneous ``k = 1``.
    """

    e2 = np.sum(np.abs(e_field) ** 2, axis=-1)
    sigma = np.asarray(conductivity, dtype=np.float64)
    k = 0.5 if time_average else 1.0
    return float(k * np.sum(sigma * e2) * cell_volume)


def _charge_correction(e_field: np.ndarray, mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Return ``grad(phi)`` enforcing ``(E - grad phi).n = 0`` on the mask boundary.

    Solves the finite-volume Neumann problem for the scalar potential so the
    corrected eddy current ``J = sigma(E - grad phi)`` is divergence-free inside
    the conductor and has no normal component at its surface (charge
    conservation in a finite sample). Homogeneous ``sigma`` within the mask.
    """

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    mask = np.asarray(mask, dtype=bool)
    ndim = mask.ndim
    h = np.asarray(spacing, dtype=np.float64)
    if h.size != ndim:
        raise ValueError("spacing must have one entry per grid axis")

    idx = -np.ones(mask.shape, dtype=np.int64)
    cells = np.argwhere(mask)
    idx[tuple(cells.T)] = np.arange(len(cells))
    n = len(cells)
    if n == 0:
        return np.zeros_like(e_field)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(n)
    for d in range(ndim):
        inv_h2 = 1.0 / h[d] ** 2
        for sign in (+1, -1):
            shifted = np.roll(mask, -sign, axis=d)
            # zero out wrap-around at the far edge
            sl = [slice(None)] * ndim
            sl[d] = -1 if sign == +1 else 0
            shifted[tuple(sl)] = False
            interior = mask & shifted  # cells with a conductor neighbor across this face
            here = np.argwhere(interior)
            nb = here.copy()
            nb[:, d] += sign
            i_here = idx[tuple(here.T)]
            i_nb = idx[tuple(nb.T)]
            # Laplacian: (phi_nb - phi_i) / h^2
            rows.extend(i_here.tolist() + i_here.tolist())
            cols.extend(i_nb.tolist() + i_here.tolist())
            data.extend([inv_h2] * len(i_here) + [-inv_h2] * len(i_here))
            # RHS: sign * face-averaged E_d / h_d
            e_here = e_field[..., d][tuple(here.T)]
            e_nb = e_field[..., d][tuple(nb.T)]
            np.add.at(rhs, i_here, sign * 0.5 * (e_here + e_nb) / h[d])

    a = csr_matrix((data, (rows, cols)), shape=(n, n))
    # Neumann system is singular (phi defined up to a constant): pin cell 0.
    a = a.tolil()
    a[0, :] = 0.0
    a[0, 0] = 1.0
    rhs[0] = 0.0
    phi_vec = spsolve(a.tocsr(), rhs)

    phi = np.zeros(mask.shape)
    phi[tuple(cells.T)] = phi_vec
    # Mask-aware gradient: central where both neighbours are conductor, one-sided
    # at the boundary. A plain np.gradient would see the zero-outside padding and
    # manufacture huge spurious gradients at the conductor surface.
    grad = np.zeros(mask.shape + (ndim,))
    for d in range(ndim):
        fwd = np.roll(mask, -1, axis=d)
        bwd = np.roll(mask, +1, axis=d)
        edge_hi = [slice(None)] * ndim
        edge_hi[d] = -1
        fwd[tuple(edge_hi)] = False
        edge_lo = [slice(None)] * ndim
        edge_lo[d] = 0
        bwd[tuple(edge_lo)] = False
        phi_fwd = np.roll(phi, -1, axis=d)
        phi_bwd = np.roll(phi, +1, axis=d)
        central = mask & fwd & bwd
        only_fwd = mask & fwd & ~bwd
        only_bwd = mask & ~fwd & bwd
        g = np.zeros(mask.shape)
        g[central] = (phi_fwd[central] - phi_bwd[central]) / (2.0 * h[d])
        g[only_fwd] = (phi_fwd[only_fwd] - phi[only_fwd]) / h[d]
        g[only_bwd] = (phi[only_bwd] - phi_bwd[only_bwd]) / h[d]
        grad[..., d] = g
    return grad


@dataclass(frozen=True)
class EddyResult:
    """Induced E-field, eddy current and deposited power in a conductive sample."""

    e_field: np.ndarray       # (..., 3) corrected induced field (V/m)
    current_density: np.ndarray  # (..., 3) eddy current density J = sigma E (A/m^2)
    power: float              # resistive power (W)
    e_field_uncorrected: np.ndarray  # (..., 3) before the charge correction


def eddy_currents(
    grid_points: np.ndarray,
    segments: Sequence[Segment],
    dcurrent_dt: float,
    *,
    conductivity: float,
    mask: np.ndarray,
    spacing: Sequence[float],
    charge_correction: bool = True,
    time_average: bool = True,
) -> EddyResult:
    """First-order eddy currents and power for a conductive sample on a grid.

    ``grid_points`` is ``(*grid, 3)`` (m); ``mask`` (``*grid`` bool) marks the
    conductor; ``spacing`` the per-axis cell size (m). Computes the induced
    field, optionally applies the :func:`_charge_correction` so the eddy current
    is tangential at the sample surface, and integrates the deposited power over
    the conductor. See the module docstring for the first-order limitations.
    """

    e_ind = induced_efield(grid_points, segments, dcurrent_dt)
    e = e_ind.copy()
    if charge_correction:
        e = e - _charge_correction(e_ind, mask, spacing)
    mask_arr = np.asarray(mask, dtype=bool)
    e[~mask_arr] = 0.0
    cell_volume = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    j = conductivity * e
    power = eddy_power(e, conductivity, cell_volume, time_average=time_average)
    return EddyResult(e_field=e, current_density=j, power=power, e_field_uncorrected=e_ind)


def geometric_loss_integral(
    grid_points: np.ndarray,
    segments: Sequence[Segment],
    *,
    conductivity: float,
    mask: np.ndarray,
    spacing: Sequence[float],
    charge_correction: bool = False,
) -> float:
    """``G = integral sigma |A_unit|^2 dV`` (ohm/(rad/s)^2) over the conductor.

    The frequency-independent geometry factor behind the reflected resistance:
    ``R_reflected(omega) = omega^2 * G`` (see :func:`reflected_resistance`).
    Computed as ``2 * power`` of :func:`eddy_currents` at unit ``dI/dt`` (so the
    induced field is exactly ``A_unit``). ``charge_correction`` defaults False:
    for an axisymmetric coil + coaxial conductor the eddy current is azimuthal and
    already divergence-free, so no correction is needed.
    """

    res = eddy_currents(
        grid_points, segments, 1.0, conductivity=conductivity, mask=mask,
        spacing=spacing, charge_correction=charge_correction, time_average=True,
    )
    return 2.0 * res.power


def reflected_resistance(
    grid_points: np.ndarray,
    segments: Sequence[Segment],
    *,
    conductivity: float,
    mask: np.ndarray,
    spacing: Sequence[float],
    frequency: float,
    charge_correction: bool = False,
    relative_permittivity: float = 1.0,
    relative_permeability: float = 1.0,
    validity_length_m: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> float:
    """Reflected series resistance ``R = omega^2 integral sigma |A_unit|^2 dV`` (ohm).

    Eddy-current loss in a conductive sample couples back into the coil (by
    reciprocity) as an added series resistance: the time-average dissipated power
    ``1/2 I0^2 R`` equals ``1/2 omega^2 I0^2 integral sigma |A_unit|^2 dV``. This
    ``R_reflected(omega)`` grows as ``omega^2`` (first-order / Born regime) and is
    what degrades the loaded Q and raises the Johnson-noise floor. First-order
    limitations apply (module docstring): no skin-effect shielding, so it
    over-estimates loss once the skin depth drops below the sample size.
    """

    assessment = _conductive_validity_assessment(
        grid_points,
        segments,
        mask=mask,
        spacing=spacing,
        conductivity=conductivity,
        frequency=float(frequency),
        relative_permittivity=relative_permittivity,
        relative_permeability=relative_permeability,
        validity_length_m=validity_length_m,
    )
    apply_validity_policy(
        assessment,
        solver_name="reflected_resistance",
        policy=validity_policy,
        stacklevel=3,
    )
    omega = 2.0 * np.pi * float(frequency)
    return omega**2 * geometric_loss_integral(
        grid_points, segments, conductivity=conductivity, mask=mask,
        spacing=spacing, charge_correction=charge_correction,
    )


def coil_inductance(
    radii: Sequence[float],
    centers: np.ndarray,
    *,
    wire_radius: float,
    axis: str = "z",
    n_segments: int = 120,
) -> float:
    """Series inductance (H) of coaxial circular turns carrying the same current.

    ``L = sum_j L_self(a_j) + sum_{j!=k} M(loop_j, loop_k)`` -- the sum of every
    entry of the turn inductance matrix (self via
    :func:`self_inductance_circular`, mutual via :func:`mutual_inductance`).
    """

    radii = np.asarray(radii, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    loops = [circular_loop(centers[k], radii[k], axis=axis, n_segments=n_segments)
             for k in range(radii.size)]
    total = 0.0
    for j in range(radii.size):
        total += self_inductance_circular(radii[j], wire_radius)
        for k in range(radii.size):
            if k != j:
                total += mutual_inductance(loops[j], loops[k])
    return float(total)


@dataclass(frozen=True)
class CoilLoading:
    """Frequency-swept sample loading of a coil by a conductive medium."""

    frequency: np.ndarray            # Hz
    reflected_resistance: np.ndarray  # ohm (~ omega^2)
    q_unloaded: np.ndarray            # omega L / R_coil
    q_loaded: np.ndarray              # omega L / (R_coil + R_reflected)
    noise_penalty: np.ndarray         # sqrt(R_total / R_coil) -- SNR / noise-floor degradation
    validity: QuasistaticAssessment | None = None


def coil_loading(
    grid_points: np.ndarray,
    segments: Sequence[Segment],
    *,
    conductivity: float,
    mask: np.ndarray,
    spacing: Sequence[float],
    frequencies: Sequence[float],
    inductance: float,
    coil_resistance: float | np.ndarray,
    charge_correction: bool = False,
    relative_permittivity: float = 1.0,
    relative_permeability: float = 1.0,
    validity_length_m: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> CoilLoading:
    """Sweep the sample-loading effect of a conductive medium across frequency.

    The geometry factor ``G`` is computed once; ``R_reflected(f) = (2 pi f)^2 G``.
    Returns the reflected resistance, the unloaded/loaded quality factors
    ``omega L / R`` and the noise penalty ``sqrt(R_total / R_coil)`` (the factor by
    which the coil's Johnson-noise floor rises, i.e. the SNR degradation when the
    sample loading is not negligible). ``coil_resistance`` may be a scalar or a
    per-frequency array (e.g. a ``sqrt(f)`` skin-effect wire model).
    """

    freqs = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    assessment = _conductive_validity_assessment(
        grid_points,
        segments,
        mask=mask,
        spacing=spacing,
        conductivity=conductivity,
        frequency=float(np.max(freqs)),
        relative_permittivity=relative_permittivity,
        relative_permeability=relative_permeability,
        validity_length_m=validity_length_m,
    )
    apply_validity_policy(
        assessment,
        solver_name="coil_loading",
        policy=validity_policy,
        stacklevel=3,
    )
    g = geometric_loss_integral(
        grid_points, segments, conductivity=conductivity, mask=mask,
        spacing=spacing, charge_correction=charge_correction,
    )
    omega = 2.0 * np.pi * freqs
    r_reflected = omega**2 * g
    r_coil = np.broadcast_to(np.asarray(coil_resistance, dtype=np.float64), freqs.shape)
    r_total = r_coil + r_reflected
    return CoilLoading(
        frequency=freqs,
        reflected_resistance=r_reflected,
        q_unloaded=omega * inductance / r_coil,
        q_loaded=omega * inductance / r_total,
        noise_penalty=np.sqrt(r_total / r_coil),
        validity=assessment,
    )


__all__ = [
    "vector_potential",
    "induced_efield",
    "mutual_inductance",
    "self_inductance_circular",
    "eddy_power",
    "eddy_currents",
    "EddyResult",
    "geometric_loss_integral",
    "reflected_resistance",
    "coil_inductance",
    "coil_loading",
    "CoilLoading",
]
