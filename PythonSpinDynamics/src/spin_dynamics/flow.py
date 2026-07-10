"""Flowing-sample NMR: residence-time kinematics, washout, and inflow.

Many NMR measurements run on a *moving* sample -- inline process monitoring in
a pipe, stopped-flow kinetics, hyphenated LC-NMR. Flow changes the measured
signal through two mechanisms modeled here:

* **Washout / inflow during acquisition.** Excited spins leave the sensitive
  volume and fresh, unexcited spins enter, so the acquired signal decays faster
  than the intrinsic ``T2``. The extra decay is set by the *residence-time
  distribution* (RTD) of the flow, not by relaxation.
* **Transit-time polarization.** Spins entering the coil carry the longitudinal
  magnetization built up during their transit through the upstream field; for a
  velocity *distribution* this is an average over the RTD of the
  single-streamline result in :mod:`spin_dynamics.prepolarization` (Phase B,
  :func:`inflow_polarization`).

Two flow regimes are provided:

* ``"plug"`` -- uniform velocity ``v = Q / A``; every spin has the same
  residence time ``tau = L / v = V / Q``. Washout is linear with a sharp
  cutoff at ``tau``.
* ``"laminar"`` -- fully developed Hagen-Poiseuille profile
  ``v(r) = 2 v_mean (1 - (r/R)^2)``; a distribution of residence times with a
  minimum ``tau/2`` at the centerline and a long ``1/t`` tail from the slow
  near-wall streamlines.

Symbols: ``Q`` volumetric flow rate (m^3/s), ``A = pi R^2`` pipe cross-section,
``R`` pipe radius, ``L`` sensitive-region length, ``V = A L`` sensitive volume,
``v_mean = Q / A`` mean (bulk) velocity, ``tau = V / Q = L / v_mean`` mean
residence time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from spin_dynamics.prepolarization import field_ratio_equilibrium, longitudinal_recovery

__all__ = [
    "FlowModel",
    "washout_fraction",
    "washout_density",
    "apply_washout",
    "flow_enhanced_rate",
    "mean_transit_time",
    "transit_time_distribution",
    "inflow_polarization",
]


@dataclass(frozen=True)
class FlowModel:
    """Pipe flow through a cylindrical sensitive region.

    ``volumetric_flow_rate`` ``Q`` (m^3/s), ``pipe_radius`` ``R`` (m), and
    ``sensitive_length`` ``L`` (m) define the kinematics; ``regime`` is
    ``"plug"`` or ``"laminar"``.
    """

    volumetric_flow_rate: float
    pipe_radius: float
    sensitive_length: float
    regime: str = "plug"

    def __post_init__(self) -> None:
        for label, value in (
            ("volumetric_flow_rate", self.volumetric_flow_rate),
            ("pipe_radius", self.pipe_radius),
            ("sensitive_length", self.sensitive_length),
        ):
            if value <= 0 or not np.isfinite(value):
                raise ValueError(f"{label} must be finite and positive")
        if self.regime not in {"plug", "laminar"}:
            raise ValueError("regime must be 'plug' or 'laminar'")

    @property
    def cross_section(self) -> float:
        """Pipe cross-sectional area ``A = pi R^2`` (m^2)."""

        return float(np.pi * self.pipe_radius**2)

    @property
    def sensitive_volume(self) -> float:
        """Sensitive volume ``V = A L`` (m^3)."""

        return self.cross_section * self.sensitive_length

    @property
    def mean_velocity(self) -> float:
        """Bulk velocity ``v_mean = Q / A`` (m/s)."""

        return self.volumetric_flow_rate / self.cross_section

    @property
    def mean_residence_time(self) -> float:
        """Volumetric mean residence time ``tau = V / Q`` (s).

        For the laminar regime the *RTD* mean diverges (near-wall streamlines
        are arbitrarily slow); this volumetric ``tau`` is still the physical
        fill time and the natural time scale, so it is what the washout and
        distribution functions are parameterized by.
        """

        return self.sensitive_volume / self.volumetric_flow_rate

    def centerline_velocity(self) -> float:
        """Peak velocity: ``2 v_mean`` (laminar) or ``v_mean`` (plug)."""

        return self.mean_velocity * (2.0 if self.regime == "laminar" else 1.0)


def washout_fraction(flow: FlowModel, time: np.ndarray) -> np.ndarray:
    """Fraction ``W(t)`` of originally-excited spins still in the volume.

    Uniformly excited spins are assumed (the sensitive volume is fully
    labelled at ``t = 0``). Averaging the remaining residence time over both
    axial position and the velocity profile:

    * plug -- ``W(t) = 1 - t/tau`` (linear, sharp cutoff at ``tau``);
    * laminar -- ``W(t) = 1 - t/tau`` for ``t <= tau/2`` and
      ``W(t) = tau/(4 t)`` beyond, i.e. the same initial slope as plug flow
      but a slow ``1/t`` tail from the near-wall streamlines.

    Both regimes share the initial slope ``-1/tau`` (hence the shared
    early-time apparent rate in :func:`flow_enhanced_rate`); they diverge only
    past ``t = tau/2``.
    """

    t = np.asarray(time, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    tau = flow.mean_residence_time
    if flow.regime == "plug":
        return np.clip(1.0 - t / tau, 0.0, 1.0)
    # laminar: 1 - t/tau up to tau/2, then the tau/(4t) tail
    safe_t = np.maximum(t, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        w = np.where(t <= tau / 2.0, 1.0 - t / tau, tau / (4.0 * safe_t))
    return np.clip(w, 0.0, 1.0)


def washout_density(flow: FlowModel, time: np.ndarray) -> np.ndarray:
    """Remaining-residence density ``E(t) = -dW/dt`` (1/s), normalized to 1.

    The distribution of *remaining* residence times of uniformly-excited spins
    -- the quantity that governs washout during acquisition, not the classic
    inlet-tracer (exit-age) RTD. Plug: uniform ``1/tau`` on ``[0, tau]``.
    Laminar: ``1/tau`` for ``t <= tau/2`` and ``tau/(4 t^2)`` beyond; both
    integrate to 1.
    """

    t = np.asarray(time, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    tau = flow.mean_residence_time
    if flow.regime == "plug":
        return np.where(t <= tau, 1.0 / tau, 0.0)
    safe_t = np.maximum(t, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t <= tau / 2.0, 1.0 / tau, tau / (4.0 * safe_t**2))


def apply_washout(
    flow: FlowModel, time: np.ndarray, signal: np.ndarray
) -> np.ndarray:
    """Multiply an acquired signal by the flow washout factor ``W(t)``.

    ``signal`` is the stationary-sample response (e.g. a CPMG echo train or an
    FID envelope) sampled at ``time``; the returned array is the flowing-sample
    signal, with excited spins progressively replaced by fresh inflowing ones.
    """

    t = np.asarray(time, dtype=np.float64)
    s = np.asarray(signal)
    if s.shape[-1] != t.shape[-1]:
        raise ValueError("signal last axis must match time")
    return s * washout_fraction(flow, t)


def flow_enhanced_rate(flow: FlowModel, t2_seconds: float) -> float:
    """Effective early-time decay rate ``1/T2 + 1/tau``.

    The washout has initial slope ``-1/tau`` in *both* regimes, so this is the
    small-time apparent rate for plug and laminar flow alike. It is only the
    initial rate: plug washout is linear (not exponential) and the laminar
    signal develops a slow tail, so evaluate the full curve through
    :func:`apply_washout` when the acquisition runs an appreciable fraction of
    ``tau``.
    """

    if t2_seconds <= 0 or not np.isfinite(t2_seconds):
        raise ValueError("t2_seconds must be finite and positive")
    return 1.0 / t2_seconds + 1.0 / flow.mean_residence_time


# --- Phase B: transit-time inflow polarization --------------------------------


def mean_transit_time(flow: FlowModel, length_meters: float) -> float:
    """Mean transit time ``L / v_mean`` through an upstream region (s)."""

    if length_meters <= 0 or not np.isfinite(length_meters):
        raise ValueError("length_meters must be finite and positive")
    return float(length_meters) / flow.mean_velocity


def transit_time_distribution(
    flow: FlowModel, length_meters: float, time: np.ndarray
) -> np.ndarray:
    """Flux-weighted exit-age RTD ``E(t)`` for transit through a length (1/s).

    The distribution of transit times of the fluid *entering* the detector,
    weighted by volumetric flux (what sets the average inflowing
    magnetization). Plug: uniform density ``1/tau`` on ``[0, tau]`` standing in
    for the Dirac at the single transit time ``tau = L / v_mean``. Laminar:
    the classic ``E(t) = tau^2 / (2 t^3)`` for ``t >= tau/2`` (mean ``tau``).
    Use :func:`inflow_polarization` for averaging; this is for inspection.
    """

    t = np.asarray(time, dtype=np.float64)
    if np.any(t < 0):
        raise ValueError("time must be non-negative")
    tau = mean_transit_time(flow, length_meters)
    if flow.regime == "plug":
        return np.where(t <= tau, 1.0 / tau, 0.0)
    safe_t = np.maximum(t, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t >= tau / 2.0, tau**2 / (2.0 * safe_t**3), 0.0)


def inflow_polarization(
    flow: FlowModel,
    *,
    polarizing_field_tesla: float,
    detection_field_tesla: float,
    prepolarizer_length_meters: float,
    t1_seconds: float,
    initial_magnetization: float = 0.0,
    detection_equilibrium_magnetization: float = 1.0,
    quadrature_points: int = 2000,
) -> float:
    """Flux-averaged longitudinal magnetization of spins entering the detector.

    Each streamline spends its transit time in the polarizing field building up
    ``Mz`` by T1 recovery; the detector sees the flux-weighted average. In the
    detection-field units of :mod:`spin_dynamics.prepolarization`, ``1.0`` is
    thermal equilibrium in the detection field and ``B_pol/B_det`` is full
    equilibrium in the polarizer.

    Plug flow reduces to the single-transit
    :func:`spin_dynamics.prepolarization.prepolarized_flow_state` value.
    Laminar flow integrates over the parabolic profile using the flux weight
    ``2(1-u)`` with ``u = (r/R)^2`` and transit time ``tau/(2(1-u))``, a
    bounded quadrature that avoids the singular ``1/t^3`` RTD tail.
    """

    if t1_seconds <= 0 or not np.isfinite(t1_seconds):
        raise ValueError("t1_seconds must be finite and positive")
    equilibrium = float(
        field_ratio_equilibrium(
            polarizing_field_tesla,
            detection_field_tesla,
            detection_equilibrium_magnetization=detection_equilibrium_magnetization,
        )
    )
    tau_pre = mean_transit_time(flow, prepolarizer_length_meters)
    if flow.regime == "plug":
        return float(
            longitudinal_recovery(initial_magnetization, equilibrium, tau_pre, t1_seconds)
        )
    if quadrature_points < 2:
        raise ValueError("quadrature_points must be >= 2")
    # Flux-weighted average over u = (r/R)^2 in (0, 1); weight 2(1-u) integrates
    # to 1. Avoid the u=1 (wall, infinite transit) endpoint with a midpoint rule.
    edges = np.linspace(0.0, 1.0, quadrature_points + 1)
    u = 0.5 * (edges[:-1] + edges[1:])
    du = edges[1] - edges[0]
    transit = tau_pre / (2.0 * (1.0 - u))
    mz = longitudinal_recovery(initial_magnetization, equilibrium, transit, t1_seconds)
    weight = 2.0 * (1.0 - u)
    return float(np.sum(weight * mz) * du)
