"""Lumped thermal network solver (Option A of docs/thermal_modeling.md).

Nodes carry heat capacities ``C = rho c_p V`` (or are fixed-temperature
baths); links carry linear conductances ``G`` (conduction, convection) and
optionally a radiation coefficient ``eps sigma_SB A`` for the nonlinear
``T^4`` exchange. The network solves

``C_i dT_i/dt = sum_links q_in,i + P_i``

either to steady state (linear solve, or damped Newton when radiation links
are present) or transiently (stiff ODE; scipy's LSODA when available, a
step-limited RK4 fallback otherwise).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)

__all__ = [
    "STEFAN_BOLTZMANN",
    "ThermalNode",
    "ThermalLink",
    "ThermalNetwork",
    "ThermalTransientResult",
    "conduction_conductance",
    "cylindrical_shell_conductance",
    "convection_conductance",
    "flow_conductance",
    "radiation_link",
]


@dataclass(frozen=True)
class ThermalNode:
    """A lumped node. ``heat_capacity=None`` marks a fixed-temperature bath."""

    name: str
    heat_capacity: float | None
    initial_temperature: float = 293.15

    def __post_init__(self) -> None:
        if self.heat_capacity is not None and (
            self.heat_capacity <= 0 or not np.isfinite(self.heat_capacity)
        ):
            raise ValueError("heat_capacity must be positive (or None for a bath)")
        if self.initial_temperature <= 0 or not np.isfinite(self.initial_temperature):
            raise ValueError("initial_temperature must be finite and positive (kelvin)")

    @property
    def is_bath(self) -> bool:
        return self.heat_capacity is None


@dataclass(frozen=True)
class ThermalLink:
    """Heat exchange between two nodes.

    ``conductance`` is the linear part in W/K (conduction and/or convection);
    ``radiation_coefficient`` is ``eps * sigma_SB * A`` in W/K^4 for the
    nonlinear exchange ``q = coeff * (T_a^4 - T_b^4)`` (small-cavity/two-node
    grey-body idealization).
    """

    node_a: str
    node_b: str
    conductance: float = 0.0
    radiation_coefficient: float = 0.0

    def __post_init__(self) -> None:
        for label, value in (
            ("conductance", self.conductance),
            ("radiation_coefficient", self.radiation_coefficient),
        ):
            if value < 0 or not np.isfinite(value):
                raise ValueError(f"{label} must be finite and non-negative")
        if self.conductance == 0.0 and self.radiation_coefficient == 0.0:
            raise ValueError("link must have a nonzero conductance or radiation coefficient")
        if self.node_a == self.node_b:
            raise ValueError("link endpoints must differ")


def conduction_conductance(conductivity: float, area: float, length: float) -> float:
    """Slab conduction ``G = k A / L`` (W/K)."""

    for label, value in (("conductivity", conductivity), ("area", area), ("length", length)):
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")
    return conductivity * area / length


def cylindrical_shell_conductance(
    conductivity: float, length: float, r_inner: float, r_outer: float
) -> float:
    """Radial conduction through a cylindrical shell, ``2 pi k L / ln(r2/r1)``."""

    for label, value in (
        ("conductivity", conductivity),
        ("length", length),
        ("r_inner", r_inner),
        ("r_outer", r_outer),
    ):
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")
    if r_outer <= r_inner:
        raise ValueError("r_outer must exceed r_inner")
    return 2.0 * np.pi * conductivity * length / np.log(r_outer / r_inner)


def convection_conductance(film_coefficient: float, area: float) -> float:
    """Convection film ``G = h A`` (W/K)."""

    for label, value in (("film_coefficient", film_coefficient), ("area", area)):
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")
    return film_coefficient * area


def flow_conductance(
    density: float, specific_heat: float, volumetric_flow_rate: float
) -> float:
    """Advective heat-removal conductance ``G = rho c_p Q`` (W/K).

    A flowing sample carries heat out of the probe: modeled as a link from the
    sample node to a bath held at the inlet temperature with this conductance,
    so the removed power is ``G (T_sample - T_in)``. Formally identical to the
    Pennes perfusion sink of :mod:`spin_dynamics.thermal.conduction` -- blood
    perfusion is itself a volumetric flow term.
    """

    for label, value in (
        ("density", density),
        ("specific_heat", specific_heat),
        ("volumetric_flow_rate", volumetric_flow_rate),
    ):
        if value <= 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and positive")
    return density * specific_heat * volumetric_flow_rate


def radiation_link(
    node_a: str, node_b: str, *, emissivity: float, area: float
) -> ThermalLink:
    """Grey-body radiation link with coefficient ``eps sigma_SB A``."""

    if not (0.0 < emissivity <= 1.0):
        raise ValueError("emissivity must be in (0, 1]")
    if area <= 0 or not np.isfinite(area):
        raise ValueError("area must be finite and positive")
    return ThermalLink(
        node_a=node_a,
        node_b=node_b,
        radiation_coefficient=emissivity * STEFAN_BOLTZMANN * area,
    )


@dataclass(frozen=True)
class ThermalTransientResult:
    """Transient temperatures: ``temperatures[name]`` is T(t) over ``times``."""

    times: np.ndarray
    temperatures: dict[str, np.ndarray]

    def final(self) -> dict[str, float]:
        return {name: float(t[-1]) for name, t in self.temperatures.items()}


def _source_power(value: Any) -> float:
    if hasattr(value, "average_power"):
        return float(value.average_power)
    return float(value)


class ThermalNetwork:
    """Nodal RC thermal network with steady-state and transient solves.

    ``sources`` maps node names to a power in watts, an object with an
    ``average_power`` attribute (:class:`ConstantSource`,
    :class:`DutyCycledSource`), or a sequence of either.
    """

    def __init__(
        self,
        nodes: Sequence[ThermalNode],
        links: Sequence[ThermalLink],
        sources: Mapping[str, Any] | None = None,
    ) -> None:
        names = [n.name for n in nodes]
        if len(set(names)) != len(names):
            raise ValueError("node names must be unique")
        self.nodes = list(nodes)
        self._index = {n.name: i for i, n in enumerate(self.nodes)}
        for link in links:
            for end in (link.node_a, link.node_b):
                if end not in self._index:
                    raise ValueError(f"link references unknown node '{end}'")
        self.links = list(links)
        power = np.zeros(len(self.nodes))
        for name, value in (sources or {}).items():
            if name not in self._index:
                raise ValueError(f"source references unknown node '{name}'")
            if isinstance(value, (list, tuple)):
                power[self._index[name]] += sum(_source_power(v) for v in value)
            else:
                power[self._index[name]] += _source_power(value)
        if np.any(power < 0):
            raise ValueError("source powers must be non-negative")
        self._power = power
        self._free = np.array([not n.is_bath for n in self.nodes], dtype=bool)
        if not np.any(self._free):
            raise ValueError("network needs at least one non-bath node")

    # ------------------------------------------------------------------
    # Flux assembly
    # ------------------------------------------------------------------

    def _net_flux(self, temps: np.ndarray) -> np.ndarray:
        """Net heat inflow per node (W) at the given temperatures."""

        q = self._power.copy()
        for link in self.links:
            ia, ib = self._index[link.node_a], self._index[link.node_b]
            flow = link.conductance * (temps[ia] - temps[ib])
            if link.radiation_coefficient > 0.0:
                flow += link.radiation_coefficient * (temps[ia] ** 4 - temps[ib] ** 4)
            q[ia] -= flow
            q[ib] += flow
        return q

    def _jacobian(self, temps: np.ndarray) -> np.ndarray:
        n = len(self.nodes)
        jac = np.zeros((n, n))
        for link in self.links:
            ia, ib = self._index[link.node_a], self._index[link.node_b]
            g = link.conductance
            if link.radiation_coefficient > 0.0:
                # d/dT of coeff*(Ta^4 - Tb^4): 4 coeff T^3 at each end.
                ga = g + 4.0 * link.radiation_coefficient * temps[ia] ** 3
                gb = g + 4.0 * link.radiation_coefficient * temps[ib] ** 3
            else:
                ga = gb = g
            jac[ia, ia] -= ga
            jac[ia, ib] += gb
            jac[ib, ia] += ga
            jac[ib, ib] -= gb
        return jac

    def _initial(self) -> np.ndarray:
        return np.array([n.initial_temperature for n in self.nodes], dtype=np.float64)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def steady_state(
        self, *, tol: float = 1e-10, max_iterations: int = 100
    ) -> dict[str, float]:
        """Steady temperatures: linear solve, or damped Newton with radiation."""

        temps = self._initial()
        free = self._free
        for _ in range(max_iterations):
            residual = self._net_flux(temps)[free]
            if np.max(np.abs(residual)) < tol * max(1.0, float(np.max(self._power)), 1e-12):
                break
            jac = self._jacobian(temps)[np.ix_(free, free)]
            step = np.linalg.solve(jac, -residual)
            # Damp so temperatures stay positive (radiation needs T > 0).
            scale = 1.0
            new = temps[free] + step
            while np.any(new <= 0.0) and scale > 1e-6:
                scale *= 0.5
                new = temps[free] + scale * step
            temps[free] = new
        else:
            raise RuntimeError("steady-state Newton iteration did not converge")
        return {n.name: float(temps[i]) for i, n in enumerate(self.nodes)}

    def transient(
        self,
        times: np.ndarray,
        *,
        max_step: float | None = None,
    ) -> ThermalTransientResult:
        """Integrate the network ODE over ``times`` (strictly increasing, s)."""

        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if times.size < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing with >= 2 samples")
        caps = np.array(
            [n.heat_capacity if not n.is_bath else np.inf for n in self.nodes],
            dtype=np.float64,
        )
        free = self._free

        def rhs_free(t: float, tf: np.ndarray) -> np.ndarray:
            temps = self._initial()
            temps[free] = tf
            return self._net_flux(temps)[free] / caps[free]

        try:
            from scipy.integrate import solve_ivp
        except ImportError:
            trajectory = self._transient_rk4(times, caps, max_step)
        else:
            sol = solve_ivp(
                rhs_free,
                (float(times[0]), float(times[-1])),
                self._initial()[free],
                method="LSODA",
                t_eval=times,
                max_step=np.inf if max_step is None else float(max_step),
                rtol=1e-8,
                atol=1e-8,
            )
            if not sol.success:
                raise RuntimeError(f"transient integration failed: {sol.message}")
            trajectory = np.tile(self._initial(), (times.size, 1))
            trajectory[:, free] = sol.y.T
        temperatures = {
            n.name: trajectory[:, i].copy() for i, n in enumerate(self.nodes)
        }
        return ThermalTransientResult(times=times, temperatures=temperatures)

    def _transient_rk4(
        self, times: np.ndarray, caps: np.ndarray, max_step: float | None
    ) -> np.ndarray:
        free = self._free
        # Resolve the fastest RC time constant.
        g_tot = np.zeros(len(self.nodes))
        for link in self.links:
            ia, ib = self._index[link.node_a], self._index[link.node_b]
            g_lin = link.conductance + 4.0 * link.radiation_coefficient * 300.0**3
            g_tot[ia] += g_lin
            g_tot[ib] += g_lin
        with np.errstate(divide="ignore", invalid="ignore"):
            tau = caps[free] / g_tot[free]
        tau = tau[np.isfinite(tau) & (tau > 0)]
        limit = float(np.min(tau)) / 10.0 if tau.size else np.inf
        if max_step is not None:
            limit = min(limit, float(max_step))
        temps = self._initial()
        out = np.zeros((times.size, len(self.nodes)))
        out[0] = temps

        def deriv(t_vec: np.ndarray) -> np.ndarray:
            d = np.zeros_like(t_vec)
            d[free] = self._net_flux(t_vec)[free] / caps[free]
            return d

        for idx in range(times.size - 1):
            span = float(times[idx + 1] - times[idx])
            nsub = max(1, int(np.ceil(span / limit))) if np.isfinite(limit) else 1
            h = span / nsub
            for _ in range(nsub):
                k1 = deriv(temps)
                k2 = deriv(temps + h * k1 / 2.0)
                k3 = deriv(temps + h * k2 / 2.0)
                k4 = deriv(temps + h * k3)
                temps = temps + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            out[idx + 1] = temps
        return out
