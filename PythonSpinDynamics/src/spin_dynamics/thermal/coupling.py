"""Quasi-static electro-thermal coupling loop (Phase 2 of thermal_modeling.md).

Closes the feedback between the thermal network and the electromagnetic
layers: coil resistance rises with coil temperature (raising the Joule source
and the Johnson noise), sample conductivity and magnetization change with
sample temperature, and the resulting powers re-enter the thermal solve. The
time-scale separation (RF microseconds versus thermal seconds-to-minutes)
makes the weak-coupling fixed point / macro-step march exact for practical
purposes -- see docs/thermal_modeling.md.

Conventions: the coil Joule source scales as ``R(T) = R_ref * (rho(T) /
rho(T_ref))**resistance_exponent`` with exponent 1 for DC/audio (gradient)
coils and 0.5 for skin-depth-limited RF coils. Sample SAR optionally scales
linearly with sample temperature via ``sar_tempco`` (biological tissue:
conductivity roughly +2%/K).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from spin_dynamics.fields.coil_properties import ConductorMaterial
from spin_dynamics.sample import Sample
from spin_dynamics.thermal.network import ThermalNetwork, ThermalTransientResult
from spin_dynamics.thermal.sources import coil_joule_power

__all__ = [
    "CoupledCoilDrive",
    "CoupledSAR",
    "ThermalCouplingResult",
    "ThermalCoupling",
    "resistance_at_temperature",
]


def resistance_at_temperature(
    material: ConductorMaterial,
    reference_resistance: float,
    reference_temperature: float,
    temperature: float,
    *,
    exponent: float = 1.0,
) -> float:
    """Scale a measured coil resistance to another temperature.

    ``R(T) = R_ref * (rho(T)/rho(T_ref))**exponent`` -- exponent 1 for a DC or
    audio-frequency (gradient) coil, 0.5 for a skin-depth-limited RF coil
    (``R_ac ~ sqrt(rho)`` when the skin depth is small against the wire).
    """

    if reference_resistance <= 0 or not np.isfinite(reference_resistance):
        raise ValueError("reference_resistance must be finite and positive")
    if not (0.0 < exponent <= 1.0):
        raise ValueError("exponent must be in (0, 1]")
    ratio = material.resistivity_at(temperature) / material.resistivity_at(
        reference_temperature
    )
    return float(reference_resistance * ratio**exponent)


@dataclass(frozen=True)
class CoupledCoilDrive:
    """Temperature-dependent coil Joule source.

    ``current`` is the sinusoidal drive amplitude (A) active for
    ``duty_cycle`` of the time; the dissipation uses
    :func:`resistance_at_temperature` at the coil node's temperature.
    """

    node: str
    current: float
    duty_cycle: float
    material: ConductorMaterial
    reference_resistance: float
    reference_temperature: float
    resistance_exponent: float = 0.5

    def resistance(self, temperature: float) -> float:
        return resistance_at_temperature(
            self.material,
            self.reference_resistance,
            self.reference_temperature,
            temperature,
            exponent=self.resistance_exponent,
        )

    def power(self, temperature: float) -> float:
        return coil_joule_power(self.current, self.resistance(temperature)) * self.duty_cycle


@dataclass(frozen=True)
class CoupledSAR:
    """Temperature-dependent sample deposition.

    ``reference_power`` is the duty-cycle-averaged SAR at
    ``reference_temperature`` (from ``sar_source_from_eddy`` or
    ``sar_power_from_loading``); ``tempco`` is the linear fractional change of
    sample conductivity (hence deposition) per kelvin.
    """

    node: str
    reference_power: float
    reference_temperature: float
    tempco: float = 0.0

    def power(self, temperature: float) -> float:
        if self.reference_power < 0 or not np.isfinite(self.reference_power):
            raise ValueError("reference_power must be finite and non-negative")
        scale = 1.0 + self.tempco * (temperature - self.reference_temperature)
        return self.reference_power * max(scale, 0.0)


@dataclass(frozen=True)
class ThermalCouplingResult:
    """Converged (or final) state of the coupling loop."""

    temperatures: dict[str, float]
    coil_resistance: float
    coil_power: float
    sample_power: float
    sample: Sample | None
    iterations: int
    transient: ThermalTransientResult | None = None

    def probe_updates(self, *, coil_node: str, inductance: float | None = None,
                      omega0: float | None = None) -> dict[str, float]:
        """Updates for an ``sp`` mapping: coil temperature, R, and Q.

        ``Q = omega0 * L / R`` is included when ``inductance`` and ``omega0``
        are given.
        """

        updates = {"T": self.temperatures[coil_node], "R": self.coil_resistance}
        if inductance is not None and omega0 is not None:
            updates["Q"] = omega0 * inductance / self.coil_resistance
        return updates


class ThermalCoupling:
    """Fixed-point and macro-step coupling of sources, network, and consumers.

    ``network_factory(sources)`` builds a :class:`ThermalNetwork` from a
    ``{node: watts}`` mapping so each iteration re-solves with updated
    temperature-dependent powers. ``sample_node`` links the thermal solution
    back to a :class:`spin_dynamics.sample.Sample` via ``with_temperature``.
    """

    def __init__(
        self,
        network_factory,
        drive: CoupledCoilDrive,
        *,
        sar: CoupledSAR | None = None,
        sample: Sample | None = None,
        sample_node: str | None = None,
        extra_sources: dict[str, float] | None = None,
    ) -> None:
        self._factory = network_factory
        self.drive = drive
        self.sar = sar
        self.sample = sample
        self.sample_node = sample_node
        self.extra = dict(extra_sources or {})
        if sample is not None and sample_node is None:
            raise ValueError("sample_node is required when a sample is supplied")

    def _sources(self, temps: dict[str, float]) -> tuple[dict[str, float], float, float]:
        t_coil = temps[self.drive.node]
        coil_power = self.drive.power(t_coil)
        sources = dict(self.extra)
        sources[self.drive.node] = sources.get(self.drive.node, 0.0) + coil_power
        sample_power = 0.0
        if self.sar is not None:
            t_dep = temps[self.sar.node]
            sample_power = self.sar.power(t_dep)
            sources[self.sar.node] = sources.get(self.sar.node, 0.0) + sample_power
        return sources, coil_power, sample_power

    def _initial_temperatures(self) -> dict[str, float]:
        network = self._factory(dict(self.extra))
        return {n.name: n.initial_temperature for n in network.nodes}

    def _result(
        self,
        temps: dict[str, float],
        coil_power: float,
        sample_power: float,
        iterations: int,
        transient: ThermalTransientResult | None = None,
    ) -> ThermalCouplingResult:
        sample = self.sample
        if sample is not None and self.sample_node is not None:
            sample = sample.with_temperature(temps[self.sample_node])
        return ThermalCouplingResult(
            temperatures=dict(temps),
            coil_resistance=self.drive.resistance(temps[self.drive.node]),
            coil_power=coil_power,
            sample_power=sample_power,
            sample=sample,
            iterations=iterations,
            transient=transient,
        )

    def fixed_point(
        self, *, tol: float = 1e-8, max_iterations: int = 200
    ) -> ThermalCouplingResult:
        """Converged steady state of the two-way coupled problem.

        Raises ``RuntimeError`` when the loop does not converge -- including
        the physical thermal-runaway case where the resistance tempco gain
        exceeds the network's ability to shed heat.
        """

        temps = self._initial_temperatures()
        coil_power = sample_power = 0.0
        for iteration in range(1, max_iterations + 1):
            sources, coil_power, sample_power = self._sources(temps)
            steady = self._factory(sources).steady_state()
            delta = max(abs(steady[k] - temps[k]) for k in steady)
            temps = steady
            if delta < tol:
                return self._result(temps, coil_power, sample_power, iteration)
        raise RuntimeError(
            "thermal coupling fixed point did not converge "
            "(possible thermal runaway: tempco feedback exceeds cooling)"
        )

    def march(
        self,
        times: np.ndarray,
        *,
        update_every: int = 1,
    ) -> ThermalCouplingResult:
        """Time-march with sources refreshed every ``update_every`` intervals.

        Valid when the macro step is short against the thermal time constants
        over which R(T) changes appreciably (weak coupling); the fixed point
        of :meth:`fixed_point` is the long-time limit.
        """

        times = np.asarray(times, dtype=np.float64).reshape(-1)
        if times.size < 2 or not np.all(np.diff(times) > 0):
            raise ValueError("times must be strictly increasing with >= 2 samples")
        if update_every < 1:
            raise ValueError("update_every must be >= 1")
        temps = self._initial_temperatures()
        names = list(temps)
        trajectory = {name: [temps[name]] for name in names}
        coil_power = sample_power = 0.0
        idx = 0
        while idx < times.size - 1:
            stop = min(idx + update_every, times.size - 1)
            sources, coil_power, sample_power = self._sources(temps)
            network = self._factory(sources)
            # Restart the network from the marched temperatures.
            network = ThermalNetwork(
                [replace(n, initial_temperature=temps[n.name]) for n in network.nodes],
                network.links,
                sources,
            )
            segment = network.transient(times[idx : stop + 1])
            for name in names:
                trajectory[name].extend(segment.temperatures[name][1:].tolist())
            temps = segment.final()
            idx = stop
        transient = ThermalTransientResult(
            times=times,
            temperatures={k: np.asarray(v) for k, v in trajectory.items()},
        )
        return self._result(temps, coil_power, sample_power, times.size, transient)
