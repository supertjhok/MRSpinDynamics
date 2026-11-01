"""Regression tests for code-review items L-42 to L-48.

* L-42: tuned inverse multistart crashed (AttributeError) when its first start
  returned a non-finite score.
* L-43: the ILT magnitude-data check cast permitted complex input straight to
  float64, raising numpy's ComplexWarning.
* L-44: ``ThermalNetwork.steady_state`` raised when the last Newton iteration
  converged, and accepted non-positive temperatures after damping.
* L-45: ``ThermalCoupling.march`` rebuilt each network from the coupling's
  sources only, dropping sources the factory adds itself.
* L-46: three hardware-response classes accepted ``oversample < 1``.
* L-47: ``sar_source_from_eddy`` mixed a cycle-averaged density with an
  instantaneous ``EddyResult.power`` when ``time_average=False``.
* L-48: ``ExperimentPredictor`` cached without bound, and the parameter
  constructors aliased ``params`` arrays to ``pp`` arrays.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import spin_dynamics.optimization.drivers as drivers
import spin_dynamics.parameters as parameters_module
from spin_dynamics.analysis.ilt import invert_t1
from spin_dynamics.design.adapters import ExperimentPredictor
from spin_dynamics.fields import coils
from spin_dynamics.fields.coil_properties import ANNEALED_COPPER
from spin_dynamics.fields.quasistatic import eddy_currents
from spin_dynamics.optimal_control.control_response import (
    GradientDriverResponse,
    MatchedProbeResponse,
    TunedProbeResponse,
    UntunedProbeResponse,
)
from spin_dynamics.thermal import (
    CoupledCoilDrive,
    ThermalCoupling,
    ThermalLink,
    ThermalNetwork,
    ThermalNode,
    sar_source_from_eddy,
)


# --------------------------------------------------------------------- L-42


def test_inverse_multistart_survives_non_finite_first_start(monkeypatch) -> None:
    reference = np.array([0.2, 1.1, 2.5])
    calls: list[np.ndarray] = []

    def fake_optimizer(start, _neff, _target_mrx, _target_snr, **_kwargs):
        calls.append(np.asarray(start, dtype=np.float64).copy())
        score = -np.inf if len(calls) == 1 else -float(len(calls))
        return SimpleNamespace(best_score=score, best_phases=np.asarray(start) + 0.1)

    monkeypatch.setattr(
        drivers.excitation_module,
        "optimize_tuned_inverse_excitation_phases",
        fake_optimizer,
    )
    result = drivers.run_tuned_inverse_excitation_multistart(
        reference.size,
        np.zeros((3, 5)),
        np.zeros(5),
        1.0,
        reference,
        num_starts=3,
        seed=0,
        random_fraction=0.0,
    )
    # With random_fraction=0 every later start is the current centre. After
    # the non-finite first start that centre is still the phase-flipped
    # reference (independent reference value), then the best finite result.
    flipped = np.mod(np.pi + reference, 2 * np.pi)
    np.testing.assert_allclose(calls[0], flipped)
    np.testing.assert_allclose(calls[1], flipped)
    np.testing.assert_allclose(calls[2], flipped + 0.1)
    assert result.best_index == 1
    assert result.best_score == -2.0


# --------------------------------------------------------------------- L-43


def test_inversion_recovery_ilt_accepts_complex_input_without_warning() -> None:
    tau = np.linspace(0.01, 2.0, 40)
    t1_axis = np.logspace(-2, 1, 30)
    signal = (1.0 - 2.0 * np.exp(-tau / 0.3)).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        complex_result = invert_t1(
            signal, tau, t1_axis, mode="inversion", regularization=1e-3
        )
    real_result = invert_t1(
        signal.real, tau, t1_axis, mode="inversion", regularization=1e-3
    )
    np.testing.assert_allclose(complex_result.distribution, real_result.distribution)


# --------------------------------------------------------------------- L-44


def _two_node_network(power: float = 2.0, conductance: float = 0.5) -> ThermalNetwork:
    nodes = [
        ThermalNode("coil", heat_capacity=10.0, initial_temperature=293.0),
        ThermalNode("bath", heat_capacity=None, initial_temperature=293.0),
    ]
    links = [ThermalLink("coil", "bath", conductance=conductance)]
    return ThermalNetwork(nodes, links, {"coil": power})


def test_steady_state_accepts_convergence_on_last_iteration() -> None:
    # A linear network converges in one Newton step; before the fix the
    # for/else raised because that step was the last allowed iteration.
    temps = _two_node_network().steady_state(max_iterations=1)
    assert temps["coil"] == pytest.approx(293.0 + 2.0 / 0.5, rel=1e-12)


def test_steady_state_refuses_non_positive_newton_step(monkeypatch) -> None:
    network = _two_node_network()
    # A Jacobian with the wrong sign makes every damped step move the free
    # temperature below zero; the solver must refuse rather than accept it.
    monkeypatch.setattr(network, "_jacobian", lambda temps: np.array([[1e-12, 0.0], [0.0, 1.0]]))
    with pytest.raises(RuntimeError, match="positive"):
        network.steady_state()


# --------------------------------------------------------------------- L-45


def test_march_keeps_factory_sources_and_reaches_fixed_point() -> None:
    bath, g, c, heater = 293.15, 0.05, 2.0, 0.3

    def factory(sources):
        merged = dict(sources)
        merged["coil"] = merged.get("coil", 0.0) + heater  # factory-internal source
        nodes = [
            ThermalNode("coil", heat_capacity=c, initial_temperature=bath),
            ThermalNode("bath", heat_capacity=None, initial_temperature=bath),
        ]
        return ThermalNetwork(nodes, [ThermalLink("coil", "bath", conductance=g)], merged)

    drive = CoupledCoilDrive(
        node="coil", current=1.0, duty_cycle=0.25, material=ANNEALED_COPPER,
        reference_resistance=2.0, reference_temperature=bath, resistance_exponent=1.0,
    )
    coupling = ThermalCoupling(factory, drive)
    steady = coupling.fixed_point().temperatures["coil"]
    tau = c / g
    marched = coupling.march(np.linspace(0.0, 30.0 * tau, 301)).temperatures["coil"]
    # Independent check of the steady rise: it exceeds the heater-only rise.
    assert steady - bath > heater / g
    # Before the fix the march ignored the heater and settled 6 K low.
    assert marched == pytest.approx(steady, abs=1e-3)


# --------------------------------------------------------------------- L-46


@pytest.mark.parametrize(
    "factory",
    [
        lambda n: TunedProbeResponse(tau=1e-6, oversample=n),
        lambda n: UntunedProbeResponse(
            inductance_h=1e-6, resistance_ohm=1.0, carrier_hz=1e7, oversample=n
        ),
        lambda n: MatchedProbeResponse(tau_main=1e-6, tau_extra=2e-7, oversample=n),
        lambda n: GradientDriverResponse(tau_rl=1e-5, oversample=n),
    ],
)
def test_responses_validate_oversample(factory) -> None:
    for bad in (0, -2, 2.5):
        with pytest.raises(ValueError, match="oversample"):
            factory(bad)
    assert factory(3).oversample == 3


# --------------------------------------------------------------------- L-47


def test_sar_source_matches_cycle_average_for_instantaneous_eddy_result() -> None:
    coil = coils.solenoid(radius=0.05, length=0.3, turns=200, axis="z", n_segments=24)
    ax = np.linspace(-0.02, 0.02, 9)
    dx = ax[1] - ax[0]
    grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1)
    mask = np.sqrt(grid[..., 0] ** 2 + grid[..., 1] ** 2) <= 0.015
    common = dict(dcurrent_dt=1e4, conductivity=1e2, mask=mask, spacing=(dx, dx, dx))
    averaged = eddy_currents(grid, coil, time_average=True, **common)
    instantaneous = eddy_currents(grid, coil, time_average=False, **common)
    assert instantaneous.power == pytest.approx(2.0 * averaged.power, rel=1e-12)

    source, density = sar_source_from_eddy(instantaneous, time_average=False)
    # Independent reference: the cycle-averaged eddy power.
    assert source.peak_power == pytest.approx(averaged.power, rel=1e-12)
    assert float(np.sum(density)) * dx**3 == pytest.approx(averaged.power, rel=1e-9)
    # The default (time-averaged result) is unchanged.
    default_source, _ = sar_source_from_eddy(averaged)
    assert default_source.peak_power == pytest.approx(averaged.power, rel=1e-12)


# --------------------------------------------------------------------- L-48


class _CountingAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, parameters, design):
        self.calls += 1
        return np.array([parameters["x"] * design])


def test_predictor_cache_is_bounded_lru() -> None:
    adapter = _CountingAdapter()
    predictor = ExperimentPredictor(adapter, max_cache_entries=2)
    assert ExperimentPredictor(_CountingAdapter()).max_cache_entries == 100_000
    values = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    first = predictor(values, 2.0)
    np.testing.assert_allclose(first[:, 0], values["x"] * 2.0)
    assert len(predictor._cache) == 2
    assert adapter.calls == 4
    # The two most recent particles are cached; the evicted ones re-simulate
    # and give identical predictions.
    again = predictor(values, 2.0)
    np.testing.assert_allclose(again, first)
    assert adapter.calls == 6
    assert len(predictor._cache) == 2
    with pytest.raises(ValueError):
        ExperimentPredictor(adapter, max_cache_entries=0)


def _array_fields(obj):
    return [
        (field.name, getattr(obj, field.name))
        for field in dataclasses.fields(obj)
        if isinstance(getattr(obj, field.name), np.ndarray)
    ]


@pytest.mark.parametrize(
    "name",
    [name for name in parameters_module.__all__ if name.startswith("set_params_")],
)
def test_parameter_constructor_arrays_are_not_shared(name: str) -> None:
    constructed = getattr(parameters_module, name)()
    arrays = [
        (f"{index}.{field}", value)
        for index, obj in enumerate(constructed)
        for field, value in _array_fields(obj)
    ]
    for (label_a, a), (label_b, b) in itertools.combinations(arrays, 2):
        assert not np.shares_memory(a, b), f"{name}: {label_a} aliases {label_b}"
