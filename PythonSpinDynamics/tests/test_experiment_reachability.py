"""Every workflow option is reachable from an experiment spec or declared.

A workflow keyword argument that ``build_kwargs`` never produces silently
falls back to its default for facade users (``CPMGIRTrain`` once could not
choose ``signal_convention``). The registry entry must therefore either wire
the option, list it in ``execution_kwargs``, or declare it ``direct_only``.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
from typing import Any, Callable

import numpy as np
import pytest

from spin_dynamics.experiment import (
    Acquisition,
    CPMGImaging,
    CPMGIRTrain,
    Experiment,
    Hardware,
    NQRSORC,
    Phantom,
    Sample,
    available_workflows,
)
from spin_dynamics.nqr import QuadrupolarSite, simulate_full_slse
from spin_dynamics.workflows import (
    run_ideal_cpmg_ir_train,
    run_ideal_receiver_array_cpmg_imaging,
)

# Functions a ``resolve_func`` can dispatch to besides the nominal ``func``.
_ALTERNATIVES: dict[str, tuple[Callable[..., Any], ...]] = {
    "run_ideal_cpmg_imaging": (run_ideal_receiver_array_cpmg_imaging,),
    "run_tuned_cpmg_imaging": (),
    "run_matched_cpmg_imaging": (),
    "simulate_slse": (simulate_full_slse,),
}


def _builder_keys(func: Callable[..., Any], seen: set[Any] | None = None) -> set[str]:
    """Collect the keyword names a kwargs builder can emit, from its source.

    Builders use dict literals, ``kwargs["name"] = ...`` and
    ``kwargs.update(name=...)``; helpers they call from the same module are
    followed. Over-collection only makes the check more permissive.
    """

    seen = set() if seen is None else seen
    func = inspect.unwrap(func)
    if func in seen:
        return set()
    seen.add(func)
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    module = sys.modules[func.__module__]
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys.update(
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            keys.add(node.slice.value)
        elif isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Attribute) and target.attr == "update":
                keys.update(kw.arg for kw in node.keywords if kw.arg)
            elif isinstance(target, ast.Name):
                helper = getattr(module, target.id, None)
                if inspect.isfunction(helper) and helper.__module__ == func.__module__:
                    keys |= _builder_keys(helper, seen)
    return keys


def _keyword_parameters(func: Callable[..., Any]) -> set[str] | None:
    """Named keyword parameters, or ``None`` for a ``**kwargs`` passthrough."""

    parameters = inspect.signature(func).parameters.values()
    if any(p.kind is p.VAR_KEYWORD for p in parameters):
        return None
    return {
        p.name
        for p in parameters
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }


def _candidates(entry: Any) -> tuple[Callable[..., Any], ...]:
    if entry.resolve_func is None:
        return (entry.func,)
    assert entry.name in _ALTERNATIVES, (
        f"{entry.name} dispatches through resolve_func; list the functions it "
        "can return in _ALTERNATIVES"
    )
    return (entry.func, *_ALTERNATIVES[entry.name])


_ENTRIES = available_workflows()


@pytest.mark.parametrize(
    "entry", _ENTRIES, ids=[f"{e.name}-{e.probe}" for e in _ENTRIES]
)
def test_every_workflow_option_is_reachable_or_declared(entry: Any) -> None:
    keys = _builder_keys(entry.build_kwargs)
    declared = keys | set(entry.execution_kwargs) | set(entry.direct_only)
    for func in _candidates(entry):
        parameters = _keyword_parameters(func)
        if parameters is None:
            continue
        unreachable = parameters - declared
        assert not unreachable, (
            f"{func.__module__}.{func.__name__} options {sorted(unreachable)} "
            f"cannot be set through {entry.sequence_type.__name__}; wire them "
            "in build_kwargs or declare them in direct_only"
        )


@pytest.mark.parametrize(
    "entry", _ENTRIES, ids=[f"{e.name}-{e.probe}" for e in _ENTRIES]
)
def test_direct_only_names_real_parameters(entry: Any) -> None:
    accepted: set[str] = set()
    for func in _candidates(entry):
        parameters = _keyword_parameters(func)
        if parameters is not None:
            accepted |= parameters
    stale = set(entry.direct_only) - accepted
    assert not stale, f"{entry.name} declares unknown direct_only options {sorted(stale)}"


def test_alternatives_table_matches_registry() -> None:
    dispatching = {e.name for e in _ENTRIES if e.resolve_func is not None}
    assert set(_ALTERNATIVES) == dispatching


def test_builder_scan_sees_conditional_keys() -> None:
    entry = next(
        e for e in _ENTRIES if e.sequence_type is CPMGIRTrain and e.probe == "ideal"
    )
    assert {"tauvect", "signal_convention", "t1_seconds"} <= _builder_keys(
        entry.build_kwargs
    )


def test_cpmg_ir_signal_convention_reaches_workflow() -> None:
    sequence = dict(num_echoes=2, echo_spacing_seconds=1e-3, tauvect=(5e-3, 0.2))
    acquisition = Acquisition(numpts=15, maxoffs=5.0, rephase_action="ignore")
    cpmg = Experiment(CPMGIRTrain(**sequence), acquisition=acquisition).run().result
    matlab = Experiment(
        CPMGIRTrain(**sequence, signal_convention="matlab"), acquisition=acquisition
    ).run().result
    direct = run_ideal_cpmg_ir_train(
        num_echoes=2,
        echo_spacing_seconds=1e-3,
        tauvect=(5e-3, 0.2),
        numpts=15,
        maxoffs=5.0,
        rephase_action="ignore",
        signal_convention="matlab",
    )
    np.testing.assert_allclose(matlab.echo_integrals, direct.echo_integrals)
    np.testing.assert_allclose(
        matlab.echo_integrals, -2.0 * cpmg.echo_integrals, rtol=1e-12, atol=1e-15
    )


def test_imaging_conventions_are_forwarded() -> None:
    sample = Sample(phantom=Phantom(rho=np.ones((3, 3))), t1_seconds=5e-3, t2_seconds=5e-3)
    base = dict(num_echoes=1, ny=3)
    default = Experiment(CPMGImaging(**base), sample).run().result
    cpmg = Experiment(CPMGImaging(**base, signal_convention="cpmg"), sample).run().result
    np.testing.assert_allclose(cpmg.kspace, 0.5 * default.kspace, rtol=1e-12)


def test_receive_mode_is_honored_only_by_tuned_imaging() -> None:
    sample = Sample(phantom=Phantom(rho=np.ones((3, 3))), t1_seconds=5e-3, t2_seconds=5e-3)
    sequence = CPMGImaging(num_echoes=1, ny=3)
    tuned = Experiment(sequence, sample, Hardware(probe="tuned", receive_mode="weighted"))
    assert not [w for w in tuned.plan().warnings if "receive_mode" in w]
    ideal = Experiment(sequence, sample, Hardware(receive_mode="weighted"))
    assert any("hardware.receive_mode" in w for w in ideal.plan().warnings)


def test_sorc_model_is_forwarded() -> None:
    site = QuadrupolarSite(spin=1, quadrupole_frequency_hz=900e3, eta=0.3)
    sequence = dict(
        pulse_duration_seconds=100e-6,
        nutation_hz=2.5e3,
        half_spacing_seconds=0.5e-3,
        num_pulses=6,
        orientations="single",
        # SORC is an off-resonance train; the "x" line is at 990 kHz.
        rf_frequency_hz=995e3,
    )
    pathway = Experiment(NQRSORC(**sequence), Sample(site=site)).run().result
    steady = Experiment(
        NQRSORC(**sequence, model="steady_state"), Sample(site=site)
    ).run().result
    # The finite pathway train approaches the steady-state limit from below.
    assert np.any(np.abs(pathway.signal_amplitudes) > 0.0)
    assert not np.allclose(
        pathway.signal_amplitudes, steady.signal_amplitudes, rtol=1e-3, atol=0.0
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: CPMGIRTrain(signal_convention="physics"),
        lambda: CPMGImaging(density_normalization="none"),
        lambda: Hardware(receive_mode="loud"),
    ],
)
def test_invalid_option_values_are_rejected(build: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match="must be"):
        build()
