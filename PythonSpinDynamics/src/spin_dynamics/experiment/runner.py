"""Execution of planned experiments via the registered workflows."""

from __future__ import annotations

import platform
import sys
import time
import warnings as _warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from spin_dynamics.experiment.plan import ExperimentPlan, plan_experiment
from spin_dynamics.experiment.registry import resolve
from spin_dynamics.experiment.specs import Experiment


class ExperimentPlanError(ValueError):
    """Raised when ``run()`` is called on an experiment whose plan has errors."""


@dataclass(frozen=True)
class RunRecord:
    """A completed experiment run: spec, native result, and provenance."""

    experiment: Experiment
    result: Any
    plan: ExperimentPlan
    provenance: dict[str, Any]

    def save(self, path: str) -> None:
        from spin_dynamics.experiment.io import save_run

        save_run(self, path)


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("python-spin-dynamics")
    except Exception:
        return "unknown"


def run_experiment(experiment: Experiment, **execution: Any) -> RunRecord:
    """Plan and execute an experiment, delegating to the registered workflow.

    ``execution`` accepts runtime knobs that do not affect the physics
    (``num_workers``, ``tau_workers``); knobs the resolved workflow does not
    take are rejected. Plan errors raise :class:`ExperimentPlanError`; plan
    warnings are emitted via :mod:`warnings` and recorded in the plan.
    """

    # Skip the runtime estimate here: run() measures the actual elapsed time,
    # and estimating would trigger the calibration dry run on first use.
    plan = plan_experiment(experiment, estimate=False)
    if not plan.ok:
        raise ExperimentPlanError(
            "experiment plan has errors:\n" + plan.report()
        )
    for message in plan.warnings:
        _warnings.warn(message, UserWarning, stacklevel=2)

    entry = resolve(type(experiment.sequence), experiment.hardware.probe)
    assert entry is not None  # plan.ok guarantees resolution

    unknown = set(execution) - set(entry.execution_kwargs)
    if unknown:
        allowed = ", ".join(sorted(entry.execution_kwargs)) or "none"
        raise TypeError(
            f"{entry.name} does not accept execution options "
            f"{sorted(unknown)}; allowed: {allowed}"
        )

    func = entry.resolved_func(experiment)
    kwargs = entry.build_kwargs(experiment)
    kwargs.update(execution)

    start = time.perf_counter()
    result = func(**kwargs)
    elapsed = time.perf_counter() - start

    provenance = {
        "workflow": func.__name__,
        "package_version": _package_version(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "execution": dict(execution),
        "plan_warnings": list(plan.warnings),
    }
    return RunRecord(
        experiment=experiment, result=result, plan=plan, provenance=provenance
    )
