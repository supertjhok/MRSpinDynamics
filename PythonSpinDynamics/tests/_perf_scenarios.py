"""Canonical hot-path scenarios for the performance acceleration work.

Phase 0 of the JAX/Numba plan (see ``docs/performance.md``) locks the output of
the current NumPy reference path so every later backend (Numba, JAX) can be
validated bit-for-bit, within tolerance, against it.

Each scenario is a zero-argument callable returning a flat ``dict`` of named
NumPy arrays (scalars are stored as 0-d arrays). Scenarios are deliberately
small and deterministic so the golden fixture is cheap to regenerate and the
parity test runs fast. They cover the four kernel shapes a compiled backend
will touch:

* ``rf_matrix_elements`` — elementwise pulse-matrix construction;
* ``arb10_cpmg`` — the core segment-loop propagator, end to end;
* ``v0crit_objective`` — a pulse-optimization objective evaluation;
* ``nqr_diagonalize`` — dense quadrupolar diagonalization.

This module is imported both by ``tests/test_perf_golden.py`` and by
``benchmarks/forward_kernel.py``; it only depends on ``spin_dynamics`` so it can
be imported once ``src`` is on ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Callable
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from spin_dynamics.core.rotations import rf_matrix_elements  # noqa: E402
from spin_dynamics.nqr.hamiltonians import diagonalize_site  # noqa: E402
from spin_dynamics.nqr.systems import QuadrupolarSite  # noqa: E402
from spin_dynamics.optimization.refocusing import (  # noqa: E402
    evaluate_ideal_v0crit_refocusing_pulse,
)
from spin_dynamics.workflows import run_ideal_cpmg_train  # noqa: E402


Scenario = Callable[[], dict[str, np.ndarray]]


def tiny_arb10_params(numpts: int = 9) -> dict:
    """A small two-pulse acquisition with free-precession gradients.

    Shared by the Numba and JAX backend parity tests so they exercise the raw
    ``sim_spin_dynamics_arb10`` kernel without workflow assembly.
    """

    del_w = np.linspace(-5.0, 5.0, numpts)
    exc = rf_matrix_elements(del_w, w1=1.0, tp=np.pi / 2, phi=0.0)
    ref = rf_matrix_elements(del_w, w1=1.0, tp=np.pi, phi=np.pi / 2)
    return {
        "tp": np.array([np.pi / 2, 1.0, np.pi, 1.3]),
        "pul": np.array([1, 0, 2, 0]),
        "amp": np.array([1.0, 0.0, 1.0, 0.0]),
        "acq": np.array([False, False, False, True]),
        "grad": np.array([0.0, 0.2, 0.0, 0.1]),
        "Rtot": [exc, ref],
        "del_w": del_w,
        "del_wg": np.ones(numpts),
        "w_1": np.ones(numpts),
        "T1n": np.full(numpts, 100.0),
        "T2n": np.full(numpts, 50.0),
        "m0": np.ones(numpts, dtype=np.complex128),
        "mth": np.zeros(numpts, dtype=np.complex128),
    }


def scenario_rf_matrix_elements() -> dict[str, np.ndarray]:
    """Elementwise RF-pulse matrix elements over an offset grid."""

    del_w = np.linspace(-10.0, 10.0, 128)
    mat = rf_matrix_elements(del_w, w1=1.2, tp=0.7, phi=0.3)
    return {
        "del_w": del_w,
        "R_00": mat.R_00,
        "R_0p": mat.R_0p,
        "R_0m": mat.R_0m,
        "R_p0": mat.R_p0,
        "R_m0": mat.R_m0,
        "R_pp": mat.R_pp,
        "R_mm": mat.R_mm,
        "R_pm": mat.R_pm,
        "R_mp": mat.R_mp,
    }


def scenario_arb10_cpmg() -> dict[str, np.ndarray]:
    """End-to-end finite ideal CPMG train through the core segment loop."""

    result = run_ideal_cpmg_train(
        numpts=129,
        maxoffs=8.0,
        num_echoes=6,
        t1_seconds=1.7,
        t2_seconds=1.1,
        num_workers=1,
        auto_refine_grid=False,
        rephase_action="ignore",
    )
    return {
        "del_w": np.asarray(result.del_w, dtype=np.float64),
        "mrx": np.asarray(result.mrx, dtype=np.complex128),
        "echo_integrals": np.asarray(result.echo_integrals, dtype=np.complex128),
    }


def scenario_v0crit_objective() -> dict[str, np.ndarray]:
    """Ideal v0crit refocusing objective evaluation (an optimizer inner step)."""

    phases = np.linspace(0.0, np.pi, 16)
    evaluation = evaluate_ideal_v0crit_refocusing_pulse(phases, numpts=101)
    return {
        "phases": phases,
        "masy": np.asarray(evaluation.masy, dtype=np.complex128),
        "v0crit": np.asarray(evaluation.v0crit, dtype=np.float64),
        "score": np.asarray(evaluation.score, dtype=np.float64),
        "axis_rms": np.asarray(evaluation.axis_rms, dtype=np.float64),
        "v0crit_average": np.asarray(evaluation.v0crit_average, dtype=np.float64),
    }


def scenario_nqr_diagonalize() -> dict[str, np.ndarray]:
    """Dense quadrupolar diagonalization for a spin-1 NaNO2-like 14N site."""

    site = QuadrupolarSite(
        spin=1.0,
        quadrupole_frequency_hz=3.75e6,
        eta=0.112,
        gamma_hz_per_t=3.077e6,
        isotope="14N",
    )
    eigensystem = diagonalize_site(site)
    return {
        "levels_hz": np.asarray(eigensystem.levels_hz, dtype=np.float64),
        "transition_hz": np.asarray(
            [t.frequency_hz for t in eigensystem.transitions], dtype=np.float64
        ),
    }


SCENARIOS: dict[str, Scenario] = {
    "rf_matrix_elements": scenario_rf_matrix_elements,
    "arb10_cpmg": scenario_arb10_cpmg,
    "v0crit_objective": scenario_v0crit_objective,
    "nqr_diagonalize": scenario_nqr_diagonalize,
}


def compute_all() -> dict[str, np.ndarray]:
    """Compute every scenario, flattened to ``"{scenario}::{key}"`` names."""

    flat: dict[str, np.ndarray] = {}
    for name, fn in SCENARIOS.items():
        for key, value in fn().items():
            flat[f"{name}::{key}"] = np.asarray(value)
    return flat
