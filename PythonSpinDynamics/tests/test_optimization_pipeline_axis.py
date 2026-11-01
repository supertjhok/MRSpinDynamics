"""Regression tests: the excitation pipeline uses the tuned-probe refocusing axis.

MATLAB ``opt_exc_pulse_tuned_repeat.m`` derives the refocusing axis with
``calc_rot_axis_tuned_probe_lp_Orig`` (probe transient plus ``tfp`` free
precession before and after the pulse). The pipeline previously used the ideal
``calc_rot_axis_arba4`` of the bare segments, which differs by up to ~2 in
``neff`` (an n_z sign flip off resonance plus a 90-degree transverse rotation).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import spin_dynamics.optimization.pipeline as pipeline_module
from spin_dynamics.optimization import run_tuned_excitation_inverse_pipeline
from spin_dynamics.optimization.spa import evaluate_tuned_refocusing_pulse
from spin_dynamics.parameters import set_params_tuned_spa
from spin_dynamics.probes.tuned import calc_rot_axis_tuned_probe_lp_orig2

NUMPTS = 21
PHASES = np.array([0.0, 0.4, 1.3, 0.0, np.pi, 2.0, 0.7, 0.0, 0.2, 1.1])
SEGMENT_FRACTION = 0.1


def _reference_axis(*, tfp_t180: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    params, sp, pp = set_params_tuned_spa(numpts=NUMPTS)
    tfp = params.tfp if tfp_t180 is None else tfp_t180 * pp.T_180
    params = params.__class__(
        **{
            **params.__dict__,
            "tref": SEGMENT_FRACTION * pp.T_180 * np.ones(PHASES.size),
            "pref": PHASES,
            "aref": np.ones(PHASES.size),
            "tfp": tfp,
        }
    )
    return sp.del_w, calc_rot_axis_tuned_probe_lp_orig2(params, sp, pp)


def _capture_neff(monkeypatch, refocusing, **kwargs) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}

    def fake_excitation(_segments, neff, **_options):
        captured["neff"] = np.asarray(neff).copy()
        target = SimpleNamespace(
            del_w=np.linspace(-10.0, 10.0, NUMPTS),
            mrx=np.ones(NUMPTS, dtype=np.complex128),
            snr=1.0,
            phases=np.zeros(1),
        )
        result = SimpleNamespace(best_evaluation=target, best_score=1.0)
        return SimpleNamespace(results=(result,), best_result=result, best_score=1.0)

    def fake_inverse(*_args, **_options):
        evaluation = SimpleNamespace(
            excitation=SimpleNamespace(mrx=-np.ones(NUMPTS, dtype=np.complex128))
        )
        result = SimpleNamespace(best_evaluation=evaluation, best_score=1.0)
        return SimpleNamespace(
            results=(result,), best_index=0, best_result=result, best_score=1.0
        )

    monkeypatch.setattr(
        pipeline_module.drivers, "run_tuned_excitation_multistart", fake_excitation
    )
    monkeypatch.setattr(
        pipeline_module.drivers, "run_tuned_inverse_excitation_multistart", fake_inverse
    )
    result = run_tuned_excitation_inverse_pipeline(
        refocusing, excitation_segments=1, numpts=NUMPTS, **kwargs
    )
    captured["del_w"] = result.del_w
    np.testing.assert_allclose(result.neff, captured["neff"])
    return captured


def test_python_tuned_result_uses_tuned_probe_axis(monkeypatch) -> None:
    evaluation = evaluate_tuned_refocusing_pulse(
        PHASES, segment_fraction=SEGMENT_FRACTION, numpts=NUMPTS
    )
    result = SimpleNamespace(
        best_phases=PHASES, best_evaluation=evaluation, best_score=evaluation.snr
    )
    refocusing = SimpleNamespace(best_result=result, results=(result,))

    captured = _capture_neff(monkeypatch, refocusing)
    del_w, expected = _reference_axis()

    np.testing.assert_allclose(captured["del_w"], del_w)
    np.testing.assert_allclose(captured["neff"], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("times_are_t180", [True, False])
def test_matlab_cell_free_precession_is_included(monkeypatch, times_are_t180) -> None:
    tfp_t180 = 1.5  # differs from the tuned SPA default of 3 T180
    times_t180 = np.concatenate(
        [[tfp_t180], SEGMENT_FRACTION * np.ones(PHASES.size), [tfp_t180]]
    )
    cell = np.empty((1, 7), dtype=object)
    cell[0, 0] = times_t180 if times_are_t180 else np.pi * times_t180
    cell[0, 1] = np.concatenate([[0.0], PHASES, [0.0]])
    cell[0, 2] = np.concatenate([[0.0], np.ones(PHASES.size), [0.0]])
    cell[0, 3] = 1.0
    cell[0, 4] = {}
    cell[0, 5] = {}
    cell[0, 6] = {}
    cells = np.empty((1, 1), dtype=object)
    cells[0, 0] = cell

    captured = _capture_neff(
        monkeypatch, cells, result_times_are_t180=times_are_t180
    )
    del_w, expected = _reference_axis(tfp_t180=tfp_t180)

    np.testing.assert_allclose(captured["del_w"], del_w)
    np.testing.assert_allclose(captured["neff"], expected, rtol=1e-12, atol=1e-12)
    # The padding matters: the default-tfp axis is measurably different.
    _, default_tfp_axis = _reference_axis()
    assert np.max(np.abs(default_tfp_axis - expected)) > 1e-2


def test_maxoffs_must_match_tuned_evaluator_grid() -> None:
    result = SimpleNamespace(
        best_phases=PHASES, best_evaluation=None, best_score=1.0
    )
    refocusing = SimpleNamespace(best_result=result, results=(result,))
    with pytest.raises(ValueError, match="maxoffs"):
        run_tuned_excitation_inverse_pipeline(
            refocusing, excitation_segments=1, numpts=NUMPTS, maxoffs=5.0
        )
