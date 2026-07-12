from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "regression_gate.py"
SPEC = importlib.util.spec_from_file_location("regression_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


def test_comparison_accepts_noise_and_small_absolute_slowdown():
    policy = gate.RegressionPolicy(min_case_delta_seconds=0.010)
    cases, geomean, failed = gate.compare_results(
        {"large": 1.0, "tiny": 0.005},
        {"large": 1.10, "tiny": 0.009},
        policy,
    )
    assert not failed
    assert not any(case.failed for case in cases)
    assert geomean == pytest.approx(1.10**0.5)


def test_comparison_rejects_large_case_regression():
    cases, _geomean, failed = gate.compare_results(
        {"kernel": 0.100, "compiler": 0.100},
        {"kernel": 0.160, "compiler": 0.100},
        gate.RegressionPolicy(max_geomean_ratio=2.0),
    )
    assert failed
    assert {case.name for case in cases if case.failed} == {"kernel"}


def test_comparison_rejects_broad_aggregate_regression():
    _cases, geomean, failed = gate.compare_results(
        {"a": 1.0, "b": 2.0},
        {"a": 1.30, "b": 2.60},
        gate.RegressionPolicy(max_case_ratio=2.0, max_geomean_ratio=1.25),
    )
    assert geomean == pytest.approx(1.30)
    assert failed


def test_improvement_does_not_mask_a_regression_in_aggregate():
    _cases, geomean, failed = gate.compare_results(
        {"improved": 1.0, "regressed": 1.0},
        {"improved": 0.5, "regressed": 1.6},
        gate.RegressionPolicy(max_case_ratio=2.0, max_geomean_ratio=1.25),
    )
    assert geomean == pytest.approx(1.6**0.5)
    assert failed


def test_comparison_rejects_case_set_or_invalid_timings():
    with pytest.raises(ValueError, match="cases differ"):
        gate.compare_results({"a": 1.0}, {"b": 1.0}, gate.RegressionPolicy())
    with pytest.raises(ValueError, match="invalid timing"):
        gate.compare_results({"a": 0.0}, {"a": 1.0}, gate.RegressionPolicy())


def test_alignment_allows_new_cases_but_rejects_removed_cases():
    baseline, candidate, added = gate.align_case_sets(
        {"stable": 1.0},
        {"stable": 1.1, "new": 0.2},
    )
    assert baseline == {"stable": 1.0}
    assert candidate == {"stable": 1.1}
    assert added == ("new",)
    with pytest.raises(ValueError, match="removed benchmark"):
        gate.align_case_sets({"removed": 1.0}, {"other": 1.0})


def test_empty_or_zero_push_baseline_falls_back_to_parent():
    assert gate._normalize_baseline_ref("") == "HEAD^"
    assert gate._normalize_baseline_ref("000000") == "HEAD^"
    assert gate._normalize_baseline_ref("origin/main") == "origin/main"
