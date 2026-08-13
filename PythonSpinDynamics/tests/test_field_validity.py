"""Quasistatic validity metrics, policy, and RF-entry-point integration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from spin_dynamics.fields import (
    BirdcageGeometry,
    QuasistaticRegion,
    QuasistaticValidityError,
    QuasistaticValidityWarning,
    apply_validity_policy,
    assess_quasistatic_validity,
    birdcage_quadrature_mode,
    coil_loading,
    coils,
    solve_birdcage_field,
)


def test_high_permittivity_region_requires_full_wave() -> None:
    region = QuasistaticRegion(
        "high-epsilon sample",
        characteristic_length_m=0.30,
        relative_permittivity=80.0,
    )
    assessment = assess_quasistatic_validity(128.0e6, regions=(region,))

    assert assessment.assessed
    assert assessment.requires_full_wave
    assert assessment.region_metrics[0].phase_span == pytest.approx(7.2, rel=0.02)
    assert any(
        finding.criterion == "material_phase_span"
        for finding in assessment.findings
    )


def test_good_conductor_outside_born_regime_recommends_mqs() -> None:
    region = QuasistaticRegion(
        "copper shield",
        characteristic_length_m=1.0e-3,
        conductivity_s_per_m=5.8e7,
        born_approximation=True,
    )
    assessment = assess_quasistatic_validity(1.0e6, regions=(region,))

    assert assessment.requires_self_consistent_mqs
    assert not assessment.requires_full_wave
    assert any(
        finding.criterion == "born_reaction_field"
        for finding in assessment.findings
    )


def test_static_assessment_passes_without_division_artifacts() -> None:
    region = QuasistaticRegion(
        "static conductor",
        characteristic_length_m=0.1,
        conductivity_s_per_m=1.0,
        born_approximation=True,
    )
    assessment = assess_quasistatic_validity(
        0.0,
        coil_extent_m=0.2,
        regions=(region,),
    )

    assert assessment.ok
    assert assessment.region_metrics[0].phase_span == 0.0
    assert assessment.region_metrics[0].displacement_to_conduction_ratio == 0.0


def test_missing_context_warns_and_error_policy_is_available() -> None:
    assessment = assess_quasistatic_validity(
        10.0e6,
        coil_extent_m=0.1,
        regions=None,
    )
    assert any(
        finding.criterion == "material_context"
        for finding in assessment.findings
    )
    with pytest.warns(QuasistaticValidityWarning, match="material context"):
        apply_validity_policy(assessment, solver_name="test solver")
    with pytest.raises(QuasistaticValidityError, match="test solver"):
        apply_validity_policy(
            assessment,
            solver_name="test solver",
            policy="error",
        )


def _birdcage() -> tuple[BirdcageGeometry, np.ndarray]:
    geometry = BirdcageGeometry(
        radius=0.05,
        length=0.10,
        n_rungs=8,
        ring_segments_per_section=3,
    )
    return geometry, np.asarray([geometry.center])


def test_birdcage_reports_unassessed_frequency_and_retains_result() -> None:
    geometry, points = _birdcage()
    mode = birdcage_quadrature_mode(geometry)

    with pytest.warns(QuasistaticValidityWarning, match="no RF frequency"):
        solution = solve_birdcage_field(geometry, mode, points)

    assert solution.validity is not None
    assert not solution.validity.assessed


def test_birdcage_low_frequency_air_case_can_be_explicitly_assessed() -> None:
    geometry, points = _birdcage()
    mode = birdcage_quadrature_mode(geometry)

    with warnings.catch_warnings():
        warnings.simplefilter("error", QuasistaticValidityWarning)
        solution = solve_birdcage_field(
            geometry,
            mode,
            points,
            frequency_hz=1.0e6,
            validity_regions=(),
        )

    assert solution.validity is not None
    assert solution.validity.ok


def test_coil_loading_retains_worst_frequency_assessment() -> None:
    loop = coils.conducting_ring(radius=0.02, n_segments=24)
    axis = np.linspace(-0.005, 0.005, 3)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.stack([x, y, z], axis=-1)
    result = coil_loading(
        points,
        loop,
        conductivity=0.1,
        mask=np.ones(points.shape[:-1], dtype=bool),
        spacing=(0.005, 0.005, 0.005),
        frequencies=(1.0e3, 2.0e3),
        inductance=1.0e-6,
        coil_resistance=1.0,
        validity_policy="ignore",
    )

    assert result.validity is not None
    assert result.validity.frequency_hz == 2.0e3
