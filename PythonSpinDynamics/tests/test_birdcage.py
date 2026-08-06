"""Prescribed-current birdcage geometry, mode, and field tests."""

from __future__ import annotations

import numpy as np
import pytest

from spin_dynamics.fields.birdcage import (
    BirdcageCurrentMode,
    BirdcageGeometry,
    birdcage_current_mode,
    birdcage_field_metrics,
    birdcage_linear_mode,
    birdcage_quadrature_mode,
    solve_birdcage_field,
)


def _geometry(*, ring_segments: int = 8) -> BirdcageGeometry:
    return BirdcageGeometry(
        radius=0.15,
        length=0.30,
        n_rungs=16,
        ring_segments_per_section=ring_segments,
    )


def test_geometry_connects_rungs_and_end_ring_sections() -> None:
    geometry = _geometry(ring_segments=3)
    rungs = geometry.rung_segments()
    positive = geometry.positive_end_ring_sections
    negative = geometry.negative_end_ring_sections

    assert len(rungs) == len(positive) == len(negative) == geometry.n_rungs
    assert all(len(section) == 3 for section in positive + negative)
    for index in range(geometry.n_rungs):
        following = (index + 1) % geometry.n_rungs
        np.testing.assert_allclose(positive[index][0][0], rungs[index][1], atol=1.0e-15)
        np.testing.assert_allclose(positive[index][-1][1], rungs[following][1], atol=1.0e-15)
        np.testing.assert_allclose(negative[index][0][0], rungs[index][0], atol=1.0e-15)
        np.testing.assert_allclose(negative[index][-1][1], rungs[following][0], atol=1.0e-15)


def test_prescribed_mode_enforces_end_ring_kcl() -> None:
    geometry = _geometry()
    mode = birdcage_linear_mode(geometry)

    assert mode.max_kcl_residual_a < 1.0e-12
    assert not mode.rung_currents_a.flags.writeable
    np.testing.assert_allclose(
        mode.negative_end_ring_currents_a,
        -mode.positive_end_ring_currents_a,
    )
    with pytest.raises(ValueError, match="sum to zero"):
        birdcage_current_mode(np.ones(geometry.n_rungs))


def test_fundamental_linear_modes_are_degenerate_and_orthogonal_at_center() -> None:
    geometry = _geometry()
    center = np.asarray([geometry.center])
    cosine = birdcage_linear_mode(geometry)
    sine = birdcage_linear_mode(
        geometry,
        azimuthal_phase_rad=np.pi / 2.0,
    )
    cosine_field = solve_birdcage_field(geometry, cosine, center).field_t[0]
    sine_field = solve_birdcage_field(geometry, sine, center).field_t[0]

    assert np.linalg.norm(cosine_field) == pytest.approx(
        np.linalg.norm(sine_field),
        rel=1.0e-12,
    )
    assert abs(np.vdot(cosine_field, sine_field)) < (
        1.0e-12 * np.linalg.norm(cosine_field) * np.linalg.norm(sine_field)
    )
    assert abs(cosine_field[2]) < 1.0e-12 * np.linalg.norm(cosine_field)
    assert abs(sine_field[2]) < 1.0e-12 * np.linalg.norm(sine_field)


def test_quadrature_modes_select_opposite_circular_components() -> None:
    geometry = _geometry()
    center = np.asarray([geometry.center])
    plus = solve_birdcage_field(
        geometry,
        birdcage_quadrature_mode(geometry, handedness=1),
        center,
    )
    minus = solve_birdcage_field(
        geometry,
        birdcage_quadrature_mode(geometry, handedness=-1),
        center,
    )

    assert abs(plus.b1_minus_t[0]) < 1.0e-12 * abs(plus.b1_plus_t[0])
    assert abs(minus.b1_plus_t[0]) < 1.0e-12 * abs(minus.b1_minus_t[0])
    np.testing.assert_allclose(
        abs(plus.b1_plus_t[0]),
        abs(minus.b1_minus_t[0]),
        rtol=1.0e-12,
    )


def test_quadrature_field_is_uniform_and_circular_in_central_roi() -> None:
    geometry = _geometry()
    axis = np.linspace(-0.06, 0.06, 13)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    roi = xx**2 + yy**2 <= 0.06**2
    solution = solve_birdcage_field(
        geometry,
        birdcage_quadrature_mode(geometry, handedness=1),
        points,
    )
    metrics = birdcage_field_metrics(
        solution,
        roi_mask=roi,
        target_handedness=1,
    )

    assert metrics.coefficient_of_variation < 0.01
    assert metrics.peak_to_peak_nonuniformity < 0.03
    assert metrics.circularity > 0.999
    assert metrics.circularity_db > 40.0
    assert metrics.transverse_fraction > 0.999


def test_end_ring_segmentation_converges_at_center() -> None:
    coarse_geometry = _geometry(ring_segments=4)
    fine_geometry = _geometry(ring_segments=16)
    center = np.asarray([coarse_geometry.center])
    coarse = solve_birdcage_field(
        coarse_geometry,
        birdcage_quadrature_mode(coarse_geometry),
        center,
    )
    fine = solve_birdcage_field(
        fine_geometry,
        birdcage_quadrature_mode(fine_geometry),
        center,
    )

    np.testing.assert_allclose(
        abs(coarse.b1_plus_t[0]),
        abs(fine.b1_plus_t[0]),
        rtol=5.0e-4,
    )


@pytest.mark.parametrize("axis", ("x", "y", "z"))
def test_quadrature_handedness_follows_cage_axis(axis: str) -> None:
    geometry = BirdcageGeometry(
        radius=0.15,
        length=0.30,
        n_rungs=16,
        axis=axis,
        ring_segments_per_section=8,
    )
    solution = solve_birdcage_field(
        geometry,
        birdcage_quadrature_mode(geometry, handedness=1),
        geometry.center,
    )

    assert abs(solution.b1_minus_t) < 1.0e-12 * abs(solution.b1_plus_t)
    axial = np.vdot(geometry.axis_vector, solution.field_t)
    assert abs(axial) < 1.0e-12 * np.linalg.norm(solution.field_t)

def test_birdcage_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="n_rungs"):
        BirdcageGeometry(radius=0.1, length=0.2, n_rungs=3)
    geometry = _geometry()
    with pytest.raises(ValueError, match="end-ring KCL"):
        BirdcageCurrentMode(
            rung_currents_a=np.zeros(4),
            positive_end_ring_currents_a=np.arange(4),
            negative_end_ring_currents_a=np.zeros(4),
        )
    with pytest.raises(ValueError, match="n_rungs/2"):
        birdcage_linear_mode(geometry, mode_index=8)
    with pytest.raises(ValueError, match="handedness"):
        birdcage_quadrature_mode(geometry, handedness=0)
    with pytest.raises(ValueError, match="same number of rungs"):
        solve_birdcage_field(
            geometry,
            birdcage_quadrature_mode(
                BirdcageGeometry(radius=0.1, length=0.2, n_rungs=8)
            ),
            np.zeros((1, 3)),
        )
