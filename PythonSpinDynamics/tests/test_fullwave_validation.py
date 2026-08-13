"""Phase 4 full-wave validation metric tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from spin_dynamics.fields.domain import SpatialDomain
from spin_dynamics.fields.fullwave_validation import (
    FullWaveValidationCheck,
    FullWaveValidationReport,
    apply_validation_report,
    integrate_conductive_loss,
    low_frequency_loop_check,
    mesh_convergence_check,
    numerical_termination_check,
    reciprocity_check,
)
from spin_dynamics.fields.harmonic import (
    MU_0,
    HarmonicConvergence,
    HarmonicEMSolution,
    HarmonicFieldNormalization,
    HarmonicPort,
    HarmonicSolverProvenance,
)
from spin_dynamics.fields.openems import _parse_openems_output


def _linear_solution(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    ports: tuple[HarmonicPort, ...] = (),
) -> HarmonicEMSolution:
    domain = SpatialDomain(axes)
    x, y, z = np.meshgrid(*axes, indexing="ij")
    electric = np.stack((1.0 + x, 2.0 * y, 3.0 * z), axis=-1).astype(complex)
    magnetic = 1.0e-6 * np.stack((2.0 + x, y - z, 1.0 + z), axis=-1)
    return HarmonicEMSolution(
        domain=domain,
        frequency_hz=32.0e6,
        phasor_convention="exp(+iwt)",
        electric_field_v_per_m=electric,
        magnetic_flux_density_t=magnetic,
        normalization=HarmonicFieldNormalization("per_ampere"),
        ports=ports,
        provenance=HarmonicSolverProvenance("synthetic"),
        convergence=HarmonicConvergence(
            False,
            relative_residual=1.0e-6,
            iterations=100,
            metadata={
                "time_domain_terminated": True,
                "end_criteria": 1.0e-5,
                "final_energy_db": -60.0,
            },
        ),
    )


def test_mesh_convergence_interpolates_complex_fields_on_common_points() -> None:
    coarse_axis = np.array([-1.0, 0.0, 1.0])
    fine_axis = np.linspace(-1.0, 1.0, 5)
    coarse = _linear_solution((coarse_axis,) * 3)
    fine = _linear_solution((fine_axis,) * 3)
    probes = np.array([[0.0, 0.0, 0.0], [0.25, -0.4, 0.7]])

    check = mesh_convergence_check(coarse, fine, probes, relative_tolerance=1e-12)

    assert check.passed
    assert check.metric == pytest.approx(0.0, abs=1.0e-15)


def test_mesh_convergence_detects_field_change() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    coarse = _linear_solution((axis,) * 3)
    fine = replace(
        coarse,
        magnetic_flux_density_t=1.2 * coarse.magnetic_flux_density_t,
    )

    check = mesh_convergence_check(
        coarse,
        fine,
        np.array([[0.0, 0.0, 0.0]]),
        relative_tolerance=0.05,
    )

    assert not check.passed
    assert check.metric > 0.05


def test_low_frequency_loop_check_recovers_biot_savart_center_field() -> None:
    radius = 0.065
    axis = np.array([-0.01, 0.0, 0.01])
    solution = _linear_solution((axis,) * 3)
    magnetic = np.zeros(solution.field_shape, dtype=complex)
    magnetic[..., 0] = MU_0 / (2.0 * radius)
    solution = replace(solution, magnetic_flux_density_t=magnetic)

    check = low_frequency_loop_check(
        solution,
        radius_m=radius,
        relative_tolerance=1.0e-12,
    )

    assert check.passed
    assert check.metric == pytest.approx(0.0)


def test_reciprocity_compares_transfer_impedance_in_both_directions() -> None:
    axis = np.array([-1.0, 0.0, 1.0])
    z_transfer = 2.0 + 3.0j
    drive_a = _linear_solution(
        (axis,) * 3,
        ports=(
            HarmonicPort(1, "a", current_a=1.0 + 0.0j, voltage_v=5.0),
            HarmonicPort(2, "b", current_a=0.1, voltage_v=z_transfer),
        ),
    )
    drive_b = _linear_solution(
        (axis,) * 3,
        ports=(
            HarmonicPort(1, "a", current_a=0.1, voltage_v=2.0 * z_transfer),
            HarmonicPort(2, "b", current_a=2.0 + 0.0j, voltage_v=6.0),
        ),
    )

    check = reciprocity_check(drive_a, drive_b, port_a=1, port_b=2)

    assert check.passed
    assert check.metric == pytest.approx(0.0)


def test_conductive_loss_integrates_complex_poynting_work_density() -> None:
    axis = np.array([0.0, 0.5, 1.0])
    shape = (3, 3, 3, 3)
    electric = np.zeros(shape, dtype=complex)
    current = np.zeros(shape, dtype=complex)
    electric[..., 0] = 2.0
    current[..., 0] = 3.0

    assert integrate_conductive_loss(electric, current, (axis,) * 3) == pytest.approx(
        3.0
    )


def test_termination_parser_and_solution_check_distinguish_mesh_validation() -> None:
    stdout = """
FDTD simulation size: 67x69x66 --> 305118 FDTD cells
FDTD timestep is: 4.53824e-12 s
[@ 25s] Timestep: 9009 || Energy: ~2e-21 (-54.48dB)
Time for 9009 iterations with 305118.00 cells : 25.57 sec
"""
    parsed = _parse_openems_output(stdout, "", end_criteria=1.0e-5)

    assert parsed["termination_verified"] is True
    assert parsed["mesh_cells"] == 305118
    assert parsed["iterations"] == 9009
    assert parsed["relative_energy"] == pytest.approx(10.0 ** -5.448)
    solution = _linear_solution((np.array([-1.0, 0.0, 1.0]),) * 3)
    check = numerical_termination_check(solution)
    assert check.passed
    assert not solution.convergence.converged


def test_report_serialization_and_application_promote_only_required_passes(
    tmp_path,
) -> None:
    report = FullWaveValidationReport(
        (
            FullWaveValidationCheck("required", True, metric=0.01, tolerance=0.05),
            FullWaveValidationCheck("advisory", False, required=False),
        ),
        metadata={"case": "synthetic"},
    )
    path = report.write_json(tmp_path / "report.json")
    solution = _linear_solution((np.array([-1.0, 0.0, 1.0]),) * 3)
    validated = apply_validation_report(solution, report)

    assert report.passed
    assert '"passed": true' in path.read_text(encoding="utf-8")
    assert validated.convergence.converged
    assert "fullwave_validation" in validated.convergence.metadata
