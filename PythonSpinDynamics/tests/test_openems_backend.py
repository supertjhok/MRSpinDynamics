"""Optional openEMS project generation, execution, and import tests."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import spin_dynamics.fields.openems as openems_module
from spin_dynamics.fields import coils
from spin_dynamics.fields.openems import (
    OPENEMS_PROJECT_SCHEMA,
    OpenEMSExecutionError,
    OpenEMSProject,
    OpenEMSWire,
    check_openems_python,
    load_openems_field_dump,
    load_openems_project,
    load_openems_solution,
    loaded_loop_openems_reference,
    run_openems,
    segments_to_polylines,
    write_openems_project,
)


def test_segment_geometry_joins_loops_without_inventing_connections() -> None:
    ring = coils.conducting_ring(radius=0.02, n_segments=12)
    ring_paths = segments_to_polylines(ring)
    assert len(ring_paths) == 1
    assert ring_paths[0].shape == (13, 3)
    np.testing.assert_allclose(ring_paths[0][0], ring_paths[0][-1], atol=1.0e-15)

    two_turns = coils.solenoid(
        radius=0.02,
        length=0.01,
        turns=2,
        n_segments=12,
    )
    solenoid_paths = segments_to_polylines(two_turns)
    assert len(solenoid_paths) == 2
    assert all(path.shape == (13, 3) for path in solenoid_paths)


def test_wire_from_common_segment_format_preserves_radius_and_paths() -> None:
    segments = coils.conducting_ring(radius=0.03, n_segments=16)
    wire = OpenEMSWire.from_segments("loop", segments, radius_m=1.0e-3)

    assert wire.name == "loop"
    assert wire.radius_m == 1.0e-3
    assert len(wire.polylines_m) == 1
    assert not wire.polylines_m[0].flags.writeable


def test_loaded_loop_reference_contains_wave_sensitive_sample_and_feed() -> None:
    project = loaded_loop_openems_reference()

    assert project.settings.frequency_hz == 128.0e6
    assert project.samples[0].relative_permittivity == 80.0
    assert project.samples[0].conductivity_s_per_m == 0.5
    assert project.samples[0].axis == "x"
    assert project.ports[0].kind == "lumped"
    assert project.ports[0].excite
    assert project.ports[0].direction == "z"
    assert not project.wires
    assert len(project.planar_conductors) == 1
    assert project.planar_conductors[0].normal_axis == "x"
    assert project.planar_conductors[0].points_m.shape == (192, 3)
    assert project.metadata["strip_width_m"] == pytest.approx(6.0e-3)
    assert np.ptp(project.ports[0].stop_m - project.ports[0].start_m) > 0.0
    assert project.field_domain.shape == (21, 17, 17)
    assert project.settings.boundary_conditions == ("PML_8",) * 6
    assert len(project.model_hash) == 64


def test_project_json_round_trip_and_standalone_driver_generation(tmp_path) -> None:
    project = loaded_loop_openems_reference(n_wire_points=24)
    project_path, driver_path = write_openems_project(project, tmp_path)
    loaded = load_openems_project(project_path)

    assert loaded.model_hash == project.model_hash
    assert loaded.to_dict() == project.to_dict()
    driver = driver_path.read_text(encoding="utf-8")
    ast.parse(driver)
    assert "AddWire" in driver
    assert "AddPolygon" in driver
    assert "AddCurvePort" in driver
    assert 'dump_type=10' in driver
    assert 'dump_type=12' in driver
    assert 'dump_type=15' in driver
    assert '"J_fd"' in driver
    assert "SetBoundaryCond" in driver
    assert 'engine="multithreaded"' in driver
    assert "numThreads=max(1, min(8, os.cpu_count() or 1))" in driver
    assert "float(np.min(coordinates))" in driver


def test_project_loader_rejects_unknown_schema(tmp_path) -> None:
    path = tmp_path / "project.json"
    path.write_text(json.dumps({"schema": "unknown/v99"}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported openEMS project schema"):
        load_openems_project(path)


def test_availability_probe_is_nonthrowing_for_any_python_environment() -> None:
    availability = check_openems_python(sys.executable)

    assert availability.python_executable == sys.executable
    assert isinstance(availability.available, bool)
    if availability.available:
        assert availability.version
    else:
        assert availability.detail


def test_external_runner_reports_driver_failure_and_retains_logs(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        openems_module,
        "_OPENEMS_DRIVER",
        "raise SystemExit('synthetic openEMS failure')\n",
    )
    with pytest.raises(OpenEMSExecutionError, match="synthetic openEMS failure"):
        run_openems(
            loaded_loop_openems_reference(n_wire_points=24),
            directory=Path("relative-openems-run"),
            python_executable=sys.executable,
            setup_only=True,
        )
    assert "synthetic openEMS failure" in (
        tmp_path / "relative-openems-run" / "openems_stderr.log"
    ).read_text(encoding="utf-8")


def _canonical_field(shape: tuple[int, int, int], scale: float) -> np.ndarray:
    grid = np.indices(shape, dtype=np.float64)
    result = np.empty(shape + (3,), dtype=np.complex128)
    result[..., 0] = scale * (1.0 + grid[0] + 1.0j * grid[1])
    result[..., 1] = scale * (2.0 + grid[1] - 0.5j * grid[2])
    result[..., 2] = scale * (3.0 + grid[2] + 0.25j * grid[0])
    return result


def _write_openems_dump(
    path: Path,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    frequency_hz: float,
    *,
    legacy: bool,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.attrs["openEMS_HDF5_version"] = 0.3
        handle.attrs["legacy_fmt"] = legacy
        mesh = handle.create_group("Mesh")
        mesh.attrs["mesh_type"] = 0
        mesh.attrs["mesh_scaling"] = 1.0e-3
        for name, axis in zip("xyz", axes):
            mesh.create_dataset(name, data=axis)
        frequency = handle.create_group("FieldData/FD")
        frequency.attrs["frequency"] = np.asarray([frequency_hz])
        component_first = np.moveaxis(values, -1, 0)
        if legacy:
            raw = np.transpose(component_first, (0, 3, 2, 1))
            real = frequency.create_dataset("f0_real", data=np.real(raw))
            frequency.create_dataset("f0_imag", data=np.imag(raw))
            real.attrs["d_order"] = "NZYX"
        else:
            dataset = frequency.create_dataset("f0", data=component_first)
            dataset.attrs["d_order"] = "NXYZ"


@pytest.mark.parametrize("legacy", [False, True])
def test_openems_hdf5_imports_current_and_legacy_vector_orders(
    tmp_path,
    legacy: bool,
) -> None:
    axes = (
        np.array([-0.01, 0.0, 0.01]),
        np.array([-0.02, 0.02]),
        np.array([-0.03, 0.0, 0.03, 0.06]),
    )
    expected = _canonical_field((3, 2, 4), 1.0)
    path = tmp_path / "field.h5"
    _write_openems_dump(path, axes, expected, 128.0e6, legacy=legacy)

    loaded = load_openems_field_dump(path, frequency_hz=128.0e6)

    assert loaded.domain.shape == (3, 2, 4)
    assert loaded.legacy_format is legacy
    np.testing.assert_allclose(loaded.values, expected)
    # openEMS stores Cartesian HDF5 mesh coordinates in physical units already;
    # mesh_scaling records the drawing-unit conversion and is not applied twice.
    np.testing.assert_allclose(loaded.domain.axes[0], axes[0])


def _write_solution_bundle(
    directory: Path,
    *,
    current_a: complex = 2.0j,
) -> tuple[OpenEMSProject, np.ndarray, np.ndarray]:
    project = loaded_loop_openems_reference(n_wire_points=24)
    write_openems_project(project, directory)
    axes = (
        np.array([-0.01, 0.0, 0.01]),
        np.array([-0.02, 0.02]),
        np.array([-0.03, 0.0, 0.03, 0.06]),
    )
    electric = _canonical_field((3, 2, 4), 10.0)
    magnetic = _canonical_field((3, 2, 4), 1.0e-6)
    _write_openems_dump(
        directory / "E_fd.h5",
        axes,
        electric,
        project.settings.frequency_hz,
        legacy=False,
    )
    _write_openems_dump(
        directory / "B_fd.h5",
        axes,
        magnetic,
        project.settings.frequency_hz,
        legacy=False,
    )
    (directory / "openems_run.json").write_text(
        json.dumps(
            {
                "completed": True,
                "backend": "openEMS",
                "backend_version": "0.0.36-test",
                "termination_verified": False,
            }
        ),
        encoding="utf-8",
    )
    (directory / "openems_port.json").write_text(
        json.dumps(
            [
                {
                    "number": 1,
                    "name": "feed",
                    "voltage_v": [100.0, 0.0],
                    "current_a": [current_a.real, current_a.imag],
                    "accepted_power_w": 25.0,
                    "reference_impedance_ohm": 50.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    return project, electric, magnetic


def test_solution_import_normalizes_complex_fields_and_keeps_provenance(
    tmp_path,
) -> None:
    project, electric, magnetic = _write_solution_bundle(tmp_path)

    solution = load_openems_solution(tmp_path, normalization="per_ampere")

    assert solution.normalization.kind == "per_ampere"
    assert solution.normalization.reference_value == pytest.approx(2.0)
    assert solution.normalization.port_index == 1
    np.testing.assert_allclose(solution.electric_field_v_per_m, electric / (2.0j))
    np.testing.assert_allclose(
        solution.magnetic_flux_density_t,
        magnetic / (2.0j),
    )
    assert solution.provenance.backend == "openEMS"
    assert solution.provenance.backend_version == "0.0.36-test"
    assert solution.provenance.model_hash == project.model_hash
    assert solution.provenance.metadata["termination_verified"] is False
    assert solution.convergence is None
    assert solution.materials[0].relative_permittivity == 80.0


def test_solution_import_supports_accepted_power_normalization(tmp_path) -> None:
    _, electric, _ = _write_solution_bundle(tmp_path)

    solution = load_openems_solution(tmp_path, normalization="per_sqrt_watt")

    assert solution.normalization.reference_value == pytest.approx(5.0)
    np.testing.assert_allclose(solution.electric_field_v_per_m, electric / 5.0)


def test_solution_import_rejects_zero_current_normalization(tmp_path) -> None:
    _write_solution_bundle(tmp_path, current_a=0.0j)

    with pytest.raises(ValueError, match="zero per_ampere"):
        load_openems_solution(tmp_path, normalization="per_ampere")


def test_project_schema_constant_is_versioned() -> None:
    assert OPENEMS_PROJECT_SCHEMA.endswith("/v1")
    assert OpenEMSProject.from_dict(
        loaded_loop_openems_reference(n_wire_points=24).to_dict()
    ).model_hash
