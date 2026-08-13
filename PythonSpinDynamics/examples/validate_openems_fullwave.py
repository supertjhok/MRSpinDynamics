"""Run or summarize the Phase 4 openEMS validation suite."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from spin_dynamics.fields.fullwave_validation import (
    FullWaveValidationReport,
    apply_validation_report,
    low_frequency_loop_check,
    mesh_convergence_check,
    numerical_termination_check,
    openems_conductive_loss_check,
    reciprocity_check,
)
from spin_dynamics.fields.harmonic import save_harmonic_em_npz
from spin_dynamics.fields.openems import (
    OpenEMSLumpedPort,
    OpenEMSPlanarConductor,
    load_openems_solution,
    loaded_loop_openems_reference,
    run_openems,
    unloaded_loop_openems_reference,
)


def _label(millimetres: float) -> str:
    return f"{millimetres:g}".replace(".", "p")


def _reciprocity_projects():
    base = unloaded_loop_openems_reference(
        frequency_hz=128.0e6,
        max_cell_size_m=5.0e-3,
        max_timesteps=100_000,
    )
    conductor = base.planar_conductors[0]
    port = base.ports[0]

    def shifted_conductor(name: str, x_offset: float) -> OpenEMSPlanarConductor:
        offset = np.array([x_offset, 0.0, 0.0])
        return OpenEMSPlanarConductor(
            name,
            conductor.points_m + offset,
            normal_axis=conductor.normal_axis,
            priority=conductor.priority,
        )

    def shifted_port(
        number: int,
        name: str,
        x_offset: float,
        excite: bool,
    ) -> OpenEMSLumpedPort:
        offset = np.array([x_offset, 0.0, 0.0])
        return OpenEMSLumpedPort(
            number,
            port.start_m + offset,
            port.stop_m + offset,
            port.resistance_ohm,
            kind="lumped",
            excite=excite,
            priority=port.priority,
            name=name,
        )

    conductors = (
        shifted_conductor("loop_a", -0.04),
        shifted_conductor("loop_b", 0.04),
    )
    bounds = np.array([[-0.17, 0.17], [-0.125, 0.125], [-0.125, 0.125]])
    projects = []
    for label, excite_a, excite_b in (("a", True, False), ("b", False, True)):
        ports = (
            shifted_port(1, "a", -0.04, excite_a),
            shifted_port(2, "b", 0.04, excite_b),
        )
        projects.append(
            replace(
                base,
                name=f"reciprocity_drive_{label}",
                wires=(),
                planar_conductors=conductors,
                ports=ports,
                simulation_bounds_m=bounds,
                metadata={"reference_case": "phase4-two-port-reciprocity"},
            )
        )
    return tuple(projects)


def _run_cases(
    output: Path,
    python_executable: str,
    mesh_sizes_mm: tuple[float, ...],
    timeout: float,
) -> None:
    base = loaded_loop_openems_reference()
    for millimetres in mesh_sizes_mm:
        project = replace(
            base,
            name=f"loaded_loop_strip_{_label(millimetres)}mm",
            settings=replace(
                base.settings,
                max_cell_size_m=millimetres * 1.0e-3,
                max_timesteps=150_000,
            ),
        )
        run_openems(
            project,
            directory=output / f"strip_{_label(millimetres)}mm",
            python_executable=python_executable,
            timeout=timeout,
        )
    run_openems(
        unloaded_loop_openems_reference(),
        directory=output / "unloaded_strip_32mhz_5mm",
        python_executable=python_executable,
        timeout=timeout,
    )
    for project, label in zip(_reciprocity_projects(), ("a", "b")):
        run_openems(
            project,
            directory=output / f"reciprocity_drive_{label}",
            python_executable=python_executable,
            timeout=timeout,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or summarize Phase 4 openEMS validation checks."
    )
    parser.add_argument(
        "--output", type=Path, default=Path(".tmp/openems_phase4_strip")
    )
    parser.add_argument("--python", default="python", help="Python containing openEMS")
    parser.add_argument("--run", action="store_true", help="run all solver cases")
    parser.add_argument(
        "--mesh-sizes-mm",
        type=float,
        nargs="+",
        default=(3.0, 2.5, 2.0),
    )
    parser.add_argument("--mesh-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--sample-boundary-exclusion-mm",
        type=float,
        default=None,
        help=(
            "exclude this distance from the loaded-sample interface during "
            "mesh comparison (default: the coarsest requested cell size)"
        ),
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    sizes = tuple(float(value) for value in args.mesh_sizes_mm)
    if len(sizes) < 2 or any(value <= 0.0 for value in sizes):
        parser.error("--mesh-sizes-mm requires at least two positive values")
    output = args.output.resolve()
    if args.run:
        _run_cases(output, args.python, sizes, args.timeout)

    loaded = {
        value: load_openems_solution(output / f"strip_{_label(value)}mm")
        for value in sizes
    }
    reference = loaded_loop_openems_reference()
    probes = np.stack(
        np.meshgrid(*reference.field_domain.axes, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    sample = reference.samples[0]
    boundary_exclusion_m = (
        max(sizes) * 1.0e-3
        if args.sample_boundary_exclusion_mm is None
        else float(args.sample_boundary_exclusion_mm) * 1.0e-3
    )
    if boundary_exclusion_m < 0.0 or boundary_exclusion_m >= min(
        sample.radius_m, 0.5 * sample.length_m
    ):
        parser.error("sample boundary exclusion must fit inside the sample")
    relative = probes - sample.center_m
    axial_index = "xyz".index(sample.axis)
    radial_indices = [index for index in range(3) if index != axial_index]
    inside_sample = (
        np.abs(relative[:, axial_index])
        <= 0.5 * sample.length_m - boundary_exclusion_m
    ) & (
        np.linalg.norm(relative[:, radial_indices], axis=1)
        <= sample.radius_m - boundary_exclusion_m
    )
    validation_probes = probes[inside_sample]
    if validation_probes.size == 0:
        parser.error("sample boundary exclusion leaves no validation probes")
    checks = [
        replace(
            numerical_termination_check(loaded[value]),
            name=f"time_domain_termination_{_label(value)}mm",
        )
        for value in sizes
    ]
    checks.extend(
        mesh_convergence_check(
            loaded[left],
            loaded[right],
            validation_probes,
            relative_tolerance=args.mesh_tolerance,
            name=f"mesh_{_label(left)}_to_{_label(right)}mm",
        )
        for left, right in zip(sizes[:-1], sizes[1:])
    )
    finest = sizes[-1]
    checks.append(openems_conductive_loss_check(output / f"strip_{_label(finest)}mm"))
    low_frequency = load_openems_solution(output / "unloaded_strip_32mhz_5mm")
    checks.append(
        replace(
            numerical_termination_check(low_frequency),
            name="time_domain_termination_unloaded_32mhz",
        )
    )
    checks.append(low_frequency_loop_check(low_frequency, radius_m=0.065))
    drive_a = load_openems_solution(output / "reciprocity_drive_a")
    drive_b = load_openems_solution(output / "reciprocity_drive_b")
    checks.append(reciprocity_check(drive_a, drive_b, port_a=1, port_b=2))
    report = FullWaveValidationReport(
        tuple(checks),
        metadata={
            "backend": "openEMS",
            "mesh_sizes_mm": list(sizes),
            "validated_region": "loaded-sample interior",
            "sample_boundary_exclusion_m": boundary_exclusion_m,
            "num_validation_probes": int(validation_probes.shape[0]),
            "interpretation": (
                "fields are trusted in the stated validation region only when "
                "every required check passes; dielectric-interface E is excluded"
            ),
        },
    )
    report_path = report.write_json(output / "fullwave_validation.json")
    validated = apply_validation_report(loaded[finest], report)
    solution_path = output / "validated_harmonic_solution.npz"
    save_harmonic_em_npz(solution_path, validated)
    print(f"Validation passed: {report.passed}")
    for check in report.checks:
        print(f"  {'PASS' if check.passed else 'FAIL'} {check.name}: {check.metric}")
    print(f"Wrote {report_path}")
    print(f"Wrote {solution_path}")


if __name__ == "__main__":
    main()
