"""Reproducible Phase 2 field, loading, network and surrogate gate."""

from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from aperture import Region, candidates, field
from spatial_loading import (
    ElectrostaticReference,
    region_integrals,
    dielectric_bounds,
    TunedCircuit,
    EPS0,
    combined_incident_integral,
)
from spin_dynamics.fields.coil_peec import extract_impedance_surface, self_capacitance

HERE = Path(__file__).resolve().parent
LIMITS = {
    "field_rms": 0.01,
    "field_max_normalized": 0.05,
    "field_path": 0.01,
    "peec_path": 0.01,
    "peec_surface": 0.01,
    "peec_full_discrepancy": 0.01,
    "peec_full_path": 0.02,
    "spectral_interpolation": 0.001,
    "voxel_electric": 0.03,
    "voxel_coupling": 0.03,
    "voxel_inductive_field": 0.05,
    "material_surrogate": 0.03,
    "material_solver_residual": 1e-10,
    "pose_integral": 0.03,
}
FREQUENCIES = [3.1024e6, 3.20335e6, 3.3043e6]


def relative(a, b):
    return float(np.max(np.abs(np.asarray(a) - b) / np.maximum(np.abs(b), 1e-30)))


def peec(name):
    cache_dir = Path(".tmp/nqr_phase2_gate/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import spin_dynamics.fields.coil_peec as peec_module

    fingerprint = hashlib.sha256(
        Path(peec_module.__file__).read_bytes()
        + candidates(64)[name].path_points.tobytes()
        + str(FREQUENCIES).encode()
    ).hexdigest()
    cache = cache_dir / (name + "_" + fingerprint + ".json")
    rows = json.loads(cache.read_text()) if cache.exists() else []
    cached = len(rows) == 7
    for path, per, form in (
        (16, 8, "chain"),
        (32, 8, "chain"),
        (64, 8, "chain"),
        (64, 16, "chain"),
        (64, 32, "chain"),
        (16, 8, "full"),
        (32, 8, "full"),
    ):
        if cached:
            continue
        c = candidates(path)[name]
        z = extract_impedance_surface(c, FREQUENCIES, n_perimeter=per, formulation=form)
        rows.append(
            {
                "path": path,
                "perimeter": per,
                "formulation": form,
                "L": z.inductance.tolist(),
                "R": z.resistance.tolist(),
                "C": self_capacitance(c),
            }
        )
        print(name, "PEEC", path, per, form, flush=True)
    cache.write_text(json.dumps(rows, indent=2))

    def discrepancy(i, j):
        return max(relative(rows[i][k], rows[j][k]) for k in ("L", "R", "C"))

    selected = rows[4]
    middle = max(
        abs((selected[k][0] + selected[k][2]) / 2 / selected[k][1] - 1)
        for k in ("L", "R")
    )
    return rows, {
        "peec_path": discrepancy(1, 2),
        "peec_surface": discrepancy(3, 4),
        "peec_full_discrepancy": discrepancy(1, 6),
        "peec_full_path": discrepancy(5, 6),
        "spectral_interpolation": middle,
    }


def maps(name, output):
    c = candidates(128)[name]
    rng = np.random.default_rng(49979687)
    low = np.array([-0.17, -0.0175, -0.315])
    high = -low
    random = rng.uniform(low, high, (1000, 3))
    # Boundary points prevent a random-only test from missing edge errors.
    boundary = np.array(
        list(
            itertools.product(
                np.linspace(low[0], high[0], 17),
                (low[1], 0.0, high[1]),
                np.linspace(low[2], high[2], 33),
            )
        )
    )
    query = np.vstack((random, boundary))
    direct = field(c, query)
    scale = np.sqrt(np.mean(np.sum(direct**2, axis=1)))
    path = field(candidates(64)[name], query)
    path_error = np.sqrt(np.mean(np.sum((path - direct) ** 2, axis=1))) / scale
    attempts = []
    for nx, ny, nz in ((33, 5, 65), (65, 9, 129), (129, 9, 193)):
        axes = [np.linspace(a, b, n) for a, b, n in zip(low, high, (nx, ny, nz))]
        points = np.array(list(itertools.product(*axes)))
        values = field(c, points).reshape(nx, ny, nz, 3)
        interpolation = RegularGridInterpolator(axes, values)
        error = np.linalg.norm(interpolation(query) - direct, axis=1) / scale
        result = {
            "shape": [nx, ny, nz],
            "rms": float(np.sqrt(np.mean(error**2))),
            "max": float(error.max()),
        }
        attempts.append(result)
        print(name, "map", result, flush=True)
        if (
            result["rms"] <= LIMITS["field_rms"]
            and result["max"] <= LIMITS["field_max_normalized"]
        ):
            break
    np.savez_compressed(
        output / (name + "_converged_map.npz"),
        x_m=axes[0],
        y_m=axes[1],
        z_m=axes[2],
        tx_t_a=values,
        rx_t_a=values,
        validation_points_m=query,
        validation_direct_t_a=direct,
    )
    return attempts, {
        "field_rms": result["rms"],
        "field_max_normalized": result["max"],
        "field_path": float(path_error),
    }


def parcels():
    return [
        Region(
            "target packet",
            (0.10, 0.002, 0.10),
            (0.0, 0.0, 0.0),
            0.001,
            1.0,
            293.15,
            3.0,
            1e-8,
        ),
        Region(
            "paper surrogate",
            (0.33, 0.001, 0.42),
            (0.0, -0.01, 0.0),
            0.020,
            0.0,
            293.15,
            3.0,
            1e-8,
        ),
    ]


def spatial(name, network):
    c = candidates(128)[name]
    reference = ElectrostaticReference(c)
    details = []
    checks = {
        "voxel_electric": 0.0,
        "voxel_coupling": 0.0,
        "voxel_inductive_field": 0.0,
    }
    for translation in ((0.0, 0.0, -0.1), (0.0, 0.0, 0.0), (0.0, 0.0, 0.1)):
        totals = []
        for shape in ((9, 3, 11), (17, 5, 21), (33, 7, 41)):
            electric = a_squared = e_dot_a = coupling = 0.0
            for region in parcels():
                e, r = region_integrals(reference, region, shape, translation)
                electric += e
                a_squared += r["a_squared"]
                e_dot_a += r["e_dot_a"]
                if region.crystalline_fraction:
                    points, mass = region.voxels(shape, translation)
                    coupling += float(
                        np.sum(np.sum(field(c, points) ** 2, axis=1) * mass)
                    )
            totals.append(
                {
                    "shape": shape,
                    "electric_integral_m": electric,
                    "a_squared": a_squared,
                    "e_dot_a": e_dot_a,
                    "mass_coupling": coupling,
                }
            )
        last, previous = totals[-1], totals[-2]
        for key, metric in (
            ("electric_integral_m", "voxel_electric"),
            ("mass_coupling", "voxel_coupling"),
            ("a_squared", "voxel_inductive_field"),
        ):
            checks[metric] = max(checks[metric], relative(previous[key], last[key]))
        details.append({"translation_m": translation, "refinements": totals})
        print(name, "voxel", translation, checks, flush=True)
    # Integrate a moving finite packet's coupling, refining transport samples.
    trajectory = []
    packet = parcels()[0]
    for n in (17, 33, 65):
        positions = np.linspace(-0.315, 0.315, n)
        points, mass = packet.voxels((5, 3, 7))
        moved = np.tile(points, (n, 1))
        moved[:, 2] += np.repeat(positions, len(points))
        b2 = np.sum(field(c, moved) ** 2, axis=1).reshape(n, -1) @ mass
        trajectory.append(float(np.trapezoid(b2, positions)))
    checks["pose_integral"] = relative(trajectory[-2], trajectory[-1])
    # Circuit corners include uncertain material, component values and temperature.
    circuit_rows = []
    for pose in details:
        last = pose["refinements"][-1]
        for fidx in (0, 2):
            frequency = FREQUENCIES[fidx]
            circuit = TunedCircuit(
                network["L"][fidx], network["R"][fidx], network["C"], frequency
            )
            for epsilon, sigma in itertools.product((1.5, 3.0, 6.0), (0.0, 1e-8, 1e-4)):
                bounds = dielectric_bounds(
                    last["electric_integral_m"], epsilon, sigma, frequency
                )
                incident = combined_incident_integral(
                    last["electric_integral_m"],
                    last["a_squared"],
                    last["e_dot_a"],
                    frequency,
                    network["R"][fidx] + 2j * np.pi * frequency * network["L"][fidx],
                )
                loss = dielectric_bounds(incident, epsilon, sigma, frequency)[
                    "shunt_conductance_s"
                ]
                results = []
                for dc, g, ls, rs, cs, temp in itertools.product(
                    bounds["delta_capacitance_f"],
                    loss,
                    (0.95, 1.05),
                    (0.95, 1.05),
                    (0.98, 1.02),
                    (273.15, 323.15),
                ):
                    results.append(
                        circuit.evaluate(
                            dc,
                            g,
                            l_scale=ls,
                            r_scale=rs,
                            c_scale=cs,
                            wire_temperature=temp,
                            parcel_temperature=temp,
                        )
                    )
                summary = {
                    key: [
                        float(min(x[key] for x in results)),
                        float(max(x[key] for x in results)),
                    ]
                    for key in results[0]
                }
                circuit_rows.append(
                    {
                        "translation_m": pose["translation_m"],
                        "frequency_hz": frequency,
                        "epsilon_r": epsilon,
                        "conductivity_s_m": sigma,
                        "dielectric_bounds": bounds,
                        "total_loss_conductance_bounds_s": loss,
                        "circuit_corner_ranges": summary,
                        "range_scope": "evaluated corners; interior extrema are not certified",
                    }
                )
    return {
        "refinements": details,
        "trajectory_integrals": trajectory,
        "circuit_cases": circuit_rows,
    }, checks


def analytic_loading_checks():
    # Independent exact isolated-sphere and slab eigenresponses in uniform E.
    error = 0.0
    cases = 0
    for eps, sigma, f, t in itertools.product(
        (1.0, 1.5, 3.0, 6.0), (0.0, 1e-8, 1e-4), FREQUENCIES, (0.0, 1 / 3, 1.0)
    ):
        volume = 1e-6
        e0 = 7.0
        integral = volume * e0**2
        chi = eps - 1 - 1j * sigma / (2 * np.pi * f * EPS0)
        actual_c = EPS0 * integral * (chi / (1 + t * chi)).real
        actual_g = sigma * integral / abs(1 + t * chi) ** 2
        bounds = dielectric_bounds(integral, eps, sigma, f)
        for actual, limits in (
            (actual_c, bounds["delta_capacitance_f"]),
            (actual_g, bounds["shunt_conductance_s"]),
        ):
            error = max(
                error,
                max(limits[0] - actual, actual - limits[1], 0.0)
                / max(limits[1], 1e-30),
            )
        cases += 1
    return {"cases": cases, "maximum_normalized_bound_violation": float(error)}


def material_benchmarks(report):
    from material_validation import solve_regions

    conductors = candidates(128)
    networks = {
        name: data["network_refinements"][4]
        for name, data in report["candidates"].items()
    }
    cases = []
    checks = {name: 0.0 for name in conductors}
    residual = {name: 0.0 for name in conductors}
    for parameters in (
        ((3.0, 1e-4), (3.0, 1e-4)),
        ((6.0, 1e-5), (2.0, 1e-4)),
        ((2.0, 1e-4), (6.0, 1e-8)),
    ):
        regions = [
            replace(region, relative_permittivity=eps, conductivity_s_m=sigma)
            for region, (eps, sigma) in zip(parcels(), parameters)
        ]
        for z in (-0.1, 0.0, 0.1):
            for frequency in (FREQUENCIES[0], FREQUENCIES[-1]):
                results = []
                # Level A reduced material mesh versus Level B finer thickness/area mesh.
                for shape in ((13, 1, 13), (17, 2, 17)):
                    results.append(
                        solve_regions(
                            regions,
                            conductors,
                            networks,
                            shape,
                            frequency,
                            (0.0, 0.0, z),
                        )
                    )
                if any(
                    relative(results[-2][name][key], results[-1][name][key])
                    > LIMITS["material_surrogate"]
                    for name in conductors
                    for key in ("delta_capacitance_f", "total_conductance_s")
                ):
                    results.append(
                        solve_regions(
                            regions,
                            conductors,
                            networks,
                            (21, 2, 21),
                            frequency,
                            (0.0, 0.0, z),
                        )
                    )
                for name in conductors:
                    checks[name] = max(
                        checks[name],
                        max(
                            relative(results[-2][name][key], results[-1][name][key])
                            for key in ("delta_capacitance_f", "total_conductance_s")
                        ),
                    )
                    residual[name] = max(
                        residual[name], results[-1][name]["relative_linear_residual"]
                    )
                cases.append(
                    {
                        "materials": parameters,
                        "translation_z_m": z,
                        "frequency_hz": frequency,
                        "levels": results,
                    }
                )
                print(
                    "Material benchmark", parameters, z, frequency, checks, flush=True
                )
    for name in conductors:
        report["candidates"][name]["checks"]["material_surrogate"] = checks[name]
        report["candidates"][name]["checks"]["material_solver_residual"] = residual[
            name
        ]
    return cases


def main():
    output = Path(".tmp/nqr_phase2_gate")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 2,
        "limits": LIMITS,
        "supported_domain": {
            "mail_clear_m": [0.34, 0.035, 0.43],
            "map_x_y_z_m": [[-0.17, 0.17], [-0.0175, 0.0175], [-0.315, 0.315]],
            "material": "uniform contrasts for analytic envelopes; explicit heterogeneous packet/paper cases for reduced-vs-fine voxel validation; 1.5<=epsilon_r<=6, 0<=sigma<=1e-4 S/m",
            "limitations": "no shielding foil or magnetic contents; no full-wave validation; numerical gate applies to reported geometry/material/pose family",
            "frequencies_hz": FREQUENCIES,
            "temperature_k": [273.15, 323.15],
            "component_tolerances": {
                "L_fraction": 0.05,
                "R_fraction": 0.05,
                "tuning_C_fraction": 0.02,
            },
        },
        "analytic_loading": analytic_loading_checks(),
        "candidates": {},
    }
    for name in candidates():
        networks, nchecks = peec(name)
        fields, fchecks = maps(name, output)
        loading, lchecks = spatial(name, networks[4])
        checks = nchecks | fchecks | lchecks
        report["candidates"][name] = {
            "network_refinements": networks,
            "field_refinements": fields,
            "spatial_loading": loading,
            "checks": checks,
            "passed": all(checks[k] <= LIMITS[k] for k in checks),
        }
        (output / "gate2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    report["material_benchmarks"] = material_benchmarks(report)
    for data in report["candidates"].values():
        data["passed"] = all(
            data["checks"][key] <= LIMITS[key] for key in data["checks"]
        )
    report["numerical_domain_passed"] = (
        all(c["passed"] for c in report["candidates"].values())
        and report["analytic_loading"]["maximum_normalized_bound_violation"] < 1e-12
    )
    # Gate is explicitly limited to the reported geometry/material/pose family.
    report["gate2_passed"] = report["numerical_domain_passed"]
    report["outside_validated_scope"] = [
        "conductive shielding foils, metal or magnetic contents",
        "full-wave/skin-effect regime",
        "material/pose families outside the reported validation cases",
    ]
    report["source_sha256"] = {
        n: hashlib.sha256(HERE.joinpath(n).read_bytes()).hexdigest()
        for n in (
            "validate_phase2.py",
            "spatial_loading.py",
            "aperture.py",
            "material_validation.py",
        )
    }
    (output / "gate2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Numerical domain passed:", report["numerical_domain_passed"], flush=True)
    if not report["gate2_passed"]:
        raise SystemExit(
            "Phase 2 numerical gate failed; inspect checks before proceeding"
        )


if __name__ == "__main__":
    main()
