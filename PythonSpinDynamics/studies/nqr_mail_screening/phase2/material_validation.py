"""Independent complex-permittivity cuboid-volume collocation benchmark.

Uses Magpylib only for its exact cuboid depolarization kernel. Magnetostatic H
and electrostatic polarization have the same scalar-potential kernel. This is
an independent validation dependency, not a replacement for package coil APIs.
"""

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from spatial_loading import EPS0, ElectrostaticReference
from spin_dynamics.fields.quasistatic import vector_potential


def depolarization_matrix(points, dimensions):
    from magpylib.core import magnet_cuboid_Bfield

    n = len(points)
    out = np.empty((n, 3, n, 3))
    for start in range(0, n, 64):
        stop = min(n, start + 64)
        relative = (points[start:stop, None, :] - points[None, :, :]).reshape(-1, 3)
        sizes = np.tile(dimensions, (stop - start, 1))
        for axis in range(3):
            polarization = np.zeros_like(relative)
            polarization[:, axis] = 1.0
            b = magnet_cuboid_Bfield(relative, sizes, polarization).reshape(
                stop - start, n, 3
            )
            # Inside source voxel: H = (B-J)/mu0. Dimensionless E kernel = mu0 H/J.
            b[np.arange(stop - start), np.arange(start, stop), axis] -= 1.0
            out[start:stop, :, :, axis] = b.transpose(0, 2, 1)
    return out.reshape(3 * n, 3 * n)


def solve_regions(
    regions,
    conductors,
    networks,
    shape=(9, 1, 9),
    frequency=3.3043e6,
    translation=(0.0, 0.0, 0.0),
):
    if not 3.1024e6 <= frequency <= 3.3043e6:
        raise ValueError("frequency outside the validated coil band")
    for region in regions:
        epsilon = region.relative_permittivity - 1j * region.conductivity_s_m / (
            2 * np.pi * frequency * EPS0
        )
        k = 2 * np.pi * frequency * np.sqrt(4e-7 * np.pi * EPS0 * epsilon)
        if (
            region.conductivity_s_m / (2 * np.pi * frequency * EPS0) > 1
            or abs(k) * np.linalg.norm(region.size_m) > 0.1
        ):
            raise ValueError(
                "material exceeds the electroquasistatic validation domain"
            )
    points = []
    dimensions = []
    contrasts = []
    conductivities = []
    for region in regions:
        p, _ = region.voxels(shape, translation)
        points.extend(p)
        dimensions.extend(np.tile(np.asarray(region.size_m) / shape, (len(p), 1)))
        contrasts.extend(
            [
                region.relative_permittivity
                - 1
                - 1j * region.conductivity_s_m / (2 * np.pi * frequency * EPS0)
            ]
            * len(p)
        )
        conductivities.extend([region.conductivity_s_m] * len(p))
    points = np.asarray(points)
    dimensions = np.asarray(dimensions)
    volumes = np.prod(dimensions, axis=1)
    chi = np.asarray(contrasts)
    kernel = depolarization_matrix(points, dimensions)
    matrix = np.eye(len(kernel), dtype=complex) - kernel * np.repeat(chi, 3)[None, :]
    factored = lu_factor(matrix)
    results = {}
    for name, conductor in conductors.items():
        e = ElectrostaticReference(conductor).field(points)
        path = conductor.path_points
        a = vector_potential(points, list(zip(path[:-1], path[1:])), current=1.0)
        network = networks[name]
        fgrid = np.linspace(3.1024e6, 3.3043e6, len(network["R"]))
        zw = np.interp(
            frequency, fgrid, network["R"]
        ) + 2j * np.pi * frequency * np.interp(frequency, fgrid, network["L"])
        # PEEC potential ramps 0 -> +1 V along the path, so current is -1/Z.
        incident = e + 2j * np.pi * frequency * a / zw
        rhs = np.column_stack((e.reshape(-1), incident.reshape(-1)))
        solution = lu_solve(factored, rhs)
        residual = np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs)
        electrostatic = solution[:, 0].reshape(-1, 3)
        total = solution[:, 1].reshape(-1, 3)
        delta = EPS0 * np.sum(chi * np.sum(e * electrostatic, axis=1) * volumes)
        conductance = float(
            np.sum(
                np.asarray(conductivities)
                * np.sum(np.abs(total) ** 2, axis=1)
                * volumes
            )
        )
        results[name] = {
            "delta_capacitance_f": float(delta.real),
            "electrostatic_conductance_s": float(-2 * np.pi * frequency * delta.imag),
            "total_conductance_s": conductance,
            "relative_linear_residual": float(residual),
            "voxels": len(points),
            "shape": shape,
        }
    return results
