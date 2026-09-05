"""Phase 2 parcel/coil model, using built-in Biot-Savart and PEEC solvers."""

from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from spin_dynamics.fields.magnetostatics import biot_savart
from spin_dynamics.fields.coil_peec import (
    Conductor,
    helical_solenoid,
    extract_impedance_surface,
    self_capacitance,
)


@dataclass(frozen=True)
class Region:
    material: str
    size_m: tuple
    centre_m: tuple
    mass_kg: float
    crystalline_fraction: float
    temperature_k: float
    relative_permittivity: float
    conductivity_s_m: float

    def __post_init__(self):
        if len(self.size_m) != 3 or len(self.centre_m) != 3 or min(self.size_m) <= 0:
            raise ValueError("need three positive region dimensions")
        if (
            self.mass_kg < 0
            or not 0 <= self.crystalline_fraction <= 1
            or self.temperature_k <= 0
        ):
            raise ValueError("invalid material amount, fraction or temperature")
        if self.relative_permittivity < 1 or self.conductivity_s_m < 0:
            raise ValueError("invalid loading properties")

    def voxels(self, shape=(5, 3, 7), translation=(0.0, 0.0, 0.0), rotation=None):
        axes = [
            ((np.arange(n) + 0.5) / n - 0.5) * d for n, d in zip(shape, self.size_m)
        ]
        local = np.array(list(itertools.product(*axes))) + np.asarray(self.centre_m)
        matrix = np.eye(3) if rotation is None else np.asarray(rotation)
        if (
            matrix.shape != (3, 3)
            or not np.allclose(matrix.T @ matrix, np.eye(3))
            or np.linalg.det(matrix) < 0
        ):
            raise ValueError("pose must be a proper rotation")
        points = local @ matrix.T + np.asarray(translation)
        return points, np.full(len(points), self.mass_kg / len(points))


def candidates(resolution=16):
    enclosing = helical_solenoid(
        diameter=0.36, length=0.46, turns=6, wire_radius=0.0015, n_per_turn=resolution
    )
    # Two open rectangular loops in series, normal to y; 1 mm feed gaps.
    paths = []
    for y in (-0.03, 0.03):
        corners = np.array(
            [
                [-0.18, y, -0.23],
                [0.18, y, -0.23],
                [0.18, y, 0.23],
                [-0.18, y, 0.23],
                [-0.18, y, -0.229],
            ]
        )
        paths.append(
            np.vstack(
                [
                    np.linspace(a, b, max(2, resolution // 4), endpoint=False)
                    for a, b in zip(corners[:-1], corners[1:])
                ]
                + [corners[-1:]]
            )
        )
    split = Conductor(np.vstack(paths), wire_radius=0.0015)
    left = helical_solenoid(
        diameter=0.36, length=0.20, turns=3, wire_radius=0.0015, n_per_turn=resolution
    ).path_points.copy()
    right = left[::-1].copy()
    left[:, 2] -= 0.13
    right[:, 2] += 0.13
    bridge = np.array(
        [left[-1], [0.21, 0.0, left[-1, 2]], [0.21, 0.0, right[0, 2]], right[0]]
    )
    gradiometer = Conductor(np.vstack((left, bridge[1:-1], right)), wire_radius=0.0015)
    return {
        "enclosing_helix": enclosing,
        "split_rectangular_pair": split,
        "axial_gradiometer": gradiometer,
    }


def field(conductor, points):
    path = conductor.path_points
    return biot_savart(np.asarray(points), list(zip(path[:-1], path[1:])), current=1.0)


def loading_sweep(inductance, resistance, self_c, frequency):
    # Explicit lumped perturbations, not a voxel dielectric-field solution.
    omega = 2 * np.pi * frequency
    total_c = 1 / (omega**2 * inductance)
    rows = []
    for extra_c in (0.0, 1e-12, 5e-12):
        for extra_r in (0.0, 0.5, 2.0):
            cap = total_c + extra_c
            r = resistance + extra_r
            resonant = 1 / (2 * np.pi * np.sqrt(inductance * cap))
            q = 2 * np.pi * resonant * inductance / r
            rows.append(
                {
                    "added_c_f": extra_c,
                    "added_r_ohm": extra_r,
                    "resonant_frequency_hz": resonant,
                    "q": q,
                    "ringdown_s": 2 * inductance / r,
                    "current_per_drive_volt_a": 1
                    / abs(r + 1j * (omega * inductance - 1 / (omega * cap))),
                    "johnson_psd_v2_hz": 4 * 1.380649e-23 * 293.15 * r,
                    "positive_external_tuning_capacitance": bool(total_c > self_c),
                }
            )
    return rows


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    regions = [
        Region(
            "fentanyl-HCl aniline",
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
    # EM properties are placeholders for later evidence-backed loading cases.
    axes = [
        np.linspace(-0.165, 0.165, 9),
        np.linspace(-0.0125, 0.0125, 5),
        np.linspace(-0.21, 0.21, 13),
    ]
    points = np.array(list(itertools.product(*axes)))
    rng = np.random.default_rng(32452843)
    probes = rng.uniform(
        [a.min() for a in axes], [a.max() for a in axes], size=(200, 3)
    )
    report = {
        "phase": 2,
        "gate2_passed": False,
        "regions": [asdict(r) for r in regions],
        "loading_evidence": "assumed region properties; lumped R/C sensitivity only",
        "remaining_gate_items": [
            "voxel-dependent dielectric/conductive loading",
            "PEEC path and surface-current convergence",
            "spatial and pose convergence",
            "surrogate error acceptance across candidates",
        ],
        "candidates": {},
    }
    fine = candidates(32)
    for name, conductor in candidates().items():
        tx = field(conductor, points)
        rx = tx.copy()  # reciprocity for the same single-port coil, T/A
        direct = field(conductor, probes)
        interpolated = np.stack(
            [
                RegularGridInterpolator(
                    axes, tx[:, j].reshape(*(len(a) for a in axes))
                )(probes)
                for j in range(3)
            ],
            axis=1,
        )
        scale = np.sqrt(np.mean(np.sum(direct**2, axis=1)))
        surrogate = (
            np.sqrt(np.mean(np.sum((interpolated - direct) ** 2, axis=1))) / scale
        )
        refined = field(fine[name], probes)
        path_error = np.sqrt(np.mean(np.sum((refined - direct) ** 2, axis=1))) / scale
        network = extract_impedance_surface(
            conductor, [3.1024e6, 3.3043e6], n_perimeter=8, formulation="chain"
        )
        cap = self_capacitance(conductor)
        metrics = []
        voxel_arrays = {}
        for pose in (-0.10, 0.0, 0.10):
            for index, region in enumerate(regions):
                locations, masses = region.voxels(translation=(0.0, 0.0, pose))
                beta = field(conductor, locations)
                norm = np.linalg.norm(beta, axis=1)
                metrics.append(
                    {
                        "material": region.material,
                        "translation_z_m": pose,
                        "mass_kg": float(masses.sum()),
                        "density_kg_m3": region.mass_kg / np.prod(region.size_m),
                        "minimum_beta_t_a": float(norm.min()),
                        "median_beta_t_a": float(np.median(norm)),
                        "mass_weighted_beta_squared_t2_a2": float(
                            np.average(norm**2, weights=masses)
                        ),
                        "interpretation": "same-port small-tip coupling proxy, not finite-pulse SNR or optimized receive weights",
                    }
                )
                key = f"r{index}_z{pose}"
                voxel_arrays[key + "_points_m"] = locations
                voxel_arrays[key + "_mass_kg"] = masses
                voxel_arrays[key + "_beta_t_a"] = beta
        np.savez_compressed(
            output / (name + ".npz"),
            grid_points_m=points,
            tx_t_a=tx,
            rx_t_a=rx,
            **voxel_arrays,
        )
        report["candidates"][name] = {
            "wire_vertices": len(conductor.path_points),
            "surrogate_normalized_rms_error": float(surrogate),
            "field_path_refinement_error": float(path_error),
            "frequencies_hz": network.frequency.tolist(),
            "inductance_h": network.inductance.tolist(),
            "resistance_ohm": network.resistance.tolist(),
            "self_capacitance_f": float(cap),
            "peec_scope": "coarse 8-perimeter chain SIBC; resistance and proximity not converged",
            "region_pose_metrics": metrics,
            "loading_sweep": loading_sweep(
                network.inductance[-1],
                network.resistance[-1],
                cap,
                network.frequency[-1],
            ),
        }
        print(name, "surrogate", surrogate, "path", path_error, flush=True)
    (output / "aperture_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    run(Path(".tmp/nqr_phase2"))
