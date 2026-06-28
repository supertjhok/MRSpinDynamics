"""In-process plumbing smoke test for the finite-displacement chain (no ABINIT).

For the real workflow that runs ABINIT, use the three-stage CLI
``examples/abinit/efg_temperature.py`` (phonon -> displace -> collect); this
script only exercises the in-process plumbing with a synthetic EFG model so the
chain can run in CI without a DFT code.

Shows how a finite-displacement DFT temperature calculation is wired together:

1. parse the equilibrium structure from a converged ABINIT input;
2. generate one displaced ABINIT input per +/- mode displacement plus a manifest
   (this is what you would run locally with ABINIT);
3. collect the per-job EFG of the target nucleus and central-difference into
   mode curvatures;
4. run the harmonic temperature sweep -> C_Q(T), eta(T), nu(T).

ABINIT is not run here -- step 3 uses a synthetic EFG-vs-geometry model so the
plumbing runs end-to-end. In a real workflow, replace ``synthetic_efg`` with
``parse_abinit_efg`` on each job's output and select the target atom.

Run:
    python examples/finite_displacement_workflow.py
"""

from __future__ import annotations

import numpy as np

from quadrupolar_dft import (
    EFGTensor,
    PhononMode,
    coupling_constant_hz,
    efg_temperature_sweep,
    generate_displacement_jobs,
    nqr_frequencies_hz,
    parse_abinit_structure,
    vibrational_modes_from_efg,
)

BASE_ABI = "examples/abinit/nano2_efg.abi"
TARGET_ATOM = 2  # first nitrogen (0-based)
Q_BARN = 0.02044  # 14N

# Synthetic equilibrium EFG and curvature for the target N (stand-in for DFT).
_V_EQ = np.diag([-1.4e21, -2.6e21, 4.0e21])
_CURVATURE = np.diag([0.6 * 5.0e68, 0.4 * 5.0e68, -5.0e68])


def synthetic_efg(normal_coordinate: float) -> EFGTensor:
    """Stand in for a DFT EFG at a displaced geometry (quadratic in Q)."""

    matrix = _V_EQ + 0.5 * _CURVATURE * normal_coordinate**2
    return EFGTensor.from_components(matrix, unit="si")


def main() -> None:
    crystal = parse_abinit_structure(open(BASE_ABI, encoding="utf-8").read())
    print(f"Parsed {crystal.natom} atoms; target = atom {TARGET_ATOM} "
          f"(Z={crystal.species_z[TARGET_ATOM]})")

    # One librational mode of the two nitrite N atoms (a DFT phonon calculation
    # would supply the real frequency and eigenvector).
    eigenvector = np.zeros((crystal.natom, 3))
    eigenvector[2, 1] = 1.0
    eigenvector[3, 1] = -1.0
    mode = PhononMode(210.0, eigenvector, label="NO2- libration")

    jobs = generate_displacement_jobs(crystal, [mode], max_displacement_angstrom=0.04)
    print(f"Generated {len(jobs)} jobs: {[j.name for j in jobs]}")
    print("  (write_jobs(base_input, jobs, out_dir, target_atom_index=...) would")
    print("   emit one .abi per job plus manifest.json to run with ABINIT.)")

    # Step 3: collect EFGs. Here from the synthetic model; normally from ABINIT.
    efg_by_job = {
        job.name: synthetic_efg(job.sign * job.delta_q_si) for job in jobs
    }
    modes = vibrational_modes_from_efg([mode], jobs, efg_by_job)
    print(f"Recovered {len(modes)} mode curvature(s) by central difference.")

    equilibrium = efg_by_job["equilibrium"]
    points = efg_temperature_sweep(
        equilibrium, modes, [0.0, 77.0, 200.0, 300.0, 400.0],
        spin=1.0, quadrupole_moment_barns=Q_BARN,
    )
    cq0 = coupling_constant_hz(equilibrium.vzz_si, Q_BARN)
    static_nu = np.sort(nqr_frequencies_hz(spin=1.0, cq_hz=cq0, eta=equilibrium.eta))[-1]
    print(f"\n  static: C_Q={cq0 / 1e6:.4f} MHz  nu_+={static_nu / 1e6:.4f} MHz")
    print("    T(K)   C_Q(MHz)   eta      nu_+(MHz)")
    for point in points:
        nu_plus = np.sort(point.frequencies_hz)[-1]
        print(
            f"   {point.temperature_k:5.0f}   {point.cq_hz / 1e6:7.4f}  "
            f"{point.eta:6.4f}   {nu_plus / 1e6:8.4f}"
        )


if __name__ == "__main__":
    main()
