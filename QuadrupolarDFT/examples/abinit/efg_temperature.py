"""End-to-end finite-displacement EFG temperature workflow (real ABINIT runs).

Three stages, with a DFT run between each:

  1. phonon   -- write the DFPT phonon + anaddb inputs from a converged static
                 EFG input.            [then run ABINIT DFPT, then anaddb]
  2. displace -- parse the anaddb eigenvectors, write one displaced EFG input per
                 +/- mode + a manifest. [then run ABINIT EFG on every input]
  3. collect  -- parse the displaced EFG outputs, central-difference the mode
                 curvatures, and report C_Q(T), eta(T), nu(T) (and dnu/dT).

No synthetic data: every number comes from your ABINIT outputs.

Examples
--------
  python efg_temperature.py phonon  --base nano2_efg.abi --out runs/nano2_ph
  # run ABINIT on runs/nano2_ph/phonon.abi, then anaddb -> runs/nano2_ph/anaddb.out

  python efg_temperature.py displace --base nano2_efg.abi \\
      --anaddb runs/nano2_ph/anaddb.out --target 2 --max-modes 6 --out runs/nano2_disp
  # run ABINIT EFG on every runs/nano2_disp/*.abi (see run_finite_displacement_wsl.sh)

  python efg_temperature.py collect  --workdir runs/nano2_disp \\
      --temperatures 0,77,150,300 --quadmom 0.02044
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from quadrupolar_dft import (
    abinit_input_with_positions,
    anaddb_input,
    coupling_constant_hz,
    collect_efg_outputs,
    efg_temperature_sweep,
    generate_displacement_jobs,
    manifest_dict,
    modes_from_arrays,
    nqr_frequencies_hz,
    parse_abinit_structure,
    parse_anaddb_modes,
    phonon_dfpt_input,
    PhononMode,
    vibrational_modes_from_collected,
)


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def cmd_phonon(args):
    base = _read(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phonon.abi").write_text(phonon_dfpt_input(base), encoding="utf-8")
    (out / "anaddb.abi").write_text(anaddb_input(), encoding="utf-8")
    print(f"Wrote {out/'phonon.abi'} and {out/'anaddb.abi'}")
    print(f"Next: bash run_phonon_wsl.sh {out}")
    print("      (runs ABINIT DFPT then anaddb; writes anaddb.out with the modes)")


def _load_modes(args, crystal):
    masses = crystal.masses_amu
    if args.modes:
        data = json.loads(_read(args.modes))
        return modes_from_arrays(
            data["wavenumbers_cm_inv"],
            np.array(data["eigendisplacements"]),
            masses,
            mass_weighted=data.get("mass_weighted", False),
        )
    return parse_anaddb_modes(_read(args.anaddb), masses, natom=crystal.natom)


def cmd_displace(args):
    base = _read(args.base)
    crystal = parse_abinit_structure(base)
    modes = _load_modes(args, crystal)
    if args.max_modes:
        modes = sorted(modes, key=lambda m: m.wavenumber_cm_inv)[: args.max_modes]
    jobs = generate_displacement_jobs(
        crystal, modes, max_displacement_angstrom=args.max_displacement
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        (out / f"{job.name}.abi").write_text(
            abinit_input_with_positions(base, job.crystal), encoding="utf-8"
        )
    manifest = manifest_dict(jobs, target_atom_index=args.target)
    manifest["modes"] = [
        {"wavenumber_cm_inv": m.wavenumber_cm_inv, "eigenvector": m.eigenvector.tolist()}
        for m in modes
    ]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(jobs)} EFG inputs and manifest.json to {out}")
    print(f"  {len(modes)} mode(s); run ABINIT EFG on every *.abi, outputs as <name>.abo")


def cmd_collect(args):
    workdir = Path(args.workdir)
    manifest = json.loads((workdir / "manifest.json").read_text(encoding="utf-8"))
    modes = [
        PhononMode(m["wavenumber_cm_inv"], np.array(m["eigenvector"]))
        for m in manifest["modes"]
    ]
    efg_by_job = collect_efg_outputs(
        manifest, workdir, output_suffix=args.suffix
    )
    vib_modes = vibrational_modes_from_collected(modes, manifest, efg_by_job)
    temperatures = [float(t) for t in args.temperatures.split(",")]
    equilibrium = efg_by_job["equilibrium"]
    points = efg_temperature_sweep(
        equilibrium, vib_modes, temperatures,
        spin=args.spin, quadrupole_moment_barns=args.quadmom,
    )
    cq0 = coupling_constant_hz(equilibrium.vzz_si, args.quadmom)
    static_nu = np.sort(nqr_frequencies_hz(spin=args.spin, cq_hz=cq0, eta=equilibrium.eta))
    print(f"static (no vibration): C_Q={cq0/1e6:.4f} MHz  eta={equilibrium.eta:.4f}")
    print(f"  lines (MHz): {[round(float(x)/1e6, 4) for x in static_nu]}")
    print("   T(K)   C_Q(MHz)   eta      lines(MHz)")
    for point in points:
        lines = ", ".join(f"{float(x)/1e6:.4f}" for x in np.sort(point.frequencies_hz))
        print(
            f"  {point.temperature_k:5.0f}   {point.cq_hz/1e6:7.4f}  "
            f"{point.eta:6.4f}   {lines}"
        )
    if len(temperatures) >= 2:
        low, high = points[0], points[-1]
        span = high.temperature_k - low.temperature_k
        lo = np.sort(low.frequencies_hz)
        hi = np.sort(high.frequencies_hz)
        print(f"  dnu/dT over {low.temperature_k:.0f}-{high.temperature_k:.0f} K (kHz/K):")
        for i in range(min(lo.size, hi.size)):
            print(f"    line {lo[i]/1e6:.4f} MHz : {(hi[i]-lo[i])/span/1e3:+.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("phonon", help="write DFPT phonon + anaddb inputs")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_phonon)

    d = sub.add_parser("displace", help="write displaced EFG inputs from modes")
    d.add_argument("--base", required=True)
    d.add_argument("--anaddb", help="anaddb output with eigenvectors")
    d.add_argument("--modes", help="modes JSON (alternative to --anaddb)")
    d.add_argument("--target", type=int, required=True, help="0-based target atom")
    d.add_argument("--max-modes", type=int, default=0, help="limit to N lowest modes")
    d.add_argument("--max-displacement", type=float, default=0.04)
    d.add_argument("--out", required=True)
    d.set_defaults(func=cmd_displace)

    c = sub.add_parser("collect", help="parse EFG outputs -> nu(T)")
    c.add_argument("--workdir", required=True)
    c.add_argument("--temperatures", default="0,77,150,300")
    c.add_argument("--spin", type=float, default=1.0)
    c.add_argument("--quadmom", type=float, default=0.02044, help="barns")
    c.add_argument("--suffix", default=".abo")
    c.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
