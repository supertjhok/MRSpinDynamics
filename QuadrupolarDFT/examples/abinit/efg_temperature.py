"""End-to-end finite-displacement EFG temperature workflow (real ABINIT runs).

An optional relaxation stage, then three stages with a DFT run between each:

  0. relax    -- write an ionic-relaxation input from a converged static EFG
                 input.                [then run ABINIT relaxation]
     relax-collect -- read the relaxed geometry from the output, write a relaxed
                 static EFG input to feed the rest of the chain.
  1. phonon   -- write the DFPT phonon + anaddb inputs from a (relaxed) static
                 EFG input.            [then run ABINIT DFPT, then anaddb]
  2. displace -- parse the anaddb eigenvectors, write one displaced EFG input per
                 +/- mode + a manifest. [then run ABINIT EFG on every input]
  3. collect  -- parse the displaced EFG outputs, central-difference the mode
                 curvatures, and report C_Q(T), eta(T), nu(T) (and dnu/dT).

Relaxing first puts the geometry at an energy minimum, which removes spurious
imaginary modes at Gamma and corrects the equilibrium eta. It is optional but
recommended; skip it only if your input is already a relaxed structure.

No synthetic data: every number comes from your ABINIT outputs.

Examples
--------
  python efg_temperature.py relax   --base nano2_efg.abi --out runs/nano2_relax
  # run ABINIT on runs/nano2_relax/relax.abi (see run_relax_wsl.sh), then:
  python efg_temperature.py relax-collect --base nano2_efg.abi \\
      --abo runs/nano2_relax/relax.abo --out runs/nano2_relax
  # -> writes runs/nano2_relax/relaxed.abi; use it as --base below

  python efg_temperature.py phonon  --base runs/nano2_relax/relaxed.abi --out runs/nano2_ph
  # run ABINIT on runs/nano2_ph/phonon.abi, then anaddb -> runs/nano2_ph/anaddb.out

  python efg_temperature.py displace --base runs/nano2_relax/relaxed.abi \\
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
    relax_input,
    relaxed_input,
    vibrational_modes_from_collected,
)


def _read(path):
    return Path(path).read_text(encoding="utf-8")


def cmd_relax(args):
    base = _read(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "relax.abi").write_text(relax_input(base), encoding="utf-8")
    print(f"Wrote {out/'relax.abi'}")
    print(f"Next: bash run_relax_wsl.sh {out}")
    print("      (runs ABINIT ionic relaxation; writes relax.abo with the geometry)")
    print(f"Then: efg_temperature.py relax-collect --base {args.base} "
          f"--abo {out/'relax.abo'} --out {out}")


def cmd_relax_collect(args):
    base = _read(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    relaxed = relaxed_input(base, _read(args.abo))
    (out / "relaxed.abi").write_text(relaxed, encoding="utf-8")
    print(f"Wrote {out/'relaxed.abi'} (static EFG input at the relaxed geometry)")
    print(f"Next: efg_temperature.py phonon --base {out/'relaxed.abi'} --out runs/<ph>")


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

    results = {
        "label": args.label or workdir.name,
        "spin": args.spin,
        "quadmom_barns": args.quadmom,
        "target_atom_index": int(manifest["target_atom_index"]),
        "n_modes": len(modes),
        "mode_wavenumbers_cm_inv": [m.wavenumber_cm_inv for m in modes],
        "static": {
            "cq_mhz": cq0 / 1e6,
            "eta": float(equilibrium.eta),
            "lines_mhz": [float(x) / 1e6 for x in static_nu],
        },
        "points": [
            {
                "temperature_k": point.temperature_k,
                "cq_mhz": point.cq_hz / 1e6,
                "eta": float(point.eta),
                "lines_mhz": [float(x) / 1e6 for x in np.sort(point.frequencies_hz)],
            }
            for point in points
        ],
        "dnu_dt_khz_per_k": [],
    }
    if len(temperatures) >= 2:
        low, high = points[0], points[-1]
        span = high.temperature_k - low.temperature_k
        lo = np.sort(low.frequencies_hz)
        hi = np.sort(high.frequencies_hz)
        results["dnu_dt_span_k"] = [low.temperature_k, high.temperature_k]
        for i in range(min(lo.size, hi.size)):
            results["dnu_dt_khz_per_k"].append(
                {"line_mhz": lo[i] / 1e6, "slope_khz_per_k": (hi[i] - lo[i]) / span / 1e3}
            )

    print(f"static (no vibration): C_Q={cq0/1e6:.4f} MHz  eta={equilibrium.eta:.4f}")
    print(f"  lines (MHz): {[round(x, 4) for x in results['static']['lines_mhz']]}")
    print("   T(K)   C_Q(MHz)   eta      lines(MHz)")
    for point in results["points"]:
        lines = ", ".join(f"{x:.4f}" for x in point["lines_mhz"])
        print(
            f"  {point['temperature_k']:5.0f}   {point['cq_mhz']:7.4f}  "
            f"{point['eta']:6.4f}   {lines}"
        )
    if results["dnu_dt_khz_per_k"]:
        lo_t, hi_t = results["dnu_dt_span_k"]
        print(f"  dnu/dT over {lo_t:.0f}-{hi_t:.0f} K (kHz/K):")
        for entry in results["dnu_dt_khz_per_k"]:
            print(f"    line {entry['line_mhz']:.4f} MHz : {entry['slope_khz_per_k']:+.2f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Wrote results JSON to {args.out_json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("relax", help="write an ionic-relaxation input")
    r.add_argument("--base", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_relax)

    rc = sub.add_parser(
        "relax-collect", help="read relaxed geometry -> write relaxed static input"
    )
    rc.add_argument("--base", required=True)
    rc.add_argument("--abo", required=True, help="ABINIT relaxation output")
    rc.add_argument("--out", required=True)
    rc.set_defaults(func=cmd_relax_collect)

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
    c.add_argument("--label", help="label for the results JSON (default: workdir name)")
    c.add_argument("--out-json", help="also write the results to this JSON file")
    c.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
