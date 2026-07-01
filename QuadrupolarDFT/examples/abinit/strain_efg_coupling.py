"""Homogeneous-strain EFG coupling workflow for piezoelectric NQR.

This staged script estimates the strain-to-quadrupolar-drive coefficient from
DFT rather than guessing it.  It does not run ABINIT itself; it writes the ABINIT
inputs, then collects completed ``.abo`` outputs.

Typical glycine workflow
------------------------
  # 0. Confirm that the CIF is a non-centrosymmetric glycine polymorph.
  python examples/abinit/strain_efg_coupling.py check

  # 1. Prepare a starter static EFG input from the bundled glycine CIF.
  python examples/abinit/strain_efg_coupling.py prepare-base \\
      --out runs/glycine_static/glycine_efg.abi

  # 2. Run ABINIT on runs/glycine_static/glycine_efg.abi and converge settings.

  # 3. Starting from that static EFG input, write +/- strain jobs.
  python examples/abinit/strain_efg_coupling.py generate \\
      --base runs/glycine_static/glycine_efg.abi \\
      --target-atom-index 1 \\
      --out runs/glycine_strain

  # 4. Run ABINIT on every runs/glycine_strain/*.abi, producing matching .abo.

  # 5. Collect EFG tensors and project dV/depsilon onto NQR transitions.
  python examples/abinit/strain_efg_coupling.py collect \\
      --workdir runs/glycine_strain \\
      --quadmom 0.02044 \\
      --strain-peak 1e-5

Atom indexing: ``--target-atom-index`` is zero-based, matching the expanded
Python structure arrays.  ``prepare-base`` prints the nitrogen candidates; for
the bundled glycine CIF they are 1 and 11.  ABINIT output atom numbers are
one-based; the collector adds one internally.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from quadrupolar_dft import (
    cif_structure_metadata,
    collect_strain_derivatives,
    generate_strain_jobs,
    glycine_static_efg_input_from_cif,
    parse_abinit_structure,
    space_group_is_likely_centrosymmetric,
    strain_transition_couplings,
    write_strain_jobs,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GLYCINE_CIF = (
    REPO_ROOT / "QuadrupolarDFT" / "structures" / "Glycine" / "189379.cif"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check CIF polymorph metadata.")
    check.add_argument("--cif", type=Path, default=DEFAULT_GLYCINE_CIF)
    check.set_defaults(func=cmd_check)

    prepare = subparsers.add_parser(
        "prepare-base",
        help="Write a starter glycine static EFG ABINIT input from the CIF.",
    )
    prepare.add_argument("--cif", type=Path, default=DEFAULT_GLYCINE_CIF)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--ecut", type=float, default=25.0)
    prepare.add_argument("--pawecutdg", type=float, default=50.0)
    prepare.add_argument(
        "--pseudo-dir",
        default="Pseudodojo_paw_pw_standard",
        help="Subdirectory under ABI_PSPDIR containing H/C/N/O PAW XML files.",
    )
    prepare.add_argument(
        "--pawovlp",
        type=float,
        default=None,
        help=(
            "Optional ABINIT PAW sphere overlap allowance in percent. "
            "Use deliberately; large values can make PAW results unreliable."
        ),
    )
    prepare.add_argument(
        "--ngkpt",
        default="2,2,2",
        help="Comma-separated k-point grid for the starter input.",
    )
    prepare.set_defaults(func=cmd_prepare_base)

    generate = subparsers.add_parser(
        "generate",
        help="Write equilibrium and +/- homogeneous-strain ABINIT EFG inputs.",
    )
    generate.add_argument("--base", type=Path, required=True,
                          help="Converged static EFG ABINIT input.")
    generate.add_argument("--out", type=Path, required=True,
                          help="Output directory for strained jobs.")
    generate.add_argument("--cif", type=Path, default=DEFAULT_GLYCINE_CIF,
                          help="CIF used to verify the glycine polymorph.")
    generate.add_argument("--target-atom-index", type=int, required=True,
                          help="Zero-based target nucleus index in the ABINIT input.")
    generate.add_argument("--strain", type=float, default=1.0e-3,
                          help="Central-difference strain amplitude.")
    generate.add_argument(
        "--components",
        default="xx,yy,zz,yz,xz,xy",
        help="Comma-separated strain components to stage.",
    )
    generate.add_argument(
        "--allow-centrosymmetric",
        action="store_true",
        help="Do not abort if the CIF space group is likely centrosymmetric.",
    )
    generate.set_defaults(func=cmd_generate)

    collect = subparsers.add_parser(
        "collect",
        help="Collect ABINIT outputs and compute transition drive couplings.",
    )
    collect.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="Directory containing strain_manifest.json and .abo files.",
    )
    collect.add_argument("--quadmom", type=float, default=0.02044,
                         help="Target quadrupole moment in barns, e.g. 14N = 0.02044.")
    collect.add_argument("--spin", type=float, default=1.0)
    collect.add_argument("--suffix", default=".abo",
                         help="ABINIT output suffix for each job.")
    collect.add_argument("--strain-peak", type=float, default=1.0e-5,
                         help="Peak acoustic strain used for Rabi-rate estimates.")
    collect.add_argument("--json", type=Path,
                         help="Optional JSON output path.")
    collect.add_argument("--csv", type=Path,
                         help="Optional CSV output path.")
    collect.set_defaults(func=cmd_collect)

    args = parser.parse_args()
    args.func(args)


def cmd_check(args) -> None:
    metadata = cif_structure_metadata(args.cif)
    _print_structure_check(metadata)


def cmd_prepare_base(args) -> None:
    metadata = cif_structure_metadata(args.cif)
    _print_structure_check(metadata)
    if space_group_is_likely_centrosymmetric(metadata.get("space_group")):
        raise SystemExit(
            "Refusing to prepare a piezoelectric glycine base input from a "
            "likely centrosymmetric structure."
        )
    ngkpt = tuple(int(item.strip()) for item in args.ngkpt.split(","))
    if len(ngkpt) != 3:
        raise SystemExit("--ngkpt must contain exactly three comma-separated ints")
    text, atoms = glycine_static_efg_input_from_cif(
        args.cif,
        ecut=args.ecut,
        pawecutdg=args.pawecutdg,
        ngkpt=ngkpt,
        pseudo_dir=args.pseudo_dir,
        pawovlp=args.pawovlp,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote starter glycine static EFG input: {args.out}")
    print("Nitrogen target candidates, zero-based:")
    for index, atom in enumerate(atoms):
        if atom["element"] != "N":
            continue
        frac = atom["fractional"]
        print(
            f"  {index}: {atom['label']} sym={atom['symmetry']} "
            f"xred=({frac[0]:.6f}, {frac[1]:.6f}, {frac[2]:.6f})"
        )
    print("Next:")
    print(f"  bash examples/abinit/run_static_efg_wsl.sh {args.out}")
    print("or run ABINIT manually on that .abi before strain generation.")


def cmd_generate(args) -> None:
    metadata = cif_structure_metadata(args.cif)
    _print_structure_check(metadata)
    if (
        space_group_is_likely_centrosymmetric(metadata.get("space_group"))
        and not args.allow_centrosymmetric
    ):
        raise SystemExit(
            "Refusing to stage piezoelectric strain jobs for a likely "
            "centrosymmetric structure. Pass --allow-centrosymmetric only for "
            "a deliberate negative/control calculation."
        )

    base = args.base.read_text(encoding="utf-8")
    crystal = parse_abinit_structure(base)
    components = tuple(
        item.strip() for item in args.components.split(",") if item.strip()
    )
    jobs = generate_strain_jobs(
        crystal,
        strain_amplitude=args.strain,
        components=components,
    )
    directory = write_strain_jobs(
        base,
        jobs,
        args.out,
        target_atom_index=args.target_atom_index,
        structure_check=metadata,
    )
    print(f"Wrote {len(jobs)} ABINIT inputs to {directory}")
    print(f"Wrote manifest: {directory / 'strain_manifest.json'}")
    print("Next: run ABINIT on each .abi so each job has a matching .abo file.")


def cmd_collect(args) -> None:
    manifest_path = args.workdir / "strain_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    equilibrium, derivatives = collect_strain_derivatives(
        manifest,
        args.workdir,
        output_suffix=args.suffix,
    )
    couplings = strain_transition_couplings(
        equilibrium,
        derivatives,
        spin=args.spin,
        quadrupole_moment_barns=args.quadmom,
    )
    rows = _coupling_rows(couplings, strain_peak=args.strain_peak)
    _print_collect_summary(equilibrium, derivatives, rows, args.strain_peak)
    if args.json:
        _write_json(args.json, manifest, equilibrium, derivatives, rows)
        print(f"Wrote JSON: {args.json}")
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"Wrote CSV: {args.csv}")


def _print_structure_check(metadata: dict) -> None:
    space_group = metadata.get("space_group")
    centrosymmetric = space_group_is_likely_centrosymmetric(space_group)
    print("Structure check")
    print(f"  CIF: {metadata.get('path')}")
    print(f"  name: {metadata.get('chemical_name_common')}")
    print(f"  formula: {metadata.get('chemical_formula_sum')}")
    print(f"  CCDC: {metadata.get('ccdc')}")
    print(f"  space group: {space_group}")
    print(f"  likely centrosymmetric: {centrosymmetric}")
    if not centrosymmetric:
        print("  piezoelectricity: symmetry-allowed")
    else:
        print("  piezoelectricity: bulk effect symmetry-forbidden/control case")


def _print_collect_summary(equilibrium, derivatives, rows, strain_peak: float) -> None:
    print("Strain-to-EFG collection")
    print("  equilibrium EFG principal components (1e21 V/m^2):")
    print(
        "   ",
        " ".join(
            f"{value / 1e21: .6g}"
            for value in equilibrium.principal_components_si
        ),
    )
    print(f"  equilibrium eta: {equilibrium.eta:.6g}")
    print(f"  strain derivatives: {len(derivatives)}")
    print(f"  transition couplings for peak strain {strain_peak:.3e}:")
    for row in rows:
        print(
            f"    {row['component']:>2s} "
            f"{row['frequency_hz'] / 1e3:9.3f} kHz "
            f"cos={row['cosine_hz_per_strain']:.3e} Hz/strain "
            f"Rabi={row['rabi_hz_at_strain']:.3e} Hz"
        )


def _coupling_rows(couplings, *, strain_peak: float) -> list[dict]:
    rows = []
    for item in couplings:
        rows.append(
            {
                "component": item.component,
                "lower": item.lower,
                "upper": item.upper,
                "frequency_hz": item.frequency_hz,
                "cosine_hz_per_strain": item.cosine_hz_per_strain,
                "rwa_rabi_hz_per_strain": item.rwa_rabi_hz_per_strain,
                "strain_peak": strain_peak,
                "rabi_hz_at_strain": item.rwa_rabi_hz_per_strain * strain_peak,
            }
        )
    return sorted(rows, key=lambda row: (row["component"], row["frequency_hz"]))


def _write_json(path: Path, manifest: dict, equilibrium, derivatives, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": manifest,
        "equilibrium_efg_si": equilibrium.matrix_si.tolist(),
        "equilibrium_eta": equilibrium.eta,
        "derivatives": [
            {
                "component": item.component,
                "strain_amplitude": item.strain_amplitude,
                "derivative_si_per_strain": item.derivative_si_per_strain.tolist(),
            }
            for item in derivatives
        ],
        "couplings": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "component",
        "lower",
        "upper",
        "frequency_hz",
        "cosine_hz_per_strain",
        "rwa_rabi_hz_per_strain",
        "strain_peak",
        "rabi_hz_at_strain",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
