#!/usr/bin/env python3
"""Parse an Elk ``EFG.OUT`` into eta, Vzz and C_Q for a target species.

Elk's EFG task (``tasks: 115``) writes the Cartesian electric-field-gradient
tensor and its eigenvalues, in atomic units, for every atom -- grouped by
species in the order they appear in ``elk.in``. This reader extracts the
eigenvalues for one species and reports the NQR parameters, so an Elk
all-electron EFG can be compared directly against the ABINIT PAW pipeline.

Convention matches ``quadrupolar_dft.EFGTensor``: principal components ordered
``|Vzz| >= |Vyy| >= |Vxx|`` and ``eta = |(Vxx - Vyy) / Vzz|``.

Usage:
    python3 parse_elk_efg.py EFG.OUT --species 2 --quadmom 0.02044
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# 1 atomic unit of EFG = e / (4 pi eps0 a0^3) = 9.71736166e21 V/m^2 (Elk manual).
EFG_AU_TO_SI = 9.71736166e21
ELEMENTARY_CHARGE_C = 1.602176634e-19
PLANCK_CONSTANT_J_S = 6.62607015e-34
BARN_M2 = 1.0e-28

_SPECIES_RE = re.compile(
    r"Species\s*:\s*(?P<species>\d+)\s*\((?P<label>[^)]*)\).*?atom\s*:\s*(?P<atom>\d+)"
)
_FLOAT = r"[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?"


def parse_efg_eigenvalues(text: str, species: int) -> list[tuple[int, list[float]]]:
    """Return ``[(atom_index, [ev0, ev1, ev2]), ...]`` for the given species."""

    results: list[tuple[int, list[float]]] = []
    current_species: int | None = None
    current_atom: int | None = None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = _SPECIES_RE.search(line)
        if match is not None:
            current_species = int(match.group("species"))
            current_atom = int(match.group("atom"))
            continue
        if current_species == species and "eigenvalues" in line.lower():
            # Eigenvalues are on the following line.
            values = re.findall(_FLOAT, lines[i + 1] if i + 1 < len(lines) else "")
            if len(values) >= 3:
                evs = [float(v.replace("D", "E").replace("d", "e")) for v in values[:3]]
                results.append((current_atom if current_atom is not None else -1, evs))
    return results


def nqr_parameters(eigenvalues: list[float], quadmom_barns: float) -> dict:
    """eta and C_Q from EFG eigenvalues (atomic units), repo convention."""

    ordered = sorted(eigenvalues, key=abs)  # |Vxx| <= |Vyy| <= |Vzz|
    vxx, vyy, vzz = ordered
    eta = 0.0 if vzz == 0.0 else abs((vxx - vyy) / vzz)
    vzz_si = vzz * EFG_AU_TO_SI
    cq_hz = ELEMENTARY_CHARGE_C * quadmom_barns * BARN_M2 * vzz_si / PLANCK_CONSTANT_J_S
    return {"eta": eta, "vzz_au": vzz, "cq_mhz": cq_hz / 1e6}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("efg_out", help="path to Elk EFG.OUT")
    parser.add_argument("--species", type=int, required=True,
                        help="1-based species number (order in elk.in atoms block)")
    parser.add_argument("--quadmom", type=float, default=0.02044,
                        help="nuclear quadrupole moment in barns (default 14N)")
    parser.add_argument("--atom", type=int, default=None,
                        help="report only this 1-based atom index within the species")
    args = parser.parse_args()

    text = Path(args.efg_out).read_text(encoding="utf-8")
    records = parse_efg_eigenvalues(text, args.species)
    if not records:
        raise SystemExit(f"no EFG record for species {args.species} in {args.efg_out}")

    for atom_index, evs in records:
        if args.atom is not None and atom_index != args.atom:
            continue
        p = nqr_parameters(evs, args.quadmom)
        print(f"species {args.species} atom {atom_index}: "
              f"eta={p['eta']:.4f}  Vzz={p['vzz_au']:+.5f} au  "
              f"C_Q={p['cq_mhz']:+.4f} MHz")


if __name__ == "__main__":
    main()
