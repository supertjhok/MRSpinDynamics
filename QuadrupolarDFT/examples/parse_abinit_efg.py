"""Parse ABINIT EFG output and print compact quadrupolar summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

from quadrupolar_dft import parse_abinit_efg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="ABINIT .abo/.out file")
    args = parser.parse_args()

    records = parse_abinit_efg(args.output.read_text(encoding="utf-8"))
    for record in records:
        print(
            f"atom={record.atom_index} typat={record.typat} "
            f"Q={record.quadrupole_moment_barns:g} barn "
            f"Cq={record.cq_mhz:g} MHz eta={record.eta:g}"
        )


if __name__ == "__main__":
    main()
