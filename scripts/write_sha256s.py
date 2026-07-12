#!/usr/bin/env python3
"""Write stable SHA-256 checksums for every regular file in a directory."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def checksum_lines(directory: Path, output: Path) -> list[str]:
    """Return sorted ``sha256  filename`` lines, excluding ``output`` itself."""

    root = directory.resolve()
    destination = output.resolve()
    lines: list[str] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.resolve() == destination:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.name}")
    if not lines:
        raise ValueError(f"no files to checksum in {root}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: DIRECTORY/SHA256SUMS)",
    )
    args = parser.parse_args(argv)
    output = args.output or args.directory / "SHA256SUMS"
    try:
        lines = checksum_lines(args.directory, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="ascii")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"wrote {output} ({len(lines)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
