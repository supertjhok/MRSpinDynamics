#!/usr/bin/env python3
"""Print the CHANGELOG.md section for one version, for use as release notes.

Usage:
    python scripts/extract_changelog.py 0.1.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def extract(version: str) -> str:
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    # Match "## [0.1.0]" optionally followed by " - date".
    header = re.compile(rf"^##\s+\[{re.escape(version)}\]")
    next_header = re.compile(r"^##\s+\[")
    out: list[str] = []
    capturing = False
    for line in lines:
        if header.match(line):
            capturing = True
            continue
        if capturing and next_header.match(line):
            break
        if capturing:
            out.append(line)
    body = "\n".join(out).strip()
    if not body:
        raise SystemExit(f"no changelog section found for version {version!r}")
    return body


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_changelog.py <version>")
    print(extract(sys.argv[1]))
