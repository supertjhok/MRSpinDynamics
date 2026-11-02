"""Build ``docs/user_manual.pdf`` from ``user_manual.tex``.

The PDF is not tracked in git: CI builds it for the documentation site and the
release assets, and contributors build it locally with this script. It runs
``pdflatex`` directly (no ``latexmk``, so no Perl), repeating until the table
of contents and cross-references settle.

Usage::

    python docs/build_manual.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys


DOCS_DIR = Path(__file__).resolve().parent
SOURCE = "user_manual.tex"
MAX_PASSES = 4


def main() -> None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        sys.exit("pdflatex not found; install TeX Live or MiKTeX")
    log = DOCS_DIR / "user_manual.log"
    for attempt in range(1, MAX_PASSES + 1):
        completed = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", SOURCE],
            cwd=DOCS_DIR,
            stdout=subprocess.DEVNULL,
        )
        text = log.read_text(encoding="latin-1")
        if completed.returncode != 0:
            errors = [line for line in text.splitlines() if line.startswith("!")]
            sys.exit(f"pdflatex failed on pass {attempt}: {errors[:5]}\nsee {log}")
        # Two passes are always needed for a fresh table of contents.
        if attempt >= 2 and "Rerun to get" not in text and "Rerun LaTeX" not in text:
            break
    else:
        sys.exit(f"cross-references did not settle after {MAX_PASSES} passes")
    pdf = DOCS_DIR / "user_manual.pdf"
    print(f"wrote {pdf.relative_to(DOCS_DIR.parent)} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
