"""Build the NQR Spectra Explorer as a static site for GitHub Pages.

The explorer normally runs as a local server (``app/explorer_server.py``).
This script precomputes every answer that server gives: ``data/stats.json``,
``data/options.json``, ``data/index.json`` (one search row per compound plus
the fields the search reads) and ``data/compounds/<file>.json`` (the detail
view). ``static-api.js`` then answers the front end's requests from those
files, so the same ``index.html``/``app.js`` work without a server.

Only the canonical tables reach the site; the Landolt review queue and page
extracts stay in the repository. ``original_record`` source blobs are dropped
because the front end does not show them.

Usage (from the repository root)::

    python NQRDatabase/scripts/build_static_explorer.py --out PythonSpinDynamics/site/nqr
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT / "app" / "explorer_static"
sys.path.insert(0, str(PROJECT / "app"))

import explorer_server as server  # noqa: E402

STATIC_FILES = ("app.js", "static-api.js", "styles.css")
DROPPED_FIELDS = frozenset({"original_record"})
_HTML_REWRITES = (
    ('<link rel="stylesheet" href="/static/styles.css">', '<link rel="stylesheet" href="styles.css">'),
    (
        '<script src="/static/app.js"></script>',
        '<script src="static-api.js"></script>\n    <script src="app.js"></script>',
    ),
)


def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key not in DROPPED_FIELDS}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def compound_file_name(compound_id: str, used: set[str]) -> str:
    """Return a unique, URL-safe JSON file name for a compound id."""

    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", compound_id).strip("-.") or "compound"
    candidate, suffix = stem, 2
    while candidate.lower() in used:
        candidate, suffix = f"{stem}-{suffix}", suffix + 1
    used.add(candidate.lower())
    return f"{candidate}.json"


def index_entries() -> list[dict[str, Any]]:
    """Return one entry per compound: its search row and the searched fields."""

    rows = server.compound_rows([], [], None)
    aliases: dict[str, list[str]] = defaultdict(list)
    frequencies: dict[str, list[float]] = defaultdict(list)
    with server.connect() as conn:
        notes = {row["id"]: row["notes"] for row in conn.execute("SELECT id, notes FROM compounds")}
        for row in conn.execute("SELECT compound_id, alias FROM compound_aliases ORDER BY alias"):
            aliases[row["compound_id"]].append(row["alias"])
        for row in conn.execute(
            """
            SELECT samples.compound_id, lines.frequency_khz
            FROM lines
            JOIN sites ON sites.id = lines.site_id
            JOIN samples ON samples.id = sites.sample_id
            WHERE lines.frequency_khz IS NOT NULL
            ORDER BY lines.frequency_khz
            """
        ):
            frequencies[row["compound_id"]].append(row["frequency_khz"])
    used: set[str] = set()
    return [
        {
            "row": row,
            "file": compound_file_name(row["id"], used),
            "notes": notes.get(row["id"]),
            "aliases": aliases.get(row["id"], []),
            "frequencies_khz": frequencies.get(row["id"], []),
        }
        for row in rows
    ]


def _write_json(path: Path, payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def static_index_html() -> str:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    for old, new in _HTML_REWRITES:
        if html.count(old) != 1:
            raise RuntimeError(f"index.html no longer contains {old!r} exactly once")
        html = html.replace(old, new)
    return html


def build(out_dir: Path) -> dict[str, int]:
    """Write the static explorer into ``out_dir`` (replaced if present)."""

    if out_dir.exists():
        shutil.rmtree(out_dir)
    compounds_dir = out_dir / "data" / "compounds"
    compounds_dir.mkdir(parents=True)
    for name in STATIC_FILES:
        shutil.copyfile(STATIC_DIR / name, out_dir / name)
    (out_dir / "index.html").write_text(static_index_html(), encoding="utf-8")

    data_dir = out_dir / "data"
    total = _write_json(data_dir / "stats.json", server.explorer_stats())
    total += _write_json(data_dir / "options.json", server.source_options())
    entries = index_entries()
    total += _write_json(data_dir / "index.json", entries)
    for entry in entries:
        detail = server.compound_detail(entry["row"]["id"])
        total += _write_json(compounds_dir / entry["file"], _strip(detail))
    return {"compounds": len(entries), "data_bytes": total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT / "build" / "explorer",
        help="output directory (replaced if it exists)",
    )
    args = parser.parse_args(argv)
    summary = build(args.out)
    print(
        f"wrote {summary['compounds']} compounds "
        f"({summary['data_bytes'] / 1e6:.1f} MB of JSON) to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
