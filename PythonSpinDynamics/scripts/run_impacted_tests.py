"""Run tests selected from changed files and a declarative impact map.

Examples::

    python scripts/run_impacted_tests.py
    python scripts/run_impacted_tests.py --group nqr
    python scripts/run_impacted_tests.py --list
    python scripts/run_impacted_tests.py --full

The full suite remains the CI/release gate. This runner is the short feedback
loop for local design changes.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
CONFIG_PATH = PROJECT_ROOT / "tests" / "test_groups.json"


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _git_lines(*arguments: str) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "-C",
            str(REPOSITORY_ROOT),
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in completed.stdout.splitlines()]


def changed_project_paths(base: str = "HEAD") -> tuple[str, ...]:
    """Return tracked and untracked PythonSpinDynamics paths relative to it."""

    prefix = f"{PROJECT_ROOT.name}/"
    tracked = _git_lines(
        "diff",
        "--name-only",
        "--diff-filter=ACMR",
        base,
        "--",
        PROJECT_ROOT.name,
    )
    untracked = _git_lines(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        PROJECT_ROOT.name,
    )
    paths = {
        path[len(prefix) :] if path.startswith(prefix) else path
        for path in (*tracked, *untracked)
        if path
    }
    return tuple(sorted(paths))


def select_groups(
    paths: tuple[str, ...] | list[str],
    config: dict,
) -> tuple[str, ...]:
    """Return impact groups whose path globs match at least one changed file."""

    selected = []
    for name, details in config["groups"].items():
        patterns = details.get("paths", [])
        if any(
            fnmatch.fnmatch(path, pattern)
            for path in paths
            for pattern in patterns
        ):
            selected.append(name)
    return tuple(selected)


def select_test_modules(
    paths: tuple[str, ...] | list[str],
    groups: tuple[str, ...] | list[str],
    config: dict,
    *,
    include_smoke: bool = True,
) -> tuple[str, ...]:
    """Return de-duplicated unittest modules for paths and impact groups."""

    modules: list[str] = []
    if include_smoke:
        modules.extend(config.get("always", []))
    for group in groups:
        if group not in config["groups"]:
            raise ValueError(f"unknown test group: {group}")
        modules.extend(config["groups"][group].get("tests", []))
    for path in paths:
        if path.startswith("tests/test_") and path.endswith(".py"):
            modules.append(path[:-3].replace("/", "."))
    return tuple(dict.fromkeys(modules))


def changed_examples(paths: tuple[str, ...] | list[str]) -> tuple[Path, ...]:
    """Return changed plotting/example scripts for targeted CLI checks."""

    return tuple(
        PROJECT_ROOT / path
        for path in paths
        if path.startswith("examples/") and path.endswith(".py")
    )


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    started = time.perf_counter()
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"  completed in {time.perf_counter() - started:.2f} s", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="Git comparison base")
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--list", action="store_true", dest="list_only")
    parser.add_argument("--full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.full:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests"])
        return

    config = _load_config()
    paths = changed_project_paths(args.base)
    automatic_groups = select_groups(paths, config)
    groups = tuple(dict.fromkeys((*automatic_groups, *args.group)))
    modules = select_test_modules(
        paths,
        groups,
        config,
        include_smoke=not args.no_smoke,
    )
    examples = changed_examples(paths)

    print("Changed paths:")
    for path in paths:
        print(f"  {path}")
    print("Impact groups:", ", ".join(groups) if groups else "fallback smoke")
    print("Test modules:", " ".join(modules) if modules else "none")
    for example in examples:
        print(f"Example CLI check: {example.relative_to(PROJECT_ROOT)} --help")
    if args.list_only:
        return

    if modules:
        _run([sys.executable, "-m", "unittest", *modules])
    for example in examples:
        _run([sys.executable, str(example), "--help"])


if __name__ == "__main__":
    main()
