"""Check the Python snippets in the user manual and the web documentation.

Static mode (the default, part of the smoke job) parses every snippet and
checks that each ``spin_dynamics`` name it imports exists. ``--run`` also
executes the snippets: each document runs top to bottom in one shared
namespace, like a notebook, inside a temporary working directory, so later
snippets may use names an earlier one defined. A snippet that uses names its
page never defines (an illustrative fragment) is reported as incomplete
rather than failed; ``--strict`` fails those too. Errors of any other kind
fail the check, which is how renamed arguments and attributes are caught.

A snippet that cannot run there (it reads a file the reader supplies, or only
sketches an API) is marked on the line directly above it with a reason::

    % snippet: skip -- reads the reader's Pulseq file        (LaTeX)
    <!-- snippet: skip -- reads the reader's Pulseq file --> (Markdown)

Skipped snippets are still checked statically. Planning documents
(``*_plan.md``) hold design sketches and are not checked.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import os
import re
import signal
import sys
import tempfile
import time
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DOCS = Path(__file__).resolve().parent
MANUAL = DOCS / "user_manual.tex"

_LISTING = re.compile(
    r"^(?P<marker>[^\n]*)\n\\begin\{lstlisting\}\[language=Python\]\n(?P<code>.*?)\\end\{lstlisting\}",
    re.S | re.M,
)
_FENCE = re.compile(
    r"^(?P<marker>[^\n]*)\n```python[^\n]*\n(?P<code>.*?)^```", re.S | re.M
)
_TEX_SKIP = re.compile(r"^\s*%\s*snippet:\s*skip\s*--\s*\S")
_MD_SKIP = re.compile(r"^\s*<!--\s*snippet:\s*skip\s*--\s*\S.*-->\s*$")


@dataclass(frozen=True)
class Snippet:
    path: Path
    line: int
    code: str
    skip: bool

    @property
    def label(self) -> str:
        return f"{self.path.relative_to(DOCS.parent).as_posix()}:{self.line}"


def documents() -> list[Path]:
    markdown = sorted(
        path for path in DOCS.rglob("*.md") if not path.name.endswith("_plan.md")
    )
    return [MANUAL, *markdown]


def snippets(path: Path) -> Iterator[Snippet]:
    text = path.read_text(encoding="utf-8")
    pattern, skip = (_LISTING, _TEX_SKIP) if path.suffix == ".tex" else (_FENCE, _MD_SKIP)
    for match in pattern.finditer(text):
        line = text.count("\n", 0, match.start("code")) + 1
        yield Snippet(path, line, match.group("code"), bool(skip.match(match.group("marker"))))


def _missing_imports(tree: ast.AST) -> list[str]:
    missing: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
            names: list[tuple[str, str]] = []
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules = [node.module]
            names = [(node.module, alias.name) for alias in node.names if alias.name != "*"]
        else:
            continue
        for module in modules:
            if module.split(".")[0] != "spin_dynamics":
                continue
            try:
                importlib.import_module(module)
            except ImportError as exc:
                missing.append(f"{module} ({exc})")
        for module, name in names:
            if module.split(".")[0] != "spin_dynamics":
                continue
            try:
                parent = importlib.import_module(module)
            except ImportError:
                continue
            if hasattr(parent, name):
                continue
            try:
                importlib.import_module(f"{module}.{name}")
            except ImportError:
                missing.append(f"{module}.{name}")
    return missing


def check_static(items: list[Snippet]) -> list[str]:
    failures = []
    for snippet in items:
        try:
            tree = ast.parse(snippet.code)
        except SyntaxError as exc:
            failures.append(f"{snippet.label}: syntax error: {exc.msg} (line {exc.lineno})")
            continue
        for name in _missing_imports(tree):
            failures.append(f"{snippet.label}: cannot import {name}")
    return failures


class _Timeout(Exception):
    pass


def _raise_timeout(signum: int, frame: object) -> None:
    raise _Timeout


_MODULE = "__doc_snippet__"


def _fresh_module() -> types.ModuleType:
    # A registered module, so dataclasses and pickling defined in a snippet work.
    module = types.ModuleType(_MODULE)
    sys.modules[_MODULE] = module
    return module


def _execute(snippet: Snippet, namespace: dict[str, object], timeout_seconds: int) -> None:
    import matplotlib.pyplot as plt

    if hasattr(signal, "SIGALRM"):
        signal.alarm(timeout_seconds)
    try:
        code = compile(snippet.code, snippet.label, "exec")
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exec(code, namespace)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        plt.close("all")


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:300]


@dataclass
class RunReport:
    failures: list[str]
    incomplete: list[str]
    ran: int = 0
    skipped: int = 0


def check_run(items: list[Snippet], timeout_seconds: int) -> RunReport:
    """Run each document's snippets in order in one namespace.

    A snippet that fails there is rerun alone, because a page often reuses a
    name such as ``site`` for unrelated examples. It fails the check only if
    it also fails alone with anything but ``NameError``. A ``NameError`` alone
    marks it incomplete: it continues context the page does not define (or an
    unrelated example's), which ``--strict`` rejects.
    """

    import matplotlib

    matplotlib.use("Agg")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _raise_timeout)
    report = RunReport(failures=[], incomplete=[])
    by_document: dict[Path, list[Snippet]] = {}
    for snippet in items:
        by_document.setdefault(snippet.path, []).append(snippet)
    start_dir = Path.cwd()
    for group in by_document.values():
        shared_module = _fresh_module()
        with tempfile.TemporaryDirectory() as workdir:
            os.chdir(workdir)
            # Examples save outputs under results/, as a reader's project would.
            Path("results").mkdir()
            try:
                for snippet in group:
                    if snippet.skip:
                        report.skipped += 1
                        continue
                    report.ran += 1
                    started = time.perf_counter()
                    try:
                        _execute(snippet, shared_module.__dict__, timeout_seconds)
                    except _Timeout:
                        report.failures.append(
                            f"{snippet.label}: exceeded {timeout_seconds} s"
                        )
                    except Exception:  # noqa: BLE001 - retry alone, then judge
                        try:
                            _execute(snippet, _fresh_module().__dict__, timeout_seconds)
                        except NameError as exc:
                            report.incomplete.append(f"{snippet.label}: {_describe(exc)}")
                        except _Timeout:
                            report.failures.append(
                                f"{snippet.label}: exceeded {timeout_seconds} s"
                            )
                        except Exception as exc:  # noqa: BLE001
                            report.failures.append(f"{snippet.label}: {_describe(exc)}")
                        finally:
                            sys.modules[_MODULE] = shared_module
                    elapsed = time.perf_counter() - started
                    if elapsed > 0.5 * timeout_seconds:
                        print(f"slow: {snippet.label} took {elapsed:.1f} s")
            finally:
                os.chdir(start_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="store_true", help="also execute the snippets")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per snippet")
    parser.add_argument(
        "--strict", action="store_true", help="with --run, also reject incomplete snippets"
    )
    parser.add_argument("paths", nargs="*", type=Path, help="limit to these documents")
    args = parser.parse_args(argv)

    selected = [path.resolve() for path in args.paths] or documents()
    items = [snippet for path in selected for snippet in snippets(path)]
    failures = check_static(items)
    print(f"checked {len(items)} snippets in {len(selected)} documents statically")
    if args.run:
        report = check_run(items, args.timeout)
        failures += report.failures
        print(
            f"ran {report.ran} snippets ({report.skipped} marked skip); "
            f"{len(report.incomplete)} incomplete (use names their page does not define)"
        )
        if args.strict:
            failures += report.incomplete
    for failure in failures:
        print(failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
