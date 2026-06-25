"""Generate a lightweight Markdown API inventory from source docstrings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "spin_dynamics"
OUTPUT = ROOT / "docs" / "python_api" / "api_reference.md"

MODULES = [
    "analysis.ilt",
    "analysis.regularization",
    "absolute_phase",
    "coupling.evolution",
    "coupling.hamiltonians",
    "coupling.isochromats",
    "coupling.j_editing",
    "coupling.operators",
    "coupling.slic",
    "coupling.systems",
    "core.echo",
    "core.isochromats",
    "core.kernels",
    "core.numerics",
    "core.rotations",
    "esr.distributions",
    "esr.hamiltonians",
    "esr.hyperfine",
    "esr.lineshapes",
    "esr.orientations",
    "esr.pulsed",
    "esr.relaxation",
    "esr.spectra",
    "esr.systems",
    "exchange",
    "motion",
    "noise",
    "nqr.full_dynamics",
    "nqr.hamiltonians",
    "nqr.inhomogeneity",
    "nqr.model_selection",
    "nqr.operators",
    "nqr.orientations",
    "nqr.pulses",
    "nqr.relaxation",
    "nqr.sequences",
    "nqr.simulation",
    "nqr.systems",
    "nqr.zeeman",
    "nqr.workflows",
    "parameters.constructors",
    "phase_cycling",
    "optimization.drivers",
    "optimization.excitation",
    "optimization.pipeline",
    "optimization.refocusing",
    "optimization.results",
    "optimization.spa",
    "pulses",
    "pulse_diagnostics",
    "radiation_damping",
    "sequences.motion",
    "workflows.acquisition",
    "workflows.cpmg",
    "workflows.cpmg_ir",
    "workflows.diffusion",
    "workflows.fid",
    "workflows.imaging",
    "workflows.imaging_frequency",
    "workflows.imaging_types",
    "workflows.pgse",
    "workflows.sweeps",
    "workflows.time_varying",
    "workflows.wurst",
]


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    signature: str
    summary: str


def _module_path(module: str) -> Path:
    parts = module.split(".")
    package_path = SRC.joinpath(*parts, "__init__.py")
    if package_path.exists():
        return package_path
    return SRC.joinpath(*parts).with_suffix(".py")


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    return ast.unparse(node)


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> str:
    if isinstance(node, ast.ClassDef):
        return ""
    args = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    defaults = [None] * (len(positional) - len(node.args.defaults)) + list(
        node.args.defaults
    )
    for arg, default in zip(positional, defaults):
        text = arg.arg
        annotation = _annotation_text(arg.annotation)
        if annotation:
            text += f": {annotation}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        args.append(text)
    if node.args.vararg is not None:
        args.append(f"*{node.args.vararg.arg}")
    elif node.args.kwonlyargs:
        args.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
        text = arg.arg
        annotation = _annotation_text(arg.annotation)
        if annotation:
            text += f": {annotation}"
        if default is not None:
            text += f" = {ast.unparse(default)}"
        args.append(text)
    if node.args.kwarg is not None:
        args.append(f"**{node.args.kwarg.arg}")
    signature = f"({', '.join(args)})"
    returns = _annotation_text(node.returns)
    if returns:
        signature += f" -> {returns}"
    return signature


def _summary(node: ast.AST) -> str:
    doc = ast.get_docstring(node) or ""
    if not doc:
        return ""
    first = doc.strip().splitlines()[0].strip()
    return first.rstrip(".") + "."


def _module_symbols(path: Path) -> list[Symbol]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    symbols = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and _is_public(node.name):
            symbols.append(Symbol("class", node.name, "", _summary(node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_public(
            node.name
        ):
            symbols.append(Symbol("function", node.name, _signature(node), _summary(node)))
    return symbols


def _render_module(module: str, symbols: list[Symbol]) -> str:
    lines = [f"## `spin_dynamics.{module}`", ""]
    if not symbols:
        lines.extend(["No public classes or functions found.", ""])
        return "\n".join(lines)
    lines.extend(["| Kind | Name | Summary |", "| --- | --- | --- |"])
    for symbol in symbols:
        name = f"`{symbol.name}{symbol.signature}`"
        lines.append(f"| {symbol.kind} | {name} | {symbol.summary} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parts = [
        "# API Reference",
        "",
        "Generated from public class and function docstrings by "
        "`docs/generate_api_reference.py`.",
        "",
        "This reference is an inventory, not a substitute for the user manual. "
        "For numerical assumptions, equations, and workflow guidance, see "
        "`docs/user_manual.tex`.",
        "",
    ]
    for module in MODULES:
        path = _module_path(module)
        if not path.exists():
            raise SystemExit(f"missing module source: {module} ({path})")
        parts.append(_render_module(module, _module_symbols(path)))
    OUTPUT.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
