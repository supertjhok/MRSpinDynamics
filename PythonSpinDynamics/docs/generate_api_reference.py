"""Generate a lightweight Markdown API inventory from source docstrings."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "spin_dynamics"
OUTPUT = ROOT / "docs" / "python_api" / "api_reference.md"

MODULES = [
    "analysis.compressed_sensing",
    "analysis.ilt",
    "analysis.regularization",
    "absolute_phase",
    "coupling.evolution",
    "coupling.hamiltonians",
    "coupling.isochromats",
    "coupling.isotopes",
    "coupling.j_editing",
    "coupling.mixed_operators",
    "coupling.multinuclear",
    "coupling.operators",
    "coupling.slic",
    "coupling.systems",
    "coupling.zulf",
    "composition",
    "core.echo",
    "core.isochromats",
    "core.kernels",
    "core.numerics",
    "core.rotations",
    "detection.base",
    "detection.gradiometer",
    "detection.inductive",
    "detection.opm",
    "detection.spatial",
    "detection.squid",
    "design.reference_ir",
    "design.adapters",
    "design.adapter_benchmarks",
    "design.benchmarks",
    "design.constraints",
    "design.costs",
    "design.diagnostics",
    "design.io",
    "design.likelihoods",
    "design.live",
    "design.models",
    "design.performance",
    "design.posterior",
    "design.priors",
    "design.robustness",
    "design.session",
    "design.spaces",
    "design.types",
    "design.utilities",
    "deprecation",
    "esr.deer",
    "esr.dipolar",
    "esr.distributions",
    "esr.endor",
    "esr.eseem",
    "esr.hamiltonians",
    "esr.hyperfine",
    "esr.hyscore",
    "esr.lineshapes",
    "esr.orientations",
    "esr.pulsed",
    "esr.relaxation",
    "esr.spectra",
    "esr.systems",
    "exchange",
    "experiment.cli",
    "experiment.config",
    "experiment.esr_adapter",
    "experiment.esr_multidim_adapter",
    "experiment.estimate",
    "experiment.hardware",
    "experiment.io",
    "experiment.nano_mr_adapter",
    "experiment.nqr_adapter",
    "experiment.plan",
    "experiment.provenance",
    "experiment.registry",
    "experiment.rules",
    "experiment.runner",
    "experiment.serialization",
    "experiment.sequence_adapter",
    "experiment.specs",
    "experiment.wiring",
    "fields.birdcage",
    "fields.birdcage_circuit",
    "fields.birdcage_multiport",
    "fields.coil_peec",
    "fields.coil_properties",
    "fields.fasthenry_interop",
    "fields.fastercap_interop",
    "fields.coils",
    "fields.domain",
    "fields.eddy_modes",
    "fields.electropermanent",
    "fields.electropermanent_array",
    "fields.electropermanent_hysteresis",
    "fields.electropermanent_transient",
    "fields.electropermanent_pulses",
    "fields.gradient_coils",
    "fields.gradient_engineering",
    "fields.gradient_shielding",
    "fields.gradient_windings",
    "fields.fullwave_validation",
    "fields.harmonic",
    "fields.openems",
    "fields.interpolate",
    "fields.magnetostatics",
    "fields.maps",
    "fields.nonlinear_magnetostatics",
    "fields.positions",
    "fields.quasistatic",
    "fields.scalar_potential_3d",
    "fields.validity",
    "flow",
    "hyperpolarization.singlet",
    "hyperpolarization.lls",
    "hyperpolarization.phip",
    "interference.active",
    "interference.cancellers",
    "interference.coils",
    "interference.diagnostics",
    "interference.masks",
    "interference.recordings",
    "interference.sources",
    "interference.trackers",
    "motion",
    "nano_mr.baths",
    "nano_mr.compiler",
    "nano_mr.correlation",
    "nano_mr.endor_qdyne",
    "nano_mr.esr_bridge",
    "nano_mr.exact",
    "nano_mr.filter_functions",
    "nano_mr.frames",
    "nano_mr.geometry",
    "nano_mr.hamiltonians",
    "nano_mr.high_resolution",
    "nano_mr.imaging",
    "nano_mr.motion",
    "nano_mr.noise",
    "nano_mr.presets",
    "nano_mr.readout",
    "nano_mr.sequences",
    "nano_mr.sensors",
    "nano_mr.statistical",
    "noise",
    "nonresonant.field_reversal",
    "nonresonant.sequences",
    "nqr.crossover",
    "nqr.crossover_sequences",
    "nqr.floquet",
    "nqr.field_relaxation",
    "nqr.full_dynamics",
    "nqr.hamiltonians",
    "nqr.inhomogeneity",
    "nqr.interference",
    "nqr.isotopes",
    "nqr.lab_frame",
    "nqr.model_selection",
    "nqr.operators",
    "nqr.orientations",
    "nqr.piezo_detection",
    "nqr.polarization_enhancement",
    "nqr.pulses",
    "nqr.relaxation",
    "nqr.rwa_validation",
    "nqr.sequences",
    "nqr.simulation",
    "nqr.structure_coupling",
    "nqr.systems",
    "nqr.zeeman",
    "nqr.workflows",
    "parameters.constructors",
    "phase_cycling",
    "optimal_control.control_response",
    "optimal_control.detection_objective",
    "optimal_control.diffusion",
    "optimal_control.drivers",
    "optimal_control.hamiltonians",
    "optimal_control.multi_axis",
    "optimal_control.objectives",
    "optimal_control.parameterization",
    "optimal_control.solvers",
    "optimization.drivers",
    "optimization.excitation",
    "optimization.pipeline",
    "optimization.refocusing",
    "optimization.results",
    "optimization.spa",
    "prepolarization",
    "probes.matched",
    "probes.tuned",
    "probes.untuned",
    "pulses",
    "pulse_diagnostics",
    "relaxation",
    "radiation_damping",
    "receiver_frontend",
    "receiver_network",
    "sample",
    "sequences.cpmg",
    "sequences.compiler",
    "sequences.ir",
    "sequences.motion",
    "sequences.plotting",
    "sequences.pulseq",
    "susceptibility",
    "spin_noise",
    "thermal.conduction",
    "thermal.coupling",
    "thermal.electromagnet",
    "thermal.materials",
    "thermal.network",
    "thermal.sources",
    "workflows.acquisition",
    "workflows.batched_sweeps",
    "workflows.bipolar",
    "workflows.bloch_siegert",
    "workflows.bssfp",
    "workflows.cpmg",
    "workflows.cpmg_ir",
    "workflows.diffusion",
    "workflows.electropermanent_imaging",
    "workflows.electropermanent_controller",
    "workflows.electropermanent_particle_imaging",
    "workflows.electropermanent_particle_spin_echo",
    "workflows.electropermanent_dynamic_inversion",
    "workflows.electropermanent_transport",
    "workflows.fid",
    "workflows.imaging",
    "workflows.imaging_3d",
    "workflows.imaging_frequency",
    "workflows.imaging_types",
    "workflows.receiver_arrays",
    "workflows.relaxation",
    "workflows.sense",
    "workflows.pgse",
    "workflows.portable_halbach",
    "workflows.qspace",
    "workflows.single_sided",
    "workflows.slice_selective",
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
            symbols.append(
                Symbol("function", node.name, _signature(node), _summary(node))
            )
    return symbols


def _unlisted_public_modules() -> list[str]:
    """Return public source modules omitted from the generated inventory."""

    missing = []
    for path in SRC.rglob("*.py"):
        relative = path.relative_to(SRC)
        if path.name == "__init__.py" or any(
            part.startswith("_") for part in relative.parts
        ):
            continue
        module = ".".join(relative.with_suffix("").parts)
        if module not in MODULES and _module_symbols(path):
            missing.append(module)
    return sorted(missing)

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
    missing = _unlisted_public_modules()
    if missing:
        raise SystemExit(
            "public modules missing from API inventory: " + ", ".join(missing)
        )

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
