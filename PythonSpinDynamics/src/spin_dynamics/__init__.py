"""Python port of the MATLABSpinDynamics simulation package."""

from importlib.metadata import PackageNotFoundError, version


try:
    __version__ = version("python-spin-dynamics")
except PackageNotFoundError:  # pragma: no cover - source tree without metadata
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "absolute_phase",
    "analysis",
    "coupling",
    "composition",
    "core",
    "deprecation",
    "detection",
    "design",
    "esr",
    "exchange",
    "experiment",
    "flow",
    "hyperpolarization",
    "interference",
    "motion",
    "nano_mr",
    "noise",
    "nonresonant",
    "nqr",
    "parameters",
    "phase_cycling",
    "prepolarization",
    "pulse_diagnostics",
    "relaxation",
    "radiation_damping",
    "sample",
    "spin_noise",
    "susceptibility",
    "thermal",
    "pulses",
    "optimization",
    "probes",
    "sequences",
    "workflows",
]
