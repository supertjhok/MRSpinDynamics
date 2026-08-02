"""Human-friendly config files for experiments (TOML / JSON).

Unlike the canonical machine encoding in
:mod:`spin_dynamics.experiment.serialization` (tagged ``__type__`` /
``__ndarray__`` dicts, used for exact result round-trips), this module
produces the readable, hand-authorable form used by the CLI and config-driven
runs. Every spec object is a table with a ``type`` key naming its class; scalar
fields are plain keys and arrays are plain lists::

    [sequence]
    type = "CPMGTrain"
    num_echoes = 8

    [sample]
    t1_seconds = 2.0
    t2_seconds = 2.0

    [hardware]
    probe = "tuned"

    [acquisition]
    numpts = 501
    maxoffs = 10.0

The top-level document describes an ``Experiment``; the ``sample``,
``hardware``, and ``acquisition`` sections may omit their ``type`` (their
class is implied) and may be omitted entirely to accept the defaults.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from spin_dynamics.esr import ESRSpinSystem, HyperfineCoupling
from spin_dynamics.experiment.hardware import (
    ImagingPlane,
    PlanarSpiralCoil,
    RxArray,
    RxCoil,
    SolenoidCoil,
    TxCoil,
    UniformB0,
)
from spin_dynamics.experiment.specs import (
    Acquisition,
    DEERDistribution,
    Experiment,
    Hardware,
    NanoMRBathComponent,
    NanoMRLayer,
    NanoMROpticalReadout,
    NanoMRSensor,
    Phantom,
    Sample,
    SEQUENCE_TYPES,
    SequenceDomain,
    TransportDomain2D,
    UniformFlow2D,
)
from spin_dynamics.noise import NoiseSpec
from spin_dynamics.nqr import QuadrupolarSite
from spin_dynamics.sequences import (
    ADCEvent,
    GradientWaveform,
    HardwareEffectsPolicy,
    RFPulse,
    SequenceBlock,
    SequenceIR,
)

_SPEC_CLASSES: tuple[type, ...] = (
    Sample,
    Hardware,
    Acquisition,
    Phantom,
    TransportDomain2D,
    UniformFlow2D,
    DEERDistribution,
    SequenceDomain,
    NanoMRSensor,
    NanoMRBathComponent,
    NanoMRLayer,
    NanoMROpticalReadout,
    *SEQUENCE_TYPES,
    SolenoidCoil,
    PlanarSpiralCoil,
    TxCoil,
    RxCoil,
    RxArray,
    UniformB0,
    ImagingPlane,
    QuadrupolarSite,
    ESRSpinSystem,
    HyperfineCoupling,
    NoiseSpec,
    HardwareEffectsPolicy,
    RFPulse,
    GradientWaveform,
    ADCEvent,
    SequenceBlock,
    SequenceIR,
)
_REGISTRY: dict[str, type] = {cls.__name__: cls for cls in _SPEC_CLASSES}
_SECTION_CLASSES: dict[str, type] = {
    "sample": Sample,
    "hardware": Hardware,
    "acquisition": Acquisition,
}


class ConfigError(ValueError):
    """Raised when a config document cannot be built into an Experiment."""


# --------------------------------------------------------------------------
# build (config mapping -> Experiment)
# --------------------------------------------------------------------------
def _build(value: Any) -> Any:
    if isinstance(value, dict):
        if "type" in value:
            type_name = value["type"]
            cls = _REGISTRY.get(type_name)
            if cls is None:
                raise ConfigError(f"unknown spec type {type_name!r}")
            fields = {k: _build(v) for k, v in value.items() if k != "type"}
            try:
                return cls(**fields)
            except TypeError as exc:
                raise ConfigError(f"{type_name}: {exc}") from exc
        # An untyped table is a plain mapping (e.g. a noise spec dict).
        return {k: _build(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_build(v) for v in value]
    return value


def _build_section(value: Any, cls: type) -> Any:
    if value is None:
        return cls()
    if not isinstance(value, dict):
        raise ConfigError(f"section for {cls.__name__} must be a table")
    declared = value.get("type")
    if declared is not None and declared != cls.__name__:
        raise ConfigError(
            f"section type {declared!r} does not match expected {cls.__name__!r}"
        )
    fields = {k: _build(v) for k, v in value.items() if k != "type"}
    try:
        return cls(**fields)
    except TypeError as exc:
        raise ConfigError(f"{cls.__name__}: {exc}") from exc


def experiment_from_config(data: dict[str, Any]) -> Experiment:
    """Build an :class:`Experiment` from a friendly config mapping."""

    if not isinstance(data, dict):
        raise ConfigError("config document must be a table")
    data = dict(data)
    top_type = data.pop("type", "Experiment")
    if top_type != "Experiment":
        raise ConfigError(f"top-level type must be 'Experiment', got {top_type!r}")
    if "sequence" not in data:
        raise ConfigError("config must define a [sequence] with a type")
    sequence = _build(data["sequence"])
    kwargs: dict[str, Any] = {"sequence": sequence}
    for section, cls in _SECTION_CLASSES.items():
        if section in data:
            kwargs[section] = _build_section(data[section], cls)
    unknown = set(data) - {"sequence", *_SECTION_CLASSES}
    if unknown:
        raise ConfigError(f"unknown config sections: {sorted(unknown)}")
    return Experiment(**kwargs)


# --------------------------------------------------------------------------
# emit (Experiment -> config mapping)
# --------------------------------------------------------------------------
def _emit(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _REGISTRY:
            raise ConfigError(
                f"{name} cannot be written to a config file; use the JSON "
                "result archive instead"
            )
        out: dict[str, Any] = {"type": name}
        for f in dataclasses.fields(value):
            current = getattr(value, f.name)
            if _is_default(f, current):
                continue
            out[f.name] = _emit(current)
        return out
    if isinstance(value, dict):
        return {k: _emit(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_emit(v) for v in value]
    return value


def _is_default(field: dataclasses.Field, value: Any) -> bool:
    if field.default is not dataclasses.MISSING:
        default = field.default
    elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        default = field.default_factory()  # type: ignore[misc]
    else:
        return False
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return np.array_equal(np.asarray(value), np.asarray(default))
    return bool(value == default)


def experiment_to_config(experiment: Experiment) -> dict[str, Any]:
    """Return a friendly config mapping for an experiment (round-trips)."""

    config: dict[str, Any] = {"sequence": _emit(experiment.sequence)}
    for section, cls in _SECTION_CLASSES.items():
        spec = getattr(experiment, section)
        if spec != cls():
            emitted = _emit(spec)
            emitted.pop("type", None)  # class is implied by the section name
            config[section] = emitted
    return config


# --------------------------------------------------------------------------
# TOML / JSON serialization
# --------------------------------------------------------------------------
def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ConfigError("non-finite floats cannot be written to TOML")
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise ConfigError(f"cannot serialize {type(value).__name__} to TOML")


def _toml_array(value: list) -> str:
    return "[" + ", ".join(_toml_value(v) for v in value) + "]"


def _toml_value(value: Any) -> str:
    if isinstance(value, list):
        return _toml_array(value)
    return _toml_scalar(value)


def _dump_toml(mapping: dict[str, Any], path: tuple[str, ...] = ()) -> list[str]:
    scalar_lines: list[str] = []
    table_lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            sub_path = path + (key,)
            table_lines.append("")
            table_lines.append(f"[{'.'.join(sub_path)}]")
            table_lines.extend(_dump_toml(value, sub_path))
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(item, dict) for item in value)
        ):
            sub_path = path + (key,)
            for item in value:
                table_lines.append("")
                table_lines.append(f"[[{'.'.join(sub_path)}]]")
                table_lines.extend(_dump_toml(item, sub_path))
        else:
            scalar_lines.append(f"{key} = {_toml_value(value)}")
    return scalar_lines + table_lines


def dumps_toml(config: dict[str, Any]) -> str:
    """Serialize a friendly config mapping to a TOML string."""

    return "\n".join(_dump_toml(config)).lstrip("\n") + "\n"


def dumps_json(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2) + "\n"


def _loads_toml(text: str) -> dict[str, Any]:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ConfigError(
                "reading TOML configs requires Python 3.11+ or the 'tomli' "
                "package; use a .json config instead"
            ) from exc
    return tomllib.loads(text)


def save_config(experiment: Experiment, path: str | Path) -> None:
    """Write an experiment to a ``.toml`` or ``.json`` config file."""

    path = Path(path)
    config = experiment_to_config(experiment)
    if path.suffix == ".json":
        text = dumps_json(config)
    elif path.suffix == ".toml":
        text = dumps_toml(config)
    else:
        raise ConfigError(
            f"config path must end in .toml or .json, got {path.suffix!r}"
        )
    path.write_text(text, encoding="utf-8")


def load_config(path: str | Path) -> Experiment:
    """Read an experiment from a ``.toml`` or ``.json`` config file."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix == ".toml":
        data = _loads_toml(text)
    else:
        raise ConfigError(
            f"config path must end in .toml or .json, got {path.suffix!r}"
        )
    return experiment_from_config(data)
