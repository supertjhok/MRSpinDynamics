"""Saving and loading experiment runs as NPZ archives with JSON provenance.

The archive holds every array field of the native result under
``result.<field>`` plus a single ``__meta__`` JSON document carrying the
experiment spec, provenance, scalar result fields, and JSON-representable
result metadata. Result metadata that cannot be serialized (e.g. objects
holding live probe models) is dropped and listed in
``unsaved_result_fields`` rather than failing the save.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from spin_dynamics.experiment.runner import RunRecord
from spin_dynamics.experiment.serialization import (
    SerializationError,
    decode,
    encode,
)
from spin_dynamics.experiment.specs import Experiment

FORMAT_NAME = "spin-dynamics-run"
FORMAT_VERSION = 1

_RESULT_TYPES: dict[str, type] = {}


def register_result_type(cls: type) -> type:
    """Register a workflow result dataclass for load-time reconstruction."""

    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    _RESULT_TYPES[cls.__name__] = cls
    return cls


def save_run(record: RunRecord, path: str) -> None:
    result = record.result
    if not dataclasses.is_dataclass(result) or isinstance(result, type):
        raise TypeError("RunRecord.result must be a workflow result dataclass")

    arrays: dict[str, np.ndarray] = {}
    scalars: dict[str, Any] = {}
    meta_fields: dict[str, Any] = {}
    unsaved: list[str] = []
    for spec_field in dataclasses.fields(result):
        name = spec_field.name
        value = getattr(result, name)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            arrays[f"result.{name}"] = value
        elif isinstance(value, (bool, int, float, str, np.bool_, np.integer, np.floating)):
            scalars[name] = value.item() if isinstance(value, np.generic) else value
        else:
            try:
                meta_fields[name] = encode(value, path=f"result.{name}")
            except SerializationError:
                unsaved.append(name)

    meta = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "experiment": record.experiment.to_dict(),
        "provenance": record.provenance,
        "result_type": type(result).__name__,
        "result_scalars": scalars,
        "result_meta": meta_fields,
        "unsaved_result_fields": unsaved,
    }
    np.savez(path, __meta__=json.dumps(meta), **arrays)


@dataclass(frozen=True)
class LoadedRun:
    """A run loaded from disk: spec, raw arrays, and best-effort result."""

    experiment: Experiment
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, Any]
    provenance: Mapping[str, Any]
    result_type: str
    unsaved_result_fields: tuple[str, ...]
    _result_meta: Mapping[str, Any]

    @property
    def result(self) -> Any:
        """Reconstruct the native result dataclass.

        Fields listed in ``unsaved_result_fields`` come back as ``None``;
        everything else round-trips exactly.
        """

        cls = _RESULT_TYPES.get(self.result_type)
        if cls is None:
            raise TypeError(
                f"result type {self.result_type!r} is not registered; "
                "use .arrays / .scalars for raw access"
            )
        fields: dict[str, Any] = {}
        for spec_field in dataclasses.fields(cls):
            name = spec_field.name
            if name in self.arrays:
                fields[name] = self.arrays[name]
            elif name in self.scalars:
                fields[name] = self.scalars[name]
            elif name in self._result_meta:
                fields[name] = decode(self._result_meta[name])
            else:
                fields[name] = None
        return cls(**fields)


def load_run(path: str) -> LoadedRun:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a {FORMAT_NAME} archive")
        arrays = {
            key.removeprefix("result."): archive[key]
            for key in archive.files
            if key.startswith("result.")
        }
    return LoadedRun(
        experiment=Experiment.from_dict(meta["experiment"]),
        arrays=arrays,
        scalars=dict(meta.get("result_scalars", {})),
        provenance=dict(meta.get("provenance", {})),
        result_type=str(meta.get("result_type", "")),
        unsaved_result_fields=tuple(meta.get("unsaved_result_fields", ())),
        _result_meta=dict(meta.get("result_meta", {})),
    )
