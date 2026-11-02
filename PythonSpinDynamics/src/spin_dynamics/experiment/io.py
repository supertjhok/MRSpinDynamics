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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from spin_dynamics.conventions import conventions
from spin_dynamics.experiment.runner import RunRecord
from spin_dynamics.experiment.provenance import (
    encoded_fingerprint,
    experiment_fingerprint,
    result_fingerprint,
)
from spin_dynamics.experiment.serialization import (
    SerializationError,
    decode,
    encode,
)
from spin_dynamics.experiment.specs import Experiment

FORMAT_NAME = "spin-dynamics-run"
FORMAT_VERSION = 2

_RESULT_TYPES: dict[str, type] = {}


def register_result_type(cls: type) -> type:
    """Register a workflow result dataclass for load-time reconstruction."""

    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    _RESULT_TYPES[cls.__name__] = cls
    return cls


def _npz_path(path: str | os.PathLike[str]) -> Path:
    """Return the file ``np.savez`` actually writes for ``path``.

    NumPy appends ``.npz`` when the name does not already end in it.
    """

    text = os.fspath(path)
    return Path(text if text.endswith(".npz") else text + ".npz")


def _existing_npz_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a load path, accepting the name passed to the matching save."""

    given = Path(os.fspath(path))
    if given.exists():
        return given
    written = _npz_path(path)
    return written if written.exists() else given


def save_run(record: RunRecord, path: str | os.PathLike[str]) -> Path:
    """Save a run archive and return the path actually written.

    As with :func:`numpy.savez`, ``.npz`` is appended when ``path`` does not
    already end in it; :func:`load_run` accepts either spelling.
    """

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
    written = _npz_path(path)
    np.savez(written, __meta__=json.dumps(meta), **arrays)
    return written


@dataclass(frozen=True)
class ReproductionReport:
    """Comparison between a saved run and a fresh execution of its spec."""

    matches: bool
    expected_result_sha256: str | None
    actual_result_sha256: str | None
    experiment_matches: bool
    archive_result_matches: bool | None
    implementation_matches: bool | None
    environment_matches: bool | None
    randomness_status: str
    notes: tuple[str, ...]
    rerun: RunRecord

    def require_match(self) -> None:
        """Raise when the rerun did not reproduce the saved result exactly."""

        if not self.matches:
            details = "; ".join(self.notes) or "result fingerprints differ"
            raise AssertionError(f"saved run was not reproduced: {details}")


@dataclass(frozen=True)
class LoadedRun:
    """A run loaded from disk: spec, raw arrays, and best-effort result."""

    experiment: Experiment
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, Any]
    provenance: Mapping[str, Any]
    format_version: int
    result_type: str
    unsaved_result_fields: tuple[str, ...]
    _result_meta: Mapping[str, Any]
    _experiment_payload: Mapping[str, Any] = dataclasses.field(default_factory=dict)

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

    @property
    def specification_matches(self) -> bool | None:
        """Whether the loaded spec matches its recorded canonical fingerprint.

        The stored spec payload is checked as saved, so an archive written
        before a spec class gained a defaulted field still matches; the
        re-encoded spec is accepted too.
        """

        expected = self.provenance.get("experiment_sha256")
        if expected is None:
            return None
        if self._experiment_payload and expected == encoded_fingerprint(
            self._experiment_payload
        ):
            return True
        return bool(expected == experiment_fingerprint(self.experiment))

    @property
    def result_matches(self) -> bool | None:
        """Whether persisted result content matches its recorded fingerprint."""

        expected = self.provenance.get("result_sha256")
        if expected is None or self.unsaved_result_fields:
            return None
        try:
            return bool(expected == result_fingerprint(self.result))
        except (TypeError, ValueError):
            return False

    def verify_reproduction(self, **execution: Any) -> ReproductionReport:
        """Rerun the saved experiment and compare canonical result identities.

        Stored execution options are reused unless explicitly overridden, and
        the rerun uses the :mod:`spin_dynamics.conventions` recorded with the
        run (archives without a record rerun under the active conventions).
        Version-1 archives without a result fingerprint can still be rerun, but
        cannot make an exact identity claim and therefore report ``matches=False``.
        """

        options = dict(self.provenance.get("execution", {}))
        options.update(execution)
        saved_conventions = self.provenance.get("conventions")
        if saved_conventions is None:
            rerun = self.experiment.run(**options)
        else:
            with conventions(saved_conventions):
                rerun = self.experiment.run(**options)
        expected = self.provenance.get("result_sha256")
        actual = rerun.provenance.get("result_sha256")
        experiment_matches = self.specification_matches
        if experiment_matches is None:
            experiment_matches = True

        old_impl = self.provenance.get("implementation")
        new_impl = rerun.provenance.get("implementation")
        implementation_matches = None if old_impl is None else old_impl == new_impl
        old_environment = self.provenance.get("environment", {}).get("sha256")
        new_environment = rerun.provenance.get("environment", {}).get("sha256")
        environment_matches = (
            None
            if old_environment is None
            else old_environment == new_environment
        )
        randomness = self.provenance.get("randomness", {}).get("status", "unknown")
        notes: list[str] = []
        if expected is None:
            notes.append("saved archive has no result fingerprint")
        elif expected != actual:
            notes.append("result fingerprints differ")
        if not experiment_matches:
            notes.append("loaded experiment fingerprint differs from the saved identity")
        archive_result_matches = self.result_matches
        if archive_result_matches is False:
            notes.append("persisted result content failed its fingerprint check")
        if implementation_matches is False:
            notes.append("workflow implementation fingerprint changed")
        if environment_matches is False:
            notes.append("numerical environment fingerprint changed")
        if randomness == "unseeded":
            notes.append("saved experiment contains an unseeded stochastic source")
        matches = bool(
            expected is not None
            and expected == actual
            and experiment_matches
            and archive_result_matches is not False
        )
        return ReproductionReport(
            matches=matches,
            expected_result_sha256=expected,
            actual_result_sha256=actual,
            experiment_matches=experiment_matches,
            archive_result_matches=archive_result_matches,
            implementation_matches=implementation_matches,
            environment_matches=environment_matches,
            randomness_status=randomness,
            notes=tuple(notes),
            rerun=rerun,
        )


def load_run(path: str | os.PathLike[str]) -> LoadedRun:
    path = _existing_npz_path(path)
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["__meta__"]))
        if meta.get("format") != FORMAT_NAME:
            raise ValueError(f"{path} is not a {FORMAT_NAME} archive")
        format_version = int(meta.get("format_version", 1))
        if format_version < 1 or format_version > FORMAT_VERSION:
            raise ValueError(
                f"unsupported {FORMAT_NAME} format version {format_version}; "
                f"this installation supports versions 1-{FORMAT_VERSION}"
            )
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
        format_version=format_version,
        result_type=str(meta.get("result_type", "")),
        unsaved_result_fields=tuple(meta.get("unsaved_result_fields", ())),
        _result_meta=dict(meta.get("result_meta", {})),
        _experiment_payload=meta["experiment"],
    )
