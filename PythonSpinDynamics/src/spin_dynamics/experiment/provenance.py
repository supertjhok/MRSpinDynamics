"""Canonical fingerprints and reproducibility metadata for facade runs."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import os
import platform
import struct
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from spin_dynamics.experiment.serialization import encode

PROVENANCE_SCHEMA_VERSION = 2
FINGERPRINT_ALGORITHM = "sha256"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def experiment_fingerprint(experiment: Any) -> str:
    """Return a canonical SHA-256 identity for a complete experiment spec."""

    return _sha256_json(encode(experiment))


def _feed(hasher: Any, value: Any) -> None:
    if value is None:
        hasher.update(b"none;")
    elif isinstance(value, (bool, np.bool_)):
        hasher.update(b"bool:1;" if bool(value) else b"bool:0;")
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        hasher.update(f"int:{int(value)};".encode())
    elif isinstance(value, (float, np.floating)):
        hasher.update(b"float:")
        hasher.update(struct.pack(">d", float(value)))
    elif isinstance(value, (complex, np.complexfloating)):
        number = complex(value)
        hasher.update(b"complex:")
        hasher.update(struct.pack(">dd", number.real, number.imag))
    elif isinstance(value, str):
        payload = value.encode("utf-8")
        hasher.update(f"str:{len(payload)}:".encode())
        hasher.update(payload)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays do not have a canonical fingerprint")
        array = np.ascontiguousarray(value)
        if array.dtype.byteorder == ">" or (
            array.dtype.byteorder == "=" and sys.byteorder == "big"
        ):
            array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        dtype = array.dtype.newbyteorder("<").str
        hasher.update(
            f"array:{dtype}:{','.join(map(str, array.shape))}:".encode()
        )
        hasher.update(array.tobytes(order="C"))
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        hasher.update(f"dataclass:{cls.__module__}.{cls.__qualname__}:".encode())
        for spec_field in dataclasses.fields(value):
            _feed(hasher, spec_field.name)
            _feed(hasher, getattr(value, spec_field.name))
    elif isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("fingerprinted mappings require string keys")
        hasher.update(b"mapping:")
        for key in sorted(value):
            _feed(hasher, key)
            _feed(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(f"sequence:{len(value)}:".encode())
        for item in value:
            _feed(hasher, item)
    else:
        raise TypeError(f"cannot fingerprint {type(value).__name__}")


def result_fingerprint(result: Any) -> str:
    """Return a content fingerprint for a native result dataclass."""

    hasher = hashlib.sha256()
    _feed(hasher, result)
    return hasher.hexdigest()


def callable_identity(func: Callable[..., Any]) -> dict[str, Any]:
    """Describe and fingerprint the resolved workflow implementation."""

    try:
        source = inspect.getsource(func).encode("utf-8")
        source_hash = hashlib.sha256(source).hexdigest()
    except (OSError, TypeError):
        source_hash = None
    module_hash = None
    try:
        source_file = inspect.getsourcefile(func)
        if source_file is not None:
            module_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    except OSError:
        pass
    return {
        "module": func.__module__,
        "qualname": func.__qualname__,
        "callable_sha256": source_hash,
        "module_sha256": module_hash,
    }


def environment_identity() -> dict[str, Any]:
    """Return numerical-runtime versions and their canonical fingerprint."""

    packages: dict[str, str] = {"numpy": np.__version__}
    try:
        import scipy

        packages["scipy"] = scipy.__version__
    except ImportError:
        pass
    numpy_build = getattr(np.__config__, "CONFIG", {})
    numpy_build = json.loads(json.dumps(numpy_build, default=str))
    thread_variables = {
        name: os.environ[name]
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        if name in os.environ
    }
    details = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": packages,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy_build": numpy_build,
        "thread_environment": thread_variables,
    }
    return {**details, "sha256": _sha256_json(details)}


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


@lru_cache(maxsize=1)
def package_source_fingerprint() -> str:
    """Fingerprint every Python source file in the installed package tree."""

    package_root = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        hasher.update(f"{len(relative)}:".encode())
        hasher.update(relative)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


@lru_cache(maxsize=1)
def source_revision() -> dict[str, Any]:
    """Return Git revision/dirty state when running from a checkout."""

    root = _repository_root()
    if root is None:
        return {
            "git_commit": None,
            "git_dirty": None,
            "package_source_sha256": package_source_fingerprint(),
        }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout.strip()
        package_root = Path(__file__).resolve().parents[3]
        try:
            package_path = str(package_root.relative_to(root))
        except ValueError:
            package_path = "."
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", package_path],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        ).stdout
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "package_source_sha256": package_source_fingerprint(),
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "git_commit": None,
            "git_dirty": None,
            "package_source_sha256": package_source_fingerprint(),
        }


def randomness_identity(experiment: Any) -> dict[str, Any]:
    """Classify stochastic facade inputs and record their seeds."""

    sources: list[dict[str, Any]] = []
    sequence = experiment.sequence
    if hasattr(sequence, "seed"):
        seed = getattr(sequence, "seed")
        sources.append({"source": "sequence", "seed": seed})

    noise = experiment.acquisition.noise
    if noise is not None:
        if isinstance(noise, Mapping):
            seed = noise.get("seed")
            external_rng = noise.get("rng") is not None
        else:
            seed = getattr(noise, "seed", None)
            external_rng = getattr(noise, "rng", None) is not None
        sources.append(
            {"source": "acquisition.noise", "seed": seed, "external_rng": external_rng}
        )

    if not sources:
        status = "deterministic"
    elif any(item.get("seed") is None for item in sources):
        status = "unseeded"
    else:
        status = "seeded"
    return {"status": status, "sources": sources}


def build_provenance(
    experiment: Any,
    result: Any,
    func: Callable[..., Any],
    *,
    package_version: str,
    elapsed_seconds: float,
    execution: Mapping[str, Any],
    plan_warnings: tuple[str, ...],
    timestamp_utc: str,
) -> dict[str, Any]:
    """Build the versioned provenance record for one completed run."""

    try:
        output_hash = result_fingerprint(result)
        fingerprint_error = None
    except TypeError as exc:
        output_hash = None
        fingerprint_error = str(exc)
    environment = environment_identity()
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        "workflow": func.__name__,
        "package_version": package_version,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "timestamp_utc": timestamp_utc,
        "elapsed_seconds": elapsed_seconds,
        "execution": dict(execution),
        "plan_warnings": list(plan_warnings),
        "experiment_sha256": experiment_fingerprint(experiment),
        "result_sha256": output_hash,
        "result_fingerprint_error": fingerprint_error,
        "implementation": callable_identity(func),
        "environment": environment,
        "source": source_revision(),
        "randomness": randomness_identity(experiment),
    }


__all__ = [
    "FINGERPRINT_ALGORITHM",
    "PROVENANCE_SCHEMA_VERSION",
    "build_provenance",
    "callable_identity",
    "environment_identity",
    "experiment_fingerprint",
    "package_source_fingerprint",
    "randomness_identity",
    "result_fingerprint",
    "source_revision",
]
