"""JSON encoding helpers for adaptive-session checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


def encode_jsonable(value: Any) -> Any:
    """Recursively encode NumPy arrays and complex scalars for JSON."""

    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            payload = {
                "real": np.real(value).tolist(),
                "imag": np.imag(value).tolist(),
            }
        else:
            payload = {"values": value.tolist()}
        return {
            "__ndarray__": payload,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return encode_jsonable(np.asarray(value))
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, Mapping):
        return {str(key): encode_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_jsonable(item) for item in value]
    return value


def decode_jsonable(value: Any) -> Any:
    """Inverse of :func:`encode_jsonable`."""

    if isinstance(value, list):
        return [decode_jsonable(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "__complex__" in value:
        real, imag = value["__complex__"]
        return complex(real, imag)
    if "__ndarray__" in value:
        payload = value["__ndarray__"]
        if "real" in payload:
            array = np.asarray(payload["real"]) + 1j * np.asarray(payload["imag"])
        else:
            array = np.asarray(payload["values"])
        return np.asarray(array, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    return {key: decode_jsonable(item) for key, item in value.items()}


def state_to_json(state: Mapping[str, Any], *, indent: int | None = 2) -> str:
    return json.dumps(encode_jsonable(dict(state)), indent=indent)


def state_from_json(payload: str) -> dict[str, Any]:
    decoded = decode_jsonable(json.loads(payload))
    if not isinstance(decoded, dict):
        raise TypeError("checkpoint payload must describe a mapping")
    return decoded


def save_design_state(path: str | Path, state: Mapping[str, Any]) -> None:
    Path(path).write_text(state_to_json(state), encoding="utf-8")


def save_design_state_atomic(path: str | Path, state: Mapping[str, Any]) -> None:
    """Atomically replace a JSON checkpoint after flushing it to disk.

    The temporary file is created in the destination directory so
    :func:`os.replace` remains an atomic same-filesystem operation. If writing
    fails, the previous checkpoint is retained and the temporary file is
    removed.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor_open = False
        with handle:
            handle.write(state_to_json(state))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            if descriptor_open:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        finally:
            raise


def load_design_state(path: str | Path) -> dict[str, Any]:
    return state_from_json(Path(path).read_text(encoding="utf-8"))
