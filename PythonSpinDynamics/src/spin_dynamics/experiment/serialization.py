"""JSON-compatible encoding/decoding for experiment specs and result metadata.

Encoded values use small tagged dicts so that frozen dataclasses, NumPy
arrays, and complex numbers round-trip through JSON. Only types registered
via :func:`register_serializable` can be reconstructed; encoding an
unregistered or non-representable object raises :class:`SerializationError`
with the offending field path so callers can switch to a plain-mapping form.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

TYPE_KEY = "__type__"
NDARRAY_KEY = "__ndarray__"
COMPLEX_KEY = "__complex__"


class SerializationError(TypeError):
    """Raised when a value cannot be encoded to or decoded from JSON form."""


_TYPE_TABLE: dict[str, type] = {}


def register_serializable(cls: type) -> type:
    """Register a dataclass so tagged dicts can be decoded back into it."""

    if not dataclasses.is_dataclass(cls):
        raise SerializationError(f"{cls.__name__} is not a dataclass")
    existing = _TYPE_TABLE.get(cls.__name__)
    if existing is not None and existing is not cls:
        raise SerializationError(
            f"a different type named {cls.__name__} is already registered"
        )
    _TYPE_TABLE[cls.__name__] = cls
    return cls


def registered_types() -> Mapping[str, type]:
    return dict(_TYPE_TABLE)


def encode(value: Any, *, path: str = "$") -> Any:
    """Encode a value into JSON-representable form."""

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise SerializationError(f"{path}: non-finite float is not serializable")
        return value
    if isinstance(value, (np.bool_, np.integer, np.floating)):
        return encode(value.item(), path=path)
    if isinstance(value, (complex, np.complexfloating)):
        cval = complex(value)
        return {COMPLEX_KEY: [cval.real, cval.imag]}
    if isinstance(value, np.ndarray):
        payload: Any
        if np.iscomplexobj(value):
            payload = {"real": value.real.tolist(), "imag": value.imag.tolist()}
        else:
            payload = value.tolist()
        return {NDARRAY_KEY: payload, "dtype": str(value.dtype)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _TYPE_TABLE:
            raise SerializationError(
                f"{path}: {name} is not registered for serialization; pass a "
                "plain mapping instead or register the type"
            )
        out: dict[str, Any] = {TYPE_KEY: name}
        for field in dataclasses.fields(value):
            out[field.name] = encode(
                getattr(value, field.name), path=f"{path}.{field.name}"
            )
        return out
    if isinstance(value, Mapping):
        encoded: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SerializationError(f"{path}: mapping keys must be strings")
            if key in (TYPE_KEY, NDARRAY_KEY, COMPLEX_KEY):
                raise SerializationError(f"{path}: reserved mapping key {key!r}")
            encoded[key] = encode(item, path=f"{path}[{key!r}]")
        return encoded
    if isinstance(value, (list, tuple)):
        return [encode(item, path=f"{path}[{i}]") for i, item in enumerate(value)]
    raise SerializationError(
        f"{path}: values of type {type(value).__name__} are not serializable"
    )


def decode(value: Any) -> Any:
    """Decode a value produced by :func:`encode`."""

    if isinstance(value, dict):
        if COMPLEX_KEY in value:
            real, imag = value[COMPLEX_KEY]
            return complex(float(real), float(imag))
        if NDARRAY_KEY in value:
            payload = value[NDARRAY_KEY]
            dtype = np.dtype(value.get("dtype", "float64"))
            if isinstance(payload, dict):
                real = np.asarray(payload["real"], dtype=np.float64)
                imag = np.asarray(payload["imag"], dtype=np.float64)
                return (real + 1j * imag).astype(dtype)
            return np.asarray(payload, dtype=dtype)
        if TYPE_KEY in value:
            name = value[TYPE_KEY]
            cls = _TYPE_TABLE.get(name)
            if cls is None:
                raise SerializationError(f"unknown serialized type {name!r}")
            fields = {
                key: decode(item) for key, item in value.items() if key != TYPE_KEY
            }
            return cls(**fields)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
