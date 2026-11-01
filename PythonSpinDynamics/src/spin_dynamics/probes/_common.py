"""Shared parameter-access helpers for the probe circuit models.

The probe functions accept MATLAB-style parameter records either as mappings
or as attribute objects (the frozen dataclasses from
:mod:`spin_dynamics.parameters`, or ``SimpleNamespace``-like objects).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np


def field(obj: Mapping[str, Any] | Any, name: str) -> Any:
    """Read ``name`` from a mapping or attribute object.

    ``in_`` falls back to a mapping's ``"in"`` key (MATLAB's ``sp.in`` is a
    Python keyword).
    """

    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        if name == "in_":
            return obj["in"]
    return getattr(obj, name)


def with_fields(obj: Mapping[str, Any] | Any, **updates: Any) -> Any:
    """Return a copy of ``obj`` with ``updates`` applied.

    Mappings come back as a new ``dict``. A dataclass whose fields include
    every updated name is copied with :func:`dataclasses.replace` (keeping its
    type); otherwise, e.g. when adding solver-only keys such as ``tp`` or
    ``C1``, the object's attributes are copied into a ``dict``.
    """

    if isinstance(obj, Mapping):
        out = dict(obj)
        out.update(updates)
        return out
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        names = {item.name for item in dataclasses.fields(obj)}
        if set(updates) <= names:
            return dataclasses.replace(obj, **updates)
    out = dict(vars(obj))
    out.update(updates)
    return out


def as_vector(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """Return ``value`` as a flat array of ``dtype``."""

    return np.asarray(value, dtype=dtype).reshape(-1)
